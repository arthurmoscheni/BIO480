import os
import pandas as pd
import numpy as np
from scipy.signal import find_peaks, bessel, sosfilt

def load_and_group_cells(file_path):
    print(f"Loading data from {file_path}...")
    if not os.path.exists(file_path):
        # Fallback to local file if path not found
        if os.path.exists(os.path.basename(file_path)):
            file_path = os.path.basename(file_path)
        else:
            raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_pickle(file_path)
    
    all_cols = df.columns.tolist()
    contact_col = next((c for c in all_cols if 'Contact' in c and 'Passive' not in c), None)
    vm_col = next((c for c in all_cols if 'Membrane' in c or 'Vm' in c), None)
    type_col = next((c for c in all_cols if 'Sweep_Type' in c or 'Type' in c), 'Sweep_Type')
    
    cols = {
        'mouse': 'Mouse_Name',
        'counter': 'Cell_Counter',
        'type': 'Cell_Type',
        'sweep_type': 'Sweep_Type',
        'vm': vm_col,
        'contacts': contact_col 
    }

    grouped_cells = df.groupby([cols['mouse'], cols['counter']])
    print(f"Data Loaded. Found {len(grouped_cells)} unique cells.\n")
    return grouped_cells, cols

def remove_aps_interpolation(vm, threshold=-20, window_samples=60):
    peaks, _ = find_peaks(vm, height=threshold)
    if len(peaks) == 0:
        return vm
    vm_clean = vm.copy()
    half_win = window_samples // 2
    for p in peaks:
        start = max(0, p - half_win)
        end = min(len(vm) - 1, p + half_win)
        vm_clean[start:end] = np.linspace(vm[start], vm[end], num=(end - start))
    return vm_clean

def calculate_metrics(trace, fs=20000):
    """Trace passed here is already baseline subtracted."""
    # 4th order Bessel, 300Hz cutoff
    sos = bessel(4, 300, 'low', fs=fs, output='sos')
    trace_filtered = sosfilt(sos, trace)
    
    pre_samples = int(0.050 * fs)
    response_period = trace_filtered[pre_samples:]
    
    # Limit search for initial peak to first 30ms
    search_window_ms = 30
    search_samples = int((search_window_ms / 1000) * fs)
    search_period = response_period[:search_samples]

    # Detect Response Polarity
    max_val = np.max(search_period)
    min_val = np.min(search_period)
    
    if abs(min_val) > abs(max_val):
        # IPSP
        peak_amp = min_val
        peak_idx_local = np.argmin(response_period)
        polarity = -1
    else:
        # EPSP
        peak_amp = max_val
        peak_idx_local = np.argmax(response_period)
        polarity = 1

    time_to_peak = (peak_idx_local / fs) * 1000  # ms

    # Robust Max Slope
    early_window = int(0.050 * fs) 
    if early_window > len(response_period):
        early_window = len(response_period)
    
    dv_dt = np.gradient(response_period[:early_window]) * fs # V/s
    
    if polarity == 1:
        max_slope = np.max(dv_dt)
    else:
        max_slope = np.min(dv_dt)

    return time_to_peak, max_slope, peak_amp

def compute_spike_rates(raw_snippet, fs=20000, pre_samples=1000, spike_threshold_mv=-20, prominence_mv=10):
    n_samples = len(raw_snippet)
    t = (np.arange(n_samples) - pre_samples) / fs 
    
    # Auto-detect units & convert to mV if likely in Volts
    if np.abs(np.mean(raw_snippet)) < 1.0:
        trace_mv = raw_snippet * 1000
    else:
        trace_mv = raw_snippet
        
    peaks, _ = find_peaks(trace_mv, height=spike_threshold_mv, prominence=prominence_mv)
    spike_times = t[peaks]

    # Artifact removal (2ms around t=0)
    valid_baseline_mask = (spike_times < -0.002) 
    baseline_duration = (pre_samples / fs) - 0.002
    if baseline_duration <= 0: baseline_duration = 0.001
    
    baseline_count = np.sum(valid_baseline_mask)
    baseline_rate = baseline_count / baseline_duration

    # Post-touch window (1ms to 50ms)
    post_mask = (spike_times >= 0.001) & (spike_times < 0.050)
    post_count = np.sum(post_mask)
    post_rate = post_count / 0.049 

    return baseline_rate, post_rate, post_rate - baseline_rate