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
    print(f"Columns in data: {all_cols}")
    contact_col = next((c for c in all_cols if 'Contact' in c and 'Passive' not in c), None)
    vm_col = next((c for c in all_cols if 'Membrane' in c or 'Vm' in c), None)
    type_col = next((c for c in all_cols if 'Sweep_Type' in c or 'Type' in c), 'Sweep_Type')
    
    cols = {
        'mouse': 'Mouse_Name',
        'counter': 'Cell_Counter',
        'type': 'Cell_Type',
        'sweep_type': 'Sweep_Type',
        'vm': vm_col,
        'contacts': contact_col,
        'whisker_angle': 'Sweep_WhiskerAngle'
    }

    grouped_cells = df.groupby([cols['mouse'], cols['counter']])
    print(f"Data Loaded. Found {len(grouped_cells)} unique cells.\n")
    return grouped_cells, cols

# def remove_aps_interpolation(vm, threshold=-20, window_samples=60):
#     peaks, _ = find_peaks(vm, height=threshold)
#     if len(peaks) == 0:
#         return vm
#     vm_clean = vm.copy()
#     half_win = window_samples // 2
#     for p in peaks:
#         start = max(0, p - half_win)
#         end = min(len(vm) - 1, p + half_win)
#         vm_clean[start:end] = np.linspace(vm[start], vm[end], num=(end - start))
#     return vm_clean


# def remove_aps_interpolation(vm, fs=20000, slope_thresh=20, win_ms=6.0, voltage_safety_thresh=-0.040):
#     """
#     Removes APs by detecting the inflection point (dV/dt) rather than peak voltage.
#     Linearly interpolates from the initiation threshold to a fixed window after.

#     Parameters:
#     -----------
#     vm : array-like
#         Membrane potential trace (assumed in Volts).
#     fs : int
#         Sampling rate in Hz (default 20000).
#     slope_thresh : float
#         Threshold for dV/dt in V/s to define AP initiation (default 20 V/s).
#         Typical cortical neurons initiate > 10-20 V/s.
#     win_ms : float
#         Duration of the cut window in ms (default 6 ms covers the full AP + AHP).
#     voltage_safety_thresh : float
#         Minimum voltage (V) required to accept a slope crossing as an AP (default -40 mV).
#         Prevents noise in deep hyperpolarization from triggering cuts.

#     Returns:
#     --------
#     vm_clean : array
#         Trace with APs removed via linear interpolation.
#     """
#     # print("using a vm threshold of ", voltage_safety_thresh)
#     vm_clean = vm.copy()
#     n_samples = len(vm)
    
#     # 1. Calculate 1st Derivative (dV/dt) in V/s
#     dv = np.gradient(vm) * fs
    
#     # 2. Find points where slope exceeds threshold AND voltage is high enough
#     # This double-check ensures we don't cut noise during Down-states
#     spike_mask = (dv > slope_thresh) & (vm > voltage_safety_thresh)
#     spike_indices = np.where(spike_mask)[0]
    
#     if len(spike_indices) == 0:
#         return vm_clean

#     # 3. Group consecutive indices to find unique event onsets
#     # We only care about the *first* point where the slope crosses the threshold
#     # np.diff(spike_indices) > 1 finds breaks in continuity
#     breaks = np.where(np.diff(spike_indices) > 1)[0]
#     onsets = [spike_indices[0]] + [spike_indices[i+1] for i in breaks]
    
#     # 4. Interpolate
#     win_samples = int((win_ms / 1000) * fs)
    
#     for onset in onsets:
#         # Define the cut window
#         # Start: slightly before the threshold crossing to smooth the join
#         start = max(0, onset - 2) 
#         # End: fixed window after onset
#         end = min(n_samples - 1, onset + win_samples)
        
#         # Linear Interpolation
#         # Connects the voltage at 'start' directly to the voltage at 'end'
#         vm_clean[start:end] = np.linspace(vm[start], vm[end], num=(end - start))
        
#     return vm_clean



def remove_aps_interpolation(vm, fs=20000, slope_thresh=20, v_peak_min=-0.020, max_win_ms=10.0):
    """
    Robustly removes APs by detecting peaks, finding the exact initiation threshold 
    (slope crossing), and dynamically searching for the return to baseline.

    Parameters:
    -----------
    vm : array-like
        Membrane potential trace (Volts).
    fs : int
        Sampling rate (Hz).
    slope_thresh : float
        dV/dt threshold (V/s) to define AP initiation (Prompt says 10, typically 10-20).
    v_peak_min : float
        Minimum peak voltage (V) to confirm an event is actually an AP (prevents noise cuts).
    max_win_ms : float
        Maximum window to look for return to baseline before forcing a cut.

    Returns:
    --------
    vm_clean : array
        Trace with APs removed via linear interpolation from Threshold -> Return.
    """
    vm_clean = vm.copy()
    n_samples = len(vm)
    dt = 1 / fs
    
    # 1. Calculate Derivative
    dv = np.gradient(vm) * fs  # V/s

    # 2. Find Peaks First (The Anchor)
    # detecting peaks ensures we only process real APs, not just noisy slopes.
    peaks, _ = find_peaks(vm, height=v_peak_min, distance=int(0.002 * fs)) # 2ms refractory
    
    if len(peaks) == 0:
        return vm_clean

    max_win_samples = int((max_win_ms / 1000) * fs)

    for peak_idx in peaks:
        # --- A. Find Initiation (Walk Backward) ---
        # Look back from peak to find where slope drops below threshold
        # We search in a reasonable window (e.g., 2ms before peak)
        search_back = int(0.002 * fs)
        start_search = max(0, peak_idx - search_back)
        
        # Get the slope segment before the peak
        slope_segment = dv[start_search : peak_idx]
        
        # Find the LAST point in this segment where slope was < threshold
        # This corresponds to the moment just before the explosive rise
        below_thresh_indices = np.where(slope_segment < slope_thresh)[0]
        
        if len(below_thresh_indices) > 0:
            # The initiation is the index just after the last sub-threshold point
            rel_idx = below_thresh_indices[-1] + 1
            onset_idx = start_search + rel_idx
        else:
            # Fallback: if slope is consistently high, just take start of search
            onset_idx = start_search
            
        # Refine onset: Ensure we don't start ON the peak
        onset_idx = min(onset_idx, peak_idx - 1)
        onset_idx = max(0, onset_idx)
        
        # Capture the voltage at initiation
        v_start = vm[onset_idx]

        # --- B. Find Return to Baseline (Walk Forward) ---
        # Look forward from peak to find when Vm returns to v_start
        end_search = min(n_samples, peak_idx + max_win_samples)
        
        post_peak_segment = vm[peak_idx : end_search]
        
        # Find points where voltage drops below the start voltage
        return_indices = np.where(post_peak_segment <= v_start)[0]
        
        if len(return_indices) > 0:
            # Found a return point
            return_idx = peak_idx + return_indices[0]
        else:
            # Did not return (e.g., on a plateau or burst). Use max window.
            return_idx = end_search - 1

        # --- C. Interpolate ---
        # Draw a straight line from Onset to Return
        vm_clean[onset_idx : return_idx] = np.linspace(
            vm[onset_idx], vm[return_idx], num=(return_idx - onset_idx)
        )

    return vm_clean




def calculate_metrics(trace, fs=20000):
    """Trace passed here is already baseline subtracted."""
    # 4th order Bessel, 2000Hz cutoff
    sos = bessel(4, 2000, 'low', fs=fs, output='sos')
    trace_filtered = sosfilt(sos, trace)
    
    pre_samples = int(0.050 * fs)
    response_period = trace_filtered[pre_samples:]
    
    # Limit search for initial peak to first 30ms
    search_window_ms = 40  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    search_samples = int((search_window_ms / 1000) * fs)

    if search_samples > len(response_period):
        search_samples = len(response_period)
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
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Define the post-touch window (e.g., 0 to 100ms, or max available)
    window_end_s = 0.100 
    
    # Check actual duration of the snippet provided
    snippet_duration_s = (len(raw_snippet) - pre_samples) / fs
    
    # If we have less data than the window, truncate the window
    if snippet_duration_s < window_end_s:
        window_end_s = snippet_duration_s

    # Post-touch window (1ms to window_end)
    post_mask = (spike_times >= 0.001) & (spike_times < window_end_s)
    post_count = np.sum(post_mask)
    
    # Divide by the ACTUAL duration analyzed (minus the 1ms blanking)
    duration = window_end_s - 0.001
    if duration <= 0: duration = 0.001 # prevent div/0
    
    post_rate = post_count / duration  

    return baseline_rate, post_rate, post_rate - baseline_rate

    # post_mask = (spike_times >= 0.001) & (spike_times < 0.100)
    # post_count = np.sum(post_mask)
    # post_rate = post_count / 0.099

    # return baseline_rate, post_rate, post_rate - baseline_rate