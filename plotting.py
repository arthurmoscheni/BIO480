import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import bessel, sosfilt
import config
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms

def debug_plot_baseline_windows(grouped_cells, cols, fs=20000, max_cells_to_search=20):
    """
    Finds one active-touch sweep with >= 2 contacts and plots baseline windows
    to verify data quality before running full analysis.
    """
    searched_cells = 0
    for (mouse, count), cell_df in grouped_cells:
        searched_cells += 1
        if searched_cells > max_cells_to_search:
            print("Searched max_cells_to_search cells, no suitable sweep found.")
            return
        
        at_sweeps = cell_df[cell_df[cols['sweep_type']].astype(str).str.contains('active touch', case=False)]
        for _, sweep in at_sweeps.iterrows():
            raw_vm = np.array(sweep[cols['vm']])
            contacts = sweep[cols['contacts']]
            if contacts is None: continue
            contacts = np.array(contacts)
            if contacts.ndim == 0 or len(contacts) == 0: continue
            if contacts.ndim == 1: contacts = contacts.reshape(1, -1)
            
            if contacts.shape[0] < 2: continue
            
            contact_times = contacts[:2, 0]
            pre_samples = int(0.050 * fs)
            post_samples = int(0.100 * fs)
            baseline_win_samples = int(0.002 * fs)
            
            fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
            for idx, (onset_time, ax) in enumerate(zip(contact_times, axes)):
                onset_idx = int(onset_time * fs)
                start_idx = onset_idx - pre_samples
                end_idx = onset_idx + post_samples
                if start_idx < 0 or end_idx > len(raw_vm): continue
                
                snippet = raw_vm[start_idx:end_idx]
                t_ms = (np.arange(len(snippet)) - pre_samples) / fs * 1000 
                
                b_end = pre_samples
                b_start = pre_samples - baseline_win_samples
                baseline_mean = np.mean(snippet[b_start:b_end])
                
                ax.plot(t_ms, snippet, label="Raw Vm")
                ax.axvspan(t_ms[b_start], t_ms[b_end-1], color='orange', alpha=0.3, label="Baseline" if idx==0 else None)
                ax.axhline(baseline_mean, color='red', linestyle='--', label="Mean" if idx==0 else None)
                ax.axvline(0, color='k', linestyle='--', label="Contact" if idx==0 else None)
                ax.set_title(f"Cell {mouse}_{count} – Contact {idx+1}")
                ax.set_ylabel("Vm (V)")
                ax.grid(alpha=0.3)
            
            axes[-1].set_xlabel("Time from contact (ms)")
            handles, labels = axes[0].get_legend_handles_labels()
            if handles: fig.legend(handles, labels, loc="upper right")
            plt.tight_layout()
            plt.savefig(os.path.join(config.FIG_DIR, "debug_baseline_contacts.png"), dpi=300)
            plt.close()
            print(f"[DEBUG] Baseline window plot saved.")
            return

# def plot_methodology_validation(grouped_cells, cols, fs=20000):
#     """
#     Generates a 2-panel figure:
#     A. Full sweep showing ICI classification (Long vs Short).
#     B. Zoomed-in event showing filtering, baseline window, and peak detection.
#     """
#     print("Searching for a suitable sweep (mixed ICIs) for plotting...")
    
#     found_sweep = None
#     contacts_arr = None
    
#     # 1. Search for a good example sweep
#     for (mouse, count), cell_df in grouped_cells:
#         at_sweeps = cell_df[cell_df[cols['sweep_type']].astype(str).str.contains('active touch', case=False)]
        
#         for _, sweep in at_sweeps.iterrows():
#             contacts = sweep[cols['contacts']]
#             if contacts is None: continue
#             contacts = np.array(contacts)
#             if contacts.ndim == 1: contacts = contacts.reshape(1, -1)
            
#             # We want at least 3 contacts to show dynamics
#             if len(contacts) < 3: continue
            
#             # Check for mix of Long (>200ms) and Short (<200ms) ICIs
#             onsets = contacts[:, 0]
#             icis = np.diff(onsets)
            
#             if np.any(icis > 0.200) and np.any(icis < 0.200):
#                 found_sweep = sweep
#                 contacts_arr = onsets
#                 break
#         if found_sweep is not None: break
    
#     if found_sweep is None:
#         print("No sweep with mixed ICIs found. Plotting functionality skipped.")
#         return
    
#     # zoom in on 2nd contact to 5th contact
#     # Handle case where we don't have enough contacts after filtering
#     if len(contacts_arr) >= 5:
#         contacts_arr = contacts_arr[1:5]
#     else:
#         contacts_arr = contacts_arr[1:]

#     # 2. Prepare Data
#     raw_vm = np.array(found_sweep[cols['vm']])
#     time_axis = np.arange(len(raw_vm)) / fs
    
#     # Filter (Bessel 300Hz)
#     sos = bessel(4, 300, 'low', fs=fs, output='sos')
#     filt_vm = sosfilt(sos, raw_vm)
    
#     # 3. Create Plot
#     fig = plt.figure(figsize=(12, 10))
#     gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.3], hspace=0.35)
    
#     # --- Panel A: Full Trace & Stratification ---
#     ax1 = fig.add_subplot(gs[0])
#     ax1.plot(time_axis, raw_vm * 1000, color='gray', alpha=0.6, label='Raw Vm', lw=0.8)
#     # ax1.plot(time_axis, filt_vm * 1000, color='black', alpha=0.8, label='Filtered', lw=1) # Optional
    
#     last_onset = -np.inf
#     for i, onset in enumerate(contacts_arr):
#         ici = onset - last_onset
#         if i == 0: ici = 999 # First contact is always Long
        
#         is_long = ici > 0.200
#         color = 'tab:blue' if is_long else 'tab:orange'
#         label_txt = "Long\n(>200ms)" if is_long else "Short\n(<200ms)"
        
#         ax1.axvline(onset, color=color, linestyle='--', lw=2, alpha=0.8)
#         # Label roughly above the trace
#         y_txt = np.max(raw_vm)*1000 + 2
#         ax1.text(onset, y_txt, label_txt, color=color, ha='center', fontsize=9, fontweight='bold')
        
#         last_onset = onset

#     ax1.set_title("A. Event Detection & ICI Stratification", loc='left', fontweight='bold', fontsize=14)
#     ax1.set_ylabel("Vm (mV)")
#     ax1.set_xlabel("Time (s)")
#     if len(contacts_arr) > 0:
#         ax1.set_xlim(contacts_arr[0]-0.1, contacts_arr[-1]+0.2)
    
#     # --- Panel B: Feature Extraction (Zoom on first Long Event) ---
#     ax2 = fig.add_subplot(gs[1])
    
#     # Zoom window: -10ms to +50ms around first contact (or specific contact)
#     # Here we pick the last one or a specific index if available
#     target_idx = 3 if len(contacts_arr) > 3 else 0
#     target_onset = contacts_arr[target_idx] 
    
#     win_pre = 0.050
#     win_post = 0.10
    
#     idx_start = int((target_onset - win_pre) * fs)
#     idx_end = int((target_onset + win_post) * fs)
    
#     # Relative time axis for zoom
#     t_zoom = (np.arange(idx_end - idx_start) / fs) - win_pre
#     vm_zoom_raw = raw_vm[idx_start:idx_end]
#     vm_zoom_filt = filt_vm[idx_start:idx_end]
    
#     # Calculate Metrics for Visualization
#     # Baseline (-15ms to 0ms) - Adjusted logic from prompt to match typically used baseline in analysis
#     # Analysis uses -2ms to 0ms. Let's visualize that or the prompt's -15 to -10 depending on preference.
#     # The prompt code uses -15 to -10 for calculation here specifically.
#     base_mask = (t_zoom >= -0.015) & (t_zoom < -0.010)
#     baseline_val = np.mean(vm_zoom_filt[base_mask])
    
#     # Peak Detection
#     resp_mask = (t_zoom >= 0.002) & (t_zoom <= 0.050)
#     resp_slice = vm_zoom_filt[resp_mask]
#     t_slice = t_zoom[resp_mask]

#     vm_delta = resp_slice - baseline_val

#     if len(vm_delta) > 0:
#         if np.max(vm_delta) >= np.abs(np.min(vm_delta)):
#             # Excitatory (EPSP)
#             peak_idx_local = np.argmax(vm_delta)
#         else:
#             # Inhibitory (IPSP)
#             peak_idx_local = np.argmin(vm_delta)
            
#         peak_val = resp_slice[peak_idx_local]
#         peak_time = t_slice[peak_idx_local]

#         # Max Slope (0-20ms)
#         slope_mask = (t_zoom >= 0) & (t_zoom <= 0.020)
#         slope_slice = vm_zoom_filt[slope_mask]
#         dv = np.gradient(slope_slice) * fs
#         slope_idx_local = np.argmax(dv)  # Focus on depolarizing slope only
#         slope_time = t_zoom[slope_mask][slope_idx_local]
#         slope_val = vm_zoom_filt[slope_mask][slope_idx_local]

#         # Plotting B
#         ax2.plot(t_zoom * 1000, vm_zoom_raw * 1000, color='lightgray', label='Raw Trace')
#         ax2.plot(t_zoom * 1000, vm_zoom_filt * 1000, color='black', lw=2, label='Filtered (Bessel 300Hz)')
        
#         # 1. Baseline Window
#         ax2.axvspan(-50, 0, color='gold', alpha=0.3, label='Baseline Window (-50ms)')
#         ax2.axhline(baseline_val * 1000, color='gold', linestyle='--', alpha=0.9)
        
#         # 2. Contact Line
#         ax2.axvline(0, color='k', linestyle='-', lw=1)
        
#         # 3. Peak & Slope
#         ax2.scatter(peak_time*1000, peak_val*1000, color='red', s=100, zorder=5, label='Detected Peak')
#         ax2.scatter(slope_time*1000, slope_val*1000, color='green', marker='s', s=80, zorder=5, label='Max Slope')
        
#         # Annotations
#         ax2.annotate(f"Latency: {peak_time*1000:.1f}ms", 
#                      xy=(peak_time*1000, peak_val*1000), 
#                      xytext=(peak_time*1000 + 5, peak_val*1000),
#                      arrowprops=dict(arrowstyle='->', color='red'))

#         ax2.set_title("B. Feature Extraction: Baseline & Peak Detection", loc='left', fontweight='bold', fontsize=14)
#         ax2.set_xlabel("Time from Contact (ms)")
#         ax2.set_ylabel("Vm (mV)")
#         ax2.legend(loc='lower right', frameon=True)
#         ax2.grid(True, alpha=0.3)
    
#     plt.tight_layout()
#     # Updated to use config path
#     plt.savefig(os.path.join(config.FIG_DIR, "methodology_check.png"), dpi=300)
#     plt.show()

def plot_methodology_validation(grouped_cells, cols, fs=20000, 
                                target_index=0, target_type=None):
    """
    target_index (int): 0 = plot 1st match, 1 = plot 2nd match, etc.
    target_type (str): If set (e.g., 'PV'), only looks for that cell type.
    """
    print(f"Searching for match #{target_index + 1} (Type: {target_type if target_type else 'Any'})...")
    
    matches_found = 0
    found_sweep = None
    contacts_arr = None
    
    # 1. Search for a good example sweep
    for (mouse, count), cell_df in grouped_cells:
        
        # A. Filter by Cell Type (if requested)
        current_type = cell_df[cols['type']].iloc[0]
        if target_type is not None and target_type != current_type:
            continue

        at_sweeps = cell_df[cell_df[cols['sweep_type']].astype(str).str.contains('active touch', case=False)]
        
        sweep_candidate = None
        contacts_candidate = None
        
        # Check sweeps for this cell
        for _, sweep in at_sweeps.iterrows():
            contacts = sweep[cols['contacts']]
            if contacts is None: continue
            contacts = np.array(contacts)
            if contacts.ndim == 1: contacts = contacts.reshape(1, -1)
            
            if len(contacts) < 5: continue
            
            onsets = contacts[:, 0]
            icis = np.diff(onsets)
            
            # Must have Mixed ICIs
            if np.any(icis > 0.200) and np.any(icis < 0.200):
                sweep_candidate = sweep
                contacts_candidate = onsets
                break # Found a good sweep for THIS cell
        
        # B. If this cell has a good sweep, check if it's the one we want
        if sweep_candidate is not None:
            if matches_found == target_index:
                found_sweep = sweep_candidate
                contacts_arr = contacts_candidate
                # Add Cell ID to title for clarity
                print(f"Found match! Cell: {mouse}_{count} ({current_type})")
                break
            else:
                matches_found += 1 # Valid match, but not the one we want yet
    
    if found_sweep is None:
        print("Target cell not found. Try reducing target_index.")
        return
    
    # 2. Prepare Data
    raw_vm = np.array(found_sweep[cols['vm']])
    time_axis = np.arange(len(raw_vm)) / fs
    
    # Filter (Bessel 2000Hz)
    sos = bessel(4, 2000, 'low', fs=fs, output='sos')
    filt_vm = sosfilt(sos, raw_vm)
    
    # 3. Create Plot
    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.3], hspace=0.35)
    
    # --- Panel A: Full Trace & Stratification ---
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(time_axis, raw_vm * 1000, color='gray', alpha=0.5, label='Raw Vm', lw=0.8)
    ax1.plot(time_axis, filt_vm * 1000, color='black', alpha=0.8, label='Filtered (300Hz)', lw=0.8)
    
    last_onset = -np.inf
    # Plot a subset of contacts to avoid clutter if too many
    plot_contacts = contacts_arr if len(contacts_arr) < 4 else contacts_arr[:4]

    for i, onset in enumerate(plot_contacts):
        ici = onset - last_onset
        if i == 0: ici = 999 # First contact is always Long
        
        is_long = ici > 0.200
        color = 'tab:blue' if is_long else 'tab:orange'
        label_txt = "Long" if is_long else "Short"
        
        ax1.axvline(onset, color=color, linestyle='--', lw=2, alpha=0.8)
        
        # Label roughly above the trace
        y_txt = np.max(filt_vm)*1000 + 5
        ax1.text(onset, y_txt, label_txt, color=color, ha='center', fontsize=9, fontweight='bold')
        
        last_onset = onset

    ax1.set_title("A. Event Detection & ICI Stratification", loc='left', fontweight='bold', fontsize=14)
    ax1.set_ylabel("Vm (mV)")
    ax1.set_xlabel("Time (s)")
    if len(plot_contacts) > 0:
        ax1.set_xlim(plot_contacts[0]-0.2, plot_contacts[-1]+0.2)
    ax1.legend(loc='upper right')
    
    # --- Panel B: Feature Extraction (Zoom on a specific Long Event) ---
    ax2 = fig.add_subplot(gs[1])
    
    # Pick the first "Long" event found (to show clean baseline)
    target_idx = 2
    # Try to find a long event that isn't the very start of the file
    for i in range(1, len(contacts_arr)):
        if (contacts_arr[i] - contacts_arr[i-1]) > 0.200:
            target_idx = i
            break
            
    target_onset = contacts_arr[target_idx] 
    
    win_pre = 0.080  # Show 80ms before
    win_post = 0.100 # Show 100ms after
    
    idx_start = int((target_onset - win_pre) * fs)
    idx_end = int((target_onset + win_post) * fs)
    
    # Relative time axis for zoom
    t_zoom = (np.arange(idx_end - idx_start) / fs) - win_pre
    vm_zoom_filt = filt_vm[idx_start:idx_end]
    
    # --- 1. Linear Drift Calculation (Visualizing the polyfit) ---
    # Baseline window: -50ms to 0ms
    base_mask = (t_zoom >= -0.050) & (t_zoom < 0)
    base_time = t_zoom[base_mask]
    base_vm = vm_zoom_filt[base_mask]
    
    if len(base_vm) > 1:
        # Fit line: y = mx + c
        poly = np.polyfit(base_time, base_vm, 1)
        drift_line = np.polyval(poly, base_time)
        drift_slope = poly[0] # V/s
    else:
        drift_line = base_vm
        drift_slope = 0
        
    baseline_mean = np.mean(base_vm)

    # --- 2. Peak Detection ---
    resp_mask = (t_zoom >= 0.002) & (t_zoom <= 0.050)
    resp_slice = vm_zoom_filt[resp_mask]
    t_slice = t_zoom[resp_mask]

    vm_delta = resp_slice - baseline_mean
    
    peak_time = 0
    peak_val = baseline_mean
    
    if len(vm_delta) > 0:
        if np.max(vm_delta) >= np.abs(np.min(vm_delta)): # EPSP
            peak_idx_local = np.argmax(vm_delta)
        else: # IPSP
            peak_idx_local = np.argmin(vm_delta)
            
        peak_val = resp_slice[peak_idx_local]
        peak_time = t_slice[peak_idx_local]

    # --- Plotting B ---
    ax2.plot(t_zoom * 1000, vm_zoom_filt * 1000, color='black', lw=2, label='Filtered Trace')
    
    # A. Baseline Window Shading
    ax2.axvspan(-50, 0, color='gold', alpha=0.2, label='Baseline Window (-50ms)')
    
    # B. Drift Line (Red)
    ax2.plot(base_time * 1000, drift_line * 1000, color='red', lw=2.5, linestyle='-', 
             label=f'Linear Drift Fit (Slope={drift_slope:.3f} V/s)')
    
    # C. Contact Line
    ax2.axvline(0, color='k', linestyle='--', lw=1)
    
    # D. Peak Marker
    ax2.scatter(peak_time*1000, peak_val*1000, color='red', s=120, zorder=5, marker='o', label='Detected Peak')
    
    # E. Annotations
    ax2.set_title("B. Quality Control: Linear Drift & Peak Detection", loc='left', fontweight='bold', fontsize=14)
    ax2.set_xlabel("Time from Contact (ms)")
    ax2.set_ylabel("Vm (mV)")
    
    # Add text box for metrics
    textstr = '\n'.join((
        r'$\mathrm{Drift\ Slope}=%.2f\ V/s$' % (drift_slope, ),
        r'$\mathrm{Baseline\ SD}=%.3f\ mV$' % (np.std(base_vm)*1000, ),
        r'$\mathrm{Peak\ Amp}=%.2f\ mV$' % ((peak_val - baseline_mean)*1000, )))
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.9)
    ax2.text(0.02, 0.95, textstr, transform=ax2.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

    ax2.legend(loc='lower right', frameon=True)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # Save using config path
    FIG_DIR = config.FIG_DIR if hasattr(config, 'FIG_DIR') else "."
    save_path = os.path.join(FIG_DIR, "methodology_check.png")
    plt.savefig(save_path, dpi=300)
    print(f"Validation plot saved to: {save_path}")
    plt.close()


def plot_neuroscience_style_validation(grouped_cells, cols, fs=20000, 
                                       target_index=0, target_type=None, 
                                       save_dir="."):
    """
    Generates a publication-quality trace plot matching the reference style:
    - Top: Whisker Angle (Green)
    - Bottom: Vm (Color-coded by cell type)
    - Vertical bars indicating Active Touch onsets.
    - Floating scale bars (No box axes).
    """
    
    # --- 1. CONFIGURATION ---
    # Colors matching standard literature (and your image)
    colors = {
        'SST': '#E69F00',  # Orange/Gold
        'PV':  '#D55E00',  # Vermillion/Red
        'VIP': '#0072B2',  # Blue
        'EXC': 'black',    # Black/Grey
        'Whisker': '#009E73' # Green
    }
    
    # --- 2. SEARCH LOGIC (Same as before) ---
    print(f"Searching for match #{target_index + 1} (Type: {target_type if target_type else 'Any'})...")
    
    matches_found = 0
    found_sweep = None
    contacts_arr = None
    cell_id_str = ""
    cell_type_str = ""

    for (mouse, count), cell_df in grouped_cells:
        current_type = cell_df[cols['type']].iloc[0]
        if target_type is not None and target_type != current_type:
            continue

        at_sweeps = cell_df[cell_df[cols['sweep_type']].astype(str).str.contains('active touch', case=False)]
        
        sweep_candidate = None
        
        for _, sweep in at_sweeps.iterrows():
            contacts = sweep[cols['contacts']]
            if contacts is None: continue
            contacts = np.array(contacts)
            if contacts.ndim == 1: contacts = contacts.reshape(1, -1)
            
            # Simple check: enough contacts to look "busy" like the image
            if len(contacts) > 3:
                sweep_candidate = sweep
                contacts_arr = contacts[:, 0]
                break 
        
        if sweep_candidate is not None:
            if matches_found == target_index:
                found_sweep = sweep_candidate
                cell_id_str = f"{mouse} Cell {count}"
                cell_type_str = current_type
                break
            else:
                matches_found += 1

    if found_sweep is None:
        print("Target cell not found.")
        return

    # --- 3. DATA PREP ---
    raw_vm = np.array(found_sweep[cols['vm']])
    
    # Try to grab whisker angle. If missing, make a flat dummy line.
    if 'whisker_angle' in cols and cols['whisker_angle'] in found_sweep:
        raw_wh = np.array(found_sweep[cols['whisker_angle']])
        # Downsample whisker if it's at same fs as Vm (usually whisker is 500Hz, Vm 20kHz)
        # Assuming they are already aligned or same length for plotting:
        if len(raw_wh) != len(raw_vm):
            # Simple resize for visualization only
            from scipy.ndimage import zoom
            zoom_factor = len(raw_vm) / len(raw_wh)
            raw_wh = zoom(raw_wh, zoom_factor)
    else:
        raw_wh = np.zeros_like(raw_vm)

    # Filter Vm for cleaner plot (remove high freq noise)
    sos = bessel(4, 1000, 'low', fs=fs, output='sos') # 1kHz filter
    filt_vm = sosfilt(sos, raw_vm)
    
    t = np.arange(len(filt_vm)) / fs

    # --- 4. PLOTTING ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, 
                                   gridspec_kw={'height_ratios': [1, 2], 'hspace': 0.05})
    
    # Remove standard axes (Frames)
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_yticks([])
        ax.set_xticks([])

    # A. WHISKER TRACE (Top)
    ax1.plot(t, raw_wh, color=colors['Whisker'], lw=1)
    ax1.text(-0.02, 0.5, "Whisker\nangle", transform=ax1.transAxes, 
             color=colors['Whisker'], ha='right', va='center', fontsize=12, fontweight='bold')

    # B. VM TRACE (Bottom)
    c_vm = colors.get(cell_type_str, 'black')
    ax2.plot(t, filt_vm * 1000, color=c_vm, lw=0.8) # Convert to mV
    
    label_y = np.percentile(filt_vm*1000, 50)
    ax2.text(-0.02, 0.5, f"{cell_id_str}\n{cell_type_str}\nVm", transform=ax2.transAxes, 
             color=c_vm, ha='right', va='center', fontsize=12, fontweight='bold')

    # C. CONTACT LINES (Grey vertical bars)
    # Span across both axes
    trans = transforms.blended_transform_factory(ax2.transData, fig.transFigure)
    
    for onset in contacts_arr:
        # Check if onset is within plot range
        if onset > t[-1]: continue
        
        # Plot grey bar spanning both subplots visual area
        # We use a trick: axvline on both, or a rect. 
        # Simpler: just lines on both axes
        ax1.axvline(onset, color='grey', alpha=0.4, lw=2)
        ax2.axvline(onset, color='grey', alpha=0.4, lw=2)
    
    # Label "Active touch" at the bottom
    ax2.text(0, -0.1, "Active\ntouch", transform=ax2.transAxes, 
             color='grey', ha='right', va='top', fontsize=12)
    # Add little ticks at the bottom for contacts
    for onset in contacts_arr:
        if onset > t[-1]: continue
        ax2.plot([onset, onset], [np.min(filt_vm*1000)-2, np.min(filt_vm*1000)-6], 
                 color='grey', lw=1.5, clip_on=False)

    # --- 5. SCALE BARS (The "Floating" Look) ---
    
    # Time Scale (Horizontal) - e.g., 2 seconds
    # Place it bottom right
    bar_len_s = 2.0
    x_start = t[-1] - bar_len_s - 1.0 # 1s padding from right
    y_start_vm = np.min(filt_vm*1000) + 5
    
    ax2.plot([x_start, x_start + bar_len_s], [y_start_vm, y_start_vm], color='black', lw=2)
    ax2.text(x_start + bar_len_s/2, y_start_vm - 2, "2 s", ha='center', va='top', fontsize=10)

    # Vm Scale (Vertical) - e.g., 10 mV
    bar_len_mv = 10
    x_scale_vm = x_start + bar_len_s + 0.2
    y_scale_vm_btm = y_start_vm + 5
    
    ax2.plot([x_scale_vm, x_scale_vm], [y_scale_vm_btm, y_scale_vm_btm + bar_len_mv], 
             color=c_vm, lw=2)
    ax2.text(x_scale_vm + 0.05, y_scale_vm_btm + bar_len_mv/2, "10 mV", 
             color=c_vm, rotation=0, va='center', fontsize=10)

    # Whisker Scale (Vertical) - e.g., 10 degrees
    # We need range of whisker to place this well
    wh_range = np.max(raw_wh) - np.min(raw_wh)
    if wh_range == 0:
        wh_range = 1 # avoid div/0
    
    bar_len_deg = 10 
    x_scale_wh = x_scale_vm
    y_scale_wh_btm = np.mean(raw_wh)
    
    ax1.plot([x_scale_wh, x_scale_wh], [y_scale_wh_btm, y_scale_wh_btm + bar_len_deg], 
             color=colors['Whisker'], lw=2)
    ax1.text(x_scale_wh + 0.05, y_scale_wh_btm + bar_len_deg/2, "10°", 
             color=colors['Whisker'], rotation=0, va='center', fontsize=10)

    # Save
    import os
    save_path = os.path.join(save_dir, f"trace_validation_{cell_id_str.replace(' ', '_')}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Neuroscience style plot saved to: {save_path}")
    plt.close()


def plot_pv_coupling_diagnostic(df_metrics):
    if df_metrics.empty or 'PV' not in df_metrics['Cell_Type'].values:
        print("No PV data available for coupling diagnostic.")
        return

    pv_data = df_metrics[df_metrics['Cell_Type'] == 'PV'].copy()

    plt.figure(figsize=(8, 7))

    # Plot: X=Amplitude (mV), Y=Delta Firing Rate (Hz)
    plt.scatter(pv_data['Peak_Amp_V'] * 1000, 
                pv_data['Delta_FR_0_50_Hz'], 
                color='red', alpha=0.7, edgecolors='black', s=60, label='PV Cells')

    # Add Reference Lines (The "Crosshair")
    plt.axhline(0, color='black', linestyle='--', linewidth=1.5)
    plt.axvline(0, color='black', linestyle='--', linewidth=1.5)

    # Add Quadrant Interpretation Labels
    xlims = plt.xlim()
    ylims = plt.ylim()

    # Top-Left: The "AHP Trap" (High Firing, Negative Vm)
    plt.text(xlims[0] + (xlims[1]-xlims[0])*0.05, 
            ylims[1] - (ylims[1]-ylims[0])*0.05, 
            "POS Firing / NEG Vm\n(Likely AHP Contamination)", 
            fontsize=10, color='darkred', fontweight='bold', ha='left', va='top',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # Top-Right: True Excitation
    plt.text(xlims[1] - (xlims[1]-xlims[0])*0.05, 
            ylims[1] - (ylims[1]-ylims[0])*0.05, 
            "POS Firing / POS Vm\n(True Excitation)", 
            fontsize=10, color='green', fontweight='bold', ha='right', va='top',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # Bottom-Left: True Inhibition
    plt.text(xlims[0] + (xlims[1]-xlims[0])*0.05, 
            ylims[0] + (ylims[1]-ylims[0])*0.05, 
            "NEG Firing / NEG Vm\n(True Inhibition)", 
            fontsize=10, color='blue', fontweight='bold', ha='left', va='bottom',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    plt.xlabel("Peak Amplitude (mV)")
    plt.ylabel("Change in Firing Rate (Hz)")
    plt.title("PV Cells: E-I Coupling Diagnostic")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save
    save_path = os.path.join(config.FIG_DIR, "debug_PV_coupling_scatter.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[DEBUG] PV Diagnostic plot saved to: {save_path}")

def generate_main_figures(df_metrics, trace_storage):
    if df_metrics.empty:
        return

    colors = config.COLORS
    time_axis = config.TIME_AXIS
    
    # 1. Standard Waveforms (Long ICI only)
    plt.figure(figsize=(10, 6))
    for ctype in ['EXC', 'PV', 'SST', 'VIP']:
        if ctype not in trace_storage: continue
        traces = trace_storage[ctype]['long']
        if len(traces) > 0:
            type_mean = np.mean(np.vstack(traces), axis=0) * 1000  # mV
            plt.plot(time_axis, type_mean, color=colors[ctype],
                     label=f"{ctype} (n={len(traces)})", linewidth=2)
    
    plt.axvline(0, color='grey', linestyle='--', label='Touch Onset')
    plt.title("Active Touch Response Dynamics (Grand Average, Long ICI)")
    plt.xlabel("Time from Contact (ms)")
    plt.ylabel("Vm Change (mV)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(-20, 60)
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIG_DIR, "fig1_waveforms_long_ici.png"), dpi=300)
    plt.close()

    # 2. Boxplots: latency + slope (long ICI)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.boxplot(x='Cell_Type', y='Time_to_Peak_ms', data=df_metrics,
                palette=colors, showfliers=False)
    plt.title("Response Latency (Time to Peak, long ICI)")
    plt.ylabel("Time (ms)")
    
    plt.subplot(1, 2, 2)
    sns.boxplot(x='Cell_Type', y='Max_Slope', data=df_metrics,
                palette=colors, showfliers=False)
    plt.title("Response Speed (Max Slope, long ICI)")
    plt.ylabel("Slope (V/s)")
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIG_DIR, "fig2_latency_and_slope_long_ici.png"), dpi=300)
    plt.close()

    # 3. Latency jitter plot
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='Cell_Type', y='Latency_Jitter_ms', data=df_metrics,
                palette=colors, showfliers=False)
    sns.stripplot(x='Cell_Type', y='Latency_Jitter_ms', data=df_metrics,
                  color='k', alpha=0.4)
    plt.title("Latency Jitter (SD of Time-to-Peak)")
    plt.ylabel("Jitter (ms)")
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIG_DIR, "fig3_latency_jitter.png"), dpi=300)
    plt.close()

    # 4. Effect of ICI on time-to-peak
    df_ici = df_metrics.melt(
        id_vars=['Cell_Type', 'Cell_ID'],
        value_vars=['Time_to_Peak_ms', 'Time_to_Peak_short_ms'],
        var_name='ICI_condition',
        value_name='Time_to_Peak_cond_ms'
    )
    df_ici['ICI_condition'] = df_ici['ICI_condition'].map({
        'Time_to_Peak_ms': 'Long ICI',
        'Time_to_Peak_short_ms': 'Short ICI'
    })
    df_ici = df_ici.dropna(subset=['Time_to_Peak_cond_ms'])

    if not df_ici.empty:
        plt.figure(figsize=(10, 4))
        sns.boxplot(x='Cell_Type', y='Time_to_Peak_cond_ms',
                    hue='ICI_condition', data=df_ici, showfliers=False)
        plt.title("Effect of ICI on Latency")
        plt.ylabel("Time to Peak (ms)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(config.FIG_DIR, "fig4_ici_effect_latency.png"), dpi=300)
        plt.close()

    # 5. Latency vs dFR
    plt.figure(figsize=(6, 5))
    for ctype, ccol in colors.items():
        sub = df_metrics[df_metrics['Cell_Type'] == ctype]
        plt.scatter(sub['Time_to_Peak_ms'], sub['Delta_FR_0_50_Hz'],
                    label=ctype, alpha=0.7, color=ccol)
    plt.axhline(0, color='grey', linestyle='--')
    plt.xlabel("Time to Peak (ms)")
    plt.ylabel("dFR (Hz)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIG_DIR, "fig5_latency_vs_deltaFR.png"), dpi=300)
    plt.close()

    # 6. Feature Space
    plt.figure(figsize=(6, 5))
    for ctype, ccol in colors.items():
        sub = df_metrics[df_metrics['Cell_Type'] == ctype]
        plt.scatter(sub['Time_to_Peak_ms'], sub['Max_Slope'],
                    label=ctype, alpha=0.7, color=ccol)
    plt.xlabel("Time to Peak (ms)")
    plt.ylabel("Max Slope (V/s)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIG_DIR, "fig6_feature_space.png"), dpi=300)
    plt.close()

    # 7. Slope Adaptation
    plt.figure(figsize=(6, 5))
    df_ratio = df_metrics.dropna(subset=['Slope_Adaptation_Ratio'])
    sns.boxplot(x='Cell_Type', y='Slope_Adaptation_Ratio', data=df_ratio, palette=colors, showfliers=False)
    plt.axhline(1.0, color='grey', linestyle='--', label='No Change')
    plt.title("Slope Adaptation (Short / Long)")
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIG_DIR, "fig7_slope_adaptation.png"), dpi=300)
    plt.close()

    # 8. EI Coupling
    plt.figure(figsize=(6, 5))
    df_coupling = df_metrics.dropna(subset=['EI_Coupling_Strength'])
    sns.boxplot(x='Cell_Type', y='EI_Coupling_Strength', data=df_coupling, palette=colors, showfliers=False)
    plt.axhline(0, color='grey', linestyle='--', label='No Coupling')
    plt.title("E-I Coupling Strength")
    plt.ylabel("Coupling (Hz/mV)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIG_DIR, "fig8_EI_coupling.png"), dpi=300)
    plt.close()

    # 9. Amplitude Adaptation Scatter
    plt.figure(figsize=(7, 7))
    for ctype in ['EXC', 'PV', 'SST', 'VIP']:
        sub = df_metrics[df_metrics['Cell_Type'] == ctype]
        plt.scatter(sub['Peak_Amp_V']*1000, sub['Peak_Amp_short_V']*1000, 
                    color=colors[ctype], label=ctype, alpha=0.7)
        
    lims = [
        np.nanmin([df_metrics['Peak_Amp_V'].min(), df_metrics['Peak_Amp_short_V'].min()])*1000,
        np.nanmax([df_metrics['Peak_Amp_V'].max(), df_metrics['Peak_Amp_short_V'].max()])*1000
    ]
    plt.plot(lims, lims, 'k--', alpha=0.5, label="Identity")
    
    plt.xlabel("Amplitude Long ICI (mV)")
    plt.ylabel("Amplitude Short ICI (mV)")
    plt.title("Amplitude Adaptation: Short vs Long ICI")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIG_DIR, "fig12_amplitude_scatter.png"), dpi=300)
    plt.close()

    # 10. Amplitude Adaptation Ratio Boxplot
    plt.figure(figsize=(6, 5))
    df_amp_ratio = df_metrics.dropna(subset=['Amp_Adaptation_Ratio'])
    # Clean extreme outliers for plotting
    df_amp_ratio = df_amp_ratio[df_amp_ratio['Amp_Adaptation_Ratio'].between(-2, 5)] 
    
    sns.boxplot(x='Cell_Type', y='Amp_Adaptation_Ratio', data=df_amp_ratio, palette=colors, showfliers=False)
    plt.axhline(1.0, color='grey', linestyle='--', label='No Change')
    plt.title("Amplitude Adaptation Ratio (Amp_Short / Amp_Long)")
    plt.ylabel("Ratio")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIG_DIR, "fig13_amplitude_ratio.png"), dpi=300)
    plt.close()

    # 11. Waveform Overlay (Long vs Short)
    plt.figure(figsize=(12, 8))
    for i, ctype in enumerate(['EXC', 'PV', 'SST', 'VIP']):
        plt.subplot(2, 2, i+1)
        if ctype not in trace_storage: continue

        traces_l = trace_storage[ctype]['long']
        traces_s = trace_storage[ctype]['short']
        
        if len(traces_l) > 0 and len(traces_s) > 0:
            mean_l = np.mean(np.vstack(traces_l), axis=0) * 1000
            mean_s = np.mean(np.vstack(traces_s), axis=0) * 1000
            
            plt.plot(time_axis, mean_l, color=colors[ctype], linestyle='-', linewidth=2, label=f'Long (n={len(traces_l)})')
            plt.plot(time_axis, mean_s, color=colors[ctype], linestyle=':', linewidth=2, label=f'Short (n={len(traces_s)})')
            
            plt.fill_between(time_axis, mean_l, mean_s, color=colors[ctype], alpha=0.1)
            
        plt.axvline(0, color='grey', linestyle='--')
        plt.title(f"{ctype}: Short vs Long Adaptation")
        plt.xlabel("Time (ms)")
        plt.ylabel("Vm (mV)")
        plt.legend(fontsize='small')
        plt.xlim(-10, 100) 
        plt.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(os.path.join(config.FIG_DIR, "fig14_waveform_comparison.png"), dpi=300)
    plt.close()
    
    print("\nFigures Generated successfully.")
    
    # === EPSP vs IPSP separation ===
    df_metrics['Response_Type'] = np.where(df_metrics['Peak_Amp_V'] >= 0, 'EPSP-dominated', 'IPSP-dominated')

    # Histogram of mean amplitude by cell type
    plt.figure(figsize=(8, 5))
    for ctype, ccol in colors.items():
        sub = df_metrics[df_metrics['Cell_Type'] == ctype]
        plt.hist(sub['Peak_Amp_V']*1000, bins=20, alpha=0.5, label=ctype, color=ccol)
    plt.xlabel("Mean Peak Amplitude (mV)")
    plt.ylabel("Cell count")
    plt.title("Distribution of touch-evoked PSP amplitudes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIG_DIR, "fig_epsp_ipsp_amplitude_hist.png"), dpi=300)
    plt.close()

    # Repeat key latency plot for EPSP-only cells
    df_epsp = df_metrics[df_metrics['Response_Type'] == 'EPSP-dominated']

    plt.figure(figsize=(6, 4))
    sns.boxplot(x='Cell_Type', y='Time_to_Peak_ms', data=df_epsp, palette=colors, showfliers=False)
    plt.title("Response Latency (EPSP-dominated cells only)")
    plt.ylabel("Time to Peak (ms)")
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIG_DIR, "fig_latency_epsp_only.png"), dpi=300)
    plt.close()

    # Repeat slope adaptation plot for EPSP-only
    plt.figure(figsize=(6, 4))
    df_epsp_ratio = df_epsp.dropna(subset=['Slope_Adaptation_Ratio'])
    sns.boxplot(x='Cell_Type', y='Slope_Adaptation_Ratio', data=df_epsp_ratio, palette=colors, showfliers=False)
    plt.axhline(1.0, color='grey', linestyle='--')
    plt.title("Slope Adaptation (EPSP-dominated cells only)")
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIG_DIR, "fig_slope_adaptation_epsp_only.png"), dpi=300)
    plt.close()


# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy import stats

# def formatting_tweaks(ax, title=None, x_label=None, y_label=None):
#     """Helper to apply consistent scientific styling to any axis."""
#     sns.despine(ax=ax, trim=False)
#     if title: ax.set_title(title, fontweight='bold', pad=10)
#     if x_label: ax.set_xlabel(x_label, fontsize=12)
#     if y_label: ax.set_ylabel(y_label, fontsize=12)
#     ax.tick_params(axis='both', which='major', labelsize=10)
#     ax.grid(True, linestyle=':', alpha=0.4)

# def generate_main_figures(df_metrics, trace_storage, config):
#     if df_metrics.empty:
#         return

#     # --- 0. Global Style Setup ---
#     # This fixes the "microscopic font" issue globally
#     sns.set_context("notebook", font_scale=1.2)
#     sns.set_style("ticks")
    
#     colors = config.COLORS
#     time_axis = config.TIME_AXIS
    
#     # Ensure output directory exists
#     os.makedirs(config.FIG_DIR, exist_ok=True)

#     # --- 1. Standard Waveforms with SEM (Grand Average) ---
#     plt.figure(figsize=(10, 6))
#     ax = plt.gca()
    
#     for ctype in ['EXC', 'PV', 'SST', 'VIP']:
#         if ctype not in trace_storage: continue
#         traces = trace_storage[ctype]['long']
        
#         if len(traces) > 0:
#             # Convert list of arrays to matrix
#             stack = np.vstack(traces) * 1000 # Convert to mV
#             mean_trace = np.mean(stack, axis=0)
#             sem_trace = stats.sem(stack, axis=0) # Standard Error of Mean
            
#             # Plot Mean
#             ax.plot(time_axis, mean_trace, color=colors[ctype], 
#                     label=f"{ctype} (n={len(traces)})", linewidth=2.5)
            
#             # Shade the error (SEM) - crucial for showing variance
#             ax.fill_between(time_axis, mean_trace - sem_trace, mean_trace + sem_trace,
#                             color=colors[ctype], alpha=0.15, edgecolor=None)
    
#     ax.axvline(0, color='#444444', linestyle='--', linewidth=1.5, alpha=0.8, label='Touch Onset')
#     formatting_tweaks(ax, title="Active Touch Response Dynamics (Grand Average)", 
#                       x_label="Time from Contact (ms)", y_label="Vm Change (mV)")
#     ax.set_xlim(-20, 60)
#     ax.legend(frameon=False, loc='upper left')
    
#     plt.tight_layout()
#     plt.savefig(os.path.join(config.FIG_DIR, "fig1_waveforms_long_ici.png"), dpi=300)
#     plt.close()

#     # --- 2. Boxplots with Raw Data Points (Honesty check) ---
#     fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
#     # Latency
#     sns.boxplot(x='Cell_Type', y='Time_to_Peak_ms', data=df_metrics,
#                 palette=colors, showfliers=False, ax=axes[0], width=0.5, boxprops={'alpha': 0.6})
#     # Add stripplot to show the N=2 for PV clearly
#     sns.stripplot(x='Cell_Type', y='Time_to_Peak_ms', data=df_metrics,
#                   color='black', size=4, alpha=0.6, jitter=True, ax=axes[0])
#     formatting_tweaks(axes[0], title="Response Latency (Time to Peak)", y_label="Time (ms)")

#     # Slope
#     sns.boxplot(x='Cell_Type', y='Max_Slope', data=df_metrics,
#                 palette=colors, showfliers=False, ax=axes[1], width=0.5, boxprops={'alpha': 0.6})
#     sns.stripplot(x='Cell_Type', y='Max_Slope', data=df_metrics,
#                   color='black', size=4, alpha=0.6, jitter=True, ax=axes[1])
#     formatting_tweaks(axes[1], title="Response Speed (Max Slope)", y_label="Slope (V/s)")
    
#     plt.tight_layout()
#     plt.savefig(os.path.join(config.FIG_DIR, "fig2_latency_and_slope.png"), dpi=300)
#     plt.close()

#     # --- 3. Latency Jitter ---
#     plt.figure(figsize=(7, 5))
#     ax = plt.gca()
#     sns.boxplot(x='Cell_Type', y='Latency_Jitter_ms', data=df_metrics,
#                 palette=colors, showfliers=False, width=0.5, boxprops={'alpha': 0.6}, ax=ax)
#     sns.stripplot(x='Cell_Type', y='Latency_Jitter_ms', data=df_metrics,
#                   color='k', alpha=0.5, size=5, ax=ax)
    
#     formatting_tweaks(ax, title="Latency Jitter (Precision)", y_label="SD of Time-to-Peak (ms)")
#     plt.tight_layout()
#     plt.savefig(os.path.join(config.FIG_DIR, "fig3_latency_jitter.png"), dpi=300)
#     plt.close()

#     # --- 4. Effect of ICI (Paired) ---
#     # NOTE: Boxplots are bad for paired data. A slopegraph or connected dots is better, 
#     # but sticking to boxplots for summary, let's make them cleaner.
#     df_ici = df_metrics.melt(
#         id_vars=['Cell_Type', 'Cell_ID'],
#         value_vars=['Time_to_Peak_ms', 'Time_to_Peak_short_ms'],
#         var_name='ICI_condition', value_name='Time_to_Peak_cond_ms'
#     ).dropna()
    
#     df_ici['ICI_condition'] = df_ici['ICI_condition'].map(
#         {'Time_to_Peak_ms': 'Long', 'Time_to_Peak_short_ms': 'Short'}
#     )

#     if not df_ici.empty:
#         plt.figure(figsize=(10, 5))
#         ax = plt.gca()
#         sns.boxplot(x='Cell_Type', y='Time_to_Peak_cond_ms', hue='ICI_condition', 
#                     data=df_ici, showfliers=False, palette="Blues", ax=ax)
#         formatting_tweaks(ax, title="Effect of Inter-Contact Interval on Latency", y_label="Latency (ms)")
#         ax.legend(title="ICI Condition", frameon=False)
#         plt.tight_layout()
#         plt.savefig(os.path.join(config.FIG_DIR, "fig4_ici_effect_latency.png"), dpi=300)
#         plt.close()

#     # --- 11. Waveform Comparison (The "Money" Plot for Adaptation) ---
#     # We prioritize this visual because it shows the mechanism
#     fig, axes = plt.subplots(2, 2, figsize=(12, 10))
#     axes = axes.flatten()
    
#     for i, ctype in enumerate(['EXC', 'PV', 'SST', 'VIP']):
#         ax = axes[i]
#         if ctype not in trace_storage: 
#             ax.axis('off')
#             continue

#         traces_l = trace_storage[ctype]['long']
#         traces_s = trace_storage[ctype]['short']
        
#         if len(traces_l) > 0 and len(traces_s) > 0:
#             # Process Long
#             stack_l = np.vstack(traces_l) * 1000
#             mean_l = np.mean(stack_l, axis=0)
#             sem_l = stats.sem(stack_l, axis=0)
            
#             # Process Short
#             stack_s = np.vstack(traces_s) * 1000
#             mean_s = np.mean(stack_s, axis=0)
#             sem_s = stats.sem(stack_s, axis=0)
            
#             # Plot Long (Solid, darker)
#             ax.plot(time_axis, mean_l, color=colors[ctype], linestyle='-', linewidth=2.5, label='Long ICI')
#             ax.fill_between(time_axis, mean_l - sem_l, mean_l + sem_l, color=colors[ctype], alpha=0.1)

#             # Plot Short (Dashed, same color but maybe lighter or just dashed)
#             ax.plot(time_axis, mean_s, color='gray', linestyle='--', linewidth=2, label='Short ICI')
#             # We use gray for Short ICI to create contrast without color clashing
            
#             # Highlight difference
#             ax.fill_between(time_axis, mean_l, mean_s, color=colors[ctype], alpha=0.05)
            
#             formatting_tweaks(ax, title=f"{ctype} Adaptation", x_label="Time (ms)", y_label="Vm (mV)")
#             ax.set_xlim(-10, 80)
#             if i == 0: ax.legend(frameon=False) # Only legend on first plot to save space
            
#     plt.tight_layout()
#     plt.savefig(os.path.join(config.FIG_DIR, "fig14_waveform_comparison.png"), dpi=300)
#     plt.close()

#     # --- 13. Amplitude Adaptation Ratio (The "Facilitation" Story) ---
#     plt.figure(figsize=(7, 6))
#     ax = plt.gca()
    
#     df_amp_ratio = df_metrics.dropna(subset=['Amp_Adaptation_Ratio'])
#     # Filter extreme outliers for plotting clarity
#     df_amp_ratio = df_amp_ratio[df_amp_ratio['Amp_Adaptation_Ratio'].between(-2, 5)]
    
#     # Add reference line FIRST so it is behind data
#     ax.axhline(1.0, color='#555555', linestyle='--', linewidth=1.5, zorder=0, label='No Adaptation')
    
#     sns.boxplot(x='Cell_Type', y='Amp_Adaptation_Ratio', data=df_amp_ratio, 
#                 palette=colors, showfliers=False, ax=ax, width=0.5, boxprops={'alpha': 0.6})
#     sns.stripplot(x='Cell_Type', y='Amp_Adaptation_Ratio', data=df_amp_ratio,
#                   color='k', size=4, alpha=0.6, jitter=True, ax=ax)
    
#     # Annotate the SST Facilitation
#     sst_median = df_amp_ratio[df_amp_ratio['Cell_Type']=='SST']['Amp_Adaptation_Ratio'].median()
#     if not np.isnan(sst_median) and sst_median > 1.0:
#         ax.text(2, sst_median + 0.2, "Facilitation", ha='center', color=colors['SST'], fontweight='bold')
    
#     formatting_tweaks(ax, title="Functional Switch: Depression vs Facilitation", 
#                       y_label="Adaptation Ratio (Short / Long)")
    
#     plt.tight_layout()
#     plt.savefig(os.path.join(config.FIG_DIR, "fig13_amplitude_ratio.png"), dpi=300)
#     plt.close()
    
#     print("\nHigh-quality figures generated successfully.")