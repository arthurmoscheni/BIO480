import numpy as np
import pandas as pd
from utils import remove_aps_interpolation, calculate_metrics, compute_spike_rates
from scipy import stats


def get_contact_responses(
    sweep, 
    cols, 
    fs=20000,
    short_ici_min=0.010,   # 10 ms: lower bound to ignore artifacts
    short_ici_max=0.080,   # 80 ms: max ICI for "short"
    long_ici_min=0.100,    # 100 ms: min ICI for "long"
    do_filter=False, 
    max_slope_thresh=2.0,
    keep_intermediate=False   # if False, drop 80–200 ms ICIs
):
    raw_vm = sweep[cols['vm']]
    contacts = sweep[cols['contacts']]
    
    if raw_vm is None or len(raw_vm) == 0: 
        return []
    if contacts is None: 
        return []
    try:
        if np.isscalar(contacts) and pd.isna(contacts): 
            return []
    except:
        pass

    contacts = np.array(contacts)
    if contacts.ndim == 0 or len(contacts) == 0: 
        return []
    if contacts.ndim == 1: 
        contacts = contacts.reshape(1, -1)
    
    raw_vm = np.array(raw_vm)
    events = []

    pre_samples = int(0.050 * fs)   # -50 ms
    post_samples = int(0.100 * fs)  # +100 ms  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    last_onset_time = None

    for i in range(len(contacts)):
        onset_time = contacts[i, 0]

        if last_onset_time is None:
            ici = np.inf   # first contact in sweep: no previous contact
        else:
            ici = onset_time - last_onset_time
        
        last_onset_time = onset_time

        # -----------------------------
        # ICI classification
        # -----------------------------
        
        # 1. Filter out Artifacts (e.g. < 10 ms)
        # Note: We keep np.inf (First contact) as it is > 0.010
        if ici < short_ici_min:
            continue

        is_short = (ici >= short_ici_min) and (ici <= short_ici_max)
        is_long  = (ici >= long_ici_min)

        if not (is_short or is_long):
            # This handles "intermediate" ICIs (e.g. 80–100 ms)
            if not keep_intermediate:
                continue
            ici_category = 'intermediate'
        else:
            ici_category = 'short' if is_short else 'long'

        onset_idx = int(onset_time * fs)
        start_idx = onset_idx - pre_samples
        end_idx = onset_idx + post_samples

        if start_idx < 0 or end_idx > len(raw_vm):
            continue

        raw_snippet = raw_vm[start_idx:end_idx]
        
        # # Assuming remove_aps_interpolation is defined in your scope
        # snippet_clean = remove_aps_interpolation(
        #     raw_snippet, fs=fs, slope_thresh=15, # was 20 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #     win_ms=6.0, voltage_safety_thresh=-0.040  #win_ms was 6.0 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # )
        snippet_clean = remove_aps_interpolation(raw_snippet, fs=fs)

        # Baseline (-20 ms to 0 ms)
        b_win_samples = int(0.020 * fs)  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        baseline_segment = snippet_clean[pre_samples - b_win_samples : pre_samples]

        if len(baseline_segment) > 0:
            baseline = np.mean(baseline_segment)
            baseline_sd = np.std(baseline_segment, ddof=1)
        else:
            baseline = snippet_clean[pre_samples-1]
            baseline_sd = 0.0
        
        psp = snippet_clean - baseline

        # Pre-touch slope (last 50 ms)
        pre_slope_window = int(0.05 * fs)
        pre_trace = snippet_clean[pre_samples - pre_slope_window : pre_samples]
        
        x = np.arange(len(pre_trace))
        if len(x) > 1:
            poly = np.polyfit(x, pre_trace, 1)
            pre_touch_slope = poly[0] * fs  # V/s
        else:
            pre_touch_slope = 0.0

        # QC filter
        if do_filter and abs(pre_touch_slope) > max_slope_thresh:
            continue 
        if baseline_sd > 0.005:  # 5 mV
            continue 

        # Vm-based metrics
        # Assuming calculate_metrics and compute_spike_rates are defined
        ttp, slope, amp = calculate_metrics(psp, fs=fs)
        _, _, delta_fr = compute_spike_rates(raw_snippet, fs=fs, pre_samples=pre_samples)

        events.append({
            'snippet': psp,
            'ici': ici,
            'ici_category': ici_category,
            'is_long': is_long,
            'is_short': is_short,
            'time_to_peak_ms': ttp,
            'max_slope': slope,
            'peak_amp': amp,
            'delta_fr_0_50_hz': delta_fr,
            'baseline_sd': baseline_sd,
            'pre_touch_slope': pre_touch_slope
        })

    return events

def analyze_cell_ici(cell_sweeps, cols, fs=20000):
    """
    Aggregates contacts for a single cell and applies inclusion criteria.
    Ref: 'included only recordings with at least 3 contacts in each category'
    """
    all_short = []
    all_long = []
    
    # 1. Aggregate all contacts from all sweeps for this cell
    for sweep in cell_sweeps:
        # Call your EXISTING function here
        events = get_contact_responses(
            sweep, 
            cols, 
            fs=fs, 
            short_ici_max=0.080,  # 80 ms
            long_ici_min=0.100,   # 100 ms
            keep_intermediate=False
        )
        
        # Sort into buckets
        for e in events:
            if e['ici_category'] == 'short':
                all_short.append(e)
            elif e['ici_category'] == 'long':
                all_long.append(e)

    # 2. Apply the "General Active Contact" exclusion (Ref: >= 5 contacts)
    # Note: The text implies usually analyzing Long ICI (clean touches) for general analysis
    if len(all_long) < 5:
        print(f"Cell excluded from General Analysis: Only {len(all_long)} clean contacts.")
        # Depending on your goal, you might return None here if strict about the 5-contact rule
    
    # 3. Apply the "ICI Analysis" exclusion (Ref: >= 3 contacts in EACH category)
    if len(all_short) < 3 or len(all_long) < 3:
        return None  # Discard this cell entirely for ICI comparison
        
    # 4. Return valid datasets
    return {
        'short_ici_events': all_short,
        'long_ici_events': all_long
    }

def run_full_analysis(grouped_cells, columns, config):
    """
    Iterates through all cells, computes metrics, and performs statistical tests.
    """
    cell_metrics = {
        'Cell_Type': [], 'Cell_ID': [], 'N_contacts_long': [], 'N_contacts_short': [],
        'Time_to_Peak_ms': [], 'Latency_Jitter_ms': [], 'Max_Slope': [], 
        'Peak_Amp_V': [], 'Peak_Amp_short_V': [], 'Amp_Adaptation_Ratio': [], 
        'Time_to_Peak_short_ms': [], 'Max_Slope_short': [], 'Delta_FR_0_50_Hz': [],
        'Slope_Adaptation_Ratio': [], 'EI_Coupling_Strength': [], 'Baseline_SD_long': []
    }

    trace_storage = {
        'EXC': {'long': [], 'short': []}, 'PV': {'long': [], 'short': []}, 
        'SST': {'long': [], 'short': []}, 'VIP': {'long': [], 'short': []}
    }

    print("\nStarting Active Touch Analysis...")
    processed_count = 0
    total_events = 0

    for (mouse, count), cell_df in grouped_cells:
        cell_type = cell_df[columns['type']].iloc[0]
        cell_id = f"{mouse}_{count}"
        
        at_sweeps = cell_df[cell_df[columns['sweep_type']].astype(str).str.contains('active touch', case=False)]
        if len(at_sweeps) == 0: continue 

        all_events = []
        for _, sweep in at_sweeps.iterrows():
            events = get_contact_responses(
                sweep, columns, fs=config.FS,
                short_ici_max=0.080,
                long_ici_min=0.100,
                do_filter=config.FILTER_UNSTABLE_TRIALS, 
                max_slope_thresh=config.MAX_PRE_TOUCH_SLOPE,
                keep_intermediate=False
            )
            all_events.extend(events)

        if len(all_events) > 0:
            total_events += len(all_events)
            long_events = [ev for ev in all_events if ev['is_long']]
            short_events = [ev for ev in all_events if not ev['is_long']]

            # Require minimum clean contacts for analysis
            if len(long_events) < 5: continue 

            try:
                # --- Long ICI Metrics (Baseline Response) ---
                latencies_long = np.array([ev['time_to_peak_ms'] for ev in long_events])
                slopes_long = np.array([ev['max_slope'] for ev in long_events])
                amps_long = np.array([ev['peak_amp'] for ev in long_events])
                delta_fr_long = np.array([ev['delta_fr_0_50_hz'] for ev in long_events])
                baseline_sd_long = np.array([ev['baseline_sd'] for ev in long_events])
                
                mean_ttp = latencies_long.mean()
                jitter = latencies_long.std(ddof=1) if len(latencies_long) > 1 else 0.0
                mean_slope = slopes_long.mean()
                mean_amp = amps_long.mean()
                mean_delta_fr = delta_fr_long.mean()
                mean_baseline_sd = baseline_sd_long.mean()

                # Store waveforms for grand average
                snippets_long = np.vstack([ev['snippet'] for ev in long_events])
                grand_avg_long = np.mean(snippets_long, axis=0)
                if len(grand_avg_long) == 3000 and cell_type in trace_storage:
                    trace_storage[cell_type]['long'].append(grand_avg_long)

                # --- Short ICI Metrics (Adaptation) ---
                if len(short_events) >= 3:
                    latencies_short = np.array([ev['time_to_peak_ms'] for ev in short_events])
                    slopes_short = np.array([ev['max_slope'] for ev in short_events])
                    amps_short = np.array([ev['peak_amp'] for ev in short_events])
                    
                    mean_ttp_short = latencies_short.mean()
                    mean_slope_short = slopes_short.mean()
                    mean_amp_short = amps_short.mean()

                    # Store waveforms
                    snippets_short = np.vstack([ev['snippet'] for ev in short_events])
                    grand_avg_short = np.mean(snippets_short, axis=0)
                    if len(grand_avg_short) == 3000 and cell_type in trace_storage:
                        trace_storage[cell_type]['short'].append(grand_avg_short)

                    # Calculate Adaptation Ratios
                    if abs(mean_slope) > 1e-6:
                        slope_ratio = mean_slope_short / mean_slope
                    else: slope_ratio = np.nan
                    
                    if abs(mean_amp) > 1e-6:
                        amp_ratio = abs(mean_amp_short) / abs(mean_amp)
                    else: amp_ratio = np.nan
                else:
                    mean_ttp_short = np.nan
                    mean_slope_short = np.nan
                    mean_amp_short = np.nan
                    slope_ratio = np.nan
                    amp_ratio = np.nan

                # EI Coupling Strength
                mean_amp_mv = mean_amp * 1000 
                if abs(mean_amp_mv) > 0.1:
                    ei_coupling = mean_delta_fr / mean_amp_mv
                else:
                    ei_coupling = np.nan

                # Save Metrics
                cell_metrics['Cell_Type'].append(cell_type)
                cell_metrics['Cell_ID'].append(cell_id)
                cell_metrics['N_contacts_long'].append(len(long_events))
                cell_metrics['N_contacts_short'].append(len(short_events))
                cell_metrics['Time_to_Peak_ms'].append(mean_ttp)
                cell_metrics['Latency_Jitter_ms'].append(jitter)
                cell_metrics['Max_Slope'].append(mean_slope)
                cell_metrics['Peak_Amp_V'].append(mean_amp)
                cell_metrics['Peak_Amp_short_V'].append(mean_amp_short)
                cell_metrics['Time_to_Peak_short_ms'].append(mean_ttp_short)
                cell_metrics['Max_Slope_short'].append(mean_slope_short)
                cell_metrics['Delta_FR_0_50_Hz'].append(mean_delta_fr)
                cell_metrics['Slope_Adaptation_Ratio'].append(slope_ratio)
                cell_metrics['Amp_Adaptation_Ratio'].append(amp_ratio)
                cell_metrics['EI_Coupling_Strength'].append(ei_coupling)
                cell_metrics['Baseline_SD_long'].append(mean_baseline_sd)

                processed_count += 1
                if processed_count % 20 == 0: print(f"Processed {processed_count} cells...")
                    
            except ValueError: pass

    print(f"\nAnalysis Complete. Processed {processed_count} cells.")
    print(f"Total contact events analyzed: {total_events}\n")
    
    df_results = pd.DataFrame(cell_metrics)
    
    # # --- Statistical Tests ---
    # print("Running Statistical Tests...\n")
    
    # # 1. Adaptation (Paired Test: Long vs Short within Cell Types)
    # print("--- Adaptation Analysis (Paired t-test: Long vs. Short) ---")
    
    # for ctype in df_results['Cell_Type'].unique():
    #     # Filter for cells that have valid short ICI data
    #     subset = df_results[(df_results['Cell_Type'] == ctype) & 
    #                         (df_results['Peak_Amp_short_V'].notna())]
        
    #     if len(subset) > 2:
    #         # We use ttest_rel for paired samples
    #         stat_amp, p_amp = stats.ttest_rel(subset['Peak_Amp_V'], subset['Peak_Amp_short_V'])
    #         stat_slope, p_slope = stats.ttest_rel(subset['Max_Slope'], subset['Max_Slope_short'])
            
    #         print(f"[{ctype}] (n={len(subset)})")
    #         print(f"  Amplitude: p={p_amp:.4e} {'*' if p_amp < 0.05 else ''}")
    #         print(f"  Slope:     p={p_slope:.4e} {'*' if p_slope < 0.05 else ''}")
    #     else:
    #         print(f"[{ctype}] Not enough data for paired test (n={len(subset)})")

    # print("\n" + "-"*40 + "\n")

    # # 2. Cell Type Differences (Kruskal-Wallis)
    # # Non-parametric ANOVA equivalent for comparing multiple independent groups
    # metrics_to_compare = [
    #     'Peak_Amp_V', 'Max_Slope', 'Time_to_Peak_ms', 
    #     'Delta_FR_0_50_Hz', 'Amp_Adaptation_Ratio'
    # ]
    
    # print("--- Cell Type Comparisons (Kruskal-Wallis) ---")
    
    # for metric in metrics_to_compare:
    #     # Prepare groups
    #     groups = []
    #     labels = []
    #     for ctype in df_results['Cell_Type'].unique():
    #         data = df_results[df_results['Cell_Type'] == ctype][metric].dropna()
    #         if len(data) > 0:
    #             groups.append(data)
    #             labels.append(ctype)
        
    #     if len(groups) > 1:
    #         stat, p_val = stats.kruskal(*groups)
    #         print(f"{metric}: p={p_val:.4e} {'*' if p_val < 0.05 else ''}")
            
    #         # Simple mean display for quick interpretation if significant
    #         if p_val < 0.05:
    #             means = [f"{lbl}:{g.mean():.2e}" for lbl, g in zip(labels, groups)]
    #             print(f"  Means -> {', '.join(means)}")
    #     else:
    #         print(f"{metric}: Not enough groups to compare.")

    return df_results, trace_storage