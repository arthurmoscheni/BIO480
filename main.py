import os
import pandas as pd
import warnings

# Local imports
import config
from utils import load_and_group_cells
from analysis import run_full_analysis
from plotting import (debug_plot_baseline_windows, generate_main_figures, 
                      plot_pv_coupling_diagnostic, plot_methodology_validation, plot_neuroscience_style_validation)
from classifiers import run_classifier_analysis

warnings.filterwarnings("ignore")

def main():
    # 1. Load Data
    try:
        grouped_cells, columns = load_and_group_cells(config.INPUT_FILE)
    except Exception as e:
        print(f"CRITICAL ERROR LOADING DATA: {e}")
        return
    print("Columns detected:")
    for col_name in columns:
        print(f" - {col_name}")
    # 2. Run or Load Analysis
    if os.path.exists(config.METRICS_FILE):
        print(f"Metrics file found at {config.METRICS_FILE}. Loading...")
        df_metrics = pd.read_csv(config.METRICS_FILE)
        # Note: If loading from CSV, trace_storage won't be available for waveform plots
        # You might need to re-run analysis if you need traces.
        trace_storage = {'EXC': {'long':[], 'short':[]}, 'PV': {'long':[], 'short':[]}, 
                         'SST': {'long':[], 'short':[]}, 'VIP': {'long':[], 'short':[]}}
    else:
        df_metrics, trace_storage = run_full_analysis(grouped_cells, columns, config)
        # Optional: Save results
        # df_metrics.to_csv(config.METRICS_FILE, index=False)

    if df_metrics.empty:
        print("No metrics computed. Exiting.")
        return

    # 3. QC Plots
    debug_plot_baseline_windows(grouped_cells, columns)
    plot_methodology_validation(grouped_cells, columns, target_index=3, target_type='SST')
    plot_neuroscience_style_validation(grouped_cells, columns, target_index=3, target_type='EXC')

    # 4. Main Figures
    generate_main_figures(df_metrics, trace_storage, config)
    plot_pv_coupling_diagnostic(df_metrics)



    # 5. Classifiers
    # 4-Class
    use_bayes_search = False
    run_classifier_analysis(df_metrics, suffix="", use_bayes_search=use_bayes_search)
    # 3-Class (EXC, PV, SST)
    run_classifier_analysis(df_metrics, allowed_classes=['EXC', 'PV', 'SST'], suffix="_3class", use_bayes_search=use_bayes_search)
    # 2-Class (EXC vs. INH)
    run_classifier_analysis(df_metrics, allowed_classes=['EXC', 'SST'], suffix="_2class", use_bayes_search=use_bayes_search)
    # 6. Stats Summary
    print("\n--- Summary Statistics ---")
    print(df_metrics.groupby('Cell_Type')[['Slope_Adaptation_Ratio', 'EI_Coupling_Strength']].median())


if __name__ == "__main__":
    main()