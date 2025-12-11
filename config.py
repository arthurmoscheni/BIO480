import os
import numpy as np

# --- Paths ---
FIG_DIR = "figures_idea3_updated"
DATA_DIR = "Data"
INPUT_FILE = os.path.join(DATA_DIR, 'data_bio482.pkl')
METRICS_FILE = os.path.join(DATA_DIR, 'cell_metrics.csv')

# Ensure directories exist
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# --- Analysis Parameters ---
FS = 20000  # Sampling frequency
FILTER_UNSTABLE_TRIALS = True
MAX_PRE_TOUCH_SLOPE = 1.0  # V/s

# --- Visualization ---
COLORS = {'EXC': 'black', 'PV': 'red', 'SST': 'orange', 'VIP': 'blue'}
TIME_AXIS = np.linspace(-50, 100, 3000)  # For plotting waveforms