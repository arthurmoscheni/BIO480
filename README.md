# Active Touch Electrophysiology Analysis Pipeline

This repository contains a modular Python pipeline for analyzing intracellular membrane potential ($V_m$) responses during active touch tasks. The code processes electrophysiological sweep data to extract response metrics, visualize waveforms, and classify cell types (EXC, PV, SST, VIP) based on their physiological signatures.
Download the data using this link (https://drive.google.com/drive/u/2/folders/0AFT2-uCkxrQ0Uk9PVA) and put the data inside the Data/ folder

## 📂 Project Structure

├── Data/                   # Input .pkl files and output .csv metrics  
├── figures_idea3_updated/  # Directory where generated plots are saved  
├── config.py               # Global configuration (paths, thresholds, colors)  
├── utils.py                # Low-level signal processing (filtering, AP removal)  
├── analysis.py             # Event detection and metric aggregation logic  
├── plotting.py             # Visualization functions (waveforms, QC plots)  
├── classifiers.py          # Machine learning (SVM, RF, Logistic Regression)  
├── main.py                 # Entry point to run the pipeline  
└── README.md               # Project documentation  

---

## ⚙️ Installation

Ensure you have **Python 3.8+** installed. Then install the required dependencies:

    pip install pandas numpy matplotlib seaborn scipy scikit-learn

---

## 🚀 Usage

### 1. Prepare Data

- Place your raw data pickle file (e.g. `data_bio482.pkl`) inside the `Data/` folder.  
- If your file name differs, update `INPUT_FILE` in `config.py`.

### 2. Run Analysis

From the project root:

    python main.py

### 3. View Results

- **Metrics**: A CSV file (`cell_metrics.csv`) containing per-cell statistics will be saved in `Data/`.  
- **Figures**: Check `figures_idea3_updated/` for quality control plots, waveform averages, and classifier results.

---

## 🧪 Methodology

The pipeline performs the following steps:

### 1. Signal Preprocessing

- **AP Removal**  
  Action potentials are detected and interpolated to isolate subthreshold $V_m$.

- **Filtering**  
  4th-order Bessel low-pass filter (300 Hz cutoff).

- **Baseline Subtraction**  
  Uses a short window immediately preceding contact onset to estimate baseline $V_m$.

### 2. Event Stratification

- **Long ICI**: Inter-contact interval (ICI) > 200 ms (assumed “rested” state).  
- **Short ICI**: ICI < 200 ms (for adaptation analysis).

### 3. Feature Extraction

For each touch event, the pipeline computes:

- **Latency** – Time to peak response (ms).  
- **Slope** – Maximum derivative ($dV/dt$) of the rising phase (V/s).  
- **Adaptation Ratios** – Ratio of Short-ICI response / Long-ICI response.  
- **E–I Coupling** – Correlation between membrane depolarization magnitude and firing rate change.

### 4. Classification

- Uses **Random Forest**, **SVM**, and **Logistic Regression** to classify cell types based on extracted physiological features.  
- Performs **PCA** for dimensionality reduction and visualization.

---

## 📊 Key Outputs

| Figure Name                     | Description                                                              |
|---------------------------------|--------------------------------------------------------------------------|
| `methodology_check.png`         | QC: Raw trace, ICI stratification, and peak detection zoom.             |
| `fig1_waveforms...`             | Grand average $V_m$ response traces per cell type.                       |
| `fig2_latency...`               | Boxplots comparing response latency and slope.                           |
| `fig7_slope_adaptation.png`     | Adaptation: how response speed changes with high-frequency touch.       |
| `fig14_waveform_comparison.png` | Overlay comparison of Long vs Short ICI waveforms.                      |
| `fig10_confusion_matrix.png`    | ML classification performance for cell-type prediction.                 |

---

## 🛠 Configuration

You can adjust analysis parameters in `config.py` without modifying the core logic:

- `ICI_THRESHOLD`: Time in seconds to define “Long” vs “Short” intervals (default: `0.200`).  
- `FILTER_UNSTABLE_TRIALS`: Enable/disable pre-touch slope filtering (boolean).  
- `MAX_PRE_TOUCH_SLOPE`: Threshold for rejecting trials with drifting baselines.
