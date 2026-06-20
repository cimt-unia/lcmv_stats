# LCMV Stats Library

**lcmv_stats** is a specialized statistical analysis and machine learning toolkit designed for **LCMV source-reconstructed EEG/MEG data**. It serves as the analytical companion to **[lcmv_xtra](https://github.com/cimt-unia/lcmv_xtra)**, providing rigorous statistical inference, time-frequency clustering, and feature engineering capabilities optimized for the **CIMT Unified Atlas** (448 ROIs).

<br>

While `lcmv_xtra` handles source reconstruction and basic connectivity estimation, `lcmv_stats` focuses on:
*   Non-parametric permutation testing for connectivity edges.
*   Cluster-based mass-univariate statistics for time-frequency spectrograms.
*   Directed connectivity analysis (GPDC) for significant networks.
*   Automated feature extraction for machine learning pipelines.

## 📦 Installation

`lcmv_stats` relies on `lcmv_xtra` for atlas definitions and core connectivity computations.

```bash
# 1. Install lcmv_xtra (Required Dependency)
pip install git+https://github.com/cimt-unia/lcmv_xtra.git

# 2. Install scientific stack
pip install mne spectral_connectivity scipy pandas numpy matplotlib

# 3. Install lcmv_stats
pip install git+https://github.com/cimt-unia/lcmv_stats.git
```

## Quick Start

### 1. Group-Level Connectivity Analysis
Compare two conditions (e.g., Rest vs. Task) using edge-wise permutation tests on the CIMT Motor Network.

```python
import pandas as pd
from pathlib import Path
from lcmv_stats.batch import prepare_group_comparison
from lcmv_stats.statistics import run_edgewise_permutation
from lcmv_stats.reporting import plot_top_edges

# Load event metadata
events_df = pd.read_csv("path/to/events.csv")
lcmv_root = Path("/data/lcmv_derivatives")

# Prepare data arrays for Condition A (Rest) and Condition B (Task)
results = prepare_group_comparison(
    events_df=events_df,
    lcmv_root=lcmv_root,
    band="low_beta",
    condition_col='task_type',
    val_a='rest',
    val_b='task'
)

# Run non-parametric permutation test
stats_df = run_edgewise_permutation(
    in_data=results['data_a'],
    out_data=results['data_b'],
    n_permutations=5000,
    alpha=0.01
)

# Visualize top 5 significant edges
plot_top_edges(stats_df, band="low_beta", n_top=5)
```

### 2. Time-Frequency Cluster Analysis
Perform cluster-based permutation testing on spectrograms for a specific ROI.

```python
from lcmv_stats.timefreq import run_roi_spectrogram_analysis

# Run end-to-end TF analysis for the Left STN
df_clusters = run_roi_spectrogram_analysis(
    events_df=events_df,
    lcmv_root=lcmv_root,
    roi_name="STN-lh",
    condition="bima_off",
    f_min=13.0,       # Low Beta
    f_max=30.0,
    baseline_duration=2.0,
    save_path_csv="results/stn_lh_clusters.csv",
    save_path_plot="results/stn_lh_tf_plot.png"
)

print(df_clusters[df_clusters['significant'] == True])
```

### 3. Machine Learning Feature Engineering
Extract spectral features from continuous signals for classification tasks.

```python
from lcmv_stats.machine_learning import process_signal_to_ml_dataframe

# Process resting state vs. movement signals
df_ml = process_signal_to_ml_dataframe(
    rest_signal=rest_ts,      # 1D numpy array
    move_signal=move_ts,      # 1D numpy array
    fs=250.0,
    epoch_duration=1.5,
    overlap_frac=0.75
)

# df_ml now contains columns: ['epoch', 'delta', 'theta', 'alpha', 'beta', ... , 'target']
print(df_ml.head())
```

## Key Modules

| Module | Description |
| :--- | :--- |
| **`_atlas.py`** | Bridges with `lcmv_xtra` to load CIMT labels and motor network indices. |
| **`batch.py`** | High-level helpers for iterating subjects, extracting epochs, and preparing arrays for stats. |
| **`connectivity.py`** | Computes WPLI via `lcmv_xtra` and implements targeted GPDC for significant edges. |
| **`epoching.py`** | Handles event-locked trial extraction and continuous sliding-window epoching. |
| **`statistics.py`** | Core inference tools: Edge-wise permutation tests and Cohen's d calculation. |
| **`timefreq.py`** | Z-scored spectrogram computation and cluster-based permutation testing (MNE-backed). |
| **`visualization.py`** | Plotting tools for connectivity matrices, PSD comparisons, and feature distributions. |
| **`reporting.py`** | Generates publication-ready markdown reports and saves spectral results with cluster overlays. |
| **`machine_learning.py`** | Utilities for Z-scoring, epoching, and Welch-based spectral feature extraction into DataFrames. |

## Architecture & Workflow

The library is designed to work seamlessly with the output structure of `lcmv_xtra`.

1.  **Input**: Raw LCMV derivatives (`.npy` trial files or continuous time courses) and event metadata (`.csv`).
2.  **Preprocessing**: `epoching.py` extracts trials or sliding windows; `utils.py` handles subject mapping and sampling frequency retrieval.
3.  **Feature Extraction**: 
    *   *Connectivity*: `connectivity.py` computes WPLI/GPDC.
    *   *Spectral*: `timefreq.py` computes Z-scored power.
    *   *ML*: `machine_learning.py` extracts band-power features.
4.  **Statistical Inference**: `statistics.py` and `timefreq.py` apply non-parametric permutation tests to control for multiple comparisons.
5.  **Reporting**: `reporting.py` and `visualization.py` generate figures and tables for publication.

## 📖 Tutorials & Examples

Detailed Jupyter notebooks are available in the `notebooks/` directory:

*   **[Test Real-Data Verification](https://github.com/cimt-unia/lcmv_stats/blob/main/notebooks/Test.ipynb)**: Validation of statistical outputs against known data.
*   **[Connectivity T-test Analysis](https://github.com/cimt-unia/lcmv_stats/blob/main/notebooks/Connectivity_Analysis.md)**: Comparing Rest vs. Task connectivity in the Motor Network.
*   **[Spectrogram Permutation Test](https://github.com/cimt-unia/lcmv_stats/blob/main/notebooks/Spectrogram_Analysis.ipynb)**: Identifying significant time-frequency clusters.
*   **[Feature Engineering for ML](https://github.com/cimt-unia/lcmv_stats/blob/main/notebooks/Feature_Engineering.ipynb)**: Preparing spectral features for classifiers.

## 📄 License

This project is licensed under the MIT License

## 🙏 Acknowledgements

*   **[lcmv_xtra](https://github.com/cimt-unia/lcmv_xtra)**: For source reconstruction and CIMT atlas definitions.
*   **[MNE-Python](https://mne.tools/)**: For cluster-based permutation testing infrastructure.
*   **[spectral_connectivity](https://github.com/edenlabllc/spectral_connectivity)**: For GPDC implementation.
