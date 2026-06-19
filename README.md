# LCMV Stats Library

**lcmv_stats** is a Python library designed for advanced statistical analysis of LCMV source-reconstructed EEG/MEG data. It acts as the statistical complement to **`lcmv_xtra`**, specifically optimized for the **CIMT Unified Atlas** (448 ROIs).

While `lcmv_xtra` handles source reconstruction, atlas extraction, and basic connectivity estimation, `lcmv_stats` provides robust tools for:
- **Group-level Statistical Inference**: Edge-wise permutation tests and cluster-based correction.
- **Advanced Connectivity Metrics**: Generalized Partial Directed Coherence (GPDC) for significant edges.
- **Time-Frequency Analysis**: Z-scored spectrograms and cluster-based permutation testing.
- **Automated Reporting**: Generation of markdown reports and publication-ready visualizations.

## 📦 Installation

### Step 1: Install Dependencies

`lcmv_stats` relies on `lcmv_xtra` for atlas definitions and core connectivity functions. Since `lcmv_xtra` is hosted on GitHub, it must be installed directly from the repository.

```bash
# 1. Install lcmv_xtra from GitHub
pip install git+https://github.com/cimt-unia/lcmv_xtra.git

# 2. Install other scientific dependencies
pip install mne spectral_connectivity scipy pandas numpy matplotlib
```

### Step 2: Install lcmv_stats

Clone this repository and install it in editable mode:

```bash
git clone <repository_url>
cd lcmv_stats
pip install -e .
```

## 🏗️ Architecture

| Feature | `lcmv_xtra` (Source & Prep) | `lcmv_stats` (Analysis & Stats) |
| :--- | :--- | :--- |
| **Core Function** | LCMV Beamforming, Source Extraction | Permutation Tests, GPDC, Clustering |
| **Atlas Support** | CIMT (448), DiFuMo (512), Glasser+Tian | Optimized for CIMT Motor Network |
| **Connectivity** | WPLI (Weighted Phase Lag Index) | WPLI + GPDC (Generalized Partial Directed Coherence) |
| **Output** | Source Time Courses, Connectivity Matrices | Statistical DataFrames, Cluster Maps, Reports |

## 🚀 Quick Start

```python
import lcmv_stats as ls
from pathlib import Path

# 1. Load CIMT Atlas metadata from lcmv_xtra
labels = ls.get_cimt_labels()
motor_indices = ls.get_motor_network_indices()

# 2. Extract Epochs using lcmv_stats helpers
in_ep, out_ep = ls.extract_event_epochs(
    subject_id="sub-001",
    lcmv_root=Path("/path/to/derivatives/lcmv"),
    events_df=my_events_df
)

# 3. Compute WPLI (delegated to lcmv_xtra)
sfreq = ls.get_subject_sfreq("sub-001", Path("/path/to/derivatives/lcmv"))
in_conn, out_conn = ls.extract_wpli_features(in_ep, out_ep, band="low_beta", sfreq=sfreq)

# 4. Run Group-Level Statistics
df_sig = ls.run_edgewise_permutation(in_data, out_data, n_permutations=5000)
```



## 📖 Key Modules

- **`epoching.py`**: Handles event-locked trial extraction and continuous sliding-window epoching with overlap support.
- **`connectivity.py`**: Computes WPLI via `lcmv_xtra` and implements targeted GPDC for statistically significant edges.
- **`statistics.py`**: Performs non-parametric edge-wise permutation tests and calculates effect sizes (Cohen's d).
- **`timefreq.py`**: Generates Z-scored spectrograms and runs cluster-based permutation tests for time-frequency data.
- **`visualization.py`**: Provides tools for plotting connectivity matrices, Power Spectral Density (PSD), and top significant edges.
- **`reporting.py`**: Automates the generation of markdown reports and saves spectral results with cluster overlays.

## 📋 Requirements

- Python >= 3.9
- NumPy, Pandas, SciPy, Matplotlib
- MNE-Python
- lcmv_xtra
- spectral_connectivity

