# LCMV Stats Library

**lcmv_stats** is the statistical analysis companion to **[lcmv_xtra](https://github.com/cimt-unia/lcmv_xtra)**. 

While `lcmv_xtra` handles LCMV source reconstruction and basic connectivity estimation, `lcmv_stats` provides the rigorous statistical framework needed for publication-ready results, specifically optimized for the **CIMT Unified Atlas** (448 ROIs).

<br>

## 📦 Installation

`lcmv_stats` requires `lcmv_xtra` as a dependency.

```bash
# 1. Install lcmv_xtra from GitHub
pip install git+https://github.com/cimt-unia/lcmv_xtra.git

# 2. Install scientific dependencies
pip install mne spectral_connectivity scipy pandas numpy matplotlib

# 3. Install lcmv_stats
pip install git+https://github.com/cimt-unia/lcmv_stats.git
```
<br>

## Tutorials & Examples

Get started with our step-by-step guides in the `notebooks/` folder:

*   **[Test](https://github.com/cimt-unia/lcmv_stats/blob/main/notebooks/Test.ipynb)**: Testing Real-Data Verification for lcmv_stats. 
*   **[Tutorial](https://github.com/cimt-unia/lcmv_stats/tree/main/notebooks)**: Flexible comparisons (e.g., Rest vs. Task).

<br>


## Key Modules

- **`epoching.py`**: Handles event-locked trial extraction and continuous sliding-window epoching with overlap support.
- **`connectivity.py`**: Computes WPLI via `lcmv_xtra` and implements targeted GPDC for statistically significant edges.
- **`statistics.py`**: Performs non-parametric edge-wise permutation tests and calculates effect sizes (Cohen's d).
- **`timefreq.py`**: Generates Z-scored spectrograms and runs cluster-based permutation tests for time-frequency data.
- **`visualization.py`**: Provides tools for plotting connectivity matrices, Power Spectral Density (PSD), and top significant edges.
- **`reporting.py`**: Automates the generation of markdown reports and saves spectral results with cluster overlays.

<br>

  ## Architecture

| Feature | `lcmv_xtra` (Source & Prep) | `lcmv_stats` (Analysis & Stats) |
| :--- | :--- | :--- |
| **Core Function** | LCMV Beamforming, Source Extraction | Permutation Tests, GPDC, Clustering |
| **Atlas Support** | CIMT (448), DiFuMo (512), Glasser+Tian | Optimized for CIMT Motor Network |
| **Connectivity** | WPLI (Weighted Phase Lag Index) | WPLI + GPDC (Generalized Partial Directed Coherence) |
| **Output** | Source Time Courses, Connectivity Matrices | Statistical DataFrames, Cluster Maps, Reports |

