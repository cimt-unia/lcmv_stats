# LCMV Stats Library

**lcmv_stats** is a specialized statistical analysis and machine learning toolkit designed for **LCMV source-reconstructed EEG/MEG data**. It serves as the analytical companion to **[lcmv_xtra](https://github.com/cimt-unia/lcmv_xtra)**, providing rigorous statistical inference, time-frequency clustering, and feature engineering capabilities optimized for the **CIMT Unified Atlas** (448 ROIs).



<br>

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
<br>


