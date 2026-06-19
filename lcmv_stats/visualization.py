# lcmv_stats/visualization.py
"""
Visualization tools for inspecting CIMT connectivity features and spectral properties.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal
import lcmv_xtra
import logging

logger = logging.getLogger(__name__)

def plot_connectivity_matrix(
    matrix: np.ndarray, 
    band: str = "low_beta", 
    condition: str = "inphase",
    title: str = ""
):
    """
    Plot a heatmap of the CIMT Motor-Basal-Executive-STN connectivity matrix.
    """
    roi_info = lcmv_xtra.connectivity.select_cimt_motor_network_rois()
    rois = roi_info['target_rois']
    
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(matrix, cmap='viridis', aspect='auto')
    
    ax.set_xticks(range(len(rois)))
    ax.set_yticks(range(len(rois)))
    ax.set_xticklabels(rois, rotation=90, fontsize=8)
    ax.set_yticklabels(rois, fontsize=8)
    
    ax.set_title(f"{title} - {condition.title()} ({band.replace('_', ' ').title()})")
    plt.colorbar(im, ax=ax, label="WPLI Value")
    plt.tight_layout()
    plt.show()

def validate_matrix_quality(matrix_path: Path) -> dict:
    """
    Check if a connectivity matrix is valid (not all zeros, no NaNs).
    """
    if not matrix_path.exists():
        return {'exists': False}
        
    try:
        data = np.load(matrix_path)
        return {
            'exists': True,
            'all_zero': bool(np.all(data == 0)),
            'has_nan': bool(np.any(np.isnan(data))),
            'mean_val': float(np.mean(data)),
            'shape': data.shape
        }
    except Exception as e:
        return {'exists': True, 'error': str(e)}

def plot_psd_rois(
    epochs_in: np.ndarray,
    epochs_out: np.ndarray,
    sfreq: float,
    freq_max: float = 100.0,
    target_rois: list = None,
    title: str = "Grand-Average PSD",
    _ymin: float = -5.5
):
    """
    Plot Power Spectral Density (PSD) for specific CIMT ROIs.
    
    Args:
        epochs_in: In-phase epochs (n_epochs, n_rois, n_times).
        epochs_out: Out-phase epochs (n_epochs, n_rois, n_times).
        sfreq: Sampling frequency in Hz.
        freq_max: Maximum frequency to plot.
        target_rois: List of ROI names to plot. If None, plots the CIMT motor network.
        title: Title for the figure.
    """
    # 1. Get Atlas Metadata
    package_dir = Path(lcmv_xtra.__file__).parent
    roi_file = package_dir / 'data' / 'cimt_atlas' / 'cimt_atlas_labels.csv'
    
    if not roi_file.exists():
        raise FileNotFoundError(f"CIMT labels not found at {roi_file}")
        
    roi_df = pd.read_csv(roi_file)
    full_roi_names = roi_df['roi_name'].tolist()
    
    # Create mapping for full names
    name_to_full = {}
    if 'region_full_name' in roi_df.columns:
        name_to_full = dict(zip(roi_df['roi_name'], roi_df['region_full_name']))
    else:
        name_to_full = dict(zip(roi_df['roi_name'], roi_df['roi_name']))

    # 2. Determine Target ROIs
    if target_rois is None:
        # Default to CIMT Motor Network
        roi_info = lcmv_xtra.connectivity.select_cimt_motor_network_rois()
        target_rois = roi_info['target_rois']
        
    # Get indices for these ROIs
    target_indices = []
    target_fullnames = []
    for roi in target_rois:
        if roi in full_roi_names:
            target_indices.append(full_roi_names.index(roi))
            target_fullnames.append(name_to_full.get(roi, roi))
            
    if not target_indices:
        raise ValueError("No valid target ROIs found in atlas.")

    # 3. Compute PSD
    def compute_psd(epochs, indices):
        if epochs is None or epochs.size == 0:
            return None, None
        n_epochs, _, n_times = epochs.shape
        nperseg = min(500, n_times // 2)
        nfft = 1024
        
        freqs, _ = signal.welch(np.zeros(n_times), fs=sfreq, nperseg=nperseg, nfft=nfft)
        psds = np.zeros((len(indices), len(freqs)))
        
        for i, idx in enumerate(indices):
            epoch_psds = []
            for e in range(n_epochs):
                _, psd = signal.welch(epochs[e, idx, :], fs=sfreq, nperseg=nperseg, nfft=nfft)
                epoch_psds.append(np.log10(psd + 1e-15))
            psds[i] = np.mean(epoch_psds, axis=0)
        return freqs, psds

    freqs, psd_in = compute_psd(epochs_in, target_indices)
    _, psd_out = compute_psd(epochs_out, target_indices)
    
    if psd_in is None and psd_out is None:
        logger.warning("No valid epochs provided for PSD calculation.")
        return

    # 4. Plotting
    n_rois = len(target_indices)
    ncols = 3
    nrows = int(np.ceil(n_rois / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3.5*nrows))
    axes = axes.flatten() if nrows * ncols > 1 else [axes]
    
    # Determine global Y limits
    all_vals = []
    if psd_in is not None: all_vals.extend(psd_in.flatten())
    if psd_out is not None: all_vals.extend(psd_out.flatten())
    
    if all_vals:
        global_ymin, global_ymax = _ymin, 0
        global_ymax = np.percentile(all_vals, 99)
 

    for i, (ax, roi_name, full_name) in enumerate(zip(axes, target_rois, target_fullnames)):
        if psd_in is not None:
            ax.plot(freqs, psd_in[i], color='#2E8B57', label='In-phase', lw=2)
        if psd_out is not None:
            ax.plot(freqs, psd_out[i], color='#DC143C', label='Out-phase', lw=2)
            
        ax.set_xlim(1, freq_max)
        ax.set_ylim(global_ymin, global_ymax)
        ax.set_ylabel("log₁₀(Power)")
        ax.set_title(f"{roi_name}\n{full_name}", fontsize=10, pad=6)
        
        # Highlight Beta and Gamma bands
        ax.axvspan(13, 30, color='lightblue', alpha=0.3)   
        ax.axvspan(30, 60, color='lightcoral', alpha=0.3) 
        
        ax.grid(True, linestyle='--', alpha=0.5)
        if i == 0: ax.legend(fontsize=8)

    # Hide unused subplots
    for ax in axes[n_rois:]:
        ax.set_visible(False)
        
    fig.suptitle(f"{title}: In-phase vs Out-phase", fontsize=14)
    fig.text(0.5, 0.02, "Frequency (Hz)", ha='center', fontsize=12)
    plt.tight_layout(rect=[0.02, 0.04, 0.98, 0.95])
    plt.show()