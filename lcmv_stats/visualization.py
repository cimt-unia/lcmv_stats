# lcmv_stats/visualization.py 

"""
Visualization tools for CIMT connectivity, spectral properties,
and ML feature distributions (Tensor-Native).

All functions accept numpy arrays. No pandas. No lcmv_xtra imports.
ROI selection uses _atlas.py exclusively.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal
from scipy.ndimage import gaussian_filter
import logging
from typing import Optional, List, Dict, Tuple

from ._atlas import get_cimt_labels, resolve_roi_indices, select_network, get_motor_network_indices

logger = logging.getLogger(__name__)


def plot_connectivity_matrix(
    matrix: np.ndarray,
    roi_names: Optional[List[str]] = None,
    band: str = "low_beta",
    condition: str = "A",
    title: str = ""
):
    """
    Plot a heatmap of a connectivity matrix.

    Args:
        matrix: (n_rois, n_rois) symmetric connectivity matrix.
        roi_names: List of ROI names for axis labels. If None, uses generic indices.
        band: Frequency band label for title.
        condition: Condition label for title.
        title: Additional title prefix.
    """
    n_rois = matrix.shape[0]

    if roi_names is None:
        roi_names = [f"ROI_{i}" for i in range(n_rois)]

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(matrix, cmap='viridis', aspect='auto')

    ax.set_xticks(range(n_rois))
    ax.set_yticks(range(n_rois))
    ax.set_xticklabels(roi_names, rotation=90, fontsize=7)
    ax.set_yticklabels(roi_names, fontsize=7)

    prefix = f"{title} - " if title else ""
    ax.set_title(f"{prefix}{condition.title()} ({band.replace('_', ' ').title()})")
    plt.colorbar(im, ax=ax, label="WPLI Value")
    plt.tight_layout()
    plt.show()


def validate_matrix_quality(matrix_path: Path) -> dict:
    """Check if a saved connectivity matrix is valid (not all zeros, no NaNs)."""
    if not matrix_path.exists():
        return {'exists': False}

    try:
        data = np.load(matrix_path)
        # Handle both raw arrays and .npz files
        if isinstance(data, np.lib.npyio.NpzFile):
            data = data[list(data.keys())[0]]
        return {
            'exists': True,
            'all_zero': bool(np.all(data == 0)),
            'has_nan': bool(np.any(np.isnan(data))),
            'mean_val': float(np.nanmean(data)),
            'shape': data.shape
        }
    except Exception as e:
        return {'exists': True, 'error': str(e)}


def plot_psd_rois(
    epochs_a: np.ndarray,
    epochs_b: np.ndarray,
    sfreq: float,
    roi_names: Optional[List[str]] = None,
    freq_max: float = 100.0,
    title: str = "Grand-Average PSD",
    ymin: float = -5.5
):
    """
    Plot PSD for specific CIMT ROIs comparing two conditions.

    Args:
        epochs_a: (n_epochs, n_rois_full, n_times) — Condition A.
        epochs_b: Same shape — Condition B.
        sfreq: Sampling frequency in Hz.
        roi_names: List of ROI names to plot. If None, plots motor network.
        freq_max: Maximum frequency to display.
        title: Figure title.
        ymin: Minimum y-axis value (log scale).
    """
    # Resolve ROI indices
    if roi_names is None:
        indices = get_motor_network_indices()
        atlas_df = get_cimt_labels()
        roi_names = atlas_df.loc[atlas_df['index'].isin(indices), 'roi_name'].tolist()
    else:
        indices = resolve_roi_indices(roi_names)

    # Get full names for display
    atlas_df = get_cimt_labels()
    name_to_full = dict(zip(atlas_df['roi_name'], atlas_df['region_full_name']))

    # Vectorized PSD computation across all epochs at once
    def compute_psd_vectorized(epochs, roi_indices):
        if epochs is None or epochs.size == 0:
            return None, None
        n_epochs, _, n_times = epochs.shape
        nperseg = min(500, n_times // 2)

        # Extract selected ROIs: (n_epochs, n_selected, n_times)
        selected = epochs[:, roi_indices, :]
        # Reshape to (n_epochs * n_selected, n_times) for batch Welch
        flat = selected.reshape(-1, n_times)

        freqs, psd_flat = signal.welch(
            flat, fs=sfreq, nperseg=nperseg, window='hann'
        )

        # Average across epochs: (n_selected, n_freqs)
        psd_per_roi = psd_flat.reshape(n_epochs, len(roi_indices), -1).mean(axis=0)
        return freqs, np.log10(psd_per_roi + 1e-15)

    freqs, psd_a = compute_psd_vectorized(epochs_a, indices)
    _, psd_b = compute_psd_vectorized(epochs_b, indices)

    if psd_a is None and psd_b is None:
        logger.warning("No valid epochs provided for PSD calculation.")
        return

    # Determine global Y limits
    all_vals = []
    if psd_a is not None:
        all_vals.extend(psd_a.flatten())
    if psd_b is not None:
        all_vals.extend(psd_b.flatten())

    ymax = np.percentile(all_vals, 99) if all_vals else 0

    # Plot
    n_rois = len(indices)
    ncols = 3
    nrows = int(np.ceil(n_rois / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = axes.flatten() if nrows * ncols > 1 else [axes]

    for i, (ax, roi_name) in enumerate(zip(axes, roi_names)):
        full_name = name_to_full.get(roi_name, roi_name)

        if psd_a is not None:
            ax.plot(freqs, psd_a[i], color='#2E8B57', label='Condition A', lw=2)
        if psd_b is not None:
            ax.plot(freqs, psd_b[i], color='#DC143C', label='Condition B', lw=2)

        ax.set_xlim(1, freq_max)
        ax.set_ylim(ymin, ymax)
        ax.set_ylabel("log₁₀(Power)")
        ax.set_title(f"{roi_name}\n{full_name}", fontsize=10, pad=6)

        # Highlight Beta and Gamma bands
        ax.axvspan(13, 30, color='lightblue', alpha=0.3)
        ax.axvspan(30, 60, color='lightcoral', alpha=0.3)

        ax.grid(True, linestyle='--', alpha=0.5)
        if i == 0:
            ax.legend(fontsize=8)

    # Hide unused subplots
    for ax in axes[n_rois:]:
        ax.set_visible(False)

    fig.suptitle(f"{title}: Condition A vs Condition B", fontsize=14)
    fig.text(0.5, 0.02, "Frequency (Hz)", ha='center', fontsize=12)
    plt.tight_layout(rect=[0.02, 0.04, 0.98, 0.95])
    plt.show()


def plot_psd_comparison(
    signals: Dict[str, np.ndarray],
    fs: float,
    freq_bands: Optional[Dict[str, Tuple[float, float]]] = None,
    freq_max: float = 120.0,
    title: str = "Power Spectral Density Comparison"
):
    """Plot Welch's PSD for multiple conditions on a single semi-log plot."""
    plt.figure(figsize=(10, 6))

    colors = ['#2E8B57', '#DC143C', '#1E90FF', '#FF8C00']

    for idx, (label, sig) in enumerate(signals.items()):
        nperseg = int(fs * 2)
        noverlap = nperseg // 2
        freqs, psd = signal.welch(sig, fs, nperseg=nperseg, noverlap=noverlap, window='hann')

        mask = freqs <= freq_max
        freqs = freqs[mask]
        psd = psd[mask]

        plt.semilogy(freqs, psd, color=colors[idx % len(colors)], lw=2, label=label)

    if freq_bands:
        for band_name, (f_low, f_high) in freq_bands.items():
            if f_high <= freq_max:
                plt.axvspan(f_low, f_high, color='gray', alpha=0.1)

    plt.xlim(0, freq_max)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (σ²/Hz)")
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.show()


def plot_spectrogram(
    signal_data: np.ndarray,
    fs: float,
    title: str = "Spectrogram",
    freq_range: Tuple[float, float] = (12, 30),
    smooth_sigma: Tuple[float, float] = (1.0, 2.0),
    percentile_clip: Tuple[float, float] = (70, 99)
):
    """Plot a smoothed time-frequency spectrogram in dB."""
    nperseg = int(fs * 1.0)
    noverlap = int(nperseg * 0.75)

    f, t, Sxx = signal.spectrogram(signal_data, fs, nperseg=nperseg, noverlap=noverlap, window='hann')

    Sxx_smooth = gaussian_filter(Sxx, sigma=smooth_sigma)
    Sxx_db = 10 * np.log10(Sxx_smooth + 1e-12)

    vmin = np.percentile(Sxx_db, percentile_clip[0])
    vmax = np.percentile(Sxx_db, percentile_clip[1])

    plt.figure(figsize=(12, 6))
    mesh = plt.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='rainbow', vmin=vmin, vmax=vmax)
    plt.ylim(freq_range)
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (sec)')
    plt.title(title)
    plt.colorbar(mesh, label='Power (dB)')
    plt.tight_layout()
    plt.show()


def plot_feature_distribution(
    features: np.ndarray,
    labels: np.ndarray,
    band_names: List[str],
    roi_names: Optional[List[str]] = None,
    title: str = "Feature Distribution by Condition"
):
    """
    Plot boxplots of band power features grouped by condition.

    Args:
        features: (n_samples, n_rois, n_bands) or (n_samples, n_bands).
        labels: (n_samples,) integer labels (0=A, 1=B).
        band_names: List of band name strings.
        roi_names: Optional ROI names. If features is 3D, averages across ROIs.
        title: Plot title.
    """
    # Average across ROIs if 3D
    if features.ndim == 3:
        features = features.mean(axis=1)  # (n_samples, n_bands)

    n_bands = features.shape[1]
    unique_labels = np.unique(labels)

    fig, axes = plt.subplots(1, n_bands, figsize=(4 * n_bands, 5), sharey=False)
    if n_bands == 1:
        axes = [axes]

    for i, band in enumerate(band_names):
        ax = axes[i]
        data_to_plot = [features[labels == lbl, i] for lbl in unique_labels]

        bp = ax.boxplot(data_to_plot, labels=[str(l) for l in unique_labels], patch_artist=True)

        colors = ['#2E8B57', '#DC143C']
        for j, patch in enumerate(bp['boxes']):
            patch.set_facecolor(colors[j % len(colors)])
            patch.set_alpha(0.7)

        ax.set_title(band.replace('_', ' ').title())
        ax.set_ylabel("Log Power")
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()
