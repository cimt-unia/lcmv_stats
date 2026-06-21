"""
Time-frequency analysis tools for CIMT source-space data.
Handles spectrogram computation, Z-scoring, and cluster-based permutation testing.
Designed for modularity: decouples data loading from statistical inference.
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import gaussian_filter
from typing import Optional, Tuple, List
from pathlib import Path
from mne.stats import permutation_cluster_test, combine_adjacency, ttest_1samp_no_p
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


def compute_zscored_spectrogram(
    epoch: np.ndarray,
    sfreq: float,
    f_min: float = 1.0,
    f_max: float = 100.0,
    pre_sec: float = 5.0,
    baseline_duration: Optional[float] = None,
    baseline_mode: str = "absolute"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a Z-scored spectrogram for a single epoch/trial.
    
    Args:
        epoch: 1D array of time-series data.
        sfreq: Sampling frequency.
        f_min/f_max: Frequency range of interest.
        pre_sec: Duration of the pre-event window (used for time-axis labeling).
        baseline_duration: 
            - If mode='absolute': Duration in seconds (None = full pre-event).
            - If mode='fractional': Fraction of total epoch (0.0 to 1.0).
        baseline_mode: 'absolute' (time-based) or 'fractional' (percentage-based).
        
    Returns:
        f: Frequency bins.
        t: Time bins (relative to event onset).
        Sxx_z: Z-scored power spectral density.
    """
    nperseg = int(sfreq * 1.0)
    if nperseg > len(epoch):
        nperseg = max(4, len(epoch) // 2)
    noverlap = int(nperseg * 0.75)

    f, t, Sxx = signal.spectrogram(
        epoch, fs=sfreq, window="hann",
        nperseg=nperseg, noverlap=noverlap,
        scaling="density", mode="psd"
    )

    freq_mask = (f >= f_min) & (f <= f_max)
    f, Sxx = f[freq_mask], Sxx[freq_mask, :]
    t_relative = t - pre_sec
    
    # Flexible Baseline Logic
    if baseline_mode == "fractional":
        fraction = baseline_duration if baseline_duration is not None else 1.0
        ref_end_time = t[0] + ((t[-1] - t[0]) * fraction)
        ref_mask = t <= ref_end_time
    else:
        if baseline_duration is None:
            ref_mask = t_relative < 0
        else:
            ref_mask = (t_relative >= -pre_sec) & (t_relative <= -pre_sec + baseline_duration)

    if not np.any(ref_mask):
        return f, t_relative, np.zeros_like(Sxx)

    ref_mean = Sxx[:, ref_mask].mean(axis=1, keepdims=True)
    ref_std = Sxx[:, ref_mask].std(axis=1, keepdims=True)
    
    # Numerical stability
    std_floor = np.maximum(np.abs(ref_mean) * 0.01, 1e-30)
    ref_std = np.where(ref_std < std_floor, std_floor, ref_std)

    return f, t_relative, (Sxx - ref_mean) / ref_std


def run_cluster_spectrogram(
    spectrograms_3d: np.ndarray,
    n_permutations: int = 1000,
    threshold: dict = {'start': 0.5, 'step': 0.1},
) -> Tuple:
    """Perform cluster-based permutation testing on time-frequency spectrograms."""
    n_units, n_freqs, n_times = spectrograms_3d.shape
    
    adj_freq = np.eye(n_freqs, k=1) + np.eye(n_freqs, k=-1)
    adj_time = np.eye(n_times, k=1) + np.eye(n_times, k=-1)
    adjacency = combine_adjacency(adj_freq, adj_time)

    return permutation_cluster_test(
        [spectrograms_3d],
        n_permutations=n_permutations,
        threshold=threshold,
        tail=0,
        stat_fun=ttest_1samp_no_p,
        adjacency=adjacency,
        out_type='indices',
        n_jobs=-1,
        seed=42
    )


def format_cluster_results_for_publication(
    T_obs: np.ndarray,
    clusters: List[Tuple],
    cluster_pv: np.ndarray,
    f: np.ndarray,
    t: np.ndarray,
    save_path: Optional[Path] = None,
    avg_sxx: Optional[np.ndarray] = None,
    roi_name: str = "ROI",
    n_units: int = 0,
    mode_label: str = "Subject",
    smooth_sigma: Optional[Tuple[float, float]] = (1.0, 2.0),
    colormap_percentile: float = 2.0,
    plot_save_path: Optional[Path] = None,
    vline_x: Optional[float] = None,
    vline_label: Optional[str] = None
) -> pd.DataFrame:
    """
    Convert raw MNE cluster permutation test outputs into a publication-ready DataFrame
    and optionally generate the corresponding time-frequency plot.
    """
    if not clusters:
        logger.info("No clusters found. Returning empty DataFrame.")
        return pd.DataFrame()
    
    results = []
    for c_idx, cluster in enumerate(clusters):
        freq_idx, time_idx = cluster
        freq_bounds = f[[freq_idx.min(), freq_idx.max()]]
        time_bounds = t[[time_idx.min(), time_idx.max()]]
        cluster_t_stats = T_obs[freq_idx][:, time_idx]
        
        results.append({
            'cluster_id': c_idx,
            'p_value': float(cluster_pv[c_idx]),
            'significant': bool(cluster_pv[c_idx] < 0.05),
            'n_points': int(len(freq_idx) * len(time_idx)),
            'freq_min_hz': round(float(freq_bounds[0]), 2),
            'freq_max_hz': round(float(freq_bounds[1]), 2),
            'time_min_s': round(float(time_bounds[0]), 3),
            'time_max_s': round(float(time_bounds[1]), 3),
            'mean_t_stat': round(float(np.mean(cluster_t_stats)), 4),
            'peak_t_stat': round(float(np.max(np.abs(cluster_t_stats))), 4)
        })
    
    df = pd.DataFrame(results).sort_values('p_value')
    
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        logger.info(f"Saved publication table to {save_path}")
    
    if avg_sxx is not None:
        _plot_tf_clusters_integrated(
            f=f, t=t, avg_sxx=avg_sxx,
            clusters=clusters, cluster_pv=cluster_pv,
            roi_name=roi_name, n_units=n_units,
            mode_label=mode_label, smooth_sigma=smooth_sigma,
            colormap_percentile=colormap_percentile,
            save_path=plot_save_path,
            vline_x=vline_x, vline_label=vline_label
        )
    
    return df


def _plot_tf_clusters_integrated(
    f: np.ndarray,
    t: np.ndarray,
    avg_sxx: np.ndarray,
    clusters: List,
    cluster_pv: np.ndarray,
    roi_name: str,
    n_units: int,
    mode_label: str,
    smooth_sigma: Optional[Tuple[float, float]],
    colormap_percentile: float,
    save_path: Optional[Path],
    vline_x: Optional[float] = None,
    vline_label: Optional[str] = None
):
    """Internal helper to generate the TF plot."""
    def _get_clim(data: np.ndarray, percentile: float = 2.0) -> Tuple[float, float]:
        lo = np.percentile(data, percentile)
        hi = np.percentile(data, 100.0 - percentile)
        v = max(abs(lo), abs(hi))
        return -v, v
    
    avg_sxx_plot = gaussian_filter(avg_sxx, sigma=smooth_sigma) if smooth_sigma else avg_sxx
    vmin, vmax = _get_clim(avg_sxx_plot, colormap_percentile)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    mesh = ax.pcolormesh(t, f, avg_sxx_plot, shading="gouraud", cmap="RdBu_r", vmin=vmin, vmax=vmax)
    
    if clusters:
        sig_mask = np.zeros_like(avg_sxx, dtype=bool)
        for c_idx, cluster in enumerate(clusters):
            if cluster_pv[c_idx] < 0.05:
                sig_mask[cluster] = True
        ax.contour(t, f, sig_mask.astype(int), levels=[0.5], colors='white', linewidths=1.5)
        
    ax.set_ylabel("Frequency (Hz)", fontsize=13)
    ax.set_xlabel("Time (s)", fontsize=13)
    ax.set_title(f"Global Average ({mode_label}) + Significant Clusters (p<0.05)\nROI: {roi_name} | N={n_units}", fontsize=14)
    
    if vline_x:
        ax.axvline(x=vline_x, color='gray', linestyle=':', linewidth=2, label=vline_label or "Boundary")
    else:
        ax.axvline(x=0, color="black", linestyle="--", linewidth=2, label="Event onset")
        
    ax.legend(loc="upper right")
    fig.colorbar(mesh, ax=ax, label="Mean Z-score")
    fig.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved TF plot to {save_path}")
    else:
        plt.show()
    plt.close()


def run_group_tf_analysis(
    epochs_list: List[np.ndarray],
    sfreq: float,
    roi_name: str = "ROI",
    f_min: float = 13.0,
    f_max: float = 30.0,
    pre_sec: float = 5.0,
    baseline_duration: Optional[float] = 2.0,
    baseline_mode: str = "absolute",
    n_permutations: int = 1000,
    smooth_sigma: Tuple[float, float] = (1.0, 2.0),
    save_path_csv: Optional[Path] = None,
    save_path_plot: Optional[Path] = None,
    vline_x: Optional[float] = None,
    vline_label: Optional[str] = None
) -> pd.DataFrame:
    """
    Generic group-level TF analysis. Takes a list of 1D epochs (one per subject/unit)
    and handles the rest: spectrograms, stats, and plotting.
    """
    if len(epochs_list) < 2:
        raise ValueError(f"Need ≥2 units for cluster test, got {len(epochs_list)}")

    units_for_stats = []
    global_f = global_t = None

    logger.info(f"Processing {len(epochs_list)} units for TF analysis...")

    for i, avg_epoch in enumerate(epochs_list):
        f, t, Sxx_z = compute_zscored_spectrogram(
            epoch=avg_epoch, sfreq=sfreq,
            f_min=f_min, f_max=f_max,
            pre_sec=pre_sec, 
            baseline_duration=baseline_duration,
            baseline_mode=baseline_mode
        )
        
        if global_f is None:
            global_f, global_t = f, t
        units_for_stats.append(Sxx_z)

    X = np.stack(units_for_stats, axis=0)
    T_obs, clusters, cluster_pv, H0 = run_cluster_spectrogram(X, n_permutations)
    
    sig_count = sum(1 for p in cluster_pv if p < 0.05)
    logger.info(f"Found {sig_count} significant clusters (p<0.05)")
    
    avg_sxx = np.mean(X, axis=0)
    
    return format_cluster_results_for_publication(
        T_obs=T_obs, clusters=clusters, cluster_pv=cluster_pv,
        f=global_f, t=global_t,
        save_path=save_path_csv,
        avg_sxx=avg_sxx, roi_name=roi_name,
        n_units=len(units_for_stats), mode_label="Subject",
        smooth_sigma=smooth_sigma, colormap_percentile=2.0,
        plot_save_path=save_path_plot,
        vline_x=vline_x, vline_label=vline_label
    )
