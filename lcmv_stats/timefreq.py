# lcmv_stats/timefreq.py

"""
Time-frequency analysis tools for CIMT source-space data.
Handles spectrogram computation, Z-scoring, and cluster-based permutation testing.
Designed for modularity: decouples data loading from statistical inference.
"""

import numpy as np
import scipy.signal as signal
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import logging
from mne.stats import permutation_cluster_test, combine_adjacency, ttest_1samp_no_p
from typing import List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


def prepare_roi_signal_from_2d(
    move_epochs_2d: np.ndarray, 
    rest_epochs_2d: np.ndarray,
    sfreq: float
) -> Tuple[np.ndarray, float]:
    """
    Averages 2D ROI epochs and calculates the exact epoch duration.
    
    Args:
        move_epochs_2d: Shape (n_move_trials, n_samples)
        rest_epochs_2d: Shape (n_rest_trials, n_samples)
        sfreq: Sampling frequency in Hz.
        
    Returns:
        Tuple of (1D concatenated signal [Move|Rest], epoch_duration_in_seconds)
    """
    if move_epochs_2d.ndim != 2 or rest_epochs_2d.ndim != 2:
        raise ValueError("Inputs must be 2D arrays (n_trials, n_samples).")
    
    n_samples_per_epoch = move_epochs_2d.shape[1]
    epoch_dur_sec = n_samples_per_epoch / sfreq
    
    move_avg = move_epochs_2d.mean(axis=0)
    rest_avg = rest_epochs_2d.mean(axis=0)
    
    concat_sig = np.concatenate([move_avg, rest_avg])
    
    return concat_sig, epoch_dur_sec


def compute_zscored_spectrogram(
    sig: np.ndarray, 
    sfreq: float, 
    f_min: float = 12.0, 
    f_max: float = 30.0,
    baseline_fraction: float = 0.8,
    nperseg_override: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a Z-scored spectrogram for a 1D signal.
    
    Args:
        nperseg_override: If provided, overrides automatic nperseg calculation.
                          Use int(sfreq * 0.5) for concatenated signals to avoid edge clipping.
    """
    if nperseg_override is not None:
        nperseg = nperseg_override
    else:
        nperseg = int(sfreq * 1.0)
    
    nperseg = max(4, min(nperseg, len(sig) // 2))
    noverlap = int(nperseg * 0.75)

    f, t, Sxx = signal.spectrogram(
        sig, fs=sfreq, window="hann",
        nperseg=nperseg, noverlap=noverlap,
        scaling="density", mode="psd",
    )
    
    freq_mask = (f >= f_min) & (f <= f_max)
    f_filt = f[freq_mask]
    Sxx_filt = Sxx[freq_mask, :]

    t_max_baseline = t[-1] * baseline_fraction
    ref_mask = t <= t_max_baseline
    
    if not np.any(ref_mask):
        logger.warning("No time points in baseline window.")
        return f_filt, t, np.zeros_like(Sxx_filt)

    ref_mean = Sxx_filt[:, ref_mask].mean(axis=1, keepdims=True)
    ref_std = Sxx_filt[:, ref_mask].std(axis=1, keepdims=True)
    
    floor = np.where(np.abs(ref_mean) < 1e-30, 1e-30, np.abs(ref_mean) * 0.01)
    ref_std = np.where(ref_std < floor, floor, ref_std)
    
    Sxx_z = (Sxx_filt - ref_mean) / ref_std
    
    return f_filt, t, Sxx_z


def plot_and_test_group_spectrograms(
    spectrograms_list: List[np.ndarray],
    f: np.ndarray,
    t: np.ndarray,
    roi_name: str,
    hemisphere: str,
    epoch_dur_sec: float,
    baseline_fraction: float = 0.8,
    n_permutations: int = 1000,
    alpha: float = 0.05,
    threshold_start: float = 0.5,
    threshold_step: float = 0.1,
    tail: int = 0,
    smooth_sigma: tuple = (1.0, 2.0),
    f_min: float = 12.0,
    f_max: float = 30.0,
    save_path: Optional[str] = None
):
    """
    Plots group-average spectrogram with configurable cluster permutation test.
    
    Args:
        alpha: Significance threshold (e.g., 0.05, 0.01, 0.001).
        threshold_start: Initial TFCE/stat threshold for clustering.
        threshold_step: Step size for threshold search.
        tail: 0=two-tailed, -1=left-tailed, 1=right-tailed.
        save_path: If provided, saves figure to this path instead of only showing.
    """
    if not spectrograms_list:
        print("No valid spectrograms to plot.")
        return

    X = np.stack(spectrograms_list, axis=0)
    
    # --- CRITICAL VALIDATION ---
    expected_time_points = len(t)
    actual_time_points = X.shape[2]
    
    if expected_time_points != actual_time_points:
        raise ValueError(
            f"Time axis mismatch! t has {expected_time_points} points "
            f"but spectrograms have {actual_time_points} time points."
        )
    
    significant_clusters = []
    
    # --- STATISTICS CHECK ---
    if X.shape[0] > 1:
        nf, nt = X.shape[1], X.shape[2]
        adj_freq = np.eye(nf, k=1) + np.eye(nf, k=-1)
        adj_time = np.eye(nt, k=1) + np.eye(nt, k=-1)
        adjacency = combine_adjacency(adj_freq, adj_time)

        try:
            _, clusters, cluster_pv, _ = permutation_cluster_test(
                [X],
                n_permutations=n_permutations,
                threshold=dict(start=threshold_start, step=threshold_step),
                tail=tail,
                stat_fun=ttest_1samp_no_p,
                adjacency=adjacency,
                out_type='indices',
                n_jobs=-1,
                seed=42,
            )
            significant_clusters = [c for c, p in zip(clusters, cluster_pv) if p < alpha]
            print(f"Found {len(significant_clusters)} significant clusters at p < {alpha}")
        except Exception as e:
            print(f"Statistics failed: {e}")
    else:
        print("Skipping statistics: Only 1 subject provided. Plotting raw Z-scores only.")

    # --- PLOTTING ---
    avg_Sxx = np.mean(X, axis=0)
    smoothed = gaussian_filter(avg_Sxx, sigma=smooth_sigma)
    
    v = max(abs(np.percentile(smoothed, 2)), abs(np.percentile(smoothed, 98)))
    region = "STN" if "STN" in roi_name.upper() else ("M1" if "M1" in roi_name.upper() else "ROI")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    mesh = ax.pcolormesh(t, f, smoothed, shading="gouraud", cmap="RdBu", vmin=-v, vmax=v)

    if significant_clusters:
        sig_mask = np.zeros_like(avg_Sxx, dtype=bool)
        for cluster in significant_clusters:
            sig_mask[cluster] = True
        ax.contour(t, f, sig_mask.astype(int), levels=[0.5], colors='white', linewidths=1.5)

    ax.axvline(x=epoch_dur_sec, color='gray', linestyle=':', linewidth=2, 
               label=f'Move→Rest boundary ({epoch_dur_sec}s)')
    ref_end = epoch_dur_sec * baseline_fraction
    if baseline_fraction < 1.0:
        ax.axvline(x=ref_end, color='red', linestyle='--', linewidth=1, alpha=0.7, 
                   label=f'Z-score ref end ({ref_end:.2f}s)')

    ax.set_ylabel("Frequency (Hz)", fontsize=13)
    ax.set_xlabel("Time (s)", fontsize=13)
    ax.set_ylim([f_min, f_max])
    ax.set_title(f"Z-Scored Spectrogram ({hemisphere.upper()} {region})\n"
                 f"ROI: {roi_name} (N={X.shape[0]}, α={alpha}, perms={n_permutations})")
    ax.legend(loc='upper right')
    fig.colorbar(mesh, ax=ax, label="Mean Z-score")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()
