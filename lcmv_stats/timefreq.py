# lcmv_stats/timefreq.py

"""
Time-frequency analysis tools for CIMT source-space data (Tensor-Native).

Operates on 5D epoched tensors (n_subjects, n_epochs, n_rois, n_samples)
produced by lcmv_stats.epoching.epoch_tensor.

Computes Z-scored spectrograms using EPOCH-AVERAGED signals 
(2s CondA avg + 2s CondB avg = 4s total).
"""

import numpy as np
import scipy.signal as signal
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import logging
from mne.stats import permutation_cluster_test, combine_adjacency, ttest_1samp_no_p
from typing import List, Optional, Tuple, Literal

from ._atlas import resolve_roi_indices, get_cimt_labels

logger = logging.getLogger(__name__)


def average_condition_epochs(
    epochs_a: np.ndarray,
    epochs_b: np.ndarray,
    roi_index: int
) -> Tuple[np.ndarray, float]:
    """
    Average epochs from two conditions for a single ROI to create a 4s signal.

    Args:
        epochs_a: (n_epochs_a, n_rois, n_samples) — Condition A epochs for one subject.
        epochs_b: (n_epochs_b, n_rois, n_samples) — Condition B epochs for one subject.
        roi_index: Index into axis 1 for the target ROI.

    Returns:
        avg_signal: 1D array of shape (2 * n_samples,) formed by 
                    [mean(A_epochs[:, roi, :]) | mean(B_epochs[:, roi, :])]
        cond_duration_sec: Duration of one condition in seconds (for boundary calc).
    """
    if epochs_a.ndim != 3 or epochs_b.ndim != 3:
        raise ValueError("Inputs must be 3D (n_epochs, n_rois, n_samples).")
    
    if epochs_a.shape[2] != epochs_b.shape[2]:
        raise ValueError("Epochs must have same number of samples.")

    # Extract single ROI and average across epochs: (n_samples,)
    avg_a = epochs_a[:, roi_index, :].mean(axis=0)
    avg_b = epochs_b[:, roi_index, :].mean(axis=0)

    # Concatenate the two 2s averages into one 4s signal
    avg_signal = np.concatenate([avg_a, avg_b])
    
    # Calculate duration of ONE condition for boundary placement
    cond_duration_sec = epochs_a.shape[2] / (epochs_a.shape[2] / epochs_a.shape[2]) 
    # Simpler: just use sample count and sfreq later, but return samples here
    return avg_signal, float(epochs_a.shape[2])


def compute_spectrogram_for_subject(
    sig: np.ndarray,
    sfreq: float,
    n_samples_per_cond: int,
    f_min: float = 12.0,
    f_max: float = 30.0,
    normalize_mode: Literal["none", "condition_a"] = "condition_a",
    nperseg_override: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Compute spectrogram for a single subject's averaged 4s signal.

    Args:
        sig: 1D concatenated signal [CondA_avg | CondB_avg] (length = 2 * n_samples_per_cond).
        sfreq: Sampling frequency in Hz.
        n_samples_per_cond: Number of samples in ONE condition (defines baseline region).
        f_min, f_max: Frequency range of interest.
        normalize_mode:
            "none": Raw PSD.
            "condition_a": Z-score using only Condition A portion as baseline.
        nperseg_override: Override automatic window size.

    Returns:
        (f_filt, t, Sxx_out, boundary_time) where Sxx_out is Z-scored or raw,
        and boundary_time is the precise time (in seconds) where CondA ends.
    """
    if nperseg_override is not None:
        nperseg = nperseg_override
    else:
        # For short 4s signals, use full length or half, whichever is smaller
        nperseg = min(len(sig), int(sfreq * 1.0))
    nperseg = max(4, nperseg)
    noverlap = int(nperseg * 0.75)

    try:
        f, t, Sxx = signal.spectrogram(
            sig, fs=sfreq, window="hann",
            nperseg=nperseg, noverlap=noverlap,
            scaling="density", mode="psd"
        )
    except Exception as e:
        logger.error(f"Spectrogram computation failed: {e}")
        return np.array([]), np.array([]), np.array([]), 0.0

    freq_mask = (f >= f_min) & (f <= f_max)
    f_filt = f[freq_mask]
    Sxx_filt = Sxx[freq_mask, :]

    # --- PRECISE BOUNDARY CALCULATION ---
    # Map the exact sample boundary to the nearest spectrogram time bin
    boundary_sample = n_samples_per_cond
    samples_per_time_bin = nperseg - noverlap
    
    # Find the closest time bin index for the boundary
    boundary_bin_idx = min(int(boundary_sample / samples_per_time_bin), len(t) - 1)
    boundary_time = float(t[boundary_bin_idx])  # Use actual t value for plotting

    # --- Baseline Normalization ---
    if normalize_mode == "condition_a":
        ref_mask = np.arange(len(t)) <= boundary_bin_idx
    else:
        ref_mask = None

    if ref_mask is not None and np.any(ref_mask):
        ref_mean = Sxx_filt[:, ref_mask].mean(axis=1, keepdims=True)
        ref_std = Sxx_filt[:, ref_mask].std(axis=1, keepdims=True)
        floor_val = np.maximum(np.abs(ref_mean) * 0.01, 1e-30)
        ref_std = np.where(ref_std < floor_val, floor_val, ref_std)
        Sxx_out = (Sxx_filt - ref_mean) / ref_std
    else:
        if normalize_mode == "condition_a":
            logger.warning("No time points in Condition A baseline. Returning raw PSD.")
        Sxx_out = Sxx_filt

    return f_filt, t, Sxx_out, boundary_time


def compute_group_spectrograms_from_epochs(
    epochs_a: np.ndarray,
    epochs_b: np.ndarray,
    roi_name: str,
    sfreq: float,
    f_min: float = 12.0,
    f_max: float = 30.0,
    normalize_mode: Literal["none", "condition_a"] = "condition_a"
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, float]:
    """
    Compute group spectrograms directly from 5D epoched tensors.
    
    Each subject's spectrogram is computed from the AVERAGE of their epochs,
    resulting in a 4-second time axis (2s CondA + 2s CondB).

    Args:
        epochs_a: 5D array (n_subjects, n_epochs_a, n_rois, n_samples).
        epochs_b: 5D array (n_subjects, n_epochs_b, n_rois, n_samples).
        roi_name: CIMT ROI name (e.g., 'L_4_ROI').
        sfreq: Sampling frequency.
        f_min, f_max: Frequency range.
        normalize_mode: Baseline normalization mode.

    Returns:
        (spectrograms_list, f, t, boundary_sec)
        spectrograms_list: List of 2D arrays (n_freqs, n_times), one per subject.
        boundary_sec: Precise time in seconds where Condition A ends (~2.0s).
    """
    if epochs_a.shape[0] != epochs_b.shape[0] or \
       epochs_a.shape[2] != epochs_b.shape[2] or \
       epochs_a.shape[3] != epochs_b.shape[3]:
        raise ValueError("Epoch arrays must have same n_subjects, n_rois, and n_samples.")

    n_subjects = epochs_a.shape[0]
    n_samples_per_cond = epochs_a.shape[3]

    # Resolve ROI index
    roi_idx = resolve_roi_indices([roi_name])[0]

    spectrograms = []
    f_out, t_out = None, None
    boundary_sec = 0.0

    logger.info(f"Computing epoch-averaged spectrograms for {n_subjects} subjects at ROI '{roi_name}'...")

    for i in range(n_subjects):
        ep_a_subj = epochs_a[i]
        ep_b_subj = epochs_b[i]

        # Average epochs instead of concatenating all of them
        avg_sig, _ = average_condition_epochs(ep_a_subj, ep_b_subj, roi_index=roi_idx)

        f, t, sxx, subj_boundary = compute_spectrogram_for_subject(
            avg_sig, sfreq, n_samples_per_cond,
            f_min=f_min, f_max=f_max, normalize_mode=normalize_mode
        )

        if sxx.size > 0:
            spectrograms.append(sxx)
            if f_out is None:
                f_out, t_out = f, t
                boundary_sec = subj_boundary

    return spectrograms, f_out, t_out, boundary_sec


def plot_and_test_group_spectrograms(
    spectrograms_list: List[np.ndarray],
    f: np.ndarray,
    t: np.ndarray,
    roi_name: str,
    hemisphere: str,
    boundary_sec: float,
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
    Plots group-average spectrogram with cluster-based permutation test.
    """
    if not spectrograms_list:
        logger.warning("No valid spectrograms to plot.")
        return

    X = np.stack(spectrograms_list, axis=0)

    if X.shape[2] != len(t):
        raise ValueError(
            f"Time axis mismatch! t has {len(t)} points "
            f"but spectrograms have {X.shape[2]} time points."
        )

    significant_clusters = []

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
            logger.info(f"Found {len(significant_clusters)} significant clusters at p < {alpha}")
        except Exception as e:
            logger.error(f"Cluster permutation test failed: {e}")
    else:
        logger.info("Skipping statistics: Only 1 subject provided.")

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

    ax.axvline(x=boundary_sec, color='gray', linestyle=':', linewidth=2,
               label=f'Condition Boundary ({boundary_sec:.2f}s)')

    ax.set_ylabel("Frequency (Hz)", fontsize=13)
    ax.set_xlabel("Time (s)", fontsize=13)
    ax.set_xlim([t[0], t[-1]])  # Ensure x-axis matches actual 4s duration
    ax.set_ylim([f_min, f_max])
    ax.set_title(
        f"Group Spectrogram (Epoch-Averaged, {hemisphere.upper()} {region})\n"
        f"ROI: {roi_name} (N={X.shape[0]}, α={alpha}, perms={n_permutations})"
    )
    ax.legend(loc='upper right')
    fig.colorbar(mesh, ax=ax, label="Mean Z-score")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved spectrogram to {save_path}")
    plt.show()
