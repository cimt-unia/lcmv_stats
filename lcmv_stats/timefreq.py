# lcmv_stats/timefreq.py

"""
Time-frequency analysis tools for CIMT source-space data (Tensor-Native).

Operates on 5D epoched tensors (n_subjects, n_epochs, n_rois, n_samples)
produced by lcmv_stats.epoching.epoch_tensor.

Concatenates epochs from two conditions along the epoch axis per subject/ROI,
then computes Z-scored spectrograms with Condition A as baseline reference.
"""

import numpy as np
import scipy.signal as signal
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import logging
from mne.stats import permutation_cluster_test, combine_adjacency, ttest_1samp_no_p
from typing import List, Optional, Tuple, Literal

from .utils import load_tensor
from .epoching import epoch_tensor
from ._atlas import resolve_roi_indices, get_cimt_labels

logger = logging.getLogger(__name__)


def concatenate_condition_epochs(
    epochs_a: np.ndarray,
    epochs_b: np.ndarray,
    roi_index: int
) -> Tuple[np.ndarray, int]:
    """
    Concatenate epochs from two conditions along the epoch axis for a single ROI.

    Args:
        epochs_a: (n_epochs_a, n_rois, n_samples) — Condition A epochs for one subject.
        epochs_b: (n_epochs_b, n_rois, n_samples) — Condition B epochs for one subject.
        roi_index: Index into axis 1 for the target ROI.

    Returns:
        concat_signal: 1D array of shape (n_epochs_total * n_samples,)
                       formed by flattening [A_epochs | B_epochs] for the ROI.
        n_epochs_a: Number of Condition A epochs (for boundary calculation).
    """
    if epochs_a.ndim != 3 or epochs_b.ndim != 3:
        raise ValueError("Inputs must be 3D (n_epochs, n_rois, n_samples).")

    # Extract single ROI: (n_epochs, n_samples)
    roi_a = epochs_a[:, roi_index, :]
    roi_b = epochs_b[:, roi_index, :]

    # Flatten each condition's epochs into a continuous 1D signal
    sig_a = roi_a.ravel()  # (n_epochs_a * n_samples,)
    sig_b = roi_b.ravel()  # (n_epochs_b * n_samples,)

    concat_signal = np.concatenate([sig_a, sig_b])
    return concat_signal, roi_a.shape[0]


def compute_spectrogram_for_subject(
    sig: np.ndarray,
    sfreq: float,
    n_samples_per_epoch: int,
    n_epochs_a: int,
    f_min: float = 12.0,
    f_max: float = 30.0,
    normalize_mode: Literal["none", "condition_a"] = "condition_a",
    nperseg_override: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute spectrogram for a single subject's concatenated epoch signal.

    Args:
        sig: 1D concatenated signal [CondA_epochs | CondB_epochs] for one ROI.
        sfreq: Sampling frequency in Hz.
        n_samples_per_epoch: Number of samples per epoch.
        n_epochs_a: Number of Condition A epochs (defines baseline region).
        f_min, f_max: Frequency range of interest.
        normalize_mode:
            "none": Raw PSD.
            "condition_a": Z-score using only Condition A portion as baseline.
        nperseg_override: Override automatic window size.

    Returns:
        (f_filt, t, Sxx_out) where Sxx_out is Z-scored or raw.
    """
    if nperseg_override is not None:
        nperseg = nperseg_override
    else:
        nperseg = min(int(sfreq * 1.0), len(sig) // 2)
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
        return np.array([]), np.array([]), np.array([])

    freq_mask = (f >= f_min) & (f <= f_max)
    f_filt = f[freq_mask]
    Sxx_filt = Sxx[freq_mask, :]

    # --- Baseline Normalization ---
    if normalize_mode == "condition_a":
        # Condition A occupies the first n_epochs_a * n_samples_per_epoch samples
        boundary_sample = n_epochs_a * n_samples_per_epoch
        # Map sample boundary to time index
        samples_per_time_bin = nperseg - noverlap
        boundary_time_idx = boundary_sample / samples_per_time_bin
        ref_mask = t <= boundary_time_idx
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

    return f_filt, t, Sxx_out


def compute_group_spectrograms_from_tensors(
    tensor_path_a: str,
    tensor_path_b: str,
    roi_name: str,
    sfreq: float,
    epoch_duration: float = 2.0,
    overlap: float = 0.5,
    f_min: float = 12.0,
    f_max: float = 30.0,
    normalize_mode: Literal["none", "condition_a"] = "condition_a"
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, float]:
    """
    End-to-end: Load two condition tensors → epoch → concatenate → spectrograms.

    Args:
        tensor_path_a: Path to Condition A .npz tensor.
        tensor_path_b: Path to Condition B .npz tensor.
        roi_name: CIMT ROI name (e.g., 'L_4_ROI').
        sfreq: Sampling frequency.
        epoch_duration: Epoch duration in seconds.
        overlap: Epoch overlap fraction.
        f_min, f_max: Frequency range.
        normalize_mode: Baseline normalization mode.

    Returns:
        (spectrograms_list, f, t, boundary_sec)
        spectrograms_list: List of 2D arrays (n_freqs, n_times), one per subject.
        boundary_sec: Time in seconds where Condition A ends.
    """
    tens_a = load_tensor(tensor_path_a)
    tens_b = load_tensor(tensor_path_b)

    if not np.array_equal(tens_a['subject_ids'], tens_b['subject_ids']):
        raise ValueError("Subject IDs mismatch between tensors.")

    # Epoch both tensors (Z-scoring applied before epoching)
    ep_a = epoch_tensor(tens_a['data'], sfreq, epoch_duration, overlap, do_zscore=True)
    ep_b = epoch_tensor(tens_b['data'], sfreq, epoch_duration, overlap, do_zscore=True)

    # Resolve ROI index
    roi_idx = resolve_roi_indices([roi_name])[0]
    n_samples_per_epoch = ep_a.shape[3]

    spectrograms = []
    f_out, t_out = None, None

    for i in range(len(tens_a['subject_ids'])):
        concat_sig, n_epochs_a = concatenate_condition_epochs(
            ep_a[i], ep_b[i], roi_index=roi_idx
        )

        f, t, sxx = compute_spectrogram_for_subject(
            concat_sig, sfreq, n_samples_per_epoch, n_epochs_a,
            f_min=f_min, f_max=f_max, normalize_mode=normalize_mode
        )

        if sxx.size > 0:
            spectrograms.append(sxx)
            if f_out is None:
                f_out, t_out = f, t

    # Boundary in seconds: n_epochs_a * epoch_duration
    # (accounts for overlap: actual time covered = n_epochs_a * step_duration)
    step_sec = epoch_duration * (1 - overlap)
    boundary_sec = n_epochs_a * step_sec

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

    Args:
        spectrograms_list: List of 2D arrays (n_freqs, n_times), one per subject.
        f, t: Frequency and time bins from compute_spectrogram_for_subject.
        roi_name: Label for the ROI.
        hemisphere: 'L' or 'R'.
        boundary_sec: Time in seconds where Condition A ends and B begins.
        n_permutations: Number of permutations for cluster test.
        alpha: Significance threshold.
        threshold_start, threshold_step: TFCE/stat threshold parameters.
        tail: 0=two-tailed, -1=left, 1=right.
        smooth_sigma: Gaussian smoothing sigma for display.
        f_min, f_max: Y-axis frequency limits.
        save_path: If provided, saves figure instead of showing.
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
    ax.set_ylim([f_min, f_max])
    ax.set_title(
        f"Group Spectrogram ({hemisphere.upper()} {region})\n"
        f"ROI: {roi_name} (N={X.shape[0]}, α={alpha}, perms={n_permutations})"
    )
    ax.legend(loc='upper right')
    fig.colorbar(mesh, ax=ax, label="Mean Z-score")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved spectrogram to {save_path}")
    plt.show()
