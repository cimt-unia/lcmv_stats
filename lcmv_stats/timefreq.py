# lcmv_stats/timefreq.py

"""
Time-frequency analysis tools for CIMT source-space data (Tensor-Native).

Operates on 4D epoch arrays (n_epochs, n_rois, n_samples) extracted from
5D epoched tensors produced by lcmv_stats.epoching.epoch_tensor.

Z-scoring is assumed to have been applied BEFORE epoching (in epoch_tensor),
so spectrograms here represent normalized power relative to the full
continuous recording baseline.
"""

import numpy as np
import scipy.signal as signal
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import logging
from mne.stats import permutation_cluster_test, combine_adjacency, ttest_1samp_no_p
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


def compute_roi_spectrogram(
    epochs: np.ndarray,
    roi_index: int,
    sfreq: float,
    f_min: float = 12.0,
    f_max: float = 30.0,
    nperseg_override: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute grand-average spectrogram for a single ROI across all epochs.

    Since Z-scoring was applied to continuous data before epoching,
    the resulting spectrogram reflects deviation from the full-recording
    mean/std — no additional baseline correction is needed.

    Args:
        epochs: Shape (n_epochs, n_rois, n_samples). From epoch_tensor[i].
        roi_index: Index into axis 1 for the target ROI.
        sfreq: Sampling frequency in Hz.
        f_min: Minimum frequency of interest.
        f_max: Maximum frequency of interest.
        nperseg_override: If provided, overrides automatic window size.

    Returns:
        (f_filt, t, Sxx_avg) where Sxx_avg is (n_freqs, n_times).
    """
    if epochs.ndim != 3:
        raise ValueError(f"Expected 3D epochs (n_epochs, n_rois, n_samples), got shape {epochs.shape}")

    # Extract single ROI: (n_epochs, n_samples)
    roi_data = epochs[:, roi_index, :]
    n_epochs, n_samples = roi_data.shape

    if n_samples < 4:
        raise ValueError(f"Epoch too short ({n_samples} samples) for spectrogram.")

    # Adaptive window: 1 second or half the epoch length, whichever is smaller
    if nperseg_override is not None:
        nperseg = nperseg_override
    else:
        nperseg = min(int(sfreq * 1.0), n_samples // 2)
    nperseg = max(4, nperseg)
    noverlap = int(nperseg * 0.75)

    # Compute PSD per epoch, then average
    freqs_all = None
    psd_sum = None

    for e in range(n_epochs):
        try:
            f, _, Sxx = signal.spectrogram(
                roi_data[e], fs=sfreq, window="hann",
                nperseg=nperseg, noverlap=noverlap,
                scaling="density", mode="psd"
            )
        except Exception as ex:
            logger.warning(f"Spectrogram failed for epoch {e}: {ex}")
            continue

        if freqs_all is None:
            freqs_all = f
            psd_sum = Sxx.copy()
        else:
            psd_sum += Sxx

    if freqs_all is None:
        return np.array([]), np.array([]), np.array([])

    Sxx_avg = psd_sum / n_epochs

    # Filter to frequency band of interest
    freq_mask = (freqs_all >= f_min) & (freqs_all <= f_max)
    f_filt = freqs_all[freq_mask]
    Sxx_filt = Sxx_avg[freq_mask, :]

    # Convert to log scale for visualization stability
    Sxx_log = np.log10(Sxx_filt + 1e-15)

    # Time axis
    t = np.arange(Sxx_filt.shape[1]) * (nperseg - noverlap) / sfreq

    return f_filt, t, Sxx_log


def plot_and_test_group_spectrograms(
    spectrograms_list: List[np.ndarray],
    f: np.ndarray,
    t: np.ndarray,
    roi_name: str,
    hemisphere: str,
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

    Each element in spectrograms_list is a 2D array (n_freqs, n_times)
    from compute_roi_spectrogram for one subject.

    Args:
        spectrograms_list: List of 2D arrays, one per subject.
        f: Frequency bins from compute_roi_spectrogram.
        t: Time bins from compute_roi_spectrogram.
        roi_name: Label for the ROI being plotted.
        hemisphere: 'L' or 'R' for title formatting.
        n_permutations: Number of permutations for cluster test.
        alpha: Significance threshold.
        threshold_start: Initial TFCE/stat threshold.
        threshold_step: Step size for threshold search.
        tail: 0=two-tailed, -1=left, 1=right.
        smooth_sigma: Gaussian smoothing sigma for display.
        f_min, f_max: Frequency limits for y-axis.
        save_path: If provided, saves figure instead of showing.
    """
    if not spectrograms_list:
        logger.warning("No valid spectrograms to plot.")
        return

    X = np.stack(spectrograms_list, axis=0)  # (N_subjects, N_freqs, N_times)

    # Validate time axis alignment
    if X.shape[2] != len(t):
        raise ValueError(
            f"Time axis mismatch! t has {len(t)} points "
            f"but spectrograms have {X.shape[2]} time points."
        )

    significant_clusters = []

    # Cluster-based permutation test (only if >1 subject)
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

    # Grand average + smoothing for display
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

    ax.set_ylabel("Frequency (Hz)", fontsize=13)
    ax.set_xlabel("Time within epoch (s)", fontsize=13)
    ax.set_ylim([f_min, f_max])
    ax.set_title(
        f"Group Spectrogram ({hemisphere.upper()} {region})\n"
        f"ROI: {roi_name} (N={X.shape[0]}, α={alpha}, perms={n_permutations})"
    )
    ax.legend(loc='upper right')
    fig.colorbar(mesh, ax=ax, label="Mean log₁₀(PSD)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved spectrogram to {save_path}")
    plt.show()
