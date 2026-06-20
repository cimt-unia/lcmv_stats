# lcmv_stats/timefreq.py

"""
Time-frequency analysis tools for CIMT source-space data.
Handles spectrogram computation, Z-scoring, and cluster-based permutation testing.
"""

import numpy as np
from scipy import signal
from typing import Optional, Tuple, List
from mne.stats import permutation_cluster_test, combine_adjacency, ttest_1samp_no_p
import logging

logger = logging.getLogger(__name__)


def compute_zscored_spectrogram(
    epoch: np.ndarray,
    sfreq: float,
    f_min: float = 1.0,
    f_max: float = 100.0,
    pre_sec: float = 5.0,
    baseline_duration: Optional[float] = None,
    safe_baseline_margin: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a Z-scored spectrogram for a single epoch with robust edge handling.

    Args:
        epoch: 1D array of time-series data (single trial or subject-average).
        sfreq: Sampling frequency.
        f_min/f_max: Frequency range of interest.
        pre_sec: Duration of the pre-event window (used to locate baseline).
        baseline_duration: Duration of baseline for Z-scoring (None = entire pre-event minus margin).
        safe_baseline_margin: Seconds to exclude before t=0 to avoid edge artifacts.

    Returns:
        f: Frequency bins.
        t: Time bins (relative to event onset).
        Sxx_z: Z-scored power spectral density.
    """
    # 1. Compute Raw Spectrogram
    nperseg = int(sfreq * 1.0)
    if nperseg > len(epoch):
        nperseg = max(4, len(epoch) // 2)
    noverlap = int(nperseg * 0.75)

    f, t, Sxx = signal.spectrogram(
        epoch, fs=sfreq, window="hann",
        nperseg=nperseg, noverlap=noverlap,
        scaling="density", mode="psd"
    )

    # Filter frequencies
    freq_mask = (f >= f_min) & (f <= f_max)
    f = f[freq_mask]
    Sxx = Sxx[freq_mask, :]

    # 2. Z-Score Normalization with SAFE BASELINE
    t_relative = t - pre_sec

    # Define baseline mask WITH MARGIN to avoid edge artifacts near t=0
    safe_baseline_end = -safe_baseline_margin

    if baseline_duration is None:
        ref_mask = t_relative < safe_baseline_end
    else:
        ref_start = -pre_sec
        ref_end = min(-pre_sec + baseline_duration, safe_baseline_end)
        ref_mask = (t_relative >= ref_start) & (t_relative <= ref_end)

    if not np.any(ref_mask):
        logger.warning(
            f"No valid baseline bins found (margin={safe_baseline_margin}s). "
            f"Returning zeros."
        )
        return f, t_relative, np.zeros_like(Sxx)

    ref_data = Sxx[:, ref_mask]
    ref_mean = np.mean(ref_data, axis=1, keepdims=True)
    ref_std = np.std(ref_data, axis=1, keepdims=True)

    # Prevent division by zero AND replace NaN/Inf in output
    std_floor = np.abs(ref_mean) * 0.01
    std_floor = np.where(std_floor < 1e-30, 1e-30, std_floor)
    ref_std = np.where(ref_std < std_floor, std_floor, ref_std)

    Sxx_z = (Sxx - ref_mean) / ref_std

    # CRITICAL FIX: Replace any remaining NaN/Inf with 0
    Sxx_z = np.nan_to_num(Sxx_z, nan=0.0, posinf=0.0, neginf=0.0)

    return f, t_relative, Sxx_z


def run_cluster_spectrogram(
    spectrograms_3d: np.ndarray,
    adjacency: Optional[np.ndarray] = None,
    n_permutations: int = 1000,
    threshold: dict = {'start': 0.5, 'step': 0.1},
) -> Tuple:
    """
    Perform cluster-based permutation testing on time-frequency spectrograms.

    Args:
        spectrograms_3d: Array of shape (n_subjects, n_freqs, n_times).
        adjacency: Pre-computed adjacency matrix. If None, a lattice adjacency is created.
        n_permutations: Number of permutations.
        threshold: TFCE threshold parameters.

    Returns:
        T_obs, clusters, cluster_pv, H0
    """
    if spectrograms_3d.ndim != 3:
        raise ValueError("Input must be a 3D array (n_subjects, n_freqs, n_times)")

    n_subj, n_freqs, n_times = spectrograms_3d.shape

    if adjacency is None:
        adj_freq = np.eye(n_freqs, k=1) + np.eye(n_freqs, k=-1)
        adj_time = np.eye(n_times, k=1) + np.eye(n_times, k=-1)
        adjacency = combine_adjacency(adj_freq, adj_time)

    logger.info(f"Running cluster permutation test on {n_subj} subjects...")

    T_obs, clusters, cluster_pv, H0 = permutation_cluster_test(
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

    return T_obs, clusters, cluster_pv, H0
