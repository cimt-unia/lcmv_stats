# lcmv_stats/timefreq.py

"""
Time-frequency analysis tools for CIMT source-space data.
Handles spectrogram computation, Z-scoring, and cluster-based permutation testing.
Replicates the robust logic from bima_spectrogram_stats_final.
"""

import numpy as np
import pandas as pd
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
    baseline_duration: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a Z-scored spectrogram for a single epoch/trial.
    
    CRITICAL: This function expects a SINGLE trial or subject-average epoch,
    NOT a grand average across subjects. Averaging before power estimation
    destroys phase coherence and causes edge artifacts.
    
    Args:
        epoch: 1D array of time-series data (single trial or subject-average).
               Shape must be (n_times,).
        sfreq: Sampling frequency.
        f_min/f_max: Frequency range of interest.
        pre_sec: Duration of the pre-event window (used to locate baseline).
        baseline_duration: Duration of baseline for Z-scoring. 
                          None = entire pre-event period.
                          Float = first N seconds of pre-event period.
        
    Returns:
        f: Frequency bins.
        t: Time bins (relative to event onset).
        Sxx_z: Z-scored power spectral density.
    """
    # 1. Compute Raw Spectrogram (exact match to working script)
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

    # 2. Z-Score Normalization (exact match to working script)
    t_relative = t - pre_sec
    
    # Determine baseline mask
    if baseline_duration is None:
        # Use entire pre-event period
        ref_mask = t_relative < 0
    else:
        # Use first N seconds of pre-event period
        ref_mask = (t_relative >= -pre_sec) & (t_relative <= -pre_sec + baseline_duration)

    if not np.any(ref_mask):
        logger.warning("No baseline bins found. Returning zeros.")
        return f, t_relative, np.zeros_like(Sxx)

    ref_data = Sxx[:, ref_mask]
    ref_mean = np.mean(ref_data, axis=1, keepdims=True)
    ref_std = np.std(ref_data, axis=1, keepdims=True)

    # Prevent division by zero
    std_floor = np.abs(ref_mean) * 0.01
    std_floor = np.where(std_floor < 1e-30, 1e-30, std_floor)
    ref_std = np.where(ref_std < std_floor, std_floor, ref_std)

    Sxx_z = (Sxx - ref_mean) / ref_std
    
    # Safety net for any remaining numerical issues
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
        spectrograms_3d: Array of shape (n_units, n_freqs, n_times).
                        n_units can be subjects OR trials depending on STATS_MODE.
        adjacency: Pre-computed adjacency matrix. If None, lattice adjacency created.
        n_permutations: Number of permutations.
        threshold: TFCE threshold parameters.
        
    Returns:
        T_obs, clusters, cluster_pv, H0
    """
    if spectrograms_3d.ndim != 3:
        raise ValueError("Input must be a 3D array (n_units, n_freqs, n_times)")

    n_units, n_freqs, n_times = spectrograms_3d.shape

    if adjacency is None:
        adj_freq = np.eye(n_freqs, k=1) + np.eye(n_freqs, k=-1)
        adj_time = np.eye(n_times, k=1) + np.eye(n_times, k=-1)
        adjacency = combine_adjacency(adj_freq, adj_time)

    logger.info(f"Running cluster permutation test on {n_units} units...")

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


def format_cluster_results_for_publication(
    T_obs: np.ndarray,
    clusters: List[Tuple],
    cluster_pv: np.ndarray,
    f: np.ndarray,
    t: np.ndarray
) -> pd.DataFrame:
    """
    Convert raw MNE cluster permutation test outputs into a publication-ready DataFrame.
    
    Maps cluster indices back to actual Frequency (Hz) and Time (s) values,
    making the results directly usable for scientific tables and CSV exports.
    
    Args:
        T_obs: Observed T-statistic map (n_freqs, n_times).
        clusters: List of tuples containing (freq_indices, time_indices) for each cluster.
        cluster_pv: P-values for each cluster.
        f: Frequency bins (Hz).
        t: Time bins (s, relative to event onset).
        
    Returns:
        DataFrame with columns: cluster_id, p_value, n_points, 
                               freq_min_hz, freq_max_hz, 
                               time_min_s, time_max_s,
                               mean_t_stat, peak_t_stat
    """
    if not clusters:
        logger.info("No clusters found. Returning empty DataFrame.")
        return pd.DataFrame()
    
    results = []
    for c_idx, cluster in enumerate(clusters):
        freq_idx, time_idx = cluster
        
        # Map indices to physical coordinates
        freq_bounds = f[[freq_idx.min(), freq_idx.max()]]
        time_bounds = t[[time_idx.min(), time_idx.max()]]
        
        # Extract T-stats within this cluster
        cluster_t_stats = T_obs[freq_idx][:, time_idx]
        
        results.append({
            'cluster_id': c_idx,
            'p_value': cluster_pv[c_idx],
            'significant': cluster_pv[c_idx] < 0.05,
            'n_points': len(freq_idx) * len(time_idx),
            'freq_min_hz': round(float(freq_bounds[0]), 2),
            'freq_max_hz': round(float(freq_bounds[1]), 2),
            'time_min_s': round(float(time_bounds[0]), 3),
            'time_max_s': round(float(time_bounds[1]), 3),
            'mean_t_stat': round(float(np.mean(cluster_t_stats)), 4),
            'peak_t_stat': round(float(np.max(np.abs(cluster_t_stats))), 4)
        })
    
    df = pd.DataFrame(results).sort_values('p_value')
    logger.info(f"Formatted {len(df)} clusters for publication.")
    return df
