# lcmv_stats/timefreq.py

"""
Time-frequency analysis tools for CIMT source-space data.
Handles spectrogram computation, Z-scoring, and cluster-based permutation testing.
Replicates the robust logic from bima_spectrogram_stats_final.
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


def extract_single_roi_epoch(
    trial_file: Path, 
    t_event: float, 
    roi_idx: int, 
    sfreq: float, 
    pre_sec: float, 
    post_sec: float,
    target_sfreq: float, 
    expected_samples: int
) -> Optional[np.ndarray]:
    """
    Extract and resample a single ROI epoch from a raw trial file.
    
    This function bridges raw multi-ROI trial data and TF analysis by handling
    ROI selection, temporal windowing, and cross-subject resampling automatically.
    
    Args:
        trial_file: Path to the .npy trial file (shape: n_rois x n_times).
        t_event: Event timestamp in seconds relative to trial start.
        roi_idx: Numeric index of the target ROI in the CIMT atlas.
        sfreq: Original sampling frequency of the trial data.
        pre_sec: Seconds before event to include.
        post_sec: Seconds after event to include.
        target_sfreq: Target sampling frequency for output.
        expected_samples: Exact number of samples expected in output.
        
    Returns:
        1D numpy array of shape (expected_samples,) or None if extraction fails.
    """
    try:
        trial_data = np.load(trial_file)
    except Exception:
        return None
        
    n_rois, n_times = trial_data.shape
    if not (0 <= roi_idx < n_rois):
        return None
        
    start_sample = int(np.round((t_event - pre_sec) * sfreq))
    end_sample = int(np.round((t_event + post_sec) * sfreq))
    
    if start_sample < 0 or end_sample > n_times:
        return None
        
    epoch = trial_data[roi_idx, start_sample:end_sample]
    
    # Resample to target frequency if needed
    if sfreq != target_sfreq:
        epoch = signal.resample(epoch, int(round(len(epoch) * target_sfreq / sfreq)))
        
    # Enforce exact length via truncation or edge-padding
    if len(epoch) >= expected_samples:
        return epoch[:expected_samples]
    return np.pad(epoch, (0, expected_samples - len(epoch)), mode="edge")


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
        ref_mask = t_relative < 0
    else:
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
    t: np.ndarray,
    save_path: Optional[Path] = None,
    # --- INTEGRATED PLOTTING PARAMETERS ---
    avg_sxx: Optional[np.ndarray] = None,
    roi_name: str = "ROI",
    n_units: int = 0,
    mode_label: str = "Subject",
    smooth_sigma: Optional[Tuple[float, float]] = (1.0, 2.0),
    colormap_percentile: float = 2.0,
    plot_save_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Convert raw MNE cluster permutation test outputs into a publication-ready DataFrame
    AND optionally generate the corresponding time-frequency plot with clusters overlaid.
    
    Args:
        T_obs: Observed T-statistic map (n_freqs, n_times).
        clusters: List of tuples containing (freq_indices, time_indices) for each cluster.
        cluster_pv: P-values for each cluster.
        f: Frequency bins (Hz).
        t: Time bins (s, relative to event onset).
        save_path: Optional path to save the resulting DataFrame as a CSV file.
        
        # Plotting Parameters (Optional - if provided, generates figure automatically)
        avg_sxx: Average Z-scored spectrogram (n_freqs, n_times) to plot.
        roi_name: Name of the ROI for the plot title.
        n_units: Number of subjects or trials averaged.
        mode_label: Label for the mode (e.g., "Subject" or "Trial").
        smooth_sigma: Sigma for Gaussian smoothing of the plot.
        colormap_percentile: Percentile for color limit scaling.
        plot_save_path: Optional path to save the generated figure. If None, displays interactively.
        
    Returns:
        DataFrame with columns: cluster_id, p_value, n_points, 
                               freq_min_hz, freq_max_hz, 
                               time_min_s, time_max_s,
                               mean_t_stat, peak_t_stat
    """
    # ─── 1. FORMAT TABLE ──────────────────────────────────────────────
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
    logger.info(f"Formatted {len(df)} clusters for publication.")
    
    # Save CSV if path provided
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        logger.info(f"Saved publication table to {save_path}")
    
    # ─── 2. GENERATE PLOT AUTOMATICALLY IF DATA PROVIDED ──────────────
    if avg_sxx is not None:
        _plot_tf_clusters_integrated(
            f=f, t=t, avg_sxx=avg_sxx,
            clusters=clusters, cluster_pv=cluster_pv,
            roi_name=roi_name, n_units=n_units,
            mode_label=mode_label, smooth_sigma=smooth_sigma,
            colormap_percentile=colormap_percentile,
            save_path=plot_save_path
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
    save_path: Optional[Path]
):
    """Internal helper to generate the TF plot when called from format_cluster_results_for_publication."""
    
    def _get_clim(data: np.ndarray, percentile: float = 2.0) -> Tuple[float, float]:
        lo = np.percentile(data, percentile)
        hi = np.percentile(data, 100.0 - percentile)
        v = max(abs(lo), abs(hi))
        return -v, v
    
    # Smooth for visualization only
    avg_sxx_plot = gaussian_filter(avg_sxx, sigma=smooth_sigma) if smooth_sigma else avg_sxx
    vmin, vmax = _get_clim(avg_sxx_plot, colormap_percentile)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot Mean
    mesh = ax.pcolormesh(t, f, avg_sxx_plot,
                         shading="gouraud", cmap="RdBu_r",
                         vmin=vmin, vmax=vmax)
    
    # Overlay Significant Clusters
    if clusters:
        sig_mask = np.zeros_like(avg_sxx, dtype=bool)
        for c_idx, cluster in enumerate(clusters):
            if cluster_pv[c_idx] < 0.05:
                sig_mask[cluster] = True
        ax.contour(t, f, sig_mask.astype(int), levels=[0.5], colors='white', linewidths=1.5)
        
    ax.set_ylabel("Frequency (Hz)", fontsize=13)
    ax.set_xlabel("Time relative to event (s)", fontsize=13)
    ax.set_title(f"Global Average ({mode_label}) + Significant Clusters (p<0.05)\nROI: {roi_name} | N={n_units}", fontsize=14)
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

# lcmv_stats/timefreq.py

def run_roi_spectrogram_analysis(
    events_df: pd.DataFrame,
    lcmv_root: Path,
    roi_name: str,
    condition: str = "bima_off",
    pre_sec: float = 5.0,
    post_sec: float = 5.0,
    target_sfreq: float = 250.0,
    f_min: float = 13.0,
    f_max: float = 30.0,
    baseline_duration: Optional[float] = 2.0,
    n_permutations: int = 1000,
    threshold: dict = {'start': 0.5, 'step': 0.1},
    smooth_sigma: Tuple[float, float] = (1.0, 2.0),
    colormap_percentile: float = 2.0,
    save_path_csv: Optional[Path] = None,
    save_path_plot: Optional[Path] = None,
    notes_filter: List[str] = ["good"]
) -> pd.DataFrame:
    """
    Complete end-to-end TF cluster analysis for a single ROI.
    
    Orchestrates trial loading, epoch extraction, resampling, spectrogram 
    computation, Z-scoring, cluster permutation testing, and publication output.
    
    Args:
        events_df: DataFrame with 'subject', 'trial', 'event', 'notes' columns.
        lcmv_root: Root path to LCMV derivatives.
        roi_name: CIMT ROI name (e.g., 'STN-lh').
        condition: Condition folder name (default: 'bima_off').
        pre_sec/post_sec: Epoch window in seconds relative to event.
        target_sfreq: Target sampling frequency for resampling.
        f_min/f_max: Frequency range of interest.
        baseline_duration: Baseline duration for Z-scoring (None = full pre-event).
        n_permutations: Number of permutations for cluster test.
        threshold: TFCE threshold parameters.
        smooth_sigma: Gaussian smoothing sigma for visualization.
        colormap_percentile: Percentile for color limit scaling.
        save_path_csv: Path to save publication CSV table.
        save_path_plot: Path to save TF cluster plot.
        notes_filter: Values in 'notes' column to include.
        
    Returns:
        Publication-ready DataFrame with cluster results.
    """
    import json
    import re
    from ._atlas import get_roi_index
    
    # ─── 1. VALIDATE & PREPARE ────────────────────────────────────────
    roi_idx = get_roi_index(roi_name)
    logger.info(f"ROI: {roi_name} (index {roi_idx})")
    
    events_df = events_df[events_df["notes"].isin(notes_filter)].copy()
    if events_df.empty:
        raise ValueError("No valid events after filtering.")
    
    total_samples = int((pre_sec + post_sec) * target_sfreq)
    units_for_stats = []
    global_f = global_t = None
    
    # ─── 2. PER-SUBJECT PROCESSING ────────────────────────────────────
    for subject_name in sorted(events_df["subject"].unique()):
        # Map subject ID
        num_match = re.findall(r'\d+', subject_name)
        subj_id = f"sub-{int(num_match[0]):03d}" if num_match else subject_name
        
        # Load metadata for original sfreq
        meta_file = lcmv_root / f"{subj_id}_{condition}" / "pipeline_metadata.json"
        if not meta_file.exists():
            logger.warning(f"Skip {subj_id}: metadata not found")
            continue
            
        with open(meta_file, 'r') as f:
            orig_sfreq = float(json.load(f)['sfreq_hz'])
        
        # Get trials for this subject
        subj_events = events_df[events_df["subject"] == subject_name]
        trial_dir = lcmv_root / f"{subj_id}_{condition}" / "cimt_trials"
        if not trial_dir.exists():
            logger.warning(f"Skip {subj_id}: trial dir not found")
            continue
        
        # Extract epochs using library function
        all_epochs = []
        for _, row in subj_events.iterrows():
            t_event = float(row['event'])
            if not np.isfinite(t_event): continue
            
            trial_file = trial_dir / f"trial_{int(row['trial']):03d}.npy"
            if not trial_file.exists(): continue
            
            epoch = extract_single_roi_epoch(
                trial_file=trial_file, t_event=t_event, roi_idx=roi_idx,
                sfreq=orig_sfreq, pre_sec=pre_sec, post_sec=post_sec,
                target_sfreq=target_sfreq, expected_samples=total_samples
            )
            if epoch is not None:
                all_epochs.append(epoch)
        
        if not all_epochs: continue
        
        # Average trials per subject BEFORE spectrogram (preserves phase coherence)
        avg_epoch = np.mean(all_epochs, axis=0)
        
        # Compute spectrogram using library function
        f, t, Sxx_z = compute_zscored_spectrogram(
            epoch=avg_epoch, sfreq=target_sfreq,
            f_min=f_min, f_max=f_max,
            pre_sec=pre_sec, baseline_duration=baseline_duration
        )
        
        if global_f is None:
            global_f, global_t = f, t
        units_for_stats.append(Sxx_z)
        logger.info(f"Processed {subj_id}: {len(all_epochs)} trials")
    
    if len(units_for_stats) < 2:
        raise ValueError(f"Need ≥2 units for cluster test, got {len(units_for_stats)}")
    
    # ─── 3. CLUSTER PERMUTATION TEST ──────────────────────────────────
    X = np.stack(units_for_stats, axis=0)
    T_obs, clusters, cluster_pv, H0 = run_cluster_spectrogram(
        spectrograms_3d=X,
        n_permutations=n_permutations,
        threshold=threshold
    )
    
    sig_count = sum(1 for p in cluster_pv if p < 0.05)
    logger.info(f"Found {sig_count} significant clusters (p<0.05)")
    
    # ─── 4. FORMAT & PLOT ─────────────────────────────────────────────
    avg_sxx = np.mean(X, axis=0)
    
    df = format_cluster_results_for_publication(
        T_obs=T_obs, clusters=clusters, cluster_pv=cluster_pv,
        f=global_f, t=global_t,
        save_path=save_path_csv,
        avg_sxx=avg_sxx, roi_name=roi_name,
        n_units=len(units_for_stats), mode_label="Subject",
        smooth_sigma=smooth_sigma, colormap_percentile=colormap_percentile,
        plot_save_path=save_path_plot
    )
    
    return df
