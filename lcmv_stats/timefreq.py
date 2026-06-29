# lcmv_stats/timefreq.py

"""
Time-frequency analysis tools for CIMT source-space data.
Refactored for Rigid Concatenation of conditions (e.g., Rest|Move).
"""

import numpy as np
import scipy.signal as signal
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import logging
from mne.stats import permutation_cluster_test, combine_adjacency, ttest_1samp_no_p
from typing import List, Optional, Tuple, Literal

logger = logging.getLogger(__name__)


def concatenate_condition_signals(
    cond_a_signal: np.ndarray,
    cond_b_signal: np.ndarray,
    sfreq: float
) -> Tuple[np.ndarray, float, float]:
    """
    Rigidly concatenates two 1D signals (Condition A | Condition B).
    
    Args:
        cond_a_signal: 1D array for Condition A (e.g., Rest). Shape: (n_samples,)
        cond_b_signal: 1D array for Condition B (e.g., Move). Shape: (n_samples,)
        sfreq: Sampling frequency in Hz.
        
    Returns:
        concat_sig: Concatenated 1D signal.
        dur_a_sec: Duration of Condition A in seconds.
        dur_total_sec: Total duration of concatenated signal in seconds.
    """
    if cond_a_signal.ndim != 1 or cond_b_signal.ndim != 1:
        raise ValueError("Inputs must be 1D arrays.")
        
    concat_sig = np.concatenate([cond_a_signal, cond_b_signal])
    dur_a_sec = len(cond_a_signal) / sfreq
    dur_total_sec = len(concat_sig) / sfreq
    
    return concat_sig, dur_a_sec, dur_total_sec


def compute_spectrogram_for_subject(
    sig: np.ndarray, 
    sfreq: float, 
    f_min: float = 12.0, 
    f_max: float = 30.0,
    normalize_mode: Literal["none", "fraction", "pre_event"] = "fraction",
    baseline_fraction: float = 0.5, # Default to 50% if using fraction (adjust as needed)
    alignment_sec: float = 0.0,     # Used only if mode="pre_event"
    nperseg_override: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute spectrogram for a single subject's concatenated signal.
    
    Args:
        sig: 1D concatenated signal [CondA | CondB].
        sfreq: Sampling frequency.
        normalize_mode: 
            "none": Raw PSD.
            "fraction": Z-score using first X% of the TOTAL signal.
            "pre_event": Z-score using time < alignment_sec.
        baseline_fraction: Fraction of total time to use as baseline if mode="fraction".
        alignment_sec: Time point (in seconds) marking the boundary/event if mode="pre_event".
        
    Returns:
        f_filt: Frequency bins.
        t: Time bins.
        Sxx_out: Spectrogram (Z-scored or raw).
    """
    if nperseg_override is not None:
        nperseg = nperseg_override
    else:
        # Adaptive window: 1 second, but capped at half signal length
        nperseg = min(int(sfreq * 1.0), len(sig) // 2)
    
    nperseg = max(4, nperseg)
    noverlap = int(nperseg * 0.75)

    try:
        f, t, Sxx = signal.spectrogram(
            sig, fs=sfreq, window="hann",
            nperseg=nperseg, noverlap=noverlap,
            scaling="density", mode="psd",
        )
    except Exception as e:
        logger.error(f"Spectrogram computation failed: {e}")
        return np.array([]), np.array([]), np.array([])

    # Filter frequencies
    freq_mask = (f >= f_min) & (f <= f_max)
    f_filt = f[freq_mask]
    Sxx_filt = Sxx[freq_mask, :]

    # --- Conditional Normalization ---
    if normalize_mode == "fraction":
        # Use first X% of the TOTAL concatenated signal as baseline
        t_max_baseline = t[-1] * baseline_fraction
        ref_mask = t <= t_max_baseline
        
    elif normalize_mode == "pre_event":
        # Use time before the alignment marker (e.g., start of Move)
        ref_mask = t < alignment_sec
        
    else:
        # No normalization
        ref_mask = None

    if ref_mask is not None:
        if np.any(ref_mask):
            ref_mean = Sxx_filt[:, ref_mask].mean(axis=1, keepdims=True)
            ref_std = Sxx_filt[:, ref_mask].std(axis=1, keepdims=True)
            
            # Floor std to avoid division by zero
            floor_val = np.maximum(np.abs(ref_mean) * 0.01, 1e-30)
            ref_std = np.where(ref_std < floor_val, floor_val, ref_std)
            
            Sxx_out = (Sxx_filt - ref_mean) / ref_std
        else:
            logger.warning("No time points found in baseline mask. Returning raw PSD.")
            Sxx_out = Sxx_filt
    else:
        Sxx_out = Sxx_filt

    return f_filt, t, Sxx_out


def plot_and_test_group_spectrograms(
    spectrograms_list: List[np.ndarray],
    f: np.ndarray,
    t: np.ndarray,
    roi_name: str,
    hemisphere: str,
    boundary_sec: float, # The time point where CondA ends and CondB begins
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
    Plots group-average spectrogram with cluster permutation test.
    
    Args:
        spectrograms_list: List of 2D arrays (n_freqs, n_times) for each subject.
        boundary_sec: Time in seconds where Condition A ends and Condition B starts.
    """
    if not spectrograms_list:
        print("No valid spectrograms to plot.")
        return

    # Stack subjects: (N_subjects, N_freqs, N_times)
    X = np.stack(spectrograms_list, axis=0)
    
    # --- CRITICAL VALIDATION ---
    if X.shape[2] != len(t):
        raise ValueError(
            f"Time axis mismatch! t has {len(t)} points "
            f"but spectrograms have {X.shape[2]} time points."
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
        print("Skipping statistics: Only 1 subject provided.")

    # --- PLOTTING ---
    # Grand Average for Plotting
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

    # Plot Boundary Line
    ax.axvline(x=boundary_sec, color='gray', linestyle=':', linewidth=2, 
               label=f'Condition Boundary ({boundary_sec:.2f}s)')

    ax.set_ylabel("Frequency (Hz)", fontsize=13)
    ax.set_xlabel("Time (s)", fontsize=13)
    ax.set_ylim([f_min, f_max])
    ax.set_title(f"Group Spectrogram ({hemisphere.upper()} {region})\n"
                 f"ROI: {roi_name} (N={X.shape[0]}, α={alpha}, perms={n_permutations})")
    ax.legend(loc='upper right')
    fig.colorbar(mesh, ax=ax, label="Mean Z-score")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()
