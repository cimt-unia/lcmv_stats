# lcmv_stats/timefreq.py

"""
Time-frequency analysis tools for CIMT source-space data.
"""

import numpy as np
import scipy.signal as signal
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import logging # Keep this import
from mne.stats import permutation_cluster_test, combine_adjacency, ttest_1samp_no_p
from typing import List, Optional, Tuple, Union


logger = logging.getLogger(__name__)


# SPECTROGRAM ANALYSIS FUNCTIONS 

def prepare_roi_signal_from_2d(
    move_epochs_2d: np.ndarray, 
    rest_epochs_2d: np.ndarray,
    sfreq: float
) -> Tuple[np.ndarray, float]:
    """
    Averages 2D ROI epochs and calculates the exact epoch duration.
    
    NOTE: This function expects 2D arrays where axis 0 is 'trials/epochs' and 
    axis 1 is 'time samples'. It averages across trials to create a single 
    representative time series per condition.
    
    Args:
        move_epochs_2d: Shape (n_move_trials, n_samples)
        rest_epochs_2d: Shape (n_rest_trials, n_samples)
        sfreq: Sampling frequency in Hz.
        
    Returns:
        Tuple of (1D concatenated signal [Move|Rest], epoch_duration_in_seconds)
    """
    if move_epochs_2d.ndim != 2 or rest_epochs_2d.ndim != 2:
        raise ValueError("Inputs must be 2D arrays (n_trials, n_samples).")
    
    # Calculate duration directly from the number of samples in one epoch
    n_samples_per_epoch = move_epochs_2d.shape[1]
    epoch_dur_sec = n_samples_per_epoch / sfreq
    
    # Average across the trial dimension (axis=0) to get a single representative signal
    move_avg = move_epochs_2d.mean(axis=0)
    rest_avg = rest_epochs_2d.mean(axis=0)
    
    # Concatenate: Move first, then Rest
    concat_sig = np.concatenate([move_avg, rest_avg])
    
    return concat_sig, epoch_dur_sec

def compute_zscored_spectrogram(
    sig: np.ndarray, 
    sfreq: float, 
    f_min: float = 12.0, 
    f_max: float = 30.0,
    baseline_fraction: float = 0.8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a Z-scored spectrogram for a 1D signal.
    
    The baseline is defined as the first 'baseline_fraction' of the total 
    signal duration. This allows for consistent normalization whether the 
    input is a single subject's average or a group average.
    """
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
    
    # Prevent division by zero in low-power regions
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
    n_permutations: int = 100,
    alpha: float = 0.05,
    smooth_sigma: tuple = (1.0, 2.0)
):
    """
    Plots the group-average spectrogram and performs cluster permutation test if N > 1.
    
    Args:
        spectrograms_list: List of 2D arrays (n_freqs, n_times). Can be from 
                           single subjects or pre-computed group averages.
        epoch_dur_sec: Duration of the 'Move' condition in seconds (for boundary line).
    """
    if not spectrograms_list:
        print("No valid spectrograms to plot.")
        return

    X = np.stack(spectrograms_list, axis=0) # Shape: (N_subjects, n_freqs, n_times)
    significant_clusters = []
    
    # --- STATISTICS CHECK ---
    # Permutation tests require variance across subjects (N > 1).
    if X.shape[0] > 1:
        nf, nt = X.shape[1], X.shape[2]
        adj_freq = np.eye(nf, k=1) + np.eye(nf, k=-1)
        adj_time = np.eye(nt, k=1) + np.eye(nt, k=-1)
        adjacency = combine_adjacency(adj_freq, adj_time)

        try:
            _, clusters, cluster_pv, _ = permutation_cluster_test(
                [X],
                n_permutations=n_permutations,
                threshold=dict(start=0.5, step=0.1),
                tail=0,
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

    # Overlay significant clusters ONLY if they exist
    if significant_clusters:
        sig_mask = np.zeros_like(avg_Sxx, dtype=bool)
        for cluster in significant_clusters:
            sig_mask[cluster] = True
        ax.contour(t, f, sig_mask.astype(int), levels=[0.5], colors='white', linewidths=1.5)

    # Draw boundaries based on the dynamically calculated epoch duration
    ax.axvline(x=epoch_dur_sec, color='gray', linestyle=':', linewidth=2, label=f'Move→Rest boundary ({epoch_dur_sec}s)')
    ref_end = epoch_dur_sec * baseline_fraction
    if baseline_fraction < 1.0:
        ax.axvline(x=ref_end, color='red', linestyle='--', linewidth=1, alpha=0.7, label=f'Z-score ref end ({ref_end:.2f}s)')

    ax.set_ylabel("Frequency (Hz)", fontsize=13)
    ax.set_xlabel("Time (s)", fontsize=13)
    ax.set_title(f"Z-Scored Spectrogram ({hemisphere.upper()} {region})\nROI: {roi_name} (N={X.shape[0]})")
    ax.legend(loc='upper right')
    fig.colorbar(mesh, ax=ax, label="Mean Z-score")
    plt.tight_layout()
    plt.show()

'''

# EXECUTION EXAMPLE


print("\n>>> Starting Spectrogram Analysis...")

# 1. Prepare Signals using the 2D ROI epochs
subjects_data = []
sfreq = fs 

try:
    # The function returns both the signal AND the exact duration derived from data shape
    concat_sig, epoch_dur = prepare_roi_signal_from_2d(move_roi_epochs, rest_roi_epochs, sfreq)
    
    print(f"Detected Epoch Duration: {epoch_dur} seconds")
    
    # Compute Spectrogram
    f, t, sxx_z = compute_zscored_spectrogram(
        concat_sig, sfreq, f_min=12.0, f_max=30.0, baseline_fraction=0.8
    )
    
    subjects_data.append(sxx_z)
    print(f"Processed {SUBJECT_ID}: Spectrogram shape {sxx_z.shape}")
    
except Exception as e:
    print(f"Error processing {SUBJECT_ID}: {e}")
    import traceback
    traceback.print_exc()

# 2. Plot and Test
if subjects_data:
    plot_and_test_group_spectrograms(
        spectrograms_list=subjects_data,
        f=f,
        t=t,
        roi_name=TARGET_ROI,
        hemisphere="right", 
        epoch_dur_sec=epoch_dur, 
        baseline_fraction=0.8,
        n_permutations=100 
    )
else:
    print("No data available for plotting.")

'''
