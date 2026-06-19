# lcmv_stats/epoching.py

"""
Epoching tools for CIMT trial data.
Handles both event-locked and continuous/resting-state extraction.
"""

import numpy as np
from pathlib import Path
from typing import Optional
import logging
from .utils import get_subject_sfreq

logger = logging.getLogger(__name__)

def extract_event_epochs(
    subject_id: str, 
    lcmv_root: Path, 
    events_df, 
    pre_sec: float = 5.0, 
    post_sec: float = 5.0,
    condition: str = "bima_off",
    epoch_duration: Optional[float] = None,
    overlap: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract in-phase and out-phase epochs for a single subject based on events.
    """
    sfreq = get_subject_sfreq(subject_id, lcmv_root, condition)
    trial_dir = lcmv_root / f"{subject_id}_{condition}" / "cimt_trials"
    
    all_pre, all_post = [], []
    n_rois_ref = None
    
    for _, row in events_df.iterrows():
        trial_file = trial_dir / f"trial_{int(row['trial']):03d}.npy"
        if not trial_file.exists(): continue
        
        try:
            trial_data = np.load(trial_file) # Shape: (n_rois, n_times)
            t_event = float(row['event'])
            
            if n_rois_ref is None:
                n_rois_ref = trial_data.shape[0]
            elif trial_data.shape[0] != n_rois_ref:
                continue

            start = int(np.round((t_event - pre_sec) * sfreq))
            end = int(np.round((t_event + post_sec) * sfreq))
            
            if start < 0 or end > trial_data.shape[1]: continue
            
            window = trial_data[:, start:end]
            mid_point = int(pre_sec * sfreq)
            
            if epoch_duration:
                epoch_samples = int(round(epoch_duration * sfreq))
                step_samples = int(round(epoch_samples * (1 - overlap)))
                
                pre_window = window[:, :mid_point]
                if pre_window.shape[1] >= epoch_samples:
                    n_pre = ((pre_window.shape[1] - epoch_samples) // step_samples) + 1
                    for i in range(n_pre):
                        s = i * step_samples
                        e = s + epoch_samples
                        all_pre.append(pre_window[:, s:e])
                        
                post_window = window[:, mid_point:]
                if post_window.shape[1] >= epoch_samples:
                    n_post = ((post_window.shape[1] - epoch_samples) // step_samples) + 1
                    for i in range(n_post):
                        s = i * step_samples
                        e = s + epoch_samples
                        all_post.append(post_window[:, s:e])
            else:
                all_pre.append(window[:, :mid_point])
                all_post.append(window[:, mid_point:])
                
        except Exception as e:
            logger.warning(f"Error processing trial for {subject_id}: {e}")
            
    if not all_pre:
        n_rois = n_rois_ref if n_rois_ref else 0
        return np.empty((0, n_rois, 0)), np.empty((0, n_rois, 0))
        
    return np.stack(all_pre), np.stack(all_post)

def extract_continuous_epochs(
    subject_id: str,
    lcmv_root: Path,
    condition: str = "bima_off",
    epoch_duration: float = 2.0,
    overlap: float = 0.0,
    do_zscore: bool = False
) -> np.ndarray:
    """
    Extract epochs from continuous CIMT time courses using a sliding window.
    Assumes data is already preprocessed (filtered/cleaned) by lcmv_xtra.
    
    Args:
        subject_id: Subject identifier.
        lcmv_root: Root path to LCMV derivatives.
        condition: Condition folder name.
        epoch_duration: Duration of each epoch in seconds.
        overlap: Overlap fraction (0.0 to 1.0).
        do_zscore: Whether to Z-score each ROI time course before epoching.
        
    Returns:
        np.ndarray: Shape (n_epochs, n_rois, n_samples).
    """
    sfreq = get_subject_sfreq(subject_id, lcmv_root, condition)
    subj_dir = lcmv_root / f"{subject_id}_{condition}"
    
    # Load continuous CIMT time courses
    cimt_file = subj_dir / "cimt_time_courses.npy"
    if not cimt_file.exists():
        raise FileNotFoundError(f"CIMT time courses not found: {cimt_file}")
        
    try:
        tc = np.load(cimt_file) # Shape: (448, T)
    except Exception as e:
        raise ValueError(f"Failed to load CIMT data: {e}")

    n_rois, n_times = tc.shape
    
    # 1. Optional Z-scoring per ROI (for resting-state normalization)
    if do_zscore:
        mu = np.mean(tc, axis=1, keepdims=True)
        sigma = np.std(tc, axis=1, keepdims=True)
        # Prevent division by zero
        sigma = np.where(sigma < 1e-12, 1.0, sigma)
        tc = (tc - mu) / sigma

    # 2. Window into epochs
    ep_samp = int(epoch_duration * sfreq)
    step = int(ep_samp * (1 - overlap))
    
    if n_times < ep_samp:
        logger.warning(f"Subject {subject_id}: Data length ({n_times}) shorter than epoch duration ({ep_samp}).")
        return np.empty((0, n_rois, ep_samp))
        
    n_ep = ((n_times - ep_samp) // step) + 1
    if n_ep <= 0:
        return np.empty((0, n_rois, ep_samp))
        
    epochs = np.empty((n_ep, n_rois, ep_samp))
    for i in range(n_ep):
        s = i * step
        epochs[i] = tc[:, s:s + ep_samp]
        
    logger.info(f"Extracted {n_ep} continuous epochs for {subject_id} at {sfreq}Hz")
    return epochs