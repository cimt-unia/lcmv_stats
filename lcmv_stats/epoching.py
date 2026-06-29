# lcmv_stats/epoching.py

import numpy as np
import logging

logger = logging.getLogger(__name__)

def epoch_tensor(
    tensor_data: np.ndarray,
    sfreq: float,
    epoch_duration: float = 2.0,
    overlap: float = 0.5,
    do_zscore: bool = False
) -> np.ndarray:
    """
    Convert a continuous 3D tensor into a 5D epoched tensor.
    Z-scoring is applied to continuous data BEFORE epoching.
    
    Args:
        tensor_data: Shape (n_subjects, n_rois, n_times).
        sfreq: Sampling frequency in Hz.
        epoch_duration: Duration of each epoch in seconds.
        overlap: Overlap fraction (0.0 to 1.0).
        do_zscore: Z-score each ROI's continuous time course before epoching.
        
    Returns:
        np.ndarray: Shape (n_subjects, n_epochs, n_rois, n_samples).
                    Returns empty array (0,0,0,0) if data is too short.
    """
    n_subj, n_rois, n_times = tensor_data.shape
    
    # ─── STEP 1: Z-score CONTINUOUS data per ROI (vectorized) ───
    if do_zscore:
        # Compute mean/std over time axis (axis=2) for each subject×ROI
        mu = np.mean(tensor_data, axis=2, keepdims=True)      # (n_subj, n_roi, 1)
        sigma = np.std(tensor_data, axis=2, keepdims=True)    # (n_subj, n_roi, 1)
        sigma = np.where(sigma < 1e-12, 1.0, sigma)           # Prevent div-by-zero
        tensor_data = (tensor_data - mu) / sigma
        logger.debug("Applied Z-score normalization to continuous data before epoching.")
    
    # ─── STEP 2: Calculate window parameters ───
    ep_samp = int(epoch_duration * sfreq)
    step = max(1, int(ep_samp * (1 - overlap)))
    
    if n_times < ep_samp:
        logger.warning(f"Time dimension ({n_times}) < epoch samples ({ep_samp}).")
        return np.empty((n_subj, 0, n_rois, ep_samp))
    
    n_ep = ((n_times - ep_samp) // step) + 1
    
    # ─── STEP 3: Vectorized epoch extraction ───
    # sliding_window_view on axis=2 creates (n_subj, n_roi, n_windows, ep_samp)
    epochs = np.lib.stride_tricks.sliding_window_view(
        tensor_data, window_shape=(ep_samp,), axis=2
    )[:, :, ::step, :]
    
    # Reorder to (n_subjects, n_epochs, n_rois, n_samples)
    epochs = np.transpose(epochs, (0, 2, 1, 3))[:, :n_ep, :, :].copy()
    
    logger.info(
        f"Epoched tensor: {n_subj} subjects × {n_ep} epochs × "
        f"{n_rois} ROIs × {ep_samp} samples at {sfreq}Hz "
        f"(zscore={do_zscore})"
    )
    return epochs

'''
# Usage Example
import numpy as np
from lcmv_stats.epoching import epoch_tensor
from lcmv_stats.utils import load_tensor

# 1. Load continuous tensor (your existing output from lcmv_xtra)
cont = load_tensor("./ml_data/study_eyes_closed.npz")
# cont['data'].shape → (12, 448, T)  ← Continuous

# 2. Epoch it (Z-scoring applied BEFORE windowing)
epochs = epoch_tensor(
    tensor_data=cont['data'],
    sfreq=cont['sfreq'],
    epoch_duration=2.0,
    overlap=0.5,
    do_zscore=True
)
# epochs.shape → (12, n_epochs, 448, 500)  ← Epoched

# 3. Save as new .npz with identical metadata structure
np.savez_compressed(
    "./ml_data/epochs_eyes_closed.npz",
    data=epochs,                    # Now 5D instead of 3D
    subject_ids=cont['subject_ids'], # Same subjects, same order
    sfreq=cont['sfreq'],            # Same sampling rate
    epoch_duration=2.0,             # NEW: epoch metadata
    overlap=0.5                     # NEW: epoch metadata
)
print(f"✅ Saved epoched tensor: {epochs.shape}")
'''
