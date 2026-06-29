# lcmv_stats/connectivity.py

"""
Connectivity feature extraction for CIMT source-space data.
Delegates to lcmv_xtra for WPLI and implements GPDC for significant edges.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Tuple
import logging
from lcmv_xtra import compute_cimt_motor_connectivity
from ._atlas import get_cimt_labels

logger = logging.getLogger(__name__)

def extract_wpli_features(
    epochs_in: np.ndarray, 
    epochs_out: np.ndarray, 
    band: str = "low_beta", 
    sfreq: float = 250.0
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Compute WPLI for the CIMT motor network.
    
    Args:
        epochs_in:  (n_epochs, n_rois, n_times) — from epoch_tensor[i]
        epochs_out: Same shape as epochs_in.
        band: Frequency band name.
        sfreq: Sampling frequency.
        
    Returns:
        Tuple of (in_conn_matrix, out_conn_matrix) or (None, None) if empty.
    """
    if epochs_in.size == 0 or epochs_out.size == 0:
        logger.warning("Empty epochs provided for WPLI extraction.")
        return None, None
        
    try:
        in_conn = compute_cimt_motor_connectivity(epochs_in, band_name=band, sfreq=sfreq)
        out_conn = compute_cimt_motor_connectivity(epochs_out, band_name=band, sfreq=sfreq)
        return in_conn.values, out_conn.values
    except Exception as e:
        logger.error(f"WPLI calculation failed: {e}")
        return None, None


def extract_gpdc_features(
    epochs_in: np.ndarray,
    epochs_out: np.ndarray,
    sig_df: pd.DataFrame,
    sfreq: float,
    band_range: Tuple[float, float],
    nw: int = 3,
    n_tapers: int = 5
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str]]:
    """
    Compute GPDC only for ROIs involved in significant edges.
    
    Args:
        epochs_in/out: Shape (n_epochs, 448, n_times).
        sig_df: DataFrame from run_edgewise_permutation with significant edges.
        sfreq: Sampling frequency.
        band_range: (low_freq, high_freq) tuple.
        
    Returns:
        (in_gpdc_matrix, out_gpdc_matrix, valid_roi_names)
    """
    if sig_df.empty or epochs_in.size == 0:
        return None, None, []

    roi_names = sorted(set(sig_df['roi1'].unique()) | set(sig_df['roi2'].unique()))
    
    atlas_df = get_cimt_labels()
    roi_indices, valid_rois = [], []
    
    for name in roi_names:
        match = atlas_df[atlas_df['roi_name'] == name]
        if not match.empty:
            idx = int(match.iloc[0]['index'])
            if idx < epochs_in.shape[1]:
                roi_indices.append(idx)
                valid_rois.append(name)
            else:
                logger.warning(f"ROI {name} index {idx} out of bounds.")
            
    if not roi_indices:
        logger.warning("No valid ROI indices found for GPDC subset.")
        return None, None, []
        
    in_subset = epochs_in[:, roi_indices, :]
    out_subset = epochs_out[:, roi_indices, :]
    
    try:
        from spectral_connectivity import Multitaper, Connectivity
    except ImportError:
        raise ImportError("spectral_connectivity is required for GPDC features.")
        
    def _compute_gpdc(epochs_sub):
        n_trials, n_rois, n_samples = epochs_sub.shape
        if n_trials == 0 or n_rois < 2:
            return None
            
        data_reshaped = np.transpose(epochs_sub, (2, 0, 1))
        data_reshaped = data_reshaped - np.mean(data_reshaped, axis=0, keepdims=True)
        
        m = Multitaper(
            time_series=data_reshaped,
            sampling_frequency=sfreq,
            time_halfbandwidth_product=nw,
            n_tapers=n_tapers,
            detrend_type='constant',
            is_low_bias=True
        )
        
        c = Connectivity.from_multitaper(m)
        gpdc_full = c.generalized_partial_directed_coherence()
        
        if gpdc_full.ndim == 4:
            gpdc_full = gpdc_full[0, :, :, :]
            
        freqs = c.frequencies
        freq_idx = np.where((freqs >= band_range[0]) & (freqs <= band_range[1]))[0]
        
        if len(freq_idx) == 0:
            return None
        return np.mean(gpdc_full[freq_idx, :, :], axis=0)

    try:
        in_gpdc = _compute_gpdc(in_subset)
        out_gpdc = _compute_gpdc(out_subset)
        return in_gpdc, out_gpdc, valid_rois
    except Exception as e:
        logger.error(f"GPDC calculation failed: {e}")
        return None, None, []
