# lcmv_stats/machine_learning.py

"""
Machine Learning Feature Engineering tools for LCMV source-reconstructed data.
Focuses on Z-score normalization, epoching, and spectral feature extraction 
into Pandas DataFrames.
"""

import numpy as np
import pandas as pd
from scipy import signal
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

# ─── Configuration Constants ───

def get_frequency_bands() -> Dict[str, Tuple[float, float]]:
    """
    Define standard frequency bands for feature extraction.
    Matches the definitions used in lcmv_xtra/lcmv_stats connectivity modules.
    """
    return {
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 12), 
        "beta": (13, 30),
        "low_gamma": (30, 50), 
        "high_gamma": (50, 100)
    }

def zscore_normalization(signal_data: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """
    Standardize the signal to mean=0, std=1.
    
    Args:
        signal_data: Input signal array.
        axis: Axis along which to compute mean and std. 
              If None, computes over the entire array.
              
    Returns:
        Z-scored signal array.
    """
    mu = np.mean(signal_data, axis=axis, keepdims=True)
    sigma = np.std(signal_data, axis=axis, keepdims=True)
    
    # Prevent division by zero
    sigma = np.where(sigma < 1e-12, 1.0, sigma)
    
    return (signal_data - mu) / sigma

def create_epochs(
    signal_data: np.ndarray, 
    fs: float, 
    epoch_duration: float = 1.5, 
    overlap_frac: float = 0.75
) -> List[np.ndarray]:
    """
    Splits continuous signal into overlapping epochs.
    
    Args:
        signal_data: 1D array of time-series data.
        fs: Sampling frequency.
        epoch_duration: Duration of each epoch in seconds.
        overlap_frac: Overlap fraction (0.0 to 1.0).
        
    Returns:
        List of 1D numpy arrays, each representing an epoch.
    """
    epoch_samples = int(epoch_duration * fs)
    step_size = int(epoch_samples * (1 - overlap_frac))
    
    if step_size == 0:
        raise ValueError("Overlap fraction is too high, resulting in zero step size.")
        
    epochs = []
    start_idx = 0
    
    while start_idx + epoch_samples <= len(signal_data):
        end_idx = start_idx + epoch_samples
        epochs.append(signal_data[start_idx:end_idx])
        start_idx += step_size
        
    return epochs

def compute_epoch_features(
    epochs: List[np.ndarray], 
    fs: float, 
    freq_bands: Optional[Dict[str, Tuple[float, float]]] = None
) -> pd.DataFrame:
    """
    Computes power in specific frequency bands for each epoch using Welch's method.
    
    Args:
        epochs: List of 1D numpy arrays.
        fs: Sampling frequency.
        freq_bands: Dictionary of band names and (low, high) tuples. 
                    Uses default bands if None.
        
    Returns:
        DataFrame with columns for each band and epoch index.
    """
    if not epochs:
        return pd.DataFrame()
        
    if freq_bands is None:
        freq_bands = get_frequency_bands()
        
    feature_list = []
    band_names = list(freq_bands.keys())
    
    # Use full epoch length for FFT resolution
    nperseg = len(epochs[0])
    noverlap = nperseg // 2
    
    for i, epoch in enumerate(epochs):
        # Compute PSD for this single epoch
        freqs, psd = signal.welch(epoch, fs, nperseg=nperseg, noverlap=noverlap, window='hann')
        
        features = {'epoch': i}
        
        for band_name, (low_freq, high_freq) in freq_bands.items():
            # Find indices corresponding to the frequency band
            idx_band = np.logical_and(freqs >= low_freq, freqs <= high_freq)
            
            # Integrate PSD over the band (sum of power * frequency resolution)
            if np.any(idx_band):
                band_power = np.trapz(psd[idx_band], freqs[idx_band])
            else:
                band_power = 0.0
                
            features[band_name] = band_power
            
        feature_list.append(features)
        
    return pd.DataFrame(feature_list, columns=['epoch'] + band_names)

def process_signal_to_ml_dataframe(
    rest_signal: np.ndarray,
    move_signal: np.ndarray,
    fs: float,
    epoch_duration: float = 1.5,
    overlap_frac: float = 0.75,
    freq_bands: Optional[Dict[str, Tuple[float, float]]] = None
) -> pd.DataFrame:
    """
    End-to-end pipeline: Z-Score -> Epoch -> Feature Extract -> Merge.
    
    Args:
        rest_signal: 1D array of resting state data.
        move_signal: 1D array of movement task data.
        fs: Sampling frequency.
        epoch_duration: Duration of each epoch in seconds.
        overlap_frac: Overlap fraction for epoching.
        freq_bands: Dictionary of frequency bands.
        
    Returns:
        Merged DataFrame with features and 'target' column.
    """
    if freq_bands is None:
        freq_bands = get_frequency_bands()

    # 1. Z-Score Normalization
    logger.info("Applying Z-score normalization...")
    rest_z = zscore_normalization(rest_signal)
    move_z = zscore_normalization(move_signal)
    
    # 2. Epoching
    logger.info(f"Creating epochs (duration={epoch_duration}s, overlap={overlap_frac})...")
    rest_epochs = create_epochs(rest_z, fs, epoch_duration, overlap_frac)
    move_epochs = create_epochs(move_z, fs, epoch_duration, overlap_frac)
    
    logger.info(f"Rest epochs: {len(rest_epochs)}, Move epochs: {len(move_epochs)}")
    
    if not rest_epochs or not move_epochs:
        raise ValueError("No epochs created. Check signal length and epoch duration.")
        
    # 3. Feature Extraction
    logger.info("Computing spectral features...")
    df_rest = compute_epoch_features(rest_epochs, fs, freq_bands)
    df_move = compute_epoch_features(move_epochs, fs, freq_bands)
    
    # 4. Merge and Label
    df_rest['target'] = 'rest'
    df_move['target'] = 'move'
    
    df_merged = pd.concat([df_rest, df_move], ignore_index=True)
    
    logger.info(f"Merged DataFrame created with {len(df_merged)} samples.")
    return df_merged
