# lcmv_stats/machine_learning.py

"""
Machine Learning Feature Engineering for CIMT source-space data (Tensor-Native).

Operates on 5D epoched tensors (n_subjects, n_epochs, n_rois, n_samples)
produced by lcmv_stats.epoching.epoch_tensor.

Computes band power features vectorized across all subjects/epochs/ROIs.
Returns pure NumPy arrays. No pandas. No raw signal processing.
Z-scoring and epoching are assumed to have been applied upstream.
"""

import numpy as np
from scipy import signal
from typing import Dict, Optional, Tuple, List
import logging

from ._atlas import resolve_roi_indices, select_network, get_cimt_labels

logger = logging.getLogger(__name__)


def get_frequency_bands() -> Dict[str, Tuple[float, float]]:
    """Standard frequency band definitions (Hz) for ML feature extraction."""
    return {
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 12),
        "low_beta": (13, 20),
        "high_beta": (20, 30),
        "beta": (13, 30),
        "low_gamma": (30, 60),
        "high_gamma": (60, 100),
    }


def extract_band_power_features(
    epochs: np.ndarray,
    sfreq: float,
    freq_bands: Optional[Dict[str, Tuple[float, float]]] = None,
    roi_indices: Optional[np.ndarray] = None,
    log_transform: bool = True
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Extract band power features from epoched tensor (vectorized, multi-subject).

    Args:
        epochs: (n_subjects, n_epochs, n_rois, n_samples) — already Z-scored.
        sfreq: Sampling frequency in Hz.
        freq_bands: Band definitions. Uses defaults if None.
        roi_indices: Optional subset of ROI indices. If None, uses all ROIs.
        log_transform: Apply log10(power + eps) for ML stability.

    Returns:
        Tuple of:
            features: (n_subjects, n_epochs, n_selected_rois, n_bands) float64
            band_names: List of band name strings
            roi_indices_used: (n_selected_rois,) int64
    """
    if epochs.ndim != 4:
        raise ValueError(f"Expected 4D epochs (n_subj, n_ep, n_roi, n_samp), got shape {epochs.shape}")

    if freq_bands is None:
        freq_bands = get_frequency_bands()

    n_subj, n_ep, n_rois, n_samp = epochs.shape
    band_names = list(freq_bands.keys())
    n_bands = len(band_names)

    # Subset ROIs if requested
    if roi_indices is not None:
        epochs = epochs[:, :, roi_indices, :]
        n_rois = len(roi_indices)

    # Vectorized Welch PSD across all subjects, epochs, and ROIs at once
    # Reshape to (n_subj * n_ep * n_rois, n_samples) for batch processing
    flat_data = epochs.reshape(-1, n_samp)

    # Use full epoch length for maximum frequency resolution
    nperseg = min(n_samp, int(sfreq * 2))
    noverlap = nperseg // 2

    freqs, psd_flat = signal.welch(
        flat_data, fs=sfreq, nperseg=nperseg,
        noverlap=noverlap, window='hann', axis=1
    )

    # Integrate power per band: (n_subj*n_ep*n_roi, n_bands)
    band_powers = np.zeros((flat_data.shape[0], n_bands), dtype=np.float64)
    for b_idx, (band_name, (f_low, f_high)) in enumerate(freq_bands.items()):
        mask = (freqs >= f_low) & (freqs <= f_high)
        if np.any(mask):
            band_powers[:, b_idx] = np.trapezoid(psd_flat[:, mask], freqs[mask], axis=1)
        else:
            logger.warning(f"No frequency bins found for band '{band_name}' ({f_low}-{f_high} Hz)")

    if log_transform:
        band_powers = np.log10(band_powers + 1e-15)

    # Reshape back to (n_subjects, n_epochs, n_rois, n_bands)
    features = band_powers.reshape(n_subj, n_ep, n_rois, n_bands)

    roi_indices_used = roi_indices if roi_indices is not None else np.arange(n_rois, dtype=np.int64)

    logger.info(
        f"Extracted ML features: {features.shape} "
        f"({n_subj} subjects × {n_ep} epochs × {n_rois} ROIs × {n_bands} bands)"
    )

    return features, band_names, roi_indices_used


def prepare_ml_dataset(
    tensor_path_a: str,
    tensor_path_b: str,
    sfreq: float,
    roi_names: Optional[List[str]] = None,
    functional_systems: Optional[List[str]] = None,
    sub_systems: Optional[List[str]] = None,
    epoch_duration: float = 2.0,
    overlap: float = 0.5,
    freq_bands: Optional[Dict[str, Tuple[float, float]]] = None,
    log_transform: bool = True
) -> Dict:
    """
    End-to-end ML dataset preparation from two condition tensors.

    Loads tensors → epochs (with Z-scoring) → extracts band power features →
    returns structured dict ready for sklearn/pytorch/tensorflow.

    Args:
        tensor_path_a: Path to Condition A .npz tensor.
        tensor_path_b: Path to Condition B .npz tensor.
        sfreq: Sampling frequency.
        roi_names: Explicit ROI names (optional).
        functional_systems: Metadata-based ROI selection (optional).
        sub_systems: Metadata-based ROI selection (optional).
        epoch_duration: Epoch duration in seconds.
        overlap: Epoch overlap fraction.
        freq_bands: Band definitions. Uses defaults if None.
        log_transform: Apply log10 transform to band powers.

    Returns:
        Dict with keys:
            'features_a': (n_subj, n_ep_a, n_rois, n_bands)
            'features_b': (n_subj, n_ep_b, n_rois, n_bands)
            'labels': (n_total_samples,) — 0 for A, 1 for B
            'subject_ids': (n_subj,)
            'band_names': list of band name strings
            'roi_names': list of ROI name strings
            'sfreq': float
            'epoch_duration': float
    """
    from .utils import load_tensor
    from .epoching import epoch_tensor

    tens_a = load_tensor(tensor_path_a)
    tens_b = load_tensor(tensor_path_b)

    if not np.array_equal(tens_a['subject_ids'], tens_b['subject_ids']):
        raise ValueError("Subject IDs mismatch between tensors.")

    # Resolve ROI indices
    if roi_names is not None:
        roi_idx = resolve_roi_indices(roi_names)
    elif functional_systems is not None or sub_systems is not None:
        roi_idx = select_network(
            functional_systems=functional_systems,
            sub_systems=sub_systems
        )
    else:
        # Default: Motor-Basal-Executive-STN network
        roi_idx = select_network(
            functional_systems=['Motor', 'BasalGanglia', 'Frontoparietal'],
            sub_systems=['Primary', 'Premotor', 'Supplementary', 'Eye',
                         'DLPFC', 'IFJ', 'IFS', 'VLPFC', 'Subthalamic']
        )

    # Epoch both conditions (Z-scoring applied before epoching)
    ep_a = epoch_tensor(tens_a['data'], sfreq, epoch_duration, overlap, do_zscore=True)
    ep_b = epoch_tensor(tens_b['data'], sfreq, epoch_duration, overlap, do_zscore=True)

    # Extract features
    feat_a, band_names, roi_idx_used = extract_band_power_features(
        ep_a, sfreq, freq_bands, roi_indices=roi_idx, log_transform=log_transform
    )
    feat_b, _, _ = extract_band_power_features(
        ep_b, sfreq, freq_bands, roi_indices=roi_idx, log_transform=log_transform
    )

    # Build labels: 0 = Condition A, 1 = Condition B
    n_ep_a = feat_a.shape[1]
    n_ep_b = feat_b.shape[1]
    labels = np.concatenate([
        np.zeros(n_ep_a, dtype=np.int8),
        np.ones(n_ep_b, dtype=np.int8)
    ])

    # Get ROI names for metadata
    atlas_df = get_cimt_labels()
    selected_roi_names = atlas_df.loc[
        atlas_df['index'].isin(roi_idx_used), 'roi_name'
    ].tolist()

    return {
        'features_a': feat_a,
        'features_b': feat_b,
        'labels': labels,
        'subject_ids': tens_a['subject_ids'],
        'band_names': band_names,
        'roi_names': selected_roi_names,
        'sfreq': sfreq,
        'epoch_duration': epoch_duration,
    }


def flatten_for_sklearn(dataset: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Flatten ML dataset into 2D feature matrix for sklearn classifiers.

    Args:
        dataset: Output from prepare_ml_dataset().

    Returns:
        Tuple of:
            X: (n_total_samples, n_rois * n_bands) float64
            y: (n_total_samples,) int8 labels
            subject_ids_per_sample: (n_total_samples,) subject ID per sample
    """
    feat_a = dataset['features_a']  # (n_subj, n_ep_a, n_roi, n_band)
    feat_b = dataset['features_b']  # (n_subj, n_ep_b, n_roi, n_band)
    n_subj = feat_a.shape[0]
    n_roi = feat_a.shape[2]
    n_band = feat_a.shape[3]

    # Flatten each subject's epochs: (n_subj * n_ep, n_roi * n_band)
    flat_a = feat_a.reshape(n_subj * feat_a.shape[1], n_roi * n_band)
    flat_b = feat_b.reshape(n_subj * feat_b.shape[1], n_roi * n_band)

    X = np.vstack([flat_a, flat_b])
    y = dataset['labels']

    # Repeat subject IDs for each epoch
    subj_ids = dataset['subject_ids']
    n_ep_a = feat_a.shape[1]
    n_ep_b = feat_b.shape[1]
    subj_per_sample = np.concatenate([
        np.repeat(subj_ids, n_ep_a),
        np.repeat(subj_ids, n_ep_b)
    ])

    logger.info(f"Flattened ML dataset: X={X.shape}, y={y.shape}")
    return X, y, subj_per_sample
