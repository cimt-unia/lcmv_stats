# lcmv_stats/connectivity.py

"""
Unified connectivity analysis module for CIMT source-space data (Tensor-Native).

Combines:
  - Feature extraction (WPLI, GPDC)
  - Batch tensor comparison
  - Statistical inference (permutation tests, effect sizes)
  - Reporting & visualization of connectivity results

All functions operate on tensors produced by lcmv_xtra.assemble_tensor
or epoch arrays produced by lcmv_stats.epoching.epoch_tensor.

OUTPUT FORMAT: All statistical results are saved as .npz files containing
structured arrays or dictionaries of numpy arrays. No CSVs are generated
for intermediate statistical data.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from mne.stats import permutation_t_test
import logging

from lcmv_xtra import compute_cimt_motor_connectivity
from ._atlas import get_cimt_labels, get_motor_network_indices, get_motor_network_metadata
from .utils import load_tensor
from .epoching import epoch_tensor

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

COLOR_INPHASE = '#2E8B57'
COLOR_OUTPHASE = '#DC143C'


# =============================================================================
# 1. FEATURE EXTRACTION
# =============================================================================

def extract_wpli_features(
    epochs_task_A: np.ndarray,
    epochs_task_B: np.ndarray,
    band: str = "low_beta",
    sfreq: float = 250.0
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Compute WPLI for the CIMT motor network for two conditions.

    Args:
        epochs_task_A: (n_epochs, n_rois, n_times) — Condition A epochs.
        epochs_task_B: (n_epochs, n_rois, n_times) — Condition B epochs.
        band: Frequency band name.
        sfreq: Sampling frequency.

    Returns:
        Tuple of (conn_matrix_A, conn_matrix_B) or (None, None) if empty.
    """
    if epochs_task_A.size == 0 or epochs_task_B.size == 0:
        logger.warning("Empty epochs provided for WPLI extraction.")
        return None, None

    try:
        conn_a = compute_cimt_motor_connectivity(epochs_task_A, band_name=band, sfreq=sfreq)
        conn_b = compute_cimt_motor_connectivity(epochs_task_B, band_name=band, sfreq=sfreq)
        return conn_a.values, conn_b.values
    except Exception as e:
        logger.error(f"WPLI calculation failed: {e}")
        return None, None


def extract_gpdc_features(
    epochs_task_A: np.ndarray,
    epochs_task_B: np.ndarray,
    sig_edge_indices: np.ndarray,
    roi_names: np.ndarray,
    sfreq: float,
    band_range: Tuple[float, float],
    nw: int = 3,
    n_tapers: int = 5
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
    """
    Compute GPDC only for ROIs involved in significant edges.
    Uses pure numpy inputs instead of pandas DataFrames.

    Args:
        epochs_task_A/B: Shape (n_epochs, 448, n_times).
        sig_edge_indices: Array of shape (n_sig_edges, 2) with ROI indices.
        roi_names: Array of ROI names corresponding to the full atlas.
        sfreq: Sampling frequency.
        band_range: (low_freq, high_freq) tuple.

    Returns:
        (gpdc_A, gpdc_B, valid_roi_names_array)
    """
    if sig_edge_indices.size == 0 or epochs_task_A.size == 0:
        return None, None, np.array([])

    # Get unique ROI indices from significant edges
    unique_roi_indices = np.unique(sig_edge_indices.flatten())
    
    # Filter to valid bounds
    valid_mask = unique_roi_indices < epochs_task_A.shape[1]
    valid_indices = unique_roi_indices[valid_mask]
    valid_names = roi_names[valid_indices]

    if len(valid_indices) == 0:
        logger.warning("No valid ROI indices found for GPDC subset.")
        return None, None, np.array([])

    in_subset = epochs_task_A[:, valid_indices, :]
    out_subset = epochs_task_B[:, valid_indices, :]

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
        gpdc_a = _compute_gpdc(in_subset)
        gpdc_b = _compute_gpdc(out_subset)
        return gpdc_a, gpdc_b, valid_names
    except Exception as e:
        logger.error(f"GPDC calculation failed: {e}")
        return None, None, np.array([])


# =============================================================================
# 2. STATISTICAL INFERENCE (NUMPY-NATIVE)
# =============================================================================

def cohens_d_paired(group_a: np.ndarray, group_b: np.ndarray) -> np.ndarray:
    """Calculate Cohen's d for paired samples (vectorized)."""
    diff = group_a - group_b
    std_diff = diff.std(axis=0, ddof=1)
    std_diff = np.where(std_diff == 0, 1.0, std_diff)
    return diff.mean(axis=0) / std_diff


def prepare_connectivity_for_stats(connectivity_matrices: list[np.ndarray]) -> np.ndarray:
    """Convert list of (n_rois, n_rois) matrices to (n_subjects, n_edges)."""
    if not connectivity_matrices:
        raise ValueError("No connectivity matrices provided.")
    n_rois = connectivity_matrices[0].shape[0]
    triu_idx = np.triu_indices(n_rois, k=1)
    vectors = [mat[triu_idx] for mat in connectivity_matrices]
    return np.stack(vectors)


def run_edgewise_permutation(
    data_a: np.ndarray,
    data_b: np.ndarray,
    n_permutations: int = 5000,
    alpha: float = 0.01
) -> Dict[str, np.ndarray]:
    """
    Edge-wise non-parametric permutation tests.
    Returns a dictionary of numpy arrays instead of a pandas DataFrame.
    
    Returns:
        Dictionary with keys:
            'edge_indices': (n_edges, 2) uint32
            't_stat': (n_edges,) float64
            'p_val': (n_edges,) float64
            'cohens_d': (n_edges,) float64
            'mean_a': (n_edges,) float64
            'mean_b': (n_edges,) float64
            'sd_a': (n_edges,) float64
            'sd_b': (n_edges,) float64
            'roi_names': (n_motor_rois,) object
    """
    if data_a.shape != data_b.shape or data_a.ndim != 2:
        raise ValueError("Input arrays must have same shape (n_subjects, n_edges)")

    n_subjects, n_edges = data_a.shape
    
    target_rois = np.array(get_motor_network_indices())
    triu_ix = np.triu_indices(len(target_rois), k=1)
    
    # Pre-allocate output arrays
    t_stats = np.zeros(n_edges, dtype=np.float64)
    p_vals = np.ones(n_edges, dtype=np.float64)
    ds = np.zeros(n_edges, dtype=np.float64)
    means_a = np.mean(data_a, axis=0)
    means_b = np.mean(data_b, axis=0)
    sds_a = np.std(data_a, axis=0, ddof=1)
    sds_b = np.std(data_b, axis=0, ddof=1)
    
    logger.info(f"Running permutation tests on {n_edges} edges across {n_subjects} subjects...")

    for i in range(n_edges):
        diffs = (data_a[:, i] - data_b[:, i]).reshape(-1, 1)
        if len(diffs) < 3:
            continue
            
        try:
            t_obs, p_val, _ = permutation_t_test(X=diffs, n_permutations=n_permutations, tail=0)
            t_stats[i] = t_obs[0]
            p_vals[i] = p_val[0]
        except Exception as e:
            logger.warning(f"Permutation test failed for edge {i}: {e}")
            
    ds = cohens_d_paired(data_a, data_b)
    
    # Build edge index pairs
    edge_indices = np.column_stack([triu_ix[0][:n_edges], triu_ix[1][:n_edges]]).astype(np.uint32)
    
    return {
        'edge_indices': edge_indices,
        't_stat': t_stats,
        'p_val': p_vals,
        'cohens_d': ds,
        'mean_a': means_a,
        'mean_b': means_b,
        'sd_a': sds_a,
        'sd_b': sds_b,
        'roi_names': target_rois
    }


# =============================================================================
# 3. BATCH TENSOR COMPARISON
# =============================================================================

def compare_tensors(
    tensor_path_a: str | Path,
    tensor_path_b: str | Path,
    band: str = "low_beta",
    epoch_duration: float = 2.0,
    overlap: float = 0.0,
    do_zscore: bool = True
) -> Dict:
    """
    End-to-end comparison of two condition tensors.
    Z-scoring is applied to continuous data before epoching.

    Returns:
        {
            'per_subject': {
                'data_a': (n_subj, n_edges),
                'data_b': (n_subj, n_edges),
                'subject_ids': (n_subj,)
            },
            'group_summary': Dict of numpy arrays (see run_edgewise_permutation),
            'metadata': dict
        }
    """
    tens_a = load_tensor(tensor_path_a)
    tens_b = load_tensor(tensor_path_b)

    if not np.array_equal(tens_a['subject_ids'], tens_b['subject_ids']):
        raise ValueError("Subject IDs mismatch between tensors.")

    sfreq = tens_a['sfreq']

    ep_a = epoch_tensor(tens_a['data'], sfreq, epoch_duration, overlap, do_zscore)
    ep_b = epoch_tensor(tens_b['data'], sfreq, epoch_duration, overlap, do_zscore)

    mats_a, mats_b, valid_subs = [], [], []

    for i, sid in enumerate(tens_a['subject_ids']):
        conn_a, _ = extract_wpli_features(ep_a[i], ep_a[i], band, sfreq)
        conn_b, _ = extract_wpli_features(ep_b[i], ep_b[i], band, sfreq)

        if conn_a is not None and conn_b is not None:
            mats_a.append(conn_a)
            mats_b.append(conn_b)
            valid_subs.append(sid)

    if not mats_a:
        raise ValueError("No valid connectivity matrices extracted.")

    subj_a = prepare_connectivity_for_stats(mats_a)
    subj_b = prepare_connectivity_for_stats(mats_b)
    stats_dict = run_edgewise_permutation(subj_a, subj_b)

    return {
        "per_subject": {
            "data_a": subj_a,
            "data_b": subj_b,
            "subject_ids": np.array(valid_subs)
        },
        "group_summary": stats_dict,
        "metadata": {
            "band": band, 
            "sfreq": sfreq,
            "epoch_duration": epoch_duration, 
            "overlap": overlap,
            "do_zscore": do_zscore
        }
    }


def save_comparison_results(result: Dict, output_path: str | Path):
    """
    Save comparison results to a single .npz file.
    Maintains clean numpy structure throughout the pipeline.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(
        output_path,
        # Per-subject data
        per_subj_data_a=result['per_subject']['data_a'],
        per_subj_data_b=result['per_subject']['data_b'],
        subject_ids=result['per_subject']['subject_ids'],
        # Group summary arrays
        edge_indices=result['group_summary']['edge_indices'],
        t_stat=result['group_summary']['t_stat'],
        p_val=result['group_summary']['p_val'],
        cohens_d=result['group_summary']['cohens_d'],
        mean_a=result['group_summary']['mean_a'],
        mean_b=result['group_summary']['mean_b'],
        sd_a=result['group_summary']['sd_a'],
        sd_b=result['group_summary']['sd_b'],
        roi_names=result['group_summary']['roi_names'],
        # Metadata
        metadata_band=result['metadata']['band'],
        metadata_sfreq=result['metadata']['sfreq'],
        metadata_epoch_duration=result['metadata']['epoch_duration'],
        metadata_overlap=result['metadata']['overlap'],
        metadata_do_zscore=result['metadata']['do_zscore']
    )
    logger.info(f"✅ Saved comparison results to {output_path}")


# =============================================================================
# 4. REPORTING & VISUALIZATION (NUMPY INPUTS)
# =============================================================================

def plot_top_edges(
    stats_dict: Dict[str, np.ndarray],
    band: str,
    n_top: int = 5,
    save_path: Optional[str] = None
):
    """
    Create enhanced visualization of top N significant edges.
    Accepts numpy dictionary from run_edgewise_permutation.
    """
    # Sort by absolute Cohen's d
    abs_d = np.abs(stats_dict['cohens_d'])
    top_idx = np.argsort(abs_d)[::-1][:n_top]
    
    if len(top_idx) == 0:
        return

    n_edges = len(top_idx)
    fig, axes = plt.subplots(1, n_edges, figsize=(6 * n_edges, 5))
    if n_edges == 1:
        axes = [axes]

    roi_names = stats_dict['roi_names']
    
    for plot_i, edge_i in enumerate(top_idx):
        ax = axes[plot_i]
        phases = ['Condition A', 'Condition B']
        means = [stats_dict['mean_a'][edge_i], stats_dict['mean_b'][edge_i]]
        sds = [stats_dict['sd_a'][edge_i], stats_dict['sd_b'][edge_i]]
        n_subj = len(stats_dict['mean_a'])  # Infer from array length
        
        sems = [sds[0] / np.sqrt(n_subj), sds[1] / np.sqrt(n_subj)]

        ax.bar(phases, means, color=[COLOR_INPHASE, COLOR_OUTPHASE], alpha=0.85)
        ax.errorbar(range(2), means, yerr=sems, fmt='none', ecolor='black', capsize=5)

        d = stats_dict['cohens_d'][edge_i]
        p = stats_dict['p_val'][edge_i]
        stats_text = f"d = {d:.2f}\np = {p:.4f}"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, ha='right', va='top')

        r1_idx, r2_idx = stats_dict['edge_indices'][edge_i]
        r1 = str(roi_names[r1_idx]).replace('_', ' ').title()
        r2 = str(roi_names[r2_idx]).replace('_', ' ').title()
        ax.set_title(f"{r1} ↔ {r2}", fontsize=10)
        ax.set_ylabel('WPLI Connectivity')

    fig.suptitle(f'Top {n_top} CIMT Connections: {band.replace("_", " ").title()} Band', fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def create_directed_effect_map(
    stats_dict: Dict[str, np.ndarray],
    gpdc_a: np.ndarray,
    gpdc_b: np.ndarray,
    gpdc_roi_names: np.ndarray,
    alpha: float = 0.05,
    condition_label: str = "A"
) -> Dict[str, np.ndarray]:
    """
    Combines undirected significance (WPLI p-val) with directed flow (GPDC).
    Returns pure numpy arrays instead of pandas DataFrame.

    Returns:
        Dictionary with keys:
            'edge_pair_indices': (n_sig, 2) uint32
            'cohens_d': (n_sig,) float64
            'p_val': (n_sig,) float64
            'directional_strength': (n_sig,) float64
            'dominant_direction': (n_sig,) object (e.g., 'ROI_A → ROI_B')
    """
    sig_mask = stats_dict['p_val'] < alpha
    if not np.any(sig_mask) or gpdc_a is None:
        return {}

    sig_indices = np.where(sig_mask)[0]
    
    # Map GPDC ROI names to indices
    gpdc_name_to_idx = {name: i for i, name in enumerate(gpdc_roi_names)}
    full_roi_names = stats_dict['roi_names']
    
    directions = []
    strengths = []
    valid_sig_indices = []
    
    for edge_i in sig_indices:
        r1_idx, r2_idx = stats_dict['edge_indices'][edge_i]
        r1_name = str(full_roi_names[r1_idx])
        r2_name = str(full_roi_names[r2_idx])
        
        if r1_name not in gpdc_name_to_idx or r2_name not in gpdc_name_to_idx:
            continue
            
        gi1 = gpdc_name_to_idx[r1_name]
        gi2 = gpdc_name_to_idx[r2_name]
        
        gpdc_mat = gpdc_a if condition_label == "A" else gpdc_b
        
        flow_1to2 = gpdc_mat[gi1, gi2]
        flow_2to1 = gpdc_mat[gi2, gi1]
        
        if flow_1to2 > flow_2to1:
            directions.append(f"{r1_name} → {r2_name}")
            strengths.append(flow_1to2 - flow_2to1)
        else:
            directions.append(f"{r2_name} → {r1_name}")
            strengths.append(flow_2to1 - flow_1to2)
            
        valid_sig_indices.append(edge_i)
    
    if not valid_sig_indices:
        return {}
        
    valid_sig_indices = np.array(valid_sig_indices)
    
    return {
        'edge_pair_indices': stats_dict['edge_indices'][valid_sig_indices],
        'cohens_d': stats_dict['cohens_d'][valid_sig_indices],
        'p_val': stats_dict['p_val'][valid_sig_indices],
        'directional_strength': np.array(strengths),
        'dominant_direction': np.array(directions, dtype=object)
    }
