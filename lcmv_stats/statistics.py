# lcmv_stats/statistics.py

"""
Core statistical inference tools.
"""
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List
from mne.stats import permutation_t_test
import logging
from ._atlas import get_motor_network_indices, get_motor_network_metadata

logger = logging.getLogger(__name__)

def cohens_d_paired(inphase: np.ndarray, outphase: np.ndarray) -> float:
    """Calculate Cohen's d for paired samples."""
    diff = inphase - outphase
    std_diff = diff.std(ddof=1)
    if std_diff == 0:
        return 0.0
    return diff.mean() / std_diff

def run_edgewise_permutation(
    in_data: np.ndarray, 
    out_data: np.ndarray, 
    n_permutations: int = 5000,
    alpha: float = 0.01
) -> pd.DataFrame:
    """
    Perform edge-wise non-parametric permutation tests.
    Automatically maps edges to CIMT Motor Network ROI names.
    
    Args:
        in_data: Array of shape (n_subjects, n_edges).
        out_data: Array of shape (n_subjects, n_edges).
        n_permutations: Number of permutations.
        alpha: Significance threshold.
        
    Returns:
        DataFrame containing statistics for each edge.
    """
    if in_data.shape != out_data.shape or in_data.ndim != 2:
        raise ValueError("Input arrays must have the same shape (n_subjects, n_edges)")
        
    n_subjects, n_edges = in_data.shape
    results = []
    
    # Get ROI names for labeling
    target_rois = get_motor_network_indices()
    triu_ix = np.triu_indices(len(target_rois), k=1)
    
    # Ensure we have enough edges to map
    if n_edges != len(triu_ix[0]):
        logger.warning(f"Number of edges ({n_edges}) does not match number of motor network edges ({len(triu_ix[0])}). Mapping may be incorrect.")
        # Fallback: use generic indices if mismatch
        edge_pairs = [(f"Edge_{i}_A", f"Edge_{i}_B") for i in range(n_edges)]
        name_map = {}
    else:
        edge_pairs = [(target_rois[i], target_rois[j]) for i, j in zip(*triu_ix)]
        # Get full names for reporting
        meta_df = get_motor_network_metadata()
        name_map = dict(zip(meta_df['roi_name'], meta_df['region_full_name'])) if 'region_full_name' in meta_df.columns else {}

    logger.info(f"Running permutation tests on {n_edges} edges across {n_subjects} subjects...")
    
    for i in range(n_edges):
        diffs = (in_data[:, i] - out_data[:, i]).reshape(-1, 1)
        
        if len(diffs) < 3:
            continue
            
        try:
            t_obs, p_val, _ = permutation_t_test(
                X=diffs,
                n_permutations=n_permutations,
                tail=0
            )
            
            d = cohens_d_paired(in_data[:, i], out_data[:, i])
            
            # Map edge index to ROI names
            if n_edges == len(triu_ix[0]):
                r1_short, r2_short = edge_pairs[i]
                r1_full = name_map.get(r1_short, r1_short)
                r2_full = name_map.get(r2_short, r2_short)
            else:
                r1_short, r2_short = f"Edge_{i}_A", f"Edge_{i}_B"
                r1_full, r2_full = r1_short, r2_short
            
            results.append({
                'edge_index': i,
                'roi1': r1_short,
                'roi2': r2_short,
                'roi1_full': r1_full,
                'roi2_full': r2_full,
                'n_subjects': n_subjects,
                'mean_inphase': np.mean(in_data[:, i]),
                'sd_inphase': np.std(in_data[:, i], ddof=1),
                'mean_outphase': np.mean(out_data[:, i]),
                'sd_outphase': np.std(out_data[:, i], ddof=1),
                'mean_diff': np.mean(diffs),
                't_stat': t_obs[0],
                'p_val': p_val[0],
                'cohens_d': d,
                'abs_d': abs(d)
            })
        except Exception as e:
            logger.warning(f"Permutation test failed for edge {i}: {e}")
            continue
            
    return pd.DataFrame(results)
