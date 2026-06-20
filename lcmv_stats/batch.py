# lcmv_stats/batch.py

"""
Batch processing helpers for lcmv_stats.
Handles subject iteration, data aggregation, and statistical preparation.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
import logging
from .utils import map_subject_to_subj, get_subject_sfreq
from .epoching import extract_event_epochs
from .connectivity import extract_wpli_features

logger = logging.getLogger(__name__)


def prepare_connectivity_for_stats(
    connectivity_matrices: List[np.ndarray]
) -> np.ndarray:
    """
    Universal processor to convert a list of connectivity matrices into 
    the (n_subjects, n_edges) array required for statistical testing.
    
    Args:
        connectivity_matrices: A list of 2D numpy arrays (N_rois x N_rois).
        
    Returns:
        A 2D numpy array of shape (n_subjects, n_edges).
    """
    if not connectivity_matrices:
        raise ValueError("No connectivity matrices provided.")
    
    # 1. Validate that all matrices are square and have the same dimensions
    n_rois = connectivity_matrices[0].shape[0]
    for i, mat in enumerate(connectivity_matrices):
        if mat.shape != (n_rois, n_rois):
            raise ValueError(f"Subject {i} has matrix shape {mat.shape}, expected ({n_rois}, {n_rois})")
        if mat.ndim != 2:
            raise ValueError(f"Subject {i} input is not a 2D matrix.")

    # 2. Extract Upper Triangle for all subjects at once
    triu_idx = np.triu_indices(n_rois, k=1)
    vectors = [mat[triu_idx] for mat in connectivity_matrices]
    
    # 3. Stack into final array
    return np.stack(vectors)


def prepare_group_comparison(
    events_df: pd.DataFrame,
    lcmv_root: Path,
    band: str = "low_beta",
    condition_col: str = 'task_type',
    val_a: str = 'rest',
    val_b: str = 'task',
    is_phase_comparison: bool = False
) -> Dict[str, any]:
    """
    Generic group-level processor for comparing any two conditions.
    
    This function replaces both legacy batch helpers. It can handle:
    1. Condition A vs. Condition B (e.g., Rest vs. Task)
    2. In-Phase vs. Out-Phase (if is_phase_comparison=True)
    
    Args:
        events_df: DataFrame containing trial metadata.
        lcmv_root: Path to LCMV derivatives.
        band: Frequency band for connectivity.
        condition_col: Column name in CSV to filter conditions.
        val_a: Value for Condition A (e.g., 'rest' or 'in').
        val_b: Value for Condition B (e.g., 'task' or 'out').
        is_phase_comparison: If True, extracts In/Out epochs from the SAME trials 
                             rather than filtering by condition_col.
        
    Returns:
        Dictionary with 'data_a', 'data_b', 'valid_subs'.
    """
    # Ensure subject IDs are standardized
    subject_ids = sorted([map_subject_to_subj(s) for s in events_df['subject'].unique()])
    
    all_mats_a = []
    all_mats_b = []
    valid_subs = []
    
    logger.info(f"Processing {len(subject_ids)} subjects...")

    for sid in subject_ids:
        try:
            # 1. Filter Events for this specific subject
            subj_events = events_df[events_df['subject'].apply(lambda x: map_subject_to_subj(x) == sid)]
            
            if is_phase_comparison:
                # For Phase comparison, we use ALL good trials for this subject
                ev_a = subj_events
                ev_b = subj_events
            else:
                # For Condition comparison, we filter by the column
                ev_a = subj_events[subj_events[condition_col] == val_a]
                ev_b = subj_events[subj_events[condition_col] == val_b]
            
            # Skip if either condition is missing for this subject
            if ev_a.empty or ev_b.empty: 
                logger.debug(f"Skipping {sid}: Missing events.")
                continue
            
            # 2. Extract Epochs
            # Note: extract_event_epochs always returns (in_epochs, out_epochs)
            in_a, out_a = extract_event_epochs(sid, lcmv_root, ev_a)
            in_b, out_b = extract_event_epochs(sid, lcmv_root, ev_b)
            
            if in_a.size == 0 or in_b.size == 0: 
                continue
            
            # 3. Compute Connectivity
            sfreq = get_subject_sfreq(sid, lcmv_root, condition="bima_off")
            
            if is_phase_comparison:
                # Compare In vs Out within the same trials
                conn_a, _ = extract_wpli_features(in_a, out_a, band, sfreq)
                # For phase, we usually compare the 'in' matrix of A vs 'out' matrix of A
                # But to keep the signature consistent (A vs B), we treat 'in' as A and 'out' as B
                conn_b = _ # This is actually the out_conn from the line above
                # Let's re-calculate to be explicit for the generic structure:
                _, conn_b = extract_wpli_features(in_a, out_a, band, sfreq)
            else:
                # Compare Condition A vs Condition B
                conn_a, _ = extract_wpli_features(in_a, out_a, band, sfreq)
                conn_b, _ = extract_wpli_features(in_b, out_b, band, sfreq)
            
            if conn_a is None or conn_b is None: 
                continue
            
            all_mats_a.append(conn_a)
            all_mats_b.append(conn_b)
            valid_subs.append(sid)
            
        except Exception as e:
            logger.warning(f"Failed to process subject {sid}: {e}")
            continue

    if not all_mats_a:
        raise ValueError("No valid data extracted for either condition. Check your event labels and paths.")

    # 4. Use the universal processor to create final arrays
    return {
        'data_a': prepare_connectivity_for_stats(all_mats_a),
        'data_b': prepare_connectivity_for_stats(all_mats_b),
        'valid_subs': valid_subs
    }
