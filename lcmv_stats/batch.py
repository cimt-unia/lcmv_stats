# lcmv_stats/batch.py

"""
Batch processing helpers for lcmv_stats.
Handles subject iteration and data aggregation.
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

def prepare_group_comparison(
    events_df: pd.DataFrame,
    lcmv_root: Path,
    band: str = "low_beta",
    condition_col: str = 'task_type',
    val_a: str = 'rest',
    val_b: str = 'task'
) -> Dict[str, any]:
    """
    Generic group-level processor for comparing any two conditions.
    
    Args:
        events_df: DataFrame containing trial metadata.
        lcmv_root: Path to LCMV derivatives.
        band: Frequency band for connectivity.
        condition_col: Column name in CSV to filter conditions.
        val_a: Value for Condition A (e.g., 'rest').
        val_b: Value for Condition B (e.g., 'task').
        
    Returns:
        Dictionary with 'data_a', 'data_b', 'valid_subs'.
    """
    # Ensure subject IDs are standardized
    subject_ids = sorted([map_subject_to_subj(s) for s in events_df['subject'].unique()])
    
    all_vec_a = []
    all_vec_b = []
    valid_subs = []
    
    logger.info(f"Processing {len(subject_ids)} subjects for '{val_a}' vs '{val_b}'...")

    for sid in subject_ids:
        try:
            # 1. Filter Events for this specific subject
            # We use apply here to handle mixed formats in the 'subject' column
            subj_events = events_df[events_df['subject'].apply(lambda x: map_subject_to_subj(x) == sid)]
            
            ev_a = subj_events[subj_events[condition_col] == val_a]
            ev_b = subj_events[subj_events[condition_col] == val_b]
            
            # Skip if either condition is missing for this subject
            if ev_a.empty or ev_b.empty: 
                logger.debug(f"Skipping {sid}: Missing events for '{val_a}' or '{val_b}'.")
                continue
            
            # 2. Extract Epochs
            in_a, out_a = extract_event_epochs(sid, lcmv_root, ev_a)
            in_b, out_b = extract_event_epochs(sid, lcmv_root, ev_b)
            
            if in_a.size == 0 or in_b.size == 0: 
                continue
            
            # 3. Compute Connectivity
            # We default to 'bima_off' for sfreq if not specified, as it's the standard pipeline output
            sfreq = get_subject_sfreq(sid, lcmv_root, condition="bima_off")
            
            conn_a, _ = extract_wpli_features(in_a, out_a, band, sfreq)
            conn_b, _ = extract_wpli_features(in_b, out_b, band, sfreq)
            
            if conn_a is None or conn_b is None: 
                continue
            
            # 4. Vectorize (Upper Triangle)
            triu_idx = np.triu_indices(conn_a.shape[0], k=1)
            all_vec_a.append(conn_a[triu_idx])
            all_vec_b.append(conn_b[triu_idx])
            valid_subs.append(sid)
            
        except Exception as e:
            logger.warning(f"Failed to process subject {sid}: {e}")
            continue

    if not all_vec_a:
        raise ValueError("No valid data extracted for either condition. Check your event labels and paths.")

    return {
        'data_a': np.stack(all_vec_a),
        'data_b': np.stack(all_vec_b),
        'valid_subs': valid_subs
    }


def prepare_group_vectors(connectivity_vectors: list[np.ndarray]) -> np.ndarray:
    """
    Safely stacks a list of upper-triangle connectivity vectors 
    into the (n_subjects, n_edges) array required by lcmv_stats statistics.
    """
    if not connectivity_vectors:
        raise ValueError("No valid connectivity vectors provided.")
    
    n_edges = connectivity_vectors[0].shape[0]
    
    if connectivity_vectors[0].ndim != 1:
        raise ValueError("Input must be a list of 1D vectors (upper triangles), not 2D matrices.")

    for i, vec in enumerate(connectivity_vectors):
        if vec.shape[0] != n_edges:
            raise ValueError(f"Subject {i} has {vec.shape[0]} edges, expected {n_edges}")
        if vec.ndim != 1:
             raise ValueError(f"Subject {i} input is not a 1D vector.")
    
    return np.stack(connectivity_vectors)


def prepare_group_connectivity(
    events_df: pd.DataFrame,
    lcmv_root: Path,
    band: str = "low_beta",
    condition_col: str = "notes",
    valid_condition: str = "good"
) -> Dict[str, any]:
    """
    Legacy helper for In/Out phase comparisons.
    """
    if condition_col in events_df.columns:
        events_df = events_df[events_df[condition_col] == valid_condition]
        
    subject_names = events_df['subject'].unique()
    subject_ids = [map_subject_to_subj(name) for name in subject_names]
    
    all_in_mats = []
    all_out_mats = []
    valid_subs = []
    sfreqs = {}
    
    logger.info(f"Processing {len(subject_ids)} subjects for band '{band}'...")
    
    for sid in subject_ids:
        try:
            s_events = events_df[events_df['subject'].apply(lambda x: map_subject_to_subj(x) == sid)]
            i_ep, o_ep = extract_event_epochs(sid, lcmv_root, s_events)
            
            if i_ep.size > 0:
                sf = get_subject_sfreq(sid, lcmv_root, condition="bima_off")
                sfreqs[sid] = sf
                
                ic, oc = extract_wpli_features(i_ep, o_ep, band, sf)
                
                if ic is not None:
                    triu_ix = np.triu_indices(ic.shape[0], k=1)
                    all_in_mats.append(ic[triu_ix])
                    all_out_mats.append(oc[triu_ix])
                    valid_subs.append(sid)
                    
        except Exception as e:
            logger.warning(f"Failed to process subject {sid}: {e}")
            continue
            
    if not all_in_mats:
        raise ValueError("No valid connectivity data extracted from any subject.")
        
    return {
        'in_data': np.stack(all_in_mats),
        'out_data': np.stack(all_out_mats),
        'valid_subs': valid_subs,
        'sfreqs': sfreqs
    }
