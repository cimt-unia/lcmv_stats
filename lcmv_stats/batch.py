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

def prepare_group_connectivity(
    events_df: pd.DataFrame,
    lcmv_root: Path,
    band: str = "low_beta",
    condition_col: str = "notes",
    valid_condition: str = "good"
) -> Dict[str, any]:
    """
    Extracts epochs and computes WPLI for all valid subjects in a group.
    
    Args:
        events_df: DataFrame with 'subject' column.
        lcmv_root: Path to LCMV derivatives.
        band: Frequency band for WPLI.
        
    Returns:
        Dictionary with 'in_data', 'out_data', 'valid_subs', and 'sfreqs'.
    """
    # Filter events
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
            # Get subject-specific events
            s_events = events_df[events_df['subject'].apply(lambda x: map_subject_to_subj(x) == sid)]
            
            # Extract epochs
            i_ep, o_ep = extract_event_epochs(sid, lcmv_root, s_events)
            
            if i_ep.size > 0:
                sf = get_subject_sfreq(sid, lcmv_root)
                sfreqs[sid] = sf
                
                # Compute WPLI
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