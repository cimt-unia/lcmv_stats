# lcmv_stats/batch.py

"""
Batch processing for tensor-native group comparisons.
Chains: load → zscore+epoch → connectivity → statistics.
"""

import numpy as np
from pathlib import Path
from typing import Dict
import logging
from .utils import load_tensor
from .epoching import epoch_tensor
from .connectivity import extract_wpli_features
from .statistics import run_edgewise_permutation, prepare_connectivity_for_stats

logger = logging.getLogger(__name__)

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
            'group_summary': pd.DataFrame with edge-wise stats,
            'metadata': dict
        }
    """
    tens_a = load_tensor(tensor_path_a)
    tens_b = load_tensor(tensor_path_b)
    
    # Strict alignment check
    if not np.array_equal(tens_a['subject_ids'], tens_b['subject_ids']):
        raise ValueError("Subject IDs mismatch between tensors.")
    
    sfreq = tens_a['sfreq']
    
    # Epoch entire tensors at once (includes Z-scoring if do_zscore=True)
    # Output: (n_subj, n_ep, n_roi, n_samp)
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
    stats_df = run_edgewise_permutation(subj_a, subj_b)
    
    return {
        "per_subject": {
            "data_a": subj_a,
            "data_b": subj_b,
            "subject_ids": np.array(valid_subs)
        },
        "group_summary": stats_df,
        "metadata": {
            "band": band, "sfreq": sfreq,
            "epoch_duration": epoch_duration, "overlap": overlap,
            "do_zscore": do_zscore
        }
    }
