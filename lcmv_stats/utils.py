# lcmv_stats/utils.py

import numpy as np
from pathlib import Path
import logging
from ._atlas import get_cimt_labels

logger = logging.getLogger(__name__)

def load_tensor(tensor_path: str | Path) -> dict:
    """
    Load a study tensor directly from disk.
    
    Returns:
        dict: {
            'data': np.ndarray (n_subj, n_roi, n_time),
            'subject_ids': np.ndarray (n_subj,),
            'sfreq': float
        }
    """
    tensor_path = Path(tensor_path)
    if not tensor_path.exists():
        raise FileNotFoundError(f"Tensor not found: {tensor_path}")
        
    master = np.load(tensor_path, allow_pickle=True)
    return {
        "data": master["data"],
        "subject_ids": master["subject_ids"],
        "sfreq": float(master["sfreq"])
    }

def get_roi_indices(roi_names: list[str]) -> list[int]:
    """
    Map CIMT ROI names to their integer indices in the tensor.
    
    Args:
        roi_names: List of ROI names (e.g., ['L_4_ROI', 'STN-lh']).
        
    Returns:
        List of integer indices corresponding to axis 1 of the tensor.
    """
    atlas = get_cimt_labels()
    indices = []
    for name in roi_names:
        match = atlas[atlas['roi_name'] == name]
        if match.empty:
            logger.warning(f"ROI '{name}' not found in CIMT atlas.")
            continue
        indices.append(int(match.iloc[0]['index']))
    return indices
