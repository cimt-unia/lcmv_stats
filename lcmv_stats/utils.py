# lcmv_stats/utils.py

import json
import re
import numpy as np
from pathlib import Path
from typing import Optional
import logging

from ._atlas import get_roi_index

logger = logging.getLogger(__name__)

def map_subject_to_subj(subject_name: str) -> str:
    """Convert Sbj001 to sub-001 format."""
    if subject_name.startswith('Sbj'):
        num = subject_name.replace("Sbj", "").lstrip("0") or "0"
    else:
        numbers = re.findall(r'\d+', subject_name)
        num = numbers[0] if numbers else "0"
    return f"sub-{int(num):03d}"

def get_subject_sfreq(subject_id: str, lcmv_root: Path, condition: str = "bima_off") -> float:
    """Retrieve sampling frequency from CIMT pipeline metadata."""
    metadata_file = lcmv_root / f"{subject_id}_{condition}" / "pipeline_metadata.json"
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_file}")
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    return float(metadata['sfreq_hz'])

def get_roi_time_series(
    subject_id: str, 
    lcmv_root: Path, 
    roi_name: str, 
    condition: str = "bima_off"
) -> Optional[np.ndarray]:
    """
    Extract the 1D time series for a specific CIMT ROI from the unified time courses.
    
    Args:
        subject_id: Subject identifier (e.g., 'sub-001').
        lcmv_root: Root path to LCMV derivatives.
        roi_name: Name of the ROI in the CIMT atlas (e.g., 'R_4_ROI').
        condition: Condition folder name (default: 'bima_off').
        
    Returns:
        1D numpy array of shape (n_times,) or None if extraction fails.
    """
    subj_dir = lcmv_root / f"{subject_id}_{condition}"
    cimt_file = subj_dir / "cimt_time_courses.npy"
    
    if not cimt_file.exists():
        logger.warning(f"CIMT time courses not found for {subject_id} at {cimt_file}")
        return None
        
    try:
        # Load full 448-ROI time courses: Shape (448, n_times)
        cimt_tc = np.load(cimt_file)
    except Exception as e:
        logger.error(f"Failed to load CIMT data for {subject_id}: {e}")
        return None
        
    # Get ROI Index
    try:
        roi_idx = get_roi_index(roi_name)
    except ValueError as e:
        logger.error(e)
        return None
        
    # Validate index bounds
    if roi_idx >= cimt_tc.shape[0]:
        logger.error(f"ROI index {roi_idx} out of bounds for data shape {cimt_tc.shape}")
        return None
        
    # Extract 1D time series
    return cimt_tc[roi_idx, :]
