# lcmv_stats/_atlas.py

"""
Internal module that bridges lcmv_stats with lcmv_xtra's CIMT definitions.
"""
from pathlib import Path
import pandas as pd
import lcmv_xtra

# ─── Single Source of Truth ───
_LCMV_XTRA_ROOT = Path(lcmv_xtra.__file__).parent
_CIMT_LABELS_PATH = _LCMV_XTRA_ROOT / "data" / "cimt_atlas" / "cimt_atlas_labels.csv"

def get_cimt_labels() -> pd.DataFrame:
    """Load CIMT atlas labels directly from lcmv_xtra bundle."""
    if not _CIMT_LABELS_PATH.exists():
        raise FileNotFoundError(
            f"CIMT labels not found in lcmv_xtra at {_CIMT_LABELS_PATH}. "
            "Please ensure lcmv_xtra is correctly installed."
        )
    return pd.read_csv(_CIMT_LABELS_PATH)

def get_roi_index(roi_name: str) -> int:
    """Get the numeric index for a CIMT ROI name."""
    df = get_cimt_labels()
    match = df[df["roi_name"] == roi_name]
    if match.empty:
        raise ValueError(f"ROI '{roi_name}' not found in CIMT atlas.")
    return int(match.iloc[0]["index"])

def get_motor_network_indices() -> list[int]:
    """
    Reuse lcmv_xtra's motor network definition directly.
    """
    from lcmv_xtra.connectivity import select_cimt_motor_network_rois
    roi_info = select_cimt_motor_network_rois()
    return roi_info["target_rois"]

def get_motor_network_metadata() -> pd.DataFrame:
    """
    Get detailed metadata (full names, systems) for the motor network.
    """
    from lcmv_xtra.connectivity import get_cimt_motor_network_metadata
    return get_cimt_motor_network_metadata()