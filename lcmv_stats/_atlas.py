# lcmv_stats/_atlas.py

"""
Atlas bridge module: index-first ROI selection for tensor operations.

Provides:
  - Cached atlas loading (read CSV once per process)
  - O(1) name → index resolution
  - Flexible network selection by functional_system and/or sub_system
  - Backward-compatible wrappers for legacy motor network functions
  - All primary outputs are np.ndarray of int64 indices for direct tensor slicing
"""

from pathlib import Path
from functools import lru_cache
import numpy as np
import pandas as pd
import lcmv_xtra
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_LCMV_XTRA_ROOT = Path(lcmv_xtra.__file__).parent
_CIMT_LABELS_PATH = _LCMV_XTRA_ROOT / "data" / "cimt_atlas" / "cimt_atlas_labels.csv"


# =============================================================================
# CACHED ATLAS ACCESS
# =============================================================================

@lru_cache(maxsize=1)
def get_cimt_labels() -> pd.DataFrame:
    """Load CIMT atlas labels (448 ROIs). Cached after first call."""
    if not _CIMT_LABELS_PATH.exists():
        raise FileNotFoundError(
            f"CIMT labels not found at {_CIMT_LABELS_PATH}. "
            "Ensure lcmv_xtra is correctly installed."
        )
    return pd.read_csv(_CIMT_LABELS_PATH)


@lru_cache(maxsize=1)
def _get_name_to_index_map() -> dict[str, int]:
    """Cached O(1) name→index mapping."""
    df = get_cimt_labels()
    return dict(zip(df["roi_name"], df["index"].astype(int)))


# =============================================================================
# INDEX-FIRST ROI RESOLUTION
# =============================================================================

def resolve_roi_indices(roi_names: list[str]) -> np.ndarray:
    """
    Resolve explicit ROI names to integer indices.

    Args:
        roi_names: List of CIMT ROI name strings.

    Returns:
        np.ndarray of shape (len(roi_names),), dtype=int64.
    """
    name_map = _get_name_to_index_map()
    missing = [n for n in roi_names if n not in name_map]
    if missing:
        raise ValueError(f"ROIs not found in CIMT atlas: {missing}")
    return np.array([name_map[n] for n in roi_names], dtype=np.int64)


def select_network(
    functional_systems: Optional[list[str]] = None,
    sub_systems: Optional[list[str]] = None,
    hemisphere: Optional[str] = None,
) -> np.ndarray:
    """
    Select ROI indices by metadata filtering on the CIMT atlas.

    All filters are combined with AND logic. At least one filter must be provided.

    Args:
        functional_systems: List of functional_system values to include.
            e.g., ['Motor', 'BasalGanglia', 'Frontoparietal']
        sub_systems: List of sub_system values to include.
            e.g., ['Primary', 'Premotor', 'Subthalamic', 'DLPFC']
        hemisphere: 'Left' or 'Right' to restrict to one hemisphere.

    Returns:
        Sorted np.ndarray of int64 indices for tensor axis-1 slicing.

    Examples:
        # Motor-Basal-Executive-STN network
        select_network(
            functional_systems=['Motor', 'BasalGanglia', 'Frontoparietal'],
            sub_systems=['Primary', 'Premotor', 'Supplementary', 'Eye',
                         'DLPFC', 'IFJ', 'IFS', 'VLPFC', 'Subthalamic']
        )

        # Only left-hemisphere motor cortex
        select_network(functional_systems=['Motor'], hemisphere='Left')

        # All cerebellar ROIs (motor + social + demand + action)
        select_network(sub_systems=['Cerebellum'])

        # STN only
        select_network(sub_systems=['Subthalamic'])

        # Ventral attention network
        select_network(functional_systems=['VentralAttention'])
    """
    if functional_systems is None and sub_systems is None and hemisphere is None:
        raise ValueError("At least one filter must be provided.")

    df = get_cimt_labels()
    mask = pd.Series(True, index=df.index)

    if functional_systems is not None:
        mask &= df["functional_system"].isin(functional_systems)

    if sub_systems is not None:
        mask &= df["sub_system"].isin(sub_systems)

    if hemisphere is not None:
        mask &= df["hemisphere"] == hemisphere

    indices = df.loc[mask, "index"].values.astype(np.int64)
    indices.sort()

    if len(indices) == 0:
        logger.warning("select_network returned 0 ROIs. Check filter values.")

    return indices


def get_available_systems() -> dict[str, list[str]]:
    """
    Return all unique functional_system and sub_system values in the atlas.
    Useful for interactive exploration and validation of filter arguments.
    """
    df = get_cimt_labels()
    return {
        "functional_systems": sorted(df["functional_system"].unique().tolist()),
        "sub_systems": sorted(df["sub_system"].unique().tolist()),
    }


# =============================================================================
# BACKWARD COMPATIBILITY WRAPPERS
# =============================================================================

def get_roi_index(roi_name: str) -> int:
    """
    Get the numeric index for a single CIMT ROI name.

    Legacy wrapper. Prefer resolve_roi_indices() for batch lookups.
    """
    name_map = _get_name_to_index_map()
    if roi_name not in name_map:
        raise ValueError(f"ROI '{roi_name}' not found in CIMT atlas.")
    return name_map[roi_name]


def get_motor_network_indices() -> np.ndarray:
    """
    Get integer indices for the Motor-Basal-Executive-STN network.

    Now uses native select_network() instead of delegating to lcmv_xtra.
    Returns np.ndarray of int64 (previously returned list[str] — breaking fix).
    """
    return select_network(
        functional_systems=['Motor', 'BasalGanglia', 'Frontoparietal'],
        sub_systems=['Primary', 'Premotor', 'Supplementary', 'Eye',
                     'DLPFC', 'IFJ', 'IFS', 'VLPFC', 'Subthalamic']
    )


def get_motor_network_metadata() -> pd.DataFrame:
    """
    Get detailed metadata for the Motor-Basal-Executive-STN network.

    Uses native select_network() for index selection, then joins with
    full atlas labels. No longer delegates to lcmv_xtra.
    """
    indices = get_motor_network_indices()
    df = get_cimt_labels()
    selected = df[df["index"].isin(indices)].copy()
    selected = selected.sort_values("roi_name").reset_index(drop=True)
    selected.insert(0, "new_index", range(len(selected)))
    return selected
