# lcmv_stats/__init__.py
"""
lcmv_stats: Tensor-native statistical analysis for LCMV source-space EEG/MEG.
Operates on standardized .npz tensors from lcmv_xtra.
Z-scoring is applied to continuous data before epoching.
"""

# Atlas & ROI utilities
from ._atlas import get_cimt_labels, get_motor_network_indices, get_roi_index, get_motor_network_metadata
from .utils import load_tensor, get_roi_indices

# Epoching (with pre-epoch Z-scoring)
from .epoching import epoch_tensor

# Connectivity
from .connectivity import extract_wpli_features, extract_gpdc_features

# Batch & Statistics
from .batch import compare_tensors
from .statistics import (
    run_edgewise_permutation, 
    cohens_d_paired, 
    prepare_connectivity_for_stats
)

__version__ = "0.2.0"

__all__ = [
    # Atlas
    "get_cimt_labels", "get_motor_network_indices", "get_roi_index",
    "get_motor_network_metadata", "get_roi_indices",
    # Tensor I/O
    "load_tensor",
    # Epoching
    "epoch_tensor",
    # Connectivity
    "extract_wpli_features", "extract_gpdc_features",
    # Batch
    "compare_tensors",
    # Statistics
    "run_edgewise_permutation", "cohens_d_paired", "prepare_connectivity_for_stats",
]
