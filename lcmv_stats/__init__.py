# lcmv_stats/__init__.py
"""
lcmv_stats: Statistical analysis tools for LCMV source-reconstructed EEG/MEG data.
Optimized for the CIMT Unified Atlas (448 ROIs) but flexible for other structures.
"""

from .epoching import extract_event_epochs, extract_continuous_epochs
from .utils import get_subject_sfreq, map_subject_to_subj
from .connectivity import extract_wpli_features, extract_gpdc_features
from .statistics import run_edgewise_permutation, cohens_d_paired
from .timefreq import compute_zscored_spectrogram, run_cluster_spectrogram
from .visualization import plot_connectivity_matrix, validate_matrix_quality, plot_psd_rois
from .reporting import plot_top_edges, generate_markdown_report, save_spectral_results, create_directed_effect_map

from ._atlas import get_cimt_labels, get_motor_network_indices
from .batch import prepare_group_comparison, prepare_connectivity_for_stats
from . import batch



__version__ = "0.1.0"

__all__ = [
    "extract_event_epochs",
    "extract_continuous_epochs",
    "get_subject_sfreq",
    "map_subject_to_subj",
    "extract_wpli_features",
    "extract_gpdc_features",
    "run_edgewise_permutation",
    "cohens_d_paired",
    "compute_zscored_spectrogram",
    "run_cluster_spectrogram",
    "plot_connectivity_matrix",
    "validate_matrix_quality",
    "plot_psd_rois",
    "plot_top_edges",
    "generate_markdown_report",
    "save_spectral_results",
    "create_directed_effect_map",
    "get_cimt_labels",
    "get_motor_network_indices",
    "prepare_group_comparison",
    "prepare_connectivity_for_stats",
    "batch",
]
