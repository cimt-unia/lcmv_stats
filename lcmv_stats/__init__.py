# lcmv_stats/__init__.py
"""
lcmv_stats: Statistical analysis and machine learning tools for LCMV source-reconstructed EEG/MEG data.
Optimized for the CIMT Unified Atlas (448 ROIs) but flexible for other structures.
"""
from ._atlas import get_cimt_labels, get_motor_network_indices, get_roi_index
from .epoching import extract_event_epochs, extract_continuous_epochs
from .utils import get_subject_sfreq, map_subject_to_subj, get_roi_time_series
from .connectivity import extract_wpli_features, extract_gpdc_features
from .statistics import run_edgewise_permutation, cohens_d_paired
from .visualization import plot_connectivity_matrix, validate_matrix_quality, plot_psd_rois, plot_psd_comparison, plot_spectrogram, plot_feature_distribution
from .reporting import plot_top_edges, generate_markdown_report, save_spectral_results, create_directed_effect_map
from .machine_learning import (
    get_frequency_bands, 
    zscore_normalization, 
    create_epochs, 
    compute_epoch_features, 
    process_signal_to_ml_dataframe
)
from .timefreq import (
    prepare_roi_signal_from_2d,
    compute_zscored_spectrogram,
    plot_and_test_group_spectrograms
)

from ._atlas import get_cimt_labels, get_motor_network_indices
from .batch import prepare_group_comparison, prepare_connectivity_for_stats
from . import batch


__version__ = "0.1.0"

__all__ = [
    # Atlas
    "get_cimt_labels",
    "get_motor_network_indices",
    "get_roi_index",  
    
    # Epoching
    "extract_event_epochs",
    "extract_continuous_epochs",
    
    # Utils
    "get_subject_sfreq",
    "map_subject_to_subj",
    "get_roi_time_series",
    
    # Connectivity
    "extract_wpli_features",
    "extract_gpdc_features",
    
    # Statistics
    "run_edgewise_permutation",
    "cohens_d_paired",
    
    # Time-Frequency
    "prepare_roi_signal_from_2d",
    "compute_zscored_spectrogram",
    "plot_and_test_group_spectrograms",
    
    # Visualization
    "plot_connectivity_matrix",
    "validate_matrix_quality",
    "plot_psd_rois",
    "plot_psd_comparison",
    "plot_spectrogram",
    "plot_feature_distribution",
    
    # Reporting
    "plot_top_edges",
    "generate_markdown_report",
    "save_spectral_results",
    "create_directed_effect_map",
    
    # Machine Learning / Feature Engineering
    "get_frequency_bands",
    "zscore_normalization",
    "create_epochs",
    "compute_epoch_features",
    "process_signal_to_ml_dataframe",
    
    # Atlas & Batch
    "get_cimt_labels",
    "get_motor_network_indices",
    "prepare_group_comparison",
    "prepare_connectivity_for_stats",
    "batch",
]
