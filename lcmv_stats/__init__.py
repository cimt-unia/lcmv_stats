# lcmv_stats/__init__.py
"""
lcmv_stats: Tensor-native statistical analysis for CIMT source-space EEG/MEG.

Operates exclusively on standardized .npz tensors from lcmv_xtra.assemble_tensor.
Z-scoring is applied to continuous data before epoching (in epoch_tensor).
All outputs are pure NumPy arrays — no pandas intermediates in computation paths.
"""

# ── Atlas & ROI Selection ──
from ._atlas import (
    get_cimt_labels,
    resolve_roi_indices,
    select_network,
    get_available_systems,
    # Backward compatibility wrappers
    get_roi_index,
    get_motor_network_indices,
    get_motor_network_metadata,
)

# ── Tensor I/O ──
from .utils import load_tensor

# ── Epoching (with pre-epoch Z-scoring) ──
from .epoching import epoch_tensor

# ── Connectivity & Statistics (unified module) ──
from .connectivity import (
    get_frequency_bands,
    compute_connectivity_matrix,
    extract_wpli_features,
    extract_gpdc_features,
    cohens_d_paired,
    prepare_connectivity_for_stats,
    run_edgewise_permutation,
    compare_tensors,
    save_comparison_results,
    plot_top_edges,
    create_directed_effect_map,
)

# ── Time-Frequency (tensor-native, epoch-averaged) ──
from .timefreq import (
    average_condition_epochs,
    compute_spectrogram_for_subject,
    compute_group_spectrograms_from_epochs,
    plot_and_test_group_spectrograms,
)

# ── Machine Learning Feature Engineering (tensor-native) ──
from .machine_learning import (
    extract_band_power_features,
    prepare_ml_dataset,
    flatten_for_sklearn,
)

# ── Visualization  ──
from .visualization import (
    plot_connectivity_matrix,
    plot_psd_rois,
    plot_psd_comparison,
    plot_spectrogram,
    plot_feature_distribution,
    validate_matrix_quality,
)

__version__ = "0.4.0"

__all__ = [
    
    # ── Atlas & ROI ──
    "get_cimt_labels",
    "resolve_roi_indices",
    "select_network",
    "get_available_systems",
    "get_roi_index",
    "get_motor_network_indices",
    "get_motor_network_metadata",
    
    # ── Tensor I/O ──
    "load_tensor",
    
    # ── Epoching ──
    "epoch_tensor",
    
    # ── Connectivity & Statistics ──
    "get_frequency_bands",
    "compute_connectivity_matrix",
    "extract_wpli_features",
    "extract_gpdc_features",
    "cohens_d_paired",
    "prepare_connectivity_for_stats",
    "run_edgewise_permutation",
    "compare_tensors",
    "save_comparison_results",
    "plot_top_edges",
    "create_directed_effect_map",
    
    # ── Time-Frequency ──
    "average_condition_epochs",
    "compute_spectrogram_for_subject",
    "compute_group_spectrograms_from_epochs",
    "plot_and_test_group_spectrograms",
    
    # ── Machine Learning ──
    "extract_band_power_features",
    "prepare_ml_dataset",
    "flatten_for_sklearn",
    
    # ── Visualization ──
    "plot_connectivity_matrix",
    "plot_psd_rois",
    "plot_psd_comparison",
    "plot_spectrogram",
    "plot_feature_distribution",
    "validate_matrix_quality",
]
