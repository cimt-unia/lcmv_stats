# lcmv_stats/__init__.py
"""
lcmv_stats: Tensor-native statistical analysis for LCMV source-space EEG/MEG.

Operates exclusively on standardized .npz tensors from lcmv_xtra.assemble_tensor.
Z-scoring is applied to continuous data before epoching (in epoch_tensor).

All connectivity results use pure NumPy arrays — no pandas intermediates.
"""

# Atlas & ROI utilities
from ._atlas import (
    get_cimt_labels,
    get_motor_network_indices,
    get_roi_index,
    get_motor_network_metadata,
)
from .utils import load_tensor, get_roi_indices

# Epoching (with pre-epoch Z-scoring)
from .epoching import epoch_tensor

# Connectivity, Statistics, Batch, Reporting (unified module)
from .connectivity import (
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

# Time-Frequency (tensor-native)
from .timefreq import (
    compute_roi_spectrogram,
    plot_and_test_group_spectrograms,
)

# Visualization (inspection tools)
from .visualization import (
    plot_connectivity_matrix,
    validate_matrix_quality,
    plot_psd_rois,
    plot_psd_comparison,
    plot_spectrogram,
    plot_feature_distribution,
)

# Machine Learning Feature Engineering
from .machine_learning import (
    get_frequency_bands,
    zscore_normalization,
    create_epochs,
    compute_epoch_features,
    process_signal_to_ml_dataframe,
)

__version__ = "0.3.0"

__all__ = [
    # ── Atlas & ROI ──
    "get_cimt_labels",
    "get_motor_network_indices",
    "get_roi_index",
    "get_motor_network_metadata",
    "get_roi_indices",
    # ── Tensor I/O ──
    "load_tensor",
    # ── Epoching ──
    "epoch_tensor",
    # ── Connectivity & Statistics ──
    "extract_wpli_features",
    "extract_gpdc_features",
    "cohens_d_paired",
    "prepare_connectivity_for_stats",
    "run_edgewise_permutation",
    "compare_tensors",
    "save_comparison_results",
    # ── Reporting ──
    "plot_top_edges",
    "create_directed_effect_map",
    # ── Time-Frequency ──
    "compute_roi_spectrogram",
    "plot_and_test_group_spectrograms",
    # ── Visualization ──
    "plot_connectivity_matrix",
    "validate_matrix_quality",
    "plot_psd_rois",
    "plot_psd_comparison",
    "plot_spectrogram",
    "plot_feature_distribution",
    # ── Machine Learning ──
    "get_frequency_bands",
    "zscore_normalization",
    "create_epochs",
    "compute_epoch_features",
    "process_signal_to_ml_dataframe",
]
