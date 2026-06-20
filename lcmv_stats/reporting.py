# lcmv_stats/reporting.py

"""
Reporting tools for saving and visualizing statistical results.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional
from scipy.ndimage import gaussian_filter
import logging  # After existing imports

logger = logging.getLogger(__name__)  # After COLOR constants

COLOR_INPHASE = '#2E8B57'
COLOR_OUTPHASE = '#DC143C'

def plot_top_edges(
    df_sig: pd.DataFrame, 
    band: str, 
    n_top: int = 5, 
    save_path: Optional[str] = None
):
    """
    Create enhanced visualization of top N significant edges.
    Expects columns: roi1_full, roi2_full, mean_inphase, mean_outphase, 
                     sd_inphase, sd_outphase, n_subjects, cohens_d, p_val, mean_diff
    """
    df_top = df_sig.nlargest(n_top, 'abs_d').copy()
    if df_top.empty: return

    n_edges = len(df_top)
    fig, axes = plt.subplots(1, n_edges, figsize=(6 * n_edges, 5))
    if n_edges == 1: axes = [axes]
    
    for idx, (_, row) in enumerate(df_top.iterrows()):
        ax = axes[idx]
        phases = ['In-Phase', 'Out-Phase']
        means = [row['mean_inphase'], row['mean_outphase']]
        sems = [row['sd_inphase']/np.sqrt(row['n_subjects']), 
                row['sd_outphase']/np.sqrt(row['n_subjects'])]
        
        ax.bar(phases, means, color=[COLOR_INPHASE, COLOR_OUTPHASE], alpha=0.85)
        ax.errorbar(range(2), means, yerr=sems, fmt='none', ecolor='black', capsize=5)
        
        stats_text = f"d = {row['cohens_d']:.2f}\np = {row['p_val']:.4f}"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, ha='right', va='top')
        
        r1 = row.get('roi1_full', row['roi1']).replace('_', ' ').title()
        r2 = row.get('roi2_full', row['roi2']).replace('_', ' ').title()
        ax.set_title(f"{r1} ↔ {r2}", fontsize=10)
        ax.set_ylabel('WPLI Connectivity')

    fig.suptitle(f'Top {n_top} CIMT Connections: {band.replace("_", " ").title()} Band', fontweight='bold')
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def create_directed_effect_map(
    df_sig: pd.DataFrame,
    in_gpdc: np.ndarray,
    out_gpdc: np.ndarray,
    roi_names: List[str],
    alpha: float = 0.05,
    condition_label: str = "In-Phase"
) -> pd.DataFrame:
    """
    Combines undirected significance (WPLI p-val) with directed flow (GPDC).
    
    Args:
        df_sig: DataFrame from run_edgewise_permutation.
        in_gpdc: GPDC matrix for condition A (n_rois x n_rois).
        out_gpdc: GPDC matrix for condition B (n_rois x n_rois).
        roi_names: List of ROI names corresponding to GPDC matrix rows/cols.
        alpha: Significance threshold for filtering edges.
        condition_label: Label for the GPDC condition being reported.
        
    Returns:
        DataFrame with columns: edge, cohens_d, p_val, dominant_direction, 
                               directional_strength, wpli_mean_diff, gpdc_condition
    """
    sig_edges = df_sig[df_sig['p_val'] < alpha].copy()
    if sig_edges.empty or in_gpdc is None:
        return pd.DataFrame()
    
    # Create ROI name to index mapping
    roi_to_idx = {name: idx for idx, name in enumerate(roi_names)}
    
    directed_results = []
    for _, row in sig_edges.iterrows():
        r1, r2 = row['roi1'], row['roi2']
        
        # Skip if ROIs not in GPDC subset
        if r1 not in roi_to_idx or r2 not in roi_to_idx:
            logger.debug(f"Skipping edge {r1}-{r2}: ROI not in GPDC subset")
            continue
            
        idx1, idx2 = roi_to_idx[r1], roi_to_idx[r2]
        
        # Use in_gpdc as primary; fall back to out_gpdc if specified
        gpdc_matrix = in_gpdc if condition_label == "In-Phase" else out_gpdc
        
        flow_1_to_2 = gpdc_matrix[idx1, idx2]
        flow_2_to_1 = gpdc_matrix[idx2, idx1]
        
        # Determine dominant direction
        if flow_1_to_2 > flow_2_to_1:
            direction = f"{r1} → {r2}"
            strength = flow_1_to_2 - flow_2_to_1
        else:
            direction = f"{r2} → {r1}"
            strength = flow_2_to_1 - flow_1_to_2
            
        directed_results.append({
            'edge': f"{r1}-{r2}",
            'cohens_d': row['cohens_d'],
            'p_val': row['p_val'],
            'dominant_direction': direction,
            'directional_strength': strength,
            'wpli_mean_diff': row['mean_diff'],
            'gpdc_condition': condition_label
        })
    
    if not directed_results:
        return pd.DataFrame()
        
    return pd.DataFrame(directed_results).sort_values('directional_strength', ascending=False)

def save_spectral_results(
    output_dir: Path,
    subject_id: str,
    band: str,
    f: np.ndarray,
    t: np.ndarray,
    avg_sxx: np.ndarray,
    clusters: List,
    cluster_pvs: np.ndarray,
    config_sigma: tuple = (1.0, 2.0)
):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(
        output_dir / f"{subject_id}_{band}_spectral_stats.npz",
        f=f, t=t, avg_sxx=avg_sxx,
        clusters=clusters, cluster_pvs=cluster_pvs
    )
    
    avg_sxx_plot = gaussian_filter(avg_sxx, sigma=config_sigma)
    fig, ax = plt.subplots(figsize=(14, 8))
    mesh = ax.pcolormesh(t, f, avg_sxx_plot, shading='gouraud', cmap='RdBu_r')
    
    if clusters:
        sig_mask = np.zeros_like(avg_sxx, dtype=bool)
        for c_idx, cluster in enumerate(clusters):
            if cluster_pvs[c_idx] < 0.05:
                sig_mask[cluster] = True
        ax.contour(t, f, sig_mask.astype(int), levels=[0.5], colors='white', linewidths=1.5)
        
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time relative to event (s)")
    ax.set_title(f"Spectral Clusters: {band.replace('_', ' ').title()}")
    ax.axvline(x=0, color="black", linestyle="--", linewidth=2)
    fig.colorbar(mesh, ax=ax, label="Mean Z-score")
    
    plot_path = output_dir / f"{subject_id}_{band}_spectral_clusters.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    return plot_path

def generate_markdown_report(df_all: pd.DataFrame, output_path: Path):
    lines = ["# CIMT Edge-Wise Permutation Test Results", ""]
    bands = df_all['band'].unique() if 'band' in df_all.columns else ['Unknown']
    
    for band in bands:
        df_band = df_all[df_all['band'] == band] if 'band' in df_all.columns else df_all
        lines.append(f"## {band.replace('_', ' ').title()} Band")
        lines.append("| Region 1 | Region 2 | Cohen's d | p-value |")
        lines.append("| :--- | :--- | :---: | :---: |")
        
        for _, row in df_band.iterrows():
            r1 = row.get('roi1_full', row.get('roi1', 'N/A'))
            r2 = row.get('roi2_full', row.get('roi2', 'N/A'))
            lines.append(f"| {r1} | {r2} | {row['cohens_d']:.2f} | {row['p_val']:.4f} |")
        lines.append("")
        
    with open(output_path, 'w') as f:
        f.write("\n".join(lines))
