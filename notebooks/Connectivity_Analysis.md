# TUTORIAL
### Advanced Analysis Workflows with LCMV Stats

This guide demonstrates how to leverage `lcmv_stats` for complex experimental designs beyond the standard "In-phase vs. Out-phase" comparison. 

The core philosophy of `lcmv_stats` is **modularity**:
1.  **Extraction** (`epoching.py`) converts raw time-courses into standardized epochs.
2.  **Feature Extraction** (`connectivity.py`) converts epochs into connectivity matrices.
3.  **Inference** (`statistics.py`) compares two sets of matrices, regardless of what they represent.

<br>

## Concept: The "Condition A vs. Condition B" Framework

`lcmv_stats` does not care if you are comparing "Pre vs. Post", "Rest vs. Move", or "Left vs. Right". It only requires two inputs:
*   **Array A**: Shape `(n_subjects, n_edges)` representing Condition 1.
*   **Array B**: Shape `(n_subjects, n_edges)` representing Condition 2.

Your job is simply to ensure that the data in Array A and Array B corresponds to the same subjects in the same order.

<br>

## Example: State-Based Comparison (Rest vs. Task)

**Use Case:** Comparing baseline resting state against an active task (e.g., "Rest" vs. "Spiral Drawing").
**Data Type:** Continuous recordings (no discrete trials).
**Key Function:** `extract_continuous_epochs`

### Why use this?
Continuous data cannot be directly compared using trial-locked statistics. We use a sliding window to break long recordings into independent "epochs" (e.g., 2-second chunks). This allows us to compute stable connectivity estimates for each state.

### Implementation (Single Subject)

```python
import lcmv_stats as ls
import numpy as np
from pathlib import Path

LCMV_ROOT = Path("/path/to/derivatives/lcmv")
SUBJECT_ID = "sub-001"

# 1. Extract Continuous Epochs
# We use a 2-second window with 50% overlap to increase statistical power.
rest_epochs = ls.extract_continuous_epochs(
    subject_id=SUBJECT_ID, 
    lcmv_root=LCMV_ROOT, 
    condition="resting_state",  # Folder name in derivatives
    epoch_duration=2.0, 
    overlap=0.5
)

task_epochs = ls.extract_continuous_epochs(
    subject_id=SUBJECT_ID, 
    lcmv_root=LCMV_ROOT, 
    condition="task",           # Folder name in derivatives
    epoch_duration=2.0, 
    overlap=0.5
)

# 2. Compute Connectivity (WPLI)
sfreq = ls.get_subject_sfreq(SUBJECT_ID, LCMV_ROOT, "resting_state")

# Note: For continuous data, we compare the state against itself 
# to get a stable estimate for that specific block of time.
rest_conn, _ = ls.extract_wpli_features(rest_epochs, rest_epochs, band="beta", sfreq=sfreq)
task_conn, _ = ls.extract_wpli_features(task_epochs, task_epochs, band="beta", sfreq=sfreq)

# 3. Prepare for Group Stats
# Use the universal helper to vectorize the matrix
rest_vector = ls.prepare_connectivity_for_stats([rest_conn])
task_vector = ls.prepare_connectivity_for_stats([task_conn])
```

<br>

## Complete Group-Level Workflow (Simplified)

Instead of writing a complex loop, you can now use the `prepare_group_comparison` helper. This function automatically handles subject iteration, event filtering, epoch extraction, and connectivity calculation for **any** two conditions.

### Option 1: Comparing Two Conditions (e.g., Rest vs. Task)

```python
import numpy as np
import pandas as pd
import lcmv_stats as ls
from pathlib import Path

# --- CONFIGURATION ---
LCMV_ROOT = Path("/path/to/derivatives/lcmv")
EVENTS_CSV = Path("/path/to/events.csv")
BAND = "low_beta"

# --- LOAD DATA & PROCESS ---
events_df = pd.read_csv(EVENTS_CSV)

# One function call replaces the entire subject loop!
group_data = ls.prepare_group_comparison(
    events_df=events_df,
    lcmv_root=LCMV_ROOT,
    band=BAND,
    condition_col='task_type', # Column in your CSV
    val_a='rest',              # Condition A name
    val_b='task',              # Condition B name
    is_phase_comparison=False  # Default: Compare two different sets of events
)

print(f"✅ Successfully processed {len(group_data['valid_subs'])} subjects.")

# --- RUN STATISTICS ---
if len(group_data['valid_subs']) >= 3:
    # Perform Permutation Test
    df_results = ls.run_edgewise_permutation(
        group_data['data_a'], 
        group_data['data_b'], 
        n_permutations=1000
    )
    
    print(f"Analysis Complete. Found {len(df_results)} edges.")
    
    # Save Report
    ls.generate_markdown_report(df_results.assign(band=BAND), "results_comparison.md")
else:
    print("❌ Not enough valid subjects for statistics.")
```

### Option 2: In-Phase vs. Out-Phase Comparison

If you want to compare synchronization within the same trials (In-phase vs. Out-phase), you can use the same function with a simple flag.

```python
# --- IN-PHASE VS OUT-PHASE WORKFLOW ---
group_data_phase = ls.prepare_group_comparison(
    events_df=events_df,
    lcmv_root=LCMV_ROOT,
    band=BAND,
    is_phase_comparison=True  # This tells the function to split trials internally
)

# Run stats on In vs Out
df_results_phase = ls.run_edgewise_permutation(
    group_data_phase['data_a'], # In-phase connectivity
    group_data_phase['data_b'], # Out-phase connectivity
    n_permutations=1000
)
```

<br>

## Advanced: Directionality with GPDC

If you find significant differences in WPLI (undirected), you can use `extract_gpdc_features` to determine the **direction** of information flow for those specific edges.

```python
# Assuming 'df_sig' contains significant edges from the permutation test above
# And 'in_a', 'out_a' are the epochs for the significant condition

sig_edges = df_sig[df_sig['p_val'] < 0.05]

if not sig_edges.empty:
    # You need the original epochs for this subject/condition
    in_gpdc, out_gpdc, roi_names = ls.extract_gpdc_features(
        epochs_in=in_a,
        epochs_out=out_a,
        sig_df=sig_edges,
        sfreq=sfreq,
        band_range=(13, 30) # Low Beta
    )
    
    print(f"GPDC computed for {len(roi_names)} ROIs involved in significant edges.")
    # in_gpdc shape: (n_rois, n_rois) representing directed influence
```

## Important Limitations

1.  **Continuous Data in Groups**: The `prepare_group_comparison` function is optimized for event-based data found in an `events.csv`. For large-scale group analysis of continuous data (Rest vs. Task), you may need to write a custom loop that uses `extract_continuous_epochs` and then passes the resulting list of matrices to `ls.prepare_connectivity_for_stats()`.
2.  **Edge Matching**: When comparing Condition A vs. Condition B, ensure that the ROI indices are identical. Since `lcmv_stats` uses the fixed CIMT atlas, this is handled automatically by `compute_cimt_motor_connectivity`.
