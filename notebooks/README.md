# TUTORIAL
### Advanced Analysis Workflows with LCMV Stats

This guide demonstrates how to leverage `lcmv_stats` for complex experimental designs beyond the standard "In-phase vs. Out-phase" comparison. 

The core philosophy of `lcmv_stats` is **modularity**:
1.  **Extraction** (`epoching.py`) converts raw time-courses into standardized epochs.
2.  **Feature Extraction** (`connectivity.py`) converts epochs into connectivity matrices.
3.  **Inference** (`statistics.py`) compares two sets of matrices, regardless of what they represent.



---

## Concept: The "Condition A vs. Condition B" Framework

`lcmv_stats` does not care if you are comparing "Pre vs. Post", "Rest vs. Move", or "Left vs. Right". It only requires two inputs:
*   **Array A**: Shape `(n_subjects, n_edges)` representing Condition 1.
*   **Array B**: Shape `(n_subjects, n_edges)` representing Condition 2.

Your job is simply to ensure that the data in Array A and Array B corresponds to the same subjects in the same order.

---

## Scenario 1: State-Based Comparison (Rest vs. Task)

**Use Case:** Comparing baseline resting state against an active task (e.g., "Rest" vs. "Spiral Drawing").
**Data Type:** Continuous recordings (no discrete trials).
**Key Function:** `extract_continuous_epochs`

### Why use this?
Continuous data cannot be directly compared using trial-locked statistics. We use a sliding window to break long recordings into independent "epochs" (e.g., 2-second chunks). This allows us to compute stable connectivity estimates for each state.

### Implementation

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

spiral_epochs = ls.extract_continuous_epochs(
    subject_id=SUBJECT_ID, 
    lcmv_root=LCMV_ROOT, 
    condition="spiral_task",    # Folder name in derivatives
    epoch_duration=2.0, 
    overlap=0.5
)

# 2. Compute Connectivity (WPLI)
sfreq = ls.get_subject_sfreq(SUBJECT_ID, LCMV_ROOT, "resting_state")

# Note: For continuous data, we often compare the state against itself 
# to get a stable estimate, then compare the resulting matrices later.
rest_conn, _ = ls.extract_wpli_features(rest_epochs, rest_epochs, band="beta", sfreq=sfreq)
spiral_conn, _ = ls.extract_wpli_features(spiral_epochs, spiral_epochs, band="beta", sfreq=sfreq)

# 3. Prepare for Group Stats
# Extract upper triangle to flatten the matrix into a vector of edges
triu_idx = np.triu_indices(rest_conn.shape[0], k=1)
rest_vector = rest_conn[triu_idx]
spiral_vector = spiral_conn[triu_idx]
```



---

## Complete Group-Level Workflow

Since `batch.py` is optimized for In/Out phase comparisons, here is a generic template to compare **any two conditions** across a group of subjects.

```python
import numpy as np
import pandas as pd
import lcmv_stats as ls
from pathlib import Path

# --- CONFIGURATION ---
LCMV_ROOT = Path("/path/to/derivatives/lcmv")
EVENTS_CSV = Path("/path/to/events.csv")
BAND = "low_beta"
CONDITION_A_COL = 'task_type' # Column in CSV to filter Condition A
CONDITION_B_COL = 'task_type' # Column in CSV to filter Condition B
VAL_A = 'spiral'
VAL_B = 'pingpong'

# --- LOAD DATA ---
events_df = pd.read_csv(EVENTS_CSV)
subject_ids = sorted([ls.utils.map_subject_to_subj(s) for s in events_df['subject'].unique()])

all_vec_a = []
all_vec_b = []
valid_subs = []

print(f"Processing {len(subject_ids)} subjects...")

for sid in subject_ids:
    try:
        # 1. Filter Events for this Subject
        subj_events = events_df[events_df['subject'].apply(lambda x: ls.utils.map_subject_to_subj(x) == sid)]
        
        ev_a = subj_events[subj_events[CONDITION_A_COL] == VAL_A]
        ev_b = subj_events[subj_events[CONDITION_B_COL] == VAL_B]
        
        if ev_a.empty or ev_b.empty:
            continue
            
        # 2. Extract Epochs
        in_a, out_a = ls.extract_event_epochs(sid, LCMV_ROOT, ev_a)
        in_b, out_b = ls.extract_event_epochs(sid, LCMV_ROOT, ev_b)
        
        if in_a.size == 0 or in_b.size == 0:
            continue
            
        # 3. Compute Connectivity
        sfreq = ls.get_subject_sfreq(sid, LCMV_ROOT)
        conn_a, _ = ls.extract_wpli_features(in_a, out_a, BAND, sfreq)
        conn_b, _ = ls.extract_wpli_features(in_b, out_b, BAND, sfreq)
        
        if conn_a is None or conn_b is None:
            continue
            
        # 4. Vectorize (Upper Triangle)
        triu_idx = np.triu_indices(conn_a.shape[0], k=1)
        all_vec_a.append(conn_a[triu_idx])
        all_vec_b.append(conn_b[triu_idx])
        valid_subs.append(sid)
        
    except Exception as e:
        print(f"Warning: Failed for {sid}: {e}")
        continue

# --- RUN STATISTICS ---
if len(valid_subs) >= 3:
    data_a = np.stack(all_vec_a) # Shape: (n_subs, n_edges)
    data_b = np.stack(all_vec_b) # Shape: (n_subs, n_edges)
    
    # Perform Permutation Test
    df_results = ls.run_edgewise_permutation(data_a, data_b, n_permutations=1000)
    
    print(f"✅ Analysis Complete. Found {len(df_results)} edges.")
    print(df_results.head())
    
    # Save Report
    ls.generate_markdown_report(df_results.assign(band=BAND), "results_comparison.md")
else:
    print("❌ Not enough valid subjects for statistics.")
```

---

## Advanced: Directionality with GPDC

If you find significant differences in WPLI (undirected), you can use `extract_gpdc_features` to determine the **direction** of information flow for those specific edges.

```python
# Assuming 'df_sig' contains significant edges from the permutation test above
# And 'in_a', 'out_a' are the epochs for the significant condition

sig_edges = df_sig[df_sig['p_val'] < 0.05]

if not sig_edges.empty:
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

1.  **Batch Helper**: The built-in `prepare_group_connectivity` in `batch.py` is hardcoded for In/Out phase comparisons. For the scenarios above, use the **Complete Group-Level Workflow** loop provided.
2.  **Continuous Data Quality**: When using `extract_continuous_epochs`, ensure your continuous data is clean (artifact-free). Sliding windows will include artifacts if they are present in the raw time course.
3.  **Edge Matching**: When comparing Condition A vs. Condition B, ensure that the ROI indices are identical. Since `lcmv_stats` uses the fixed CIMT atlas, this is handled automatically, but be careful if you subset ROIs manually.
