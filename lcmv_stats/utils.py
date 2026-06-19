# lcmv_stats/utils.py

import json
import re
from pathlib import Path

def map_subject_to_subj(subject_name: str) -> str:
    """Convert Sbj001 to sub-001 format."""
    if subject_name.startswith('Sbj'):
        num = subject_name.replace("Sbj", "").lstrip("0") or "0"
    else:
        numbers = re.findall(r'\d+', subject_name)
        num = numbers[0] if numbers else "0"
    return f"sub-{int(num):03d}"

def get_subject_sfreq(subject_id: str, lcmv_root: Path, condition: str = "bima_off") -> float:
    """Retrieve sampling frequency from CIMT pipeline metadata."""
    metadata_file = lcmv_root / f"{subject_id}_{condition}" / "pipeline_metadata.json"
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_file}")
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    return float(metadata['sfreq_hz'])
