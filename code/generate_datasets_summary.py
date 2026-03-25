#!/usr/bin/env python3
"""
Generate an overview datasets.tsv in the repository root summarizing all datasets
by reading individual participants.tsv files.

Auto-computed columns (from participants.tsv):
    num_subjects, num_sites, sex_M, sex_F, sex_unknown,
    age_mean, age_std, age_min, age_max

Manually maintained columns (edit DATASET_METADATA below when adding new datasets):
    coverage  -- anatomical region covered (e.g., "cervical spine", "whole spine")
    contrast  -- MRI contrast (e.g., "T2w")
    resolution -- nominal voxel size (e.g., "0.8mm iso")
    release   -- dataset version/release tag (e.g., "r20250314", "1.1.2")
    link      -- URL to the original raw dataset

Usage:
    python code/generate_datasets_summary.py

Output:
    datasets.tsv in the repository root
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Manually maintained metadata — update this dict when a new dataset is added.
# ---------------------------------------------------------------------------
DATASET_METADATA = {
    'spine-generic_multi-subject': {
        # Multi-site cervical spine dataset; 42 sites, various scanner vendors.
        # Raw data: https://github.com/spine-generic/data-multi-subject/releases/tag/r20250314
        'coverage': 'cervical spine',
        'contrast': 'T2w',
        'resolution': '0.8mm iso',
        'release': 'r20250314',
        'link': 'https://github.com/spine-generic/data-multi-subject',
    },
    'whole-spine': {
        # Two-site whole-spine dataset; Aix-Marseille University (AMU) + Neuroimaging Functional Unit (UNF), Polytechnique Montréal; Siemens scanners.
        # Raw data: https://openneuro.org/datasets/ds005616
        'coverage': 'whole spine',
        'contrast': 'T2w',
        'resolution': '1.0mm iso',
        'release': '1.1.2',
        'link': 'https://openneuro.org/datasets/ds005616',
    },
}

MANUAL_COLUMNS = ['coverage', 'contrast', 'resolution', 'release', 'link']
PLACEHOLDER = 'n/a'
# ---------------------------------------------------------------------------


def compute_stats(df):
    """Compute summary statistics from a participants DataFrame."""
    n = len(df)

    # Replace 'n/a' strings with NaN for consistent handling
    df = df.replace('n/a', np.nan)

    # Number of unique acquisition sites
    if 'institution' in df.columns:
        n_sites = int(df['institution'].dropna().nunique())
    else:
        n_sites = PLACEHOLDER

    # Sex counts
    sex_counts = df['sex'].value_counts()
    n_m = int(sex_counts.get('M', 0))
    n_f = int(sex_counts.get('F', 0))
    n_unknown = int(df['sex'].isna().sum())

    # Age stats
    age = pd.to_numeric(df['age'], errors='coerce')
    age_valid = age.dropna()
    if len(age_valid) > 0:
        age_mean = round(age_valid.mean(), 1)
        age_std = round(age_valid.std(), 1)
        age_min = int(age_valid.min())
        age_max = int(age_valid.max())
    else:
        age_mean = age_std = age_min = age_max = PLACEHOLDER

    return {
        'num_subjects': n,
        'num_sites': n_sites,
        'sex_M': n_m,
        'sex_F': n_f,
        'sex_unknown': n_unknown,
        'age_mean': age_mean,
        'age_std': age_std,
        'age_min': age_min,
        'age_max': age_max,
    }


def main():
    repo_root = Path(__file__).resolve().parent.parent

    rows = []
    for tsv_path in sorted(repo_root.rglob('participants.tsv')):
        rel_parts = tsv_path.parent.relative_to(repo_root).parts
        # Expected structure: <metric>/<dataset>/participants.tsv
        metric = rel_parts[0] if len(rel_parts) >= 1 else 'unknown'
        dataset = rel_parts[1] if len(rel_parts) >= 2 else 'unknown'

        df = pd.read_csv(tsv_path, sep='\t', dtype=str)
        stats = compute_stats(df)

        meta = DATASET_METADATA.get(dataset, {})
        manual = {col: meta.get(col, PLACEHOLDER) for col in MANUAL_COLUMNS}

        rows.append({'metric': metric, 'dataset': dataset, **stats, **manual})

    out_df = pd.DataFrame(rows)
    out_path = repo_root / 'datasets.tsv'
    out_df.to_csv(out_path, sep='\t', index=False)

    print(f"Saved summary for {len(rows)} dataset(s) to {out_path}\n")
    print(out_df.to_string(index=False))


if __name__ == '__main__':
    main()
