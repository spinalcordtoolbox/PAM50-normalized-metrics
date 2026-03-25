#!/usr/bin/env python3
"""
Generate an overview datasets.tsv in the repository root summarizing all datasets
by reading individual participants.tsv files. Also updates the Datasets Overview
table in README.md.

Auto-computed columns (from participants.tsv):
    num_subjects, num_sites, sex_M, sex_F, sex_unknown,
    age_mean, age_std, age_min, age_max

Manually maintained columns (edit DATASET_METADATA below when adding new datasets):
    coverage  -- anatomical region covered (e.g., "cervical spine", "whole spine")
    contrast  -- MRI contrast (e.g., "T2w")
    resolution -- nominal voxel size (e.g., "0.8mm iso")
    link      -- URL to the original raw dataset
    link_text -- display text for the link in README (e.g., "spine-generic/data-multi-subject")

Usage:
    python code/generate_datasets_summary.py

Output:
    datasets.tsv and README.md (Datasets Overview table) in the repository root
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Manually maintained metadata — update this dict when a new dataset is added.
# ---------------------------------------------------------------------------
DATASET_METADATA = {
    'spine-generic_multi-subject': {
        # Multi-site cervical spine dataset; 42 sites, various scanner vendors.
        # Raw data: https://github.com/spine-generic/data-multi-subject
        'coverage': 'cervical spine',
        'contrast': 'T2w',
        'resolution': '0.8mm iso',
        'link': 'https://github.com/spine-generic/data-multi-subject',
        'link_text': 'spine-generic/data-multi-subject',
    },
    'whole-spine': {
        # Two-site whole-spine dataset; Aix-Marseille University (AMU) + Neuroimaging Functional Unit (UNF), Polytechnique Montréal; Siemens scanners.
        # Raw data: https://openneuro.org/datasets/ds005616
        'coverage': 'whole spine',
        'contrast': 'T2w',
        'resolution': '1.0mm iso',
        'link': 'https://openneuro.org/datasets/ds005616',
        'link_text': 'openneuro/ds005616',
    },
}

MANUAL_COLUMNS = ['coverage', 'contrast', 'resolution', 'link', 'link_text']
PLACEHOLDER = 'n/a'

README_TABLE_START = '<!-- datasets-table-start -->'
README_TABLE_END = '<!-- datasets-table-end -->'
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


def build_readme_table(rows):
    """Render rows as a GitHub-flavoured Markdown table."""
    header = '| metric | dataset | num_subjects | num_sites | sex (M/F/unknown) | age (mean\u00b1SD [min\u2013max]) | coverage | contrast | resolution | link |'
    separator = '|--------|---------|-------------:|----------:|:-----------------:|:-----------------------:|----------|----------|------------|------|'

    lines = [header, separator]
    for r in rows:
        sex = f"{r['sex_M']}/{r['sex_F']}/{r['sex_unknown']}"
        if r['age_mean'] == PLACEHOLDER:
            age = PLACEHOLDER
        else:
            age = f"{r['age_mean']}\u00b1{r['age_std']} [{r['age_min']}\u2013{r['age_max']}]"
        link = f"[{r['link_text']}]({r['link']})" if r['link'] != PLACEHOLDER else PLACEHOLDER
        lines.append(
            f"| {r['metric']} | {r['dataset']} | {r['num_subjects']} | {r['num_sites']} "
            f"| {sex} | {age} | {r['coverage']} | {r['contrast']} | {r['resolution']} | {link} |"
        )
    return '\n'.join(lines)


def update_readme(readme_path, table_md):
    """Replace the content between the marker comments in README.md."""
    text = readme_path.read_text()
    new_block = f'{README_TABLE_START}\n{table_md}\n{README_TABLE_END}'
    pattern = re.escape(README_TABLE_START) + r'.*?' + re.escape(README_TABLE_END)
    updated, n = re.subn(pattern, new_block, text, flags=re.DOTALL)
    if n == 0:
        raise ValueError(f'Markers not found in {readme_path}. '
                         f'Add "{README_TABLE_START}" and "{README_TABLE_END}" around the table.')
    readme_path.write_text(updated)
    print(f'Updated README table in {readme_path}')


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

    # Write datasets.tsv (drop link_text — it's only for README rendering)
    out_df = pd.DataFrame(rows).drop(columns=['link_text'])
    out_path = repo_root / 'datasets.tsv'
    out_df.to_csv(out_path, sep='\t', index=False)
    print(f'Saved summary for {len(rows)} dataset(s) to {out_path}')

    # Update README.md
    table_md = build_readme_table(rows)
    update_readme(repo_root / 'README.md', table_md)

    print()
    print(out_df.to_string(index=False))


if __name__ == '__main__':
    main()
