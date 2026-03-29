#!/usr/bin/env python3
"""
Generate an overview datasets.tsv in the repository root summarizing all datasets
by reading individual participants.tsv files. Also updates the Datasets Overview
table in README.md.

Auto-computed columns (from participants.tsv):
    num_subjects, num_sessions, num_sites, sex_M, sex_F, sex_unknown,
    age_mean, age_std, age_min, age_max

Manually maintained columns (from dataset_description.json in each dataset folder):
    coverage   -- anatomical region covered (e.g., "cervical spine", "whole spine")
    contrast   -- MRI contrast (e.g., "T2w")
    resolution -- nominal voxel size (e.g., "0.8mm iso")
    population -- subject population (e.g., "healthy adults", "healthy adults and children")
    link       -- URL to the original raw dataset
    link_text  -- display text for the link in README (e.g., "spine-generic/data-multi-subject")

Usage:
    python code/generate_datasets_summary.py

Output:
    datasets.tsv and README.md (Datasets Overview table) in the repository root
"""

import json
import re
import pandas as pd
import numpy as np
from pathlib import Path


MANUAL_COLUMNS = ['name', 'order', 'coverage', 'contrast', 'resolution', 'population', 'link', 'link_text']
PLACEHOLDER = 'n/a'

README_TABLE_START = '<!-- datasets-table-start -->'
README_TABLE_END = '<!-- datasets-table-end -->'


def compute_stats(df):
    """Compute summary statistics from a participants DataFrame."""
    n_sessions = len(df)

    # Replace 'n/a' strings with NaN for consistent handling
    df = df.replace('n/a', np.nan)

    # Deduplicate by participant_id for per-subject stats (sex, age, subject count)
    if 'participant_id' in df.columns:
        df_subjects = df.drop_duplicates(subset='participant_id')
    else:
        df_subjects = df
    n_subjects = len(df_subjects)

    # Number of unique acquisition sites
    if 'institution' in df.columns:
        n_sites = int(df['institution'].dropna().nunique())
    else:
        n_sites = PLACEHOLDER

    # Sex counts (per unique subject)
    sex_counts = df_subjects['sex'].value_counts()
    n_m = int(sex_counts.get('M', 0))
    n_f = int(sex_counts.get('F', 0))
    n_unknown = int(df_subjects['sex'].isna().sum())

    # Age stats (per unique subject)
    age = pd.to_numeric(df_subjects['age'], errors='coerce')
    age_valid = age.dropna()
    if len(age_valid) > 0:
        age_mean = round(age_valid.mean(), 1)
        age_std = round(age_valid.std(), 1)
        age_min = int(age_valid.min())
        age_max = int(age_valid.max())
    else:
        age_mean = age_std = age_min = age_max = PLACEHOLDER

    return {
        'num_subjects': n_subjects,
        'num_sessions': n_sessions,
        'num_sites': n_sites,
        'sex_M': n_m,
        'sex_F': n_f,
        'sex_unknown': n_unknown,
        'age_mean': age_mean,
        'age_std': age_std,
        'age_min': age_min,
        'age_max': age_max,
    }


def load_description(dataset_dir):
    """Load dataset_description.json from a dataset directory, or return placeholders."""
    json_path = dataset_dir / 'dataset_description.json'
    if json_path.exists():
        with open(json_path) as f:
            meta = json.load(f)
    else:
        print(f'  WARNING: {json_path} not found — using placeholders.')
        meta = {}
    result = {col: meta.get(col, PLACEHOLDER) for col in MANUAL_COLUMNS}
    if 'links' in meta:
        result['links'] = meta['links']
        # Store all URLs joined by ', ' in the 'link' column of datasets.tsv
        result['link'] = ', '.join(e['link'] for e in meta['links'])
        result['link_text'] = ', '.join(e['link_text'] for e in meta['links'])
    return result


def build_readme_table(rows):
    """Render rows as a GitHub-flavoured Markdown table."""
    header = '| metric | name | num_subjects | num_sessions | num_sites | population | sex (M/F/unknown) | age (mean\u00b1SD [min\u2013max]) | coverage | contrast | resolution | link |'
    separator = '|--------|------|-------------:|-------------:|----------:|------------|:-----------------:|:-----------------------:|----------|----------|------------|------|'

    lines = [header, separator]
    for r in rows:
        sex = f"{r['sex_M']}/{r['sex_F']}/{r['sex_unknown']}"
        if r['age_mean'] == PLACEHOLDER:
            age = PLACEHOLDER
        else:
            age = f"{r['age_mean']}\u00b1{r['age_std']} [{r['age_min']}\u2013{r['age_max']}]"
        # Support multiple links via optional 'links' list of {link, link_text} dicts
        if r.get('links'):
            link = ', '.join(f"[{e['link_text']}]({e['link']})" for e in r['links'])
        elif r['link'] != PLACEHOLDER:
            link = f"[{r['link_text']}]({r['link']})"
        else:
            link = PLACEHOLDER
        lines.append(
            f"| {r['metric']} | {r['name']} | {r['num_subjects']} | {r['num_sessions']} | {r['num_sites']} "
            f"| {r['population']} | {sex} | {age} | {r['coverage']} | {r['contrast']} "
            f"| {r['resolution']} | {link} |"
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
        dataset_dir = tsv_path.parent
        rel_parts = dataset_dir.relative_to(repo_root).parts
        # Expected structure: <metric>/<dataset>/participants.tsv
        metric = rel_parts[0] if len(rel_parts) >= 1 else 'unknown'
        dataset = rel_parts[1] if len(rel_parts) >= 2 else 'unknown'

        df = pd.read_csv(tsv_path, sep='\t', dtype=str)
        stats = compute_stats(df)
        manual = load_description(dataset_dir)

        rows.append({'metric': metric, **stats, **manual})

    rows.sort(key=lambda r: r.get('order', 999))

    # Write datasets.tsv (drop link_text — it's only for README rendering)
    # Column order: metric, name, then computed stats, then remaining manual fields
    tsv_columns = ['metric', 'name', 'num_subjects', 'num_sessions', 'num_sites', 'population', 'sex_M', 'sex_F',
                   'sex_unknown', 'age_mean', 'age_std', 'age_min', 'age_max', 'coverage', 'contrast',
                   'resolution', 'link']
    out_df = pd.DataFrame(rows)[tsv_columns]
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
