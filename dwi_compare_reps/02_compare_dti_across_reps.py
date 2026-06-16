#
# Compare DTI metrics (FA, MD, RD, AD) and the tensor-fit residual (RMS) computed
# from a single DWI repetition between:
#   - 1rep, first 31 volumes vs
#   - the full acquisition (2rep)
#
# Comparison is done for WM at C2 and C3 using the sct_extract_metric output
#
# For each metric the script reports, across all subject x level pairs:
#   - mean +/- SD for 1rep and 2rep
#   - signed, absolute, and relative (%) difference (1rep - 2rep)
#   - Pearson correlation
# and saves a scatter figure (1rep vs 2rep) with the identity line.
#
# Usage:
#     python dwi_compare_reps/02_compare_dti_across_reps.py \
#         -path-results /path/to/results/dwi_compare_reps
#
# Author: Jan Valosek
#

import os
import glob
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

METRICS = ['FA', 'MD', 'RD', 'AD', 'RMS']

# Diffusivity metrics are stored in mm²/s (~1e-3); scale them by 1000 for readability.
scaling_factor = {
    'FA': 1,
    'MD': 1000,
    'AD': 1000,
    'RD': 1000,
    'RMS': 1,
    }

METRIC_TO_AXIS = {
    'FA':  'Fractional Anisotropy [a.u.]',
    'MD':  'Mean Diffusivity [×10⁻³ mm²/s]',
    'RD':  'Radial Diffusivity [×10⁻³ mm²/s]',
    'AD':  'Axial Diffusivity [×10⁻³ mm²/s]',
    'RMS': 'Fit residual RMS [a.u.]',
}

LABELS_FONT_SIZE = 22
TICKS_FONT_SIZE = LABELS_FONT_SIZE - 2


def get_parser():
    parser = argparse.ArgumentParser(
        description='Compare DTI metrics (FA, MD, RD, AD, RMS) between 1 and 2 DWI '
                    'repetitions from sct_extract_metric CSV outputs.')
    parser.add_argument(
        '-path-results', required=True,
        help='Path to the dwi_compare_reps folder containing *_dwi_{metric}_{n}rep.csv files.')
    parser.add_argument(
        '-path-out', required=True,
        help='Output directory for the summary CSV and figure.')
    return parser


def load_csvs(path_results):
    """
    Load all CSVs into a single dataframe.

    Args:
        path_results (str): directory containing *_dwi_{metric}_{n}rep.csv files.

    Returns:
        pd.DataFrame: columns participant_id, metric, VertLevel, nrep, value.
    """
    rows = []
    for csv_file in sorted(glob.glob(os.path.join(path_results, '*_dwi_*rep.csv'))):
        basename = os.path.basename(csv_file)
        m = re.match(r'(sub-\w+)_dwi_(\w+)_(\d)rep\.csv', basename)
        # sub-01_dwi_FA_1rep.csv ->  sub-01, FA, 1
        subject, metric, nrep = m.group(1), m.group(2), int(m.group(3))
        df = pd.read_csv(csv_file).rename(columns={'MAP()': 'value'})
        df['participant_id'] = subject
        df['metric'] = metric
        df['nrep'] = nrep
        rows.append(df[['participant_id', 'metric', 'VertLevel', 'nrep', 'value']])
    df = pd.concat(rows, ignore_index=True)
    df['value'] *= df['metric'].map(scaling_factor)
    return df


def pivot_df(df):
    """
    Pivot the long dataframe to one row per (participant, metric, level) with the
    1rep and 2rep values side by side.

    Returns:
        pd.DataFrame: columns participant_id, metric, VertLevel, 1rep, 2rep.
    """
    wide = (df.pivot_table(index=['participant_id', 'metric', 'VertLevel'],
                           columns='nrep', values='value')
              .rename(columns={1: '1rep', 2: '2rep'})
              .reset_index())
    wide.columns.name = None
    return wide.dropna(subset=['1rep', '2rep'])


def compute_summary(wide_df):
    """
    Compute per-metric agreement statistics across subject and levels (C2, C3).

    Returns:
        pd.DataFrame: one row per metric with n_subjects, means, diff, abs diff,
                      relative diff (%) and Pearson r.
    """
    summary = []
    for metric in METRICS:
        metric_df = wide_df[wide_df['metric'] == metric]
        x, y = metric_df['1rep'].values, metric_df['2rep'].values
        # Absolute difference
        diff = x - y                          # 1rep - 2rep (2rep = full acquisition)
        # Relative difference in %
        rel = diff / y * 100
        summary.append({
            'metric': metric,
            'n_subjects': metric_df['participant_id'].nunique(),
            '1rep (mean±SD)': f'{x.mean():.4g} ± {x.std():.2g}',
            '2rep (mean±SD)': f'{y.mean():.4g} ± {y.std():.2g}',
            'diff (1rep-2rep)': f'{diff.mean():.3g}',
            'abs diff': f'{np.abs(diff).mean():.3g}',
            'rel diff [%]': f'{np.abs(rel).mean():.2f}',
            'Pearson r': f'{pearsonr(x, y)[0]:.3f}',
        })
    return pd.DataFrame(summary)


def create_scatter(wide, path_out):
    """
    Create scatter plot 1rep vs 2rep per metric.
    One row per vertebral level (C2, C3).
    """
    mpl.rcParams['font.family'] = 'Arial'
    levels = [2, 3, 4, 5]  # one row per vertebral level (C2, C3, ...)
    fig, axes = plt.subplots(len(levels), len(METRICS), figsize=(len(METRICS) * 4, len(levels) * 4), sharex='col')
    # Loop across levels (rows)
    for row, level in enumerate(levels):
        # Loop across metrics (columns)
        for col, metric in enumerate(METRICS):
            ax = axes[row, col]
            d = wide[(wide['metric'] == metric) & (wide['VertLevel'] == level)]
            x, y = d['1rep'].values, d['2rep'].values
            ax.scatter(x, y, s=60, color='steelblue', alpha=0.7, edgecolor='none')
            lims = [min(x.min(), y.min()), max(x.max(), y.max())]
            ax.plot(lims, lims, color='black', linestyle='--', alpha=0.5)
            # Linear regression line
            slope, intercept = np.polyfit(x, y, 1)
            xfit = np.array([x.min(), x.max()])
            ax.plot(xfit, slope * xfit + intercept, color='red', linewidth=1.5)
            # Show metric name only on the first row
            if row == 0:
                ax.set_title(metric, fontsize=LABELS_FONT_SIZE)
            # Show x-axis label only on the last row
            if row == len(levels) - 1:
                ax.set_xlabel('1 repetition', fontsize=TICKS_FONT_SIZE)
            # Show y-axis label only on the first column
            if col == 0:
                ax.set_ylabel(f'C{level}\n2 repetitions', fontsize=TICKS_FONT_SIZE)
            ax.tick_params(axis='both', labelsize=TICKS_FONT_SIZE - 4)
            ax.set_box_aspect(1)  # square subplot box (independent of data range)
            ax.spines[['top', 'right']].set_visible(False)
            r = pearsonr(x, y)[0]
            sign = '+' if intercept >= 0 else '−'
            ax.text(0.05, 0.95, f'y = {slope:.2f}x {sign} {abs(intercept):.2f}\nr = {r:.2f}',
                    transform=ax.transAxes, ha='left', va='top', fontsize=TICKS_FONT_SIZE - 2)
    plt.tight_layout()
    path_filename = os.path.join(path_out, 'scatter_dti_compare_reps.png')
    plt.savefig(path_filename, dpi=300, bbox_inches='tight')
    print(f'\nFigure saved: {path_filename}')
    plt.close()


def main():
    parser = get_parser()
    args = parser.parse_args()
    os.makedirs(args.path_out, exist_ok=True)

    df = load_csvs(args.path_results)
    # Pivot the dataframe to have one row per (participant, metric, level) with 1rep and 2rep values side by side
    wide_df = pivot_df(df)
    # 21 participants × 5 metrics × 2 levels (C2, C3) = 210 rows. Then 210 / 5 (metrics) = 42
    n_pairs = len(wide_df) // wide_df['metric'].nunique()
    print(f'Loaded {wide_df["participant_id"].nunique()} subjects, {n_pairs} subject × level pairs per metric.\n')

    summary = compute_summary(wide_df)
    print(summary.to_string(index=False))
    path_csv = os.path.join(args.path_out, 'compare_dti_across_reps.csv')
    summary.to_csv(path_csv, index=False)
    print(f'\nSummary saved: {path_csv}')

    create_scatter(wide_df, args.path_out)
    print('\nDone.')


if __name__ == '__main__':
    main()
