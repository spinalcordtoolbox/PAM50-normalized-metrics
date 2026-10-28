#!/usr/bin/env python3
#
# Compare DTI metrics (FA, MD, RD, AD) extracted in the native DWI space with different
# sct_extract_metric methods (e.g. -method map, ml, wa) from per-slice CSVs produced by
# process_data_spine-generic_dwi.sh (dwi_native_perslice_<method>/*_dwi_<metric>_native_perslice.csv).
#
# Native slice numbers do not correspond across subjects, so per-slice values are first averaged within
# each vertebral level (VertLevel column, from the PAM50 levels warped to the DWI) per subject, and the
# per-level subject values are then summarized across subjects (mean ± SD).
#
# Outputs (in -path-out):
#   native_perlevel_<label>.png         : 2x2 figure (one subplot per DTI metric), one line per method
#   native_perlevel_<metric>_labels.png : one subplot per label, one line per method
#   native_perlevel_summary.csv         : per-method mean/SD over subjects and differences vs. the first method
#
# Usage:
#     python statistics/generate_figures_dti_native_perlevel.py \
#         -path-results <results>/dwi_native_perslice_map <results>/dwi_native_perslice_ml \
#                       <results>/dwi_native_perslice_wa \
#         -dataset-labels MAP ML WA \
#         -path-out <path>/figures/native_map_ml_wa
#
# Author: Jan Valosek
#

import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# Shared with the group-level script (same folder)
from generate_figures_dti_PAM50_all_subjects import (DTI_METRICS, METRIC_TO_AXIS, METRIC_SCALE,
                                                     LABEL_DISPLAY_NAMES, LINESTYLE_DASHES, vert_num_to_label)

LABELS_FONT_SIZE = 16
TICKS_FONT_SIZE = LABELS_FONT_SIZE - 2
# Categorical colors (validated palette, slots 1-3), one per method
COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
LINESTYLES = ['solid', 'dashed', 'dotted', 'dashdot']


def get_parser():
    parser = argparse.ArgumentParser(
        description='Compare sct_extract_metric methods for DTI metrics extracted per slice in the native '
                    'DWI space, summarized per vertebral level.')
    parser.add_argument('-path-results', required=True, nargs='+',
                        help='Folder(s) with *_dwi_{FA,MD,RD,AD}_native_perslice.csv files, one per method.')
    parser.add_argument('-dataset-labels', nargs='+', default=None,
                        help='Legend label for each -path-results entry. Default: folder names.')
    parser.add_argument('-labels-to-plot', nargs='+',
                        default=['white matter', 'gray matter', 'dorsal columns', 'lateral funiculi',
                                 'ventral funiculi', '4,5', '12,13'],
                        help='Tract/region labels for the per-label figure. The 2x2 per-metric figure is '
                             'generated for the first one. Default: all labels of the processing script.')
    parser.add_argument('-vert-range', nargs=2, type=int, default=[2, 5], metavar=('FIRST', 'LAST'),
                        help='Inclusive range of vertebral levels (C1=1 … C7=7). Default: 2 5.')
    parser.add_argument('-path-out', required=True, help='Output folder.')
    return parser


def load_native_csvs(path_results, dataset_label, vert_range):
    """Load per-slice native CSVs and average per subject, vertebral level, label and metric."""
    frames = []
    for metric in DTI_METRICS:
        for csv_file in sorted(glob.glob(os.path.join(path_results, f'*_dwi_{metric}_native_perslice.csv'))):
            df = pd.read_csv(csv_file)
            value_col = [c for c in df.columns if re.fullmatch(r'[A-Z]+\(\)', c) and c != 'STD()'][0]
            df = df.rename(columns={value_col: 'value'})
            df['value'] = df['value'] * METRIC_SCALE[metric]
            df['participant_id'] = os.path.basename(csv_file).split('_dwi_')[0]
            df['metric'] = metric
            frames.append(df[['participant_id', 'VertLevel', 'Label', 'metric', 'value']])
    if not frames:
        raise FileNotFoundError(f'No *_native_perslice.csv files found in {path_results}')
    df = pd.concat(frames, ignore_index=True).dropna(subset=['value', 'VertLevel'])
    df = df[df['VertLevel'].between(*vert_range)].astype({'VertLevel': int})
    df = df.groupby(['participant_id', 'VertLevel', 'Label', 'metric'], as_index=False)['value'].mean()
    df['dataset'] = dataset_label
    return df


def plot_methods(ax, df, datasets, levels):
    """Mean ± SD across subjects per level, one line per method (slightly offset to avoid overlap)."""
    for k, ds in enumerate(datasets):
        s = df[df['dataset'] == ds].groupby('VertLevel')['value'].agg(['mean', 'std']).reindex(levels)
        offset = (k - (len(datasets) - 1) / 2) * 0.06
        dashes = LINESTYLE_DASHES[LINESTYLES[k % len(LINESTYLES)]]
        ax.errorbar(np.array(levels) + offset, s['mean'], yerr=s['std'], color=COLORS[k % len(COLORS)],
                    linestyle=(0, dashes) if dashes else '-', linewidth=2,
                    marker='o', markersize=5, capsize=3, label=ds)
    ax.set_xticks(levels)
    ax.set_xticklabels([vert_num_to_label(v) for v in levels])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', which='major', labelsize=TICKS_FONT_SIZE)


def _slug(label):
    return re.sub(r'[^A-Za-z0-9]+', '_', LABEL_DISPLAY_NAMES.get(label, label)).strip('_')


def main():
    args = get_parser().parse_args()
    args.path_results = [os.path.expanduser(p) for p in args.path_results]
    args.path_out = os.path.expanduser(args.path_out)
    datasets = args.dataset_labels or [os.path.basename(os.path.normpath(p)) for p in args.path_results]
    if len(datasets) != len(args.path_results):
        raise SystemExit('-dataset-labels must have the same number of entries as -path-results')
    os.makedirs(args.path_out, exist_ok=True)
    mpl.rcParams['font.family'] = 'Arial'

    df = pd.concat([load_native_csvs(p, ds, args.vert_range) for p, ds in zip(args.path_results, datasets)],
                   ignore_index=True)
    # Paired comparison: keep subject/level/label/metric combinations available for all methods
    n_methods = df.groupby(['participant_id', 'VertLevel', 'Label', 'metric'])['dataset'].transform('nunique')
    df = df[n_methods == len(datasets)]
    levels = list(range(args.vert_range[0], args.vert_range[1] + 1))
    print(f'{df["participant_id"].nunique()} subjects; subjects per level: '
          f'{df.groupby("VertLevel")["participant_id"].nunique().to_dict()}')

    # --- Summary table: per-subject mean over levels, per method, and paired differences vs. 1st method ---
    per_subject = df.groupby(['participant_id', 'Label', 'metric', 'dataset'])['value'].mean().unstack('dataset')
    rows = []
    for (label, metric), g in per_subject.groupby(['Label', 'metric']):
        row = {'Label': label, 'metric': metric, 'n': len(g)}
        for ds in datasets:
            row[f'{ds}_mean'], row[f'{ds}_sd'] = g[ds].mean(), g[ds].std()
            row[f'{ds}_n_negative'] = int((df[(df['Label'] == label) & (df['metric'] == metric) &
                                              (df['dataset'] == ds)]['value'] < 0).sum())
        for ds in datasets[1:]:
            d = g[ds] - g[datasets[0]]
            row[f'{ds}-{datasets[0]}_mean'] = d.mean()
            row[f'{ds}-{datasets[0]}_r'] = g[ds].corr(g[datasets[0]])
        rows.append(row)
    summary = pd.DataFrame(rows)
    fname = os.path.join(args.path_out, 'native_perlevel_summary.csv')
    summary.to_csv(fname, index=False)
    print(f'Saved: {fname}')

    # --- 2x2 figure (one subplot per DTI metric) for the first label ---
    label = args.labels_to_plot[0]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, metric in zip(axes.ravel(), DTI_METRICS):
        plot_methods(ax, df[(df['Label'] == label) & (df['metric'] == metric)], datasets, levels)
        ax.set_ylabel(METRIC_TO_AXIS[metric], fontsize=TICKS_FONT_SIZE)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0), ncol=len(datasets),
               frameon=False, fontsize=TICKS_FONT_SIZE)
    fig.suptitle(f'{LABEL_DISPLAY_NAMES.get(label, label)}, native space (mean ± SD across subjects, '
                 f'n={df["participant_id"].nunique()})', fontsize=LABELS_FONT_SIZE + 2)
    plt.tight_layout()
    fname = os.path.join(args.path_out, f'native_perlevel_{_slug(label)}.png')
    plt.savefig(fname, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Figure saved: {fname}')

    # --- One figure per metric, one subplot per label ---
    for metric in DTI_METRICS:
        ncols = int(np.ceil(len(args.labels_to_plot) / 2))
        fig, axes = plt.subplots(2, ncols, figsize=(4.5 * ncols, 9), sharey=True)
        axs = np.array(axes).ravel()
        for ax in axs[len(args.labels_to_plot):]:
            ax.set_visible(False)
        for ax, lab in zip(axs, args.labels_to_plot):
            plot_methods(ax, df[(df['Label'] == lab) & (df['metric'] == metric)], datasets, levels)
            ax.set_title(LABEL_DISPLAY_NAMES.get(lab, lab), fontsize=LABELS_FONT_SIZE)
        for ax in axs[::ncols]:
            ax.set_ylabel(METRIC_TO_AXIS[metric], fontsize=TICKS_FONT_SIZE)
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0), ncol=len(datasets),
                   frameon=False, fontsize=TICKS_FONT_SIZE)
        fig.suptitle(f'{METRIC_TO_AXIS[metric]}, native space (mean ± SD across subjects)',
                     fontsize=LABELS_FONT_SIZE + 2)
        plt.tight_layout()
        fname = os.path.join(args.path_out, f'native_perlevel_{metric}_labels.png')
        plt.savefig(fname, dpi=200, bbox_inches='tight')
        plt.close()
        print(f'Figure saved: {fname}')


if __name__ == '__main__':
    main()
