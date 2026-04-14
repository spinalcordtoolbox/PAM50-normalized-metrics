#!/usr/bin/env python3
#
# Plot DTI metrics (FA, MD, RD, AD) computed per-slice in the PAM50 space
# from sct_extract_metric output CSVs produced by process_data_spine-generic_dwi.sh.
#
# Usage (single run):
#     python statistics/generate_figures_dti.py \
#         -path-results /path/to/results/dwi
#
# Usage (compare two preprocessing pipelines):
#     python statistics/generate_figures_dti.py \
#         -path-results /path/to/run_A/results/dwi /path/to/run_B/results/dwi \
#         -dataset-labels "Centerline 30mm" "Deepseg 35mm"
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
import seaborn as sns

DTI_METRICS = ['FA', 'MD', 'RD', 'AD']

METRIC_TO_AXIS = {
    'FA': 'Fractional Anisotropy [a.u.]',
    'MD': 'Mean Diffusivity [mm²/s]',
    'RD': 'Radial Diffusivity [mm²/s]',
    'AD': 'Axial Diffusivity [mm²/s]',
}

# Human-readable names for tract labels from sct_extract_metric output
# Legend: https://spinalcordtoolbox.com/overview/concepts/pam50.html#white-and-gray-matter-atlas
LABEL_DISPLAY_NAMES = {
    'white matter':   'White Matter',
    'gray matter':    'Gray Matter',
    'dorsal columns': 'Dorsal Columns',
    'lateral funiculi':  'Lateral Funiculi',
    'ventral funiculi':  'Ventral Funiculi',
    '0,1':   'Fasciculus Gracilis (L+R)',
    '2,3':   'Fasciculus Cuneatus (L+R)',
    '4,5':   'Lateral Corticospinal Tract (L+R)',
    '12,13': 'Spinal Lemniscus (L+R)',
    '30,31': 'Ventral Horn (L+R)',
}

LABELS_FONT_SIZE = 14
TICKS_FONT_SIZE = 12


def get_parser():
    parser = argparse.ArgumentParser(
        description='Plot DTI metrics (FA, MD, RD, AD) per slice in PAM50 space from '
                    'sct_extract_metric CSV outputs.')
    parser.add_argument(
        '-path-results', required=True, nargs='+',
        help='Path(s) to results/dwi folder(s) containing *_dwi_{FA,MD,RD,AD}_PAM50.csv '
             'files. Provide multiple paths to overlay runs (e.g. to compare preprocessing '
             'pipelines).')
    parser.add_argument(
        '-dataset-labels', required=False, nargs='+', default=None,
        help='Legend labels for each -path-results entry. Must match the number of paths. '
             'Defaults to the folder name of each path.')
    parser.add_argument(
        '-labels-to-plot', required=False, nargs='+',
        default=['white matter'],
        help='Tract/region label(s) to plot. Default: "white matter". '
             f'Available: {list(LABEL_DISPLAY_NAMES.keys())}')
    parser.add_argument(
        '-path-out', required=False, default='stats',
        help='Output directory for saved figures. Default: stats/')
    return parser


def load_dti_csvs(path_results, dataset_label):
    """
    Load all DTI CSVs (FA, MD, RD, AD) from a results directory.

    Args:
        path_results (str): directory containing *_dwi_{metric}_PAM50.csv files.
        dataset_label (str): label to assign to all rows (used for multi-run comparison).

    Returns:
        pd.DataFrame: long-format dataframe with columns:
            participant_id, Slice (I->S), VertLevel, Label, metric, value, std, dataset
    """
    frames = []
    for metric in DTI_METRICS:
        pattern = os.path.join(path_results, f'*_dwi_{metric}_PAM50.csv')
        csv_files = sorted(glob.glob(pattern))
        if not csv_files:
            print(f'  Warning: no {metric} CSVs found in {path_results}')
            continue
        for csv_file in csv_files:
            basename = os.path.basename(csv_file)
            m = re.match(r'(sub-\w+)_dwi_', basename)
            if not m:
                print(f'  Warning: could not parse subject from {basename}, skipping')
                continue
            subject = m.group(1)
            df = pd.read_csv(csv_file)
            df['participant_id'] = subject
            df['metric'] = metric
            df['dataset'] = dataset_label
            df = df.rename(columns={'MAP()': 'value', 'STD()': 'std'})
            frames.append(df[['participant_id', 'Slice (I->S)', 'VertLevel', 'Label',
                               'metric', 'value', 'std', 'dataset']])
    if not frames:
        raise FileNotFoundError(f'No DTI CSVs found in {path_results}')
    return pd.concat(frames, ignore_index=True)


def get_vert_indices(df_single_trace):
    """
    Compute slice positions of intervertebral disc boundaries and mid-vertebrae
    from a single reference trace (one subject, one metric, one label).

    Args:
        df_single_trace (pd.DataFrame): subset with columns Slice (I->S) and VertLevel.

    Returns:
        disc_slices  (list[int]): slice numbers at VertLevel transitions (for dashed lines).
        mid_slices   (list[int]): slice numbers at the middle of each vertebral level.
        vert_at_mid  (list[int]): vertebral level number at each mid-slice.
    """
    df = df_single_trace.sort_values('Slice (I->S)').reset_index(drop=True)
    vert = df['VertLevel'].tolist()
    slices = df['Slice (I->S)'].tolist()

    # Find indices where VertLevel changes
    changes = [0]
    for i in range(1, len(vert)):
        if vert[i] != vert[i - 1]:
            changes.append(i)
    changes.append(len(df))

    # Slice positions at disc boundaries (skip the very first boundary at index 0)
    disc_slices = [slices[i] for i in changes[1:-1]]

    # Mid-slice position and vertebral level for each level segment
    mid_slices, vert_at_mid = [], []
    for i in range(len(changes) - 1):
        mid_idx = (changes[i] + changes[i + 1]) // 2
        mid_slices.append(slices[mid_idx])
        vert_at_mid.append(int(vert[mid_idx]))

    return disc_slices, mid_slices, vert_at_mid


def vert_num_to_label(v):
    """Convert integer vertebral level to string (e.g. 5 → 'C5', 8 → 'T1')."""
    return f'T{v - 7}' if v > 7 else f'C{v}'


def annotate_vertebrae(ax, disc_slices, mid_slices, vert_at_mid, n_per_level, ymin):
    """
    Overlay vertebral disc boundary lines and level text labels on an axis.

    Args:
        ax            : matplotlib Axes.
        disc_slices   : slice numbers for dashed disc boundary lines.
        mid_slices    : slice numbers for vertebral level text labels.
        vert_at_mid   : vertebral level numbers at mid_slices positions.
        n_per_level   : dict mapping VertLevel → subject count.
        ymin (float)  : bottom of the y-axis (anchor for text labels).
    """
    for s in disc_slices:
        ax.axvline(s, color='black', linestyle='--', alpha=0.5, zorder=0)
    for s, v in zip(mid_slices, vert_at_mid):
        n = n_per_level.get(v, 0)
        ax.text(s, ymin, f'{vert_num_to_label(v)}\nn={n}',
                ha='center', va='bottom', fontsize=TICKS_FONT_SIZE, color='black')


def style_ax(ax):
    """Apply shared axis styling (spines, grid, tick size, x-axis inversion)."""
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.invert_xaxis()
    ax.tick_params(axis='both', which='major', labelsize=TICKS_FONT_SIZE)


def create_lineplot_dti(df, metric, labels, path_out, use_hue):
    """
    Create a figure with one subplot per ROI label for a single DTI metric,
    allowing direct comparison of metric values across tracts/regions.

    Layout: 3 columns, ceil(n_labels / 3) rows.

    Args:
        df       (pd.DataFrame): long-format dataframe for all metrics / subjects / datasets.
        metric   (str): DTI metric to plot (e.g. 'FA').
        labels   (list[str]): tract/region labels to show as subplots.
        path_out (str): output directory.
        use_hue  (bool): True when multiple datasets are present → colour by 'dataset'.
    """
    mpl.rcParams['font.family'] = 'Arial'

    df_metric = df[df['metric'] == metric].copy()

    ncols = 3
    nrows = int(np.ceil(len(labels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 5))
    axs = np.array(axes).ravel()

    # Hide unused subplots
    for i in range(len(labels), len(axs)):
        axs[i].set_visible(False)

    n_datasets = df_metric['dataset'].nunique()
    palette = sns.color_palette('tab10', n_colors=n_datasets)

    for ax, label in zip(axs, labels):
        df_label = df_metric[df_metric['Label'] == label]
        if df_label.empty:
            print(f'  Warning: no data for label "{label}", skipping subplot')
            ax.set_visible(False)
            continue

        # Vertebral index reference from first subject/dataset
        first_sub = df_label['participant_id'].iloc[0]
        first_ds = df_label['dataset'].iloc[0]
        ref = df_label[
            (df_label['participant_id'] == first_sub) &
            (df_label['dataset'] == first_ds)
        ]
        disc_slices, mid_slices, vert_at_mid = get_vert_indices(ref)

        n_per_level = (
            df_label.dropna(subset=['value'])
            .groupby('VertLevel')['participant_id']
            .nunique()
            .to_dict()
        )

        if use_hue:
            sns.lineplot(ax=ax, x='Slice (I->S)', y='value', data=df_label,
                         hue='dataset', errorbar='sd', linewidth=2, palette=palette)
            ax.legend(loc='upper right', fontsize=TICKS_FONT_SIZE)
        else:
            sns.lineplot(ax=ax, x='Slice (I->S)', y='value', data=df_label,
                         errorbar='sd', linewidth=2, color='steelblue')

        display_label = LABEL_DISPLAY_NAMES.get(label, label)
        ax.set_title(display_label, fontsize=LABELS_FONT_SIZE)
        ax.set_ylabel(METRIC_TO_AXIS[metric], fontsize=LABELS_FONT_SIZE)
        ax.set_xlabel('Axial Slice #', fontsize=LABELS_FONT_SIZE)
        style_ax(ax)

        ymin, _ = ax.get_ylim()
        annotate_vertebrae(ax, disc_slices, mid_slices, vert_at_mid, n_per_level, ymin)

    n_total = df_metric['participant_id'].nunique()
    fig.suptitle(
        f'{metric} (PAM50 space, n={n_total})',
        fontsize=LABELS_FONT_SIZE + 2, y=1.01
    )
    plt.tight_layout()

    filename = f'lineplot_dti_{metric}.png'
    path_filename = os.path.join(path_out, filename)
    plt.savefig(path_filename, dpi=300, bbox_inches='tight')
    print(f'  Figure saved: {path_filename}')
    plt.close()


def main():
    parser = get_parser()
    args = parser.parse_args()

    # Validate -dataset-labels count
    if args.dataset_labels is not None:
        if len(args.dataset_labels) != len(args.path_results):
            parser.error('-dataset-labels must have the same number of entries as -path-results')
        dataset_labels = args.dataset_labels
    else:
        dataset_labels = [os.path.basename(os.path.normpath(p)) for p in args.path_results]

    os.makedirs(args.path_out, exist_ok=True)

    # Load data from all paths
    print('Loading DTI CSVs...')
    all_frames = []
    for path, ds_label in zip(args.path_results, dataset_labels):
        print(f'  {path}  →  dataset="{ds_label}"')
        df = load_dti_csvs(path, ds_label)
        n_subjects = df['participant_id'].nunique()
        print(f'    {n_subjects} subject(s), {len(df)} rows loaded')
        all_frames.append(df)
    df_all = pd.concat(all_frames, ignore_index=True)

    use_hue = df_all['dataset'].nunique() > 1

    # One figure per DTI metric, all requested labels as subplots
    for metric in DTI_METRICS:
        print(f'\nPlotting: {metric}')
        create_lineplot_dti(df_all, metric, args.labels_to_plot, args.path_out, use_hue)

    print('\nDone.')


if __name__ == '__main__':
    main()
