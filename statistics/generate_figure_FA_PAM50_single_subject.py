"""
Plot a DTI metric (FA by default; see -metric) in PAM50 space for a single subject, comparing several extraction pipelines
(e.g. warped vs. interpolation, and/or sct_extract_metric -method map vs. wa).

Usage (warped vs. interpolation):
    python generate_figure_FA_PAM50_single_subject.py \
        -path-csv     <path>/dwi_PAM50/sub-XX_dwi_FA_PAM50.csv \
                      <path>/dwi_interpolation_to_PAM50/sub-XX_dwi_FA_interpolated_to_PAM50.csv \
        -dataset-labels Warped Interpolated \
        -path-out     <path>/figures/sub-XX_compare_FA_PAM50.png \
        -subject      sub-XX

Usage (warped vs. interpolation × MAP vs. WA, with one legend per factor):
    python generate_figure_FA_PAM50_single_subject.py \
        -path-csv     <path>/dwi_PAM50_map/sub-XX_dwi_FA_PAM50.csv \
                      <path>/dwi_PAM50_wa/sub-XX_dwi_FA_PAM50.csv \
                      <path>/dwi_interpolation_to_PAM50_map/sub-XX_dwi_FA_interpolated_to_PAM50.csv \
                      <path>/dwi_interpolation_to_PAM50_wa/sub-XX_dwi_FA_interpolated_to_PAM50.csv \
        -dataset-labels "Warped, MAP" "Warped, WA" "Interpolated, MAP" "Interpolated, WA" \
        -dataset-colors tab:blue tab:blue tab:orange tab:orange \
        -dataset-linestyles solid dashed solid dashed \
        -path-out     <path>/figures/sub-XX_compare_FA_PAM50.png \
        -subject      sub-XX
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Shared with the group-level script (same folder): color/line-style mapping and two-factor legends
from generate_figures_dti_PAM50_all_subjects import (LINESTYLE_DASHES, get_dataset_style, add_factor_legends,
                                                     DTI_METRICS, METRIC_TO_AXIS, METRIC_SCALE)

LABELS_FONT_SIZE = 20
TICKS_FONT_SIZE = LABELS_FONT_SIZE - 2

XLIM_VERT_RANGE = (2, 5)


def get_metric_col(df):
    """Return the name of the metric column (e.g. 'MAP()', 'WA()')."""
    candidates = ['MAP()', 'WA()', 'ML()', 'BIN()', 'MEDIAN()', 'MAX()']
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"No recognized metric column found. Columns: {list(df.columns)}")


def vert_num_to_label(v):
    """Convert integer vertebral level to string (e.g. 5 → 'C5', 8 → 'T1')."""
    return f'T{v - 7}' if v > 7 else f'C{v}'


def get_vert_indices(df_single_trace):
    """
    Compute slice positions of intervertebral disc boundaries and mid-vertebrae
    from a single reference trace.

    Returns:
        disc_slices  (list[int]): slice numbers at VertLevel transitions (dashed lines).
        mid_slices   (list[int]): slice numbers at the middle of each vertebral level.
        vert_at_mid  (list[int]): vertebral level number at each mid-slice.
    """
    df = (df_single_trace.dropna(subset=['VertLevel'])
                         .sort_values('Slice (I->S)')
                         .reset_index(drop=True))
    vert = df['VertLevel'].tolist()
    slices = df['Slice (I->S)'].tolist()

    changes = [0]
    for i in range(1, len(vert)):
        if vert[i] != vert[i - 1]:
            changes.append(i)
    changes.append(len(df))

    # Place vertical lines between the last slice of one level and the first slice of the next
    disc_slices = [(slices[i - 1] + slices[i]) / 2 for i in changes[1:-1]]

    mid_slices, vert_at_mid = [], []
    for i in range(len(changes) - 1):
        mid_idx = (changes[i] + changes[i + 1]) // 2
        mid_slices.append(slices[mid_idx])
        vert_at_mid.append(int(vert[mid_idx]))

    return disc_slices, mid_slices, vert_at_mid


def style_ax(ax):
    """Apply shared axis styling (spines, grid, tick size, x-axis inversion)."""
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', which='major', labelsize=TICKS_FONT_SIZE)


def plot(dfs, dataset_labels, dataset_style, subject, output, label, metric='FA'):
    """
    Args:
        dfs            (list[pd.DataFrame]): sct_extract_metric CSVs, one per pipeline.
        dataset_labels (list[str]): legend label per CSV.
        dataset_style  (dict or None): output of get_dataset_style().
        subject        (str): subject label for the title.
        output         (str): output PNG path.
        label          (str): tract/region label to plot.
        metric         (str): DTI metric of the CSVs (FA, MD, RD, AD); sets the axis label and scaling.
    """
    frames = []
    for df, ds in zip(dfs, dataset_labels):
        df = df[df['Label'] == label].sort_values('Slice (I->S)')
        df = df.rename(columns={get_metric_col(df): 'value'}).assign(dataset=ds)
        # DTI maps warped to PAM50 are zero-filled outside the acquired FOV; 0 means no data
        df.loc[df['value'] == 0, 'value'] = np.nan
        # Diffusivities are displayed in ×10⁻³ mm²/s
        df['value'] *= METRIC_SCALE[metric]
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    # Vertebral annotations + slice range from the first trace (reference); in PAM50 space the
    # slice-to-level mapping is the same for all traces.
    df_ref = frames[0]
    disc_slices, mid_slices, vert_at_mid = get_vert_indices(df_ref)
    ref_in_range = df_ref[df_ref['VertLevel'].between(*XLIM_VERT_RANGE)]
    xlim_min = ref_in_range['Slice (I->S)'].min()
    xlim_max = ref_in_range['Slice (I->S)'].max()
    df = df[df['Slice (I->S)'].between(xlim_min, xlim_max)]

    mpl.rcParams['font.family'] = 'Arial'
    fig, ax = plt.subplots(figsize=(6, 5))

    if dataset_style is not None:
        palette, dashes = dataset_style['palette'], dataset_style['dashes']
    else:
        palette, dashes = sns.color_palette('tab10', n_colors=len(dataset_labels)), True
    # style=dataset varies linestyle; alpha lets overlapping traces be seen through one another.
    sns.lineplot(ax=ax, x='Slice (I->S)', y='value', data=df,
                 hue='dataset', style='dataset', hue_order=dataset_labels, style_order=dataset_labels,
                 linewidth=2, palette=palette, dashes=dashes, alpha=0.7)

    ax.set_title(label.title(), fontsize=LABELS_FONT_SIZE)
    ax.set_xlabel('Axial Slice #', fontsize=LABELS_FONT_SIZE)
    ax.set_ylabel(METRIC_TO_AXIS[metric], fontsize=LABELS_FONT_SIZE)
    style_ax(ax)
    ax.set_xlim(xlim_max, xlim_min)  # inverted x-axis (rostral on the left)

    two_legends = dataset_style is not None and 'color_legend' in dataset_style
    if two_legends:
        ax.get_legend().remove()
    else:
        ax.legend(loc='upper right', fontsize=TICKS_FONT_SIZE - 3)

    # Vertebral disc boundaries + level labels, restricted to the visible range.
    ymin = ax.get_ylim()[0]
    for s in disc_slices:
        if xlim_min <= s <= xlim_max:
            ax.axvline(s, color='black', linestyle='--', alpha=0.5, zorder=0)
    for s, v in zip(mid_slices, vert_at_mid):
        if xlim_min <= s <= xlim_max:
            ax.text(s, ymin, vert_num_to_label(v),
                    ha='center', va='bottom', fontsize=TICKS_FONT_SIZE, color='black')

    suptitle = f'{METRIC_TO_AXIS[metric]}, {subject}' if subject else METRIC_TO_AXIS[metric]
    fig.suptitle(suptitle, fontsize=LABELS_FONT_SIZE + 2, y=1.01)
    plt.tight_layout()
    if two_legends:
        # Below the plot, so that the legends do not hide the (single-subject, unsmoothed) traces
        add_factor_legends(fig, dataset_style, TICKS_FONT_SIZE - 3)
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"Saved: {output}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-path-csv', required=True, nargs='+',
                        help='sct_extract_metric CSV(s) of the subject, one per pipeline to compare.')
    parser.add_argument('-dataset-labels', nargs='+', default=None,
                        help='Legend label for each -path-csv entry. Default: CSV folder names.')
    parser.add_argument('-dataset-colors', nargs='+', default=None,
                        help='Line color for each -path-csv entry (e.g. "tab:blue"). Default: tab10.')
    parser.add_argument('-dataset-linestyles', nargs='+', default=None,
                        choices=list(LINESTYLE_DASHES.keys()),
                        help='Line style for each -path-csv entry. If both -dataset-colors and '
                             '-dataset-linestyles are provided and each -dataset-labels entry has the '
                             'form "<A>, <B>", two separate legends are drawn.')
    parser.add_argument('-path-out', required=True, help='Output PNG path.')
    parser.add_argument('-subject', default='', help='Subject label for the plot title.')
    parser.add_argument('-metric', default='FA', choices=DTI_METRICS,
                        help='DTI metric of the -path-csv files (default: FA). Sets the y-axis label and '
                             'displays MD/RD/AD in ×10⁻³ mm²/s.')
    parser.add_argument('-label', default='white matter',
                        help='Tract/region label to extract (default: "white matter").')
    args = parser.parse_args()
    # Expand "~" also when the script is not started from a shell (e.g. from an IDE run configuration)
    args.path_csv = [os.path.expanduser(p) for p in args.path_csv]
    args.path_out = os.path.expanduser(args.path_out)

    dataset_labels = args.dataset_labels or \
        [os.path.basename(os.path.dirname(os.path.abspath(p))) for p in args.path_csv]
    for name, values in [('-dataset-labels', dataset_labels), ('-dataset-colors', args.dataset_colors),
                         ('-dataset-linestyles', args.dataset_linestyles)]:
        if values is not None and len(values) != len(args.path_csv):
            parser.error(f'{name} must have the same number of entries as -path-csv')
    dataset_style = get_dataset_style(dataset_labels, args.dataset_colors, args.dataset_linestyles)

    os.makedirs(os.path.dirname(os.path.abspath(args.path_out)), exist_ok=True)
    dfs = [pd.read_csv(p) for p in args.path_csv]
    plot(dfs, dataset_labels, dataset_style, args.subject, args.path_out, args.label, args.metric)


if __name__ == '__main__':
    main()
