#!/usr/bin/env python3
#
# Plot DTI metrics (FA, MD, RD, AD) computed per-slice in the PAM50 space
# from sct_extract_metric output CSVs produced by process_data_spine-generic_dwi.sh.
#
# Usage (single run):
#     python statistics/generate_figures_dti_PAM50_all_subjects.py \
#         -path-results /path/to/results/dwi
#
# Usage (compare two preprocessing pipelines):
#     python statistics/generate_figures_dti_PAM50_all_subjects.py \
#         -path-results /path/to/run_A/results/dwi /path/to/run_B/results/dwi \
#         -dataset-labels "Centerline 30mm" "Deepseg 35mm"
#
# Usage (one figure per ROI with all four DTI metrics; here warped vs interpolated for WM):
#     python statistics/generate_figures_dti_PAM50_all_subjects.py \
#         -path-results /path/to/results/dwi_PAM50 /path/to/results/dwi_interpolation_to_PAM50 \
#         -dataset-labels "Warped" "Interpolated" \
#         -labels-to-plot "white matter" -figure-type per-label
#
# Usage (healthy controls as mean ± SD, patients as one thin line per subject):
#     python statistics/generate_figures_dti_PAM50_all_subjects.py \
#         -path-results /path/to/HC/results/dwi_interpolation_to_PAM50 /path/to/DCM/results/dwi_interpolation_to_PAM50 \
#         -dataset-labels "HC" "DCM" -dataset-individual 0 1 -figure-type per-label
#
# Usage (WM FA, one subplot per site against spine-generic; pre-/post-op sessions in one subplot):
#     python statistics/generate_figures_dti_PAM50_all_subjects.py \
#         -path-results <HC dir> "<site A dir>/sub-*" "<site B dir>/*_ses-M0" "<site B dir>/*_ses-M6" \
#         -dataset-labels "HC" "Site A" "Site B pre-op" "Site B post-op" -dataset-panel 1 2 2 \
#         -dataset-linestyles solid solid solid dashed -common-subjects 0 \
#         -metrics FA -figure-type per-dataset-grid
#
# Usage (one figure per DTI metric with all ROIs overlaid in different colors):
#     python statistics/generate_figures_dti_PAM50_all_subjects.py \
#         -path-results /path/to/results/dwi_interpolation_to_PAM50 \
#         -labels-to-plot "dorsal columns" "lateral funiculi" "ventral funiculi" "4,5" "12,13" \
#         -figure-type all-labels
# (use -figure-type all-labels-grid for a single 2x2 figure with all four DTI metrics)
#
# Usage (single figure, one 2x2 block of the four DTI metrics per ROI in a 2x4 grid, PAM50 ROI next to each title):
#     python statistics/generate_figures_dti_PAM50_all_subjects.py \
#         -path-results /path/to/results/dwi_interpolation_to_PAM50 \
#         -labels-to-plot "white matter" "gray matter" "dorsal columns" "lateral funiculi" "ventral funiculi" \
#                         "4,5" "12,13" \
#         -figure-type per-label-grid -roi-insets 1 -metric-abbrev 1 -show-n 0 -level-labels-bottom-only 1
#
# Usage (one 2x4 figure per DTI metric, one subplot per ROI with its PAM50 atlas ROI drawn next to the title):
#     python statistics/generate_figures_dti_PAM50_all_subjects.py \
#         -path-results /path/to/results/dwi_interpolation_to_PAM50 \
#         -labels-to-plot "white matter" "gray matter" "dorsal columns" "lateral funiculi" "ventral funiculi" \
#                         "4,5" "12,13" \
#         -figure-type per-metric -roi-insets 1
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
from matplotlib.lines import Line2D
import seaborn as sns
from tqdm import tqdm

DTI_METRICS = ['FA', 'MD', 'RD', 'AD']

METRIC_TO_AXIS = {
    'FA': 'Fractional Anisotropy',
    'MD': r'Mean Diffusivity [$\times10^{-3}$ mm²/s]',
    'RD': r'Radial Diffusivity [$\times10^{-3}$ mm²/s]',
    'AD': r'Axial Diffusivity [$\times10^{-3}$ mm²/s]',
}

# Diffusivity metrics (MD/RD/AD) are stored in mm²/s but conventionally displayed
# in ×10⁻³ mm²/s (values ~0.3-1.5 instead of ~0.0003-0.0015). FA is unitless.
METRIC_SCALE = {'FA': 1, 'MD': 1000, 'RD': 1000, 'AD': 1000}

# Abbreviated axis labels (-metric-abbrev 1), e.g. for compact multi-panel figures
METRIC_TO_AXIS_ABBREV = {
    'FA': 'FA',
    'MD': r'MD [$\times10^{-3}$ mm²/s]',
    'RD': r'RD [$\times10^{-3}$ mm²/s]',
    'AD': r'AD [$\times10^{-3}$ mm²/s]',
}

# Human-readable names for tract labels from sct_extract_metric output
# Legend: https://spinalcordtoolbox.com/overview/concepts/pam50.html#white-and-gray-matter-atlas
LABEL_DISPLAY_NAMES = {
    'white matter':   'White Matter',
    'gray matter':    'Gray Matter',
    'dorsal columns': 'Dorsal Columns',
    'lateral funiculi':  'Lateral Funiculi',
    'ventral funiculi':  'Ventral Funiculi',
    '0,1':   'Fasciculus Gracilis',
    '2,3':   'Fasciculus Cuneatus',
    '4,5':   'Lateral Corticospinal Tract',
    '12,13': 'Spinal Lemniscus',
    '30,31': 'Ventral Horn',
}

# Fixed color per tract label (-figure-type all-labels, and the single line of -figure-type per-label).
# The color follows the tract, not its position in -labels-to-plot, so a tract keeps its color across figures.
LABEL_PALETTE = {
    'white matter':      '#4cc9f0',
    'gray matter':       '#c71585',
    'dorsal columns':    '#2a78d6',
    'lateral funiculi':  '#eb6834',
    'ventral funiculi':  '#1baf7a',
    '4,5':               '#eda100',
    '12,13':             '#e87ba4',
    '0,1':               '#008300',
    '2,3':               '#4a3aa7',
    '30,31':             '#e34948',
}

# PAM50 atlas IDs (PAM50_atlas_<ID>.nii.gz) of the combined labels, from $SCT_DIR/data/PAM50/atlas/info_label.txt;
# other labels are lists of IDs, e.g. "4,5". Used to draw the ROIs (-roi-insets).
LABEL_ATLAS_IDS = {
    'white matter':     range(0, 30),
    'gray matter':      range(30, 36),
    'dorsal columns':   range(0, 4),
    'lateral funiculi': range(4, 14),
    'ventral funiculi': range(14, 30),
}
# ROI drawn in the -roi-insets: atlas partial-volume fraction threshold, vertebral level of the (middle) PAM50 slice,
# padding (voxels) around the spinal cord, and inset height (fraction of the subplot height)
ROI_THRESH = 0.5
ROI_VERT_LEVEL = 3
ROI_PAD = 8
ROI_INSET_HEIGHT = 0.3
# -figure-type per-label-grid: height of the title (and ROI) row of each 2x2 block, as a fraction of the block height
BLOCK_TITLE_HEIGHT = 0.16

FIGURE_TYPES = ['per-metric', 'per-label', 'per-label-grid', 'all-labels', 'all-labels-grid', 'per-dataset-grid']

DEFAULT_FONT_SIZE = 20
LABELS_FONT_SIZE = DEFAULT_FONT_SIZE
TICKS_FONT_SIZE = LABELS_FONT_SIZE-2
# Font size of the per-level subject counts (n=...) below the vertebral level labels
N_FONT_SIZE = TICKS_FONT_SIZE-6

# X-axis is restricted to this inclusive range of vertebral levels (C1=1 … C7=7, T1=8 …).
# Default; can be changed with -vert-range.
XLIM_VERT_RANGE = (2, 5)

# Band around the mean across subjects: 'sd' (± 1 SD) or None (lines only). Can be changed with -errorbar.
ERRORBAR = 'sd'

# Where the number of subjects is shown: False → per vertebral level below the level labels on the x-axis;
# True → per dataset (or sex) in the legend. Can be changed with -n-in-legend.
N_IN_LEGEND = False
# Show the number of subjects per vertebral level on the x-axis. Can be changed with -show-n.
SHOW_N = True
# Vertebral level labels (C2, C3, ...) only in the bottom row of 2x2 figures. See -level-labels-bottom-only.
LEVEL_LABELS_BOTTOM_ONLY = False
# Fixed number of decimals of the y-axis tick labels (None: matplotlib default). See -ytick-decimals.
YTICK_DECIMALS = None
# Width of the mean lines. Can be changed with -linewidth.
LINEWIDTH = 2
# Datasets plotted as one thin line per subject instead of the mean (± SD) across subjects, and the width
# of these lines. See -dataset-individual, -individual-linewidth.
INDIVIDUAL_DATASETS = set()
INDIVIDUAL_LINEWIDTH = 0.8
# Dataset colors and line styles (get_dataset_style()), set in main(); used for the per-subject legend entries
DATASET_STYLE = None
# Legend placement: subplot (DTI metric, -figure-type per-label) and location. See -legend-metric, -legend-loc.
LEGEND_METRIC = 'FA'
LEGEND_LOC = 'upper right'

# Dash patterns for -dataset-linestyles (seaborn format: '' is solid, otherwise (dash, gap))
LINESTYLE_DASHES = {'solid': '', 'dashed': (4, 1.5), 'dotted': (1, 1), 'dashdot': (3, 1.25, 1.5, 1.25)}

# Width/height ratio of the country flags drawn by draw_flag() (-dataset-flags)
FLAG_ASPECT = {'CH': 1.0, 'CZ': 1.5, 'US': 1.9}

SEX_PALETTE = {'Males': 'blue', 'Females': 'red'}
SEX_LABEL_MAP = {'M': 'Males', 'F': 'Females'}


def get_parser():
    parser = argparse.ArgumentParser(
        description='Plot DTI metrics (FA, MD, RD, AD) per slice in PAM50 space from '
                    'sct_extract_metric CSV outputs.')
    parser.add_argument(
        '-path-results', required=True, nargs='+',
        help='Path(s) to results/dwi folder(s) containing *_dwi_{FA,MD,RD,AD}_PAM50.csv '
             'files. Provide multiple paths to overlay runs (e.g. to compare preprocessing '
             'pipelines). A path can end with a filename prefix pattern to load only matching '
             'files, e.g. "<dir>/*_ses-M0" for one session of a longitudinal dataset. "none" adds a '
             'dataset without data (e.g. pending processing), shown as an empty subplot with '
             '-figure-type per-dataset-grid.')
    parser.add_argument(
        '-dataset-labels', required=False, nargs='+', default=None,
        help='Legend labels for each -path-results entry. Must match the number of paths. '
             'Defaults to the folder name of each path.')
    parser.add_argument(
        '-dataset-colors', required=False, nargs='+', default=None,
        help='Line color (any matplotlib color, e.g. "tab:blue") for each -path-results entry. '
             'Must match the number of paths. Default: tab10 palette, one color per dataset.')
    parser.add_argument(
        '-dataset-linestyles', required=False, nargs='+', default=None,
        choices=list(LINESTYLE_DASHES.keys()),
        help='Line style for each -path-results entry. Must match the number of paths. '
             'Default: a different line style per dataset. Together with -dataset-colors, this '
             'allows encoding two factors, e.g. color = registration, line style = -method. '
             'If both are provided and each -dataset-labels entry has the form "<A>, <B>", two '
             'separate legends are drawn: one for colors (<A>) and one for line styles (<B>).')
    parser.add_argument(
        '-dataset-first-level', required=False, nargs='+', type=int, default=None,
        help='First vertebral level (C1=1 … C7=7) to plot for each -path-results entry; more rostral '
             'levels are dropped for that dataset (e.g. to hide levels covered by only a few '
             'subjects). Must match the number of paths. Default: no restriction.')
    parser.add_argument(
        '-dataset-last-level', required=False, nargs='+', type=int, default=None,
        help='Last vertebral level (C1=1 … C7=7) to plot for each -path-results entry; more caudal '
             'levels are dropped for that dataset (e.g. to hide the edge of the acquired field of view). '
             'Must match the number of paths. Default: no restriction.')
    parser.add_argument(
        '-dataset-individual', required=False, nargs='+', type=int, choices=[0, 1], default=None,
        help='For each -path-results entry: 1 → one thin line per subject (e.g. patients), 0 → mean '
             '(± SD, see -errorbar) across subjects. Must match the number of paths. Default: all 0. '
             'Only with several datasets (-figure-type per-metric, per-label).')
    parser.add_argument(
        '-individual-linewidth', required=False, type=float, default=INDIVIDUAL_LINEWIDTH,
        help=f'Width of the per-subject lines of -dataset-individual. Default: {INDIVIDUAL_LINEWIDTH}.')
    parser.add_argument(
        '-labels-to-plot', required=False, nargs='+',
        default=['white matter'],
        help='Tract/region label(s) to plot. Default: "white matter". '
             f'Available: {list(LABEL_DISPLAY_NAMES.keys())}')
    parser.add_argument(
        '-errorbar', required=False, choices=['sd', 'none'], default='sd',
        help='Band around the mean across subjects: "sd" (± 1 SD, default) or "none" (lines only, '
             'e.g. when many datasets are overlaid).')
    parser.add_argument(
        '-n-in-legend', required=False, type=int, choices=[0, 1], default=0,
        help='0: number of subjects per vertebral level below the level labels (default). 1: number of '
             'subjects per dataset (or sex) in the legend instead, no counts on the x-axis.')
    parser.add_argument(
        '-show-n', required=False, type=int, choices=[0, 1], default=1,
        help='1: number of subjects per vertebral level below the level labels (default). 0: no counts.')
    parser.add_argument(
        '-level-labels-bottom-only', required=False, type=int, choices=[0, 1], default=0,
        help='1: vertebral level labels (C2, C3, ...) only in the bottom row of 2x2 figures '
             '(-figure-type per-label, all-labels-grid). Default: 0 (all subplots).')
    parser.add_argument(
        '-metric-abbrev', required=False, type=int, choices=[0, 1], default=0,
        help='1: abbreviated y-axis labels (e.g. "FA" instead of "Fractional Anisotropy"). '
             'Default: 0.')
    parser.add_argument(
        '-ytick-decimals', required=False, type=int, default=None,
        help='Fixed number of decimals of the y-axis tick labels (e.g. 1). Default: matplotlib default.')
    parser.add_argument(
        '-linewidth', required=False, type=float, default=LINEWIDTH,
        help=f'Width of the mean lines. Default: {LINEWIDTH}.')
    parser.add_argument(
        '-font-size', required=False, type=int, default=LABELS_FONT_SIZE,
        help=f'Font size of axis labels; tick labels and legends are scaled accordingly. '
             f'Default: {LABELS_FONT_SIZE}.')
    parser.add_argument(
        '-legend-metric', required=False, choices=DTI_METRICS, default=DTI_METRICS[0],
        help='-figure-type per-label: subplot (DTI metric) holding the legend. Default: FA.')
    parser.add_argument(
        '-legend-loc', required=False, default='upper right',
        help='-figure-type per-label/per-metric: legend location within its subplot (matplotlib loc, '
             'e.g. "upper left"). Default: "upper right".')
    parser.add_argument(
        '-vert-range', required=False, nargs=2, type=int, default=list(XLIM_VERT_RANGE),
        metavar=('FIRST', 'LAST'),
        help='Inclusive range of vertebral levels shown on the x-axis (C1=1 … C7=7). '
             f'Default: {XLIM_VERT_RANGE[0]} {XLIM_VERT_RANGE[1]} (C2-C5).')
    parser.add_argument(
        '-dataset-panel', required=False, nargs='+', type=int, default=None,
        help='-figure-type per-dataset-grid: subplot number (1, 2, …) for each -path-results entry except '
             'the first (reference); datasets with the same number share a subplot (e.g. pre- and '
             'post-op sessions). Must have one entry less than the number of paths. Default: one '
             'subplot per dataset.')
    parser.add_argument(
        '-grid-ncols', required=False, type=int, default=4,
        help='-figure-type per-dataset-grid, per-label-grid: number of subplot (block) columns. Default: 4.')
    parser.add_argument(
        '-dataset-flags', required=False, nargs='+', default=None, choices=list(FLAG_ASPECT) + ['none'],
        help='-figure-type per-dataset-grid: country flag drawn left of the subplot title, one '
             'per -path-results entry ("none" for no flag; the first entry, the reference, is ignored). '
             'Flags are drawn instead of emojis, which matplotlib cannot render. Default: no flags.')
    parser.add_argument(
        '-dataset-legend', required=False, nargs='+', default=None,
        help='-figure-type per-dataset-grid: legend entry (e.g. pathology "tSCI") for each -path-results '
             'entry (the first entry, the reference, is ignored); the number of subjects is appended. With '
             'this option, the subplot title shows only -dataset-labels (e.g. site and scanner), without '
             'the number of subjects. Default: no legend, number of subjects in the title.')
    parser.add_argument(
        '-footnote', required=False, nargs='+', default=None,
        help='-figure-type per-dataset-grid: text line(s) shown below the figure, e.g. to explain '
             'abbreviations. Default: none.')
    parser.add_argument(
        '-common-subjects', required=False, type=int, choices=[0, 1], default=1,
        help='1: with several -path-results, keep only subjects with complete data in all of them '
             '(paired comparison of methods on the same subjects). 0: keep all subjects (comparison '
             'of independent cohorts, e.g. two datasets). Default: 1.')
    parser.add_argument(
        '-metrics', required=False, nargs='+', choices=DTI_METRICS, default=DTI_METRICS,
        help='DTI metric(s) to plot with -figure-type per-metric, all-labels and per-dataset-grid. Default: all.')
    parser.add_argument(
        '-figure-type', required=False, nargs='+', choices=FIGURE_TYPES, default=['per-metric'],
        help='Figure layout(s) to generate. '
             '"per-metric": one figure per DTI metric, one subplot per label (default). '
             '"per-label": one figure per label, one subplot per DTI metric (FA, MD, RD, AD). '
             '"per-label-grid": a single figure with one 2x2 "per-label" block per label, blocks in a grid of '
             '-grid-ncols columns (single -path-results only; see -roi-insets). '
             '"all-labels": one figure per DTI metric, all labels overlaid in a single axis in '
             'different colors (single -path-results only). '
             '"all-labels-grid": same as "all-labels", but a single 2x2 figure with one subplot '
             'per DTI metric. '
             '"per-dataset-grid": one figure per DTI metric (-metrics) and label, one subplot per '
             'dataset (see -dataset-panel): the first -path-results entry (reference, e.g. healthy '
             'controls) as mean ± SD in every subplot, the other datasets as one thin line per subject.')
    parser.add_argument(
        '-roi-insets', required=False, type=int, choices=[0, 1], default=0,
        help='-figure-type per-metric, per-label-grid: 1 → draw the PAM50 atlas ROI of each label (probability ≥ '
             f'{ROI_THRESH}, mid-{vert_num_to_label(ROI_VERT_LEVEL)} axial slice) next to the subplot (block) title. '
             'Requires nibabel and the PAM50 template (see -path-pam50). Default: 0.')
    parser.add_argument(
        '-path-pam50', required=False, default=os.path.join(os.environ.get('SCT_DIR', ''), 'data', 'PAM50'),
        help='PAM50 folder with template/ and atlas/ subfolders (-roi-insets). Default: $SCT_DIR/data/PAM50.')
    parser.add_argument(
        '-participant-file', required=False, default=None,
        help='Path to participants.tsv with at least participant_id and sex columns. '
             'Required to generate per-sex figures.')
    parser.add_argument(
        '-path-out', required=False, default='stats',
        help='Output directory for saved figures. Default: stats/')
    return parser


def load_dti_csvs(path_results, dataset_label):
    """
    Load all DTI CSVs (FA, MD, RD, AD) from a results directory.

    Args:
        path_results (str): directory containing *_dwi_{metric}_PAM50.csv files, or such a directory
            followed by a filename prefix pattern (e.g. "<dir>/*_ses-M0") to load only matching files.
        dataset_label (str): label to assign to all rows (used for multi-run comparison).

    Returns:
        pd.DataFrame: long-format dataframe with columns:
            participant_id, Slice (I->S), VertLevel, Label, metric, value, std, dataset
    """
    frames = []
    for metric in DTI_METRICS:
        # Matches both warped (`*_dwi_FA_PAM50.csv`) and interpolated
        # (`*_dwi_FA_interpolated_to_PAM50.csv`) naming.
        if os.path.isdir(path_results):
            pattern = os.path.join(path_results, f'*_dwi_{metric}_*PAM50.csv')
        else:
            pattern = f'{path_results}_dwi_{metric}_*PAM50.csv'
        csv_files = sorted(glob.glob(pattern))
        if not csv_files:
            print(f'  Warning: no {metric} CSVs found in {path_results}')
            continue
        for csv_file in tqdm(csv_files, desc=f'    {metric}', unit='subj', leave=False):
            basename = os.path.basename(csv_file)
            m = re.match(r'(sub-[^_]+(?:_ses-[^_]+)?)_dwi_', basename)
            if not m:
                print(f'  Warning: could not parse subject from {basename}, skipping')
                continue
            subject = m.group(1)
            df = pd.read_csv(csv_file)
            df['participant_id'] = subject
            df['metric'] = metric
            df['dataset'] = dataset_label
            # Value column is named after the sct_extract_metric -method, e.g. MAP() or WA()
            value_col = [c for c in df.columns if re.fullmatch(r'[A-Z]+\(\)', c) and c != 'STD()'][0]
            df = df.rename(columns={value_col: 'value', 'STD()': 'std'})
            if 'std' not in df.columns:
                df['std'] = np.nan
            # DTI maps warped to PAM50 are zero-filled outside the acquired FOV, so slices without
            # any data get an exact 0 instead of NaN. Mask them to not bias the across-subject mean.
            df.loc[df['value'] == 0, ['value', 'std']] = np.nan
            df['value'] *= METRIC_SCALE[metric]
            df['std'] *= METRIC_SCALE[metric]
            frames.append(df[['participant_id', 'Slice (I->S)', 'VertLevel', 'Label',
                               'metric', 'value', 'std', 'dataset']])
    if not frames:
        raise FileNotFoundError(f'No DTI CSVs found in {path_results}')
    return pd.concat(frames, ignore_index=True)


def get_dataset_style(dataset_labels, colors, linestyles):
    """
    Build the seaborn palette and dashes mappings for -dataset-colors and -dataset-linestyles.

    Args:
        dataset_labels (list[str]): dataset labels, one per -path-results entry.
        colors     (list[str] or None): color per dataset; None → tab10 palette.
        linestyles (list[str] or None): LINESTYLE_DASHES key per dataset; None → seaborn defaults.

    Returns:
        dict or None: {'palette': {dataset: color}, 'dashes': {dataset: dashes} or True};
                      None if neither colors nor linestyles are provided.
    """
    if colors is None and linestyles is None:
        return None
    if colors is None:
        colors = sns.color_palette('tab10', n_colors=len(dataset_labels))
    dashes = True if linestyles is None else \
        {ds: LINESTYLE_DASHES[ls] for ds, ls in zip(dataset_labels, linestyles)}
    dataset_style = {'palette': dict(zip(dataset_labels, colors)), 'dashes': dashes}

    # Two factors encoded by color and line style: "<A>, <B>" labels → one legend per factor
    parts = [ds.split(', ') for ds in dataset_labels]
    if colors is not None and linestyles is not None and all(len(p) == 2 for p in parts):
        color_legend = dict((p[0], c) for p, c in zip(parts, colors))
        linestyle_legend = dict((p[1], LINESTYLE_DASHES[ls]) for p, ls in zip(parts, linestyles))
        # Each factor level has to map to a single color / line style
        if len(color_legend) == len({(p[0], c) for p, c in zip(parts, colors)}) and \
                len(linestyle_legend) == len({(p[1], ls) for p, ls in zip(parts, linestyles)}):
            dataset_style['color_legend'] = color_legend
            dataset_style['linestyle_legend'] = linestyle_legend
        else:
            print('  Warning: -dataset-labels parts do not map consistently to colors/line styles; '
                  'drawing a single legend')
    return dataset_style


def add_factor_legends(target, dataset_style, fontsize):
    """
    Draw two legends, one for colors and one for line styles (see get_dataset_style()), side by
    side below the figure (target=Figure) or stacked in the upper right corner (target=Axes).

    Args:
        target        : matplotlib Figure or Axes.
        dataset_style (dict): output of get_dataset_style() with color_legend and linestyle_legend.
        fontsize      (int): legend font size.
    """
    color_handles = [Line2D([], [], color=c, linewidth=2) for c in dataset_style['color_legend'].values()]
    # Line styles are shown in neutral gray, so they do not suggest one of the colors
    linestyle_handles = [Line2D([], [], color='dimgray', linewidth=2, dashes=d if d else (None, None))
                         for d in dataset_style['linestyle_legend'].values()]
    legend_kwargs = dict(fontsize=fontsize, handlelength=3)
    if isinstance(target, plt.Figure):
        legend_kwargs.update(frameon=False, ncol=2)
        target.legend(color_handles, dataset_style['color_legend'].keys(),
                      loc='upper right', bbox_to_anchor=(0.48, 0), **legend_kwargs)
        target.legend(linestyle_handles, dataset_style['linestyle_legend'].keys(),
                      loc='upper left', bbox_to_anchor=(0.52, 0), **legend_kwargs)
    else:
        legend_color = target.legend(color_handles, dataset_style['color_legend'].keys(),
                                     loc='upper right', **legend_kwargs)
        target.add_artist(legend_color)
        target.legend(linestyle_handles, dataset_style['linestyle_legend'].keys(),
                      loc='upper right', bbox_to_anchor=(1, 0.8), **legend_kwargs)


def keep_common_subjects(df, labels):
    """
    Keep only subjects with all DTI metrics and all requested labels in every dataset, so that
    datasets are compared on the same subjects (e.g. when a run is incomplete for some of them).

    Args:
        df     (pd.DataFrame): long-format dataframe for all metrics / subjects / datasets.
        labels (list[str]): tract/region labels that have to be available.

    Returns:
        pd.DataFrame: df restricted to the common subjects.
    """
    df_labels = df[df['Label'].isin(labels)].dropna(subset=['value'])
    n_combinations = (df_labels.groupby('participant_id')[['dataset', 'metric', 'Label']]
                      .apply(lambda x: len(x.drop_duplicates())))
    n_expected = df['dataset'].nunique() * len(DTI_METRICS) * len(labels)
    subjects = n_combinations[n_combinations == n_expected].index
    excluded = sorted(set(df['participant_id']) - set(subjects))
    if excluded:
        print(f'  Excluding {len(excluded)} subject(s) without complete data in all datasets: '
              f'{", ".join(excluded)}')
    print(f'  {len(subjects)} subject(s) common to all datasets')
    return df[df['participant_id'].isin(subjects)]


def format_counts(counts):
    """
    Format the number of subjects per dataset for a vertebral level label: a single number when all
    datasets have the same count (e.g. paired comparison of methods), one number per dataset for two
    datasets (e.g. "202/12"), and the min–max range for more datasets (to keep the label short).
    """
    if counts.nunique() == 1:
        return str(counts.iloc[0])
    if len(counts) <= 2:
        return '/'.join(str(n) for n in counts)
    return f'{counts.min()}–{counts.max()}'


def get_vert_reference(df, metric, label):
    """
    Get everything needed to restrict the x-axis to XLIM_VERT_RANGE and to annotate vertebral
    levels. Slices are in the PAM50 space, so the slice-to-level mapping is the same for all
    subjects and is pooled across them (a single subject does not have to cover all levels).

    Args:
        df     (pd.DataFrame): long-format dataframe for all metrics / subjects / datasets.
        metric (str): DTI metric used as reference (e.g. 'FA').
        label  (str): tract/region label used as reference.

    Returns:
        disc_slices, mid_slices, vert_at_mid: see get_vert_indices(); restricted to the x-axis range.
        xlim_min, xlim_max (int): slice range covering XLIM_VERT_RANGE.
        n_per_level (dict): VertLevel → number of subjects with data (str; one number per dataset,
                            separated by '/').
    """
    df_ref = df[(df['metric'] == metric) & (df['Label'] == label)]
    ref = df_ref.dropna(subset=['VertLevel']).drop_duplicates(subset='Slice (I->S)')
    disc_slices, mid_slices, vert_at_mid = get_vert_indices(ref)

    # Slice range covering the requested vertebral levels (used to clip data and xlim).
    ref_in_range = ref[ref['VertLevel'].between(*XLIM_VERT_RANGE)]
    xlim_min = ref_in_range['Slice (I->S)'].min()
    xlim_max = ref_in_range['Slice (I->S)'].max()

    # Keep only annotations within the visible range
    disc_slices = [s for s in disc_slices if xlim_min <= s <= xlim_max]
    mid_slices, vert_at_mid = zip(*[
        (s, v) for s, v in zip(mid_slices, vert_at_mid)
        if xlim_min <= s <= xlim_max
    ]) if mid_slices else ([], [])

    # Number of subjects with data per level; per dataset (e.g. "202/12") when several are plotted
    n_per_level = (
        df_ref.dropna(subset=['value'])
        .groupby(['VertLevel', 'dataset'])['participant_id']
        .nunique()
        .unstack(fill_value=0)
        .reindex(columns=pd.unique(df_ref['dataset']), fill_value=0)
    )
    n_per_level = {v: format_counts(row) for v, row in n_per_level.iterrows()}

    return disc_slices, mid_slices, vert_at_mid, xlim_min, xlim_max, n_per_level


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
    df = (df_single_trace.dropna(subset=['VertLevel'])
                         .sort_values('Slice (I->S)')
                         .reset_index(drop=True))
    vert = df['VertLevel'].tolist()
    slices = df['Slice (I->S)'].tolist()

    # Find indices where VertLevel changes
    changes = [0]
    for i in range(1, len(vert)):
        if vert[i] != vert[i - 1]:
            changes.append(i)
    changes.append(len(df))

    # Slice positions at disc boundaries (skip the very first boundary at index 0).
    # Place vertical lines between the last slice of one level and the first slice of the next
    disc_slices = [(slices[i - 1] + slices[i]) / 2 for i in changes[1:-1]]

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


def annotate_vertebrae(ax, disc_slices, mid_slices, vert_at_mid, n_per_level, ymin, show_levels=True,
                       show_n=True):
    """
    Overlay vertebral disc boundary lines and level text labels on an axis.

    Args:
        ax            : matplotlib Axes.
        disc_slices   : slice numbers for dashed disc boundary lines.
        mid_slices    : slice numbers for vertebral level text labels.
        vert_at_mid   : vertebral level numbers at mid_slices positions.
        n_per_level   : dict mapping VertLevel → subject count.
        ymin (float)  : bottom of the y-axis (anchor for text labels).
        show_levels (bool): draw the level labels (and subject counts); False draws only the disc lines.
        show_n (bool): draw the subject counts below the level labels (if not N_IN_LEGEND and SHOW_N).
    """
    for s in disc_slices:
        ax.axvline(s, color='black', linestyle='--', alpha=0.5, zorder=0)
    if not show_levels:
        return
    for s, v in zip(mid_slices, vert_at_mid):
        if N_IN_LEGEND or not SHOW_N or not show_n:
            # Subject counts are in the legend; vertebral level label at the bottom
            ax.text(s, ymin, vert_num_to_label(v),
                    ha='center', va='bottom', fontsize=TICKS_FONT_SIZE, color='black')
            continue
        n = n_per_level.get(v, 0)
        # Subject count in a smaller font at the bottom, vertebral level label right above it
        ax.text(s, ymin, f'n={n}',
                ha='center', va='bottom', fontsize=N_FONT_SIZE, color='black')
        ax.annotate(vert_num_to_label(v), xy=(s, ymin), xytext=(0, N_FONT_SIZE * 1.3),
                    textcoords='offset points', ha='center', va='bottom',
                    fontsize=TICKS_FONT_SIZE, color='black')


def lineplot_datasets(ax, df, hue, palette, dashes):
    """
    Seaborn lineplot of the mean (± ERRORBAR) across subjects per hue level; datasets in
    INDIVIDUAL_DATASETS are instead drawn as one thin line per subject, in the color and line style of
    the dataset. The legend is rebuilt by add_legend().

    Args:
        ax      : matplotlib Axes.
        df      (pd.DataFrame): data of one metric and one label.
        hue     (str): hue column ('dataset' or 'sex').
        palette : seaborn palette (dict {level: color} when INDIVIDUAL_DATASETS is used).
        dashes  : seaborn dashes (dict {level: dashes} or True).
    """
    is_indiv = df[hue].isin(INDIVIDUAL_DATASETS) if hue == 'dataset' else pd.Series(False, index=df.index)
    df_mean = df[~is_indiv]
    if not df_mean.empty:
        # style=hue varies linestyle per dataset (1st solid, 2nd dashed, ...)
        # and alpha lets overlapping traces be seen through one another.
        sns.lineplot(ax=ax, x='Slice (I->S)', y='value', data=df_mean,
                     hue=hue, style=hue, errorbar=ERRORBAR, linewidth=LINEWIDTH,
                     palette=palette, dashes=dashes, alpha=0.7)
    for (dataset, _), df_sub in df[is_indiv].dropna(subset=['value']).groupby([hue, 'participant_id']):
        df_sub = df_sub.sort_values('Slice (I->S)')
        ls = dashes.get(dataset, '') if isinstance(dashes, dict) else ''
        ax.plot(df_sub['Slice (I->S)'], df_sub['value'], color=palette[dataset],
                linestyle=(0, ls) if ls else '-', linewidth=INDIVIDUAL_LINEWIDTH, alpha=0.8,
                zorder=1.5)  # above the SD band (zorder 1), below the mean lines (zorder 2)


def add_legend(ax, df, hue, loc, fontsize):
    """
    Draw the seaborn hue legend on an axis; with N_IN_LEGEND, append the number of subjects with data
    in the plotted range to each entry (e.g. "spine-generic, n=202").

    Args:
        ax       : matplotlib Axes with a seaborn lineplot (hue=hue).
        df       (pd.DataFrame): plotted data (one metric, one label, plotted slice range).
        hue      (str): hue column ('dataset' or 'sex').
        loc      (str): matplotlib legend location.
        fontsize (int): legend font size.
    """
    handles, labels = ax.get_legend_handles_labels()
    individual = sorted(set(df[hue]) & INDIVIDUAL_DATASETS) if hue == 'dataset' else []
    if individual:
        # Handles of the per-subject datasets (not in the seaborn legend), in the -path-results order
        style = {lab: (h.get_color(), h.get_linestyle()) for h, lab in zip(handles, labels)}
        for ds in individual:
            ls = DATASET_STYLE['dashes'].get(ds, '') if isinstance(DATASET_STYLE['dashes'], dict) else ''
            style[ds] = (DATASET_STYLE['palette'][ds], (0, ls) if ls else '-')
        labels = [ds for ds in DATASET_STYLE['palette'] if ds in style]
        # Thinner legend lines for the per-subject datasets
        handles = [Line2D([], [], color=style[ds][0], linestyle=style[ds][1],
                          linewidth=INDIVIDUAL_LINEWIDTH * 2 if ds in INDIVIDUAL_DATASETS else LINEWIDTH)
                   for ds in labels]
    if N_IN_LEGEND:
        n = df.dropna(subset=['value']).groupby(hue)['participant_id'].nunique()
        labels = [f'{lab}, n={n.get(lab, 0)}' for lab in labels]
    ax.legend(handles, labels, loc=loc, fontsize=fontsize)


def fig_scale():
    """
    Figure size factor for fonts larger than the default (-font-size): grows with the square root of the
    font size ratio, so that the text gets relatively larger while the axis labels still have room.
    """
    return max(1.0, (LABELS_FONT_SIZE / DEFAULT_FONT_SIZE) ** 0.5)


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
    if YTICK_DECIMALS is not None:
        # Tick spacing of 1, 2 or 5 × 10^k (not e.g. 0.25), and at least 10^-decimals, so that every tick
        # value is shown exactly with the fixed number of decimals
        step = 10 ** -YTICK_DECIMALS
        locator = mpl.ticker.MaxNLocator(steps=[1, 2, 5, 10])
        ticks = locator.tick_values(*ax.get_ylim())
        if len(ticks) > 1 and np.diff(ticks).min() < step - 1e-12:
            locator = mpl.ticker.MultipleLocator(step)
        ax.yaxis.set_major_locator(locator)
        ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter(f'%.{YTICK_DECIMALS}f'))


def get_label_atlas_ids(label):
    """PAM50 atlas IDs of a label: combined label (LABEL_ATLAS_IDS) or comma-separated IDs, e.g. '4,5' → [4, 5]."""
    if label in LABEL_ATLAS_IDS:
        return list(LABEL_ATLAS_IDS[label])
    return [int(i) for i in label.split(',')]


def load_pam50_rois(path_pam50, labels):
    """
    Load the PAM50 T2 axial slice in the middle of vertebral level ROI_VERT_LEVEL and the atlas ROI of each
    label (partial-volume fraction ≥ ROI_THRESH) on it, cropped around the spinal cord.

    Returns:
        background (np.ndarray): 2D T2 slice scaled to 0-1, displayed with A at the top and R on the left.
        rois       (dict): label → 2D boolean mask, same layout as background.
    """
    import nibabel as nib

    def load_slice(fname, z):
        return np.asarray(nib.load(fname).dataobj[:, :, z], dtype=float)

    # PAM50 is LAS: the third axis is inferior → superior
    levels = np.asanyarray(nib.load(os.path.join(path_pam50, 'template', 'PAM50_levels.nii.gz')).dataobj)
    z = int(np.median(np.where(levels == ROI_VERT_LEVEL)[2]))
    cord = load_slice(os.path.join(path_pam50, 'template', 'PAM50_cord.nii.gz'), z) > 0.5
    xs, ys = np.where(cord)
    x0, x1 = max(xs.min() - ROI_PAD, 0), xs.max() + ROI_PAD + 1
    y0, y1 = max(ys.min() - ROI_PAD, 0), ys.max() + ROI_PAD + 1

    def to_display(arr):
        # Transpose + flip once: A at the top, R on the left (as in FSLeyes)
        return np.flipud(arr[x0:x1, y0:y1].T)

    background = to_display(load_slice(os.path.join(path_pam50, 'template', 'PAM50_t2.nii.gz'), z))
    background = (background - background.min()) / (np.ptp(background) or 1)
    rois, fractions = {}, {}
    for label in labels:
        frac = np.zeros_like(background)
        for atlas_id in get_label_atlas_ids(label):
            if atlas_id not in fractions:
                fractions[atlas_id] = to_display(load_slice(
                    os.path.join(path_pam50, 'atlas', f'PAM50_atlas_{atlas_id:02d}.nii.gz'), z))
            frac += fractions[atlas_id]
        rois[label] = frac >= ROI_THRESH
    return background, rois


def draw_roi_inset(ax, background, mask, color, title):
    """
    Draw the ROI (mask filled with color on the PAM50 T2 background) above the top-left corner of the axis,
    with the title to its right.
    """
    rgb = np.repeat(background[..., None], 3, axis=2)
    rgb[mask] = mpl.colors.to_rgb(color)
    # Inset width in axes coordinates that keeps the image aspect ratio
    bbox = ax.get_window_extent()
    height = ROI_INSET_HEIGHT
    width = height * rgb.shape[1] / rgb.shape[0] * bbox.height / bbox.width
    inset = ax.inset_axes([0, 1.02, width, height])
    inset.imshow(rgb, interpolation='nearest')
    inset.set_axis_off()
    ax.text(width + 0.03, 1.02 + height / 2, title, transform=ax.transAxes, ha='left', va='center',
            fontsize=LABELS_FONT_SIZE)


def create_lineplot_dti(df, metric, labels, path_out, hue=None, dataset_style=None, rois=None):
    """
    Create a grid figure (nrows=2, ncols=ceil(n_labels/2)) for a single DTI metric,
    with one subplot per ROI label.

    Y-axis limits are shared across all subplots to allow cross-ROI comparison.
    Y-axis label is shown only on the first column to avoid repetition.
    X-axis label and tick numbers shown only on the bottom row.

    Args:
        df       (pd.DataFrame): long-format dataframe for all metrics / subjects / datasets.
        metric   (str): DTI metric to plot (e.g. 'FA').
        labels   (list[str]): tract/region labels, one subplot each.
        path_out (str): output directory.
        hue      (str or None): column to colour lines by — 'dataset', 'sex', or None
                                (single steelblue line).
        dataset_style (dict or None): {'palette': {dataset: color}, 'dashes': {dataset: dashes}}
                                      used when hue='dataset'; see get_dataset_style().
        rois     (tuple or None): (background, {label: mask}) from load_pam50_rois(); the ROI of each label is
                                  drawn next to the subplot title, and the line has the color of the label.
    """
    mpl.rcParams['font.family'] = 'Arial'

    nrows = 2
    ncols = int(np.ceil(len(labels) / nrows))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6 * fig_scale(), nrows * 5 * fig_scale()))
    axs = np.array(axes).ravel()

    # Hide unused subplots (when n_labels is odd)
    for i in range(len(labels), len(axs)):
        axs[i].set_visible(False)

    if hue == 'sex':
        palette = SEX_PALETTE
    elif hue == 'dataset':
        n_datasets = df['dataset'].nunique()
        palette = sns.color_palette('tab10', n_colors=n_datasets)
    else:
        palette = None
    dashes = True
    if hue == 'dataset' and dataset_style is not None:
        palette, dashes = dataset_style['palette'], dataset_style['dashes']

    # Pre-compute vertebral indices once (slice positions are the same across all labels)
    disc_slices, mid_slices, vert_at_mid, xlim_min, xlim_max, n_per_level = \
        get_vert_reference(df, metric, labels[0])
    df = df[df['Slice (I->S)'].between(xlim_min, xlim_max)]

    # Print n per vertebral level to terminal
    print('  Subjects per vertebral level:')
    for v in sorted(n_per_level):
        print(f'    {vert_num_to_label(int(v))}: {n_per_level[v]}')

    # --- First pass: draw lineplots ---
    active_axes = []
    for i, (ax, label) in enumerate(zip(axs, labels)):
        df_label = df[(df['metric'] == metric) & (df['Label'] == label)].copy()
        if hue == 'sex' and 'sex' in df_label.columns:
            df_label['sex'] = df_label['sex'].map(SEX_LABEL_MAP).fillna(df_label['sex'])
        if df_label.empty:
            print(f'  Warning: no data for label "{label}", skipping subplot')
            ax.set_visible(False)
            continue

        if hue is not None:
            lineplot_datasets(ax, df_label, hue, palette, dashes)
            if i == 0:
                if hue == 'dataset' and dataset_style is not None and 'color_legend' in dataset_style:
                    if ax.get_legend() is not None:
                        ax.get_legend().remove()
                    add_factor_legends(ax, dataset_style, TICKS_FONT_SIZE-5)
                else:
                    add_legend(ax, df_label, hue, LEGEND_LOC, TICKS_FONT_SIZE-5)
            elif ax.get_legend() is not None:
                ax.get_legend().remove()
        else:
            color = LABEL_PALETTE.get(label, 'steelblue') if rois is not None else 'steelblue'
            sns.lineplot(ax=ax, x='Slice (I->S)', y='value', data=df_label,
                         errorbar=ERRORBAR, linewidth=LINEWIDTH, color=color)

        if rois is None:
            ax.set_title(LABEL_DISPLAY_NAMES.get(label, label), fontsize=LABELS_FONT_SIZE)
        # X-axis label and ticks only on bottom row
        if i >= ncols:
            ax.set_xlabel('Axial Slice #', fontsize=LABELS_FONT_SIZE)
        else:
            ax.set_xlabel('')
            ax.tick_params(axis='x', labelbottom=False)
        # Y-axis label only on first column
        if i % ncols == 0:
            ax.set_ylabel(METRIC_TO_AXIS[metric], fontsize=LABELS_FONT_SIZE)
        else:
            ax.set_ylabel('')
        style_ax(ax)
        active_axes.append(ax)

    # --- Shared y-axis limits across all subplots ---
    all_ymins = [ax.get_ylim()[0] for ax in active_axes]
    all_ymaxs = [ax.get_ylim()[1] for ax in active_axes]
    shared_ymin, shared_ymax = min(all_ymins), max(all_ymaxs)
    for ax in active_axes:
        ax.set_ylim(shared_ymin, shared_ymax)

    for ax in active_axes:
        ax.set_xlim(xlim_max, xlim_min)  # inverted x-axis (rostral on the left)

    # --- Second pass: vertebral annotations (need final ymin) ---
    for ax in active_axes:
        annotate_vertebrae(ax, disc_slices, mid_slices, vert_at_mid, n_per_level, shared_ymin)

    if rois is not None:
        # tight_layout ignores the insets: reserve room above each subplot for the ROI and title
        plt.tight_layout()
        fig.subplots_adjust(top=1 - ROI_INSET_HEIGHT * 0.5, hspace=ROI_INSET_HEIGHT * 2)
        fig.suptitle(METRIC_TO_AXIS[metric], fontsize=LABELS_FONT_SIZE + 2, y=1.04)
        background, masks = rois
        for ax, label in zip(axs, labels):
            if ax in active_axes:
                draw_roi_inset(ax, background, masks[label], LABEL_PALETTE.get(label, 'steelblue'),
                               LABEL_DISPLAY_NAMES.get(label, label))
    else:
        fig.suptitle(METRIC_TO_AXIS[metric], fontsize=LABELS_FONT_SIZE + 2, y=1.01)
        plt.tight_layout()

    suffix = f'_per{hue}' if hue is not None else ''
    if rois is not None:
        suffix += '_rois'
    filename = f'lineplot_dti_{metric}{suffix}.png'
    path_filename = os.path.join(path_out, filename)
    plt.savefig(path_filename, dpi=300, bbox_inches='tight')
    print(f'  Figure saved: {path_filename}')
    plt.close()


def plot_label_metrics(axs, df, label, hue=None, palette=None, dashes=True, dataset_style=None):
    """
    Draw the four DTI metrics (FA, MD, RD, AD) of a single label into a 2x2 grid of axes (row-major).

    Args:
        axs   (np.ndarray): four axes.
        df    (pd.DataFrame): long-format dataframe of the label.
        label (str): tract/region label.
        hue, palette, dashes, dataset_style: see create_lineplot_dti_per_label().
    """
    ncols = 2
    for i, (ax, metric) in enumerate(zip(axs, DTI_METRICS)):
        disc_slices, mid_slices, vert_at_mid, xlim_min, xlim_max, n_per_level = \
            get_vert_reference(df, metric, label)
        df_metric = df[(df['metric'] == metric) & df['Slice (I->S)'].between(xlim_min, xlim_max)]
        if hue is not None:
            lineplot_datasets(ax, df_metric, hue, palette, dashes)
            if metric == LEGEND_METRIC:
                if hue == 'dataset' and dataset_style is not None and 'color_legend' in dataset_style:
                    if ax.get_legend() is not None:
                        ax.get_legend().remove()
                else:
                    add_legend(ax, df_metric, hue, LEGEND_LOC, TICKS_FONT_SIZE-5)
            elif ax.get_legend() is not None:
                ax.get_legend().remove()
        else:
            sns.lineplot(ax=ax, x='Slice (I->S)', y='value', data=df_metric,
                         errorbar=ERRORBAR, linewidth=LINEWIDTH, color=LABEL_PALETTE.get(label, 'steelblue'))
        # X-axis label and ticks only on bottom row
        if i >= ncols:
            ax.set_xlabel('Axial Slice #', fontsize=LABELS_FONT_SIZE)
        else:
            ax.set_xlabel('')
            ax.tick_params(axis='x', labelbottom=False)
        ax.set_ylabel(METRIC_TO_AXIS[metric], fontsize=LABELS_FONT_SIZE)
        style_ax(ax)
        ax.set_xlim(xlim_max, xlim_min)  # inverted x-axis (rostral on the left)
        annotate_vertebrae(ax, disc_slices, mid_slices, vert_at_mid, n_per_level, ax.get_ylim()[0],
                           show_levels=not LEVEL_LABELS_BOTTOM_ONLY or i >= ncols)


def create_lineplot_dti_per_label(df, label, path_out, hue=None, dataset_style=None):
    """
    Create a 2x2 figure for a single ROI label, with one subplot per DTI metric (FA, MD, RD, AD).

    Y-axis limits are independent across subplots (metrics have different units and ranges).

    Args:
        df       (pd.DataFrame): long-format dataframe for all metrics / subjects / datasets.
        label    (str): tract/region label to plot.
        path_out (str): output directory.
        hue      (str or None): column to colour lines by — 'dataset', 'sex', or None
                                (single steelblue line).
        dataset_style (dict or None): {'palette': {dataset: color}, 'dashes': {dataset: dashes}}
                                      used when hue='dataset'; see get_dataset_style().
    """
    mpl.rcParams['font.family'] = 'Arial'

    df = df[df['Label'] == label].copy()
    if df.empty:
        print(f'  Warning: no data for label "{label}", skipping figure')
        return
    if hue == 'sex':
        df['sex'] = df['sex'].map(SEX_LABEL_MAP).fillna(df['sex'])
        palette = SEX_PALETTE
    elif hue == 'dataset':
        palette = sns.color_palette('tab10', n_colors=df['dataset'].nunique())
    else:
        palette = None
    dashes = True
    if hue == 'dataset' and dataset_style is not None:
        palette, dashes = dataset_style['palette'], dataset_style['dashes']

    nrows, ncols = 2, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6 * fig_scale(), nrows * 5 * fig_scale()))
    axs = np.array(axes).ravel()

    plot_label_metrics(axs, df, label, hue, palette, dashes, dataset_style)

    fig.suptitle(LABEL_DISPLAY_NAMES.get(label, label), fontsize=LABELS_FONT_SIZE + 2, y=1.01)
    plt.tight_layout()
    if hue == 'dataset' and dataset_style is not None and 'color_legend' in dataset_style:
        add_factor_legends(fig, dataset_style, TICKS_FONT_SIZE-2)

    suffix = f'_per{hue}' if hue is not None else ''
    label_fname = re.sub(r'[^A-Za-z0-9]+', '_', LABEL_DISPLAY_NAMES.get(label, label)).strip('_')
    path_filename = os.path.join(path_out, f'lineplot_dti_{label_fname}{suffix}.png')
    plt.savefig(path_filename, dpi=300, bbox_inches='tight')
    print(f'  Figure saved: {path_filename}')
    plt.close()


def create_lineplot_dti_per_label_grid(df, labels, path_out, rois=None, ncols=4):
    """
    Create a single figure with one 2x2 block (FA, MD, RD, AD) per ROI label, the blocks arranged in a grid
    (ncols columns), e.g. 2x4 for seven labels (the last cell stays empty). Each block is drawn as with
    -figure-type per-label; its title gives the label and, with rois, the PAM50 atlas ROI drawn left of it.

    Args:
        df       (pd.DataFrame): long-format dataframe (single dataset).
        labels   (list[str]): tract/region labels, one block each.
        path_out (str): output directory.
        rois     (tuple or None): (background, {label: mask}) from load_pam50_rois().
        ncols    (int): number of block columns.
    """
    mpl.rcParams['font.family'] = 'Arial'

    nrows = int(np.ceil(len(labels) / ncols))
    # Block size as for -figure-type per-label, plus room at the top for the title and ROI
    block_w, block_h = 2 * 6 * fig_scale(), 2 * 5 * fig_scale() / (1 - BLOCK_TITLE_HEIGHT)
    fig = plt.figure(figsize=(ncols * block_w, nrows * block_h))
    subfigs = fig.subfigures(nrows, ncols, squeeze=False).ravel()
    renderer = fig.canvas.get_renderer()

    for subfig in subfigs[len(labels):]:
        subfig.set_visible(False)
    for subfig, label in zip(subfigs, labels):
        df_label = df[df['Label'] == label]
        if df_label.empty:
            print(f'  Warning: no data for label "{label}", skipping block')
            subfig.set_visible(False)
            continue
        print(f'  Plotting: {label}')
        axs = subfig.subplots(2, 2).ravel()
        plot_label_metrics(axs, df_label, label)
        subfig.subplots_adjust(left=0.14, right=0.98, bottom=0.12, top=1 - BLOCK_TITLE_HEIGHT,
                               wspace=0.42, hspace=0.12)

        # Title (and ROI), centered over the block
        title = LABEL_DISPLAY_NAMES.get(label, label)
        y_center = 1 - BLOCK_TITLE_HEIGHT / 2
        text = subfig.text(0, y_center, title, ha='left', va='center', fontsize=LABELS_FONT_SIZE + 2)
        text_w = text.get_window_extent(renderer).transformed(subfig.transSubfigure.inverted()).width
        inset_w, inset_h, gap = 0, 0, 0
        if rois is not None:
            background, masks = rois
            inset_h = BLOCK_TITLE_HEIGHT * 0.9
            inset_w = inset_h * background.shape[1] / background.shape[0] * block_h / block_w
            gap = 0.02
        x0 = (0.14 + 0.98) / 2 - (inset_w + gap + text_w) / 2
        text.set_x(x0 + inset_w + gap)
        if rois is not None:
            rgb = np.repeat(background[..., None], 3, axis=2)
            rgb[masks[label]] = mpl.colors.to_rgb(LABEL_PALETTE.get(label, 'steelblue'))
            inset = subfig.add_axes([x0, y_center - inset_h / 2, inset_w, inset_h])
            inset.imshow(rgb, interpolation='nearest')
            inset.set_axis_off()

    suffix = '_rois' if rois is not None else ''
    path_filename = os.path.join(path_out, f'lineplot_dti_per_label_grid{suffix}.png')
    plt.savefig(path_filename, dpi=150, bbox_inches='tight')
    print(f'  Figure saved: {path_filename}')
    plt.close()


def plot_all_labels(ax, df, metric, labels, show_levels=True):
    """
    Draw a single DTI metric with all ROI labels overlaid into an axis, one color (and linestyle)
    per label. The seaborn legend is left on the axis for the caller to place or remove.

    Args:
        ax       : matplotlib Axes.
        df       (pd.DataFrame): long-format dataframe for all metrics / subjects (single dataset).
        metric   (str): DTI metric to plot (e.g. 'FA').
        labels   (list[str]): tract/region labels, one line each.
        show_levels (bool): draw the vertebral level labels (see annotate_vertebrae()).
    """
    disc_slices, mid_slices, vert_at_mid, xlim_min, xlim_max, n_per_level = \
        get_vert_reference(df, metric, labels[0])
    df = df[(df['metric'] == metric) & df['Label'].isin(labels) &
            df['Slice (I->S)'].between(xlim_min, xlim_max)].copy()
    df['Label'] = df['Label'].map(lambda x: LABEL_DISPLAY_NAMES.get(x, x))
    display_names = [LABEL_DISPLAY_NAMES.get(label, label) for label in labels]
    palette = {LABEL_DISPLAY_NAMES.get(label, label): LABEL_PALETTE.get(label, 'gray')
               for label in labels}

    # style varies linestyle per label, so that labels are not distinguished by color alone
    sns.lineplot(ax=ax, x='Slice (I->S)', y='value', data=df,
                 hue='Label', hue_order=display_names, style='Label', style_order=display_names,
                 errorbar=ERRORBAR, err_kws={'alpha': 0.1}, linewidth=LINEWIDTH, palette=palette)

    ax.set_xlabel('Axial Slice #', fontsize=LABELS_FONT_SIZE)
    ax.set_ylabel(METRIC_TO_AXIS[metric], fontsize=LABELS_FONT_SIZE)
    style_ax(ax)
    ax.set_xlim(xlim_max, xlim_min)  # inverted x-axis (rostral on the left)
    annotate_vertebrae(ax, disc_slices, mid_slices, vert_at_mid, n_per_level, ax.get_ylim()[0],
                       show_levels=show_levels)


def create_lineplot_dti_all_labels(df, metric, labels, path_out):
    """
    Create a single-axis figure for a single DTI metric with all ROI labels overlaid, one color
    (and linestyle) per label.

    Args:
        df       (pd.DataFrame): long-format dataframe for all metrics / subjects (single dataset).
        metric   (str): DTI metric to plot (e.g. 'FA').
        labels   (list[str]): tract/region labels, one line each.
        path_out (str): output directory.
    """
    mpl.rcParams['font.family'] = 'Arial'

    fig, ax = plt.subplots(figsize=(14, 7))
    plot_all_labels(ax, df, metric, labels)
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), frameon=False,
              fontsize=TICKS_FONT_SIZE-3, handlelength=3)

    plt.tight_layout()
    path_filename = os.path.join(path_out, f'lineplot_dti_{metric}_all_labels.png')
    plt.savefig(path_filename, dpi=300, bbox_inches='tight')
    print(f'  Figure saved: {path_filename}')
    plt.close()


def create_lineplot_dti_all_labels_grid(df, labels, path_out):
    """
    Create a 2x2 figure with one subplot per DTI metric (FA, MD, RD, AD), each with all ROI labels
    overlaid, one color (and linestyle) per label. A single legend is shared by all subplots.

    Args:
        df       (pd.DataFrame): long-format dataframe for all metrics / subjects (single dataset).
        labels   (list[str]): tract/region labels, one line each.
        path_out (str): output directory.
    """
    mpl.rcParams['font.family'] = 'Arial'

    nrows, ncols = 2, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6 * fig_scale(), nrows * 5 * fig_scale()))
    axs = np.array(axes).ravel()

    for i, (ax, metric) in enumerate(zip(axs, DTI_METRICS)):
        plot_all_labels(ax, df, metric, labels, show_levels=not LEVEL_LABELS_BOTTOM_ONLY or i >= ncols)
        handles, legend_labels = ax.get_legend_handles_labels()
        ax.get_legend().remove()
        # X-axis label and ticks only on bottom row
        if i < ncols:
            ax.set_xlabel('')
            ax.tick_params(axis='x', labelbottom=False)

    plt.tight_layout()
    # Extra space between columns: tight_layout leaves too little room for the right column's y-axis
    # label with large fonts (-font-size)
    plt.subplots_adjust(wspace=0.3 * fig_scale())
    # Shared legend below the subplots
    fig.legend(handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, 0),
               ncol=3, frameon=False, fontsize=TICKS_FONT_SIZE-2, handlelength=3)

    path_filename = os.path.join(path_out, 'lineplot_dti_all_metrics_all_labels.png')
    plt.savefig(path_filename, dpi=300, bbox_inches='tight')
    print(f'  Figure saved: {path_filename}')
    plt.close()


def draw_flag(ax, code):
    """
    Draw a simplified country flag filling an axes (data coordinates: x 0 → FLAG_ASPECT[code], y 0 → 1).

    Args:
        ax   : matplotlib Axes (e.g. from fig.add_axes) with the flag's aspect ratio.
        code (str): ISO country code, a key of FLAG_ASPECT.
    """
    from matplotlib.patches import Rectangle, Polygon
    w = FLAG_ASPECT[code]
    if code == 'CH':
        ax.add_patch(Rectangle((0, 0), w, 1, color='#da291c'))
        for x, y, cw, ch in [(0.4, 0.19, 0.2, 0.62), (0.19, 0.4, 0.62, 0.2)]:
            ax.add_patch(Rectangle((x, y), cw, ch, color='white'))
    elif code == 'CZ':
        ax.add_patch(Rectangle((0, 0.5), w, 0.5, color='white'))
        ax.add_patch(Rectangle((0, 0), w, 0.5, color='#d7141a'))
        ax.add_patch(Polygon([(0, 0), (0, 1), (w / 2, 0.5)], color='#11457e'))
    elif code == 'US':
        for i in range(13):
            ax.add_patch(Rectangle((0, i / 13), w, 1 / 13, color='#b22234' if i % 2 == 0 else 'white'))
        ax.add_patch(Rectangle((0, 6 / 13), 0.76, 7 / 13, color='#3c3b6e'))
        # Stars as small dots
        xs, ys = np.meshgrid(np.linspace(0.07, 0.69, 6), np.linspace(6 / 13 + 0.05, 1 - 0.05, 5))
        ax.scatter(xs.ravel(), ys.ravel(), s=0.3, color='white', linewidths=0)
    # Thin frame, so that white parts are visible on the white background
    ax.add_patch(Rectangle((0, 0), w, 1, fill=False, edgecolor='0.6', linewidth=0.5))
    ax.set_xlim(0, w)
    ax.set_ylim(0, 1)
    ax.axis('off')


def add_title_flags(fig, axes_flags):
    """
    Add a country flag left of each subplot title (after the layout is final).

    Args:
        fig        : matplotlib Figure.
        axes_flags (list[tuple]): (Axes, ISO country code) pairs.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for ax, code in axes_flags:
        bbox = ax.title.get_window_extent(renderer)
        (x0, y0), (_, y1) = inv.transform([(bbox.x0, bbox.y0), (bbox.x1, bbox.y1)])
        # Flag height ~ title text height; width from the aspect ratio (converted to figure coordinates)
        height = (y1 - y0) * 0.85
        fig_w, fig_h = fig.get_size_inches()
        width = height * FLAG_ASPECT[code] * fig_h / fig_w
        ax_flag = fig.add_axes([x0 - width - 0.004, y0 + (y1 - y0 - height) / 2, width, height])
        draw_flag(ax_flag, code)


def create_lineplot_per_dataset_grid(df, metric, label, path_out, dataset_labels, panels, ncols,
                                     dataset_style=None, footnote=None, flags=None, legend_labels=None,
                                     legend_loc=LEGEND_LOC):
    """
    Create a figure for one DTI metric and one ROI label with one subplot per dataset (or group of
    datasets, see -dataset-panel). Every subplot shows the reference dataset (first -path-results entry)
    as mean ± ERRORBAR, and the subplot's dataset(s) as one thin line per subject. Y-axis limits are
    shared across subplots.

    Args:
        df             (pd.DataFrame): long-format dataframe for all metrics / subjects / datasets.
        metric         (str): DTI metric to plot.
        label          (str): tract/region label to plot.
        path_out       (str): output directory.
        dataset_labels (list[str]): dataset labels in the -path-results order; the first is the reference.
        panels         (list[int]): subplot number for each non-reference dataset.
        ncols          (int): number of subplot columns.
        dataset_style  (dict or None): see get_dataset_style(); None → tab10 colors, solid lines.
        footnote       (list[str] or None): text line(s) below the figure.
        flags          (list[str] or None): country flag (FLAG_ASPECT key or 'none') per non-reference
                                            dataset, drawn left of the subplot title.
        legend_labels  (list[str] or None): legend entry (e.g. pathology) per non-reference dataset, shown
                                            with the number of subjects in a per-subplot legend; the
                                            title then holds only the dataset label.
        legend_loc     (str): matplotlib location of the per-subplot legend.
    """
    mpl.rcParams['font.family'] = 'Arial'

    df = df[(df['metric'] == metric) & (df['Label'] == label)]
    if df.empty:
        print(f'  Warning: no {metric} data for label "{label}", skipping figure')
        return
    if dataset_style is None:
        dataset_style = get_dataset_style(dataset_labels, None, ['solid'] * len(dataset_labels))
    palette, dashes = dataset_style['palette'], dataset_style['dashes']
    reference = dataset_labels[0]

    disc_slices, mid_slices, vert_at_mid, xlim_min, xlim_max, _ = get_vert_reference(df, metric, label)
    df = df[df['Slice (I->S)'].between(xlim_min, xlim_max)]
    n_subjects = df.dropna(subset=['value']).groupby('dataset')['participant_id'].nunique()

    panel_ids = list(dict.fromkeys(panels))
    nrows = int(np.ceil(len(panel_ids) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5 * fig_scale(), nrows * 4.5 * fig_scale()),
                             sharey=True, squeeze=False)
    axs = axes.ravel()

    axes_flags = []
    for i, panel in enumerate(panel_ids):
        ax = axs[i]
        datasets = [ds for ds, p in zip(dataset_labels[1:], panels) if p == panel]
        if flags is not None:
            # Flag of the first dataset of the subplot
            code = flags[panels.index(panel)]
            if code != 'none':
                axes_flags.append((ax, code))
        df_ref = df[df['dataset'] == reference]
        sns.lineplot(ax=ax, x='Slice (I->S)', y='value', data=df_ref, errorbar=ERRORBAR,
                     linewidth=LINEWIDTH, color=palette[reference], alpha=0.7)
        for ds in datasets:
            ls = dashes.get(ds, '') if isinstance(dashes, dict) else ''
            for _, df_sub in df[df['dataset'] == ds].dropna(subset=['value']).groupby('participant_id'):
                df_sub = df_sub.sort_values('Slice (I->S)')
                ax.plot(df_sub['Slice (I->S)'], df_sub['value'], color=palette[ds],
                        linestyle=(0, ls) if ls else '-', linewidth=INDIVIDUAL_LINEWIDTH, alpha=0.8)
        # Title with the number of subjects; with several datasets in one subplot, one line per dataset
        # and the non-solid line style named
        title_lines = []
        legend_handles = []
        for ds in datasets:
            ls_name = next((name for name, d in LINESTYLE_DASHES.items()
                            if isinstance(dashes, dict) and d == dashes[ds] and d), None)
            suffix = f' ({ls_name})' if len(datasets) > 1 and ls_name else ''
            n = n_subjects.get(ds, 0)
            if legend_labels is not None:
                title_lines.append(ds if n else f'{ds} (pending)')
                ls = dashes.get(ds, '') if isinstance(dashes, dict) else ''
                legend_handles.append(Line2D([0], [0], color=palette[ds], linestyle=(0, ls) if ls else '-',
                                             linewidth=INDIVIDUAL_LINEWIDTH * 1.5,
                                             label=f'{legend_labels[dataset_labels.index(ds) - 1]}, n={n}'))
            else:
                title_lines.append(f'{ds}, n={n}{suffix}' if n else f'{ds} (pending)')
        ax.set_title('\n'.join(title_lines), fontsize=TICKS_FONT_SIZE - 2)
        if not any(n_subjects.get(ds, 0) for ds in datasets):
            # Dataset(s) without data (-path-results none): empty subplot
            for artist in ax.lines + ax.collections:
                artist.remove()
            ax.text(0.5, 0.5, 'pending', transform=ax.transAxes, ha='center', va='center',
                    fontsize=TICKS_FONT_SIZE, color='gray')
        elif legend_handles:
            # Legend box limited to above the vertebral level labels at the bottom of the subplot
            ax.legend(handles=legend_handles, loc=legend_loc, bbox_to_anchor=(0, 0.1, 1, 0.9),
                      fontsize=TICKS_FONT_SIZE - 4, frameon=True)

        # X-axis label and ticks only on bottom row (of each column)
        if i + ncols >= len(panel_ids):
            ax.set_xlabel('Axial Slice #', fontsize=LABELS_FONT_SIZE - 2)
        else:
            ax.set_xlabel('')
            ax.tick_params(axis='x', labelbottom=False)
        ax.set_ylabel(METRIC_TO_AXIS[metric] if i % ncols == 0 else '', fontsize=LABELS_FONT_SIZE - 2)
        style_ax(ax)
        ax.set_xlim(xlim_max, xlim_min)  # inverted x-axis (rostral on the left)
    for ax in axs[len(panel_ids):]:
        ax.set_visible(False)
    # Vertebral levels after all data are drawn, so that they sit at the bottom of the shared y-axis
    for i, ax in enumerate(axs[:len(panel_ids)]):
        annotate_vertebrae(ax, disc_slices, mid_slices, vert_at_mid, {}, ax.get_ylim()[0],
                           show_levels=not LEVEL_LABELS_BOTTOM_ONLY or i + ncols >= len(panel_ids),
                           show_n=False)

    err = ' ± SD' if ERRORBAR == 'sd' else ''
    fig.suptitle(f'{LABEL_DISPLAY_NAMES.get(label, label)}: {reference}, n={n_subjects.get(reference, 0)} '
                 f'(gray, mean{err}) vs. individual subjects', fontsize=LABELS_FONT_SIZE)
    plt.tight_layout()
    if footnote:
        fig.text(0.01, 0, '\n'.join(footnote), ha='left', va='top', fontsize=TICKS_FONT_SIZE - 4)
    add_title_flags(fig, axes_flags)

    label_fname = re.sub(r'[^A-Za-z0-9]+', '_', LABEL_DISPLAY_NAMES.get(label, label)).strip('_')
    path_filename = os.path.join(path_out, f'lineplot_{metric}_{label_fname}_perdataset_grid.png')
    plt.savefig(path_filename, dpi=300, bbox_inches='tight')
    print(f'  Figure saved: {path_filename}')
    plt.close()


def main():
    parser = get_parser()
    args = parser.parse_args()
    # Expand "~" also when the script is not started from a shell (e.g. from an IDE run configuration)
    args.path_results = [os.path.expanduser(p) for p in args.path_results]
    args.path_out = os.path.expanduser(args.path_out)
    if args.participant_file is not None:
        args.participant_file = os.path.expanduser(args.participant_file)

    global XLIM_VERT_RANGE, ERRORBAR, N_IN_LEGEND, LEGEND_METRIC, LEGEND_LOC
    global INDIVIDUAL_DATASETS, INDIVIDUAL_LINEWIDTH, DATASET_STYLE
    global SHOW_N, LEVEL_LABELS_BOTTOM_ONLY, YTICK_DECIMALS, LINEWIDTH, LABELS_FONT_SIZE, TICKS_FONT_SIZE, N_FONT_SIZE
    XLIM_VERT_RANGE = tuple(args.vert_range)
    ERRORBAR = None if args.errorbar == 'none' else args.errorbar
    N_IN_LEGEND, LEGEND_METRIC, LEGEND_LOC = bool(args.n_in_legend), args.legend_metric, args.legend_loc
    SHOW_N, LEVEL_LABELS_BOTTOM_ONLY = bool(args.show_n), bool(args.level_labels_bottom_only)
    YTICK_DECIMALS, LINEWIDTH = args.ytick_decimals, args.linewidth
    LABELS_FONT_SIZE = args.font_size
    TICKS_FONT_SIZE = LABELS_FONT_SIZE - 2
    N_FONT_SIZE = TICKS_FONT_SIZE - 6
    if args.metric_abbrev:
        METRIC_TO_AXIS.update(METRIC_TO_AXIS_ABBREV)

    # Validate -dataset-labels count
    if args.dataset_labels is not None:
        if len(args.dataset_labels) != len(args.path_results):
            parser.error('-dataset-labels must have the same number of entries as -path-results')
        dataset_labels = args.dataset_labels
    else:
        dataset_labels = [os.path.basename(os.path.normpath(p)) for p in args.path_results]

    for arg_name in ['dataset_colors', 'dataset_linestyles', 'dataset_first_level', 'dataset_last_level',
                     'dataset_individual', 'dataset_flags', 'dataset_legend']:
        values = getattr(args, arg_name)
        if values is not None and len(values) != len(args.path_results):
            parser.error(f'-{arg_name.replace("_", "-")} must have the same number of entries as -path-results')
    dataset_style = get_dataset_style(dataset_labels, args.dataset_colors, args.dataset_linestyles)
    if args.dataset_individual is not None:
        INDIVIDUAL_DATASETS = {ds for ds, ind in zip(dataset_labels, args.dataset_individual) if ind}
        # Per-subject lines need an explicit color (and line style) per dataset
        DATASET_STYLE = dataset_style or get_dataset_style(dataset_labels, None, ['solid'] * len(dataset_labels))
        dataset_style = DATASET_STYLE
    INDIVIDUAL_LINEWIDTH = args.individual_linewidth

    os.makedirs(args.path_out, exist_ok=True)

    # Load data from all paths
    print('Loading DTI CSVs...')
    all_frames = []
    for path, ds_label in zip(args.path_results, dataset_labels):
        print(f'  {path}  →  dataset="{ds_label}"')
        if path == 'none':
            # Placeholder without data (e.g. pending processing)
            all_frames.append(pd.DataFrame(columns=['participant_id', 'Slice (I->S)', 'VertLevel', 'Label',
                                                    'metric', 'value', 'std', 'dataset']))
            continue
        df = load_dti_csvs(path, ds_label)
        if args.dataset_first_level is not None:
            first_level = args.dataset_first_level[len(all_frames)]
            df = df[df['VertLevel'] >= first_level]
        if args.dataset_last_level is not None:
            df = df[df['VertLevel'] <= args.dataset_last_level[len(all_frames)]]
        n_subjects = df['participant_id'].nunique()
        print(f'    {n_subjects} subject(s), {len(df)} rows loaded')
        all_frames.append(df)
    df_all = pd.concat(all_frames, ignore_index=True)

    # Compare datasets on the same subjects
    if len(args.path_results) > 1 and args.common_subjects:
        df_all = keep_common_subjects(df_all, args.labels_to_plot)

    # Merge sex from participants.tsv when provided
    if args.participant_file is not None:
        df_participants = pd.read_csv(args.participant_file, sep='\t')
        if 'sex' in df_participants.columns:
            df_all = df_all.merge(
                df_participants[['participant_id', 'sex']],
                on='participant_id', how='left'
            )
            n_with_sex = df_all['participant_id'][df_all['sex'].notna()].nunique()
            print(f'  Sex merged for {n_with_sex} subject(s)')
        else:
            print('  Warning: participants.tsv has no "sex" column — per-sex figures skipped')

    default_hue = 'dataset' if df_all['dataset'].nunique() > 1 else None

    has_sex = 'sex' in df_all.columns and df_all['sex'].notna().any()

    # One figure per DTI metric, one subplot per ROI
    if 'per-metric' in args.figure_type:
        rois = load_pam50_rois(os.path.expanduser(args.path_pam50), args.labels_to_plot) if args.roi_insets else None
        for metric in args.metrics:
            print(f'\nPlotting: {metric}')
            create_lineplot_dti(df_all, metric, args.labels_to_plot, args.path_out, hue=default_hue,
                                dataset_style=dataset_style, rois=rois)
            if has_sex:
                print(f'  Plotting per-sex: {metric}')
                create_lineplot_dti(df_all, metric, args.labels_to_plot, args.path_out, hue='sex')

    # One figure per ROI, one subplot per DTI metric
    if 'per-label' in args.figure_type:
        for label in args.labels_to_plot:
            print(f'\nPlotting: {label}')
            create_lineplot_dti_per_label(df_all, label, args.path_out, hue=default_hue,
                                          dataset_style=dataset_style)
            if has_sex:
                print(f'  Plotting per-sex: {label}')
                create_lineplot_dti_per_label(df_all, label, args.path_out, hue='sex')

    # Single figure, one 2x2 block (DTI metrics) per ROI
    if 'per-label-grid' in args.figure_type:
        print('\nPlotting: one 2x2 block per label')
        rois = load_pam50_rois(os.path.expanduser(args.path_pam50), args.labels_to_plot) if args.roi_insets else None
        create_lineplot_dti_per_label_grid(df_all, args.labels_to_plot, args.path_out, rois, args.grid_ncols)

    # All ROIs overlaid in a single axis: one figure per DTI metric and/or a single 2x2 figure
    if 'all-labels' in args.figure_type or 'all-labels-grid' in args.figure_type:
        if df_all['dataset'].nunique() > 1:
            parser.error('-figure-type all-labels/all-labels-grid supports a single -path-results only')
    if 'all-labels' in args.figure_type:
        for metric in args.metrics:
            print(f'\nPlotting all labels: {metric}')
            create_lineplot_dti_all_labels(df_all, metric, args.labels_to_plot, args.path_out)
    if 'all-labels-grid' in args.figure_type:
        print('\nPlotting all labels: all metrics')
        create_lineplot_dti_all_labels_grid(df_all, args.labels_to_plot, args.path_out)

    # One subplot per dataset, each against the reference (first -path-results entry)
    if 'per-dataset-grid' in args.figure_type:
        if len(dataset_labels) < 2:
            parser.error('-figure-type per-dataset-grid needs at least two -path-results')
        panels = args.dataset_panel or list(range(1, len(dataset_labels)))
        if len(panels) != len(dataset_labels) - 1:
            parser.error('-dataset-panel must have one entry less than the number of -path-results')
        for label in args.labels_to_plot:
            for metric in args.metrics:
                print(f'\nPlotting per-dataset grid: {metric}, {label}')
                create_lineplot_per_dataset_grid(df_all, metric, label, args.path_out, dataset_labels,
                                                 panels, args.grid_ncols, dataset_style=dataset_style,
                                                 footnote=args.footnote,
                                                 flags=args.dataset_flags[1:] if args.dataset_flags else None,
                                                 legend_labels=(args.dataset_legend[1:] if args.dataset_legend
                                                                else None),
                                                 legend_loc=args.legend_loc)

    print('\nDone.')


if __name__ == '__main__':
    main()
