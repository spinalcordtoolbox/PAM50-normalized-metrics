#!/usr/bin/env python3
#
# Axial PAM50 T2 slice(s) at mid-vertebral level(s) overlaid with the PAM50 atlas ROIs used for the DTI
# curves (generate_figures_dti_PAM50_all_subjects.py), one color per ROI. ROIs are split into panels
# (-panels), e.g. WM + GM in one panel and the WM tracts in another.
#
# ROIs follow sct_extract_metric -combine 1: a combined label (e.g. "white matter") or comma-separated
# atlas IDs (e.g. "4,5") are the sum of the corresponding probabilistic atlas maps. ROIs are kept
# probabilistic: the overlay opacity is proportional to the atlas probability, and voxels below
# -threshold are not shown. Within a panel, ROIs are drawn in the given order (later on top).
#
# Requires the SCT environment (venv_sct), e.g.:
#     ~/code/sct_latest/python/envs/venv_sct/bin/python statistics/generate_figure_PAM50_rois_axial.py \
#         -path-pam50 ~/code/sct_latest/data/PAM50 -levels 3 -path-out <path>/PAM50_rois_axial_C3.png
#
# Author: Jan Valosek
#

import os
import textwrap
import argparse
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from spinalcordtoolbox.image import Image

# Shared with the group-level script (same folder)
from generate_figures_dti_PAM50_all_subjects import LABEL_DISPLAY_NAMES, LABEL_PALETTE, vert_num_to_label

DEFAULT_FONT_SIZE = 14
FONT_SIZE = DEFAULT_FONT_SIZE
# Gamma applied to the T2 template intensities (< 1 → brighter)
BACKGROUND_GAMMA = 0.7


def get_parser():
    parser = argparse.ArgumentParser(
        description='Axial PAM50 T2 slice(s) overlaid with probabilistic PAM50 atlas ROIs.')
    parser.add_argument('-path-pam50', default='~/code/sct_latest/data/PAM50',
                        help='PAM50 folder (with template/ and atlas/). Default: ~/code/sct_latest/data/PAM50')
    parser.add_argument('-panels', nargs='+',
                        default=['white matter;gray matter', 'dorsal columns;lateral funiculi;ventral funiculi',
                                 '4,5;12,13'],
                        help='ROIs per panel, separated by ";" (one panel per argument). ROIs: combined label '
                             'names from atlas/info_label.txt or comma-separated atlas IDs. '
                             'Default: "white matter;gray matter" "dorsal columns;lateral funiculi;'
                             'ventral funiculi" "4,5;12,13".')
    parser.add_argument('-levels', nargs='+', type=int, default=[3],
                        help='Vertebral levels (C1=1 … C7=7); the middle slice of each level is shown, one '
                             'row per level. Default: 3.')
    parser.add_argument('-threshold', type=float, default=0.5,
                        help='Atlas probability below which ROI voxels are not shown. Default: 0.5.')
    parser.add_argument('-font-size', type=int, default=FONT_SIZE,
                        help=f'Font size of the orientation labels; legends and title are 2 pt larger. '
                             f'Default: {FONT_SIZE}.')
    parser.add_argument('-path-out', required=True, help='Output PNG path.')
    return parser


def read_info_label(path_atlas):
    """Return {atlas ID: file} and {combined label name: [atlas IDs]} from atlas/info_label.txt."""
    files, combined, section = {}, {}, None
    for line in open(os.path.join(path_atlas, 'info_label.txt')):
        line = line.strip()
        if line.startswith('# Keyword='):
            section = line.split('=')[1].split()[0]
            continue
        if not line or line.startswith('#'):
            continue
        parts = [p.strip() for p in line.split(',')]
        if section == 'IndivLabels':
            files[int(parts[0])] = parts[-1]
        elif section == 'CombinedLabels':
            first, last = map(int, parts[-1].split(':'))
            combined[parts[1]] = list(range(first, last + 1))
    return files, combined


def main():
    global FONT_SIZE
    args = get_parser().parse_args()
    FONT_SIZE = args.font_size
    path_pam50 = os.path.expanduser(args.path_pam50)
    path_out = os.path.expanduser(args.path_out)
    path_atlas = os.path.join(path_pam50, 'atlas')
    files, combined = read_info_label(path_atlas)
    panels = [[lab.strip() for lab in p.split(';')] for p in args.panels]

    def load(fname):
        return Image(fname).change_orientation('RPI').data

    t2 = load(os.path.join(path_pam50, 'template', 'PAM50_t2.nii.gz'))
    levels = load(os.path.join(path_pam50, 'template', 'PAM50_levels.nii.gz'))
    cord = load(os.path.join(path_pam50, 'template', 'PAM50_cord.nii.gz'))
    # Probabilistic ROIs (sum of the atlas probability maps, clipped to 1)
    atlas, rois = {}, {}
    for label in {lab for p in panels for lab in p}:
        ids = combined[label] if label in combined else [int(i) for i in label.split(',')]
        for i in ids:
            if i not in atlas:
                atlas[i] = load(os.path.join(path_atlas, files[i]))
        rois[label] = np.clip(sum(atlas[i] for i in ids), 0, 1)

    # Middle slice of each vertebral level (RPI: 3rd axis inferior → superior)
    z_mid = {}
    for level in args.levels:
        z_level = np.unique(np.where(levels == level)[2])
        if z_level.size == 0:
            raise SystemExit(f'Level {level} not found in PAM50_levels.nii.gz')
        z_mid[level] = int(z_level[len(z_level) // 2])
    # Same crop size for all levels (largest cord extent + margin for the CSF), centered on the cord,
    # so that all slices are shown at the same scale
    margin = 12
    half_x = max(np.ptp(np.where(cord[:, :, z] > 0.5)[0]) for z in z_mid.values()) // 2 + margin
    half_y = max(np.ptp(np.where(cord[:, :, z] > 0.5)[1]) for z in z_mid.values()) // 2 + margin

    mpl.rcParams['font.family'] = 'Arial'
    nrows, ncols = len(args.levels), len(panels)
    # Larger figure for fonts larger than the default (grows with the square root of the font size ratio,
    # so that the text still gets relatively larger)
    scale = max(1.0, (FONT_SIZE / DEFAULT_FONT_SIZE) ** 0.5)
    # Figure height from the crop aspect ratio, so that there is no empty space around the panels
    aspect = (2 * half_y + 1) / (2 * half_x + 1)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols * scale, 4.2 * aspect * nrows * scale + 0.6), squeeze=False)
    for row, level in enumerate(args.levels):
        z = z_mid[level]
        xs, ys = np.where(cord[:, :, z] > 0.5)
        xc, yc = (xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2
        x0, x1 = xc - half_x, xc + half_x + 1
        y0, y1 = yc - half_y, yc + half_y + 1
        for col, panel in enumerate(panels):
            ax = axes[row, col]
            # x (R → L) as columns, y (P → A) as rows with origin='lower' → anterior up
            # Brighter background: gamma < 1 brightens the dark cord and the surrounding tissue
            t2_crop = t2[x0:x1, y0:y1, z].astype(float)
            t2_crop = (t2_crop / t2_crop.max()) ** BACKGROUND_GAMMA
            ax.imshow(t2_crop.T, cmap='gray', origin='lower', interpolation='bilinear', vmin=0, vmax=1)
            for label in panel:
                prob = rois[label][x0:x1, y0:y1, z].T
                rgba = np.zeros(prob.shape + (4,))
                rgba[..., :3] = mpl.colors.to_rgb(LABEL_PALETTE.get(label, 'red'))
                # Opacity = atlas probability; voxels below the threshold are not shown
                rgba[..., 3] = np.where(prob >= args.threshold, prob, 0)
                ax.imshow(rgba, origin='lower', interpolation='nearest')
            # Orientation labels only in the first panel
            if row == 0 and col == 0:
                for x, y, txt, ha, va in [(0.03, 0.5, 'R', 'left', 'center'), (0.97, 0.5, 'L', 'right', 'center'),
                                          (0.5, 0.97, 'A', 'center', 'top'), (0.5, 0.03, 'P', 'center', 'bottom')]:
                    ax.text(x, y, txt, transform=ax.transAxes, color='white', fontsize=FONT_SIZE,
                            fontweight='bold', ha=ha, va=va)
            ax.axis('off')
            if nrows > 1 and col == 0:
                ax.text(-0.04, 0.5, vert_num_to_label(level), transform=ax.transAxes, rotation=90,
                        fontsize=FONT_SIZE + 2, ha='right', va='center')
            # Per-panel legend below the panel (only under the last row)
            if row == nrows - 1:
                handles = [Patch(facecolor=LABEL_PALETTE.get(lab, 'red'), edgecolor='none') for lab in panel]
                # Long ROI names wrapped to two lines, so that legends of neighboring panels do not overlap
                names = [LABEL_DISPLAY_NAMES.get(lab, lab) for lab in panel]
                ax.legend(handles, [textwrap.fill(name, 28) for name in names],
                          loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=1,
                          fontsize=FONT_SIZE + 2, frameon=False)

    level_str = ', '.join(vert_num_to_label(lv) for lv in args.levels)
    fig.suptitle(f'PAM50 atlas ROIs, mid-{level_str} (probability ≥ {args.threshold})', fontsize=FONT_SIZE + 4)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path_out)), exist_ok=True)
    plt.savefig(path_out, dpi=300, bbox_inches='tight')
    print(f'Figure saved: {path_out}')


if __name__ == '__main__':
    main()
