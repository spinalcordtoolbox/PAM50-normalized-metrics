"""
Plot FA in PAM50 space for a single subject: warped vs. interpolation.

Usage:
    python compare_FA_PAM50_single_subject.py \
        -path-warp    <path>/dwi_PAM50/sub-XX_dwi_FA_PAM50.csv \
        -path-interp  <path>/dwi_interpolation_to_PAM50/sub-XX_dwi_FA_interpolated_to_PAM50.csv \
        -path-out     <path>/figures/sub-XX_compare_FA_PAM50.png \
        -subject      sub-amu01
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

LABELS_FONT_SIZE = 20
TICKS_FONT_SIZE = LABELS_FONT_SIZE - 2

XLIM_VERT_RANGE = (2, 5)

DATASET_LABELS = ('Warped', 'Interpolated')


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


def plot(df_direct, df_interp, subject, output, label):
    df_direct = df_direct[df_direct['Label'] == label].sort_values('Slice (I->S)')
    df_interp = df_interp[df_interp['Label'] == label].sort_values('Slice (I->S)')
    metric_col = get_metric_col(df_direct)

    # Long-format dataframe with a 'dataset' column for seaborn hue/style.
    df_direct = df_direct.assign(dataset=DATASET_LABELS[0])
    df_interp = df_interp.assign(dataset=DATASET_LABELS[1])
    df = pd.concat([df_direct, df_interp], ignore_index=True)
    df = df.rename(columns={metric_col: 'value'})

    # Vertebral annotations + slice range from the direct trace (reference).
    disc_slices, mid_slices, vert_at_mid = get_vert_indices(df_direct)
    ref_in_range = df_direct[df_direct['VertLevel'].between(*XLIM_VERT_RANGE)]
    xlim_min = ref_in_range['Slice (I->S)'].min()
    xlim_max = ref_in_range['Slice (I->S)'].max()
    df = df[df['Slice (I->S)'].between(xlim_min, xlim_max)]

    mpl.rcParams['font.family'] = 'Arial'
    # Match a single subplot size from generate_figures_dti.py (6 wide × 5 tall).
    fig, ax = plt.subplots(figsize=(6, 5))

    palette = sns.color_palette('tab10', n_colors=2)
    # style=dataset varies linestyle (1st solid, 2nd dashed); alpha lets overlapping
    # traces be seen through one another.
    sns.lineplot(ax=ax, x='Slice (I->S)', y='value', data=df,
                 hue='dataset', style='dataset', linewidth=2,
                 palette=palette, alpha=0.7)

    ax.set_title(label.title(), fontsize=LABELS_FONT_SIZE)
    ax.set_xlabel('Axial Slice #', fontsize=LABELS_FONT_SIZE)
    ax.set_ylabel('Fractional Anisotropy [a.u.]', fontsize=LABELS_FONT_SIZE)
    style_ax(ax)
    ax.set_xlim(xlim_max, xlim_min)  # inverted x-axis (rostral on the left)

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

    suptitle = f'Fractional Anisotropy [a.u.], {subject}' if subject else 'Fractional Anisotropy [a.u.]'
    fig.suptitle(suptitle, fontsize=LABELS_FONT_SIZE + 2, y=1.01)
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"Saved: {output}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-path-warp', required=True,
                        help='CSV with FA warped directly to PAM50.')
    parser.add_argument('-path-interp', required=True,
                        help='CSV with native FA interpolated to PAM50.')
    parser.add_argument('-path-out', required=True, help='Output PNG path.')
    parser.add_argument('-subject', default='', help='Subject label for the plot title.')
    parser.add_argument('-label', default='white matter',
                        help='Tract/region label to extract (default: "white matter").')
    args = parser.parse_args()

    df_direct = pd.read_csv(args.path_warp)
    df_interp = pd.read_csv(args.path_interp)
    plot(df_direct, df_interp, args.subject, args.path_out, args.label)


if __name__ == '__main__':
    main()
