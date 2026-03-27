"""
Compute within-subject, within-level CSA SD from a directory of per-subject CSV files.

For each subject and each vertebral level, the SD of MEAN(area) across all slices within
that level is calculated. The script then reports the distribution of these per-subject SDs
across subjects (mean, median, 75th and 95th percentile) for each vertebral level.

This is useful for choosing a data-driven --std-threshold for print_last_nonzero_csa.py.

Usage:
    python compute_within_level_csa_sd.py [directory]

Example:
    python compute_within_level_csa_sd.py ~/code/PAM50-normalized-metrics/spinal_cord/spine-generic_multi-subject

Default directory: current working directory
"""

import sys
import glob
import os
import pandas as pd


def main():
    directory =  os.path.expanduser(sys.argv[1]) if len(sys.argv) > 1 else '.'
    files = sorted(glob.glob(os.path.join(directory, '*.csv')))
    if not files:
        print(f"No CSV files found in: {directory}")
        return

    records = []
    for path in files:
        subject = os.path.basename(path)
        df = pd.read_csv(path)
        non_zero = df[df['MEAN(area)'].notna() & (df['MEAN(area)'] > 0)]
        for level, group in non_zero.groupby('VertLevel'):
            if len(group) < 2:
                continue
            records.append({
                'subject': subject,
                'VertLevel': level,
                'SD': group['MEAN(area)'].std(),
                'n_slices': len(group),
            })

    if not records:
        print("No data found.")
        return

    df_sd = pd.DataFrame(records)

    summary = df_sd.groupby('VertLevel')['SD'].agg(
        n_subjects='count',
        mean='mean',
        median='median',
        p75=lambda x: x.quantile(0.75),
        p95=lambda x: x.quantile(0.95),
        max='max',
    ).round(2)

    print(f"\nWithin-subject, within-level CSA SD (mm²) — {len(files)} subjects")
    print(f"Directory: {os.path.abspath(directory)}\n")
    print(summary.to_string())
    print(f"\nOverall median SD across all levels and subjects: {df_sd['SD'].median():.2f} mm²")
    print(f"Overall 95th percentile:                          {df_sd['SD'].quantile(0.95):.2f} mm²")


if __name__ == '__main__':
    main()
