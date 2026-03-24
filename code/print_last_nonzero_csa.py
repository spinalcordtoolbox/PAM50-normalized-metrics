"""
Print the first N slices with non-zero CSA at the highest vertebral level for each CSV file,
and flag vertebral levels where within-level CSA SD exceeds a threshold.

Usage:
    python print_last_nonzero_csa.py [path] [--n-slices N] [--std-threshold T]

Arguments:
    path            CSV file or directory of CSV files (default: current directory)
    --n-slices      Number of slices to print at the most caudal level (default: 10)
    --std-threshold Flag vertebral levels with within-level CSA SD above this value
                    in mm^2 (default: 10)
    --log           Path to write QC log file (optional)
"""

import argparse
import glob
import os
import pandas as pd


def print_last_nonzero_csa(csv_path, n, std_threshold, log_entries):
    df = pd.read_csv(csv_path)
    non_zero = df[df['MEAN(area)'].notna() & (df['MEAN(area)'] > 0)]
    print(f"\n{os.path.basename(csv_path)}")
    if non_zero.empty:
        print("  No non-zero CSA rows found.")
        return

    # Print first N slices at the most caudal (highest-numbered) vertebral level
    max_level = non_zero['VertLevel'].max()
    level_rows = non_zero[non_zero['VertLevel'] == max_level]
    print(f"  Last {n} slices at most caudal level (VertLevel: {max_level}):")
    print(level_rows[['Slice (I->S)', 'VertLevel', 'MEAN(area)']].head(n).to_string(index=False))

    # Within-level SD QC: flag levels exceeding the threshold
    for level, group in non_zero.groupby('VertLevel'):
        std = group['MEAN(area)'].std()
        if std > std_threshold:
            print(f" ⚠️ [QC] VertLevel {level}: SD={std:.2f} mm^2 (n={len(group)} slices)")
            if log_entries is not None:
                log_entries.append({
                    'file': os.path.basename(csv_path),
                    'VertLevel': level,
                    'SD': round(std, 2),
                    'n_slices': len(group),
                })


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('path', nargs='?', default='.', help='CSV file or directory (default: .)')
    parser.add_argument('--n-slices', type=int, default=10, 
                        help='Number of slices to print (default: 10). '
                             'Note that the slices are normalized to the PAM50 spinal cord template space and have '
                             '0.5mm slice thickness, so 10 slices correspond to 5mm of spinal cord length.')
    parser.add_argument('--std-threshold', type=float, default=10.0,
                        help='Flag levels with within-level CSA SD above this value in mm^2 (default: 10)')
    parser.add_argument('--log', type=str, default=None,
                        help='Path to write QC log CSV (optional)')
    args = parser.parse_args()

    if os.path.isfile(args.path):
        files = [args.path]
    else:
        files = sorted(glob.glob(os.path.join(args.path, '*.csv')))
        if not files:
            print(f"No CSV files found in: {args.path}")
            return

    log_entries = [] if args.log else None
    for f in files:
        print_last_nonzero_csa(f, n=args.n_slices, std_threshold=args.std_threshold, log_entries=log_entries)

    if args.log and log_entries is not None:
        log_df = pd.DataFrame(log_entries, columns=['file', 'VertLevel', 'SD', 'n_slices'])
        log_df.to_csv(args.log, index=False)
        print(f"\nQC log written to: {args.log} ({len(log_entries)} flagged entries)")


if __name__ == '__main__':
    main()