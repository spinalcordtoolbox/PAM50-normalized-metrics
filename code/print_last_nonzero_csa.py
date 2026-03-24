"""
Print the first N slices with non-zero CSA at the highest vertebral level for each CSV file,
and flag vertebral levels where within-level CSA SD exceeds a threshold.

Sweep mode (--sweep) iterates over increasing numbers of excluded caudal slices (0, 2, 4, ...)
and saves a separate QC log for each, to help identify the optimal exclusion count.

Usage:
    python print_last_nonzero_csa.py [path] [--n-slices N] [--std-threshold T] [--log FILE]
    python print_last_nonzero_csa.py [path] [--std-threshold T] [--sweep] [--sweep-max M] [--log FILE]

Arguments:
    path            CSV file or directory of CSV files (default: current directory)
    --n-slices      Number of slices to print at the most caudal level (default: 10)
    --std-threshold Flag vertebral levels with within-level CSA SD above this value
                    in mm^2 (default: 5)
    --log           Path to write QC log CSV (optional); in sweep mode used as base name
    --sweep         Sweep over excluded caudal slice counts and save one log per count
    --sweep-max     Maximum number of caudal slices to exclude in sweep (default: 10)
"""

import argparse
import glob
import os
import pandas as pd


def compute_qc_flags(non_zero, exclude_caudal, std_threshold):
    """Return list of (level, sd, n_slices) for levels exceeding the SD threshold,
    after dropping the first `exclude_caudal` slices from the most caudal level."""
    data = non_zero.copy()
    if exclude_caudal > 0:
        max_level = data['VertLevel'].max()
        caudal_idx = data[data['VertLevel'] == max_level].index[:exclude_caudal]
        data = data.drop(index=caudal_idx)

    flagged = []
    for level, group in data.groupby('VertLevel'):
        std = group['MEAN(area)'].std()
        if std > std_threshold:
            flagged.append((level, round(std, 2), len(group)))
    return flagged


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

    # Within-level SD QC
    for level, std, n_slices in compute_qc_flags(non_zero, exclude_caudal=0, std_threshold=std_threshold):
        print(f" ⚠️ [QC] VertLevel {level}: SD={std:.2f} mm^2 (n={n_slices} slices)")
        if log_entries is not None:
            log_entries.append({
                'file': os.path.basename(csv_path),
                'VertLevel': level,
                'SD': std,
                'n_slices': n_slices,
            })


def run_sweep(files, std_threshold, sweep_max, log_base):
    """For each exclusion count (0, 2, 4, ..., sweep_max), compute QC flags across all
    files and save a summary log. Prints a summary table to stdout."""
    base, ext = os.path.splitext(log_base) if log_base else ('qc_sweep', '.csv')
    if not ext:
        ext = '.csv'

    print(f"\n{'Excluded slices':>16}  {'Flagged files':>13}  {'Total flags':>11}  {'Log'}")
    print("-" * 70)

    for exclude in range(0, sweep_max + 1, 2):
        entries = []
        for csv_path in files:
            df = pd.read_csv(csv_path)
            non_zero = df[df['MEAN(area)'].notna() & (df['MEAN(area)'] > 0)]
            if non_zero.empty:
                continue
            for level, std, n_slices in compute_qc_flags(non_zero, exclude_caudal=exclude,
                                                          std_threshold=std_threshold):
                entries.append({
                    'file': os.path.basename(csv_path),
                    'VertLevel': level,
                    'SD': std,
                    'n_slices': n_slices,
                    'excluded_slices': exclude,
                })

        log_path = f"{base}_exclude{exclude:02d}{ext}"
        log_df = pd.DataFrame(entries, columns=['file', 'VertLevel', 'SD', 'n_slices', 'excluded_slices'])
        log_df.to_csv(log_path, index=False)

        n_flagged_files = log_df['file'].nunique() if not log_df.empty else 0
        print(f"{exclude:>16}  {n_flagged_files:>13}  {len(entries):>11}  {log_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('path', nargs='?', default='.', help='CSV file or directory (default: .)')
    parser.add_argument('--n-slices', type=int, default=10,
                        help='Number of slices to print (default: 10). '
                             'Note that the slices are normalized to the PAM50 spinal cord template space and have '
                             '0.5mm slice thickness, so 10 slices correspond to 5mm of spinal cord length.')
    parser.add_argument('--std-threshold', type=float, default=5.0,
                        help='Flag levels with within-level CSA SD above this value in mm^2 (default: 5)')
    parser.add_argument('--log', type=str, default=None,
                        help='Path to write QC log CSV; in sweep mode used as base name (optional)')
    parser.add_argument('--sweep', action='store_true',
                        help='Sweep over excluded caudal slice counts and save one log per count')
    parser.add_argument('--sweep-max', type=int, default=20,
                        help='Maximum number of caudal slices to exclude in sweep (default: 20)')
    args = parser.parse_args()

    if os.path.isfile(args.path):
        files = [args.path]
    else:
        files = sorted(glob.glob(os.path.join(args.path, '*.csv')))
        if not files:
            print(f"No CSV files found in: {args.path}")
            return

    if args.sweep:
        run_sweep(files, std_threshold=args.std_threshold, sweep_max=args.sweep_max,
                  log_base=args.log or 'qc_sweep')
    else:
        log_entries = [] if args.log else None
        for f in files:
            print_last_nonzero_csa(f, n=args.n_slices, std_threshold=args.std_threshold, log_entries=log_entries)

        if args.log and log_entries is not None:
            log_df = pd.DataFrame(log_entries, columns=['file', 'VertLevel', 'SD', 'n_slices'])
            log_df.to_csv(args.log, index=False)
            print(f"\nQC log written to: {args.log} ({len(log_entries)} flagged entries)")


if __name__ == '__main__':
    main()
