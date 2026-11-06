"""
Flag vertebral levels missing morphometric values despite valid data both above and below,
either entirely blank (no slice in the level holds a value) or partial (only some slices missing).

Usage:
    python find_empty_slices.py [path ...]

Arguments:
    path        CSV file(s) or directory(ies) of CSV files.
                (default: ../spinal_cord, i.e. every spinal cord dataset when run from code/)

Example:
    python find_empty_slices.py
    python find_empty_slices.py >> empty_slices_2026-07-15.log
    python find_empty_slices.py ../spinal_cord/BLSA
"""

import argparse
import glob
import os
import pandas as pd

# Only C1 to T1 are analyzed (range plotted by statistics/generate_figures.py)
MAX_VERT_LEVEL = 8


def level_name(level):
    """Convert a VertLevel number to a label (8 -> T1, 5 -> C5)."""
    return f'T{level - 7}' if level > 7 else f'C{level}'


def find_level_gaps(csv_path, metric='MEAN(area)'):
    """Return (gaps, context) for one CSV, where `gaps` is a list of (level, n_valid, n_rows)
    for vertebral levels missing some or all values of `metric` while lying inside the
    subject's imaged span, and `context` is a list of (level, n_valid, n_rows) for every level
    in that span. A level with n_valid == 0 is entirely blank, otherwise it is partial."""
    df = pd.read_csv(csv_path).sort_values('Slice (I->S)')
    df = df[(df['VertLevel'] >= 1) & (df['VertLevel'] <= MAX_VERT_LEVEL)].reset_index(drop=True)
    if df.empty or metric not in df.columns:
        return [], []

    valid = (df[metric].notna() & (df[metric] > 0)).to_numpy()
    if not valid.any():
        return [], []

    # Imaged span = first to last slice carrying a value; anything outside is uncovered template
    first, last = valid.argmax(), len(valid) - 1 - valid[::-1].argmax()
    covered = df.iloc[first:last + 1]

    gaps, context = [], []
    # Loop across levels
    for level, group in covered.groupby('VertLevel'):
        n_valid = int((group[metric].notna() & (group[metric] > 0)).sum())
        context.append((level, n_valid, len(group)))
        if n_valid < len(group):
            gaps.append((level, n_valid, len(group)))
    return gaps, context


def collect_files(paths):
    """Get a sorted list of CSV files."""
    files = []
    for path in paths:
        path = os.path.expanduser(path)
        if os.path.isfile(path):
            files.append(path)
        elif os.path.isdir(path):
            found = sorted(glob.glob(os.path.join(path, 'sub*.csv')))
            if not found:
                found = sorted(glob.glob(os.path.join(path, '*', 'sub*.csv')))
            files.extend(found)
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('path', nargs='*', default=['../spinal_cord'],
                        help='CSV file(s) or directory(ies) (default: ../spinal_cord)')
    args = parser.parse_args()

    files = collect_files(args.path)
    if not files:
        print(f"No CSV files found in: {', '.join(args.path)}")
        return

    entries = []
    for csv_path in files:
        gaps, context = find_level_gaps(csv_path)
        if not gaps:
            continue
        dataset = os.path.basename(os.path.dirname(os.path.abspath(csv_path)))
        print(f"\n{dataset} / {os.path.basename(csv_path)}")
        for level, n_valid, n_rows in gaps:
            status = 'blank' if n_valid == 0 else 'partial'
            if status == 'blank':
                print(f" - VertLevel {level} ({level_name(level)}) (n={n_rows} slices) blank")
            else:
                print(f" - VertLevel {level} ({level_name(level)}) "
                      f"(n={n_rows - n_valid} of {n_rows} slices missing) partial")
            entries.append({
                'dataset': dataset,
                'file': os.path.basename(csv_path),
                'VertLevel': level,
                'level': level_name(level),
                'status': status,
                'n_missing': n_rows - n_valid,
                'n_slices': n_rows,
            })
        # Show valid/total slices per level across the imaged span, so a gap can be seen in
        # context of the adjacent levels (e.g. C5 33/33 | C6 0/31 | C7 34/34)
        print('    ' + ' | '.join(f'{level_name(lvl)} {n_valid}/{n_rows}' for lvl, n_valid, n_rows in context))

    if not entries:
        print(f"\nNo missing morphometrics found in {len(files)} files.")
        return

    # Classify each file by its most severe finding, so a file holding an entirely blank level
    # is not also counted as partial
    df_entries = pd.DataFrame(entries)
    per_file = (df_entries.groupby(['dataset', 'file'])['status']
                .agg(lambda s: 'blank' if 'blank' in set(s) else 'partial').reset_index())

    n_blank = int((per_file['status'] == 'blank').sum())
    n_partial = int((per_file['status'] == 'partial').sum())
    print(f"\n{len(per_file)} of {len(files)} files miss morphometrics within the imaged span: "
          f"{n_blank} with an entirely blank vertebral level, {n_partial} with partial gaps only")

    summary = per_file.pivot_table(index='dataset', columns='status', values='file',
                                   aggfunc='count', fill_value=0)
    print(f"\nFlagged files per dataset:\n{summary.to_string()}")


if __name__ == '__main__':
    main()
