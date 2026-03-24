"""
Print the first 5 slices with non-zero CSA at the highest vertebral level for each CSV file in a directory.
Usage: python print_last_nonzero_csa.py [directory]
       python print_last_nonzero_csa.py path/to/file.csv
Default directory: current working directory
"""

import sys
import glob
import os
import pandas as pd


def print_last_nonzero_csa(csv_path, n=5):
    df = pd.read_csv(csv_path)
    non_zero = df[df['MEAN(area)'].notna() & (df['MEAN(area)'] > 0)]
    print(f"\n{os.path.basename(csv_path)}")
    if non_zero.empty:
        print("  No non-zero CSA rows found.")
    else:
        max_level = non_zero['VertLevel'].max()
        level_rows = non_zero[non_zero['VertLevel'] == max_level]
        print(f"  VertLevel: {max_level}")
        print(level_rows[['Slice (I->S)', 'VertLevel', 'MEAN(area)']].head(n).to_string(index=False))


if __name__ == '__main__':
    target = sys.argv[1] if len(sys.argv) > 1 else '.'

    if os.path.isfile(target):
        files = [target]
    else:
        files = sorted(glob.glob(os.path.join(target, '*.csv')))
        if not files:
            print(f"No CSV files found in: {target}")
            sys.exit(1)

    for f in files:
        print_last_nonzero_csa(f)
