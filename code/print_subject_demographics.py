#!/usr/bin/env python3
"""
Print demographics for a given subject from the filtered_spinalcord_csa.csv file.

Usage:
    python code/print_subject_demographics.py --demog-file <path/to/filtered_spinalcord_csa.csv> --dataset BLSA --subject sub-BLSA0270
"""

import argparse
import pandas as pd


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--demog-file', required=True,
                        help="Path to filtered_spinalcord_csa.csv")
    parser.add_argument('--dataset', required=True,
                        help="Dataset name (e.g. BLSA, AOMIC, UCLA_LA5c)")
    parser.add_argument('--subject', required=True,
                        help="Subject ID (e.g. sub-BLSA0270)")
    return parser


def main():
    args = get_parser().parse_args()

    df = pd.read_csv(args.demog_file, low_memory=False)
    ds = df[df['dataset'] == args.dataset]
    row = ds[ds['subject'] == args.subject]

    if row.empty:
        print(f"No rows found for dataset='{args.dataset}', subject='{args.subject}'")
        return

    cols = ['subject', 'session', 'age', 'sex', 'handedness', 'scanner', 'race', 'diagnosis']
    cols = [c for c in cols if c in row.columns]
    print(row[cols].to_string(index=False))


if __name__ == '__main__':
    main()
