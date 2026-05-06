#!/usr/bin/env python3
"""
Create participants.tsv for the VUIIS-MS dataset.

VUIIS-MS is a longitudinal cervical-spine T1w dataset with two participant groups:
  sub-BMSHC* — healthy controls (HC)
  sub-BMSMS* — multiple sclerosis patients (MS)

When --demog-file is provided, demographics (sex, age, scanner) are read from
Kurt's dataset-specific CSV.  Handedness, race, weight, height, and BMI are not
available for this dataset.

Usage:
    python code/create_VUIIS-MS_participants.py --demog-file /path/to/kurtms_spinalcord_csa_with_demographics.csv
    python code/create_VUIIS-MS_participants.py          # pathology from subject ID only

Requirements:
    pip install pandas
"""

import sys
import json
import argparse
import pandas as pd
from pathlib import Path
from participants_json_template import make_participants_json


OUT_COLS = ['participant_id', 'session_id', 'sex', 'age', 'pathology',
            'handedness', 'scanner', 'race', 'weight', 'height', 'BMI']

DIAG_MAP = {"HC": "HC", "MS": "MS"}


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--demog-file',
                        help="Path to kurtms_spinalcord_csa_with_demographics.csv (Kurt's file); "
                             "primary source for sex, age, and scanner.")
    return parser


def get_csv_pairs(dataset_dir: Path) -> list[tuple[str, str]]:
    pairs = []
    for f in sorted(dataset_dir.iterdir()):
        if f.name.startswith("sub-") and f.suffix == ".csv":
            parts = f.stem.split("_")
            sub = parts[0]
            ses = next((p for p in parts if p.startswith("ses-")), None)
            if ses:
                pairs.append((sub, ses))
    return pairs


def derive_pathology(participant_id: str) -> str:
    if "BMSHC" in participant_id:
        return "HC"
    elif "BMSMS" in participant_id:
        return "MS"
    return "n/a"


def main():
    args = get_parser().parse_args()

    repo_root   = Path(__file__).resolve().parent.parent
    dataset_dir = repo_root / "spinal_cord" / "VUIIS-MS"

    if not dataset_dir.exists():
        sys.exit(f"ERROR: Directory not found: {dataset_dir}")

    pairs = get_csv_pairs(dataset_dir)
    print(f"Found {len(pairs)} (participant_id, session_id) pairs with CSV files "
          f"in {dataset_dir.relative_to(repo_root)}")

    if args.demog_file:
        demog_path = Path(args.demog_file)
        if not demog_path.exists():
            sys.exit(f"ERROR: Demographics file not found: {demog_path}")

        df_kurt = pd.read_csv(demog_path, low_memory=False)
        df_kurt = df_kurt.rename(columns={"subject": "participant_id", "session": "session_id"})
        print(f"  Kurt CSV: {len(df_kurt)} rows")

        pair_set = set(pairs)
        df_kurt = df_kurt[
            df_kurt.apply(lambda r: (r["participant_id"], r["session_id"]) in pair_set, axis=1)
        ].reset_index(drop=True)
        print(f"  Matched {len(df_kurt)} rows to CSV files")

        rows = pd.DataFrame({
            "participant_id": df_kurt["participant_id"],
            "session_id":     df_kurt["session_id"],
            "sex":            df_kurt["sex"].fillna("n/a"),
            "age":            pd.to_numeric(df_kurt["age"], errors="coerce"),
            "pathology":      df_kurt["diagnosis"].map(DIAG_MAP).fillna("n/a"),
            "handedness":     "n/a",
            "scanner":        "n/a",
            "race":           "n/a",
            "weight":         "n/a",
            "height":         "n/a",
            "BMI":            "n/a",
        })
    else:
        rows = pd.DataFrame([
            {
                "participant_id": sub,
                "session_id":     ses,
                "sex":            "n/a",
                "age":            "n/a",
                "pathology":      derive_pathology(sub),
                "handedness":     "n/a",
                "scanner":        "n/a",
                "race":           "n/a",
                "weight":         "n/a",
                "height":         "n/a",
                "BMI":            "n/a",
            }
            for sub, ses in pairs
        ], columns=OUT_COLS)

    df = rows[OUT_COLS].sort_values(["participant_id", "session_id"]).reset_index(drop=True)

    out_tsv = dataset_dir / "participants.tsv"
    df.to_csv(out_tsv, sep="\t", index=False, na_rep="n/a")
    print(f"Written {len(df)} rows → {out_tsv.relative_to(repo_root)}")
    print(f"  pathology value counts: {df['pathology'].value_counts().to_dict()}")

    out_json = dataset_dir / "participants.json"
    with open(out_json, "w") as f:
        json.dump(make_participants_json(
            OUT_COLS,
            pathology_levels={'HC': 'Healthy Control', 'MS': 'Multiple Sclerosis'}
        ), f, indent=4)
    print(f"Written: {out_json.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
