#!/usr/bin/env python3
"""
Create participants.tsv for the TempleSocial dataset.

TempleSocial is a single-session fMRI dataset on social and non-social reward
processing from Temple University (OpenNeuro ds005123).

When --demog-file is provided, Kurt's CSV is used as the primary source for
sex, age, pathology, and race.  Weight, height, and BMI are always fetched
from OpenNeuro (ds005123 v1.1.3) because they are not present in Kurt's CSV.
Without --demog-file all demographics come from OpenNeuro.

Usage:
    python code/create_TempleSocial_participants.py --demog-file /path/to/filtered_spinalcord_csa.csv
    python code/create_TempleSocial_participants.py          # OpenNeuro only

Requirements:
    pip install pandas requests
"""

import sys
import json
import argparse
import requests
import pandas as pd
from pathlib import Path
from io import StringIO
from participants_json_template import make_participants_json


DATASET_ID = "ds005123"
VERSION    = "1.1.3"
PARTICIPANTS_URL = (
    f"https://raw.githubusercontent.com/OpenNeuroDatasets/{DATASET_ID}/"
    f"refs/tags/{VERSION}/participants.tsv"
)

SEX_MAP  = {0.0: "F", 1.0: "M", 0: "F", 1: "M"}
DIAG_MAP = {"cn": "CN"}

OUT_COLS = ['participant_id', 'sex', 'age', 'pathology',
            'handedness', 'scanner', 'race', 'weight', 'height', 'BMI']


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--demog-file',
                        help="Path to filtered_spinalcord_csa.csv (Kurt's file); "
                             "primary source for sex, age, pathology, and race.")
    return parser


def get_csv_subjects(dataset_dir: Path) -> set:
    subjects = set()
    for f in dataset_dir.iterdir():
        if f.name.startswith("sub-") and f.suffix == ".csv":
            subjects.add(f.stem.split("_")[0])
    return subjects


def fetch_openneuro(url: str) -> pd.DataFrame:
    print(f"Fetching {url} ...")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text), sep="\t", dtype=str)


def main():
    args = get_parser().parse_args()

    repo_root   = Path(__file__).resolve().parent.parent
    dataset_dir = repo_root / "spinal_cord" / "TempleSocial"

    if not dataset_dir.exists():
        sys.exit(f"ERROR: Directory not found: {dataset_dir}")

    csv_subjects = get_csv_subjects(dataset_dir)
    print(f"Found {len(csv_subjects)} subjects with CSV files in {dataset_dir.relative_to(repo_root)}")

    # Always fetch OpenNeuro for weight, height, BMI
    try:
        df_on = fetch_openneuro(PARTICIPANTS_URL)
    except requests.RequestException as e:
        sys.exit(f"ERROR: Could not fetch participants.tsv: {e}")
    print(f"  {len(df_on)} rows fetched from OpenNeuro")
    df_on = df_on[df_on["participant_id"].isin(csv_subjects)].reset_index(drop=True)

    on_extra = pd.DataFrame({
        "participant_id": df_on["participant_id"],
        "sex_on":         df_on["sex"].fillna("n/a"),
        "age_on":         pd.to_numeric(df_on["age"],    errors="coerce"),
        "race_on":        df_on["race"].fillna("n/a") if "race" in df_on.columns else "n/a",
        "weight":         pd.to_numeric(df_on["weight"], errors="coerce"),
        "height":         pd.to_numeric(df_on["height"], errors="coerce"),
        "BMI":            pd.to_numeric(df_on["BMI"],    errors="coerce"),
    })

    if args.demog_file:
        # Kurt's CSV → primary source for sex, age, pathology, race
        df_kurt = pd.read_csv(args.demog_file, low_memory=False)
        df_kurt = df_kurt[df_kurt["dataset"] == "TempleSocial"].rename(columns={"subject": "participant_id"})
        df_kurt = df_kurt.drop_duplicates("participant_id")
        print(f"  Kurt CSV: {len(df_kurt)} TempleSocial rows")
        df_kurt = df_kurt[df_kurt["participant_id"].isin(csv_subjects)].reset_index(drop=True)
        print(f"  Kept {len(df_kurt)} rows (those with CSV files)")

        kurt_demog = pd.DataFrame({
            "participant_id": df_kurt["participant_id"],
            "sex":            df_kurt["sex"].map(SEX_MAP).fillna("n/a"),
            "age":            pd.to_numeric(df_kurt["age"], errors="coerce"),
            "pathology":      df_kurt["diagnosis"].map(DIAG_MAP).fillna("CN"),
            "handedness":     "n/a",
            "scanner":        "n/a",
            "race":           df_kurt["race"].fillna("n/a") if "race" in df_kurt.columns else "n/a",
        })

        # Merge: Kurt's demog + OpenNeuro weight/height/BMI
        out = kurt_demog.merge(
            on_extra[["participant_id", "weight", "height", "BMI"]],
            on="participant_id", how="left"
        )
    else:
        print("No --demog-file provided; using OpenNeuro for all demographics.")
        out = pd.DataFrame({
            "participant_id": on_extra["participant_id"],
            "sex":            on_extra["sex_on"],
            "age":            on_extra["age_on"],
            "pathology":      "CN",
            "handedness":     "n/a",
            "scanner":        "n/a",
            "race":           on_extra["race_on"],
            "weight":         on_extra["weight"],
            "height":         on_extra["height"],
            "BMI":            on_extra["BMI"],
        })

    out = out[OUT_COLS].sort_values("participant_id").reset_index(drop=True)

    out_tsv = dataset_dir / "participants.tsv"
    out.to_csv(out_tsv, sep="	", index=False, na_rep="n/a")
    print(f"\nWritten {len(out)} rows → {out_tsv.relative_to(repo_root)}")

    out_json = dataset_dir / "participants.json"
    with open(out_json, "w") as f:
        json.dump(make_participants_json(
            OUT_COLS,
            pathology_levels={'CN': 'Cognitively Normal'}
        ), f, indent=4)
    print(f"Written: {out_json.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
