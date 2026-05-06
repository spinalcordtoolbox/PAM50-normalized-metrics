#!/usr/bin/env python3
"""
Create participants.tsv for the UCLA LA5c dataset.

UCLA LA5c is the Consortium for Neuropsychiatric Phenomics dataset (OpenNeuro ds000030)
including healthy controls and participants diagnosed with schizophrenia, bipolar
disorder, or ADHD.

The script:
  1. Fetches the participants.tsv from OpenNeuro (ds000030 v1.0.0) to get the list
     of available subjects with their sex and age
  2. Retains only rows whose participant_id has a corresponding CSV file in
     spinal_cord/UCLA_LA5c/
  3. Optionally merges additional demographics (handedness, scanner, race, diagnosis)
     from an external CSV file (e.g. filtered_spinalcord_csa.csv from Kurt)
  4. Writes participants.tsv and participants.json to spinal_cord/UCLA_LA5c/

Usage:
    python code/create_UCLA_LA5c_participants.py
    python code/create_UCLA_LA5c_participants.py --demog-file /path/to/filtered_spinalcord_csa.csv

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


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_ID = "ds000030"
VERSION    = "1.0.0"

PARTICIPANTS_URL = (
    f"https://raw.githubusercontent.com/OpenNeuroDatasets/{DATASET_ID}/"
    f"refs/tags/{VERSION}/participants.tsv"
)

DIAG_MAP = {"cn": "CN", "schz": "SCHZ", "bipolar": "bipolar", "adhd": "ADHD"}
SEX_MAP  = {0.0: "F", 1.0: "M", 0: "F", 1: "M"}

# ---------------------------------------------------------------------------


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--demog-file',
                        help="Path to filtered_spinalcord_csa.csv from Kurt; "
                             "used as the primary source for sex, age, diagnosis, "
                             "handedness, scanner, and race.")
    return parser


def fetch_participants(url: str) -> pd.DataFrame:
    """Download and parse the participants.tsv from OpenNeuro."""
    print(f"Fetching {url} ...")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return pd.read_csv(StringIO(response.text), sep="\t", dtype=str)


def get_csv_subjects(dataset_dir: Path) -> set:
    """Return set of participant_ids that have a CSV file."""
    subjects = set()
    for f in dataset_dir.iterdir():
        if f.name.startswith("sub-") and f.suffix == ".csv":
            subjects.add(f.stem.split("_")[0])
    return subjects


def main():
    args = get_parser().parse_args()

    repo_root   = Path(__file__).resolve().parent.parent
    dataset_dir = repo_root / "spinal_cord" / "UCLA_LA5c"

    if not dataset_dir.exists():
        sys.exit(f"ERROR: Directory not found: {dataset_dir}")

    # 1. Fetch participants from OpenNeuro (for subject list + sex/age fallback)
    try:
        df_openneuro = fetch_participants(PARTICIPANTS_URL)
    except requests.RequestException as e:
        sys.exit(f"ERROR: Could not fetch participants.tsv: {e}")

    print(f"  {len(df_openneuro)} rows fetched, "
          f"{df_openneuro['participant_id'].nunique()} unique participants")

    # 2. Filter to subjects with CSV files
    csv_subjects = get_csv_subjects(dataset_dir)
    before = len(df_openneuro)
    df_openneuro = df_openneuro[df_openneuro["participant_id"].isin(csv_subjects)].reset_index(drop=True)
    print(f"  Kept {len(df_openneuro)} / {before} rows "
          f"(those with CSV files in {dataset_dir.relative_to(repo_root)})")

    # OpenNeuro sex column is 'gender' with F/M values; diagnosis is CONTROL/SCHZ/…
    openneuro_diag_map = {"CONTROL": "CN", "SCHZ": "SCHZ", "BIPOLAR": "bipolar", "ADHD": "ADHD"}
    df_openneuro["sex"]       = df_openneuro["gender"].fillna("n/a")
    df_openneuro["age"]       = pd.to_numeric(df_openneuro["age"], errors="coerce")
    df_openneuro["pathology"] = df_openneuro["diagnosis"].map(openneuro_diag_map).fillna("n/a")
    df_openneuro["handedness"] = "n/a"
    df_openneuro["scanner"]    = "n/a"
    df_openneuro["race"]       = "n/a"
    df_openneuro["weight"]     = "n/a"
    df_openneuro["height"]     = "n/a"
    df_openneuro["BMI"]        = "n/a"

    out = df_openneuro[["participant_id", "sex", "age", "pathology",
                         "handedness", "scanner", "race", "weight", "height", "BMI"]].copy()

    # 3. Optionally merge from Kurt's CSV (primary source)
    if args.demog_file:
        demog_path = Path(args.demog_file)
        if not demog_path.exists():
            sys.exit(f"ERROR: Demographics file not found: {demog_path}")

        df_kurt = pd.read_csv(demog_path, low_memory=False)
        df_kurt = df_kurt[df_kurt["dataset"] == "UCLA_LA5c"].copy()
        df_kurt = df_kurt.rename(columns={"subject": "participant_id"})
        df_kurt = df_kurt.drop_duplicates("participant_id")
        print(f"  Kurt CSV: {len(df_kurt)} UCLA_LA5c rows")

        df_kurt["sex_kurt"]       = df_kurt["sex"].map(SEX_MAP).fillna("n/a")
        df_kurt["age_kurt"]       = pd.to_numeric(df_kurt["age"], errors="coerce")
        df_kurt["pathology_kurt"] = df_kurt["diagnosis"].map(DIAG_MAP).fillna("n/a")
        df_kurt["handedness_kurt"] = df_kurt["handedness"].fillna("n/a") \
            if "handedness" in df_kurt.columns else "n/a"
        df_kurt["scanner_kurt"] = df_kurt["scanner"].fillna("n/a") \
            if "scanner" in df_kurt.columns else "n/a"
        df_kurt["race_kurt"] = df_kurt["race"].fillna("n/a") \
            if "race" in df_kurt.columns else "n/a"

        merged = out.merge(
            df_kurt[["participant_id", "sex_kurt", "age_kurt", "pathology_kurt",
                     "handedness_kurt", "scanner_kurt", "race_kurt"]],
            on="participant_id", how="left"
        )
        matched = merged["sex_kurt"].notna().sum()
        print(f"  Matched {matched} / {len(merged)} subjects to Kurt CSV")

        out["sex"]       = merged["sex_kurt"].fillna(out["sex"])
        out["age"]       = merged["age_kurt"].combine_first(out["age"])
        out["pathology"] = merged["pathology_kurt"].fillna(out["pathology"])
        out["handedness"] = merged["handedness_kurt"].fillna("n/a")
        out["scanner"]    = merged["scanner_kurt"].fillna("n/a")
        out["race"]       = merged["race_kurt"].fillna("n/a")

    # 4. Write participants.tsv
    out_tsv = dataset_dir / "participants.tsv"
    out.to_csv(out_tsv, sep="\t", index=False)
    print(f"\nWritten {len(out)} rows → {out_tsv.relative_to(repo_root)}")
    print(f"  pathology value counts: {out['pathology'].value_counts().to_dict()}")

    # 5. Write participants.json
    out_json = dataset_dir / "participants.json"
    with open(out_json, "w") as f:
        json.dump(make_participants_json(
            ['participant_id', 'sex', 'age', 'pathology', 'handedness', 'scanner', 'race', 'weight', 'height', 'BMI'],
            pathology_levels={
                'CN':      'Cognitively Normal',
                'SCHZ':    'Schizophrenia',
                'bipolar': 'Bipolar Disorder',
                'ADHD':    'Attention-Deficit/Hyperactivity Disorder',
            }
        ), f, indent=4)
    print(f"Written: {out_json.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
