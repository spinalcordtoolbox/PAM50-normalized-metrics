#!/usr/bin/env python3
"""
Create participants.tsv for the TempleSocial dataset.

TempleSocial is a single-session fMRI dataset on social and non-social reward
processing from Temple University (OpenNeuro ds005123).

The script:
  1. Fetches the participants.tsv from OpenNeuro (ds005123 v1.1.3)
  2. Retains only rows whose participant_id has a corresponding CSV file in
     spinal_cord/TempleSocial/
  3. Writes the result to spinal_cord/TempleSocial/participants.tsv

Usage:
    python code/create_TempleSocial_participants.py

Requirements:
    pip install pandas requests
"""

import sys
import requests
import pandas as pd
from pathlib import Path
from io import StringIO


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_ID = "ds005123"
VERSION    = "1.1.3"

PARTICIPANTS_URL = (
    f"https://raw.githubusercontent.com/OpenNeuroDatasets/{DATASET_ID}/"
    f"refs/tags/{VERSION}/participants.tsv"
)

# ---------------------------------------------------------------------------


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
    repo_root    = Path(__file__).resolve().parent.parent
    dataset_dir  = repo_root / "spinal_cord" / "TempleSocial"

    if not dataset_dir.exists():
        sys.exit(f"ERROR: Directory not found: {dataset_dir}")

    # 1. Fetch participants from OpenNeuro
    try:
        df = fetch_participants(PARTICIPANTS_URL)
    except requests.RequestException as e:
        sys.exit(f"ERROR: Could not fetch participants.tsv: {e}")

    print(f"  {len(df)} rows fetched, "
          f"{df['participant_id'].nunique()} unique participants")

    # 2. Filter to subjects with CSV files
    csv_subjects = get_csv_subjects(dataset_dir)
    before = len(df)
    df = df[df["participant_id"].isin(csv_subjects)].reset_index(drop=True)
    print(f"  Kept {len(df)} / {before} rows "
          f"(those with CSV files in {dataset_dir.relative_to(repo_root)})")

    # 3. Map to output format (sex already F/M; age, weight, height, BMI are numeric)
    out = pd.DataFrame({
        "participant_id": df["participant_id"],
        "sex":            df["sex"].fillna("n/a"),
        "age":            pd.to_numeric(df["age"],    errors="coerce"),
        "race":           df["race"].fillna("n/a"),
        "weight":         pd.to_numeric(df["weight"], errors="coerce"),
        "height":         pd.to_numeric(df["height"], errors="coerce"),
        "BMI":            pd.to_numeric(df["BMI"],    errors="coerce"),
    })

    # 4. Write participants.tsv
    out_tsv = dataset_dir / "participants.tsv"
    out.to_csv(out_tsv, sep="\t", index=False)
    print(f"\nWritten {len(out)} rows → {out_tsv.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
