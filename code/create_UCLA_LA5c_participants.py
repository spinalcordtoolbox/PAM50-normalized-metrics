#!/usr/bin/env python3
"""
Create participants.tsv and dataset_description.json for the UCLA LA5c dataset.

UCLA LA5c is the Consortium for Neuropsychiatric Phenomics dataset (OpenNeuro ds000030)
including healthy controls and participants diagnosed with schizophrenia, bipolar
disorder, or ADHD.

The script:
  1. Fetches the participants.tsv from OpenNeuro (ds000030 v1.0.0)
  2. Retains only rows whose participant_id has a corresponding CSV file in
     spinal_cord/UCLA_LA5c/
  3. Writes the result to spinal_cord/UCLA_LA5c/participants.tsv
  4. Writes spinal_cord/UCLA_LA5c/dataset_description.json

Usage:
    python code/create_UCLA_LA5c_participants.py

Requirements:
    pip install pandas requests
"""

import json
import sys
import requests
import pandas as pd
from pathlib import Path
from io import StringIO


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_ID = "ds000030"
VERSION    = "1.0.0"

PARTICIPANTS_URL = (
    f"https://raw.githubusercontent.com/OpenNeuroDatasets/{DATASET_ID}/"
    f"refs/tags/{VERSION}/participants.tsv"
)

DATASET_DESCRIPTION = {
    "name": "UCLA LA5c",
    "order": 9,
    "coverage": "cervical spine",
    "contrast": "T1w",
    "resolution": "1.0mm iso",
    "link": f"https://openneuro.org/datasets/{DATASET_ID}",
    "link_text": f"openneuro/{DATASET_ID}",
    "population": "healthy controls and neuropsychiatric patients",
}
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
    repo_root   = Path(__file__).resolve().parent.parent
    dataset_dir = repo_root / "spinal_cord" / "UCLA_LA5c"

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

    # 3. Map to output format
    # Sex column is 'gender' with values 'F'/'M'
    out = pd.DataFrame({
        "participant_id": df["participant_id"],
        "sex":            df["gender"].fillna("n/a"),
        "age":            pd.to_numeric(df["age"], errors="coerce"),
        "diagnosis":      df["diagnosis"].fillna("n/a"),
    })

    # 4. Write participants.tsv
    out_tsv = dataset_dir / "participants.tsv"
    out.to_csv(out_tsv, sep="\t", index=False)
    print(f"\nWritten {len(out)} rows → {out_tsv.relative_to(repo_root)}")

    # 5. Write dataset_description.json (preserve existing order field)
    out_json = dataset_dir / "dataset_description.json"
    if out_json.exists():
        with open(out_json) as f:
            existing = json.load(f)
        DATASET_DESCRIPTION["order"] = existing.get("order", DATASET_DESCRIPTION["order"])
    with open(out_json, "w") as f:
        json.dump(DATASET_DESCRIPTION, f, indent=4)
        f.write("\n")
    print(f"Written: {out_json.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
