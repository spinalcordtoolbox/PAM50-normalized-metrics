#!/usr/bin/env python3
"""
Create participants.tsv for the MASiVar dataset.

The script:
  1. Fetches the participants.tsv from OpenNeuro (ds003416 v2.0.2)
  2. Retains only rows whose (participant_id, session_id) pair has a
     corresponding CSV file in the spinal_cord/MASiVar/ directory
  3. Writes the result to spinal_cord/MASiVar/participants.tsv

Usage:
    python code/create_MASiVar_participants.py

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
DATASET_ID = "ds003416"
VERSION    = "2.0.2"

# OpenNeuro datasets are mirrored on GitHub; fetch raw file via tag URL
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


def get_csv_pairs(masivar_dir: Path) -> set:
    """Return (participant_id, session_id) pairs that have a CSV file."""
    pairs = set()
    for f in masivar_dir.iterdir():
        if f.name.startswith("sub-") and f.suffix == ".csv":
            parts = f.stem.split("_")
            sub = parts[0]
            ses = next((p for p in parts if p.startswith("ses-")), None)
            if ses:
                pairs.add((sub, ses))
    return pairs


def main():
    repo_root   = Path(__file__).resolve().parent.parent
    masivar_dir = repo_root / "spinal_cord" / "MASiVar"

    if not masivar_dir.exists():
        sys.exit(f"ERROR: Directory not found: {masivar_dir}")

    # 1. Fetch participants from OpenNeuro
    try:
        df = fetch_participants(PARTICIPANTS_URL)
    except requests.RequestException as e:
        sys.exit(f"ERROR: Could not fetch participants.tsv: {e}")

    print(f"  {len(df)} session-level rows, "
          f"{df['participant_id'].nunique()} unique participants")

    # 2. Filter to rows with a matching CSV file
    csv_pairs = get_csv_pairs(masivar_dir)
    before = len(df)
    df = df[
        df.apply(lambda r: (r["participant_id"], r["session_id"]) in csv_pairs, axis=1)
    ].reset_index(drop=True)
    print(f"  Kept {len(df)} / {before} rows "
          f"(those with CSV files in {masivar_dir.relative_to(repo_root)})")

    # 3. Map columns to output format
    sex_map = {"male": "M", "female": "F"}
    out = pd.DataFrame({
        "participant_id": df["participant_id"],
        "session_id":     df["session_id"],
        "sex":            df["sex"].map(sex_map).fillna("n/a"),
        "age":            pd.to_numeric(df["age"], errors="coerce"),
    })

    # 4. Write participants.tsv
    out_tsv = masivar_dir / "participants.tsv"
    out.to_csv(out_tsv, sep="\t", index=False)
    print(f"  Written: {out_tsv.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
