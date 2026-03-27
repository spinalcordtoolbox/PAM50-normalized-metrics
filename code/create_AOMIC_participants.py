#!/usr/bin/env python3
"""
Create participants.tsv for the AOMIC dataset.

AOMIC (Amsterdam Open MRI Collection) comprises three sub-datasets that are
all stored under spinal_cord/AOMIC/ with distinct subject-ID prefixes:

  sub-ID1000xNNNN  → ds003097 v1.2.1  (N=928 healthy young adults)
  sub-PIOP1xNNNN   → ds002785         (N=216 healthy young adults)
  sub-PIOP2xNNNN   → ds002790         (N=226 healthy young adults)

The script:
  1. Fetches the participants.tsv for each sub-dataset from OpenNeuro (via GitHub mirror)
  2. Maps participant IDs to the naming used in the local CSV files
  3. Retains only rows whose participant_id has a corresponding CSV file in
     spinal_cord/AOMIC/
  4. Combines the three sub-datasets and writes participants.tsv

Usage:
    python code/create_AOMIC_participants.py

Requirements:
    pip install pandas requests
"""

import sys
import requests
import pandas as pd
from pathlib import Path
from io import StringIO


# ---------------------------------------------------------------------------
# Configuration – (dataset_id, github_ref, local_prefix, sex_map)
# ---------------------------------------------------------------------------
SUBDATASETS = [
    {
        "dataset_id": "ds003097",
        "version": "1.2.1",
        "local_prefix": "ID1000",
        "sex_map": {"female": "F", "male": "M"},
    },
    {
        "dataset_id": "ds002785",
        "version": None,   # use main branch
        "local_prefix": "PIOP1",
        "sex_map": {"F": "F", "M": "M"},
    },
    {
        "dataset_id": "ds002790",
        "version": None,   # use main branch
        "local_prefix": "PIOP2",
        "sex_map": {"F": "F", "M": "M"},
    },
]

# ---------------------------------------------------------------------------


def fetch_participants(dataset_id: str, version: str | None) -> pd.DataFrame:
    """Download and parse a participants.tsv from OpenNeuro via GitHub mirror."""
    ref = f"refs/tags/{version}" if version else "main"
    url = f"https://raw.githubusercontent.com/OpenNeuroDatasets/{dataset_id}/{ref}/participants.tsv"
    print(f"  Fetching {url} ...")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return pd.read_csv(StringIO(response.text), sep="\t", dtype=str)


def get_csv_subjects(aomic_dir: Path) -> set:
    """Return set of participant_ids that have a CSV file."""
    subjects = set()
    for f in aomic_dir.iterdir():
        if f.name.startswith("sub-") and f.suffix == ".csv":
            subjects.add(f.stem.split("_")[0])
    return subjects


def main():
    repo_root = Path(__file__).resolve().parent.parent
    aomic_dir = repo_root / "spinal_cord" / "AOMIC"

    if not aomic_dir.exists():
        sys.exit(f"ERROR: Directory not found: {aomic_dir}")

    csv_subjects = get_csv_subjects(aomic_dir)
    print(f"Found {len(csv_subjects)} subjects with CSV files in {aomic_dir.relative_to(repo_root)}")

    all_rows = []

    for cfg in SUBDATASETS:
        prefix = cfg["local_prefix"]
        try:
            df = fetch_participants(cfg["dataset_id"], cfg["version"])
        except requests.RequestException as e:
            sys.exit(f"ERROR: Could not fetch participants.tsv for {cfg['dataset_id']}: {e}")

        print(f"  {cfg['dataset_id']}: {len(df)} rows fetched")

        # Map OpenNeuro sub-NNNN → local sub-<prefix>xNNNN
        # IDs in OpenNeuro are like sub-0001; local IDs are like sub-ID1000x0001
        df["participant_id"] = df["participant_id"].str.replace(
            r"^sub-(\d+)$", lambda m: f"sub-{prefix}x{m.group(1).zfill(4)}", regex=True
        )

        # Filter to subjects with CSV files
        before = len(df)
        df = df[df["participant_id"].isin(csv_subjects)].reset_index(drop=True)
        print(f"  {cfg['dataset_id']}: kept {len(df)} / {before} rows")

        # Normalise sex
        df["sex"] = df["sex"].map(cfg["sex_map"]).fillna("n/a")
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
        df["BMI"] = pd.to_numeric(df["BMI"], errors="coerce")

        all_rows.append(df[["participant_id", "sex", "age", "BMI"]])

    combined = pd.concat(all_rows, ignore_index=True).sort_values("participant_id")

    # Write participants.tsv
    out_tsv = aomic_dir / "participants.tsv"
    combined.to_csv(out_tsv, sep="\t", index=False)
    print(f"\nWritten {len(combined)} rows → {out_tsv.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
