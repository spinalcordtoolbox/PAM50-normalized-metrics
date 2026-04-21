#!/usr/bin/env python3
"""
Create participants.tsv for the KurtMS dataset.

KurtMS is a longitudinal cervical-spine T1w dataset with two participant groups:
  sub-BMSHC* — healthy controls (HC)
  sub-BMSMS* — multiple sclerosis patients (MS)

Sex and age are not publicly available for this dataset, so they are set to n/a.
Pathology is derived from the participant ID prefix.

The script:
  1. Scans spinal_cord/KurtMS/ for CSV files to determine available
     (participant_id, session_id) pairs
  2. Derives pathology from the participant ID (BMSHC → HC, BMSMS → MS)
  3. Writes the result to spinal_cord/KurtMS/participants.tsv

Usage:
    python code/create_KurtMS_participants.py
"""

import sys
import pandas as pd
from pathlib import Path


def get_csv_pairs(dataset_dir: Path) -> list[tuple[str, str]]:
    """Return sorted (participant_id, session_id) pairs that have a CSV file."""
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
    """Derive pathology from participant ID prefix (BMSHC → HC, BMSMS → MS)."""
    if "BMSHC" in participant_id:
        return "HC"
    elif "BMSMS" in participant_id:
        return "MS"
    return "n/a"


def main():
    repo_root   = Path(__file__).resolve().parent.parent
    dataset_dir = repo_root / "spinal_cord" / "KurtMS"

    if not dataset_dir.exists():
        sys.exit(f"ERROR: Directory not found: {dataset_dir}")

    pairs = get_csv_pairs(dataset_dir)
    print(f"Found {len(pairs)} (participant_id, session_id) pairs with CSV files "
          f"in {dataset_dir.relative_to(repo_root)}")

    rows = []
    for sub, ses in pairs:
        rows.append({
            "participant_id": sub,
            "session_id":     ses,
            "sex":            "n/a",
            "age":            "n/a",
            "pathology":      derive_pathology(sub),
        })

    df = pd.DataFrame(rows)

    out_tsv = dataset_dir / "participants.tsv"
    df.to_csv(out_tsv, sep="\t", index=False)
    print(f"Written {len(df)} rows → {out_tsv.relative_to(repo_root)}")
    print(f"  pathology value counts: {df['pathology'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
