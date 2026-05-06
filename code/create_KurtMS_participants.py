#!/usr/bin/env python3
"""
Create participants.tsv for the KurtMS dataset.

KurtMS is a longitudinal cervical-spine T1w dataset with two participant groups:
  sub-BMSHC* — healthy controls (HC)
  sub-BMSMS* — multiple sclerosis patients (MS)

Sex, age, handedness, scanner, and race are not publicly available for this
dataset and are set to n/a.  Pathology is derived from the participant ID prefix.
Kurt's demographics CSV contains no data for this dataset.

The script:
  1. Scans spinal_cord/KurtMS/ for CSV files to determine available
     (participant_id, session_id) pairs
  2. Derives pathology from the participant ID (BMSHC → HC, BMSMS → MS)
  3. Writes participants.tsv and participants.json to spinal_cord/KurtMS/

Usage:
    python code/create_KurtMS_participants.py
"""

import sys
import json
import pandas as pd
from pathlib import Path
from participants_json_template import make_participants_json


OUT_COLS = ['participant_id', 'session_id', 'sex', 'age', 'pathology', 'handedness', 'scanner', 'race', 'weight', 'height', 'BMI']


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
            "handedness":     "n/a",
            "scanner":        "n/a",
            "race":           "n/a",
            "weight":         "n/a",
            "height":         "n/a",
            "BMI":            "n/a",
        })

    df = pd.DataFrame(rows, columns=OUT_COLS)

    out_tsv = dataset_dir / "participants.tsv"
    df.to_csv(out_tsv, sep="\t", index=False)
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
