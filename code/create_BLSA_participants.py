#!/usr/bin/env python3
"""
Create participants.tsv for the BLSA dataset.

BLSA (Baltimore Longitudinal Study of Aging) data are not publicly available
on OpenNeuro; access is granted upon request via https://www.blsa.nih.gov.
Participant demographics are therefore not included — only participant_id and
session_id are derived from the CSV filenames.

The script:
  1. Discovers all *PAM50.csv files in spinal_cord/BLSA/
  2. Extracts participant_id and session_id from each filename
  3. Writes the result to spinal_cord/BLSA/participants.tsv

Usage:
    python code/create_BLSA_participants.py

Requirements:
    pip install pandas
"""

import sys
import pandas as pd
from pathlib import Path


def main():
    repo_root = Path(__file__).resolve().parent.parent
    blsa_dir  = repo_root / "spinal_cord" / "BLSA"

    if not blsa_dir.exists():
        sys.exit(f"ERROR: Directory not found: {blsa_dir}")

    rows = []
    for f in sorted(blsa_dir.glob("sub-*_PAM50.csv")):
        parts = f.stem.split("_")
        sub = parts[0]
        ses = next((p for p in parts if p.startswith("ses-")), None)
        rows.append({"participant_id": sub, "session_id": ses})

    if not rows:
        sys.exit("ERROR: No PAM50 CSV files found.")

    df = pd.DataFrame(rows).drop_duplicates()
    print(f"  {df['participant_id'].nunique()} unique participants, {len(df)} sessions")

    out_tsv = blsa_dir / "participants.tsv"
    df.to_csv(out_tsv, sep="\t", index=False)
    print(f"  Written: {out_tsv.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
