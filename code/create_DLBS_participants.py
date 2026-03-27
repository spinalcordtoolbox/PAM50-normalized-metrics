#!/usr/bin/env python3
"""
Create participants.tsv and dataset_description.json for the DLBS dataset.

DLBS (Dallas Lifespan Brain Study) is a longitudinal dataset with up to three
imaging waves. Subjects are identified as sub-XXXX and sessions as ses-wave1,
ses-wave2, ses-wave3. The OpenNeuro participants.tsv is subject-level and
stores wave-specific MRI ages in columns AgeMRI_W1, AgeMRI_W2, AgeMRI_W3.

The script:
  1. Fetches the participants.tsv from OpenNeuro (ds004856 v1.3.0)
  2. Expands the subject-level table into session-level rows
  3. Retains only rows whose (participant_id, session_id) pair has a
     corresponding CSV file in spinal_cord/DLBS/
  4. Writes the result to spinal_cord/DLBS/participants.tsv
  5. Writes spinal_cord/DLBS/dataset_description.json

Usage:
    python code/create_DLBS_participants.py

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
DATASET_ID = "ds004856"
VERSION    = "1.3.0"

PARTICIPANTS_URL = (
    f"https://raw.githubusercontent.com/OpenNeuroDatasets/{DATASET_ID}/"
    f"refs/tags/{VERSION}/participants.tsv"
)

# Mapping from wave suffix (in OpenNeuro column names) to BIDS session label
WAVE_TO_SESSION = {
    "W1": "ses-wave1",
    "W2": "ses-wave2",
    "W3": "ses-wave3",
}

DATASET_DESCRIPTION = {
    "name": "DLBS",
    "order": 6,
    "coverage": "cervical spine",
    "contrast": "T1w",
    "resolution": "n/a",
    "link": f"https://openneuro.org/datasets/{DATASET_ID}",
    "link_text": f"openneuro/{DATASET_ID}",
    "population": "healthy adults",
}
# ---------------------------------------------------------------------------


def fetch_participants(url: str) -> pd.DataFrame:
    """Download and parse the participants.tsv from OpenNeuro."""
    print(f"Fetching {url} ...")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return pd.read_csv(StringIO(response.text), sep="\t", dtype=str)


def get_csv_pairs(dlbs_dir: Path) -> set:
    """Return (participant_id, session_id) pairs that have a CSV file."""
    pairs = set()
    for f in dlbs_dir.iterdir():
        if f.name.startswith("sub-") and f.suffix == ".csv":
            parts = f.stem.split("_")
            sub = parts[0]
            ses = next((p for p in parts if p.startswith("ses-")), None)
            if ses:
                pairs.add((sub, ses))
    return pairs


def main():
    repo_root = Path(__file__).resolve().parent.parent
    dlbs_dir  = repo_root / "spinal_cord" / "DLBS"

    if not dlbs_dir.exists():
        sys.exit(f"ERROR: Directory not found: {dlbs_dir}")

    # 1. Fetch participants from OpenNeuro
    try:
        df = fetch_participants(PARTICIPANTS_URL)
    except requests.RequestException as e:
        sys.exit(f"ERROR: Could not fetch participants.tsv: {e}")

    print(f"  {len(df)} subject-level rows fetched")

    # 2. Expand to session-level rows
    sex_map = {"m": "M", "f": "F"}
    rows = []
    for _, row in df.iterrows():
        pid = row["participant_id"]
        sex = sex_map.get(str(row.get("Sex", "")).lower(), "n/a")
        for wave, session in WAVE_TO_SESSION.items():
            age_col = f"AgeMRI_{wave}"
            age = pd.to_numeric(row.get(age_col, "n/a"), errors="coerce")
            if not pd.isna(age):  # only add session if MRI age exists for that wave
                rows.append({
                    "participant_id": pid,
                    "session_id":     session,
                    "sex":            sex,
                    "age":            age,
                    "height":         pd.to_numeric(row.get(f"Height_{wave}", "n/a"), errors="coerce"),
                    "weight":         pd.to_numeric(row.get(f"Weight_{wave}", "n/a"), errors="coerce"),
                    "BMI":            pd.to_numeric(row.get(f"BMI_{wave}",    "n/a"), errors="coerce"),
                })

    session_df = pd.DataFrame(rows)
    print(f"  Expanded to {len(session_df)} session-level rows")

    # 3. Filter to rows with a matching CSV file
    csv_pairs = get_csv_pairs(dlbs_dir)
    before = len(session_df)
    session_df = session_df[
        session_df.apply(lambda r: (r["participant_id"], r["session_id"]) in csv_pairs, axis=1)
    ].reset_index(drop=True)
    print(f"  Kept {len(session_df)} / {before} rows "
          f"(those with CSV files in {dlbs_dir.relative_to(repo_root)})")

    # 4. Write participants.tsv
    out_tsv = dlbs_dir / "participants.tsv"
    session_df.to_csv(out_tsv, sep="\t", index=False)
    print(f"\nWritten {len(session_df)} rows → {out_tsv.relative_to(repo_root)}")

    # 5. Write dataset_description.json (preserve existing order field)
    out_json = dlbs_dir / "dataset_description.json"
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
