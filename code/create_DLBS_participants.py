#!/usr/bin/env python3
"""
Create participants.tsv for the DLBS dataset.

DLBS (Dallas Lifespan Brain Study) is a longitudinal dataset with up to three
imaging waves.  Subjects are identified as sub-XXXX and sessions as
ses-wave1, ses-wave2, ses-wave3.

When --demog-file is provided, Kurt's CSV is used as the primary source for
sex, age, pathology, and race.  Height, weight, and BMI are always fetched
from OpenNeuro (ds004856 v1.3.0) because they are not present in Kurt's CSV.
Without --demog-file all demographics come from OpenNeuro.

Usage:
    python code/create_DLBS_participants.py --demog-file /path/to/filtered_spinalcord_csa.csv
    python code/create_DLBS_participants.py          # OpenNeuro only

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


DATASET_ID = "ds004856"
VERSION    = "1.3.0"
PARTICIPANTS_URL = (
    f"https://raw.githubusercontent.com/OpenNeuroDatasets/{DATASET_ID}/"
    f"refs/tags/{VERSION}/participants.tsv"
)

WAVE_TO_SESSION = {"W1": "ses-wave1", "W2": "ses-wave2", "W3": "ses-wave3"}

SEX_MAP  = {0.0: "F", 1.0: "M", 0: "F", 1: "M"}
DIAG_MAP = {"cn": "CN"}

OUT_COLS = ['participant_id', 'session_id', 'sex', 'age', 'pathology',
            'handedness', 'scanner', 'race', 'weight', 'height', 'BMI']


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--demog-file',
                        help="Path to filtered_spinalcord_csa.csv (Kurt's file); "
                             "primary source for sex, age, pathology, and race.")
    return parser


def get_csv_pairs(dataset_dir: Path) -> set:
    pairs = set()
    for f in dataset_dir.iterdir():
        if f.name.startswith("sub-") and f.suffix == ".csv":
            parts = f.stem.split("_")
            sub = parts[0]
            ses = next((p for p in parts if p.startswith("ses-")), None)
            if ses:
                pairs.add((sub, ses))
    return pairs


def fetch_openneuro(url: str) -> pd.DataFrame:
    print(f"Fetching {url} ...")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text), sep="\t", dtype=str)


def expand_openneuro(df: pd.DataFrame, csv_pairs: set) -> pd.DataFrame:
    """Expand subject-level OpenNeuro TSV into session-level rows."""
    sex_map_on = {"m": "M", "f": "F"}
    rows = []
    for _, row in df.iterrows():
        pid  = row["participant_id"]
        sex  = sex_map_on.get(str(row.get("Sex", "")).lower(), "n/a")
        race = str(row.get("Race", "n/a")).strip().lower()
        if race in ("nan", "", "n/a"):
            race = "n/a"
        for wave, session in WAVE_TO_SESSION.items():
            if (pid, session) not in csv_pairs:
                continue
            age = pd.to_numeric(row.get(f"AgeMRI_{wave}", ""), errors="coerce")
            if pd.isna(age):
                continue
            rows.append({
                "participant_id": pid,
                "session_id":     session,
                "sex":            sex,
                "age":            age,
                "pathology":      "CN",
                "handedness":     "n/a",
                "scanner":        "n/a",
                "race":           race,
                "weight":         pd.to_numeric(row.get(f"Weight_{wave}", ""), errors="coerce"),
                "height":         pd.to_numeric(row.get(f"Height_{wave}", ""), errors="coerce"),
                "BMI":            pd.to_numeric(row.get(f"BMI_{wave}",    ""), errors="coerce"),
            })
    return pd.DataFrame(rows)


def main():
    args = get_parser().parse_args()

    repo_root   = Path(__file__).resolve().parent.parent
    dataset_dir = repo_root / "spinal_cord" / "DLBS"

    if not dataset_dir.exists():
        sys.exit(f"ERROR: Directory not found: {dataset_dir}")

    csv_pairs = get_csv_pairs(dataset_dir)
    print(f"Found {len(csv_pairs)} (participant_id, session_id) pairs with CSV files "
          f"in {dataset_dir.relative_to(repo_root)}")

    # Always fetch OpenNeuro for height, weight, BMI
    try:
        df_on = fetch_openneuro(PARTICIPANTS_URL)
    except requests.RequestException as e:
        sys.exit(f"ERROR: Could not fetch participants.tsv: {e}")
    print(f"  {len(df_on)} subject-level rows fetched from OpenNeuro")
    df_on_sessions = expand_openneuro(df_on, csv_pairs)

    if args.demog_file:
        # Kurt's CSV → primary source for sex, age, pathology, race
        df_kurt = pd.read_csv(args.demog_file, low_memory=False)
        df_kurt = df_kurt[df_kurt["dataset"] == "DLBS"].rename(
            columns={"subject": "participant_id", "session": "session_id"}
        )
        print(f"  Kurt CSV: {len(df_kurt)} DLBS rows")

        df_kurt = df_kurt[
            df_kurt.apply(lambda r: (r["participant_id"], r["session_id"]) in csv_pairs, axis=1)
        ].reset_index(drop=True)
        print(f"  Matched {len(df_kurt)} rows to CSV files")

        kurt_demog = pd.DataFrame({
            "participant_id": df_kurt["participant_id"],
            "session_id":     df_kurt["session_id"],
            "sex":            df_kurt["sex"].map(SEX_MAP).fillna("n/a"),
            "age":            pd.to_numeric(df_kurt["age"], errors="coerce"),
            "pathology":      df_kurt["diagnosis"].map(DIAG_MAP).fillna("CN"),
            "handedness":     "n/a",
            "scanner":        "n/a",
            "race":           df_kurt["race"].fillna("n/a") if "race" in df_kurt.columns else "n/a",
        })

        # Merge: Kurt's demog + OpenNeuro height/weight/BMI
        out = kurt_demog.merge(
            df_on_sessions[["participant_id", "session_id", "height", "weight", "BMI"]],
            on=["participant_id", "session_id"], how="left"
        )
    else:
        print("No --demog-file provided; using OpenNeuro for all demographics.")
        out = df_on_sessions

    out = out[OUT_COLS].sort_values(["participant_id", "session_id"]).reset_index(drop=True)

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
