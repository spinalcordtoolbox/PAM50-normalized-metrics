#!/usr/bin/env python3
"""
Create participants.tsv for the AOMIC dataset.

AOMIC (Amsterdam Open MRI Collection) comprises three sub-datasets stored under
spinal_cord/AOMIC/ with distinct subject-ID prefixes:

  sub-ID1000xNNNN  → ds003097 v1.2.1  (N=928 healthy young adults)
  sub-PIOP1xNNNN   → ds002785         (N=216 healthy young adults)
  sub-PIOP2xNNNN   → ds002790         (N=226 healthy young adults)

Demographics are read from an external demographics CSV (Kurt's file) when
provided via --demog-file.  If omitted, the script falls back to fetching each
sub-dataset's participants.tsv from OpenNeuro and mapping IDs to the local
naming convention.

Usage:
    python code/create_AOMIC_participants.py --demog-file /path/to/filtered_spinalcord_csa.csv
    python code/create_AOMIC_participants.py          # OpenNeuro fallback

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


SEX_MAP  = {0.0: "F", 1.0: "M", 0: "F", 1: "M"}
DIAG_MAP = {"cn": "CN"}

# OpenNeuro sub-dataset config used only for the fallback path
SUBDATASETS = [
    {"dataset_id": "ds003097", "version": "1.2.1", "local_prefix": "ID1000",
     "sex_map": {"female": "F", "male": "M"}},
    {"dataset_id": "ds002785", "version": None,     "local_prefix": "PIOP1",
     "sex_map": {"F": "F", "M": "M"}},
    {"dataset_id": "ds002790", "version": None,     "local_prefix": "PIOP2",
     "sex_map": {"F": "F", "M": "M"}},
]

OUT_COLS = ['participant_id', 'sex', 'age', 'pathology', 'handedness', 'scanner', 'race', 'weight', 'height', 'BMI']


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--demog-file',
                        help="Path to filtered_spinalcord_csa.csv (Kurt's file); "
                             "used as the primary source for demographics.")
    return parser


def get_csv_subjects(dataset_dir: Path) -> set:
    subjects = set()
    for f in dataset_dir.iterdir():
        if f.name.startswith("sub-") and f.suffix == ".csv":
            subjects.add(f.stem.split("_")[0])
    return subjects


def fetch_openneuro(dataset_id: str, version: str | None) -> pd.DataFrame:
    ref = f"refs/tags/{version}" if version else "main"
    url = f"https://raw.githubusercontent.com/OpenNeuroDatasets/{dataset_id}/{ref}/participants.tsv"
    print(f"  Fetching {url} ...")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text), sep="\t", dtype=str)


def from_kurt(demog_file: str, csv_subjects: set) -> pd.DataFrame:
    """Build participants DataFrame from Kurt's demographics CSV."""
    df_kurt = pd.read_csv(demog_file, low_memory=False)
    df_kurt = df_kurt[df_kurt["dataset"] == "AOMIC"].rename(columns={"subject": "participant_id"})
    df_kurt = df_kurt.drop_duplicates("participant_id")
    print(f"  Kurt CSV: {len(df_kurt)} AOMIC rows")

    before = len(df_kurt)
    df_kurt = df_kurt[df_kurt["participant_id"].isin(csv_subjects)].reset_index(drop=True)
    print(f"  Kept {len(df_kurt)} / {before} rows (those with CSV files)")

    return pd.DataFrame({
        "participant_id": df_kurt["participant_id"],
        "sex":            df_kurt["sex"].map(SEX_MAP).fillna("n/a"),
        "age":            pd.to_numeric(df_kurt["age"], errors="coerce"),
        "pathology":      df_kurt["diagnosis"].map(DIAG_MAP).fillna("CN"),
        "handedness":     df_kurt["handedness"].fillna("n/a") if "handedness" in df_kurt.columns else "n/a",
        "scanner":        "n/a",
        "race":           "n/a",
        "weight":         "n/a",
        "height":         "n/a",
        "BMI":            "n/a",
    })


def from_openneuro(csv_subjects: set) -> pd.DataFrame:
    """Build participants DataFrame by fetching from OpenNeuro (fallback)."""
    all_rows = []
    for cfg in SUBDATASETS:
        prefix = cfg["local_prefix"]
        try:
            df = fetch_openneuro(cfg["dataset_id"], cfg["version"])
        except requests.RequestException as e:
            sys.exit(f"ERROR fetching {cfg['dataset_id']}: {e}")

        df["participant_id"] = df["participant_id"].str.replace(
            r"^sub-(\d+)$",
            lambda m: f"sub-{prefix}x{m.group(1).zfill(4)}",
            regex=True,
        )
        before = len(df)
        df = df[df["participant_id"].isin(csv_subjects)].reset_index(drop=True)
        print(f"  {cfg['dataset_id']}: kept {len(df)} / {before} rows")

        df["sex"] = df["sex"].map(cfg["sex_map"]).fillna("n/a")
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
        df["pathology"]  = "CN"
        df["handedness"] = "n/a"
        df["scanner"]    = "n/a"
        df["race"]       = "n/a"
        df["weight"]     = "n/a"
        df["height"]     = "n/a"
        df["BMI"]        = "n/a"
        all_rows.append(df[OUT_COLS])

    return pd.concat(all_rows, ignore_index=True).sort_values("participant_id")


def main():
    args = get_parser().parse_args()

    repo_root   = Path(__file__).resolve().parent.parent
    dataset_dir = repo_root / "spinal_cord" / "AOMIC"

    if not dataset_dir.exists():
        sys.exit(f"ERROR: Directory not found: {dataset_dir}")

    csv_subjects = get_csv_subjects(dataset_dir)
    print(f"Found {len(csv_subjects)} subjects with CSV files in {dataset_dir.relative_to(repo_root)}")

    if args.demog_file:
        out = from_kurt(args.demog_file, csv_subjects)
    else:
        print("No --demog-file provided; falling back to OpenNeuro.")
        out = from_openneuro(csv_subjects)

    out = out.sort_values("participant_id").reset_index(drop=True)

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
