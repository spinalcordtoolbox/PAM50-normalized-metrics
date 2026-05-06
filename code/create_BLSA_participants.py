#!/usr/bin/env python3
"""
Create participants.tsv and participants.json for the BLSA dataset.

Note that the BLSA (Baltimore Longitudinal Study of Aging) dataset is not publicly available.

The script:
  1. Discovers all *PAM50.csv files in spinal_cord/BLSA/ to determine which
     sessions have processed data
  2. Merges with an internal demographics CSV to add
     sex, age, pathology, handedness, scanner, and race for each session
  3. Writes participants.tsv and participants.json to spinal_cord/BLSA/

Usage:
    python code/create_BLSA_participants.py --demog-file <path/to/filtered_spinalcord_csa.csv>

    Without --demog-file, only participant_id and session_id are written
    (same behaviour as the original script).

Requirements:
    pip install pandas
"""

import sys
import json
import argparse
import pandas as pd
from pathlib import Path


PARTICIPANTS_JSON = {
    "participant_id": {
        "Description": "Unique Participant ID",
        "LongName": "Participant ID"
    },
    "session_id": {
        "Description": "Session ID derived from the CSV filename",
        "LongName": "Session ID"
    },
    "sex": {
        "Description": "Sex of the participant as reported by the participant",
        "LongName": "Sex",
        "Levels": {
            "M": "male",
            "F": "female"
        }
    },
    "age": {
        "Description": "Participant age at the time of the session",
        "LongName": "Participant age",
        "Units": "years"
    },
    "pathology": {
        "Description": "Cognitive status / diagnosis of the participant",
        "LongName": "Pathology name",
        "Levels": {
            "CN": "Typically Developing/Aging Cognitively Normal",
            "MCI": "Mild Cognitive Impairment",
            "dementia": "Dementia"
        }
    },
    "handedness": {
        "Description": "Dominant hand of the participant",
        "LongName": "Handedness"
    },
    "scanner": {
        "Description": "Scanner identifier",
        "LongName": "Scanner ID"
    },
    "race": {
        "Description": "Self-reported race of the participant",
        "LongName": "Race"
    }
}


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--demog-file', required=False, default=None,
                        help="Path to the external demographics CSV "
                             "(e.g. filtered_spinalcord_csa.csv). "
                             "If omitted, only participant_id and session_id are written.")
    return parser


def main():
    args = get_parser().parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    blsa_dir = repo_root / "spinal_cord" / "BLSA"

    if not blsa_dir.exists():
        sys.exit(f"ERROR: Directory not found: {blsa_dir}")

    # Discover sessions from CSV filenames
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

    if args.demog_file:
        demog_path = Path(args.demog_file)
        if not demog_path.exists():
            sys.exit(f"ERROR: Demographics file not found: {demog_path}")

        print(f"  Loading demographics from {demog_path}")
        demog = pd.read_csv(demog_path, low_memory=False)
        blsa_demog = demog[demog['dataset'] == 'BLSA'][
            ['subject', 'session', 'diagnosis', 'age', 'sex', 'handedness', 'scanner', 'race']
        ].copy()
        blsa_demog.columns = ['participant_id', 'session_id', 'diagnosis', 'age', 'sex', 'handedness', 'scanner', 'race']

        df = df.merge(blsa_demog, on=['participant_id', 'session_id'], how='left')

        # Map sex: 0 -> F, 1 -> M
        df['sex'] = df['sex'].map({0.0: 'F', 1.0: 'M'})

        # Map diagnosis to pathology
        df['pathology'] = df['diagnosis'].map({'cn': 'CN', 'mci': 'MCI', 'dementia': 'dementia'})

        # Round age to 1 decimal
        df['age'] = df['age'].round(1)

        df = df[['participant_id', 'session_id', 'sex', 'age', 'pathology', 'handedness', 'scanner', 'race']]
        df = df.fillna('n/a')

        matched = (df['sex'] != 'n/a').sum()
        print(f"  Demographics matched for {matched}/{len(df)} sessions")
        print(f"  Pathology distribution: {df['pathology'].value_counts().to_dict()}")

        # Write participants.json
        out_json = blsa_dir / "participants.json"
        with open(out_json, 'w') as f:
            json.dump(PARTICIPANTS_JSON, f, indent=4)
        print(f"  Written: {out_json.relative_to(repo_root)}")

    out_tsv = blsa_dir / "participants.tsv"
    df.to_csv(out_tsv, sep="\t", index=False)
    print(f"  Written: {out_tsv.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
