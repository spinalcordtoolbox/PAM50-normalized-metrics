#!/usr/bin/env python3
"""
Remove subjects/sessions that failed QC from the spinal_cord/ datasets.

Reads a Kurt-provided QA/QC CSV with columns:
    dataset, subject, session, labels_pass, segmentations_pass, ...

For every row where ``labels_pass == 0``, the corresponding
``sub-<id>[_ses-<ses>]_*_PAM50.csv`` file is deleted from
``spinal_cord/<dataset>/``.

The QC dataset name "KurtMS" maps to the repo directory "VUIIS-MS"; other
datasets in the QC file that are not present in this repo are ignored.

Usage:
    python code/remove_failed_qc.py --qc-file /path/to/qa_results_ALL.csv
    python code/remove_failed_qc.py --qc-file ... --dry-run
"""

import sys
import argparse
import pandas as pd
from pathlib import Path


# QC-file dataset name → repo directory name
DATASETS = {
    "AOMIC":        "AOMIC",
    "BLSA":         "BLSA",
    "DLBS":         "DLBS",
    "MASiVar":      "MASiVar",
    "TempleSocial": "TempleSocial",
    "UCLA_LA5c":    "UCLA_LA5c",
    "KurtMS":       "VUIIS-MS",
}


def get_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--qc-file", required=True,
                   help="Path to Kurt's qa_results_ALL.csv")
    p.add_argument("--dry-run", action="store_true",
                   help="List files that would be removed without deleting.")
    return p


def split_filename(stem: str) -> tuple[str, str | None]:
    """Return (subject, session_or_None) extracted from a CSV stem."""
    parts = stem.split("_")
    sub = parts[0]
    ses = next((p for p in parts if p.startswith("ses-")), None)
    return sub, ses


def main():
    args = get_parser().parse_args()

    qc_path = Path(args.qc_file)
    if not qc_path.exists():
        sys.exit(f"ERROR: QC file not found: {qc_path}")

    repo_root = Path(__file__).resolve().parent.parent
    spinal_cord = repo_root / "spinal_cord"

    df = pd.read_csv(qc_path)
    fails = df[df["labels_pass"] == 0].copy()
    fails["session"] = fails["session"].where(fails["session"].notna(), None)

    total_removed = 0
    for qc_name, dir_name in DATASETS.items():
        dataset_dir = spinal_cord / dir_name
        if not dataset_dir.exists():
            print(f"  [skip] {dir_name}: directory not found")
            continue

        # Build (sub, ses) → Path map of PAM50 CSVs in the dataset dir
        repo_files: dict[tuple[str, str | None], Path] = {}
        for f in dataset_dir.glob("sub-*.csv"):
            if "_PAM50" not in f.name:
                continue
            sub, ses = split_filename(f.stem)
            repo_files[(sub, ses)] = f

        ds_fails = fails[fails["dataset"] == qc_name]
        to_remove: list[Path] = []
        missing: list[tuple[str, str | None]] = []
        for _, row in ds_fails.iterrows():
            key = (row["subject"], row["session"])
            if key in repo_files:
                to_remove.append(repo_files[key])
            else:
                missing.append(key)

        action = "would remove" if args.dry_run else "removing"
        print(f"{dir_name:25} {action} {len(to_remove):4} file(s) "
              f"({len(ds_fails)} fails in QC; {len(missing)} not in repo)")

        for f in to_remove:
            if args.dry_run:
                print(f"  [dry-run] {f.relative_to(repo_root)}")
            else:
                f.unlink()
                total_removed += 1

    if args.dry_run:
        print("\nDry run complete. No files deleted.")
    else:
        print(f"\nRemoved {total_removed} file(s) total.")


if __name__ == "__main__":
    main()
