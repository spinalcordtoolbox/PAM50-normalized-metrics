from pathlib import Path
import subprocess
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "dataset_root",
    type=Path,
    help="Path to the local ds004507 dataset"
)
args = parser.parse_args()

ROOT = args.dataset_root.resolve()

SESSIONS = [
    "ses-headDown",
    "ses-headNormal",
    "ses-headUp",
]

subjects = sorted(
    p.name for p in ROOT.glob("sub-*") if p.is_dir()
)

print("=" * 70)
print("ds004507 - PAM50 CSV generation")
print("=" * 70)
print(f"Subjects found: {len(subjects)}")
print(f"Sessions per subject: {len(SESSIONS)}")
print(f"Expected CSVs: {len(subjects) * len(SESSIONS)}")
print()

success = []
skipped = []
failed = []
missing = []

for subject in subjects:
    for session in SESSIONS:

        print()
        print("=" * 70)
        print(f"Processing: {subject} / {session}")
        print("=" * 70)

        derivative_anat = (
            ROOT
            / "derivatives"
            / "labels"
            / subject
            / session
            / "anat"
        )

        raw_anat = (
            ROOT
            / subject
            / session
            / "anat"
        )

        output_folder = ROOT / subject / session

        output_csv = (
            output_folder
            / f"{subject}_{session}_PAM50.csv"
        )

        # 1. Skip if CSV already exists
        if output_csv.exists():
            print(f"[SKIP] CSV already exists:")
            print(output_csv)
            skipped.append((subject, session))
            continue

        # 2. Segmentation: only use manual derivative for now
        seg_file = (
            derivative_anat
            / f"{subject}_{session}_T2w_seg-manual.nii.gz"
        )

        if not seg_file.exists():
            print("[MISSING] Spinal cord segmentation not found.")
            print(seg_file)
            missing.append((subject, session, "segmentation"))
            continue

        # 3. Disc labels: first manual derivative, then automatic fallback
        disc_manual = (
            derivative_anat
            / f"{subject}_{session}_T2w_labels-disc-manual.nii.gz"
        )

        disc_auto = (
            raw_anat
            / f"{subject}_{session}_T2w_totalspineseg_discs.nii.gz"
        )

        if disc_manual.exists():
            disc_file = disc_manual
            disc_source = "manual derivative"

        elif disc_auto.exists():
            disc_file = disc_auto
            disc_source = "automatic TotalSpineSeg"

        else:
            print("[MISSING] Disc labels not found.")
            print("Checked:")
            print(disc_manual)
            print(disc_auto)
            missing.append((subject, session, "disc labels"))
            continue

        output_folder.mkdir(parents=True, exist_ok=True)

        print("Segmentation:")
        print(seg_file)

        print(f"Disc labels ({disc_source}):")
        print(disc_file)

        print("Output:")
        print(output_csv)

        command = [
            "sct_process_segmentation",
            "-i", str(seg_file),
            "-discfile", str(disc_file),
            "-perslice", "1",
            "-normalize-PAM50", "1",
            "-o", str(output_csv),
        ]

        try:
            subprocess.run(
                command,
                check=True
            )

            print(f"[OK] {output_csv.name}")
            success.append((subject, session))

        except subprocess.CalledProcessError as error:
            print(f"[FAILED] {subject} / {session}")
            print(f"Return code: {error.returncode}")
            failed.append((subject, session))

        except FileNotFoundError:
            print()
            print("[ERROR] sct_process_segmentation was not found.")
            print("Make sure SCT is available in this Command Prompt.")
            sys.exit(1)

print()
print("=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(f"Generated now    : {len(success)}")
print(f"Already existing : {len(skipped)}")
print(f"Failed           : {len(failed)}")
print(f"Missing inputs   : {len(missing)}")
print(f"Expected total   : {len(subjects) * len(SESSIONS)}")

if missing:
    print()
    print("MISSING INPUTS:")
    for subject, session, what in missing:
        print(f"  {subject} / {session}: {what}")

if failed:
    print()
    print("FAILED:")
    for subject, session in failed:
        print(f"  {subject} / {session}")

print()
print("Done.")