#!/bin/bash

# Process OpenNeuro ds004507 and generate PAM50-normalized per-slice morphometric CSVs.
#
# Usage from Git Bash:
#   bash processing_scripts/process_data_ds004507.sh /c/Users/marco/Desktop/ds004507
#
# Author: Marco Egidio Ressa

set -e -o pipefail

DATASET_ROOT="$1"

if [[ -z "$DATASET_ROOT" ]]; then
    echo "Usage:"
    echo "bash processing_scripts/process_data_ds004507.sh /path/to/ds004507"
    exit 1
fi

SESSIONS=(
    "ses-headDown"
    "ses-headNormal"
    "ses-headUp"
)

QC_DIR="${DATASET_ROOT}/qc_ds004507"
mkdir -p "$QC_DIR"

echo "============================================================"
echo "ds004507 - PAM50 CSV generation"
echo "Dataset root: $DATASET_ROOT"
echo "QC folder:    $QC_DIR"
echo "============================================================"

for subject_path in "${DATASET_ROOT}"/sub-*; do

    [[ -d "$subject_path" ]] || continue

    subject=$(basename "$subject_path")

    for session in "${SESSIONS[@]}"; do

        echo
        echo "============================================================"
        echo "Processing: ${subject} / ${session}"
        echo "============================================================"

        raw_anat="${DATASET_ROOT}/${subject}/${session}/anat"
        derivative_anat="${DATASET_ROOT}/derivatives/labels/${subject}/${session}/anat"

        image="${raw_anat}/${subject}_${session}_T2w.nii.gz"

        output_dir="${DATASET_ROOT}/${subject}/${session}"
        output_csv="${output_dir}/${subject}_${session}_PAM50.csv"

        mkdir -p "$output_dir"

        # ------------------------------------------------------
        # Spinal cord segmentation
        # ------------------------------------------------------

        seg_manual="${derivative_anat}/${subject}_${session}_T2w_seg-manual.nii.gz"
        seg_generated="${raw_anat}/${subject}_${session}_T2w_seg.nii.gz"

        if [[ -f "$seg_manual" ]]; then
            seg_file="$seg_manual"
            echo "Using manual spinal cord segmentation:"
            echo "$seg_file"

        elif [[ -f "$seg_generated" ]]; then
            seg_file="$seg_generated"
            echo "Using generated spinal cord segmentation:"
            echo "$seg_file"

        else
            echo "[MISSING] Spinal cord segmentation"
            echo "Checked:"
            echo "$seg_manual"
            echo "$seg_generated"
            continue
        fi

        # ------------------------------------------------------
        # Disc labels
        # ------------------------------------------------------

        disc_manual_derivative="${derivative_anat}/${subject}_${session}_T2w_labels-disc-manual.nii.gz"
        disc_manual_generated="${raw_anat}/${subject}_${session}_T2w_labels-disc-manual.nii.gz"
        disc_auto="${raw_anat}/${subject}_${session}_T2w_totalspineseg_discs.nii.gz"

        if [[ -f "$disc_manual_derivative" ]]; then
            disc_file="$disc_manual_derivative"
            echo "Using manual derivative disc labels:"
            echo "$disc_file"

        elif [[ -f "$disc_manual_generated" ]]; then
            disc_file="$disc_manual_generated"
            echo "Using manually generated disc labels:"
            echo "$disc_file"

        elif [[ -f "$disc_auto" ]]; then
            disc_file="$disc_auto"
            echo "Using TotalSpineSeg disc labels:"
            echo "$disc_file"

        else
            echo "[MISSING] Disc labels"
            echo "Checked:"
            echo "$disc_manual_derivative"
            echo "$disc_manual_generated"
            echo "$disc_auto"
            continue
        fi

        # ------------------------------------------------------
        # QC
        # ------------------------------------------------------

        echo "Generating spinal cord segmentation QC..."

        sct_qc \
            -i "$image" \
            -s "$seg_file" \
            -p sct_deepseg_sc \
            -qc "$QC_DIR" \
            -qc-dataset ds004507 \
            -qc-subject "${subject}_${session}"

        # ------------------------------------------------------
        # PAM50-normalized morphometrics
        # ------------------------------------------------------

        if [[ -f "$output_csv" ]]; then
			echo "[SKIP] PAM50 CSV already exists:"
			echo "$output_csv"
		else
			echo "Running sct_process_segmentation..."

			sct_process_segmentation \
			-i "$seg_file" \
			-discfile "$disc_file" \
			-perslice 1 \
			-normalize-PAM50 1 \
			-o "$output_csv"

			echo "[OK] $output_csv"
fi

    done
done

echo
echo "============================================================"
echo "Processing completed."
echo "CSV files are stored in the individual subject/session folders."
echo "QC report:"
echo "${QC_DIR}/index.html"
echo "============================================================"