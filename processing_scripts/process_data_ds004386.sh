#!/bin/bash

# Process OpenNeuro ds004386 and generate contrast-agnostic spinal cord segmentations,
# PAM50-normalized per-slice morphometric CSVs, and QC reports.
#
# Usage from Git Bash:
#   bash processing_scripts/process_data_ds004386.sh /path/to/ds004386
#
# Author: Marco Egidio Ressa

set -e -o pipefail

DATASET_ROOT="$1"

if [[ -z "$DATASET_ROOT" ]]; then
    echo "Usage:"
    echo "bash processing_scripts/process_data_ds004386.sh /path/to/ds004386"
    exit 1
fi

SEG_DIR="${DATASET_ROOT}/contrast_agnostic_segmentations"
OUTPUT_DIR="${DATASET_ROOT}/contrast_agnostic_outputs"
QC_DIR="${DATASET_ROOT}/qc_contrast_agnostic"

mkdir -p "$SEG_DIR"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$QC_DIR"

echo "============================================================"
echo "ds004386 - Contrast-agnostic segmentation and PAM50 metrics"
echo "Dataset root: $DATASET_ROOT"
echo "Segmentations: $SEG_DIR"
echo "Outputs: $OUTPUT_DIR"
echo "QC: $QC_DIR"
echo "============================================================"

for subject_path in "${DATASET_ROOT}"/sub-ZS*; do

    [[ -d "$subject_path" ]] || continue

    subject=$(basename "$subject_path")

    echo
    echo "============================================================"
    echo "Processing: ${subject}"
    echo "============================================================"

    image="${DATASET_ROOT}/${subject}/anat/${subject}_T2w.nii.gz"
    disc_file="${DATASET_ROOT}/derivatives/${subject}/anat/disc_labels_edited.nii.gz"

    seg_file="${SEG_DIR}/${subject}_T2w_contrast-agnostic.nii.gz"
    output_csv="${OUTPUT_DIR}/${subject}_contrast-agnostic_PAM50.csv"

    if [[ ! -f "$image" ]]; then
        echo "[MISSING] T2w image:"
        echo "$image"
        continue
    fi

    if [[ ! -f "$disc_file" ]]; then
        echo "[MISSING] Intervertebral disc labels:"
        echo "$disc_file"
        continue
    fi

    echo "T2w image:"
    echo "$image"

    echo "Intervertebral disc labels:"
    echo "$disc_file"

    echo
    echo "Generating contrast-agnostic spinal cord segmentation..."

    sct_deepseg spinalcord \
        -i "$image" \
        -o "$seg_file"

    echo
    echo "Generating QC for spinal cord segmentation..."

    sct_qc \
        -i "$image" \
        -s "$seg_file" \
        -p sct_deepseg_sc \
        -qc "$QC_DIR" \
        -qc-dataset ds004386 \
        -qc-subject "$subject"

    echo
    echo "Generating PAM50-normalized morphometric CSV..."

    sct_process_segmentation \
        -i "$seg_file" \
        -discfile "$disc_file" \
        -perslice 1 \
        -normalize-PAM50 1 \
        -o "$output_csv"

    echo
    echo "Created:"
    echo "$output_csv"

done

echo
echo "============================================================"
echo "Done."
echo "Segmentations: $SEG_DIR"
echo "Outputs: $OUTPUT_DIR"
echo "QC reports: $QC_DIR"
echo "============================================================"