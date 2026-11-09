#!/bin/bash

# Process OpenNeuro ds004616 and generate contrast-agnostic spinal cord segmentations,
# automatic intervertebral disc labels, PAM50-normalized per-slice morphometric CSVs,
# and QC reports.
#
# Usage from Git Bash:
#   bash processing_scripts/process_data_ds004616.sh /path/to/ds004616
#
# Author: Marco Egidio Ressa

set -e -o pipefail

DATASET_ROOT="$1"

if [[ -z "$DATASET_ROOT" ]]; then
    echo "Usage:"
    echo "bash processing_scripts/process_data_ds004616.sh /path/to/ds004616"
    exit 1
fi

SEG_DIR="${DATASET_ROOT}/contrast_agnostic_segmentations"
OUTPUT_DIR="${DATASET_ROOT}/contrast_agnostic_outputs"
QC_DIR="${DATASET_ROOT}/qc_contrast_agnostic"
QC_SPINE_DIR="${DATASET_ROOT}/qc_spine"

mkdir -p "$SEG_DIR"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$QC_DIR"
mkdir -p "$QC_SPINE_DIR"

echo "============================================================"
echo "ds004616 - Contrast-agnostic segmentation and PAM50 metrics"
echo "Dataset root: $DATASET_ROOT"
echo "============================================================"

for image in "${DATASET_ROOT}"/sub-*/ses-*/anat/*_T2w.nii.gz; do

    [[ -f "$image" ]] || continue

    subject=$(basename "$(dirname "$(dirname "$(dirname "$image")")")")
    session=$(basename "$(dirname "$(dirname "$image")")")
    base="${subject}_${session}_T2w"

    seg_file="${SEG_DIR}/${base}_seg.nii.gz"
    disc_file="${DATASET_ROOT}/${subject}/${session}/anat/${base}_totalspineseg_discs.nii.gz"
    output_csv="${OUTPUT_DIR}/${subject}_${session}_PAM50.csv"

    echo
    echo "============================================================"
    echo "Processing: ${subject} ${session}"
    echo "============================================================"

    echo "Generating contrast-agnostic spinal cord segmentation..."

    sct_deepseg spinalcord \
        -i "$image" \
        -o "$seg_file"

    echo "Generating QC for spinal cord segmentation..."

    sct_qc \
        -i "$image" \
        -s "$seg_file" \
        -p sct_deepseg_sc \
        -qc "$QC_DIR" \
        -qc-dataset ds004616 \
        -qc-subject "${subject}_${session}"

    echo "Generating automatic spine and intervertebral disc labels..."

    sct_deepseg spine \
        -i "$image" \
        -qc "$QC_SPINE_DIR"

    echo "Generating PAM50-normalized morphometric CSV..."

    sct_process_segmentation \
        -i "$seg_file" \
        -discfile "$disc_file" \
        -perslice 1 \
        -normalize-PAM50 1 \
        -o "$output_csv"

    echo "Created:"
    echo "$output_csv"

done

echo
echo "============================================================"
echo "Done."
echo "Segmentations: $SEG_DIR"
echo "Outputs: $OUTPUT_DIR"
echo "Spinal cord QC: $QC_DIR"
echo "Spine/disc QC: $QC_SPINE_DIR"
echo "============================================================"
