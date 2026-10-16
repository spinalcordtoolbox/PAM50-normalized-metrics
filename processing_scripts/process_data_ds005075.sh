#!/bin/bash

# Process OpenNeuro ds005075 and generate PAM50-normalized per-slice morphometric CSVs.
#
# Usage from Git Bash:
#   bash processing_scripts/process_data_ds005075.sh /path/to/ds005075
#
# Author: Marco Egidio Ressa

set -e -o pipefail

DATASET_ROOT="$1"

if [[ -z "$DATASET_ROOT" ]]; then
    echo "Usage:"
    echo "bash processing_scripts/process_data_ds005075.sh /path/to/ds005075"
    exit 1
fi

OUTPUT_DIR="${DATASET_ROOT}/outputs"
mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "ds005075 - PAM50 CSV generation"
echo "Dataset root: $DATASET_ROOT"
echo "Output folder: $OUTPUT_DIR"
echo "============================================================"

for subject_path in "${DATASET_ROOT}"/sub-*; do

    [[ -d "$subject_path" ]] || continue

    subject=$(basename "$subject_path")

    echo
    echo "============================================================"
    echo "Processing: ${subject}"
    echo "============================================================"

    image="${DATASET_ROOT}/${subject}/anat/${subject}_T1w.nii.gz"

    derivative_anat="${DATASET_ROOT}/derivatives/preprocessing/${subject}/anat/T1w_seg_CSF"

    seg_file="${derivative_anat}/${subject}_desc-T1w_space-orig_label-sc_mask.nii.gz"
    disc_file="${derivative_anat}/${subject}_desc-T1w_space-orig_label-ivd_mask.nii.gz"

    output_csv="${OUTPUT_DIR}/${subject}_PAM50.csv"

    if [[ ! -f "$image" ]]; then
        echo "[MISSING] T1w image:"
        echo "$image"
        continue
    fi

    if [[ ! -f "$seg_file" ]]; then
        echo "[MISSING] Spinal cord segmentation:"
        echo "$seg_file"
        continue
    fi

    if [[ ! -f "$disc_file" ]]; then
        echo "[MISSING] Intervertebral disc labels:"
        echo "$disc_file"
        continue
    fi

    echo "T1w image:"
    echo "$image"

    echo "Spinal cord segmentation:"
    echo "$seg_file"

    echo "Intervertebral disc labels:"
    echo "$disc_file"

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
echo "Outputs: $OUTPUT_DIR"
echo "============================================================"