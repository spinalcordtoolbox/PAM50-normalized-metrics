#!/bin/bash
#
# Reconstruct a T2*w image from the MEGRE of each subject of a BIDS dataset (e.g., balgrist_sci_tasci, which has
# no T2w). The T2*w image is then used as the anatomical image by process_data_spine-generic_dwi.sh.
# Run this script before process_data_spine-generic_dwi.sh, on a machine with FSL (mcflirt).
#
# Follows https://github.com/valosekj/model-seg_lumbar/blob/231a0a467ce35e2191121e61b2ad15ccb72c391b/reconstruct_uncropped_megre.sh:
#   1. per measurement: concatenate the first 3 echoes and RMS-combine them
#   2. concatenate the measurements, motion-correct them against each other (FSL mcflirt), then mean across them
#
# Input:  <PATH_DATA>/sub-XX/ses-XX/anat/sub-XX_ses-XX_meas-<1..N>_echo-<1..5>_MEGRE.nii.gz
# Output: <PATH_DATA>/sub-XX/ses-XX/anat/sub-XX_ses-XX_T2starw.nii.gz (+ JSON sidecar)
#
# Usage:
#     ./reconstruct_megre_t2starw.sh <PATH_DATA>
#
# Requires: SCT (sct_image, sct_maths) and FSL (mcflirt).
#
# Author: Jan Valosek
#

set -e -o pipefail

PATH_DATA=$1
if [[ -z $PATH_DATA ]]; then
  echo "Usage: $0 <PATH_DATA>"
  exit 1
fi
for cmd in sct_image sct_maths mcflirt; do
  if ! command -v $cmd > /dev/null; then
    echo "$cmd not found"
    exit 1
  fi
done

for path_anat in ${PATH_DATA}/sub-*/ses-*/anat ${PATH_DATA}/sub-*/anat; do
  [[ -d $path_anat ]] || continue
  # sub-XX_ses-XX (or sub-XX without sessions)
  subject_session=${path_anat#${PATH_DATA}/}
  subject_session=${subject_session%/anat}
  file=${subject_session//\//_}
  if [[ ! -e ${path_anat}/${file}_meas-1_echo-1_MEGRE.nii.gz ]]; then
    echo "⚠️ ${subject_session}: no MEGRE found, skipping"
    continue
  fi
  file_out=${path_anat}/${file}_T2starw.nii.gz
  if [[ -e $file_out ]]; then
    echo "${subject_session}: ${file_out} already exists, skipping"
    continue
  fi
  echo "👉 ${subject_session}"

  tmp=$(mktemp -d)
  files_rms=()
  for meas in $(cd ${path_anat}; compgen -G "${file}_meas-[0-9]_echo-1_MEGRE.nii.gz" | sed 's/.*_meas-\([0-9]\)_echo-1_MEGRE.nii.gz/\1/' | sort); do
    # Step 1: RMS of the first 3 echoes of this measurement
    sct_image -i ${path_anat}/${file}_meas-${meas}_echo-{1,2,3}_MEGRE.nii.gz -concat t \
      -o ${tmp}/meas-${meas}_echo-123.nii.gz -v 0
    sct_maths -i ${tmp}/meas-${meas}_echo-123.nii.gz -rms t -o ${tmp}/meas-${meas}_echo-rms.nii.gz -v 0
    files_rms+=(${tmp}/meas-${meas}_echo-rms.nii.gz)
  done
  # Step 2: motion correction across measurements, then mean
  sct_image -i ${files_rms[@]} -concat t -o ${tmp}/meas-all_echo-rms.nii.gz -v 0
  mcflirt -in ${tmp}/meas-all_echo-rms.nii.gz -spline_final -cost leastsquares -o ${tmp}/meas-all_echo-rms_mc.nii.gz
  sct_maths -i ${tmp}/meas-all_echo-rms_mc.nii.gz -mean t -o ${file_out} -v 0
  rm -rf ${tmp}

  cat > ${file_out%.nii.gz}.json <<EOF
{
  "Description": "T2*w image reconstructed from the MEGRE: per measurement, RMS of echoes 1-3; then motion correction across the ${#files_rms[@]} measurements (FSL mcflirt -spline_final -cost leastsquares) and mean.",
  "Sources": [$(for m in $(seq 1 ${#files_rms[@]}); do printf '"%s"' "${file}_meas-${m}_echo-{1,2,3}_MEGRE.nii.gz"; [[ $m -lt ${#files_rms[@]} ]] && printf ', '; done)],
  "Script": "https://github.com/spinalcordtoolbox/PAM50-normalized-metrics/blob/main/processing_scripts/reconstruct_megre_t2starw.sh",
  "Date": "$(date '+%Y-%m-%d %H:%M:%S')"
}
EOF
  echo "✅ ${file_out}"
done
