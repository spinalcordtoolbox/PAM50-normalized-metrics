#!/bin/bash
#
# Process T2w and DWI data from spine-generic multi-subject dataset to build a
# normative database of microstructural metrics (FA, MD, RD, AD) normalized to
# PAM50 template space.
#
# Requires:
#   - T2w processing must run first (or be included): the T2w-to-template warp
#     (warp_template2T2w.nii.gz, warp_T2w2template.nii.gz) is used as the
#     initial transformation for DWI registration.
#   - SC segmentation and disc labels from derivatives/labels (spine-generic
#     multi-subject dataset: https://github.com/spine-generic/data-multi-subject)
#
# Usage:
#     sct_run_batch -c <PATH_TO_REPO>/etc/config_process_data_dwi.json
#
# Preprocessing flags (passed via "script_args" in the config or -script-args on the CLI):
#   --mask-method  centerline|deepseg  Method for creating the mask used in motion correction.
#                                      centerline: sct_get_centerline (default; spine-generic pipeline)
#                                      deepseg:    sct_deepseg spinalcord (SCT tutorial pipeline)
#   --mask-size    30|35               Mask radius in mm (default: 30; SCT tutorial uses 35).
#   --denoise      0|1                 Apply sct_dmri_denoise_patch2self after motion correction
#                                      (default: 1).
#   --seg-method   deepseg_sc|deepseg  DWI cord segmentation method.
#                                      deepseg_sc: sct_deepseg_sc (default; compatible with
#                                                  spine-generic derivatives/labels)
#                                      deepseg:    sct_deepseg spinalcord (newer contrast-agnostic model)
#
# Example config_process_data_dwi.json (all defaults shown explicitly):
# {
#   "path_data"   : "<PATH_TO_SPINE_GENERIC_DATASET>",
#   "path_output" : "<PATH_TO_OUTPUT>",
#   "script"      : "<PATH_TO_REPO>/processing_scripts/process_data_spine-generic_dwi.sh",
#   "script_args" : "--mask-method centerline --mask-size 30 --seg-method deepseg_sc --denoise 1",
#   "jobs"        : 8
# }
# To test an alternative configuration, change "script_args", e.g.:
# "script_args" : "--mask-method deepseg --mask-size 35 --seg-method deepseg --denoise 0"
#
# The following global variables are retrieved from the caller sct_run_batch:
# PATH_DATA_PROCESSED="~/data_processed"
# PATH_RESULTS="~/results"
# PATH_LOG="~/log"
# PATH_QC="~/qc"
#
# Manual segmentations or labels should be located under:
#   PATH_DATA/derivatives/labels/SUBJECT/anat/   (T2w)
#   PATH_DATA/derivatives/labels/SUBJECT/dwi/    (DWI)
#
# Output (per-subject CSVs, one per DTI metric):
#   PATH_RESULTS/dwi/sub-XXXX_dwi_FA_PAM50.csv
#   PATH_RESULTS/dwi/sub-XXXX_dwi_MD_PAM50.csv
#   PATH_RESULTS/dwi/sub-XXXX_dwi_RD_PAM50.csv
#   PATH_RESULTS/dwi/sub-XXXX_dwi_AD_PAM50.csv
#
#
# Authors: Jan Valosek
# Inspired by:
#   https://github.com/valosekj/dcm-olomouc/blob/main/scripts/01_process_data.sh
#   https://github.com/spine-generic/spine-generic/blob/master/process_data.sh
#

# List of WM/GM tracts to extract metrics from
# Legend: https://spinalcordtoolbox.com/overview/concepts/pam50.html#white-and-gray-matter-atlas
# Inspired by: Valosek et al., 2021, DOI: 10.1111/ene.15027
tracts=(
  "51"      # white matter (combined)
  "52"      # gray matter (combined)
  "53"      # dorsal columns
  "54"      # lateral columns
  "55"      # ventral columns
  "0,1"     # left and right fasciculus gracilis
  "2,3"     # left and right fasciculus cuneatus
  "4,5"     # left and right lateral corticospinal tract
  "12,13"   # left and right spinal lemniscus (spinothalamic + spinoreticular)
  "30,31"   # ventral gray matter horns
)

# Uncomment for full verbose
set -x

# Immediately exit if error
set -e -o pipefail

# Exit if user presses CTRL+C (Linux) or CMD+C (OSX)
trap "echo Caught Keyboard Interrupt within script. Exiting now.; exit" INT

# Print retrieved variables from sct_run_batch to the log
echo "Retrieved variables from the caller sct_run_batch:"
echo "PATH_DATA: ${PATH_DATA}"
echo "PATH_DATA_PROCESSED: ${PATH_DATA_PROCESSED}"
echo "PATH_RESULTS: ${PATH_RESULTS}"
echo "PATH_LOG: ${PATH_LOG}"
echo "PATH_QC: ${PATH_QC}"

PATH_DERIVATIVES="${PATH_DATA}/derivatives"

# Retrieve subject from sct_run_batch
SUBJECT=$1
shift

# ---- Preprocessing flags (controlled via script_args in config or -script-args CLI) ----
MASK_METHOD="centerline"   # centerline | deepseg
MASK_SIZE=30               # mm; spine-generic uses 30mm, SCT tutorial uses 35mm
SEG_METHOD="deepseg_sc"    # deepseg_sc | deepseg
DENOISE=1                  # 1=yes, 0=no

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mask-method) MASK_METHOD="$2"; shift 2 ;;
    --mask-size)   MASK_SIZE="$2";   shift 2 ;;
    --seg-method)  SEG_METHOD="$2";  shift 2 ;;
    --denoise)     DENOISE="$2";     shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

echo "SUBJECT: ${SUBJECT}"
echo "Preprocessing options:"
echo "  MASK_METHOD : ${MASK_METHOD}"
echo "  MASK_SIZE   : ${MASK_SIZE}mm"
echo "  SEG_METHOD  : ${SEG_METHOD}"
echo "  DENOISE     : ${DENOISE}"

# Log preprocessing arguments to file
echo "[$(date '+%Y-%m-%d %H:%M:%S')] SUBJECT: ${SUBJECT} | MASK_METHOD: ${MASK_METHOD} | MASK_SIZE: ${MASK_SIZE}mm | SEG_METHOD: ${SEG_METHOD} | DENOISE: ${DENOISE}" >> "${PATH_LOG}/DWI_preprocessing_args.log"

# get starting time
start=`date +%s`

# ==============================================================================
# FUNCTIONS
# ==============================================================================

# Check if manual SC segmentation exists in derivatives/labels. If yes, copy it;
# if no, run automatic segmentation.
# Sets global variable FILESEG.
segment_if_does_not_exist() {
  local file="$1"
  local contrast="$2"          # 't2', 'dwi'
  local seg_method="${3:-deepseg_sc}"  # deepseg_sc | deepseg (only used for auto-seg fallback)
  if [[ $contrast == "dwi" ]]; then
    folder_contrast="dwi"
  else
    folder_contrast="anat"
  fi
  FILESEG="${file}_label-SC_seg"
  FILESEGMANUAL="${PATH_DERIVATIVES}/labels/${SUBJECT}/${folder_contrast}/${FILESEG}.nii.gz"
  # Define log prefix to have consistent log files for T2w and DWI files
  local log_prefix
  if [[ $contrast == "t2" ]]; then log_prefix="T2w"; else log_prefix="${contrast}"; fi
  echo
  echo "Looking for manual segmentation: $FILESEGMANUAL"
  if [[ -e $FILESEGMANUAL ]]; then
    echo "✏️ [$(date '+%Y-%m-%d %H:%M:%S')] Found! Using manual segmentation."
    echo "✏️ [$(date '+%Y-%m-%d %H:%M:%S')] ${FILESEG}.nii.gz found --> using manual segmentation" >> "${PATH_LOG}/${log_prefix}_SC_segmentations.log"
    rsync -avzh $FILESEGMANUAL ${FILESEG}.nii.gz
    sct_qc -i ${file}.nii.gz -s ${FILESEG}.nii.gz -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}
  else
    echo "🤖 [$(date '+%Y-%m-%d %H:%M:%S')] Not found. Proceeding with automatic segmentation (${seg_method})."
    echo "🤖 [$(date '+%Y-%m-%d %H:%M:%S')] ${FILESEG}.nii.gz NOT found --> segmenting automatically with ${seg_method}" >> "${PATH_LOG}/${log_prefix}_SC_segmentations.log"
    if [[ "${seg_method}" == "deepseg" ]]; then
      sct_deepseg spinalcord -i ${file}.nii.gz -o ${FILESEG}.nii.gz -qc ${PATH_QC} -qc-subject ${SUBJECT}
    else
      sct_deepseg_sc -i ${file}.nii.gz -c ${contrast} -o ${FILESEG}.nii.gz -qc ${PATH_QC} -qc-subject ${SUBJECT}
    fi
  fi
}

# Check if manual disc labels exist in derivatives/labels. If yes, use them to
# generate labeled segmentation; if no, run automatic labeling.
# Sets global variable FILELABEL.
label_if_does_not_exist() {
  local file="$1"
  local file_seg="$2"
  local contrast="$3"
  FILELABEL="${file}_label-discs_dlabel"
  FILELABELMANUAL="${PATH_DERIVATIVES}/labels/${SUBJECT}/anat/${FILELABEL}.nii.gz"
  echo "Looking for manual disc labels: $FILELABELMANUAL"
  if [[ -e $FILELABELMANUAL ]]; then
    echo "✏️ [$(date '+%Y-%m-%d %H:%M:%S')] Found! Using manual disc labels."
    echo "✏️ [$(date '+%Y-%m-%d %H:%M:%S')] ${FILELABEL}.nii.gz found --> using manual disc labels" >> "${PATH_LOG}/T2w_disc_labels.log"
    rsync -avzh $FILELABELMANUAL ${FILELABEL}.nii.gz
    sct_image -i ${file}.nii.gz -set-sform-to-qform
    sct_image -i ${file_seg}.nii.gz -set-sform-to-qform
    sct_image -i ${FILELABEL}.nii.gz -set-sform-to-qform
    sct_label_vertebrae -i ${file}.nii.gz -s ${file_seg}.nii.gz -discfile ${FILELABEL}.nii.gz -c ${contrast} -qc ${PATH_QC} -qc-subject ${SUBJECT}
  else
    echo "🤖 [$(date '+%Y-%m-%d %H:%M:%S')] Not found. Proceeding with automatic labeling."
    echo "🤖 [$(date '+%Y-%m-%d %H:%M:%S')] ${FILELABEL}.nii.gz NOT found --> using automatic labeling" >> "${PATH_LOG}/T2w_disc_labels.log"
    sct_label_vertebrae -i ${file}.nii.gz -s ${file_seg}.nii.gz -c ${contrast} -qc ${PATH_QC} -qc-subject ${SUBJECT}
  fi
}

# ==============================================================================
# SCRIPT STARTS HERE
# ==============================================================================
sct_check_dependencies -short

cd $PATH_DATA_PROCESSED

# Copy T2w and DWI source data into subject-specific subfolders
mkdir -p ${SUBJECT}/anat ${SUBJECT}/dwi
rsync -avzh ${PATH_DATA}/${SUBJECT}/anat/${SUBJECT}_*T2w.* ${SUBJECT}/anat/
rsync -avzh ${PATH_DATA}/${SUBJECT}/dwi/${SUBJECT}_dwi.* ${SUBJECT}/dwi/

# ==============================================================================
# T2w
# ==============================================================================
cd ${SUBJECT}/anat

file_t2="${SUBJECT}_T2w"
echo "👉 Processing T2w: ${file_t2}"

# Segment spinal cord
segment_if_does_not_exist ${file_t2} 't2'
file_t2_seg=$FILESEG

# Label vertebrae from manual disc labels (or automatically)
label_if_does_not_exist ${file_t2} ${file_t2_seg} 't2'

# Register T2w to PAM50 template
# Step 1: centermassrot accounts for cord rotation
# Step 2: syn for small-scale deformations
sct_register_to_template \
  -i ${file_t2}.nii.gz \
  -s ${file_t2_seg}.nii.gz \
  -ldisc ${file_t2_seg}_labeled_discs.nii.gz \
  -c t2 \
  -param step=1,type=seg,algo=centermassrot:step=2,type=seg,algo=syn,metric=MeanSquares,slicewise=1,smooth=0,iter=5 \
  -qc ${PATH_QC} -qc-subject ${SUBJECT}

# Rename warping fields for clarity
mv warp_template2anat.nii.gz warp_template2T2w.nii.gz
mv warp_anat2template.nii.gz warp_T2w2template.nii.gz

# ==============================================================================
# DWI
# ==============================================================================
cd ../dwi

file_dwi="${SUBJECT}_dwi"
file_bval=${file_dwi}.bval
file_bvec=${file_dwi}.bvec

echo "👉 Processing DWI: ${file_dwi}"

# ----------
# Preprocessing
# ----------

# Separate b=0 and DWI volumes; creates *_dwi_mean.nii.gz
sct_dmri_separate_b0_and_dwi -i ${file_dwi}.nii.gz -bvec ${file_bvec}

# Create a mask around the cord to restrict motion correction and speed up processing.
# Use either spinal cord segmentation or spinal cord centerline.
# Method controlled by --mask-method.
if [[ "${MASK_METHOD}" == "deepseg" ]]; then
  # SCT tutorial approach: segment cord, pass segmentation as the centerline reference
  sct_deepseg spinalcord \
    -i ${file_dwi}_dwi_mean.nii.gz \
    -o ${file_dwi}_dwi_mean_seg.nii.gz \
    -qc ${PATH_QC} -qc-subject ${SUBJECT}
  centerline_for_mask="${file_dwi}_dwi_mean_seg.nii.gz"
else
  # spine-generic approach: extract centerline (default)
  sct_get_centerline -i ${file_dwi}_dwi_mean.nii.gz -c dwi -qc ${PATH_QC} -qc-subject ${SUBJECT}
  centerline_for_mask="${file_dwi}_dwi_mean_centerline.nii.gz"
fi

# Size of the mask (usually 30mm or 35mm).
# Size controlled by --mask-size.
sct_create_mask \
  -i ${file_dwi}_dwi_mean.nii.gz \
  -p centerline,${centerline_for_mask} \
  -size ${MASK_SIZE}mm

# Motion correction using the mask
sct_dmri_moco -i ${file_dwi}.nii.gz -bvec ${file_bvec} -m mask_${file_dwi}_dwi_mean.nii.gz -x spline
file_dwi=${file_dwi}_moco

# Rename mean DWI to the BIDS rec-average convention used by spine-generic derivatives
# (derivatives/labels uses sub-XX_rec-average_dwi_label-SC_seg.nii.gz)
mv ${file_dwi}_dwi_mean.nii.gz ${SUBJECT}_rec-average_dwi.nii.gz
file_dwi_mean="${SUBJECT}_rec-average_dwi"

# Denoise with patch2self (controlled by --denoise flag; default: enabled)
if [[ "${DENOISE}" == "1" ]]; then
  sct_dmri_denoise_patch2self \
    -i ${file_dwi}.nii.gz \
    -b ${file_bval} \
    -o ${file_dwi}_denoised.nii.gz
  file_dwi=${file_dwi}_denoised
fi

# ----------
# Spinal cord segmentation in DWI space
# ----------

# Segment SC in mean DWI image (check for manual seg first; method controlled by --seg-method)
# sct_deepseg spinalcord or sct_deepseg_sc
segment_if_does_not_exist ${file_dwi_mean} "dwi" "${SEG_METHOD}"
file_dwi_seg=$FILESEG

# ----------
# Template registration
# ----------
# Register PAM50 template to DWI space.
# Key: use warp_template2T2w.nii.gz as -initwarp
# Three-step registration:
#   1. centermass: coarse alignment accounting for cord position
#   2. bsplinesyn (regularized SyN) on seg: preserves internal WM/GM structure
#   3. syn on image: fine-scale deformation using image intensity (metric=CC)
sct_register_multimodal \
  -i $SCT_DIR/data/PAM50/template/PAM50_t2.nii.gz \
  -iseg $SCT_DIR/data/PAM50/template/PAM50_cord.nii.gz \
  -d ${file_dwi_mean}.nii.gz \
  -dseg ${file_dwi_seg}.nii.gz \
  -param step=1,type=seg,algo=centermass:step=2,type=seg,algo=bsplinesyn,metric=MeanSquares,slicewise=1,iter=3:step=3,type=im,algo=syn,metric=CC,iter=3,slicewise=1 \
  -initwarp ../anat/warp_template2T2w.nii.gz \
  -initwarpinv ../anat/warp_T2w2template.nii.gz \
  -owarp warp_template2dwi.nii.gz \
  -owarpinv warp_dwi2template.nii.gz \
  -qc ${PATH_QC} -qc-subject ${SUBJECT}

# Warp PAM50 template and WM atlas to DWI space for QC purposes
sct_warp_template \
  -d ${file_dwi_mean}.nii.gz \
  -w warp_template2dwi.nii.gz \
  -ofolder label_${file_dwi} \
  -qc ${PATH_QC} -qc-subject ${SUBJECT}
# QC: overlay PAM50 vertebral levels on mean DWI (checks registration quality)
sct_qc \
  -i ${file_dwi_mean}.nii.gz \
  -s label_${file_dwi}/template/PAM50_levels.nii.gz \
  -p sct_label_vertebrae \
  -qc ${PATH_QC} -qc-subject "PAM50_levels_dwi"

# QC: bring DWI vertebral levels into T2w space to check DWI FOV coverage
sct_register_multimodal \
  -i label_${file_dwi}/template/PAM50_levels.nii.gz \
  -d ../anat/${SUBJECT}_T2w.nii.gz \
  -identity 1 -x nn
rm -f warp_PAM50_levels2${SUBJECT}_T2w.nii.gz warp_${SUBJECT}_T2w2PAM50_levels.nii.gz ${SUBJECT}_T2w_reg.nii.gz
sct_qc \
  -i ../anat/${SUBJECT}_T2w.nii.gz \
  -s PAM50_levels_reg.nii.gz \
  -p sct_label_vertebrae \
  -qc ${PATH_QC} -qc-subject "PAM50_levels_DWI_to_T2w"

# ----------
# DTI computation
# ----------
# Compute FA, MD, RD, AD maps from the preprocessed (cropped, moco, denoised) data
sct_dmri_compute_dti \
  -i ${file_dwi}.nii.gz \
  -bvec ${file_bvec} \
  -bval ${file_bval} \
  -method standard \
  -o ${file_dwi}_

# ----------
# Warp DTI maps to PAM50 template space and extract metrics
# ----------
# DTI metrics are warped to the PAM50 template space before extraction so that
# each slice index corresponds to the same anatomical location across subjects.
# This is the DWI equivalent of 'sct_process_segmentation -normalize-PAM50':
# instead of extracting in native space (where slice counts per level differ
# between subjects), all maps are brought into a common coordinate frame first.
# The PAM50 atlas and levels files are used directly from the template, since
# the metric maps are already in that space.
mkdir -p ${PATH_RESULTS}/dwi

dti_metrics=(FA MD RD AD)

# Process DTI metrics in parallel for faster processing
pids=()
for dti_metric in "${dti_metrics[@]}"; do
  (
    # Warp DTI map to PAM50 template space using the inverse warp from registration
    sct_apply_transfo \
      -i ${file_dwi}_${dti_metric}.nii.gz \
      -d $SCT_DIR/data/PAM50/template/PAM50_t2.nii.gz \
      -w warp_dwi2template.nii.gz \
      -o ${file_dwi}_${dti_metric}_PAM50.nii.gz

    file_out="${PATH_RESULTS}/dwi/${SUBJECT}_dwi_${dti_metric}_PAM50.csv"
    echo "👉 Extracting ${dti_metric} metrics in PAM50 space..."

    for tract in "${tracts[@]}"; do
      sct_extract_metric \
        -i ${file_dwi}_${dti_metric}_PAM50.nii.gz \
        -f $SCT_DIR/data/PAM50/atlas \
        -l ${tract} \
        -combine 1 \
        -method map \
        -vert 2:5 \
        -vertfile $SCT_DIR/data/PAM50/template/PAM50_levels.nii.gz \
        -perslice 1 \
        -o ${file_out} \
        -append 1
    done
  ) &
  pids+=($!)
done

# Wait for all parallel jobs and propagate any failure
for pid in "${pids[@]}"; do
  wait "$pid"
done

# Go back to subject folder
cd ..

echo "✅ Done: ${SUBJECT}"

# ==============================================================================
# Check output files
# ==============================================================================
FILES_TO_CHECK=(
  "anat/${SUBJECT}_T2w_label-SC_seg.nii.gz"
  "dwi/${SUBJECT}_rec-average_dwi_label-SC_seg.nii.gz"
  "dwi/${file_dwi}_FA.nii.gz"
  "dwi/${file_dwi}_MD.nii.gz"
  "dwi/warp_template2dwi.nii.gz"
)
for file_to_check in ${FILES_TO_CHECK[@]}; do
  if [[ ! -e $file_to_check ]]; then
    echo "${SUBJECT}/${file_to_check} does not exist" >> $PATH_LOG/_error_check_output_files.log
  fi
done

# ==============================================================================
# End
# ==============================================================================
end=`date +%s`
runtime=$((end-start))
echo
echo "~~~"
echo "SCT version: `sct_version`"
echo "Ran on:      `uname -nsr`"
echo "Duration:    $(($runtime / 3600))hrs $((($runtime / 60) % 60))min $(($runtime % 60))sec"
echo "~~~"