#!/bin/bash
#
# Run sct_process_segmentation with the '-normalize-PAM50' flag on whole spine T2w images
#
# Spinal cord segmentation and disc labels from /derivatives are used
# Note: files in /derivatives were created from reoriented and resampled T2w images.
#
# Usage:
#     sct_run_batch -c <PATH_TO_REPO>/etc/config_process_data.json
#
# The following global variables are retrieved from the caller sct_run_batch
# but could be overwritten by uncommenting the lines below:
# PATH_DATA_PROCESSED="~/data_processed"
# PATH_RESULTS="~/results"
# PATH_LOG="~/log"
# PATH_QC="~/qc"
#
# Authors: Jan Valosek, Sandrine Bedard, Julien Cohen-Adad
#

# Uncomment for full verbose
set -x

# Immediately exit if error
set -e -o pipefail

# Exit if user presses CTRL+C (Linux) or CMD+C (OSX)
trap "echo Caught Keyboard Interrupt within script. Exiting now.; exit" INT

# Print retrieved variables from the sct_run_batch script to the log (to allow easier debug)
echo "Retrieved variables from from the caller sct_run_batch:"
echo "PATH_DATA: ${PATH_DATA}"
echo "PATH_DATA_PROCESSED: ${PATH_DATA_PROCESSED}"
echo "PATH_RESULTS: ${PATH_RESULTS}"
echo "PATH_LOG: ${PATH_LOG}"
echo "PATH_QC: ${PATH_QC}"

# Save script path
#path_source=$(dirname $PATH_DATA)
#PATH_DERIVATIVES="${path_source}/labels"


# CONVENIENCE FUNCTIONS
# ======================================================================================================================
# Check if manual spinal cord segmentation file already exists. If it does, copy it locally.
# If it doesn't, perform automatic spinal cord segmentation
segment_if_does_not_exist() {
  local file="$1"
  # Update global variable with segmentation file name
  FILESEG="${file}_seg"
  FILESEGMANUAL="${PATH_DATA}/derivatives/labels_contrast-agnostic/${SUBJECT}/anat/${file}_label-SC_seg.nii.gz"
  echo
  echo "Looking for manual segmentation: $FILESEGMANUAL"
  if [[ -e $FILESEGMANUAL ]]; then
    echo "Found! Using manual segmentation."
    rsync -avzh $FILESEGMANUAL ${FILESEG}.nii.gz
    sct_qc -i ${file}.nii.gz -s ${FILESEG}.nii.gz -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}
  else
    echo "Not found. Proceeding with automatic segmentation."
    # Segment spinal cord with contrast-agnostic
    sct_deepseg -i ${file}.nii.gz -task seg_sc_contrast_agnostic -largest 1 -qc ${PATH_QC} -qc-subject ${SUBJECT} -o ${FILESEG}.nii.gz
  fi
}


find_segmentation() {
  local file="$1"
  # Update global variable with segmentation file name
  FILESEG="${file}_seg-manual"
  FILESEGMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${file}_label-SC_seg.nii.gz"
  echo
  echo "Looking for manual segmentation: $FILESEGMANUAL"
  if [[ -e $FILESEGMANUAL ]]; then
    echo "Found! Using manual segmentation."
    rsync -avzh $FILESEGMANUAL ${FILESEG}.nii.gz
    sct_qc -i ${file}.nii.gz -s ${FILESEG}.nii.gz -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}
  else
    echo "Not found."
    # Segment spinal cord with contrast-agnostic
    #sct_deepseg -i ${file}.nii.gz -task seg_sc_contrast_agnostic -qc ${PATH_QC} -qc-subject ${SUBJECT} -o ${FILESEG}.nii.gz
  fi
}

# Check if manual label already exists. If it does, generate labeled segmentation from manual disc labels.
# If it doesn't, perform automatic spinal cord labeling
label_if_does_not_exist(){
  local file="$1"
  local file_seg="$2"
  contrast='t2' # TODO: change if we add someday T1w
  # Update global variable with segmentation file name
  FILELABEL="${file}_labels-disc"
  FILELABELMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${FILELABEL}-manual.nii.gz"
  echo "Looking for manual label: $FILELABELMANUAL"
  if [[ -e $FILELABELMANUAL ]]; then
    echo "Found! Using manual labels."
    rsync -avzh $FILELABELMANUAL ${FILELABEL}.nii.gz
    # Generate labeled segmentation from manual disc labels
    sct_image -i ${file}.nii.gz -set-sform-to-qform
    sct_image -i ${file_seg}.nii.gz -set-sform-to-qform
    sct_image -i ${FILELABEL}.nii.gz -set-sform-to-qform
    sct_label_vertebrae -i ${file}.nii.gz -s ${file_seg}.nii.gz -discfile ${FILELABEL}.nii.gz -c ${contrast} -qc ${PATH_QC} -qc-subject ${SUBJECT}
  else
    echo "Not found. Proceeding with automatic labeling."
    # Generate labeled segmentation automatically (no manual disc labels provided)
    sct_label_vertebrae -i ${file}.nii.gz -s ${file_seg}.nii.gz -c ${contrast} -qc ${PATH_QC} -qc-subject ${SUBJECT}
  fi
}

# Retrieve input params and other params
SUBJECT=$1

# get starting time:
start=`date +%s`

# ------------------------------------------------------------------------------
# SCRIPT STARTS HERE
# ------------------------------------------------------------------------------
# Display useful info for the log, such as SCT version, RAM and CPU cores available
sct_check_dependencies -short

# Go to folder where data will be copied and processed
cd $PATH_DATA_PROCESSED

# Copy source T2w images
# Note: we use '/./' in order to include the sub-folder 'ses-0X'
rsync -Ravzh ${PATH_DATA}/./${SUBJECT}/anat/${SUBJECT}_*T2w.* .
rsync -Ravzh ${PATH_DATA}/./${SUBJECT}/anat/${SUBJECT}_*T1w.* .
# Go to subject folder for source images
cd ${SUBJECT}/anat

# ------------------------------------------------------------------------------
# T2w
# ------------------------------------------------------------------------------
# Define variables
# We do a substitution '/' --> '_' in case there is a subfolder 'ses-0X/'
#file_t2="${SUBJECT//[\/]/_}"_T2w
file_t2="${SUBJECT}_T2w"

# Copy SC segmentation from /derivatives
#segment_if_does_not_exist ${file_t2}
#file_t2_seg=$FILESEG
# Find manual segmentation
find_segmentation ${file_t2}
file_t2_seg_manual=$FILESEG

# Run sct_deepseg_sc on T2w image
#sct_deepseg_sc -i ${file_t2}.nii.gz -c t2 -qc ${PATH_QC} -qc-subject ${SUBJECT} -o ${file_t2}_seg_DeepSeg.nii.gz
#file_t2_seg_deepseg="${file_t2}_seg_DeepSeg"
# Create labeling from manual disc labels located at /derivatives
label_if_does_not_exist ${file_t2} ${file_t2_seg_manual} 't2'

# Compute metrics from SC segmentation and normalize them to PAM50 ('-normalize-PAM50' flag)
# Note: '-v 2' flag is used to get all available vertebral levels from PAM50 template. This assures that the output CSV
# files will have the same number of rows, regardless of the subject's vertebral levels.
#mkdir -p ${PATH_RESULTS}/spinalcord_T2w/
mkdir -p ${PATH_RESULTS}/spinalcord/
sct_process_segmentation -i ${file_t2_seg_manual}.nii.gz -vertfile ${file_t2_seg_manual}_labeled.nii.gz -perslice 1 -normalize-PAM50 1 -v 2 -o ${PATH_RESULTS}/spinalcord/${file_t2}_PAM50.csv

# ------------------------------------------------------------------------------
# End
# ------------------------------------------------------------------------------

# Display results (to easily compare integrity across SCT versions)
end=`date +%s`
runtime=$((end-start))
echo
echo "~~~"
echo "SCT version: `sct_version`"
echo "Ran on:      `uname -nsr`"
echo "Duration:    $(($runtime / 3600))hrs $((($runtime / 60) % 60))min $(($runtime % 60))sec"
echo "~~~"
