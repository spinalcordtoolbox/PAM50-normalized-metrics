#!/bin/bash
#
# Run sct_process_segmentation with the '-normalize-PAM50' flag on spine-generic multi-subject T2w images
#
# Spinal cord segmentation and disc labels from /derivatives are used
# Note: files in /derivatives were created from reoriented and resampled T2w images. Thus, the same preprocessing steps
# are used also within this script.
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
path_source=$(dirname $PATH_DATA)
PATH_DERIVATIVES="${PATH_DATA}/derivatives"


# CONVENIENCE FUNCTIONS
# ======================================================================================================================
# Check if manual spinal cord segmentation file already exists. If it does, copy it locally.
# If it doesn't, perform automatic spinal cord segmentation
segment_if_does_not_exist() {
  local file="$1"
  folder_contrast='anat'
  # Update global variable with segmentation file name
  FILESEG="${file}_label-SC_seg"
  FILESEGMANUAL="${PATH_DERIVATIVES}/labels/${SUBJECT}/${folder_contrast}/${FILESEG}.nii.gz"
  echo
  echo "Looking for manual segmentation: $FILESEGMANUAL"
  if [[ -e $FILESEGMANUAL ]]; then
    echo "Found! Using manual segmentation."
    rsync -avzh $FILESEGMANUAL ${FILESEG}.nii.gz
    sct_qc -i ${file}.nii.gz -s ${FILESEG}.nii.gz -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}
  else
    echo "Not found. Proceeding with automatic segmentation."
    # Segment spinal cord
    sct_deepseg spinalcord -i ${file}.nii.gz -o ${FILESEG}.nii.gz -qc ${PATH_QC} -qc-subject ${SUBJECT}
  fi
}

# Check if manual spinal canal segmentation file already exists. If it does, copy it locally.
# If it doesn't, perform automatic spinal cord segmentation
segment_canal_if_does_not_exist() {
  local file="$1"
  folder_contrast='anat'
  # Update global variable with segmentation file name
  FILESEG="${file}_label-canal_seg"
  FILESEGMANUAL="${PATH_DERIVATIVES}/labels/${SUBJECT}/${folder_contrast}/${FILESEG}.nii.gz"
  echo
  echo "Looking for manual segmentation: $FILESEGMANUAL"
  if [[ -e $FILESEGMANUAL ]]; then
    echo "Found! Using manual segmentation."
    rsync -avzh $FILESEGMANUAL ${FILESEG}.nii.gz
    sct_qc -i ${file}.nii.gz -s ${FILESEG}.nii.gz -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}
  else
    echo "Not found. Proceeding with automatic segmentation."
    # Segment spinal cord
    sct_deepseg sc_canal_t2 -i ${file}.nii.gz -o ${FILESEG}.nii.gz -qc ${PATH_QC} -qc-subject ${SUBJECT}
  fi
}


# Check if manual label already exists. If it does, generate labeled segmentation from manual disc labels.
# If it doesn't, perform automatic spinal cord labeling
label_if_does_not_exist(){
  local file="$1"
  local file_seg="$2"
  local contrast="$3"
  # Update global variable with segmentation file name
  FILELABEL="${file}_label-discs_dlabel"
  FILELABELMANUAL="${PATH_DERIVATIVES}/labels/${SUBJECT}/anat/${FILELABEL}.nii.gz"
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
# Go to subject folder for source images
cd ${SUBJECT}/anat

# ------------------------------------------------------------------------------
# T2w
# ------------------------------------------------------------------------------
# Define variables
file_t2="${SUBJECT}_T2w"

# Reorient and resample (to match spine-generic derivatives/labels files)

# Copy SC segmentation from /derivatives
segment_if_does_not_exist ${file_t2} 't2'
file_t2_seg=$FILESEG

# Create labeling from manual disc labels located at /derivatives
label_if_does_not_exist ${file_t2} ${file_t2_seg} 't2'

# Compute metrics from SC segmentation and normalize them to PAM50 ('-normalize-PAM50' flag)
# Note: '-v 2' flag is used to get all available vertebral levels from PAM50 template. This assures that the output CSV
# files will have the same number of rows, regardless of the subject's vertebral levels.
mkdir -p ${PATH_RESULTS}/spinalcord
sct_process_segmentation -i ${file_t2_seg}.nii.gz -vertfile ${file_t2_seg}_labeled.nii.gz -perslice 1 -normalize-PAM50 1 -v 2 -o ${PATH_RESULTS}/spinalcord/${file_t2}_PAM50.csv

# ------------------------------------------------------------------------------
# Segment canal
segment_canal_if_does_not_exist ${file_t2} 't2'
file_t2_seg_canal=$FILESEG
mkdir -p ${PATH_RESULTS}/canal
sct_process_segmentation -i ${file_t2_seg_canal}.nii.gz -vertfile ${file_t2_seg}_labeled.nii.gz -perslice 1 -normalize-PAM50 1 -v 2 -o ${PATH_RESULTS}/canal/${file_t2}_PAM50.csv


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
