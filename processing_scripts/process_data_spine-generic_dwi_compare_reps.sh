#!/bin/bash
#
# Compare DTI metrics (FA, MD, RD, AD) and root mean square (RMS) across a different number of DWI repetitions,
# using spine-generic multi-subject subjects that were acquired with >1 repetition.
#
# Context:
#   - https://github.com/spinalcordtoolbox/PAM50-normalized-metrics/issues/54
#
# Volume structure of the spine-generic multi-rep sites (b=800):
#   1 rep = 1 b=0 + 30 DW directions = 31 volumes
#   amu, strasbourg          : 2 reps                    = 62 volumes
#   mgh, mountSinai, tokyoSkyra: 2 reps + 3 trailing b=0 = 65 volumes
#   -> N reps = first 31*N volumes; the full sequence (2 reps) keeps all volumes
#      (including the trailing b=0 of the 65-volume sites).
#
# For each rep count we run the full DWI preprocessing + DTI fit independently, so
# the comparison reflects what each subset alone would contribute to the database.
#
# Usage:
#     sct_run_batch -c <PATH_TO_REPO>/etc/config_process_data_dwi_compare_reps.json
#
# Output (per-subject CSVs, one per metric and rep count):
#   PATH_RESULTS/dwi_compare_reps/sub-XXXX_dwi_FA_1rep.csv
#   PATH_RESULTS/dwi_compare_reps/sub-XXXX_dwi_FA_2rep.csv
#   ... (FA, MD, RD, AD, RMS) x (1rep, 2rep)
#
# Requires SCT v7.3 or higher and FSL (dtifit, fslmaths).
#
# Authors: Jan Valosek
#

# Only WM for simplicity
tracts=(
  "51"      # white matter (combined)
)

# DTI metrics to extract (RMS = per-volume fit residual, see below)
metrics=(FA MD RD AD RMS)

# 1 rep = 1 b=0 + 30 DW directions
vols_per_rep=31

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

echo "SUBJECT: ${SUBJECT}"

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
    echo "🤖 [$(date '+%Y-%m-%d %H:%M:%S')] Not found. Proceeding with automatic segmentation (sct_deepseg spinalcord)."
    echo "🤖 [$(date '+%Y-%m-%d %H:%M:%S')] ${FILESEG}.nii.gz NOT found --> segmenting automatically with sct_deepseg spinalcord" >> "${PATH_LOG}/${log_prefix}_SC_segmentations.log"
    sct_deepseg spinalcord -i ${file}.nii.gz -o ${FILESEG}.nii.gz -qc ${PATH_QC} -qc-subject ${SUBJECT}
  fi
}

# Check if manual disc labels exist in derivatives/labels. If yes, use them
# directly; if no, run TotalSpineSeg (sct_deepseg spine -label-vert 0) for
# automatic disc labeling.
# Sets global variable FILELABEL (disc-label file without .nii.gz) for -ldisc.
label_if_does_not_exist() {
  local file="$1"
  FILELABEL="${file}_label-discs_dlabel"
  FILELABELMANUAL="${PATH_DERIVATIVES}/labels/${SUBJECT}/anat/${FILELABEL}.nii.gz"
  echo "Looking for manual disc labels: $FILELABELMANUAL"
  if [[ -e $FILELABELMANUAL ]]; then
    echo "✏️ [$(date '+%Y-%m-%d %H:%M:%S')] Found! Using manual disc labels."
    echo "✏️ [$(date '+%Y-%m-%d %H:%M:%S')] ${FILELABEL}.nii.gz found --> using manual disc labels" >> "${PATH_LOG}/T2w_disc_labels.log"
    # Use the manual disc labels directly
    rsync -avzh $FILELABELMANUAL ${FILELABEL}.nii.gz
    sct_qc -i ${file}.nii.gz -s ${FILELABEL}.nii.gz -p sct_label_vertebrae -qc ${PATH_QC} -qc-subject ${SUBJECT}
  else
    echo "🤖 [$(date '+%Y-%m-%d %H:%M:%S')] Not found. Running TotalSpineSeg for disc labeling."
    echo "🤖 [$(date '+%Y-%m-%d %H:%M:%S')] ${FILELABEL}.nii.gz NOT found --> labeling with TotalSpineSeg (sct_deepseg spine)" >> "${PATH_LOG}/T2w_disc_labels.log"
    sct_deepseg spine -i ${file}.nii.gz -label-vert 0 -qc ${PATH_QC} -qc-subject ${SUBJECT}
    FILELABEL="${file}_totalspineseg_discs"
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
# T2w (needed for the T2w-to-template warp used as -initwarp for DWI registration)
# ==============================================================================
cd ${SUBJECT}/anat

file_t2="${SUBJECT}_T2w"
echo "👉 Processing T2w: ${file_t2}"

# Segment spinal cord
segment_if_does_not_exist ${file_t2} 't2'
file_t2_seg=$FILESEG

# Label vertebrae from manual disc labels (or automatically)
label_if_does_not_exist ${file_t2}

# Register T2w to PAM50 template
# Step 1: centermassrot accounts for cord rotation
# Step 2: syn for small-scale deformations
sct_register_to_template \
  -i ${file_t2}.nii.gz \
  -s ${file_t2_seg}.nii.gz \
  -ldisc ${FILELABEL}.nii.gz \
  -c t2 \
  -param step=1,type=seg,algo=centermassrot:step=2,type=seg,algo=syn,metric=MeanSquares,slicewise=1,smooth=0,iter=5 \
  -qc ${PATH_QC} -qc-subject ${SUBJECT}

# Rename warping fields for clarity
mv warp_template2anat.nii.gz warp_template2T2w.nii.gz
mv warp_anat2template.nii.gz warp_T2w2template.nii.gz

# ==============================================================================
# DWI: process each rep count independently and extract metrics
# ==============================================================================
cd ../dwi

# Full sequence to subset from (lives in the dwi/ folder, one level above the rep folders)
file_full="${SUBJECT}_dwi"
# Total number of volumes = full sequence = 2 reps (62 or 65 volumes)
nvol_total=$(wc -w < ${file_full}.bval)
echo "👉 ${SUBJECT}: full sequence has ${nvol_total} volumes"

mkdir -p ${PATH_RESULTS}/dwi_compare_reps

for nrep in 1 2; do

  # Number of volumes to keep for this many reps (the full sequence keeps all volumes)
  if [[ ${nrep} -eq 2 ]]; then
    nvol=${nvol_total}
  else
    nvol=$((vols_per_rep * nrep))
  fi
  qc_subject="${SUBJECT}_${nrep}rep"
  echo "👉 Processing ${nrep} rep(s) (${nvol} volumes)"

  # Set up a working folder for this rep count
  ofolder="${nrep}rep"
  mkdir -p ${ofolder}
  cd ${ofolder}

  file_dwi="${SUBJECT}_dwi"
  file_bval=${file_dwi}.bval
  file_bvec=${file_dwi}.bvec

  # Keep the first ${nvol} volumes / bvals / bvecs from the full sequence
  keepvol=$(seq -s, 0 $((nvol-1)))   # BSD seq adds a trailing comma
  sct_image -i ../${file_full}.nii.gz -keep-vol ${keepvol%,} -o ${file_dwi}.nii.gz
  awk -v n=${nvol} '{out=$1; for(i=2;i<=n;i++) out=out" "$i; print out}' ../${file_full}.bval > ${file_bval}
  awk -v n=${nvol} '{out=$1; for(i=2;i<=n;i++) out=out" "$i; print out}' ../${file_full}.bvec > ${file_bvec}

  # ----------
  # Preprocessing
  # ----------
  # Separate b=0 and DWI volumes; creates *_dwi_mean.nii.gz
  sct_dmri_separate_b0_and_dwi -i ${file_dwi}.nii.gz -bvec ${file_bvec}

  # Create a 35mm mask around the cord to restrict motion correction and speed up processing
  sct_get_centerline -i ${file_dwi}_dwi_mean.nii.gz -c dwi -qc ${PATH_QC} -qc-subject ${qc_subject}
  sct_create_mask -i ${file_dwi}_dwi_mean.nii.gz -p centerline,${file_dwi}_dwi_mean_centerline.nii.gz -size 35mm

  # ----------
  # Motion correction
  # ----------
  # Context for 'metric=CC':
  # https://github.com/sct-pipeline/spine-park/commit/924e332c3b4836baa087ea740a7837120d0b7cbf
  sct_dmri_moco -i ${file_dwi}.nii.gz -bvec ${file_bvec} -m mask_${file_dwi}_dwi_mean.nii.gz -x spline -param metric=CC
  file_dwi=${file_dwi}_moco
  file_dwi_mean=${file_dwi}_dwi_mean

  # ----------
  # Spinal cord segmentation in DWI space
  # ----------
  # Segment automatically per rep count (a manual seg under derivatives of the full rec-average could bias the 1-rep
  # cord mask).
  sct_deepseg spinalcord -i ${file_dwi_mean}.nii.gz -o ${file_dwi_mean}_label-SC_seg.nii.gz -qc ${PATH_QC} -qc-subject ${qc_subject}
  file_dwi_seg=${file_dwi_mean}_label-SC_seg

  # ----------
  # DTI
  # ----------
  # Note: using FSL dtifit as it outputs the sum of squared errors (SSE) of the fit, which we use to compute the root
  #  mean square (RMS)
  # OLS tensor fit; outputs dtifit_FA, dtifit_MD, dtifit_L1/L2/L3 and dtifit_sse (--sse).
  dtifit -k ${file_dwi}.nii.gz -b ${file_bval} -r ${file_bvec} -m ${file_dwi_seg}.nii.gz -o dtifit --sse
  # Axial diffusivity = primary eigenvalue; radial diffusivity = mean of the two minor eigenvalues
  cp dtifit_L1.nii.gz dtifit_AD.nii.gz
  fslmaths dtifit_L2.nii.gz -add dtifit_L3.nii.gz -div 2 dtifit_RD.nii.gz
  # RMS of the per-volume fit residual = sqrt(SSE / nvol).
  # Comparable across rep counts (raw SSE is a sum over volumes, so it scales with nvol).
  fslmaths dtifit_sse.nii.gz -div ${nvol} -sqrt dtifit_RMS.nii.gz

  # ----------
  # Template registration (PAM50 -> DWI), using the T2w-to-template warp as -initwarp
  # ----------
  #   1. centermass: account for cord rotations
  #   2. bsplinesyn (regularized SyN) on seg: fine cord shape adjustment
  #   3. syn on image: fine-tune deformation using image intensity (metric=CC)
  sct_register_multimodal \
    -i $SCT_DIR/data/PAM50/template/PAM50_t2.nii.gz \
    -iseg $SCT_DIR/data/PAM50/template/PAM50_cord.nii.gz \
    -d ${file_dwi_mean}.nii.gz \
    -dseg ${file_dwi_seg}.nii.gz \
    -param step=1,type=seg,algo=centermass:step=2,type=seg,algo=bsplinesyn,metric=MeanSquares,slicewise=1,iter=3:step=3,type=im,algo=syn,metric=CC,iter=3,slicewise=1 \
    -initwarp ../../anat/warp_template2T2w.nii.gz \
    -initwarpinv ../../anat/warp_T2w2template.nii.gz \
    -owarp warp_template2dwi.nii.gz \
    -owarpinv warp_dwi2template.nii.gz \
    -qc ${PATH_QC} -qc-subject ${qc_subject}

  # Warp PAM50 template and WM atlas to DWI space to extract metrics in native DWI space
  sct_warp_template \
    -d ${file_dwi_mean}.nii.gz \
    -w warp_template2dwi.nii.gz \
    -ofolder label_${file_dwi} \
    -qc ${PATH_QC} -qc-subject ${qc_subject}

  # ----------
  # Extract metrics from WM in C2 and C3 vertebral levels (per level)
  # ----------
  for metric in "${metrics[@]}"; do
    file_out="${PATH_RESULTS}/dwi_compare_reps/${SUBJECT}_dwi_${metric}_${nrep}rep.csv"
    echo "👉 Extracting ${metric} (${nrep} rep) in WM at C2-C3..."

    rm -f "${file_out}"
    for tract in "${tracts[@]}"; do
      sct_extract_metric \
        -i dtifit_${metric}.nii.gz \
        -f label_${file_dwi}/atlas \
        -l ${tract} \
        -combine 1 \
        -method map \
        -vertfile label_${file_dwi}/template/PAM50_levels.nii.gz \
        -perlevel 1 \
        -vert 2,3 \
        -o "${file_out}" \
        -append 1
    done
  done

  cd ..
done

# Go back to subject folder
cd ..

echo "✅ Done: ${SUBJECT}"

# ==============================================================================
# Check output files
# ==============================================================================
FILES_TO_CHECK=(
  "anat/${SUBJECT}_T2w_label-SC_seg.nii.gz"
  "dwi/1rep/dtifit_FA.nii.gz"
  "dwi/2rep/dtifit_FA.nii.gz"
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
