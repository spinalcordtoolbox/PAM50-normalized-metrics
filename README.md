# PAM50 normalized metrics

This repository contains morphometric measures in the [PAM50](https://pubmed.ncbi.nlm.nih.gov/29061527/) anatomical dimensions computed from 203 healthy adult volunteers.

 - `spinal_cord` directory - CSV files of spinal cord morphometric measures

👉 Please cite this work if you use it or if you are inspired by it:

```
Valošek, Bédard et al., (2024). A database of the healthy human spinal cord morphometry in the PAM50 template space. Imaging Neuroscience, 2 1–15, https://doi.org/10.1162/imag_a_00075
```

[![DOI](https://img.shields.io/badge/ImagingNeuroscience-10.1162/imag_a_00075-status.svg)](https://doi.org/10.1162/imag_a_00075)

👉 For interactive figures, please visit the [NeuroLibre preprint](https://preprint.neurolibre.org/10.55458/neurolibre.00017/):

```
Valošek, Bédard et al., (2023). A database of the healthy human spinal cord morphometry in the PAM50 template space. NeuroLibre Reproducible Preprints, 17, https://doi.org/10.55458/neurolibre.00017
```

[![DOI](https://neurolibre.org/papers/10.55458/neurolibre.00017/status.svg)](https://doi.org/10.55458/neurolibre.00017)

## Usage

### As part of SCT

The repository is downloaded automatically during the [SCT](https://github.com/spinalcordtoolbox/spinalcordtoolbox/) installation.

CSV files from this repository are used by the following SCT functions:

- `sct_process_segmentation -normalize-PAM50 1`, example:

```console
sct_process_segmentation -i sub-001_T2w_label-SC_mask.nii.gz -vertfile sub-001_T2w_label-SC_mask_labeled.nii.gz -perslice 1 -normalize-PAM50 1 -o sub-001_T2w_metrics_PAM50.csv
```

- `sct_compute_compression -normalize-hc 1`, example: 

```console
sct_compute_compression -i sub-001_T2w_label-SC_mask.nii.gz -vertfile sub-001_T2w_label-SC_mask_labeled.nii.gz -l sub-001_T2w_label-compression.nii.gz -normalize-hc 1 -o sub-001_T2w_compression_metrics.csv
```

### Standalone usage

If you want to use the morphometric measures outside of SCT in your own research, you can simply download this repository and use the CSV files directly:

```console
git clone https://github.com/spinalcordtoolbox/PAM50-normalized-metrics.git
```

## Details

### `spinal_cord` directory

The CSV files were generated using [process_data_spine-generic.sh, r20230222](https://github.com/sct-pipeline/dcm-metric-normalization/blob/r20230222/scripts/process_data_spine-generic.sh) script from the [spine-generic/data-multi-subject, r20230223](https://github.com/spine-generic/data-multi-subject/tree/r20230223) dataset.
Spinal cord segmentation masks from [derivatives/labels](https://github.com/spine-generic/data-multi-subject/tree/r20230223/derivatives/labels) were used (files with the `seg-manual.nii.gz` suffix). These masks were produced by `sct_deepseg_sc` and manually corrected.
[SCT v6.0](https://github.com/spinalcordtoolbox/spinalcordtoolbox/tree/6.0) was used to compute the morphometric measures.

