# PAM50 normalized metrics

This repository contains CSV files of spinal cord morphometric measures in the PAM50 anatomical dimensions computed from 203 healthy adult volunteers.

👉 Please cite this work if you use it or if you are inspired by it:

```
Valošek et al., (2023). A database of the healthy human spinal cord morphometry in the PAM50 template space. NeuroLibre Reproducible Preprints, 17, https://doi.org/10.55458/neurolibre.00017
```

[![DOI](https://neurolibre.org/papers/10.55458/neurolibre.00017/status.svg)](https://doi.org/10.55458/neurolibre.00017)

## Usage

The repository is part of [SCT](https://github.com/spinalcordtoolbox/spinalcordtoolbox/) and is downloaded automatically during SCT installation.

CSV files from this repository are used by the following SCT functions:

- `sct_process_segmentation -normalize-PAM50 1`, example:

```console
sct_process_segmentation -i sub-001_T2w_label-SC_mask.nii.gz -vertfile sub-001_T2w_label-SC_mask_labeled.nii.gz -perslice 1 -normalize-PAM50 1 -o sub-001_T2w_metrics_PAM50.csv
```

- `sct_compute_compression -normalize-hc 1`, example: 

```console
sct_compute_compression -i sub-001_T2w_label-SC_mask.nii.gz -vertfile sub-001_T2w_label-SC_mask_labeled.nii.gz -l sub-001_T2w_label-compression.nii.gz -normalize-hc 1 -o sub-001_T2w_compression_metrics.csv
```

## Dependencies

- [SCT v6.0](https://github.com/spinalcordtoolbox/spinalcordtoolbox/tree/6.0)

The CSV files were generated using [process_data_spine-generic.sh, r20230222](https://github.com/sct-pipeline/dcm-metric-normalization/blob/r20230222/scripts/process_data_spine-generic.sh) script from the [spine-generic/data-multi-subject, r20230223](https://github.com/spine-generic/data-multi-subject/tree/r20230223) dataset.
