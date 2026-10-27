# spine-generic multi-subject: DTI metrics in PAM50 space

Diffusion tensor imaging (DTI) metrics of 202 healthy adults from the [spine-generic multi-subject](https://github.com/spine-generic/data-multi-subject) dataset, interpolated to the PAM50 template space.

## Files

One CSV per subject and metric: `sub-<id>_dwi_<metric>_interpolated_to_PAM50.csv`, with `<metric>`:

- `FA`: fractional anisotropy
- `MD`: mean diffusivity [mm²/s]
- `RD`: radial diffusivity [mm²/s]
- `AD`: axial diffusivity [mm²/s]

Labels (`Label` column): `white matter`, `gray matter`, `dorsal columns`, `lateral funiculi`, `ventral funiculi`, `4,5` (lateral corticospinal tracts), `12,13` (spinal lemniscus).

## Processing

`processing_scripts/process_data_spine-generic_dwi.sh`:

1. DWI motion correction (`sct_dmri_moco`) and DTI fitting (`sct_dmri_compute_dti`).
2. Registration of the PAM50 template and white matter atlas to the DWI (`sct_register_to_template`, then `sct_register_multimodal`).
3. DTI metric extraction in DWI native space per slice and label with weighted average (`wa`) and linear interpolation of each slice to the PAM50 space (`sct_extract_metric -method wa -perslice 1 -normalize-PAM50 1`).
