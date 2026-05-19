#!/usr/bin/env python3
"""
Re-derive DTI scalar metrics (FA, MD, RD, AD) from sorted eigenvalue maps
(E1 >= E2 >= E3), as produced by `sct_dmri_compute_dti -evecs 1`.

Used as a sanity check: warping E1/E2/E3 to PAM50 and recomputing the scalars
in template space should match warping the scalars (FA/MD/RD/AD) directly.

Usage:
    python compute_dti_from_evals.py \
        -e1 <prefix>E1.nii.gz \
        -e2 <prefix>E2.nii.gz \
        -e3 <prefix>E3.nii.gz \
        -o  <output_prefix>

Outputs:
    <output_prefix>FA.nii.gz, <output_prefix>MD.nii.gz,
    <output_prefix>RD.nii.gz, <output_prefix>AD.nii.gz
"""

import argparse
import numpy as np
import nibabel as nib


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-e1', required=True, help='Largest eigenvalue map.')
    parser.add_argument('-e2', required=True, help='Middle eigenvalue map.')
    parser.add_argument('-e3', required=True, help='Smallest eigenvalue map.')
    parser.add_argument('-o', required=True, help='Output prefix (e.g. "sub-XX_dwi_from_evals_").')
    args = parser.parse_args()

    e1_img = nib.load(args.e1)
    e1 = e1_img.get_fdata().astype(np.float64)
    e2 = nib.load(args.e2).get_fdata().astype(np.float64)
    e3 = nib.load(args.e3).get_fdata().astype(np.float64)

    md = (e1 + e2 + e3) / 3.0
    ad = e1
    rd = (e2 + e3) / 2.0
    sumsq = e1**2 + e2**2 + e3**2
    with np.errstate(divide='ignore', invalid='ignore'):
        fa = np.sqrt(1.5) * np.sqrt((e1 - md)**2 + (e2 - md)**2 + (e3 - md)**2) / np.sqrt(sumsq)
    fa = np.nan_to_num(fa, nan=0.0, posinf=0.0, neginf=0.0)

    affine, header = e1_img.affine, e1_img.header
    for arr, name in [(fa, 'FA'), (md, 'MD'), (rd, 'RD'), (ad, 'AD')]:
        out = f'{args.o}{name}.nii.gz'
        nib.Nifti1Image(arr.astype(np.float32), affine, header).to_filename(out)
        print(f'Saved {out}')


if __name__ == '__main__':
    main()