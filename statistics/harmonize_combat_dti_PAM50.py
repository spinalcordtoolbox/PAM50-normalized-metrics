#!/usr/bin/env python3
#
# TL;DR: for each feature (PAM50 slice × DTI metric × WM/GM region), ComBat models the value as:
#               y = α + Xβ + γ_scanner + δ_scanner · ε
#
#         - α is the mean of the feature across all subjects
#         - Xβ is the effect of the covariates (age, sex), estimated on the healthy controls
#         - γ is the additive scanner effect (shift) and δ the multiplicative one (scale)
#
#
# Harmonize per-slice DTI metrics (FA, MD, RD, AD) in PAM50 space across MRI scanners with ComBat
# (Johnson et al., 2007; Fortin et al., 2017; neuroCombat implementation), and optionally apply the
# harmonization learned on a healthy multi-scanner dataset (spine-generic) to a new dataset.
# Inspired by Middleton et al., 2023, Sci Rep, doi:10.1038/s41598-023-46465-6 (per-level ComBat);
# here, the PAM50 per-slice normalization is used instead of per-level averages.
#
# Design:
#   - Features: every PAM50 slice × DTI metric × region within -vert-range (e.g., C2-C5).
#     A single ComBat model is fitted on all features, so the empirical Bayes priors pool information
#     across slices.
#   - Batch: scanner model (participants.tsv column, -batch-col), e.g. Siemens Prisma, Prisma-fit, Skyra,
#     Philips Achieva-dStream, Ingenia, GE MR750 for spine-generic; subjects with the same model from
#     different sites share a batch. ComBat is fitted on the training dataset only (healthy subjects),
#     with age and sex as biological covariates to preserve.
#   - ComBat cannot be fitted with missing values, so only training subjects covering all slices of
#     -vert-range are used for fitting. Once fitted, the correction is independent for each feature
#     (slice), so the estimates are then applied to all other subjects on the slices they cover:
#       * training subjects with partial coverage,
#       * new-dataset subjects (optional) scanned on a scanner model present in the training dataset.
#         Note that diagnosis is never used to estimate the scanner effects, so disease effects are
#         preserved. No covariates (sex, age) are needed for them (the average covariate effect of the
#         training dataset is used, as in neuroCombatFromTraining).
#     Subjects on a scanner model absent from the training dataset (or from a batch with <2 subjects
#     for fitting) are kept unharmonized and flagged in the output, unless the new dataset provides healthy
#     controls on that scanner model (-new-controls): these are then harmonized against the training dataset
#     with a reference-batch ComBat (the harmonized training dataset is the reference and stays unchanged;
#     no covariates), and the resulting correction is applied to all new subjects on that scanner model.
#     Note that the controls of that batch then match the training dataset by construction.
#
# Why not ComBat across datasets directly: when a site contains only patients and the other
# sites only healthy controls, site and diagnosis are confounded and ComBat would remove the
# disease effect together with the site effect.
#
# ComBat does the following with the data:
#       1. Remove what you want to keep. It first models the biological effects listed as covariates (i.e., age and sex).
#          These are set aside so they aren't mistaken for scanner effects.
#       2. Measure each scanner's "fingerprint". For each batch (i.e., scanner model) it estimates two numbers:
#               - shift: how much this scanner reads higher or lower than average (e.g., MD is 0.05 higher on Philips);
#               - scale: how much more or less spread its values have.
#          It does this separately for every feature (i.e., every slice × tract × metric).
#       3. Borrow strength (the "empirical Bayes" part). With only 4 subjects on one scanner, its fingerprint estimate
#           is noisy. ComBat assumes the shifts across all features of one scanner look roughly alike, and pulls extreme
#           estimates toward that common pattern. This is why it works with small batches.
#       4. Each value is adjusted: remove the scanner's shift, undo its extra spread, then put the age and sex effects back.
#
# The result is what each subject would roughly look like on an "average scanner". This average is the pooled reference,
# not any one specific scanner.
#
# Outputs (in -path-out):
#   combat_perslice.csv                  : long table with raw and harmonized per-slice values
#   combat_between_scanner_cov.csv       : between-scanner CoV per slice (training dataset), raw vs harmonized
#   combat_scanners_<label>.png          : per-scanner means of the training dataset, raw vs harmonized
#   combat_datasets_grid_<label>.png     : raw (top) vs harmonized (bottom), training dataset as mean ± SD,
#                                          new-dataset sites as lines (site means), levels and n in the legend;
#                                          panel titles give the between-scanner-model CoV of the training
#                                          dataset (mean over slices; across scanner models whatever -batch-col)
#   combat_datasets_grid_subjects_<label>.png : same, with one line per new-dataset subject instead of site means
#
# Usage:
#     python statistics/harmonize_combat_dti_PAM50.py \
#         -path-results-train <spine-generic>/results/dwi_interpolation_to_PAM50_map \
#         -participants-train <data-multi-subject>/participants.tsv \
#         -path-results-new   <oklahoma_standford>/results/dwi_interpolation_to_PAM50_map \
#                             <dcm-olomouc>/results/dwi_interpolation_to_PAM50_map \
#         -path-data-new      <oklahoma_standford BIDS dataset> <dcm-olomouc BIDS dataset> \
#         -site-names "sub-CSM*=OU" "sub-NSDCM*=Stanford" "sub-UTCSM*=UTSW" \
#                     "*_ses-M0=Olomouc pre-op" "*_ses-M6=Olomouc post-op" \
#         -site-first-level OU=4 Stanford=4 UTSW=4 \
#         -path-out           <path>/figures/combat/interpolated_map
#
# Example with new healthy controls (CSVs only, no BIDS) on a scanner model absent from spine-generic:
#     python statistics/harmonize_combat_dti_PAM50.py \
#         -path-results-train <spine-generic>/results/dwi_PAM50_map \
#         -participants-train <data-multi-subject>/participants.tsv \
#         -path-results-new   <NSHC CSVs> -path-data-new none \
#         -new-models "sub-NSHC*=SIGNA_Premier" -new-controls "sub-NSHC*" \
#         -labels "white matter" -vert-range 3 5 -site-names "sub-NSHC*=Stanford HC" \
#         -path-out           <path>/combat_reference
#
# Requires: neuroCombat (pip install neuroCombat)
#
# Author: Jan Valosek
#

import os
import re
import textwrap
import fnmatch
import glob
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from neuroCombat import neuroCombat

# Shared with the group-level script (same folder)
from generate_figures_dti_PAM50_all_subjects import (DTI_METRICS, METRIC_TO_AXIS, LABEL_DISPLAY_NAMES,
                                                     load_dti_csvs, get_vert_indices, vert_num_to_label)

LABELS_FONT_SIZE = 16
TICKS_FONT_SIZE = LABELS_FONT_SIZE - 2
SLICE = 'Slice (I->S)'
# Training dataset (mean ± SD band) and default new-dataset site colors (in -site-names order)
TRAIN_COLOR = 'tab:gray'
SITE_COLORS = ['tab:orange', 'tab:red', 'tab:purple', 'tab:blue', 'tab:green', 'tab:brown', 'tab:pink']
LINESTYLES = {'solid': '-', 'dashed': '--', 'dotted': ':', 'dashdot': '-.'}
# -batch-col (participants.tsv) → corresponding key of the BIDS JSON sidecars of the new subjects
BATCH_COL_TO_JSON_KEY = {'manufacturers_model_name': 'ManufacturersModelName', 'manufacturer': 'Manufacturer'}


def get_parser():
    parser = argparse.ArgumentParser(
        description='Harmonize per-slice DTI metrics in PAM50 space across scanner models with ComBat, '
                    'fitted on a healthy training dataset and optionally applied to a new dataset.')
    parser.add_argument('-path-results-train', required=True, nargs='+',
                        help='Folder(s) with the training dataset sct_extract_metric CSVs '
                             '(*_dwi_{FA,MD,RD,AD}_*PAM50.csv), e.g. spine-generic. Several folders are pooled '
                             'into one training dataset (e.g. spine-generic + other healthy controls).')
    parser.add_argument('-participants-train', required=True, nargs='+',
                        help='participants.tsv of the training dataset(s) with participant_id, the -batch-col '
                             'column and the -covariates columns. Several files are concatenated.')
    parser.add_argument('-covariates', nargs='*', default=['age', 'sex'],
                        help='Biological covariates to preserve (columns of -participants-train); sex is '
                             'categorical, the others continuous. Pass "-covariates" without value for no '
                             'covariates (e.g. when they are unknown for part of the training dataset). '
                             'Default: age sex.')
    parser.add_argument('-batch-col', default='manufacturers_model_name',
                        help='Column of -participants-train defining the ComBat batch, e.g. manufacturer '
                             '(vendor). For the new subjects, the corresponding BIDS JSON key is read '
                             f'({BATCH_COL_TO_JSON_KEY}). Default: manufacturers_model_name (scanner model).')
    parser.add_argument('-path-results-new', nargs='+', default=None,
                        help='Folder(s) with the new dataset(s) sct_extract_metric CSVs (optional).')
    parser.add_argument('-path-data-new', nargs='+', default=None,
                        help='BIDS folder of each new dataset (same order as -path-results-new); the scanner '
                             'model of each subject (or session) is read from ManufacturersModelName in its '
                             'DWI JSON sidecar. Required with -path-results-new; "none" for a dataset '
                             'without BIDS folder (then use -new-models).')
    parser.add_argument('-new-models', nargs='+', default=[],
                        help='Scanner model of new subjects, as <pattern>=<model> (pattern matched against the '
                             'participant ID), e.g. "sub-NSHC*=SIGNA_Premier". Overrides the BIDS JSON '
                             'sidecars; needed for datasets without BIDS folder.')
    parser.add_argument('-new-controls', nargs='+', default=[],
                        help='Participant ID pattern(s) of healthy controls in the new dataset(s), e.g. '
                             '"sub-NSHC*". For their scanner model(s) absent from the training dataset, they '
                             'are harmonized against the training dataset (reference-batch ComBat), and the '
                             'correction is applied to all new subjects on that scanner model.')
    parser.add_argument('-labels', nargs='+', default=None,
                        help='Tract/region label(s) to harmonize, e.g. "white matter" (when a new dataset '
                             'provides only these). Default: all labels of the training dataset.')
    parser.add_argument('-site-names', nargs='+', default=[],
                        help='Display names of the new-dataset sites (plotted as separate lines), as '
                             '<pattern>=<name>, where <pattern> is matched against the participant ID '
                             '(sub-XX or sub-XX_ses-YY), e.g. "sub-CSM*=OU" "*_ses-M0=Olomouc pre-op". A '
                             'pattern without "sub-" or "*" is a subject prefix (the letters after "sub-"), '
                             'e.g. CSM=OU. The first matching pattern wins; the order sets the legend order.')
    parser.add_argument('-site-first-level', nargs='+', default=[],
                        help='First vertebral level to plot per site, as <name>=<level>, e.g. OU=4 (to hide '
                             'levels covered by only a few subjects). Default: the first level of -vert-range.')
    parser.add_argument('-site-colors', nargs='+', default=[],
                        help='Line color per site, as <name>=<color>, e.g. "Olomouc pre-op=tab:blue".')
    parser.add_argument('-site-linestyles', nargs='+', default=[],
                        help='Line style per site, as <name>=<style> (solid, dashed, dotted, dashdot), e.g. '
                             '"Olomouc post-op=dashed". Default: solid.')
    parser.add_argument('-footnote', default=None,
                        help='Footnote below the legend of the grid figures, e.g. abbreviations: '
                             '"tSCI = Traumatic spinal cord injury; DCM = Degenerative cervical myelopathy".')
    parser.add_argument('-dataset-labels', nargs=2, default=['Spine-generic', 'OU/Stanford/UTSW'],
                        metavar=('TRAIN', 'NEW'), help='Display names of the training and new datasets.')
    parser.add_argument('-vert-range', nargs=2, type=int, default=[2, 5], metavar=('FIRST', 'LAST'),
                        help='Inclusive range of vertebral levels to harmonize (C1=1 … C7=7). Training '
                             'subjects have to cover all its slices to be used for fitting. Default: 2 5.')
    parser.add_argument('-labels-to-plot', nargs='+', default=['white matter'],
                        help='Tract/region label(s) to plot. All labels are harmonized. '
                             'Default: "white matter".')
    parser.add_argument('-path-out', required=True, help='Output folder.')
    return parser


def normalize_model_name(name):
    """Make scanner model names comparable between datasets, e.g. 'MAGNETOM Prisma' → 'Prisma',
    'SIGNA_Premier' → 'SIGNA-Premier', 'Prisma_fit' → 'Prisma-fit'."""
    name = re.sub(r'^MAGNETOM\s+', '', str(name).strip())
    return re.sub(r'[\s_]+', '-', name)


def get_site_prefix(participant_id):
    """'sub-NSDCM032' → 'NSDCM'."""
    m = re.match(r'sub-([A-Za-z]+)', participant_id)
    return m.group(1) if m else participant_id


def parse_site_patterns(site_names):
    """-site-names entries → [(pattern, name)]; a bare prefix (e.g. 'CSM') becomes 'sub-CSM*'."""
    patterns = []
    for entry in site_names:
        pattern, name = entry.split('=', 1)
        if '*' not in pattern and not pattern.startswith('sub-'):
            pattern = f'sub-{pattern}*'
        patterns.append((pattern, name))
    return patterns


def get_site(participant_id, site_patterns):
    """Display name of the first -site-names pattern matching the participant ID; else its prefix."""
    for pattern, name in site_patterns:
        if fnmatch.fnmatch(participant_id, pattern):
            return name
    return get_site_prefix(participant_id)


def get_models_from_bids(path_data, participant_ids, json_key='ManufacturersModelName'):
    """Read json_key (ManufacturersModelName by default) from the DWI JSON sidecar of each subject (or
    session, for participant IDs of the form sub-XX_ses-YY)."""
    models = {}
    for sub in participant_ids:
        # sub-018_ses-M0 → <path_data>/sub-018/ses-M0/dwi/
        files = sorted(glob.glob(os.path.join(path_data, *sub.split('_'), 'dwi', f'{sub}*_dwi.json')))
        if not files:
            print(f'  Warning: no DWI JSON sidecar for {sub}; it will not be harmonized')
            continue
        with open(files[0]) as f:
            models[sub] = normalize_model_name(json.load(f).get(json_key, 'n/a'))
    return models


def to_wide(df):
    """Long per-slice table → subjects × (metric, Label, slice) matrix (NaN where a slice is not covered)."""
    return df.pivot_table(index='participant_id', columns=['metric', 'Label', SLICE], values='value')


def apply_combat_estimates(dat, batch_idx, estimates):
    """
    Apply ComBat estimates fitted on a training dataset to new samples (same math as
    neuroCombat.neuroCombatFromTraining, which in neuroCombat 0.2.12 cannot match batch names because
    the stored batches are integer codes). Each feature is corrected independently, so NaN (slices
    not covered by a subject) stay NaN without affecting the other features.

    Args:
        dat       (np.ndarray): features × samples.
        batch_idx (np.ndarray): index of each sample's batch in the training batch levels.
        estimates (dict): neuroCombat(...)['estimates'].
    """
    var_pooled = estimates['var.pooled']                                # features × 1
    # Grand mean + average covariate effect of the training samples (no covariates for new samples)
    stand_mean = (estimates['stand.mean'][:, 0] + estimates['mod.mean'].mean(axis=1))[:, None]
    gamma = estimates['gamma.star'][batch_idx, :].T                     # features × samples
    delta = estimates['delta.star'][batch_idx, :].T
    s_data = (dat - stand_mean) / np.sqrt(var_pooled)
    return ((s_data - gamma) / np.sqrt(delta)) * np.sqrt(var_pooled) + stand_mean


def fit_to_reference(wide_ref, wide_ctrl):
    """
    ComBat with the (harmonized) training dataset as reference batch, kept unchanged, and the healthy
    controls of a scanner model absent from it as the only other batch. No covariates (the controls'
    age/sex may be unknown).

    Returns:
        estimates (dict): neuroCombat(...)['estimates'].
        idx       (int): index of the controls' batch in the estimates.
    """
    covars = pd.DataFrame({'batch': ['reference'] * len(wide_ref) + ['controls'] * len(wide_ctrl)})
    # neuroCombat 0.2.12 uses np.int (removed in numpy 1.24) when a reference batch is given
    if not hasattr(np, 'int'):
        np.int = int
    idx = int(np.searchsorted(np.unique(covars['batch']), 'controls'))
    # neuroCombat modifies covars in place, hence the copy
    out = neuroCombat(dat=pd.concat([wide_ref, wide_ctrl]).T.values, covars=covars.copy(), batch_col='batch',
                      ref_batch='reference')
    return out['estimates'], idx


def harmonize(wide_train, covars_train, wide_new, batch_new, controls_new=(), covariates=('age', 'sex')):
    """
    Fit ComBat on the training subjects covering all features, then apply the estimates to all other
    subjects whose batch (scanner model, by default) was fitted. New batches absent from the training
    dataset that contain healthy controls (controls_new) are harmonized against the harmonized training
    dataset (fit_to_reference).

    Returns:
        wide_harm (pd.DataFrame): harmonized values, same layout as the concatenated inputs.
        status    (pd.Series): per subject — 'fit', 'applied', 'fit (reference)', 'applied (reference)' or
                               'not harmonized'.
    """
    complete = wide_train.notna().all(axis=1)
    covars_fit = covars_train.loc[wide_train.index[complete]]
    # ComBat needs ≥2 subjects per batch to estimate the batch variance
    n_per_batch = covars_fit['batch'].value_counts()
    print(f'  Scanner models (n={len(n_per_batch)}) in the training dataset: {", ".join(n_per_batch.index)}')
    small = n_per_batch[n_per_batch < 2].index
    if len(small):
        print(f'  Batch(es) with <2 subjects covering all slices (not harmonized): {", ".join(small)}')
    fit = complete & ~covars_train.loc[wide_train.index, 'batch'].isin(small)
    covars_fit = covars_train.loc[wide_train.index[fit]]
    print(f'  Fitting ComBat on {fit.sum()} training subject(s) × {wide_train.shape[1]} features '
          f'({covars_fit["batch"].nunique()} batches); {(~fit).sum()} training subject(s) with partial '
          f'coverage get the fitted correction applied')

    # Batch = scanner model (see -batch-col); covariates (sex, age by default) are biological effects to preserve
    covariates = list(covariates)
    out = neuroCombat(dat=wide_train[fit].T.values,
                      covars=covars_fit[['batch'] + covariates].reset_index(drop=True),
                      batch_col='batch', categorical_cols=[c for c in covariates if c == 'sex'],
                      continuous_cols=[c for c in covariates if c != 'sex'])
    batch_levels = np.unique(covars_fit['batch'])   # neuroCombat encodes batches in this (sorted) order

    wide_raw = pd.concat([wide_train, wide_new]) if wide_new is not None else wide_train
    batch_all = pd.concat([covars_train.loc[wide_train.index, 'batch'],
                           batch_new.reindex(wide_new.index) if wide_new is not None else pd.Series(dtype=str)])
    wide_harm = wide_raw.copy()
    status = pd.Series('not harmonized', index=wide_raw.index)

    wide_harm.loc[wide_train.index[fit]] = out['data'].T
    status[wide_train.index[fit]] = 'fit'
    to_apply = wide_raw.index[~wide_raw.index.isin(wide_train.index[fit]) & batch_all.isin(batch_levels).values]
    if len(to_apply):
        idx = np.searchsorted(batch_levels, batch_all[to_apply].values)
        wide_harm.loc[to_apply] = apply_combat_estimates(wide_raw.loc[to_apply].T.values, idx,
                                                         out['estimates']).T
        status[to_apply] = 'applied'

    # New batches absent from the training dataset, with healthy controls
    if wide_new is not None:
        wide_ref = wide_harm.loc[wide_train.index[fit]]
        for batch in sorted(set(batch_all[list(controls_new)]) - set(batch_levels)):
            subs = batch_all.index[(batch_all == batch).values].intersection(wide_new.index)
            ctrl = [s for s in controls_new if s in subs and wide_raw.loc[s].notna().all()]
            if len(ctrl) < 2:
                print(f'  {batch}: <2 healthy controls covering all slices; not harmonized')
                continue
            print(f'  {batch}: harmonizing {len(ctrl)} healthy control(s) covering all slices against '
                  f'{len(wide_ref)} training subject(s) (reference-batch ComBat); correction applied to '
                  f'{len(subs) - len(ctrl)} other subject(s) on {batch}')
            estimates, idx = fit_to_reference(wide_ref, wide_raw.loc[ctrl])
            wide_harm.loc[subs] = apply_combat_estimates(wide_raw.loc[subs].T.values,
                                                         np.full(len(subs), idx), estimates).T
            status[subs] = 'applied (reference)'
            status[ctrl] = 'fit (reference)'
    return wide_raw, wide_harm, batch_all, status


def to_long(wide_raw, wide_harm):
    raw = wide_raw.stack(['metric', 'Label', SLICE], future_stack=True).rename('value_raw')
    harm = wide_harm.stack(['metric', 'Label', SLICE], future_stack=True).rename('value_harmonized')
    return pd.concat([raw, harm], axis=1).dropna(subset=['value_raw']).reset_index()


def between_scanner_cov(df):
    """CoV (std/mean) of the per-scanner means, per slice/label/metric, raw and harmonized."""
    means = df.groupby([SLICE, 'VertLevel', 'Label', 'metric', 'batch'])[['value_raw', 'value_harmonized']].mean()
    cov = means.groupby([SLICE, 'VertLevel', 'Label', 'metric']).agg(lambda x: x.std() / x.mean())
    return cov.rename(columns={'value_raw': 'cov_raw', 'value_harmonized': 'cov_harmonized'}).reset_index()


def style_ax(ax, slice_to_level):
    """Shared styling; x-axis inverted (rostral on the left) with vertebral level annotations."""
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', which='major', labelsize=TICKS_FONT_SIZE)
    disc_slices, mid_slices, vert_at_mid = get_vert_indices(slice_to_level)
    xmin, xmax = slice_to_level[SLICE].min(), slice_to_level[SLICE].max()
    ax.set_xlim(xmax, xmin)
    for s in disc_slices:
        ax.axvline(s, color='black', linestyle='--', alpha=0.5, zorder=0)
    for s, v in zip(mid_slices, vert_at_mid):
        ax.text(s, 0.01, vert_num_to_label(v), transform=ax.get_xaxis_transform(),
                ha='center', va='bottom', fontsize=TICKS_FONT_SIZE)


def share_ylim(axes_col):
    lims = [ax.get_ylim() for ax in axes_col]
    for ax in axes_col:
        ax.set_ylim(min(l[0] for l in lims), max(l[1] for l in lims))


def plot_scanners(df_train, label, slice_to_level, path_out):
    """Rows: raw / harmonized; columns: DTI metrics; one line per scanner model (training dataset)."""
    df = df_train[df_train['Label'] == label]
    batches = sorted(df['batch'].unique())
    colors = plt.get_cmap('tab20')(np.linspace(0, 1, 20))[:len(batches)]
    fig, axes = plt.subplots(2, len(DTI_METRICS), figsize=(5 * len(DTI_METRICS), 9), sharex=True)
    for j, metric in enumerate(DTI_METRICS):
        means = (df[df['metric'] == metric]
                 .groupby(['batch', SLICE])[['value_raw', 'value_harmonized']].mean())
        for i, (col, title) in enumerate([('value_raw', 'Raw'), ('value_harmonized', 'Harmonized')]):
            ax = axes[i, j]
            for b, c in zip(batches, colors):
                n = df[df['batch'] == b]['participant_id'].nunique()
                m = means.loc[b][col]
                ax.plot(m.index, m.values, color=c, linewidth=1.5, label=f'{b} (n={n})')
            ax.set_title(title, fontsize=LABELS_FONT_SIZE)
            ax.set_ylabel(METRIC_TO_AXIS[metric], fontsize=TICKS_FONT_SIZE)
            if i == 1:
                ax.set_xlabel('Axial Slice #', fontsize=TICKS_FONT_SIZE)
        share_ylim(axes[:, j])
        for ax in axes[:, j]:
            style_ax(ax, slice_to_level)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0), ncol=5, frameon=False,
               fontsize=TICKS_FONT_SIZE - 2)
    fig.suptitle(f'{LABEL_DISPLAY_NAMES.get(label, label)}: mean per scanner model',
                 fontsize=LABELS_FONT_SIZE + 2)
    plt.tight_layout()
    fname = os.path.join(path_out, f'combat_scanners_{_slug(label)}.png')
    plt.savefig(fname, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Figure saved: {fname}')


def plot_datasets_grid(df, label, slice_to_level, dataset_labels, sites, site_style, site_first_level,
                       path_out, per_subject=False, cov=None, footnote=None):
    """
    Rows: raw / harmonized; columns: DTI metrics. Training dataset as mean ± SD band, new-dataset sites
    as site means (lines only), or as one line per subject (per_subject). The legend gives the vertebral
    levels plotted and the number of subjects of each site.

    Args:
        sites            (list[str]): site display names, in legend order.
        site_style       (dict): site → (color, linestyle).
        site_first_level (dict): site → first vertebral level to plot.
        per_subject      (bool): plot individual subjects of the new-dataset sites instead of site means.
        cov              (pd.DataFrame): between-scanner-model CoV of the training dataset (metric × cov_raw,
                                         cov_harmonized), shown in the panel titles (optional).
        footnote         (str): text below the legend, e.g. abbreviations (optional).
    """
    df = df[df['Label'] == label]
    train = df[df['dataset'] == 'train']
    new = df[df['dataset'] == 'new']

    def levels_str(d):
        return f'{vert_num_to_label(int(d["VertLevel"].min()))}–{vert_num_to_label(int(d["VertLevel"].max()))}'

    # Panel width growing with the number of vertebral levels: 5.5 in for 4 levels (e.g. C2-C5), 4.1 in for 2
    width_scale = (slice_to_level['VertLevel'].nunique() + 4) / 8
    narrow = width_scale < 1
    fig, axes = plt.subplots(2, len(DTI_METRICS), figsize=(5.5 * width_scale * len(DTI_METRICS), 9), sharex=True)
    for j, metric in enumerate(DTI_METRICS):
        tm = train[train['metric'] == metric]
        for i, (col, title) in enumerate([('value_raw', 'Raw'), ('value_harmonized', 'Harmonized')]):
            ax = axes[i, j]
            t = tm.groupby(SLICE)[col].agg(['mean', 'std'])
            ax.plot(t.index, t['mean'], color=TRAIN_COLOR, linewidth=2,
                    label=f'{dataset_labels[0]}, mean ± SD, {levels_str(tm)}, n={tm["participant_id"].nunique()}')
            ax.fill_between(t.index, t['mean'] - t['std'], t['mean'] + t['std'], color=TRAIN_COLOR,
                            alpha=0.2, linewidth=0)
            for site in sites:
                s = new[(new['metric'] == metric) & (new['site'] == site)]
                s = s[s['VertLevel'] >= site_first_level.get(site, 0)]
                if s.empty:
                    continue
                harmonized = (s['status'] != 'not harmonized').all()
                # Sites that could not be harmonized (scanner model absent from the training dataset): raw row only
                if not harmonized and col == 'value_harmonized':
                    continue
                color, linestyle = site_style[site]
                site_label = (f'{site}, {levels_str(s)}, n={s["participant_id"].nunique()}'
                              f'{"" if harmonized else ", raw only (not harmonized)"}')
                if per_subject:
                    for k, (_, sub) in enumerate(s.groupby('participant_id')):
                        m = sub.set_index(SLICE)[col].sort_index()
                        ax.plot(m.index, m.values, color=color, linestyle=linestyle, linewidth=1, alpha=0.7,
                                label=site_label if k == 0 else None)
                else:
                    m = s.groupby(SLICE)[col].mean()
                    ax.plot(m.index, m.values, color=color, linestyle=linestyle, linewidth=2, label=site_label)
            cov_str = f' (CoV {cov.loc[metric, col.replace("value", "cov")] * 100:.1f}%)' if cov is not None else ''
            ax.set_title(f'{title}: {metric}{cov_str}', fontsize=LABELS_FONT_SIZE)
            ax.set_ylabel(METRIC_TO_AXIS[metric], fontsize=TICKS_FONT_SIZE)
            if i == 1:
                ax.set_xlabel('Axial Slice #', fontsize=TICKS_FONT_SIZE)
        share_ylim(axes[:, j])
        for ax in axes[:, j]:
            style_ax(ax, slice_to_level)
    # Raw row: includes the sites that could not be harmonized
    handles, labels = axes[0, 0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0), ncol=2 if narrow else 3,
                        frameon=False,
                        fontsize=TICKS_FONT_SIZE)
    for line in legend.get_lines():
        line.set_linewidth(2)
        line.set_alpha(1)
    suptitle = LABEL_DISPLAY_NAMES.get(label, label)
    if cov is not None:
        cov_title = (f'CoV: between-scanner-model coefficient of variation of {dataset_labels[0]} '
                     f'(mean over slices)')
        suptitle += '\n' + textwrap.fill(cov_title, int(140 * width_scale))
    fig.suptitle(suptitle, fontsize=LABELS_FONT_SIZE + 2)
    plt.tight_layout()
    if footnote:
        # Below the legend, left-aligned with the figure
        fig.canvas.draw()
        legend_bbox = legend.get_window_extent().transformed(fig.transFigure.inverted())
        fig.text(0.01, legend_bbox.y0 - 0.01, textwrap.fill(footnote, int(180 * width_scale)),
                 ha='left', va='top', fontsize=TICKS_FONT_SIZE - 2)
    fname = os.path.join(path_out, f'combat_datasets_grid{"_subjects" if per_subject else ""}_{_slug(label)}.png')
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Figure saved: {fname}')


def _slug(label):
    return re.sub(r'[^A-Za-z0-9]+', '_', LABEL_DISPLAY_NAMES.get(label, label)).strip('_')


def main():
    parser = get_parser()
    args = parser.parse_args()
    args.path_out = os.path.expanduser(args.path_out)
    for arg in ['path_results_train', 'participants_train', 'path_results_new', 'path_data_new']:
        if getattr(args, arg) is not None:
            setattr(args, arg, [os.path.expanduser(p) for p in getattr(args, arg)])
    if args.path_results_new and (not args.path_data_new or len(args.path_data_new) != len(args.path_results_new)):
        parser.error('-path-data-new (one BIDS folder per -path-results-new entry) is required with '
                     '-path-results-new (to read the scanner models)')
    site_patterns = parse_site_patterns(args.site_names)
    site_first_level = {k: int(v) for k, v in (e.split('=', 1) for e in args.site_first_level)}
    site_colors = dict(e.split('=', 1) for e in args.site_colors)
    site_linestyles = {k: LINESTYLES[v] for k, v in (e.split('=', 1) for e in args.site_linestyles)}
    vert_range = tuple(args.vert_range)
    os.makedirs(args.path_out, exist_ok=True)

    # --- Training dataset ---
    print('Loading training dataset...')
    df_train = pd.concat([load_dti_csvs(p, 'train') for p in args.path_results_train], ignore_index=True)
    if args.labels:
        df_train = df_train[df_train['Label'].isin(args.labels)]
    # In PAM50 space, the slice-to-level mapping is the same for all subjects
    slice_to_level = (df_train.dropna(subset=['VertLevel']).drop_duplicates(SLICE)[[SLICE, 'VertLevel']]
                      .astype({'VertLevel': int}))
    slice_to_level = slice_to_level[slice_to_level['VertLevel'].between(*vert_range)].sort_values(SLICE)
    # Get unique PAM50 slices in the -vert-range; training subjects must cover all of them to be used for fitting ComBat
    slices = set(slice_to_level[SLICE])
    df_train = df_train[df_train[SLICE].isin(slices)].dropna(subset=['value'])

    participants = pd.concat([pd.read_csv(p, sep='\t') for p in args.participants_train],
                             ignore_index=True).set_index('participant_id')
    # ComBat batch: scanner model by default (-batch-col manufacturers_model_name)
    # batch (scanner model) and covariates (sex, age by default)
    covars_train = participants[[args.batch_col] + args.covariates].rename(columns={args.batch_col: 'batch'})
    covars_train['batch'] = covars_train['batch'].map(normalize_model_name)
    missing = sorted(set(df_train['participant_id']) - set(covars_train.dropna().index))
    if missing:
        print(f'  Excluding {len(missing)} subject(s) without {"/".join([args.batch_col] + args.covariates)}: '
              f'{", ".join(missing)}')
        df_train = df_train[~df_train['participant_id'].isin(missing)]
    wide_train = to_wide(df_train)
    # Same features (columns) for all subjects, in the same order
    wide_train = wide_train.reindex(columns=pd.MultiIndex.from_product(
        [DTI_METRICS, sorted(df_train['Label'].unique()), sorted(slices)], names=['metric', 'Label', SLICE]))
    # rows: subjects
    # columns: features (metric × label × slice), e.g., FA × white matter × slice 801, FA × white matter × slice 802, …
    wide_train = wide_train.loc[:, wide_train.notna().any()]

    # --- New dataset ---
    wide_new, batch_new, controls_new = None, pd.Series(dtype=str), []
    if args.path_results_new:
        print('Loading new dataset(s)...')
        frames, models = [], {}
        for path_results, path_data in zip(args.path_results_new, args.path_data_new):
            d = load_dti_csvs(path_results, 'new')
            frames.append(d)
            # ComBat batch of the new subjects: scanner model from their DWI JSON sidecars
            subs = sorted(d['participant_id'].unique())
            # Scanner model from -new-models, else from the BIDS JSON sidecars
            new_models = {sub: normalize_model_name(model) for sub in subs
                          for pattern, model in reversed(parse_site_patterns(args.new_models))
                          if fnmatch.fnmatch(sub, pattern)}
            models.update(new_models)
            if path_data.lower() != 'none':
                models.update(get_models_from_bids(path_data, [s for s in subs if s not in new_models],
                                                   BATCH_COL_TO_JSON_KEY.get(args.batch_col, 'ManufacturersModelName')))
        df_new = pd.concat(frames, ignore_index=True)
        df_new = df_new[df_new[SLICE].isin(slices)].dropna(subset=['value'])
        wide_new = to_wide(df_new).reindex(columns=wide_train.columns)
        batch_new = pd.Series(models)
        controls_new = [s for s in wide_new.index if any(fnmatch.fnmatch(s, p) for p in args.new_controls)]
        train_batches = set(covars_train.loc[wide_train.index, 'batch'])
        for model, subs in batch_new.groupby(batch_new):
            n_ctrl = len(set(subs.index) & set(controls_new))
            if model in train_batches:
                status = 'in training dataset'
            elif n_ctrl:
                status = f'NOT in training dataset; {n_ctrl} healthy control(s) → reference-batch ComBat'
            else:
                status = 'NOT in training dataset → not harmonized'
            print(f'  {model}: {len(subs)} subject(s) ({status})')

    # --- ComBat on all slices × metrics × labels ---
    print(f'\nHarmonizing {vert_num_to_label(vert_range[0])}–{vert_num_to_label(vert_range[1])} '
          f'({len(slices)} slices)...')
    wide_raw, wide_harm, batch_all, status = harmonize(wide_train, covars_train, wide_new, batch_new,
                                                       controls_new, args.covariates)

    df = to_long(wide_raw, wide_harm)
    df['VertLevel'] = df[SLICE].map(slice_to_level.set_index(SLICE)['VertLevel'])
    df['dataset'] = np.where(df['participant_id'].isin(wide_train.index), 'train', 'new')
    df['batch'] = df['participant_id'].map(batch_all)
    df['status'] = df['participant_id'].map(status)
    df['site'] = np.where(df['dataset'] == 'train', df['batch'],
                          df['participant_id'].map(lambda p: get_site(p, site_patterns)))
    df = df[['participant_id', 'dataset', 'site', 'batch', 'status', SLICE, 'VertLevel', 'Label', 'metric',
             'value_raw', 'value_harmonized']]
    print('\nSubjects per dataset, site and status:')
    print(df.groupby(['dataset', 'site', 'status'])['participant_id'].nunique().to_string())
    fname = os.path.join(args.path_out, 'combat_perslice.csv')
    df.to_csv(fname, index=False)
    print(f'\nSaved: {fname}')

    # --- Between-scanner variability of the training dataset ---
    train = df[(df['dataset'] == 'train') & (df['status'] != 'not harmonized')]
    cov = between_scanner_cov(train)
    fname = os.path.join(args.path_out, 'combat_between_scanner_cov.csv')
    cov.to_csv(fname, index=False)
    print(f'Saved: {fname}')
    for label in args.labels_to_plot:
        c = cov[cov['Label'] == label].groupby('metric')[['cov_raw', 'cov_harmonized']].mean().reindex(DTI_METRICS)
        print(f'\nBetween-scanner CoV of {label} (training dataset, mean over slices):')
        print((c * 100).round(1).astype(str).add(' %').to_string())
    # Between-scanner-model CoV (whatever -batch-col), to compare harmonization strategies (shown in the figures)
    scanner_model = participants['manufacturers_model_name'].map(normalize_model_name)
    cov_model = between_scanner_cov(train.assign(batch=train['participant_id'].map(scanner_model)))
    cov_model = {label: cov_model[cov_model['Label'] == label].groupby('metric')[['cov_raw', 'cov_harmonized']].mean()
                 for label in args.labels_to_plot}
    if args.batch_col != 'manufacturers_model_name':
        for label, c in cov_model.items():
            print(f'\nBetween-scanner-model CoV of {label} (training dataset, mean over slices):')
            print((c.reindex(DTI_METRICS) * 100).round(1).astype(str).add(' %').to_string())

    # --- Figures ---
    # New-dataset sites in -site-names order, then any other (unnamed) sites
    named = [name for _, name in site_patterns]
    sites = list(dict.fromkeys(n for n in named if n in set(df.loc[df['dataset'] == 'new', 'site'])))
    sites += sorted(set(df.loc[df['dataset'] == 'new', 'site']) - set(sites))
    site_style = {site: (site_colors.get(site, SITE_COLORS[k % len(SITE_COLORS)]), site_linestyles.get(site, '-'))
                  for k, site in enumerate(sites)}
    for label in args.labels_to_plot:
        plot_scanners(train, label, slice_to_level, args.path_out)
        if wide_new is not None:
            for per_subject in [False, True]:
                plot_datasets_grid(df, label, slice_to_level, args.dataset_labels, sites, site_style,
                                   site_first_level, args.path_out, per_subject, cov_model[label],
                                   args.footnote)


if __name__ == '__main__':
    main()
