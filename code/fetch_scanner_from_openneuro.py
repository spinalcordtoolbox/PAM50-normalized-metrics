"""
Fill the `scanner` column of a dataset's participants.tsv from the BIDS JSON
sidecars hosted on OpenNeuro (GitHub mirror OpenNeuroDatasets/<dsID>).

For open-access datasets we can read the scanner from each subject's imaging
JSON sidecar (fields `Manufacturer`, `ManufacturersModelName`,
`MagneticFieldStrength`).
See https://github.com/spinalcordtoolbox/PAM50-normalized-metrics/issues/49

The OpenNeuro dataset ID(s) and the contrast are read from the dataset's
`dataset_description.json` (`link`/`links` + `contrast`).

Some OpenNeuro datasets do not store the scanner model in their JSON
sidecars (e.g. UCLA LA5c, DLBS, AOMIC only expose field strength or nothing).
For those, the scanner is taken from the dataset's publication;
see the `LITERATURE_SCANNER` table below.

Usage:
    # one dataset
    python code/fetch_scanner_from_openneuro.py --dataset TempleSocial

    # all datasets that have a `scanner` column and an OpenNeuro link
    python code/fetch_scanner_from_openneuro.py --dataset all

    # preview without writing
    python code/fetch_scanner_from_openneuro.py --dataset DLBS --dry-run

Requirements:
    pip install pandas requests
"""

import re
import sys
import json
import argparse
import requests
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

GITHUB_API = "https://api.github.com/repos/OpenNeuroDatasets/{ds}/git/trees/{branch}?recursive=1"
RAW_URL = "https://raw.githubusercontent.com/OpenNeuroDatasets/{ds}/{branch}/{path}"
BRANCHES = ("main", "master")  # OpenNeuro mirrors use one of these as default

# AOMIC is three OpenNeuro datasets stored under one folder with prefixed IDs,
# e.g. local `sub-ID1000x0002` -> ds003097 `sub-0002`. See create_AOMIC_participants.py.
AOMIC_PREFIX_TO_DS = {"ID1000": "ds003097", "PIOP1": "ds002785", "PIOP2": "ds002790"}

# Scanner model taken from dataset's publication, used when the BIDS JSON sidecars
# do not contain Manufacturer/ManufacturersModelName.
# When an entry is present the sidecars are NOT fetched (the paper is
# authoritative and dataset-wide). AOMIC is keyed per OpenNeuro sub-dataset
# because the same physical Philips 3T scanner was upgraded between the studies.
LITERATURE_SCANNER = {
    # Poldrack et al. 2016, Sci Data 3:160110 - two 3T Siemens Trio scanners.
    # https://doi.org/10.1038/sdata.2016.110
    "UCLA_LA5c": "Siemens Trio 3T",
    # Rieck et al. 2025, Sci Data - single 3T Philips Achieva (8-ch head coil).
    # https://doi.org/10.1038/s41597-025-04847-7
    "DLBS": "Philips Achieva 3T",
    # BLSA neuroimaging - all scans on a 3T Philips Achieva (the scannerNN code
    # in session_id is the physical NIA unit, same model). Not on OpenNeuro
    # (blsa.nih.gov), so no sidecars to read.
    # https://doi.org/10.1038/s41598-024-59965-w
    "BLSA": "Philips Achieva 3T",
    # Snoek et al. 2021, Sci Data 8:85 - one Philips 3T, upgraded between studies.
    # https://doi.org/10.1038/s41597-021-00870-6
    "AOMIC": {
        "ds003097": "Philips Intera 3T",           # ID1000
        "ds002785": "Philips Achieva 3T",          # PIOP1
        "ds002790": "Philips Achieva dStream 3T",  # PIOP2
    },
}

# Datasets whose scanner is encoded in the BIDS session label (not in the JSON
# sidecars). Each entry is an ordered list of (regex-on-session_id, scanner);
# the first match wins. MASiVar (Cai et al. 2021, MRM 86:3304,
# https://doi.org/10.1002/mrm.28926) names sessions ses-s<site><scanner>x<n> and
# spans four 3T scanners across three sites; the curated subset here covers site
# 1 scanner B, site 2, and site 3.
SESSION_SCANNER = {
    "MASiVar": [
        (r"s1[AB]", "Philips Achieva 3T"),      # site 1 (scanners A and B)
        (r"s2",     "GE Discovery MR750 3T"),   # site 2
        (r"s3",     "Siemens Skyra 3T"),        # site 3
    ],
}

# Datasets whose participants.tsv uses BIDS-style split columns (like
# spine-generic) instead of a single `scanner` column. Maps each local column to
# the JSON sidecar field it is filled from. These datasets ARE read per-subject
# from the sidecars (e.g. whole-spine is multi-site: Siemens Verio at amu,
# Siemens TrioTim at unf).
MULTICOL_FIELDS = {
    "whole-spine": {
        "manufacturer": "Manufacturer",
        "manufacturers_model_name": "ManufacturersModelName",
        "receive_coil_name": "ReceiveCoilName",
        "software_versions": "SoftwareVersions",
    },
}


def get_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', required=True,
                        help="Dataset folder name under spinal_cord/ (e.g. TempleSocial), "
                             "or 'all' to process every dataset with a scanner column and "
                             "an OpenNeuro link.")
    parser.add_argument('--dry-run', action='store_true',
                        help="Print the scanner values that would be written without "
                             "modifying participants.tsv.")
    parser.add_argument('--jobs', type=int, default=8,
                        help="Number of parallel HTTP requests when fetching sidecars "
                             "(default: 8).")
    return parser


def openneuro_ids(desc: dict) -> list:
    """Extract OpenNeuro dataset IDs (dsNNNNNN) from a dataset_description.json."""
    links = desc.get("links") or ([{"link": desc.get("link", "")}])
    ids = []
    for item in links:
        m = re.search(r"(ds\d{6})", item.get("link", ""))
        if m and m.group(1) not in ids:
            ids.append(m.group(1))
    return ids


def map_subject(dataset: str, participant_id: str, ds_ids: list) -> tuple:
    """Return (openneuro_ds_id, openneuro_subject_id) for a local participant_id."""
    if dataset == "AOMIC":
        body = participant_id[len("sub-"):]          # e.g. ID1000x0002
        prefix, num = body.split("x", 1)
        return AOMIC_PREFIX_TO_DS[prefix], f"sub-{num}"
    # default: identity mapping onto the single OpenNeuro dataset
    return ds_ids[0], participant_id


def fetch_tree(ds: str) -> tuple:
    """Fetch the recursive git tree of an OpenNeuro mirror; return (branch, paths)."""
    for branch in BRANCHES:
        r = requests.get(GITHUB_API.format(ds=ds, branch=branch), timeout=60)
        if r.status_code == 200:
            data = r.json()
            if data.get("truncated"):
                print(f"  WARNING: git tree for {ds} is truncated; some sidecars may be missed.")
            return branch, [t["path"] for t in data.get("tree", []) if t["type"] == "blob"]
    r.raise_for_status()
    return None, []


def index_sidecars(paths: list, contrast: str) -> dict:
    """Map openneuro subject -> {session_or_None: json_path} for a given contrast.

    Also stores a dataset-root fallback under subject key '' (BIDS inheritance,
    e.g. AOMIC ds003097 has only a root-level T1w.json).
    """
    index = {}
    suffix = f"_{contrast}.json"
    root_json = f"{contrast}.json"
    for p in paths:
        if p == root_json:
            index.setdefault("", {})[None] = p
            continue
        if not (p.endswith(suffix) and "/anat/" in p):
            continue
        sub = p.split("/", 1)[0]                       # sub-XXXX
        m = re.search(r"_(ses-[A-Za-z0-9]+)_", "/" + p.rsplit("/", 1)[1])
        ses = m.group(1) if m else None
        index.setdefault(sub, {}).setdefault(ses, p)   # keep first path per session
    return index


def scanner_string(sidecar: dict) -> str:
    """Build a human-readable scanner ID from BIDS JSON scanner fields."""
    parts = [sidecar.get("Manufacturer"), sidecar.get("ManufacturersModelName")]
    label = " ".join(str(x) for x in parts if x)
    field = sidecar.get("MagneticFieldStrength")
    if field:
        label = f"{label} {float(field):g}T".strip()
    return label or "n/a"


def resolve_path(index: dict, sub: str, session: str | None) -> str | None:
    """Pick a JSON path for a subject, preferring an exact-session match."""
    by_ses = index.get(sub)
    if not by_ses:
        return index.get("", {}).get(None)             # dataset-root fallback
    if session and session in by_ses:
        return by_ses[session]
    return next(iter(by_ses.values()))                 # first available session


def emit_scanner(df: pd.DataFrame, scanners: list, tsv_file: Path,
                 name: str, label: str, dry_run: bool) -> None:
    """Set the `scanner` column (inserting it after session_id if absent),
    print a summary, and write participants.tsv unless dry_run."""
    if "scanner" in df.columns:
        df["scanner"] = scanners
    else:
        pos = df.columns.get_loc("session_id") + 1 if "session_id" in df.columns else len(df.columns)
        df.insert(pos, "scanner", scanners)
    counts = pd.Series(scanners).value_counts()
    print(f"[{name}]   scanner ({label}): " +
          ", ".join(f"{v}×'{k}'" for k, v in counts.items()))
    if dry_run:
        print(f"[{name}]   dry-run: participants.tsv NOT modified")
        return
    df.to_csv(tsv_file, sep="\t", index=False, na_rep="n/a")
    print(f"[{name}]   written -> {tsv_file}")


def process_dataset(dataset_dir: Path, jobs: int, dry_run: bool) -> None:
    name = dataset_dir.name
    desc_file = dataset_dir / "dataset_description.json"
    tsv_file = dataset_dir / "participants.tsv"
    if not desc_file.exists() or not tsv_file.exists():
        print(f"[{name}] SKIP: missing dataset_description.json or participants.tsv")
        return

    desc = json.loads(desc_file.read_text())
    ds_ids = openneuro_ids(desc)
    contrast = desc.get("contrast", "T1w")
    df = pd.read_csv(tsv_file, sep="\t", dtype=str)
    print(f"[{name}] {('OpenNeuro ' + ', '.join(ds_ids)) if ds_ids else 'publication'}"
          f" | contrast {contrast} | {len(df)} rows")

    # Mode 1: scanner encoded in the session label (e.g. MASiVar); may add the
    # column since these datasets often lack it.
    session_map = SESSION_SCANNER.get(name)
    if session_map:
        if "session_id" not in df.columns:
            print(f"[{name}] SKIP: no session_id column to derive scanner from")
            return
        scanners = [next((val for pat, val in session_map if re.search(pat, str(s))), "n/a")
                    for s in df["session_id"]]
        emit_scanner(df, scanners, tsv_file, name, "from session label", dry_run)
        return

    # Mode 2: scanner documented in the reference publication (dataset-wide, more
    # complete than the sidecars); skip the HTTP fetches entirely.
    lit = LITERATURE_SCANNER.get(name)
    if lit is not None:
        scanners = [(lit[map_subject(name, pid, ds_ids)[0]] if isinstance(lit, dict) else lit)
                    for pid in df["participant_id"]]
        emit_scanner(df, scanners, tsv_file, name, "from publication", dry_run)
        return

    # Mode 3: read from the OpenNeuro JSON sidecars.
    if not ds_ids:
        print(f"[{name}] SKIP: no OpenNeuro link in dataset_description.json")
        return
    multicol = MULTICOL_FIELDS.get(name)
    if multicol:
        missing = [c for c in multicol if c not in df.columns]
        if missing:
            print(f"[{name}] SKIP: participants.tsv missing columns {missing}")
            return
    elif "scanner" not in df.columns:
        print(f"[{name}] SKIP: participants.tsv has no 'scanner' column")
        return

    # Fetch + index the git tree of every OpenNeuro (sub-)dataset once.
    indexes = {}
    for ds in ds_ids:
        branch, paths = fetch_tree(ds)
        indexes[ds] = (branch, index_sidecars(paths, contrast))
        print(f"[{name}]   {ds}@{branch}: indexed "
              f"{sum(len(v) for k, v in indexes[ds][1].items() if k)} subject sidecars")

    has_session = "session_id" in df.columns
    session_col = df["session_id"] if has_session else pd.Series([None] * len(df))

    # Resolve one sidecar URL per row (cache identical URLs so we fetch each once).
    tasks = {}   # url -> None (deduplicated); row_url[i] -> url or None
    row_url = []
    for pid, ses in zip(df["participant_id"], session_col):
        ds, on_sub = map_subject(name, pid, ds_ids)
        branch, index = indexes[ds]
        path = resolve_path(index, on_sub, ses if pd.notna(ses) else None)
        url = RAW_URL.format(ds=ds, branch=branch, path=path) if path else None
        row_url.append(url)
        if url:
            tasks[url] = None

    def fetch(url: str) -> dict:
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:                          # noqa: BLE001 - report and move on
            print(f"[{name}]   WARN could not read {url}: {e}")
            return {}

    print(f"[{name}]   fetching {len(tasks)} unique sidecars ...")
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        sidecars = dict(zip(tasks, pool.map(fetch, tasks)))
    row_json = [sidecars.get(u, {}) if u else {} for u in row_url]

    if multicol:
        for col, field in multicol.items():
            df[col] = [str(sc.get(field)) if sc.get(field) is not None else "n/a"
                       for sc in row_json]
        # Site-consensus back-fill: for a subject with no sidecar (e.g. absent
        # from the public OpenNeuro release), reuse the values if every peer at
        # the same `institution` that DID resolve share one identical scanner.
        if "institution" in df.columns:
            cols = list(multicol)
            for i in df.index:
                if not all(df.at[i, c] == "n/a" for c in cols):
                    continue
                peers = df[(df["institution"] == df.at[i, "institution"]) & (df.index != i)]
                resolved = peers.loc[peers[cols].ne("n/a").any(axis=1), cols].drop_duplicates()
                if len(resolved) == 1:
                    df.loc[i, cols] = resolved.iloc[0].values
                    print(f"[{name}]   {df.at[i, 'participant_id']}: no sidecar, "
                          f"filled from '{df.at[i, 'institution']}' site consensus")
        summary = (df[list(multicol)].astype(str)
                   .agg(" | ".join, axis=1).value_counts())
    else:
        scanners = [scanner_string(sc) for sc in row_json]
        df["scanner"] = scanners
        summary = pd.Series(scanners).value_counts()
    print(f"[{name}]   scanner values: " +
          ", ".join(f"{v}×'{k}'" for k, v in summary.items()))

    if dry_run:
        print(f"[{name}]   dry-run: participants.tsv NOT modified")
        return

    df.to_csv(tsv_file, sep="\t", index=False, na_rep="n/a")
    print(f"[{name}]   written -> {tsv_file}")


def main():
    args = get_parser().parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    spinal_cord = repo_root / "spinal_cord"

    if args.dataset == "all":
        dataset_dirs = sorted(p for p in spinal_cord.iterdir() if p.is_dir())
    else:
        dataset_dirs = [spinal_cord / args.dataset]
        if not dataset_dirs[0].exists():
            sys.exit(f"ERROR: dataset folder not found: {dataset_dirs[0]}")

    for d in dataset_dirs:
        process_dataset(d, jobs=args.jobs, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
