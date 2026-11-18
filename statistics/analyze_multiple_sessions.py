"""
Analyze datasets containing repeated sessions/visits for issue #35.

For each multi-session dataset, this script:
1. compares repeated measurements using per-slice lineplots;
2. computes within-subject CoV across repeated measurements;
3. saves per-level CoV summaries and detailed participant/slice values.

Dataset-specific handling is used when the meaning of "session" differs
between datasets.
"""

import os
from pathlib import Path

import pandas as pd

import generate_figures as gf


DATA_ROOT = Path("../spinal_cord")
OUTPUT_ROOT = Path("multiple_sessions")


def prepare_plot_df(df):
    """
    Prepare a copy of the dataframe for create_lineplot().
    Solidity is converted from 0-1 to percentage, as in generate_figures.py.
    """
    plot_df = df.copy()

    if "MEAN(solidity)" in plot_df.columns:
        plot_df["MEAN(solidity)"] = plot_df["MEAN(solidity)"] * 100

    return plot_df


def save_cov_results(df, repeat_col, output_dir, repeat_label):
    """
    Compute, plot, and save within-subject CoV results.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary, detailed = gf.compute_within_subject_cv(
        df,
        repeat_col=repeat_col,
        min_repeats=2
    )

    if summary.empty:
        print("No repeated measurements available for CoV analysis.")
        return

    gf.create_within_subject_cv_plot(
        summary,
        str(output_dir),
        repeat_label=repeat_label
    )

    summary_file = output_dir / f"within_subject_cov_per_{repeat_label}.csv"
    detailed_file = output_dir / f"within_subject_cov_per_{repeat_label}_detailed.csv"

    summary.to_csv(summary_file, index=False)
    detailed.to_csv(detailed_file, index=False)

    print(f"Created: {summary_file}")
    print(f"Created: {detailed_file}")


def analyze_standard_dataset(dataset_name):
    """
    Analyze datasets where session_id directly represents the repeated
    measurement to compare.
    """
    print("\n" + "=" * 80)
    print(f"Dataset: {dataset_name}")
    print("=" * 80)

    dataset_path = DATA_ROOT / dataset_name
    output_dir = OUTPUT_ROOT / dataset_name

    df, _, _ = gf.read_csv_files(
        str(dataset_path),
        participant_file="",
        dataset_name=dataset_name
    )

    if "session_id" not in df.columns:
        print("No session_id column found. Skipping.")
        return

    # Issue #35 focuses on C1-T1.
    df = df[df["VertLevel"].isin(range(1, 9))].copy()

    repeated_subjects = (
        df[["participant_id", "session_id"]]
        .drop_duplicates()
        .groupby("participant_id")["session_id"]
        .nunique()
    )

    n_repeated = (repeated_subjects >= 2).sum()

    print(f"Subjects with >=2 sessions: {n_repeated}")

    output_dir.mkdir(parents=True, exist_ok=True)

    plot_df = prepare_plot_df(df)

    gf.create_lineplot(
        plot_df,
        "session_id",
        str(output_dir),
        show_n=False
    )

    save_cov_results(
        df,
        repeat_col="session_id",
        output_dir=output_dir,
        repeat_label="session"
    )


def analyze_blsa():
    """
    BLSA-specific analysis.

    BLSA session IDs contain both visit and scanner information
    (e.g. ses-250scanner10).

    Visits are ordered chronologically within each participant.
    Only the first five visits are compared.

    When the same participant/visit was acquired on multiple scanners,
    scanner10 is preferred because it is the dominant scanner in the dataset.
    """
    dataset_name = "BLSA"

    print("\n" + "=" * 80)
    print("Dataset: BLSA")
    print("=" * 80)

    dataset_path = DATA_ROOT / dataset_name
    output_dir = OUTPUT_ROOT / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    participants = pd.read_csv(
        dataset_path / "participants.tsv",
        sep="\t"
    )

    participants["visit"] = (
        participants["session_id"]
        .str.extract(r"ses-(\d+)scanner")[0]
        .astype(int)
    )

    participants["scanner"] = (
        participants["session_id"]
        .str.extract(r"scanner(\d+)")[0]
    )

    unique_visits = (
        participants[["participant_id", "visit"]]
        .drop_duplicates()
        .sort_values(["participant_id", "visit"])
    )

    unique_visits["visit_index"] = (
        unique_visits
        .groupby("participant_id")
        .cumcount() + 1
    )

    participants = participants.merge(
        unique_visits,
        on=["participant_id", "visit"],
        how="left"
    )

    # Keep the first five longitudinal visits.
    participants = participants[
        participants["visit_index"] <= 5
    ].copy()

    # Prefer scanner10 if multiple scanners exist for the same visit.
    participants["scanner10_priority"] = (
        participants["scanner"] == "10"
    ).astype(int)

    selected = (
        participants
        .sort_values(
            [
                "participant_id",
                "visit_index",
                "scanner10_priority"
            ],
            ascending=[True, True, False]
        )
        .drop_duplicates(
            ["participant_id", "visit_index"],
            keep="first"
        )
    )

    selected = selected[
        [
            "participant_id",
            "session_id",
            "visit_index"
        ]
    ]

    print(f"Selected subject-visits: {len(selected)}")

    print("\nSubjects per visit:")
    print(
        selected.groupby("visit_index")["participant_id"]
        .nunique()
    )

    # BLSA is too large to load all CSVs with read_csv_files().
    # Load only the selected subject/visit files and keep only the
    # columns and vertebral levels needed for this analysis.
    selected_lookup = {
        (row.participant_id, row.session_id): row.visit_index
        for row in selected.itertuples(index=False)
    }

    metric_columns = [
        "MEAN(area)",
        "MEAN(diameter_AP)",
        "MEAN(diameter_RL)",
        "MEAN(eccentricity)",
        "MEAN(solidity)",
    ]

    columns_to_keep = [
        "Slice (I->S)",
        "VertLevel",
    ] + metric_columns

    dfs = []
    csv_files = [
        filename for filename in os.listdir(dataset_path)
        if "PAM50.csv" in filename
    ]

    n_selected = 0

    for filename in csv_files:
        parts = filename.split("_")

        if len(parts) < 2:
            continue

        participant_id = parts[0]
        session_id = parts[1]

        key = (participant_id, session_id)

        if key not in selected_lookup:
            continue

        current = pd.read_csv(
            dataset_path / filename,
            usecols=lambda column: column in columns_to_keep
        )

        current = current[
            current["VertLevel"].isin(range(1, 9))
        ].copy()

        current.insert(0, "participant_id", participant_id)
        current.insert(1, "session_id", session_id)
        current["visit_index"] = selected_lookup[key]

        dfs.append(current)
        n_selected += 1

        if n_selected % 500 == 0:
            print(f"  {n_selected}/{len(selected_lookup)} selected files read")

    print(f"Selected BLSA files read: {n_selected}")

    df = pd.concat(
        dfs,
        axis=0,
        ignore_index=True
    )

    # Derived AP/RL ratio, normally created by read_csv_files().
    df["MEAN(compression_ratio)"] = (
        df["MEAN(diameter_AP)"] /
        df["MEAN(diameter_RL)"]
    )

    df["visit_label"] = (
        "Visit " + df["visit_index"].astype(str)
    )

    # Lineplot: C1-T1.
    plot_df = df[
        df["VertLevel"].isin(range(1, 9))
    ].copy()

    plot_df = prepare_plot_df(plot_df)

    gf.create_lineplot(
        plot_df,
        "visit_label",
        str(output_dir),
        show_n=False
    )

    # CoV: C1-C7.
    # T1 had too few paired subjects in exploratory analysis.
    cov_df = df[
        df["VertLevel"].isin(range(1, 8))
    ].copy()

    save_cov_results(
        cov_df,
        repeat_col="visit_index",
        output_dir=output_dir,
        repeat_label="visit"
    )


def analyze_masivar():
    """
    MASiVar-specific analysis.

    MASiVar contains cohorts with different repeated-measurement designs,
    therefore cohorts are analyzed separately rather than pooling all sessions.

    Cohort II:
        inter-scanner comparison, but only one multi-session participant.

    Cohort III:
        repeated/longitudinal acquisitions on the same scanner; small sample.

    Cohort IV:
        pediatric longitudinal acquisitions; main quantitative cohort.
    """
    dataset_name = "MASiVar"

    print("\n" + "=" * 80)
    print("Dataset: MASiVar")
    print("=" * 80)

    dataset_path = DATA_ROOT / dataset_name

    df, _, _ = gf.read_csv_files(
        str(dataset_path),
        participant_file="",
        dataset_name=dataset_name
    )

    cohorts = {
        "II": r"^sub-cIIs",
        "III": r"^sub-cIIIs",
        "IV": r"^sub-cIVs",
    }

    for cohort_name, pattern in cohorts.items():

        print("\n" + "-" * 60)
        print(f"MASiVar cohort {cohort_name}")
        print("-" * 60)

        cohort_df = df[
            df["participant_id"].str.match(pattern)
        ].copy()

        if cohort_df.empty:
            print("No data found.")
            continue

        cohort_df = cohort_df[
            cohort_df["VertLevel"].isin(range(1, 9))
        ].copy()

        session_counts = (
            cohort_df[
                ["participant_id", "session_id"]
            ]
            .drop_duplicates()
            .groupby("participant_id")["session_id"]
            .nunique()
        )

        n_repeated = (session_counts >= 2).sum()

        print(f"Subjects with >=2 sessions: {n_repeated}")

        output_dir = (
            OUTPUT_ROOT /
            dataset_name /
            f"cohort_{cohort_name}"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        plot_df = prepare_plot_df(cohort_df)

        gf.create_lineplot(
            plot_df,
            "session_id",
            str(output_dir),
            show_n=False
        )

        # Cohort II has only one repeated participant,
        # therefore no group-level quantitative CoV is reported.
        if cohort_name == "II" and n_repeated < 2:
            print(
                "Skipping group-level CoV: "
                "fewer than 2 multi-session subjects."
            )
            continue

        # Cohort IV has useful paired coverage mainly through C5.
        if cohort_name == "IV":
            cov_df = cohort_df[
                cohort_df["VertLevel"].isin(range(1, 6))
            ].copy()
        else:
            cov_df = cohort_df

        save_cov_results(
            cov_df,
            repeat_col="session_id",
            output_dir=output_dir,
            repeat_label="session"
        )


def main():

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    # Datasets where session_id can be compared directly.
    standard_datasets = [
        "ds004507",
        "VUIIS-MS",
        "DLBS",
        "ds004616",
    ]

    for dataset_name in standard_datasets:
        analyze_standard_dataset(dataset_name)

    # Dataset-specific designs.
    analyze_blsa()
    analyze_masivar()


if __name__ == "__main__":
    main()
