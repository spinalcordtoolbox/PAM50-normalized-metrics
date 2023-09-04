import pandas as pd

METRICS_DTYPE = {
    'MEAN(diameter_AP)': 'float64',
    'MEAN(area)': 'float64',
    'MEAN(diameter_RL)': 'float64',
    'MEAN(eccentricity)': 'float64',
    'MEAN(solidity)': 'float64'
}

# URL to GitHub repo (raw is used to read TSV and CSV files)
URL = 'https://raw.githubusercontent.com/spinalcordtoolbox/PAM50-normalized-metrics/r20230222/'


def _read_participant_file():
    """
    Read participants.tsv file from GitHub
    """
    path_participant_file = URL + 'participants.tsv'
    df_participants = pd.read_csv(path_participant_file, sep='\t')

    return df_participants


def read_csv_files_from_github():
    """
    Read CSV files with morphometric measures for individual subjects from GitHub
    Return a single concatenated Pandas DataFrame
    """
    # Read participants.tsv
    df_participants = _read_participant_file()
    # Get list of subjects
    list_of_subjects = df_participants['participant_id']

    # Initialize pandas dataframe where data across all subjects will be stored
    df = pd.DataFrame()
    # Loop across subjects
    for subject in list_of_subjects:
        # Contruct URL to CSV file for each subject
        path_csv_file = URL + subject + '_T2w_PAM50.csv'
        # Read CSV file from URL as Pandas DataFrame
        df_subject = pd.read_csv(path_csv_file, dtype=METRICS_DTYPE)

        # Compute compression ratio (CR) as MEAN(diameter_AP) / MEAN(diameter_RL)
        df_subject['MEAN(compression_ratio)'] = df_subject['MEAN(diameter_AP)'] / df_subject['MEAN(diameter_RL)']

        # Concatenate DataFrame objects
        df = pd.concat([df, df_subject], axis=0, ignore_index=True)

    # Get sub-id (e.g., sub-amu01) from Filename column and insert it as a new column called participant_id
    # Subject ID is the first characters of the filename till slash
    df.insert(0, 'participant_id', df['Filename'].str.split('/').str[0])

    # Insert columns sex, age and manufacturer from df_participants into df
    df = df.merge(df_participants[["age", "sex", "manufacturer", "participant_id"]], on='participant_id')

    df = df.dropna(axis=1, how='all')
    df = df.dropna(axis=0, how='any').reset_index(drop=True)
    # Keep only VertLevel from C1 to Th1
    df = df[df['VertLevel'] <= 8]
    # Recode age into age bins by 10 years
    df['age'] = pd.cut(df['age'], bins=[10, 20, 30, 40, 50, 60], labels=['10-20', '20-30', '30-40', '40-50', '50-60'])

    # Multiply solidity by 100 to get percentage (sct_process_segmentation computes solidity in the interval 0-1)
    df['MEAN(solidity)'] = df['MEAN(solidity)'] * 100

    return df
