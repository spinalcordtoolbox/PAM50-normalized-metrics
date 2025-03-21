import os
import argparse
import pandas as pd

def merge_participants(file1, file2, output_file):
    """Merges two participants.tsv files based on participant_id, ensuring format consistency."""
    
    # Load the TSV files
    df1 = pd.read_csv(file1, sep='\t', dtype=str)
    df2 = pd.read_csv(file2, sep='\t', dtype=str)

    # Standardize column names in df2 to match df1 where possible
    rename_dict = {
        'height (cm)': 'height',
        'weight (kg)': 'weight'
    }
    df2.rename(columns=rename_dict, inplace=True)

    # Ensure 'species' column exists and set all values to 'homo sapiens'

    # Merge institution_id and institution into one column
    df1['institution'] = df1['institution_id'] + ' - ' + df1['institution']
    df1.drop(columns=['institution_id'], inplace=True, errors='ignore')

    # Merge the dataframes on 'participant_id' using an outer join
    merged_df = pd.merge(df1, df2, on="participant_id", how="outer", suffixes=("_file1", "_file2"))
    merged_df['species'] = 'homo sapiens'

    # Handle duplicate columns (keep the value from df1 when available)
    for col in ['sex', 'age', 'height', 'weight', 'pathology', 'institution']:
        col_file1 = f"{col}_file1"
        col_file2 = f"{col}_file2"
        if col_file1 in merged_df.columns and col_file2 in merged_df.columns:
            merged_df[col] = merged_df[col_file1].combine_first(merged_df[col_file2])
            merged_df.drop([col_file1, col_file2], axis=1, inplace=True)

    # Ensure all missing values are replaced with "n/a"
    merged_df.fillna("n/a", inplace=True)

    # Maintain the original order of columns from df1 and add extra columns at the end
    ordered_columns = list(df1.columns) + [col for col in merged_df.columns if col not in df1.columns]
    merged_df = merged_df[ordered_columns]

    # Save the merged dataframe as a TSV
    merged_df.to_csv(output_file, sep='\t', index=False)
    print(f"Merged file saved as: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Merge two participants.tsv files.")
    parser.add_argument("-file1", type=str, help="Path to the first participants.tsv file")
    parser.add_argument("-file2", type=str, help="Path to the second participants.tsv file")
    parser.add_argument("-o", type=str, help="Path to save the merged TSV file")
    args = parser.parse_args()

    merge_participants(args.file1, args.file2, args.o)

if __name__ == "__main__":
    main()
