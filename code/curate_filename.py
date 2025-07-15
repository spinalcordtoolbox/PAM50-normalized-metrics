import os
import argparse
import pandas as pd
import re

def process_filename(original_path):
    """
    Extracts and modifies the filename path based on given rules.
    Specifically, from the absolute input path, keep only `sub-XXX/anat/sub-XXX_T2w_label-SC_seg.nii.gz`.
    """
    processed_path = os.path.join(*original_path.split("/")[-3::]) 
    match = re.search(r"(sub-[^/]+/anat/sub-[^/]+)_T2w_seg-manual\.nii\.gz$", processed_path)
    if match:
        return f"{match.group(1)}_T2w_label-SC_seg.nii.gz"
    return processed_path  # Return unchanged if it doesn't match the expected format

def process_csv(input_path, output_folder):
    """Processes a single CSV file, modifying relevant filename columns and saving in the output folder."""
    df = pd.read_csv(input_path)

    # Identify columns that contain file paths
    for col in df.columns:
        if df[col].dtype == object and df[col].str.contains("nii.gz").any():
            df[col] = df[col].apply(process_filename)

    # Save the modified CSV in the output folder
    output_path = os.path.join(output_folder, os.path.basename(input_path))
    df.to_csv(output_path, index=False)
    print(f"Processed and saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Modify filenames in CSV files.")
    parser.add_argument("-folder", type=str, help="Folder containing CSV files", required=True)
    parser.add_argument("-ofolder", type=str, help="Folder to save modified CSV files", required=True)
    parser.add_argument("-participants", type=str, help="Participants.tsv to only inlcude participants with", required=True)
    args = parser.parse_args()

    folder = args.folder
    participants_file = args.participants
    if not os.path.isdir(folder):
        print(f"Error: Folder '{folder}' does not exist.")
        return
    if not os.path.isfile(participants_file):
        print(f"Error: Participants file '{participants_file}' does not exist.")
        return
    participants_df = pd.read_csv(participants_file, sep="\t")
    csv_files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    participant_ids = set(participants_df['participant_id'].astype(str))
    csv_files = [
        f for f in csv_files
        if any(pid in f for pid in participant_ids)
    ]
    if not csv_files:
        print("No CSV files found in the folder.")
        return

    os.makedirs(args.ofolder, exist_ok=True)
    for csv_file in csv_files:
        process_csv(os.path.join(folder, csv_file), args.ofolder)

if __name__ == "__main__":
    main()
