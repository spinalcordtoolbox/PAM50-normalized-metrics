import os
import argparse
import pandas as pd
import re

def process_filename(original_path):
    """Extracts and modifies the filename path based on given rules."""
    match = re.search(r"(sub-[^/]+/anat/sub-[^/]+)_T2w_seg-manual\.nii\.gz$", original_path)
    if match:
        return f"{match.group(1)}_T2w_label-SC_seg.nii.gz"
    return original_path  # Return unchanged if it doesn't match the expected format

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
    args = parser.parse_args()

    folder = args.folder

    if not os.path.isdir(folder):
        print(f"Error: Folder '{folder}' does not exist.")
        return

    csv_files = [f for f in os.listdir(folder) if f.endswith(".csv")]

    if not csv_files:
        print("No CSV files found in the folder.")
        return

    for csv_file in csv_files:
        process_csv(os.path.join(folder, csv_file), args.ofolder)

if __name__ == "__main__":
    main()
