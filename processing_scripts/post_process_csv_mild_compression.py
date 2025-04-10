#!/usr/bin/env python
# -*- coding: utf-8

# Author: Sandrine Bédard

import os
import argparse
import pandas as pd

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i-folder",
                        required=True,
                        type=str,
                        help="Folder with csv files to process")
    parser.add_argument("-o-folder",
                        type=str,
                        required=True,
                        help="Folder to write results")

    parser.add_argument("-participants",
                        type=str,
                        required=False,
                        help="Participants.tsv to get stenosis location")
    return parser
levels={
        "C2-C3": 3,
        "C3-C4": 4,
        "C4-C5": 5,
        "C5-C6": 6,
        "C6-C7": 7,
        "C7-T1": 8,
        "T1-T2": 9,
        "T2-T3": 10,
        "T3-T4": 11,
        "T4-T5": 12,
        "T5-T6": 13,
        "T6-T7": 14,
        "T7-T8": 15,
        "T8-T9": 16,
        "T9-T10": 17,
        "T10-T11": 18,
        "T11-T12": 19,
        "T12-L1": 20,
        "L1-L2": 21,
        "L2-L3": 22,
        "L3-L4": 23,
        "L4-L5": 24,
        "L5-S1": 25
    }

def remove_mild_processing(input_file, output_folder, compression_levels, nb_slices=10):
    # Read the input CSV file
    df = pd.read_csv(input_file)
    # Convert compression_level to a list of numbers using the levels dictionary
    print(f"compression levels: {compression_levels}")
    compression_levels = compression_levels.split(',')
    compression_indices = [levels[level] for level in compression_levels if level in levels]
    # Remove rows where the 'slice' column is within the specified compression levels
    #df = df[~df['slice'].isin(compression_indices)]
    print(f"compression levels: {compression_indices}")
    # Find rows where 'VertLevel' is in compression_indices or the level before
    for level in compression_indices:
        indices_to_remove_below = df[df['VertLevel'].isin([level])].index[-nb_slices//2:]
        indices_to_remove_above = df[df['VertLevel'].isin([level - 1])].index[0:nb_slices//2]

        print(f"Indices to removes for level {level} : {indices_to_remove_below}")
        print(f"Indices to removes for level {level-1} : {indices_to_remove_above}")
        # Replace specified columns with NaN for the identified indices
        columns_to_nan = [
            "MEAN(area)", "STD(area)", "MEAN(angle_AP)", "STD(angle_AP)", 
            "MEAN(angle_RL)", "STD(angle_RL)", "MEAN(diameter_AP)", "STD(diameter_AP)", 
            "MEAN(diameter_RL)", "STD(diameter_RL)", "MEAN(eccentricity)", "STD(eccentricity)", 
            "MEAN(orientation)", "STD(orientation)", "MEAN(solidity)", "STD(solidity)", 
            "SUM(length)"
        ]
        indices_to_nan = indices_to_remove_below.union(indices_to_remove_above)
        df.loc[indices_to_nan, columns_to_nan] = pd.NA
    # Write the processed DataFrame to the output CSV file
    output_file = os.path.join(output_folder, os.path.basename(input_file))
    df.to_csv(output_file, index=False)

def analyse_stenosis(df_participants):
    # Initialize a dictionary to count occurrences of each compression level
    compression_occurrences = {level: 0 for level in levels.keys()}

    # Iterate through the 'stenosis' column to count occurrences of each level
    for stenosis in df_participants['stenosis']:
        if pd.notna(stenosis):
            stenosis_levels = stenosis.split(',')
            for level in stenosis_levels:
                if level in compression_occurrences:
                    compression_occurrences[level] += 1

    # Convert the dictionary to a DataFrame for better visualization
    df_compression_occurrences = pd.DataFrame(list(compression_occurrences.items()), columns=['Compression Level', 'Occurrences'])
    df_compression_occurrences = df_compression_occurrences[df_compression_occurrences['Occurrences'] > 0]
    print(df_compression_occurrences)
    #return df_compression_occurrences


def main():

    args = get_parser().parse_args()
    # Get input argments
    input_folder = os.path.abspath(args.i_folder)
    output_folder = os.path.abspath(args.o_folder)
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    participants_file= os.path.abspath(args.participants)
    # Read participants file
    df_participants = pd.read_csv(participants_file, sep="\t")
    # Filter rows where the 'stenosis' column has a value
    df_participants = df_participants[df_participants['stenosis'].notna()]
    # Get the list of csv files in the input folder
    csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
    for csv in csv_files:
        if any(participant_id in csv for participant_id in df_participants['participant_id']):
            print(f"processing {csv}...")
            participant = csv.split('_')[0]
            remove_mild_processing(os.path.join(input_folder, csv), output_folder, df_participants.loc[df_participants['participant_id']==participant, 'stenosis'].values[0])
    analyse_stenosis(df_participants)

if __name__ == "__main__":
    main()
