#
# Functions to plot morphometric metrics computed from normative database (spine-generic dataset in PAM50 space)
# perslice and vertebral levels
#
# Example usage:
#       python generate_figures.py
#       -path-HC $SCT_DIR/data/PAM50_normalized_metrics
#       -participant-file $SCT_DIR/data/PAM50_normalized_metrics/participants.tsv
#
# Author: Sandrine Bédard, Jan Valosek
#

import os
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

METRICS = ['MEAN(diameter_AP)', 'MEAN(area)', 'MEAN(diameter_RL)', 'MEAN(eccentricity)', 'MEAN(solidity)',
           'MEAN(compression_ratio)']

METRICS_DTYPE = {
    'MEAN(diameter_AP)': 'float64',
    'MEAN(area)': 'float64',
    'MEAN(diameter_RL)': 'float64',
    'MEAN(eccentricity)': 'float64',
    'MEAN(solidity)': 'float64'
}

METRIC_TO_TITLE = {
    'MEAN(diameter_AP)': 'AP Diameter',
    'MEAN(area)': 'Cross-Sectional Area',
    'MEAN(diameter_RL)': 'RL Diameter',
    'MEAN(eccentricity)': 'Eccentricity',
    'MEAN(solidity)': 'Solidity',
    'MEAN(compression_ratio)': 'Compression Ratio',
}

METRIC_TO_AXIS = {
    'MEAN(diameter_AP)': 'AP Diameter [mm]',
    'MEAN(area)': 'Cross-Sectional Area [mm²]',
    'MEAN(diameter_RL)': 'RL Diameter [mm]',
    'MEAN(eccentricity)': 'Eccentricity [a.u.]',
    'MEAN(solidity)': 'Solidity [%]',
    'MEAN(compression_ratio)': 'Compression Ratio [a.u.]',
}

DEMOGRAPHIC_TO_AXIS = {
    'age': 'Age [years]',
    'BMI': 'BMI [kg/m²]',
    'height': 'Height [cm]',
    'weight': 'Weight [kg]',
}

# ylim max offset (used for showing text)
METRICS_TO_YLIM = {
    'MEAN(diameter_AP)': 0.4,
    'MEAN(area)': 6,
    'MEAN(diameter_RL)': 0.7,
    'MEAN(eccentricity)': 0.03,
    'MEAN(solidity)': 1,
    'MEAN(compression_ratio)': 0.03,
}

LABELS_FONT_SIZE = 14
TICKS_FONT_SIZE = 12

# # To be same as spine-generic figures (https://github.com/spine-generic/spine-generic/blob/master/spinegeneric/cli/generate_figure.py#L114)
# When the colors are overlapping, they do not look good. So we default colors.
# PALLETE = {
#     "GE": "black",
#     "Philips": "dodgerblue",
#     "Siemens": "limegreen",
# }


def get_parser():
    parser = argparse.ArgumentParser(
        description="Plot morphometric metrics computed from normative database (spine-generic dataset in PAM50 "
                    "space) perslice and vertebral levels ")
    parser.add_argument('-path-HC', required=True, type=str,
                        help="Path to data of normative dataset computed perslice.")
    parser.add_argument('-participant-file', required=False, type=str,
                        help="Path to participants.tsv file.")
    parser.add_argument('-path-out', required=False, type=str, default='figures',
                        help="Output directory name.")

    return parser


def csv2dataFrame(filename):
    """
    Loads a .csv file and builds a pandas dataFrame of the data
    Args:
        filename (str): filename of the .csv file
    Returns:
        data (pd.dataFrame): pandas dataframe of the .csv file's data
    """
    data = pd.read_csv(filename)
    return data


def get_vert_indices(df):
    """
    Get indices of slices corresponding to mid-vertebrae
    Args:
        df (pd.dataFrame): dataframe with CSA values
    Returns:
        vert (pd.Series): vertebrae levels across slices
        ind_vert (np.array): indices of slices corresponding to the beginning of each level
        ind_vert_mid (np.array): indices of slices corresponding to mid-levels
    """
    # Get vert levels for one certain subject
    vert = df[df['participant_id'] == 'sub-amu01']['VertLevel']
    # Get indexes of where array changes value
    ind_vert = vert.diff()[vert.diff() != 0].index.values
    ind_vert_mid = []
    for i in range(len(ind_vert)):
        ind_vert_mid.append(int(ind_vert[i:i + 2].mean()))
    ind_vert_mid.insert(0, ind_vert[0] - 20)
    ind_vert_mid = ind_vert_mid

    return vert, ind_vert, ind_vert_mid


def create_lineplot(df, hue, path_out, show_cv=False):
    """
    Create lineplot for individual metrics per vertebral levels.
    Note: we are ploting slices not levels to avoid averaging across levels.
    Args:
        df (pd.dataFrame): dataframe with metric values
        hue (str): column name of the dataframe to use for grouping; if None, no grouping is applied
        path_out (str): path to output directory
        show_cv (bool): if True, show coefficient of variation
    """
    # Loop across metrics
    for metric in METRICS:
        fig, ax = plt.subplots()
        # Note: we are ploting slices not levels to avoid averaging across levels
        sns.lineplot(ax=ax, x="Slice (I->S)", y=metric, data=df, errorbar='sd', hue=hue)
        # Move y-axis to the right
        plt.tick_params(axis='y', which='both', labelleft=False, labelright=True)
        plt.grid(color='lightgrey', zorder=0)
        plt.title('Spinal Cord ' + METRIC_TO_TITLE[metric], fontsize=LABELS_FONT_SIZE)
        # Adjust ymlim for solidity (it has low variance)
        if metric == 'MEAN(solidity)':
            ax.set_ylim(90, 100)
        ymin, ymax = ax.get_ylim()
        ax.set_ylabel(METRIC_TO_AXIS[metric], fontsize=LABELS_FONT_SIZE)
        ax.set_xlabel('Vertebral Level (S->I)', fontsize=LABELS_FONT_SIZE)
        # Remove xticks
        ax.set_xticks([])

        # Get indices of slices corresponding to mid-vertebrae
        vert, ind_vert, ind_vert_mid = get_vert_indices(df)
        # Insert a vertical line for each vertebral level
        for idx, x in enumerate(ind_vert[1:]):
            plt.axvline(df.loc[x, 'Slice (I->S)'], color='black', linestyle='--', alpha=0.5)

        # Insert a text label for each vertebral level
        for idx, x in enumerate(ind_vert, 1):
            if show_cv:
                cv = compute_cv(df[(df['VertLevel'] == vert[x])], metric)
            if vert[x] > 7:
                level = 'T' + str(vert[x] - 7)
                ax.text(df.loc[ind_vert_mid[idx], 'Slice (I->S)'], ymin, level, horizontalalignment='center',
                        verticalalignment='bottom', color='black')
                # Show CV
                if show_cv:
                    ax.text(df.loc[ind_vert_mid[idx], 'Slice (I->S)'], ymax-METRICS_TO_YLIM[metric],
                            str(round(cv, 1)) + '%', horizontalalignment='center',
                            verticalalignment='bottom', color='black')
            # Deal with C1 label position
            elif vert[x] == 1:
                level = 'C' + str(vert[x])
                ax.text(df.loc[ind_vert_mid[idx], 'Slice (I->S)']+15, ymin, level, horizontalalignment='center',
                        verticalalignment='bottom', color='black')
                # Show CV
                if show_cv:
                    ax.text(df.loc[ind_vert_mid[idx], 'Slice (I->S)']+15, ymax-METRICS_TO_YLIM[metric],
                            str(round(cv, 1)) + '%', horizontalalignment='center',
                            verticalalignment='bottom', color='black')
            else:
                level = 'C' + str(vert[x])
                ax.text(df.loc[ind_vert_mid[idx], 'Slice (I->S)'], ymin, level, horizontalalignment='center',
                        verticalalignment='bottom', color='black')
                # Show CV
                if show_cv:
                    ax.text(df.loc[ind_vert_mid[idx], 'Slice (I->S)'], ymax-METRICS_TO_YLIM[metric],
                            str(round(cv, 1)) + '%', horizontalalignment='center',
                            verticalalignment='bottom', color='black')
            if show_cv:
                print(f'{metric}, {level}, COV: {cv}')

        # Invert x-axis
        ax.invert_xaxis()

        # Save figure
        if hue:
            filename = metric + '_lineplot_per' + hue + '.png'
        else:
            filename = metric + '_lineplot.png'
        path_filename = os.path.join(path_out, filename)
        plt.savefig(path_filename)
        print('Figure saved: ' + path_filename)


def create_regplot(df, path_out):
    """
    Plot data and a linear regression model fit. Slices in X and Coefficient of Variation (CoV) in Y.
    Args:
        df (pd.dataFrame): dataframe with metric values
        path_out (str): path to output directory
    """

    # Loop across metrics
    for metric in METRICS:
        slices_list = []
        cv_list = []
        # Loop across slices
        for slice in df['Slice (I->S)'].unique():
            # Get metric value for each slice
            df_slice = df[df['Slice (I->S)'] == slice]
            cv_list.append(compute_cv(df_slice, metric))
            slices_list.append(slice)

        fig, ax = plt.subplots()
        sns.regplot(ax=ax, x=slices_list, y=cv_list)
        # Move y-axis to the right
        #plt.tick_params(axis='y', which='both', labelleft=False, labelright=True)
        # Add title
        plt.title('Spinal Cord ' + METRIC_TO_TITLE[metric], fontsize=LABELS_FONT_SIZE)
        # Add labels
        ax.set_xlabel('Slice (I->S)', fontsize=LABELS_FONT_SIZE)
        ax.set_ylabel('Coefficient of Variation (%)', fontsize=LABELS_FONT_SIZE)
        # Add horizontal grid
        ax.grid(color='lightgrey', axis='y')

        # Get indices of slices corresponding to mid-vertebrae
        vert, ind_vert, ind_vert_mid = get_vert_indices(df)
        # Insert a vertical line for each vertebral level
        for idx, x in enumerate(ind_vert[1:]):
            plt.axvline(df.loc[x, 'Slice (I->S)'], color='black', linestyle='--', alpha=0.5)

        # Set the same y-axis limits across metrics
        ax.set_ylim([0, 16])

        # Place text box with COV values
        # Note: we invert xaxis, thus xmax is used for the left limit
        plt.text(.02, .93, 'mean COV: {}%'.format(round(np.mean(cv_list), 1)),
                 horizontalalignment='left', verticalalignment='center', transform=ax.transAxes,
                 bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
        # Move the text box to the front
        ax.set_zorder(1)

        ymin, ymax = ax.get_ylim()
        # Insert a text label for each vertebral level
        for idx, x in enumerate(ind_vert, 1):
            if vert[x] > 7:
                level = 'T' + str(vert[x] - 7)
                ax.text(df.loc[ind_vert_mid[idx], 'Slice (I->S)'], ymin, level, horizontalalignment='center',
                        verticalalignment='bottom', color='black')
            # Deal with C1 label position
            elif vert[x] == 1:
                level = 'C' + str(vert[x])
                ax.text(df.loc[ind_vert_mid[idx], 'Slice (I->S)']+15, ymin, level, horizontalalignment='center',
                        verticalalignment='bottom', color='black')
            else:
                level = 'C' + str(vert[x])
                ax.text(df.loc[ind_vert_mid[idx], 'Slice (I->S)'], ymin, level, horizontalalignment='center',
                        verticalalignment='bottom', color='black')

        # Invert x-axis
        ax.invert_xaxis()

        # Save figure
        filename = metric + '_cov_scatterplot.png'
        path_filename = os.path.join(path_out, filename)
        plt.savefig(path_filename)
        print('Figure saved: ' + path_filename)


def create_regplot_per_sex(df, path_out):
    """
    Plot data and a linear regression model fit. Slices in X and Coefficient of Variation (CoV) in Y. Per sex.
    Args:
        df (pd.dataFrame): dataframe with metric values
        path_out (str): path to output directory
    """

    # Loop across metrics
    for metric in METRICS:
        mean_cov = dict()
        fig, ax = plt.subplots()
        # Loop across sex
        for sex in df['sex'].unique():
            slices_list = []
            cv_list = []
            # Loop across slices
            for slice in df['Slice (I->S)'].unique():
                # Get metric value for each slice
                df_slice = df[(df['Slice (I->S)'] == slice) & (df['sex'] == sex)]
                cv_list.append(compute_cv(df_slice, metric))
                slices_list.append(slice)

            mean_cov[sex] = np.mean(cv_list)
            sns.regplot(ax=ax, x=slices_list, y=cv_list, label=sex)
        # Move y-axis to the right
        #plt.tick_params(axis='y', which='both', labelleft=False, labelright=True)
        # Add title
        plt.title('Spinal Cord ' + METRIC_TO_TITLE[metric], fontsize=LABELS_FONT_SIZE)
        # Add labels
        ax.set_xlabel('Slice (I->S)', fontsize=LABELS_FONT_SIZE)
        ax.set_ylabel('Coefficient of Variation (%)', fontsize=LABELS_FONT_SIZE)
        # Add horizontal grid
        ax.grid(color='lightgrey', axis='y')
        # Show legend including title
        plt.legend(title='sex')

        # Get indices of slices corresponding to mid-vertebrae
        vert, ind_vert, ind_vert_mid = get_vert_indices(df)
        # Insert a vertical line for each vertebral level
        for idx, x in enumerate(ind_vert[1:]):
            plt.axvline(df.loc[x, 'Slice (I->S)'], color='black', linestyle='--', alpha=0.5)

        # Set the same y-axis limits across metrics
        ax.set_ylim([0, 16])

        # Place text box with COV values
        # Note: we invert xaxis, thus xmax is used for the left limit
        plt.text(.02, .93, 'F mean COV: {}%\nM mean COV: {}%'.format(round(mean_cov['F'], 1),
                                                                              round(mean_cov['M'], 1)),
                 horizontalalignment='left', verticalalignment='center', transform=ax.transAxes,
                 bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
        # Move the text box to the front
        ax.set_zorder(1)

        ymin, ymax = ax.get_ylim()
        # Insert a text label for each vertebral level
        for idx, x in enumerate(ind_vert, 1):
            if vert[x] > 7:
                level = 'T' + str(vert[x] - 7)
                ax.text(df.loc[ind_vert_mid[idx], 'Slice (I->S)'], ymin, level, horizontalalignment='center',
                        verticalalignment='bottom', color='black')
            # Deal with C1 label position
            elif vert[x] == 1:
                level = 'C' + str(vert[x])
                ax.text(df.loc[ind_vert_mid[idx], 'Slice (I->S)']+15, ymin, level, horizontalalignment='center',
                        verticalalignment='bottom', color='black')
            else:
                level = 'C' + str(vert[x])
                ax.text(df.loc[ind_vert_mid[idx], 'Slice (I->S)'], ymin, level, horizontalalignment='center',
                        verticalalignment='bottom', color='black')
        # Invert x-axis
        ax.invert_xaxis()

        # Save figure
        filename = metric + '_cov_scatterplot_persex.png'
        path_filename = os.path.join(path_out, filename)
        plt.savefig(path_filename)
        print('Figure saved: ' + path_filename)


def compute_cv(df, metric):
    """
    Compute coefficient of variation (CV) of a given metric.
    Args:
        df (pd.dataFrame): dataframe with CSA values
        metric (str): column name of the dataframe to compute CV
    Returns:
        cv (float): coefficient of variation
    """
    cv = df[metric].std() / df[metric].mean()
    cv = cv * 100
    return cv


def format_pvalue(p_value, alpha=0.001, decimal_places=3, include_space=True, include_equal=True):
    """
    Format p-value.
    If the p-value is lower than alpha, format it to "<0.001", otherwise, round it to three decimals

    :param p_value: input p-value as a float
    :param alpha: significance level
    :param decimal_places: number of decimal places the p-value will be rounded
    :param include_space: include space or not (e.g., ' = 0.06')
    :param include_equal: include equal sign ('=') to the p-value (e.g., '=0.06') or not (e.g., '0.06')
    :return: p_value: the formatted p-value (e.g., '<0.05') as a str
    """
    if include_space:
        space = ' '
    else:
        space = ''

    # If the p-value is lower than alpha, return '<alpha' (e.g., <0.001)
    if p_value < alpha:
        p_value = space + "<" + space + str(alpha)
    # If the p-value is greater than alpha, round it number of decimals specified by decimal_places
    else:
        if include_equal:
            p_value = space + '=' + space + str(round(p_value, decimal_places))
        else:
            p_value = space + str(round(p_value, decimal_places))

    return p_value


def compute_c2_c3_stats(df):
    """
    Compute mean and std from C2 and C3 levels across sex and compare females and males.
    """

    # Compute mean and std from C2 and C3 levels
    df_c2_c3 = df[(df['VertLevel'] == 2) | (df['VertLevel'] == 3)]
    c2_c3_persex = df_c2_c3.groupby('sex')['MEAN(area)'].agg([np.mean, np.std])
    print(c2_c3_persex)

    # Compare C2-C3 CSA between females and males
    c2_c3_f = df_c2_c3[df_c2_c3['sex'] == 'F']['MEAN(area)']
    c2_c3_m = df_c2_c3[df_c2_c3['sex'] == 'M']['MEAN(area)']
    # Run normality test
    stat, pval = stats.normaltest(c2_c3_f)
    print(f'Normality test C2-C3 females: p-value{format_pvalue(pval)}')
    stat, pval = stats.normaltest(c2_c3_m)
    print(f'Normality test C2-C3 males: p-value{format_pvalue(pval)}')
    # Compute Mann-Whitney U test
    stat, pval = stats.mannwhitneyu(c2_c3_f, c2_c3_m)
    print(f'Mann-Whitney U test between females and males: p-value{format_pvalue(pval)}')


def compare_metrics_across_sex(df):
    """
    Compute Mann-Whitney U test between males and females across all levels for each metric.
    """
    for metric in METRICS:
        stat, pval = stats.mannwhitneyu(df[df['sex'] == 'M'][metric], df[df['sex'] == 'F'][metric])
        print(f'{metric}, all levels: Mann-Whitney U test between females and males: p-value{format_pvalue(pval)}')


def gen_chart_weight_height(df, df_participants, path_out):
    """
    Plot weight and height relationship per sex
    """

    plt.figure()
    fig, ax = plt.subplots()

    # Make df_participants['participant_id'] as index
    df_participants.set_index('participant_id', inplace=True)
    # Keep only "sex", "height", "weight" columns in df_participants
    df_participants = df_participants[["sex", "height", "weight"]]

    # Get values only from C2 and C3 levels
    df_c2_c3 = df[(df['VertLevel'] == 2) | (df['VertLevel'] == 3)]
    # Average slices to get mean value per subject
    df_c2_c3_average = df_c2_c3.groupby('participant_id')['MEAN(area)'].mean()

    # Combine both dataframes
    df_participants.merge(df_c2_c3_average.to_frame(), left_index=True, right_index=True)

    # Drop nan for weight and height
    print(f'Number of subjects before dropping nan for weight and height: {len(df_participants)}')
    df_participants.dropna(axis=0, subset=['weight', 'height'], inplace=True)
    print(f'Number of subjects after dropping nan for weight and height: {len(df_participants)}')

    sns.regplot(x='weight', y='height', data=df_participants[df_participants['sex'] == 'M'], label='Male', color='blue')
    sns.regplot(x='weight', y='height', data=df_participants[df_participants['sex'] == 'F'], label='Female', color='red')
    # add legend to top right corner of plot
    plt.legend(loc='upper right')
    # x axis label
    plt.xlabel(DEMOGRAPHIC_TO_AXIS['weight'])
    # y axis label
    plt.ylabel(DEMOGRAPHIC_TO_AXIS['height'])
    # add title
    plt.title('Weight vs Height persex', fontsize=LABELS_FONT_SIZE)

    # Compute correlation coefficient and p-value between BMI and metric
    corr_m, pval_m = stats.pearsonr(df_participants[df_participants['sex'] == 'M']['weight'], df_participants[df_participants['sex'] == 'M']['height'])
    corr_f, pval_f = stats.pearsonr(df_participants[df_participants['sex'] == 'F']['weight'], df_participants[df_participants['sex'] == 'F']['height'])

    # Add correlation coefficient and p-value to plot
    plt.text(0.03, 0.90,
             f'Male: r = {round(corr_m, 2)}, p{format_pvalue(pval_m, alpha=0.001, include_space=True)}\n'
             f'Female: r = {round(corr_f, 2)}, p{format_pvalue(pval_f, alpha=0.001, include_space=True)}',
             fontsize=10, transform=ax.transAxes,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor='lightgrey'))

    # save figure
    fname_fig = os.path.join(path_out, 'regplot_weight_height_relationship_persex.png')
    plt.savefig(fname_fig, dpi=200, bbox_inches="tight")
    plt.close()
    print(f'Created: {fname_fig}.\n')


def create_regplot_demographics_vs_metrics(df, path_out):
    """
    Plot relationship between demographics (BMI, weight, height) and MRI metrics persex
    """
    # Compute BMI
    df['BMI'] = df['weight'] / ((df['height'] / 100) ** 2)

    # Get values only from C2 and C3 levels
    df_c2_c3 = df[(df['VertLevel'] == 2) | (df['VertLevel'] == 3)]

    # Loop across metrics
    for metric in METRICS:
        for demographic in ['BMI', 'weight', 'height']:

            plt.figure()
            fig, ax = plt.subplots()

            # Average slices to get mean value per subject
            metric_series = df_c2_c3.groupby('participant_id')[metric].mean()
            bmi_series = df_c2_c3.groupby('participant_id')[demographic].mean()
            # Get sex per subject based on index
            sex = df_c2_c3.drop_duplicates(subset=['participant_id', 'sex'])[['participant_id', 'sex']]
            sex.set_index('participant_id', inplace=True)

            # Merge metric and bmi series with sex dataframe
            final_df = sex.merge(metric_series.to_frame(), left_index=True, right_index=True)
            final_df = final_df.merge(bmi_series.to_frame(), left_index=True, right_index=True)

            sns.regplot(x=demographic, y=metric, data=final_df[final_df['sex'] == 'M'], label='Male', color='blue')
            sns.regplot(x=demographic, y=metric, data=final_df[final_df['sex'] == 'F'], label='Female', color='red')
            # add legend to top right corner of plot
            plt.legend(loc='upper right')
            # x axis label
            plt.xlabel(DEMOGRAPHIC_TO_AXIS[demographic])
            # y axis label
            plt.ylabel(METRIC_TO_AXIS[metric])
            # add title
            plt.title('Spinal Cord ' + METRIC_TO_TITLE[metric], fontsize=LABELS_FONT_SIZE)
            # Compute correlation coefficient and p-value between BMI and metric
            corr_m, pval_m = stats.pearsonr(final_df[final_df['sex'] == 'M'][demographic],
                                            final_df[final_df['sex'] == 'M'][metric])
            corr_f, pval_f = stats.pearsonr(final_df[final_df['sex'] == 'F'][demographic],
                                            final_df[final_df['sex'] == 'F'][metric])

            # Add correlation coefficient and p-value to plot
            plt.text(0.03, 0.90,
                     f'Male: r = {round(corr_m, 2)}, p{format_pvalue(pval_m, alpha=0.001, include_space=True)}\n'
                     f'Female: r = {round(corr_f, 2)}, p{format_pvalue(pval_f, alpha=0.001, include_space=True)}',
                     fontsize=10, transform=ax.transAxes,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor='lightgrey'))

            # save figure
            fname_fig = os.path.join(path_out, 'regplot_' + demographic + '_' + metric + '_relationship_persex.png')
            plt.savefig(fname_fig, dpi=200, bbox_inches="tight")
            plt.close()
            print(f'Created: {fname_fig}.\n')


def compute_descriptive_stats(df_participants):
    """
    Compute descriptive statistics (mean and std age, weight, height)
    Args:
        df_participants: pandas dataframe containing participants information (participants.tsv)
    """
    # Whole group
    print('Whole cohort')
    print(round(df_participants[['age', 'weight', 'height']].agg(['mean', 'std']), 1))

    # Get number of males and females
    print('\nNumber of males and females')
    print(df_participants.groupby(['sex'])['participant_id'].count())
    # Per-sex
    print('\nPer-sex')
    print(round(df_participants.groupby(['sex'])[['age', 'weight', 'height']].agg(['mean', 'std']), 1))
    # Per-vendor
    print('\nPer-vendor')
    print(round(df_participants.groupby(['manufacturer'])[['age', 'weight', 'height']].agg(['mean', 'std']), 1))


def main():
    parser = get_parser()
    args = parser.parse_args()
    path_HC = args.path_HC
    path_out = args.path_out
    # If the output folder directory is not present, then create it.
    if not os.path.exists(path_out):
        os.makedirs(path_out)
    # Initialize pandas dataframe where data across all subjects will be stored
    df = pd.DataFrame()
    # Loop through .csv files of healthy controls
    for file in os.listdir(path_HC):
        if 'PAM50.csv' in file:
            # Read csv file as pandas dataframe for given subject
            df_subject = pd.read_csv(os.path.join(path_HC, file), dtype=METRICS_DTYPE)

            # Compute compression ratio (CR) as MEAN(diameter_AP) / MEAN(diameter_RL)
            df_subject['MEAN(compression_ratio)'] = df_subject['MEAN(diameter_AP)'] / df_subject['MEAN(diameter_RL)']

            # Concatenate DataFrame objects
            df = pd.concat([df, df_subject], axis=0, ignore_index=True)
    # Get sub-id (e.g., sub-amu01) from Filename column and insert it as a new column called participant_id
    # Subject ID is the first characters of the filename till slash
    df.insert(0, 'participant_id', df['Filename'].str.split('/').str[0])
    # Get number of unique subjects (unique strings under Filename column)
    subjects = df['Filename'].unique()
    # If a participants.tsv file is provided, insert columns sex, age and manufacturer from df_participants into df
    if args.participant_file:
        df_participants = pd.read_csv(args.participant_file, sep='\t')
        df = df.merge(df_participants[["age", "sex", "height", "weight", "manufacturer", "participant_id"]],
                      on='participant_id')
    # Print number of subjects
    print(f'Number of subjects: {str(len(subjects))}\n')

    # Compute descriptive statistics (mean and std age, weight, height)
    compute_descriptive_stats(df_participants)

    df = df.dropna(axis=1, how='all')
    df = df.dropna(axis=0, how='any').reset_index(drop=True) # do we want to compute mean with missing levels for some subjects?
    # Keep only VertLevel from C1 to Th1
    df = df[df['VertLevel'] <= 8]
    # Recode age into age bins by 10 years
    df['age'] = pd.cut(df['age'], bins=[10, 20, 30, 40, 50, 60], labels=['10-20', '20-30', '30-40', '40-50', '50-60'])

    # Compute Mann-Whitney U test between males and females for across levels for each metric.
    compare_metrics_across_sex(df)

    # Plot correlation between weight and height per sex
    gen_chart_weight_height(df, df_participants, args.path_out)

    # Plot relationship between demographics (BMI, weight, height) and MRI metrics persex
    create_regplot_demographics_vs_metrics(df, args.path_out)

    # Compute mean and std from C2 and C3 levels across sex and compare females and males
    compute_c2_c3_stats(df)

    # Multiply solidity by 100 to get percentage (sct_process_segmentation computes solidity in the interval 0-1)
    df['MEAN(solidity)'] = df['MEAN(solidity)'] * 100

    # Uncomment to save aggregated dataframe with metrics across all subjects as .csv file
    #df.to_csv(os.path.join(path_out, 'HC_metrics.csv'), index=False)

    # Create plots
    create_lineplot(df, None, args.path_out, show_cv=True)        # across all subjects
    create_lineplot(df, 'age', args.path_out)       # across age
    create_lineplot(df, 'sex', args.path_out)       # across sex
    create_lineplot(df, 'manufacturer', args.path_out)  # across manufacturer (vendors)

    # Plot scatterplot metrics vs COV
    create_regplot(df, args.path_out)

    # Plot scatterplot metrics vs COV per sex
    create_regplot_per_sex(df, path_out)


if __name__ == '__main__':
    main()
