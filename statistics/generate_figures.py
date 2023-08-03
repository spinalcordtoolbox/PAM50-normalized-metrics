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
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats as stats

METRICS = ['MEAN(area)', 'MEAN(diameter_AP)', 'MEAN(diameter_RL)', 'MEAN(compression_ratio)', 'MEAN(solidity)',
           'MEAN(eccentricity)']

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

DISCS_DICT = {
    7: 'C7-T1',
    6: 'C6-C7',
    5: 'C5-C6',
    4: 'C4-C5',
    3: 'C3-C4',
    2: 'C2-C3',
    1: 'C1-C2'
}

MID_VERT_DICT = {
    8: 'T1',
    7: 'C7',
    6: 'C6',
    5: 'C5',
    4: 'C4',
    3: 'C3',
    2: 'C2',
    1: 'C1'
}

VENDORS = ['Siemens', 'Philips', 'GE']
AGE_DECADES = ['10-20', '21-30', '31-40', '41-50', '51-60']

LABELS_FONT_SIZE = 14
TICKS_FONT_SIZE = 12

COLORS_SEX = {
    'M': 'blue',
    'F': 'red'
    }

# To be same as spine-generic figures (https://github.com/spine-generic/spine-generic/blob/master/spinegeneric/cli/generate_figure.py#L114)
PALLET = {
    'sex': {'M': 'blue', 'F': 'red'},
    'manufacturer': {'Siemens': 'limegreen', 'Philips': 'dodgerblue', 'GE': 'black'},
    'age': {'10-20': 'blue', '21-30': 'green', '31-40': 'orange', '41-50': 'red', '51-60': 'purple'},
    }


def get_parser():
    parser = argparse.ArgumentParser(
        description="Plot morphometric metrics computed from normative database (spine-generic dataset in PAM50 "
                    "space) perslice and vertebral levels ")
    parser.add_argument('-path-HC', required=True, type=str,
                        help="Path to data of normative dataset computed perslice.")
    parser.add_argument('-participant-file', required=False, type=str,
                        help="Path to participants.tsv file.")
    parser.add_argument('-path-out', required=False, type=str, default='stats',
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
        ind_vert (np.array): indices of slices corresponding to the beginning of each level (=intervertebral disc)
        ind_vert_mid (np.array): indices of slices corresponding to mid-levels
    """
    # Get vert levels for one certain subject
    vert = df[df['participant_id'] == 'sub-amu01']['VertLevel']
    # Get indexes of where array changes value
    ind_vert = vert.diff()[vert.diff() != 0].index.values
    # Get the beginning of C1
    ind_vert = np.append(ind_vert, vert.index.values[-1])
    ind_vert_mid = []
    # Get indexes of mid-vertebrae
    for i in range(len(ind_vert)-1):
        ind_vert_mid.append(int(ind_vert[i:i+2].mean()))

    return vert, ind_vert, ind_vert_mid


def create_lineplot(df, hue, path_out, show_cv=False):
    """
    Create lineplot for individual metrics per vertebral levels.
    Note: we are ploting slices not levels to avoid averaging across levels.
    Args:
        df (pd.dataFrame): dataframe with metric values
        hue (str): column name of the dataframe to use for grouping; if None, no grouping is applied
        path_out (str): path to output directory
        show_cv (bool): if True, include coefficient of variation for each vertebral level to the plot
    """

    mpl.rcParams['font.family'] = 'Arial'

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    axs = axes.ravel()

    # Loop across metrics
    for index, metric in enumerate(METRICS):
        # Note: we are ploting slices not levels to avoid averaging across levels
        if hue == 'sex' or hue == 'manufacturer' or hue == 'age':
            sns.lineplot(ax=axs[index], x="Slice (I->S)", y=metric, data=df, errorbar='sd', hue=hue,
                         palette=PALLET[hue])
            if index == 0:
                axs[index].legend(loc='upper right', fontsize=TICKS_FONT_SIZE)
            else:
                axs[index].get_legend().remove()
        else:
            sns.lineplot(ax=axs[index], x="Slice (I->S)", y=metric, data=df, errorbar='sd', hue=hue)

        # Adjust ymlim for solidity (it has low variance)
        if metric == 'MEAN(solidity)':
            axs[index].set_ylim(90, 100)
        ymin, ymax = axs[index].get_ylim()

        # Add labels
        axs[index].set_ylabel(METRIC_TO_AXIS[metric], fontsize=LABELS_FONT_SIZE)
        axs[index].set_xlabel('Vertebral Level (S->I)', fontsize=LABELS_FONT_SIZE)
        # Increase xticks and yticks font size
        axs[index].tick_params(axis='both', which='major', labelsize=TICKS_FONT_SIZE)

        # Remove spines
        axs[index].spines['right'].set_visible(False)
        axs[index].spines['left'].set_visible(False)
        axs[index].spines['top'].set_visible(False)
        axs[index].spines['bottom'].set_visible(True)

        # Get indices of slices corresponding vertebral levels
        vert, ind_vert, ind_vert_mid = get_vert_indices(df)
        # Insert a vertical line for each intervertebral disc
        for idx, x in enumerate(ind_vert[1:-1]):
            axs[index].axvline(df.loc[x, 'Slice (I->S)'], color='black', linestyle='--', alpha=0.5, zorder=0)

        # Insert a text label for each vertebral level
        for idx, x in enumerate(ind_vert_mid, 0):
            if show_cv:
                cv = compute_cv(df[(df['VertLevel'] == vert[x])], metric)
            # Deal with T1 label (C8 -> T1)
            if vert[x] > 7:
                level = 'T' + str(vert[x] - 7)
                axs[index].text(df.loc[ind_vert_mid[idx], 'Slice (I->S)'], ymin, level, horizontalalignment='center',
                                verticalalignment='bottom', color='black', fontsize=TICKS_FONT_SIZE)
                # Show CV
                if show_cv:
                    axs[index].text(df.loc[ind_vert_mid[idx], 'Slice (I->S)'], ymax-METRICS_TO_YLIM[metric],
                                    str(round(cv, 1)) + '%', horizontalalignment='center', verticalalignment='bottom',
                                    color='black')
            else:
                level = 'C' + str(vert[x])
                axs[index].text(df.loc[ind_vert_mid[idx], 'Slice (I->S)'], ymin, level, horizontalalignment='center',
                                verticalalignment='bottom', color='black', fontsize=TICKS_FONT_SIZE)
                # Show CV
                if show_cv:
                    axs[index].text(df.loc[ind_vert_mid[idx], 'Slice (I->S)'], ymax-METRICS_TO_YLIM[metric],
                                    str(round(cv, 1)) + '%', horizontalalignment='center', verticalalignment='bottom',
                                    color='black')
            if show_cv:
                print(f'{metric}, {level}, COV: {cv}')

        # Invert x-axis
        axs[index].invert_xaxis()
        # Add only horizontal grid lines
        axs[index].yaxis.grid(True)
        # Move grid to background (i.e. behind other elements)
        axs[index].set_axisbelow(True)

    # Save figure
    if hue:
        filename = 'lineplot_per' + hue + '.png'
    else:
        filename = 'lineplot.png'
    path_filename = os.path.join(path_out, filename)
    plt.savefig(path_filename, dpi=300, bbox_inches='tight')
    print('Figure saved: ' + path_filename)


def create_regplot(df, path_out, show_cv=False):
    """
    Plot data and a linear regression model fit. Slices in X and Coefficient of Variation (CoV) in Y.
    Args:
        df (pd.dataFrame): dataframe with metric values
        path_out (str): path to output directory
        show_cv (bool): if True, include coefficient of variation for each vertebral level to the plot
    """

    mpl.rcParams['font.family'] = 'Arial'

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    axs = axes.ravel()

    # Loop across metrics
    for index, metric in enumerate(METRICS):
        slices_list = []
        cv_list = []
        # Loop across slices
        for slice in df['Slice (I->S)'].unique():
            # Get metric value for each slice
            df_slice = df[df['Slice (I->S)'] == slice]
            cv_list.append(compute_cv(df_slice, metric))
            slices_list.append(slice)

        sns.regplot(ax=axs[index], x=slices_list, y=cv_list, scatter_kws={'alpha': 0.5})

        # Add labels
        axs[index].set_xlabel('Vertebral Level (S->I)', fontsize=LABELS_FONT_SIZE)
        axs[index].set_ylabel(f'{METRIC_TO_TITLE[metric]} COV (%)', fontsize=LABELS_FONT_SIZE)
        # Increase xticks and yticks font size
        axs[index].tick_params(axis='both', which='major', labelsize=TICKS_FONT_SIZE)

        # Remove spines
        axs[index].spines['right'].set_visible(False)
        axs[index].spines['left'].set_visible(False)
        axs[index].spines['top'].set_visible(False)
        axs[index].spines['bottom'].set_visible(True)

        # Get indices of slices corresponding vertebral levels
        vert, ind_vert, ind_vert_mid = get_vert_indices(df)
        # Insert a vertical line for each intervertebral disc
        for idx, x in enumerate(ind_vert[1:-1]):
            axs[index].axvline(df.loc[x, 'Slice (I->S)'], color='black', linestyle='--', alpha=0.5, zorder=0)

        # Set the same y-axis limits across metrics
        axs[index].set_ylim([0, 18])

        # Place text box with mean COV value
        axs[index].text(.5, .94, 'COV: {} ± {} %'.format(round(np.mean(cv_list), 1), round(np.std(cv_list), 1)),
                 horizontalalignment='center', verticalalignment='center', transform=axs[index].transAxes,
                 fontsize=TICKS_FONT_SIZE, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
        # Move the text box to the front
        axs[index].set_zorder(1)

        ymin, ymax = axs[index].get_ylim()
        # Insert a text label for each vertebral level
        for idx, x in enumerate(ind_vert_mid, 0):
            if show_cv:
                cv = compute_cv(df[(df['VertLevel'] == vert[x])], metric)
            # Deal with T1 label (C8 -> T1)
            if vert[x] > 7:
                level = 'T' + str(vert[x] - 7)
                axs[index].text(df.loc[ind_vert_mid[idx], 'Slice (I->S)'], ymin, level, horizontalalignment='center',
                                verticalalignment='bottom', color='black', fontsize=TICKS_FONT_SIZE)
                # Show CV
                if show_cv:
                    axs[index].text(df.loc[ind_vert_mid[idx], 'Slice (I->S)'], 14.8,
                                    str(round(cv, 1)) + '%', horizontalalignment='center', verticalalignment='bottom',
                                    color='black')
            else:
                level = 'C' + str(vert[x])
                axs[index].text(df.loc[ind_vert_mid[idx], 'Slice (I->S)'], ymin, level, horizontalalignment='center',
                                verticalalignment='bottom', color='black', fontsize=TICKS_FONT_SIZE)
                # Show CV
                if show_cv:
                    axs[index].text(df.loc[ind_vert_mid[idx], 'Slice (I->S)'], 14.8, str(round(cv, 1)) + '%',
                                    horizontalalignment='center', verticalalignment='bottom', color='black')
            if show_cv:
                print(f'{metric}, {level}, COV: {cv}')

        # Invert x-axis
        axs[index].invert_xaxis()
        # Add only horizontal grid lines
        axs[index].yaxis.grid(True)
        # Move grid to background (i.e. behind other elements)
        axs[index].set_axisbelow(True)

    # Save figure
    filename = 'cov_scatterplot.png'
    path_filename = os.path.join(path_out, filename)
    plt.savefig(path_filename, dpi=300, bbox_inches='tight')
    print('Figure saved: ' + path_filename)


def create_regplot_per_sex(df, path_out):
    """
    Plot data and a linear regression model fit. Slices in X and Coefficient of Variation (CoV) in Y. Per sex.
    Args:
        df (pd.dataFrame): dataframe with metric values
        path_out (str): path to output directory
    """

    mpl.rcParams['font.family'] = 'Arial'

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    axs = axes.ravel()

    # Loop across metrics
    for index, metric in enumerate(METRICS):
        mean_cov = dict()
        std_cov = dict()
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
            std_cov[sex] = np.std(cv_list)
            sns.regplot(ax=axs[index], x=slices_list, y=cv_list, label=sex, scatter_kws={'alpha': 0.5},
                        color=PALLET['sex'][sex])

        if index == 0:
            axs[index].legend(loc='upper right', fontsize=TICKS_FONT_SIZE)

        # Add labels
        axs[index].set_xlabel('Vertebral Level (S->I)', fontsize=LABELS_FONT_SIZE)
        axs[index].set_ylabel(f'{METRIC_TO_TITLE[metric]} COV (%)', fontsize=LABELS_FONT_SIZE)
        # Increase xticks and yticks font size
        axs[index].tick_params(axis='both', which='major', labelsize=TICKS_FONT_SIZE)

        # Remove spines
        axs[index].spines['right'].set_visible(False)
        axs[index].spines['left'].set_visible(False)
        axs[index].spines['top'].set_visible(False)
        axs[index].spines['bottom'].set_visible(True)

        # Get indices of slices corresponding vertebral levels
        vert, ind_vert, ind_vert_mid = get_vert_indices(df)
        # Insert a vertical line for each intervertebral disc
        for idx, x in enumerate(ind_vert[1:-1]):
            axs[index].axvline(df.loc[x, 'Slice (I->S)'], color='black', linestyle='--', alpha=0.5, zorder=0)

        # Set the same y-axis limits across metrics
        axs[index].set_ylim([0, 18])

        # Place text box with COV values
        # Note: we invert xaxis, thus xmax is used for the left limit
        axs[index].text(.5, .90, 'F COV: {} ± {} %\nM COV: {} ± {} %'.format(round(mean_cov['F'], 1),
                                                                             round(std_cov['F'], 1),
                                                                             round(mean_cov['M'], 1),
                                                                             round(std_cov['M'], 1)),
                        horizontalalignment='center', verticalalignment='center', transform=axs[index].transAxes,
                        fontsize=TICKS_FONT_SIZE, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
        # Move the text box to the front
        axs[index].set_zorder(1)

        ymin, ymax = axs[index].get_ylim()
        # Insert a text label for each vertebral level
        for idx, x in enumerate(ind_vert_mid, 0):
            # Deal with T1 label (C8 -> T1)
            if vert[x] > 7:
                level = 'T' + str(vert[x] - 7)
                axs[index].text(df.loc[ind_vert_mid[idx], 'Slice (I->S)'], ymin, level, horizontalalignment='center',
                                verticalalignment='bottom', color='black', fontsize=TICKS_FONT_SIZE)
            else:
                level = 'C' + str(vert[x])
                axs[index].text(df.loc[ind_vert_mid[idx], 'Slice (I->S)'], ymin, level, horizontalalignment='center',
                                verticalalignment='bottom', color='black', fontsize=TICKS_FONT_SIZE)

        # Invert x-axis
        axs[index].invert_xaxis()
        # Add only horizontal grid lines
        axs[index].yaxis.grid(True)
        # Move grid to background (i.e. behind other elements)
        axs[index].set_axisbelow(True)

    # Save figure
    filename = 'cov_scatterplot_persex.png'
    path_filename = os.path.join(path_out, filename)
    plt.savefig(path_filename, dpi=300, bbox_inches='tight')
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
    Compute Wilcoxon rank-sum tests between males and females across all levels for each metric.
    """

    print("")

    for metric in METRICS:
        print(f"\n{metric}")

        # Get mean values for each slice
        slices_M = df[df['sex'] == 'M'].groupby(['Slice (I->S)'])[metric].mean()
        slices_F = df[df['sex'] == 'F'].groupby(['Slice (I->S)'])[metric].mean()

        # Run normality test
        stat, pval = stats.shapiro(slices_M)
        print(f'Normality test M: p-value{format_pvalue(pval)}')
        stat, pval = stats.shapiro(slices_F)
        print(f'Normality test F: p-value{format_pvalue(pval)}')
        # Run Wilcoxon rank-sum test (groups are independent)
        stat, pval = stats.ranksums(x=slices_M, y=slices_F)
        print(f'{metric}, all levels: Wilcoxon rank-sum test between females and males: '
              f'p-value{format_pvalue(pval)}')


def compare_metrics_across_vendors(df):
    """
    Compute Wilcoxon rank-sum tests between MRI vendors across all levels for each metric.
    """

    print("")

    for metric in METRICS:
        print(f"\n{metric}")

        #stat, pval = stats.mannwhitneyu(df[df['sex'] == 'M'][metric], df[df['sex'] == 'F'][metric])
        #print(f'{metric}, all levels: Mann-Whitney U test between females and males: p-value{format_pvalue(pval)}')

        # Get mean values for each slice
        slices_siemens = df[df['manufacturer'] == 'Siemens'].groupby(['Slice (I->S)'])[metric].mean()
        slices_philips = df[df['manufacturer'] == 'Philips'].groupby(['Slice (I->S)'])[metric].mean()
        slices_ge = df[df['manufacturer'] == 'GE'].groupby(['Slice (I->S)'])[metric].mean()

        # Run normality test
        for i, slices in enumerate([slices_siemens, slices_philips, slices_ge]):
            stat, pval = stats.shapiro(slices)
            print(f'Normality test {VENDORS[i]}: p-value{format_pvalue(pval)}')

        # Run Wilcoxon rank-sum test (groups are independent)
        stat, pval = stats.ranksums(x=slices_siemens, y=slices_philips)
        print(f'{metric}, all levels: Wilcoxon rank-sum test between Siemens and Phlips: p-value{format_pvalue(pval)}')
        stat, pval = stats.ranksums(x=slices_siemens, y=slices_ge)
        print(f'{metric}, all levels: Wilcoxon rank-sum test between Siemens and GE: p-value{format_pvalue(pval)}')
        stat, pval = stats.ranksums(x=slices_philips, y=slices_ge)
        print(f'{metric}, all levels: Wilcoxon rank-sum test between Philips and GE: p-value{format_pvalue(pval)}')


def analyze_metrics_across_age_decades(df):
    """
    Get mean values for each age decade across all levels for each metric.
    """

    print("")

    for metric in METRICS:
        print(f"\n{metric}")

        # Get number of subjects for each age decade
        for age_decade in AGE_DECADES:
            number_of_subejcts = len(df[df['age'] == age_decade].groupby(['participant_id'])[metric])
            print(f'Number of subjects in {age_decade}: {number_of_subejcts}')

        # Get mean values for each slice
        slices_10_20 = df[df['age'] == '10-20'].groupby(['Slice (I->S)'])[metric].mean()
        slices_21_30 = df[df['age'] == '21-30'].groupby(['Slice (I->S)'])[metric].mean()
        slices_31_40 = df[df['age'] == '31-40'].groupby(['Slice (I->S)'])[metric].mean()
        slices_41_50 = df[df['age'] == '41-50'].groupby(['Slice (I->S)'])[metric].mean()
        slices_51_60 = df[df['age'] == '51-60'].groupby(['Slice (I->S)'])[metric].mean()

        # Note: groups are highly unbalanced --> no comparison


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

    sns.regplot(x='weight', y='height', data=df_participants[df_participants['sex'] == 'M'],
                label='Male', color=COLORS_SEX['M'])
    sns.regplot(x='weight', y='height', data=df_participants[df_participants['sex'] == 'F'],
                label='Female', color=COLORS_SEX['F']  )
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

            sns.regplot(x=demographic, y=metric, data=final_df[final_df['sex'] == 'M'],
                        label='Male', color=COLORS_SEX['M'])
            sns.regplot(x=demographic, y=metric, data=final_df[final_df['sex'] == 'F'],
                        label='Female', color=COLORS_SEX['F'])
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


def compute_descriptive_stats(df_participants, path_out_figures):
    """
    Compute descriptive statistics (mean and std age, weight, height)
    Also plot age distribution
    Args:
        df_participants: pandas dataframe containing participants information (participants.tsv)
    """
    # Whole group
    print('Whole cohort')
    print(round(df_participants[['age', 'weight', 'height']].agg(['mean', 'std']), 1))

    # Drop nan for weight and height
    print(f'Number of subjects before dropping nan for weight and height: {len(df_participants)}')
    df_temp = df_participants.dropna(axis=0, subset=['weight', 'height'])
    print(f'Number of subjects after dropping nan for weight and height: {len(df_temp)}')

    for metric in ['age', 'weight', 'height']:
        # Run normality test
        stat, pval = stats.shapiro(df_temp[df_temp['sex'] == 'M'][metric])
        print(f'{metric}: Normality test M: p-value{format_pvalue(pval)}')
        stat, pval = stats.shapiro(df_temp[df_temp['sex'] == 'F'][metric])
        print(f'{metric}: Normality test F: p-value{format_pvalue(pval)}')
        # Run Wilcoxon rank-sum test (groups are independent)
        stat, pval = stats.ranksums(x=df_temp[df_temp['sex'] == 'M'][metric],
                                    y=df_temp[df_temp['sex'] == 'F'][metric])
        print(f'{metric}: Wilcoxon rank-sum test between females and males: '
              f'p-value{format_pvalue(pval)}')

    # Get number of males and females per sex
    print('\nNumber of males and females')
    print(df_participants.groupby(['sex'])['participant_id'].count())
    # Per-sex age, weight, height
    print('\nPer-sex')
    print(round(df_participants.groupby(['sex'])[['age', 'weight', 'height']].agg(['mean', 'std']), 1))

    # Get number of males and females per vendor
    print('\nNumber of males and females')
    print(df_participants.groupby(['manufacturer'])['participant_id'].count())
    # Per-vendor age, weight, height
    print('\nPer-vendor')
    print(round(df_participants.groupby(['manufacturer'])[['age', 'weight', 'height']].agg(['mean', 'std']), 1))

    # Plot age distribution
    plt.figure()
    fig, ax = plt.subplots()
    sns.displot(df_participants['age'], kde=True, color='black')
    # add title
    plt.title('Age distribution', fontsize=LABELS_FONT_SIZE)
    # save figure
    fname_fig = os.path.join(path_out_figures, 'age_distribution.png')
    plt.savefig(fname_fig, dpi=200, bbox_inches="tight")
    plt.close()
    print(f'Created: {fname_fig}.\n')


def compute_normative_values(df, path_out):
    """
    Compute normative values for each metric and save them in a csv file
    Args:
        df:
        path_out:

    Returns:
    """
    for metric in METRICS:
        print(f"\n{metric}")
        # Compute mean and std for each slice across all subjects
        slices_mean = df.groupby(['Slice (I->S)'])[metric].mean()
        slices_std = df.groupby(['Slice (I->S)'])[metric].std()

        # Get indices of slices corresponding vertebral levels
        vert, ind_vert, ind_vert_mid = get_vert_indices(df)

        d = []
        # Loop across intervertebral discs
        for x in reversed(ind_vert[1:-1]):
            slice_number = df.loc[x, 'Slice (I->S)']
            disc = DISCS_DICT[vert[x]]
            slice_mean = slices_mean.loc[slice_number]
            slice_std = slices_std.loc[slice_number]
            print(f'Disc {disc}, slice {slice_number}: {round(slice_mean, 2)} ± {round(slice_std, 2)}')
            d.append(
                {
                    'Disc': disc,
                    'Slice': slice_number,
                    'Mean ± STD': f'{round(slice_mean, 2)} ± {round(slice_std, 2)}'
                }
            )

        fname_csv = os.path.join(path_out, metric + '_disc_normative_values.csv')
        pd.DataFrame(d).to_csv(fname_csv, index=False)
        print(f'Created: {fname_csv}.\n')

        d = []
        # Loop across mid-vertebral slices
        for x in reversed(ind_vert_mid):
            slice_number = df.loc[x, 'Slice (I->S)']
            mid_level = MID_VERT_DICT[vert[x]]
            slice_mean = slices_mean.loc[slice_number]
            slice_std = slices_std.loc[slice_number]
            print(f'Level {mid_level}, slice {slice_number}: {round(slice_mean, 2)} ± {round(slice_std, 2)}')
            d.append(
                {
                    'Level': mid_level,
                    'Slice': slice_number,
                    'Mean ± STD': f'{round(slice_mean, 2)} ± {round(slice_std, 2)}'
                }
            )

        fname_csv = os.path.join(path_out, metric + '_mid_level_normative_values.csv')
        pd.DataFrame(d).to_csv(fname_csv, index=False)
        print(f'Created: {fname_csv}.\n')


def compute_normative_values_persex(df, path_out):
    """
    Compute normative values for each metric persex and save them in a csv file
    Args:
        df:
        path_out:

    Returns:
    """
    for metric in METRICS:
        print(f"\n{metric}")
        # Compute mean and std for each slice across all subjects
        slices_mean = df.groupby(['Slice (I->S)', 'sex'])[metric].mean()
        slices_std = df.groupby(['Slice (I->S)', 'sex'])[metric].std()

        # Get indices of slices corresponding vertebral levels
        vert, ind_vert, ind_vert_mid = get_vert_indices(df)

        d = []
        # Loop across intervertebral discs
        for x in reversed(ind_vert[1:-1]):
            slice_number = df.loc[x, 'Slice (I->S)']
            disc = DISCS_DICT[vert[x]]
            # males
            slice_mean_M = slices_mean.loc[slice_number]['M']
            slice_std_M = slices_std.loc[slice_number]['M']
            # females
            slice_mean_F = slices_mean.loc[slice_number]['F']
            slice_std_F = slices_std.loc[slice_number]['F']
            print(f'M, Disc {disc}, slice {slice_number}: {round(slice_mean_M, 2)} ± {round(slice_std_M, 2)}')
            print(f'F, Disc {disc}, slice {slice_number}: {round(slice_mean_F, 2)} ± {round(slice_std_F, 2)}')
            d.append(
                {
                    'Disc': disc,
                    'Slice': slice_number,
                    'M, Mean ± STD': f'{round(slice_mean_M, 2)} ± {round(slice_std_M, 2)}',
                    'F, Mean ± STD': f'{round(slice_mean_F, 2)} ± {round(slice_std_F, 2)}'
                }
            )

        fname_csv = os.path.join(path_out, metric + '_disc_normative_values_persex.csv')

        pd.DataFrame(d).to_csv(fname_csv, index=False)
        print(f'Created: {fname_csv}.\n')

        d = []
        # Loop across mid-vertebral slices
        for x in reversed(ind_vert_mid):
            slice_number = df.loc[x, 'Slice (I->S)']
            mid_level = MID_VERT_DICT[vert[x]]
            # males
            slice_mean_M = slices_mean.loc[slice_number]['M']
            slice_std_M = slices_std.loc[slice_number]['M']
            # females
            slice_mean_F = slices_mean.loc[slice_number]['F']
            slice_std_F = slices_std.loc[slice_number]['F']
            print(f'M, Level {mid_level}, slice {slice_number}: {round(slice_mean_M, 2)} ± {round(slice_std_M, 2)}')
            print(f'F, Level {mid_level}, slice {slice_number}: {round(slice_mean_F, 2)} ± {round(slice_std_F, 2)}')
            d.append(
                {
                    'Level': mid_level,
                    'Slice': slice_number,
                    'M, Mean ± STD': f'{round(slice_mean_M, 2)} ± {round(slice_std_M, 2)}',
                    'F, Mean ± STD': f'{round(slice_mean_F, 2)} ± {round(slice_std_F, 2)}'
                }
            )

        fname_csv = os.path.join(path_out, metric + '_mid_level_normative_values_persex.csv')
        pd.DataFrame(d).to_csv(fname_csv, index=False)
        print(f'Created: {fname_csv}.\n')


def plot_metrics_relative_to_age(df, path_out_figures):
    """
    # Plot averaged metrics from C2-C3 levels as a function of age separated for sex
    Args:
        df:
        path_out_figures:

    Returns:

    """
    # Compute mean and std from C2 and C3 levels per sex
    df_c2_c3 = df[(df['VertLevel'] == 2) | (df['VertLevel'] == 3)]

    # Recode sex to 0 and 1 to all pd.mean() at the next step
    df_c2_c3['sex'].replace(['F', 'M'], [0, 1], inplace=True)

    for metric in METRICS:
        # Get mean values for each subject
        df_c2_c3_mean = df_c2_c3.groupby(['participant_id'])[[metric, 'age', 'sex']].mean()

        # Plot averaged metrics from C2-C3 levels as a function of age
        fig, ax = plt.subplots(figsize=(20, 6))
        sns.regplot(x='age', y=metric, data=df_c2_c3_mean[df_c2_c3_mean['sex'] == 1],
                    label='Male', color=COLORS_SEX['M'])
        sns.regplot(x='age', y=metric, data=df_c2_c3_mean[df_c2_c3_mean['sex'] == 0],
                    label='Female', color=COLORS_SEX['F'])
        ax.legend()
        ax.set_xlabel('Age (years)')
        ax.set_ylabel(METRIC_TO_AXIS[metric])
        ax.set_title(f'{METRIC_TO_TITLE[metric]} from C2-C3 levels as a function of age')

        # Save figure
        fname_fig = os.path.join(path_out_figures, metric + '_C2_C3_vs_age_persex.png')
        fig.savefig(fname_fig, bbox_inches='tight')
        print(f'Created: {fname_fig}.\n')


def explore_linearity(df, path_out_figures):
    """
    Plot the following:
        - correlation matrix to examine the pairwise correlations between morphometric measures
        - scatter plots for each pair of variables
    Args:
        df: dataframe
        path_out_figures: path to output figures
    """

    # Keep only columns of interest
    df = df[METRIC_TO_TITLE.keys()]

    # Calculate the correlation matrix
    correlation_matrix = df.corr()

    # Display the correlation matrix as a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix')
    # Save figure
    fname_fig = os.path.join(path_out_figures, 'correlation_matrix.png')
    plt.savefig(fname_fig, bbox_inches='tight', dpi=300)
    print(f'Created: {fname_fig}.\n')

    # Plot scatter plots for each pair of variables
    fname_fig = os.path.join(path_out_figures, 'scatter_plots.png')
    sns.pairplot(df).savefig(fname_fig, bbox_inches='tight', dpi=300)
    print(f'Created: {fname_fig}.\n')


def main():
    parser = get_parser()
    args = parser.parse_args()
    path_HC = args.path_HC
    path_out_figures = os.path.join(args.path_out, 'figures')
    path_out_csv = os.path.join(args.path_out, 'csv')
    # If the output folder directory is not present, then create it.
    if not os.path.exists(path_out_figures):
        os.makedirs(path_out_figures)
    if not os.path.exists(path_out_csv):
        os.makedirs(path_out_csv)
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
    compute_descriptive_stats(df_participants, path_out_figures)

    df = df.dropna(axis=1, how='all')
    df = df.dropna(axis=0, how='any').reset_index(drop=True) # do we want to compute mean with missing levels for some subjects?
    # Keep only VertLevel from C1 to Th1
    df = df[df['VertLevel'] <= 8]

    explore_linearity(df, path_out_figures)

    # Plot metrics as a function of age
    plot_metrics_relative_to_age(df, path_out_figures)

    # Recode age into age bins by 10 years (decades)
    df['age'] = pd.cut(df['age'], bins=[10, 20, 30, 40, 50, 60], labels=AGE_DECADES)

    # Compute Wilcoxon rank-sum test between males and females for across levels for each metric
    compare_metrics_across_sex(df)
    # Compute Wilcoxon rank-sum tests between MRI vendors across all levels for each metric
    compare_metrics_across_vendors(df)
    # Get mean values for each age decade
    analyze_metrics_across_age_decades(df)

    # Plot correlation between weight and height per sex
    gen_chart_weight_height(df, df_participants, path_out_figures)

    # Plot relationship between demographics (BMI, weight, height) and MRI metrics persex
    create_regplot_demographics_vs_metrics(df, path_out_figures)

    # Compute mean and std from C2 and C3 levels across sex and compare females and males
    compute_c2_c3_stats(df)

    # Multiply solidity by 100 to get percentage (sct_process_segmentation computes solidity in the interval 0-1)
    df['MEAN(solidity)'] = df['MEAN(solidity)'] * 100

    # Uncomment to save aggregated dataframe with metrics across all subjects as .csv file
    #df.to_csv(os.path.join(path_out_csv, 'HC_metrics.csv'), index=False)

    # Compute normative values
    compute_normative_values(df, path_out_csv)
    compute_normative_values_persex(df, path_out_csv)

    # Create plots
    create_lineplot(df, None, path_out_figures, show_cv=True)        # across all subjects
    create_lineplot(df, 'age', path_out_figures)       # across age
    create_lineplot(df, 'sex', path_out_figures)       # across sex
    create_lineplot(df, 'manufacturer', path_out_figures)  # across manufacturer (vendors)

    # Plot scatterplot metrics vs COV
    create_regplot(df, path_out_figures, show_cv=True)

    # Plot scatterplot metrics vs COV per sex
    create_regplot_per_sex(df, path_out_figures)


if __name__ == '__main__':
    main()
