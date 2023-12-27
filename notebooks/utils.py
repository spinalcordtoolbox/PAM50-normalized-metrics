import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from repo2data.repo2data import Repo2Data

METRICS = ['MEAN(area)', 'MEAN(diameter_AP)', 'MEAN(diameter_RL)', 'MEAN(compression_ratio)', 'MEAN(eccentricity)',
           'MEAN(solidity)']

FIGURE_SIZE = [960, 649]

# vertical_spacing, horizontal_spacing
SUBPLOT_SPACING = [0.08, 0.06]

# Figure margins in px
FIGURE_MARGIN = dict(t=40, r=50, b=10, l=50)

METRIC_TO_TITLE = {
    'MEAN(area)': 'Cross-Sectional Area',
    'MEAN(diameter_AP)': 'AP Diameter',
    'MEAN(diameter_RL)': 'Transverse Diameter',
    'MEAN(compression_ratio)': 'Compression Ratio',
    'MEAN(eccentricity)': 'Eccentricity',
    'MEAN(solidity)': 'Solidity',
}

METRIC_TO_AXIS = {
    'MEAN(diameter_AP)': 'AP Diameter [mm]',
    'MEAN(area)': 'Cross-Sectional Area [mm²]',
    'MEAN(diameter_RL)': 'Transverse Diameter [mm]',
    'MEAN(eccentricity)': 'Eccentricity [a.u.]',
    'MEAN(solidity)': 'Solidity [%]',
    'MEAN(compression_ratio)': 'Compression Ratio [a.u.]',
}

# Set ylim to do not overlap horizontal grid with vertebrae labels
METRICS_TO_YLIM = {
    'MEAN(diameter_AP)': (5.7, 9.3),
    'MEAN(area)': (35, 95),
    'MEAN(diameter_RL)': (8.5, 14.5),
    'MEAN(eccentricity)': (0.51, 0.89),
    'MEAN(solidity)': (90, 99.9),
    'MEAN(compression_ratio)': (0.41, 0.84),
}

# ylim offset (used for showing text)
METRICS_TO_YLIM_OFFSET = {
    'MEAN(diameter_AP)': 0.15,
    'MEAN(area)': 2.5,
    'MEAN(diameter_RL)': 0.25,
    'MEAN(eccentricity)': 0.015,
    'MEAN(solidity)': 0.4,
    'MEAN(compression_ratio)': 0.02,
}

# x-axis tick values
XTICKVALS = [950, 900, 850, 800, 750, 700]

LABELS_FONT_SIZE = 16
TICKS_FONT_SIZE = 16
TICKS_FONT_SIZE_SUBPLOT = 14

LEGEND_ITEMS = {
    'sex': {'M': 'Males', 'F': 'Females'},
    'vendor': {'Siemens': 'Siemens', 'Philips': 'Philips', 'GE': 'GE'},
    'age': {'10-20': '10-20', '21-30': '21-30', '31-40': '31-40', '41-50': '41-50', '51-60': '51-60'},
}

PALETTE = {
    'sex': {'M': 'blue', 'F': 'red'},
    'vendor': {'Siemens': 'green', 'Philips': 'dodgerblue', 'GE': 'black'},
    'age': {'10-20': 'blue', '21-30': 'green', '31-40': 'black', '41-50': 'red', '51-60': 'purple'},
}

# paletter with 0.2 opacity -- used for fillcolor
PALETTE_RGBA = {
    'sex': {'M': 'rgba(0, 0, 255, 0.2)', 'F': 'rgba(255, 0, 0, 0.2)'},
    'vendor': {'Siemens': 'rgba(0, 128, 0, 0.2)', 'Philips': 'rgba(30, 144, 255, 0.2)', 'GE': 'rgba(0, 0, 0, 0.2)'},
    'age': {'10-20': 'rgba(0, 0, 255, 0.2)', '21-30': 'rgba(0, 128, 0, 0.2)', '31-40': 'rgba(0, 0, 0, 0.2)',
            '41-50': 'rgba(255, 0, 0, 0.2)', '51-60': 'rgba(128, 0, 128, 0.2)'}
}


def create_subplot(df, output='show', output_fname=None):
    """
    Create 2x3 subplot with lineplots for all MRI metric per vertebral levels.
    Note: we are ploting slices not levels to avoid averaging across levels.
    Args:
        df (pd.dataFrame): dataframe with metric values across all healthy subjects
        output (str): show (show figure in jupyter notebook), save (save figure as png) or html (return plotly object,
        which can be rendered as html in jupyter book)
        output_fname: output figure filename; valid only for output='save'
    """

    slices = df["Slice (I->S)"]

    # Create subplots with 2 rows and 3 columns
    fig = make_subplots(rows=2, cols=3, vertical_spacing=SUBPLOT_SPACING[0], horizontal_spacing=SUBPLOT_SPACING[1])

    # Iterate over each metric and plot on the corresponding subplot
    for i, metric in enumerate(METRICS):
        row = (i // 3) + 1
        col = (i % 3) + 1

        # Calculating mean and standard deviation
        mean = df.groupby("Slice (I->S)")[metric].mean()
        std = df.groupby("Slice (I->S)")[metric].std()

        # Add trace for upper standard deviation
        fig.add_trace(
            go.Scatter(
                x=slices,
                y=mean + std,
                mode='lines',
                line=dict(color=PALETTE['sex']['M'],
                          width=0.5),
                name='',
                hovertemplate=
                'STD: %{y:.2f}' +
                '<br>Slice: %{x}'
            ),
            row=row,
            col=col
        )
        # Add trace for lower standard deviation and fill to the upper standard deviation
        fig.add_trace(
            go.Scatter(
                x=slices,
                y=mean - std,
                mode='lines',
                line=dict(color=PALETTE['sex']['M'],
                          width=0.5),
                fill='tonexty',
                fillcolor=PALETTE_RGBA['sex']['M'],
                name='',
                hovertemplate=
                'STD: %{y:.2f}' +
                '<br>Slice: %{x}'
            ),
            row=row,
            col=col
        )
        # Add trace for mean
        fig.add_trace(
            go.Scatter(
                x=slices,
                y=mean,
                mode='lines',
                line=dict(color=PALETTE['sex']['M'],
                          width=3),
                name='',
                hovertemplate=
                'Mean: %{y:.2f}' +
                '<br>Slice: %{x}',
            ),
            row=row,
            col=col
        )

        # Insert a vertical line for each vertebral level
        # Get indices of slices corresponding to mid-vertebrae
        vert, ind_vert, ind_vert_mid = get_vert_indices(df)
        for idx, x in enumerate(ind_vert[1:-1]):
            fig.add_trace(
                go.Scatter(
                    x=[df.loc[x, 'Slice (I->S)'], df.loc[x, 'Slice (I->S)']],
                    y=[METRICS_TO_YLIM[metric][0], METRICS_TO_YLIM[metric][1]],
                    mode='lines',
                    line=dict(
                        color='black',
                        width=1,
                        dash='dash'
                    ),
                    showlegend=False,
                    hoverinfo='none'
                ),
                row=row,
                col=col
            )

        # Hide the legend for each trace
        for trace in fig.data:
            trace.showlegend = False

        # Update the x-axis settings
        fig.update_xaxes(
            autorange="reversed",  # Reverse the x-axis for axial slices
            title="Axial Slice #",  # Set the x-axis label
            title_font=dict(size=TICKS_FONT_SIZE_SUBPLOT),  # Adjust title font size
            tickfont=dict(size=TICKS_FONT_SIZE_SUBPLOT),  # Adjust tick font size
            title_standoff=0,  # Set title standoff
            showgrid=False,  # Hide grid lines
            tickvals=XTICKVALS,  # Set tick values
            showline=True,  # Show axis line
            linecolor='gray',  # Set axis line color
            showticklabels=True,  # Show tick labels
            row=row,  # Specify the row of the subplot
            col=col  # Specify the column of the subplot
        )
        # Update y-axis settings
        fig.update_yaxes(
            title=METRIC_TO_AXIS[metric],  # Set the y-axis label based on the metric
            title_font=dict(size=TICKS_FONT_SIZE_SUBPLOT),  # Adjust title font size
            tickfont=dict(size=TICKS_FONT_SIZE_SUBPLOT),  # Adjust tick font size
            title_standoff=0,  # Set title standoff
            range=[METRICS_TO_YLIM[metric][0], METRICS_TO_YLIM[metric][1]],  # Set y-axis range
            row=row,  # Specify the row of the subplot
            col=col,  # Specify the column of the subplot
            showgrid=True,  # Show grid lines
            gridcolor='lightgray'  # Set grid color
        )

        # Insert a text label for each vertebral level
        for idx, x in enumerate(ind_vert_mid, 0):
            # Th
            if vert[x] > 7:
                level = 'T' + str(vert[x] - 7)
            # Other levels
            else:
                level = 'C' + str(vert[x])

            fig.add_annotation(
                x=df.loc[ind_vert_mid[idx], 'Slice (I->S)'],
                y=METRICS_TO_YLIM[metric][0] + METRICS_TO_YLIM_OFFSET[metric],
                text=level,
                showarrow=False,
                font=dict(size=TICKS_FONT_SIZE_SUBPLOT),
                row=row, col=col
            )

        # Add custom tick marks using annotations
        for tick_val in XTICKVALS:
            fig.add_annotation(
                x=tick_val,  # X position of the tick
                y=METRICS_TO_YLIM[metric][0],  # Y position of the tick
                text='',  # You can add a label here if needed
                showarrow=True,
                arrowhead=0,
                ax=0,  # Adjust this to control the tick length
                ay=5,  # Adjust this to position the tick properly
                row=row,
                col=col
            )

    # Update the subplot sizes
    fig.update_layout(
        width=FIGURE_SIZE[0],
        height=FIGURE_SIZE[1],
        plot_bgcolor='white',
        margin=FIGURE_MARGIN
    )

    if output == 'show':
        fig.show()
    elif output == 'save':
        save_figure(fig, output_fname)
    elif output == 'html':
        return fig


def create_subplot_hue(df, hue, output='show', output_fname=None):
    """
    Create 2x3 subplot with lineplots for all MRI metric per vertebral levels for a specific hue ('age', 'sex', 'vendor').
    Note: we are ploting slices not levels to avoid averaging across levels.
    Args:
        df (pd.dataFrame): dataframe with metric values across all healthy subjects
        hue (str): column name of the dataframe to use for grouping
        output (str): show (show figure in jupyter notebook), save (save figure as png) or html (return plotly object,
        which can be rendered as html in jupyter book)
        output_fname: output figure filename; valid only for output='save'
    """

    # Get a list of unique categories for the hue variable
    categories = df[hue].unique()
    categories.sort()
    slices = df["Slice (I->S)"]

    # Create subplots with 2 rows and 3 columns
    fig = make_subplots(rows=2, cols=3, vertical_spacing=SUBPLOT_SPACING[0], horizontal_spacing=SUBPLOT_SPACING[1])

    # Iterate over each metric and plot on the corresponding subplot
    for i, metric in enumerate(METRICS):
        row = (i // 3) + 1
        col = (i % 3) + 1

        # Calculating mean and standard deviation for each category
        for j, category in enumerate(categories):
            category_data = df[df[hue] == category]
            mean = category_data.groupby("Slice (I->S)")[metric].mean()
            std = category_data.groupby("Slice (I->S)")[metric].std()

            # Add trace for upper standard deviation
            fig.add_trace(
                go.Scatter(
                    x=slices,
                    y=mean + std,
                    mode='lines',
                    line=dict(
                        color=PALETTE[hue][category],
                        width=0.5
                    ),
                    name=LEGEND_ITEMS[hue][category],
                    legendgroup=category,
                    hovertemplate=
                    'STD: %{y:.2f}' +
                    '<br>Slice: %{x}',
                    showlegend=False
                ),
                row=row,
                col=col
            )
            # Add trace for lower standard deviation and fill to the upper standard deviation
            fig.add_trace(
                go.Scatter(
                    x=slices,
                    y=mean - std,
                    mode='lines',
                    line=dict(
                        color=PALETTE[hue][category],
                        width=0.5
                    ),
                    fill='tonexty',
                    fillcolor=PALETTE_RGBA[hue][category],
                    name=LEGEND_ITEMS[hue][category],
                    legendgroup=category,
                    hovertemplate=
                    'STD: %{y:.2f}' +
                    '<br>Slice: %{x}',
                    showlegend=False
                ),
                row=row,
                col=col
            )

            # Add trace for mean
            # Note: legend is added only for the first subplot to avoid legend item duplications
            if i == 0:
                fig.add_trace(
                    go.Scatter(
                        x=slices,
                        y=mean,
                        mode='lines',
                        line=dict(
                            color=PALETTE[hue][category],
                            width=3
                        ),
                        name=LEGEND_ITEMS[hue][category],
                        legendgroup=category,
                        hovertemplate=
                        'Mean: %{y:.2f}' +
                        '<br>Slice: %{x}',
                    ),
                    row=row,
                    col=col
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=slices,
                        y=mean,
                        mode='lines',
                        line=dict(
                            color=PALETTE[hue][category],
                            width=3
                        ),
                        name=LEGEND_ITEMS[hue][category],
                        legendgroup=category,
                        hovertemplate=
                        'Mean: %{y:.2f}' +
                        '<br>Slice: %{x}',
                        showlegend=False
                    ),
                    row=row,
                    col=col
                )

        # Insert a vertical line for each vertebral level
        # Get indices of slices corresponding to mid-vertebrae
        vert, ind_vert, ind_vert_mid = get_vert_indices(df)
        for idx, x in enumerate(ind_vert[1:-1]):
            fig.add_trace(
                go.Scatter(
                    x=[df.loc[x, 'Slice (I->S)'], df.loc[x, 'Slice (I->S)']],
                    y=[METRICS_TO_YLIM[metric][0], METRICS_TO_YLIM[metric][1]],
                    mode='lines',
                    line=dict(
                        color='black',
                        width=1,
                        dash='dash'
                    ),
                    showlegend=False,
                    hoverinfo='none'
                ),
                row=row, col=col
            )

        # Update the x-axis settings
        fig.update_xaxes(
            autorange="reversed",  # Reverse the x-axis for axial slices
            title="Axial Slice #",  # Set the x-axis label
            title_font=dict(size=TICKS_FONT_SIZE_SUBPLOT),  # Adjust title font size
            tickfont=dict(size=TICKS_FONT_SIZE_SUBPLOT),  # Adjust tick font size
            title_standoff=0,  # Set title standoff
            showgrid=False,  # Hide grid lines
            tickvals=XTICKVALS,  # Set tick values
            showline=True,  # Show axis line
            linecolor='gray',  # Set axis line color
            showticklabels=True,  # Show tick labels
            row=row,  # Specify the row of the subplot
            col=col  # Specify the column of the subplot
        )
        # Update y-axis settings
        fig.update_yaxes(
            title=METRIC_TO_AXIS[metric],  # Set the y-axis label based on the metric
            title_font=dict(size=TICKS_FONT_SIZE_SUBPLOT),  # Adjust title font size
            tickfont=dict(size=TICKS_FONT_SIZE_SUBPLOT),  # Adjust tick font size
            title_standoff=0,  # Set title standoff
            range=[METRICS_TO_YLIM[metric][0], METRICS_TO_YLIM[metric][1]],  # Set y-axis range
            row=row,  # Specify the row of the subplot
            col=col,  # Specify the column of the subplot
            showgrid=True,  # Show grid lines
            gridcolor='lightgray'  # Set grid color
        )

        # Insert a text label for each vertebral level
        for idx, x in enumerate(ind_vert_mid, 0):
            # Th
            if vert[x] > 7:
                level = 'T' + str(vert[x] - 7)
            # Other levels
            else:
                level = 'C' + str(vert[x])

            fig.add_annotation(
                x=df.loc[ind_vert_mid[idx], 'Slice (I->S)'],
                y=METRICS_TO_YLIM[metric][0] + METRICS_TO_YLIM_OFFSET[metric],
                text=level,
                showarrow=False,
                font=dict(size=TICKS_FONT_SIZE_SUBPLOT),
                row=row,
                col=col
            )

        # Add custom tick marks using annotations
        for tick_val in XTICKVALS:
            fig.add_annotation(
                x=tick_val,  # X position of the tick
                y=METRICS_TO_YLIM[metric][0],  # Y position of the tick
                text='',  # You can add a label here if needed
                showarrow=True,
                arrowhead=0,
                ax=0,  # Adjust this to control the tick length
                ay=5,  # Adjust this to position the tick properly
                row=row,
                col=col
            )

    # Update the subplot sizes
    fig.update_layout(
        width=FIGURE_SIZE[0],
        height=FIGURE_SIZE[1],
        plot_bgcolor='white',
        legend_title_text=hue,
        legend={'traceorder': 'normal'},
        margin=FIGURE_MARGIN
    )

    if output == 'show':
        fig.show()
    elif output == 'save':
        save_figure(fig, output_fname)
    elif output == 'html':
        return fig


def create_regplot(df, show_cv=False, output='show', output_fname=None):
    """
    Plot data and a linear regression model fit. Slices in X and Coefficient of Variation (CoV) in Y.
    Args:
        df (pd.dataFrame): dataframe with metric values
        show_cv (bool): if True, include coefficient of variation for each vertebral level to the plot
        output (str): show (show figure in jupyter notebook), save (save figure as png) or html (return plotly object,
        which can be rendered as html in jupyter book)
        output_fname: output figure filename; valid only for output='save'
    """

    # Set the same y-lim for all subplots
    YLIMS = [-0.5, 19.5]

    fig = make_subplots(rows=2, cols=3, vertical_spacing=SUBPLOT_SPACING[0], horizontal_spacing=SUBPLOT_SPACING[1])

    # Loop across metrics
    for i, metric in enumerate(METRICS):

        row = (i // 3) + 1
        col = (i % 3) + 1

        slices_list = []
        cv_list = []
        # Loop across slices
        for slice in df['Slice (I->S)'].unique():
            # Get metric value for each slice
            df_slice = df[df['Slice (I->S)'] == slice]
            cv_list.append(compute_cv(df_slice, metric))
            slices_list.append(slice)

        # Create a scatter plot with linear regression line using Plotly
        fig.add_trace(
            go.Scatter(
                x=slices_list,
                y=cv_list,
                mode='markers',
                marker=dict(size=6, color=PALETTE['sex']['M'], opacity=0.5),
                name='',
                hovertemplate=
                'COV: %{y:.2f}%' +
                '<br>Slice: %{x}',
            ),
            row=row,
            col=col
        )

        # Insert a vertical line for each vertebral level
        # Get indices of slices corresponding to mid-vertebrae
        vert, ind_vert, ind_vert_mid = get_vert_indices(df)
        for idx, x in enumerate(ind_vert[1:-1]):
            fig.add_trace(
                go.Scatter(
                    x=[df.loc[x, 'Slice (I->S)'], df.loc[x, 'Slice (I->S)']],
                    y=YLIMS,
                    mode='lines',
                    line=dict(
                        color='black',
                        width=1,
                        dash='dash'
                    ),
                    showlegend=False,
                    hoverinfo='none'
                ),
                row=row, col=col
            )

            # Hide the legend for each trace
        for trace in fig.data:
            trace.showlegend = False

        # Update the x-axis settings
        fig.update_xaxes(
            autorange="reversed",  # Reverse the x-axis for axial slices
            title="Axial Slice #",  # Set the x-axis label
            title_font=dict(size=TICKS_FONT_SIZE_SUBPLOT),  # Adjust title font size
            tickfont=dict(size=TICKS_FONT_SIZE_SUBPLOT),  # Adjust tick font size
            title_standoff=0,  # Set title standoff
            showgrid=False,  # Hide grid lines
            tickvals=XTICKVALS,  # Set tick values
            showline=True,  # Show axis line
            linecolor='gray',  # Set axis line color
            showticklabels=True,  # Show tick labels
            row=row,  # Specify the row of the subplot
            col=col  # Specify the column of the subplot
        )

        # Update y-axis settings
        fig.update_yaxes(
            range=YLIMS,  # Set y-axis range
            title=METRIC_TO_TITLE[metric] + ' COV [%]',  # Set the y-axis label based on the metric
            title_font=dict(size=TICKS_FONT_SIZE_SUBPLOT),  # Adjust title font size
            tickfont=dict(size=TICKS_FONT_SIZE_SUBPLOT),  # Adjust tick font size
            title_standoff=0,  # Set title standoff
            row=row,  # Specify the row of the subplot
            col=col,  # Specify the column of the subplot
            showgrid=True,  # Show grid lines
            gridcolor='lightgray'  # Set grid color
        )

        # Insert a text label for each vertebral level
        for idx, x in enumerate(ind_vert_mid, 0):
            # Th
            if vert[x] > 7:
                level = 'T' + str(vert[x] - 7)
            # Other levels
            else:
                level = 'C' + str(vert[x])

            fig.add_annotation(
                x=df.loc[ind_vert_mid[idx], 'Slice (I->S)'],
                y=0.3,
                text=level,
                showarrow=False,
                font=dict(size=TICKS_FONT_SIZE_SUBPLOT),
                row=row,
                col=col
            )

        # Add custom tick marks using annotations
        for tick_val in XTICKVALS:
            fig.add_annotation(
                x=tick_val,  # X position of the tick
                y=YLIMS[0],  # Y position of the tick
                text='',  # You can add a label here if needed
                showarrow=True,
                arrowhead=0,
                ax=0,  # Adjust this to control the tick length
                ay=5,  # Adjust this to position the tick properly
                row=row,
                col=col
            )

        # Add COV text labels for each vert level
        for idx, x in enumerate(ind_vert_mid, 0):
            if show_cv:
                cv = compute_cv(df[(df['VertLevel'] == vert[x])], metric)
                cv_text = f'{cv:.1f}%'

                fig.add_annotation(
                    x=df.loc[ind_vert_mid[idx], 'Slice (I->S)'],
                    y=16,
                    text=cv_text,
                    showarrow=False,
                    font=dict(size=TICKS_FONT_SIZE_SUBPLOT - 6),
                    row=row,
                    col=col
                )

        # Place text box with mean and std COV value
        middle_slice = int(len(slices_list) / 2)
        middle_slice_value = slices_list[middle_slice]
        mean_cov_text = 'COV: {}±{}%'.format(round(np.mean(cv_list), 1), round(np.std(cv_list), 1))
        fig.add_annotation(
            text=mean_cov_text,
            x=middle_slice_value, y=18,
            showarrow=False,
            font=dict(size=TICKS_FONT_SIZE_SUBPLOT),
            bordercolor='black',
            bgcolor='white',
            borderwidth=1,
            row=row,
            col=col
        )

    # Update layout and save figure
    fig.update_layout(
        width=FIGURE_SIZE[0],
        height=FIGURE_SIZE[1],
        plot_bgcolor='white',
        margin=FIGURE_MARGIN
    )

    if output == 'show':
        fig.show()
    elif output == 'save':
        save_figure(fig, output_fname)
    elif output == 'html':
        return fig


def save_figure(fig, fname_fig):
    """
    Save figure as png into figures folder
    """
    if not os.path.exists('figures'):
        os.mkdir('figures')
    fig.write_image(os.path.join('figures', fname_fig), scale=4)
    print(f"Created: {os.path.join('figures', fname_fig)}.")


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
    for i in range(len(ind_vert) - 1):
        ind_vert_mid.append(int(ind_vert[i:i + 2].mean()))

    return vert, ind_vert, ind_vert_mid


def read_metrics(path_csv):
    """
    Read the CSV file with aggregated metrics for all subjects
    Args:
        path_csv:
    Returns:
        df (pd.dataFrame): dataframe with aggregated metrics
    """
    # Read the CSV file with aggregated metrics for all subjects
    df = pd.read_csv(path_csv)
    # Rename manufacturer column to vendor to match manuscript
    df = df.rename(columns={'manufacturer': 'vendor'})

    return df


def fetch_data():
    """
    Fetch data using repo2data
    """
    # define data requirement path
    data_req_path = os.path.join("data_requirement.json")
    # download data
    repo2data = Repo2Data(data_req_path)
    data_path = repo2data.install()

