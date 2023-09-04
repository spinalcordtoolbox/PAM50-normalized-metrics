import pandas as pd
import plotly.graph_objects as go


FONT_STYLE = {'family': 'Arial', 'color': 'black'}
FIGURE_MARGIN = dict(t=40, r=50, b=10, l=50)


def table_demographic_characteristics(path_csv):
    """
    Show table from csv file
    Args:
        path_csv:
    """

    header_values = ['',
                     'Whole Cohort',
                     'Males',
                     'Females',
                     'p-value Males-Females']

    df = pd.read_csv(path_csv)

    # Add HTML <b> tags to make the header text bold
    header_values = ['<b>' + val + '</b>' if val != '' else '' for val in header_values]

    # Define the font and fill colors
    font_style = FONT_STYLE

    # Calculate the height based on the number of rows
    table_height = 40 + len(df) * 40  # Assuming 40px height per row

    fig = go.Figure(data=[go.Table(
        header=dict(values=header_values,
                    align=['left', 'center'],
                    fill_color='grey',
                    font=font_style),
        cells=dict(values=df.transpose().values.tolist(),
                   align=['left', 'center'],
                   fill_color=[['white', 'lightgrey'] * (len(df) + 1)],
                   font=font_style))
    ])

    fig.update_layout(
        margin=FIGURE_MARGIN,
        height=table_height)

    fig.show()


def table_whole_cohort(path_csv):
    """
    Show table from csv file
    Args:
        path_csv:
    """

    header_values = ['',
                     'Axial Slice #',
                     'CSA [mm²]',
                     'AP Diameter [mm]',
                     'RL Diameter [mm]',
                     'Compression Ratio [a.u.]',
                     'Eccentricity [a.u.]',
                     'Solidity [%]']

    df = pd.read_csv(path_csv)

    # Add HTML <b> tags to make the header text bold
    header_values = ['<b>' + val + '</b>' if val != '' else '' for val in header_values]

    # Define the font and fill colors
    font_style = FONT_STYLE

    # Calculate the height based on the number of rows
    table_height = 40 + len(df) * 40  # Assuming 40px height per row

    fig = go.Figure(data=[go.Table(
        header=dict(values=header_values,
                    align=['left', 'center'],
                    fill_color='grey',
                    font=font_style),
        cells=dict(values=df.transpose().values.tolist(),
                   align=['left', 'center'],
                   fill_color=[['white', 'lightgrey'] * (len(df) + 1)],
                   font=font_style))
    ])

    fig.update_layout(
        margin=FIGURE_MARGIN,
        height=table_height)

    fig.show()


def table_persex(path_csv):
    """
    Show table from csv file
    Args:
        path_csv:
    """

    header_values = ['',
                     'Axial Slice #',
                     'CSA [mm²]',
                     '',
                     'AP Diameter [mm]',
                     '',
                     'RL Diameter [mm]',
                     '',
                     'Compression Ratio [a.u.]',
                     '',
                     'Eccentricity [a.u.]',
                     '',
                     'Solidity [%]']

    df = pd.read_csv(path_csv)

    # Add HTML <b> tags to make the header text bold
    header_values = ['<b>' + val + '</b>' if val != '' else '' for val in header_values]

    # Define the font and fill colors
    font_style = FONT_STYLE

    # Calculate the height based on the number of rows
    table_height = 40 + len(df) * 40  # Assuming 40px height per row

    fig = go.Figure(data=[go.Table(
        header=dict(values=header_values,
                    align=['left', 'center'],
                    fill_color='grey',
                    font=font_style),
        cells=dict(values=df.transpose().values.tolist(),
                   align=['left', 'center'],
                   fill_color=[['white', 'lightgrey'] * (len(df) + 1)],
                   font=font_style))
    ])

    fig.update_layout(
        margin=FIGURE_MARGIN,
        height=table_height)

    fig.show()
