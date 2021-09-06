"""This module contains helper functions for plotting."""
from typing import Tuple, Union, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import helpsk.pandas as hpandas
from helpsk.exceptions import HelpskParamValueError

STANDARD_HEIGHT = 10
GOLDEN_RATIO = 1.61803398875
STANDARD_WIDTH = STANDARD_HEIGHT / GOLDEN_RATIO
STANDARD_HEIGHT_WIDTH = (STANDARD_HEIGHT, STANDARD_WIDTH)


def plot_value_frequency(series: pd.Series, sort_by_frequency: bool = True,
                         figure_size: Tuple[int, int] = STANDARD_HEIGHT_WIDTH,
                         x_axis_rotation: int = 30) -> None:
    """Shows the unique values and corresponding frequencies.

    Args:
        series:
            a Pandas series, either categorical or with integers.
        sort_by_frequency:
            if True then sort by frequency desc; otherwise sort by index (either numerically ascending if
            series is numeric, or alphabetically if non-ordered categoric, or by category if ordered categoric
        figure_size:
            tuple containing `(height, width)` of plot. The default height is defined by
            `STANDARD_HEIGHT`, and the default width is `STANDARD_HEIGHT / GOLDEN_RATIO`
        x_axis_rotation:
            the angle to rotate the x-axis text.
    """
    value_frequencies = hpandas.value_frequency(series=series, sort_by_frequency=sort_by_frequency)
    plot_object = value_frequencies.drop(columns='Percent').plot(kind='bar', rot=10,
                                                                 ylabel='Frequency',
                                                                 title=f'Name: `{series.name}`',
                                                                 legend=False,
                                                                 figsize=figure_size)
    for idx, label in enumerate(list(value_frequencies.index)):
        if pd.isna(label):
            frequency = value_frequencies[value_frequencies.index.isna()]['Frequency'].iloc[0]
            percent = value_frequencies[value_frequencies.index.isna()]['Percent'].iloc[0]
        else:
            frequency = value_frequencies.loc[label, 'Frequency']
            percent = value_frequencies.loc[label, 'Percent']
        plot_object.annotate(frequency,
                             xy=(idx, frequency),
                             xytext=(0, 2),
                             ha='center',
                             textcoords='offset points')
        plot_object.annotate("{0:.0f}%".format(percent * 100),
                             xy=(idx, max(frequency - 10, 2)),
                             xytext=(0, 0),
                             color='white',
                             ha='center',
                             textcoords='offset points')

    plot_object.set_xticklabels(labels=value_frequencies.index.values, rotation=x_axis_rotation, ha='right')
    plt.tight_layout()


def plot_correlation_heatmap(dataframe: pd.DataFrame,
                             threshold: Optional[float] = None,
                             title: Optional[str] = None,
                             figure_size: tuple = (STANDARD_HEIGHT, STANDARD_HEIGHT),
                             round_by: int = 2,
                             features_to_highlight: Optional[list] = None) -> None:
    """Creates a heatmap of the correlations between all of the numeric columns.

    Args:
        dataframe:
            dataframe to get correlations from. Automatically chooses all numeric columns.
        threshold:
            threshold: the heatmap only includes columns that have a correlation value, corresponding to at
        least one other column, where the absolute value is higher than the threshold.

        So for example, if the threshold is `0.8` then all columns that have a correlation (absolute)
            value of `>=.80` anywhere in the correlation matrix (i.e. with any other column), will show in
            the heatmap, even though a specific column might not have high correlations with every other
            column included.
        title:
            title of plot
        figure_size:
            tuple (height, width)
        round_by:
            the number of decimal places to round to when showing the correlations in the heatmap
        features_to_highlight:
            feature labels to highlight in red
    """
    
    correlations = dataframe.corr()

    if threshold is not None:
        features = correlations.columns.values
        correlation_matrix = np.abs(correlations.values)
        np.fill_diagonal(correlation_matrix, np.NaN)
        meets_threshold = np.apply_along_axis(lambda x: np.any(x >= threshold), 0, correlation_matrix)
        if not meets_threshold.any():
            raise HelpskParamValueError('correlation `threshold` set too high.')

        features_meets_threshold = features[meets_threshold]
        correlations = correlations.loc[features_meets_threshold, features_meets_threshold]

    if title is None:
        title = ''

    # remove duplicates (i.e. mirror image in second half of heatmap
    mask = np.zeros_like(correlations)
    mask[np.triu_indices_from(mask)] = True

    with sns.axes_style("white"):
        f, ax = plt.subplots()
        f.set_size_inches(figure_size[0], figure_size[1])
        sns.heatmap(correlations,
                    mask=mask,
                    annot=True,
                    fmt='.{}f'.format(round_by),
                    cmap=sns.diverging_palette(220, 10, as_cmap=True),
                    vmin=-1,
                    vmax=1,
                    square=True, ax=ax,
                    center=0)
        plt.xticks(rotation=20, ha='right')
        plt.yticks(rotation=0)
        plt.title(title)
        plt.tight_layout()

        if features_to_highlight is not None:
            indexes_to_highlight = [index for index, x in enumerate(correlations.index.values)
                                    if x in features_to_highlight]

            for index_to_highlight in indexes_to_highlight:
                plt.gca().get_xticklabels()[index_to_highlight].set_color('red')
                plt.gca().get_yticklabels()[index_to_highlight].set_color('red')

    plt.tight_layout()
