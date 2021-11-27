"""This module contains helper functions for plotting."""
from typing import Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import helpsk.pandas as hpandas
from helpsk.exceptions import HelpskParamValueError

STANDARD_WIDTH = 10
GOLDEN_RATIO = 1.61803398875
STANDARD_HEIGHT = STANDARD_WIDTH / GOLDEN_RATIO
STANDARD_WIDTH_HEIGHT = (STANDARD_WIDTH, STANDARD_HEIGHT)


def plot_value_frequency(series: pd.Series, sort_by_frequency: bool = True,
                         figure_size: Tuple[int, int] = STANDARD_WIDTH_HEIGHT,
                         x_axis_rotation: int = 30) -> None:
    """Shows the unique values and corresponding frequencies.

    Args:
        series:
            a Pandas series, either categorical or with integers.
        sort_by_frequency:
            if True then sort by frequency desc; otherwise sort by index (either numerically ascending if
            series is numeric, or alphabetically if non-ordered categoric, or by category if ordered categoric
        figure_size:
            tuple containing `(width, height)` of plot. The default height is defined by
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


# pylint: disable=too-many-arguments
def plot_correlation_heatmap(dataframe: pd.DataFrame,
                             threshold: Optional[float] = None,
                             title: Optional[str] = None,
                             figure_size: tuple = STANDARD_WIDTH_HEIGHT,
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
            tuple containing `(width, height)` of plot. The default height is defined by
            `STANDARD_HEIGHT`, and the default width is `STANDARD_HEIGHT / GOLDEN_RATIO`
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

        features = features[meets_threshold]
        correlations = correlations.loc[features, features]

    if title is None:
        title = ''

    # remove duplicates (i.e. mirror image in second half of heatmap
    mask = np.zeros_like(correlations)
    mask[np.triu_indices_from(mask)] = True

    with sns.axes_style("white"):
        figure, axis = plt.subplots()
        figure.set_size_inches(figure_size[0], figure_size[1])
        sns.heatmap(correlations,
                    mask=mask,
                    annot=True,
                    fmt='.{}f'.format(round_by),
                    cmap=sns.diverging_palette(220, 10, as_cmap=True),
                    vmin=-1,
                    vmax=1,
                    square=True,
                    ax=axis,
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


# pylint: disable=too-many-locals
def plot_dodged_barchart(dataframe: pd.DataFrame, outer_column, inner_column,
                         figure_size: Tuple[int, int] = STANDARD_WIDTH_HEIGHT,
                         missing_value_replacement: str = '<Missing>'):
    """First attempt at dodged barchart.. It needs some work.

    args:
        dataframe:
            pandas data.frame
        outer_column:
            column name associated with outer bars
        inner_column:
            column name associated with inner bars
        figure_size:
            tuple containing `(width, height)` of plot. The default height is defined by
            `STANDARD_HEIGHT`, and the default width is `STANDARD_HEIGHT / GOLDEN_RATIO`
        missing_value_replacement
    """
    dataframe = dataframe[[outer_column, inner_column]].copy()

    # Replace all Boolean values with string because boolean causes issues
    dataframe[outer_column] = hpandas.replace_all_bools_with_strings(dataframe[outer_column])
    dataframe[inner_column] = hpandas.replace_all_bools_with_strings(dataframe[inner_column])

    dataframe[outer_column] = hpandas.fill_na(series=dataframe[outer_column],
                                              missing_value_replacement=missing_value_replacement)

    dataframe[inner_column] = hpandas.fill_na(series=dataframe[inner_column],
                                              missing_value_replacement=missing_value_replacement)

    outer_labels = dataframe[outer_column].unique().tolist()
    if hpandas.is_series_categorical(dataframe[outer_column]):
        categories = list(dataframe[outer_column].cat.categories)
        outer_labels = [x for x in categories if x in outer_labels]

    inner_labels = dataframe[inner_column].unique().tolist()
    if hpandas.is_series_categorical(dataframe[inner_column]):
        categories = list(dataframe[inner_column].cat.categories)
        inner_labels = [x for x in categories if x in inner_labels]

    grouped_data = dataframe.groupby([outer_column, inner_column]).size()

    outer_totals = [grouped_data[index].sum() for index in outer_labels]
    group_locations = np.arange(len(outer_labels))  # the x locations for the groups

    bar_midpoints = group_locations  # + (width / len(inner_labels))

    fig, axis = plt.subplots()
    fig.set_size_inches(figure_size[0], figure_size[1])

    outer_bar_width = 0.9
    inner_bar_width = outer_bar_width / len(inner_labels)

    ax_totals = axis.bar(x=bar_midpoints,
                         height=outer_totals,
                         width=outer_bar_width,
                         color='black', alpha=0.15)

    # from matplotlib.pyplot import cm
    # colors = cm.rainbow(np.linspace(0, 1, 10)[::-1])  # noqa
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k'] * 100

    if len(inner_labels) > len(colors):
        raise NotImplementedError(
            'Need to update implementation to use >' + str(len(colors)) + ' colors :(')

    ax_list = []
    for index in range(len(inner_labels)):
        # the 'if' is because it's not guaranteed that every class of the target variable will be found
        # in each category of 'outer_column', especially if the category is very small in size.
        counts = [grouped_data[x, inner_labels[index]]
                  if inner_labels[index] in grouped_data.loc[x].index else 0
                  for x in outer_labels]

        # `starting_offset` is the amount we have to subtract from our inner bars so the start of the
        # inner/outer are aligned
        starting_offset = (outer_bar_width / 2) - (inner_bar_width / 2)
        # (inner_bar_width * index) is the offset for each of the inner bars
        # i.e. it's the distance we have to add so that we aren't overlapping inner bars
        ax_list.append(
            axis.bar(x=group_locations - starting_offset + (inner_bar_width * index),
                     height=tuple(counts),
                     width=inner_bar_width,
                     color=colors[index]))

    # add some text for labels, title and axes ticks
    axis.set_ylabel('Count')
    axis.set_xlabel(outer_column)
    axis.set_title(f'`{outer_column}` vs. `{inner_column}`')
    axis.set_xticks(group_locations)
    axis.set_xticklabels(labels=outer_labels, rotation=20, ha='right')

    axis.legend([ax[0] for ax in ax_list] + [ax_totals[0]], inner_labels + ['Total'],
                title=inner_column)

    plt.tight_layout()


def plot_histogram_with_categorical(dataframe: pd.DataFrame,
                                    numeric_column: str,
                                    categorical_column: str,
                                    missing_value_replacement: str = '<Missing>') -> None:
    """Plots a categorical histogram within numeric histogram.

    Args:
        dataframe:
            TBD
        numeric_column:
            TBD
        categorical_column:
            TBD
        missing_value_replacement:
            the value
    """
    cut_dataframe = pd.DataFrame(pd.cut(dataframe[numeric_column],
                                        bins=10,
                                        right=True,
                                        include_lowest=True))

    cut_dataframe[numeric_column] = hpandas.fill_na(cut_dataframe[numeric_column],
                                                    missing_value_replacement=missing_value_replacement)
    categories = [str(x) for x in cut_dataframe[numeric_column].cat.categories]
    cut_dataframe[numeric_column] = [str(x) for x in cut_dataframe[numeric_column]]
    cut_dataframe[numeric_column] = cut_dataframe[numeric_column].astype('category')
    # only keep the categories that appear in the data, otherwise we get an error when calling
    # reorder_categories with extra categories
    categories = [x for x in categories if x in cut_dataframe[numeric_column].cat.categories]
    cut_dataframe[numeric_column] = cut_dataframe[numeric_column].cat.reorder_categories(categories)

    cut_dataframe[categorical_column] = dataframe[categorical_column].copy()

    plot_dodged_barchart(dataframe=cut_dataframe, outer_column=numeric_column,
                         inner_column=categorical_column)
