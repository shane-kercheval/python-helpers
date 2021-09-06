"""This module contains helper functions for plotting."""
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd

import helpsk.pandas as hpandas

STANDARD_PLOT_HEIGHT = 10
GOLDEN_RATIO = 1.61803398875
STANDARD_PLOT_WIDTH = STANDARD_PLOT_HEIGHT / GOLDEN_RATIO
STANDARD_PLOT_HEIGHT_WIDTH = (STANDARD_PLOT_HEIGHT, STANDARD_PLOT_WIDTH)


def plot_value_frequency(series: pd.Series, sort_by_frequency: bool = True,
                         figure_size: Tuple[int, int] = STANDARD_PLOT_HEIGHT_WIDTH,
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
            `STANDARD_PLOT_HEIGHT`, and the default width is `STANDARD_PLOT_HEIGHT / GOLDEN_RATIO`
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
