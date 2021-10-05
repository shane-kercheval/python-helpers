import unittest

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from helpsk.utility import suppress_warnings
from tests.helpers import check_plot, get_test_path


def plot_colors(colors, title, sort_colors=True):
    """
    Code from:
        https://matplotlib.org/stable/gallery/color/named_colors.html
    """
    cell_width = 212
    cell_height = 22
    swatch_width = 48
    margin = 12
    top_margin = 40

    # Sort colors by hue, saturation, value and name.
    if sort_colors is True:
        by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(color))),
                         name)
                        for name, color in colors.items())
        names = [name for hsv, name in by_hsv]
    else:
        names = list(colors)

    nrows = 7
    # n = len(names)
    # ncols = n // nrows + int(n % nrows > 0)
    # ncols = 4 - empty_cols
    # nrows = n // ncols + int(n % ncols > 0)

    width = cell_width * 4 + 2 * margin
    height = cell_height * nrows + margin + top_margin
    dpi = 72

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.subplots_adjust(margin/width, margin/height,
                        (width-margin)/width, (height-top_margin)/height)
    ax.set_xlim(0, cell_width * 4)
    ax.set_ylim(cell_height * (nrows-0.5), -cell_height/2.)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_axis_off()
    ax.set_title(title, fontsize=24, loc="left", pad=10)

    for i, name in enumerate(names):
        row = i % nrows
        col = i // nrows
        y = row * cell_height

        swatch_start_x = cell_width * col
        text_pos_x = cell_width * col + swatch_width + 7

        ax.text(text_pos_x, y, name, fontsize=14,
                horizontalalignment='left',
                verticalalignment='center')

        with suppress_warnings():
            ax.add_patch(
                Rectangle(xy=(swatch_start_x, y-9), width=swatch_width,
                          height=18, color=colors[name], edgecolor='0.7')
            )

    return fig


class TestColors(unittest.TestCase):

    def test_plot_colors(self):
        from helpsk.color import Colors

        color_names = [e.name for e in Colors]
        color_values = [e.value for e in Colors]

        check_plot(file_name=get_test_path() + '/test_files/color/colors.png',
                   plot_function=lambda: plot_colors(dict(zip(color_names, color_values)),
                                                     title="Colors Enum", sort_colors=False))
