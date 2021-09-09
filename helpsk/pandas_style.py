"""Collection of helpers methods that help to style tables in Jupyter Notebooks
"""
from typing import Union, Optional

import pandas as pd
import numpy as np

from pandas.io.formats.style import Styler
from pandas.api.types import is_list_like  # noqa
from pandas._typing import Axis  # noqa

def format(styler: Union[pd.DataFrame, "pandas.io.formats.style.Styler"],  # noqa
           round_to: int = 2,
           fill_missing_value: Optional[str] = '<Missing>',
           thousands: Optional[str] = ',') -> Styler:

    if isinstance(styler, pd.DataFrame):
        styler = styler.style



    return styler.format(precision=round_to,  # noqa
                         na_rep=fill_missing_value,  # noqa
                         thousands=thousands)  # noqa

# pylint: disable=too-many-arguments
def __bar_inverse(style_object, align: str, colors: list[str], width: float = 100, min_value: float = None,
                  max_value: float = None):
    """
    CODE MODIFIED FROM
        https://github.com/pandas-dev/pandas/blob/v1.3.2/pandas/io/formats/style.py#L2178-L2258

    Draw bar chart in dataframe cells.
    """
    # Get input value range.
    object_min = np.nanmin(style_object.to_numpy()) if min_value is None else min_value
    object_max = np.nanmax(style_object.to_numpy()) if max_value is None else max_value
    if align == "mid":
        object_min = min(0, object_min)  # noqa
        object_max = max(0, object_max)  # noqa
    elif align == "zero":
        # For "zero" mode, we want the range to be symmetrical around zero.
        object_max = max(abs(object_min), abs(object_max))
        object_min = -object_max
    # Transform to percent-range of linear-gradient
    normed = width * (style_object.to_numpy(dtype=float) - object_min) / (object_max - object_min + 1e-12)
    zero = -width * object_min / (object_max - object_min + 1e-12)

    def css_bar(start: float, end: float, color: str) -> str:
        """
        Generate CSS code to draw a bar from start to end.
        """
        css = "width: 10em; height: 80%;"  # noqa
        if end > start:
            css += "background: linear-gradient(90deg,"
            if start > 0:
                css += f" {color} {start:.1f}%, transparent {start:.1f}%, "
            min_e = min(end, width)
            css += f"transparent {min_e:.1f}%, {color} {min_e:.1f}%)"
        elif end == start == 0:
            css += "background: linear-gradient(90deg,"
            css += f" {color} {100:.1f}%, transparent {100:.1f}%, "
            css += f"transparent {100:.1f}%, {color} {100:.1f}%)"
        return css

    def css(row_item):
        if pd.isna(row_item):
            return ""

        # avoid deprecated indexing `colors[x > zero]`
        color = colors[1] if row_item > zero else colors[0]

        if align == "left":
            return css_bar(0, row_item, color)

        return css_bar(min(row_item, zero), max(row_item, zero), color)

    if style_object.ndim == 1:
        # print(css(normed[10]))
        return [css(x) for x in normed]

    return pd.DataFrame(
        [[css(x) for x in row] for row in normed],
        index=style_object.index,
        columns=style_object.columns,
    )


def bar_inverse(
        styler: Union[pd.DataFrame, "pandas.io.formats.style.Styler"],  # noqa
        subset: "Subset" = None,  # noqa
        axis: Axis = 0,
        color="#d65f5f",
        width: float = 100,
        align: str = "left",
        min_value: float = None,
        max_value: float = None,
    ) -> Styler:
    """
    CODE MODIFIED FROM
        https://github.com/pandas-dev/pandas/blob/v1.3.2/pandas/io/formats/style.py#L2178-L2258

    Draw (inverse) bar chart in the cell backgrounds.
    Parameters
    ----------
    styler: either a pandas DataFrame or object returned by pd.DataFrame.style
    subset : label, array-like, IndexSlice, optional
        A valid 2d input to `DataFrame.loc[<subset>]`, or, in the case of a 1d input
        or single key, to `DataFrame.loc[:, <subset>]` where the columns are
        prioritised, to limit ``data`` to *before* applying the function.
    axis : {0 or 'index', 1 or 'columns', None}, default 0
        Apply to each column (``axis=0`` or ``'index'``), to each row
        (``axis=1`` or ``'columns'``), or to the entire DataFrame at once
        with ``axis=None``.
    color : str or 2-tuple/list
        If a str is passed, the color is the same for both
        negative and positive numbers. If 2-tuple/list is used, the
        first element is the color_negative and the second is the
        color_positive (eg: ['#d65f5f', '#5fba7d']).
    width : float, default 100
        A number between 0 or 100. The largest value will cover `width`
        percent of the cell's width.
    align : {'left', 'zero',' mid'}, default 'left'
        How to align the bars with the cells.
        - 'left' : the min value starts at the left of the cell.
        - 'zero' : a value of zero is located at the center of the cell.
        - 'mid' : the center of the cell is at (max-min)/2, or
          if values are all negative (positive) the zero is aligned
          at the right (left) of the cell.
    min_value : float, optional
        Minimum bar value, defining the left hand limit
        of the bar drawing range, lower values are clipped to `min_value`.
        When None (default): the minimum value of the data will be used.
    max_value : float, optional
        Maximum bar value, defining the right hand limit
        of the bar drawing range, higher values are clipped to `max_value`.
        When None (default): the maximum value of the data will be used.
    Returns
    -------
    styler
    """
    if isinstance(styler, pd.DataFrame):
        styler = styler.style

    if align not in ("left", "zero", "mid"):
        raise ValueError("`align` must be one of {'left', 'zero',' mid'}")

    if not is_list_like(color):
        color = [color, color]
    elif len(color) == 1:
        color = [color[0], color[0]]
    elif len(color) > 2:
        raise ValueError(
            "`color` must be string or a list-like "
            "of length 2: [`color_neg`, `color_pos`] "
            "(eg: color=['#d65f5f', '#5fba7d'])"
        )

    if subset is None:
        subset = styler.data.select_dtypes(include=np.number).columns

    # noqa
    styler.apply(
        __bar_inverse,  # noqa
        subset=subset,
        axis=axis,
        align=align,  # noqa
        colors=color,
        width=width,  # noqa
        min_value=min_value,  # noqa
        max_value=max_value,  # noqa
    )

    return styler