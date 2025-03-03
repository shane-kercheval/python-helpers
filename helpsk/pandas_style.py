"""Collection of helpers methods that help to style tables in Jupyter Notebooks."""

from __future__ import annotations
from typing import TypeVar
from html import escape

import pandas as pd
import numpy as np

from pandas.io.formats.style import Styler
from pandas.api.types import is_list_like
from pandas._typing import Axis
from seaborn import color_palette

from helpsk import color


import helpsk.pandas as hpandas
from helpsk.validation import any_none_nan


Subset = TypeVar('Subset')


def add_bar(
        df_style: Styler,
        column_name: str,
        min_value: float | None = None,
        max_value: float | None = None,
        round_by: int = 0,
        color: str = color.GRAY) -> Styler:
    """
    Add bars to a particular column with the ability to specify a min/max value.

    Args:
        df_style: style object e.g. df.style
        column_name: name of the column to apply bars
        min_value: min value of bars
        max_value: max value of bars
        round_by: number of decimals to round by
        color: color of bars
    """
    return df_style.\
        bar(subset=[column_name], color=color, vmin=min_value, vmax=max_value).\
        format(subset=[column_name], precision=round_by)


def add_background_gradient(
        df_style: Styler,
        column_name: str,
        min_value: float | None = None,
        max_value: float | None = None,
        round_by: int = 0,
        color_cmap: str | None = None) -> Styler:
    """
    Add gradient to a particular column with the ability to specify a min/max value.

    Args:
        df_style: style object e.g. df.style
        column_name: name of the column to apply gradient
        min_value: min value of gradient
        max_value: max value of gradient
        round_by: number of decimals to round by
        color_cmap: cmap of gradient (sns.color_palette("vlag", as_cmap=True) is default)
    """
    if not color_cmap:
        color_cmap = color_palette("vlag", as_cmap=True)

    return df_style.\
        background_gradient(
            subset=[column_name],
            cmap=color_cmap,
            vmin=min_value,
            vmax=max_value,
        ).\
        format(subset=[column_name], precision=round_by)


def format(styler: pd.DataFrame | Styler,  # noqa: A001
           subset: list[str] | None = None,
           round_by: int = 2,
           fill_missing_value: str | None = '<NA>',
           missing_color: str | None = color.WARNING,
           thousands: str | None = ',',
           hide_index: bool = False) -> Styler:
    """
    Applies basic formatting to pandas Dataframe.

    Args:
        styler:
            either pd.Dataframe or pd.Dataframe.style
        subset:
            A valid 2d input to DataFrame.loc[<subset>], or, in the case of a 1d input or single
            key, to DataFrame.loc[:, <subset>] where the columns are prioritised, to limit data to
            before applying the function.
        round_by:
            number of digits to round numeric columns to
        fill_missing_value:
            the value to replace missing data (e.g. NaN)
        missing_color:
            The background color for cells that have missing values.
        thousands:
            the separator used for thousands e.g. `'` will result in `10,000` while ` ` will
            result in `10 000`.
        hide_index:
            Hide the index of the dataframe.

    Returns:
        styler
    """
    if isinstance(styler, pd.DataFrame):
        styler = styler.style

    styler = styler.highlight_null(missing_color)

    if hide_index:
        styler = styler.hide(axis='index')

    return styler.format(
        subset=subset,
        precision=round_by,
        na_rep=escape(fill_missing_value),
        thousands=thousands,
    )


def background_color(
        styler: pd.DataFrame | Styler,
        palette: str = 'Blues',
        **kwargs: dict) -> Styler:
    """
    Applies a background color to pandas Dataframe.

    Args:
        styler:
            either pd.Dataframe or pd.Dataframe.style
        palette:
            name of the palette (value passed into seaborn `color_palette()`
        kwargs:
            additional arguments that will be passed to the pandas `.background_gradient()`
            function.

    Returns:
        styler
    """
    if isinstance(styler, pd.DataFrame):
        styler = styler.style

    # color_map = sns.light_palette("green", as_cmap=True)
    # color_map = sns.color_palette("dark:salmon_r", as_cmap=True)
    # color_map = sns.color_palette(['red', 'blue', 'green'], as_cmap=True)
    # color_map = sns.color_palette("light:#5A9", as_cmap=True)
    color_map = color_palette(palette, as_cmap=True)
    return styler.background_gradient(cmap=color_map, **kwargs)


def __bar_inverse(
        style_object: Styler,
        align: str,
        colors: list[str],
        width: float = 100,
        min_value: float | None = None,
        max_value: float | None = None) -> pd.DataFrame:
    """
    CODE MODIFIED FROM FOLLOWING.

        https://github.com/pandas-dev/pandas/blob/v1.3.2/pandas/io/formats/style.py#L2178-L2258

    Draw bar chart in dataframe cells.
    """
    # Get input value range.
    object_min = np.nanmin(style_object.to_numpy()) if min_value is None else min_value
    object_max = np.nanmax(style_object.to_numpy()) if max_value is None else max_value
    if align == "mid":
        object_min = min(0, object_min)
        object_max = max(0, object_max)
    elif align == "zero":
        # For "zero" mode, we want the range to be symmetrical around zero.
        object_max = max(abs(object_min), abs(object_max))
        object_min = -object_max
    # Transform to percent-range of linear-gradient
    normed = width \
        * (style_object.to_numpy(dtype=float) - object_min) \
        / (object_max - object_min + 1e-12)
    zero = -width * object_min / (object_max - object_min + 1e-12)

    def css_bar(start: float, end: float, color: str) -> str:
        """Generate CSS code to draw a bar from start to end."""
        css = "width: 10em; height: 80%;"
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

    def css(row_item: float) -> str:
        if pd.isna(row_item):
            return ""

        # avoid deprecated indexing `colors[x > zero]`
        color_value = colors[1] if row_item > zero else colors[0]
        if align == "left":
            return css_bar(0, row_item, color_value)

        return css_bar(min(row_item, zero), max(row_item, zero), color_value)

    if style_object.ndim == 1:
        # print(css(normed[10]))
        return [css(x) for x in normed]

    return pd.DataFrame(
        [[css(x) for x in row] for row in normed],
        index=style_object.index,
        columns=style_object.columns,
    )


def bar_inverse(  # noqa: D417
        styler: pd.DataFrame | Styler,
        subset: Subset = None,
        axis: Axis = 0,
        color: str = "#d65f5f",
        width: float = 100,
        align: str = "left",
        min_value: float | None = None,
        max_value: float | None = None) -> Styler:
    """
    CODE MODIFIED FROM FOLLOWING.

        https://github.com/pandas-dev/pandas/blob/v1.3.2/pandas/io/formats/style.py#L2178-L2258

    Draw (inverse) bar chart in the cell backgrounds.
    Parameters
    ----------
    styler: either a pandas DataFrame or object returned by pd.DataFrame.style
    subset: label, array-like, IndexSlice, optional
        A valid 2d input to `DataFrame.loc[<subset>]`, or, in the case of a 1d input
        or single key, to `DataFrame.loc[:, <subset>]` where the columns are
        prioritised, to limit ``data`` to *before* applying the function.
    axis: {0 or 'index', 1 or 'columns', None}, default 0
        Apply to each column (``axis=0`` or ``'index'``), to each row
        (``axis=1`` or ``'columns'``), or to the entire DataFrame at once
        with ``axis=None``.
    color: str or 2-tuple/list
        If a str is passed, the color is the same for both
        negative and positive numbers. If 2-tuple/list is used, the
        first element is the color_negative and the second is the
        color_positive (eg: ['#d65f5f', '#5fba7d']).
    width: float, default 100
        A number between 0 or 100. The largest value will cover `width`
        percent of the cell's width.
    align: {'left', 'zero',' mid'}, default 'left'
        How to align the bars with the cells.
        - 'left': the min value starts at the left of the cell.
        - 'zero': a value of zero is located at the center of the cell.
        - 'mid': the center of the cell is at (max-min)/2, or
          if values are all negative (positive) the zero is aligned
          at the right (left) of the cell.
    min_value: float, optional
        Minimum bar value, defining the left hand limit
        of the bar drawing range, lower values are clipped to `min_value`.
        When None (default): the minimum value of the data will be used.
    max_value: float, optional
        Maximum bar value, defining the right hand limit
        of the bar drawing range, higher values are clipped to `max_value`.
        When None (default): the maximum value of the data will be used.
    Returns
    -------
    styler
    """  # noqa: D410, D411
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
            "(eg: color=['#d65f5f', '#5fba7d'])",
        )

    if subset is None:
        subset = styler.data.select_dtypes(include=np.number).columns

    styler.apply(
        __bar_inverse,
        subset=subset,
        axis=axis,
        align=align,
        colors=color,
        width=width,
        min_value=min_value,
        max_value=max_value,
    )

    return styler


def html_escape_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    HTML `escapes` all string and categorical columns and indexes in the `dataframe`.

    This can be used when displaying pd.DataFrames in Jupyter notebook using `.style`;
    e.g. `<XXX>` is displayed as blank because it is not encoded.

    Args:
        dataframe: pd.DataFrame

    Returns:
        a copy of the `dataframe` with string values replaced after being html encoded via
        `html.escape()`
    """
    def __escape(value: str | None) -> str | None:
        if not any_none_nan([value]) and isinstance(value, str):
            return escape(value)
        return value

    dataframe = dataframe.copy()
    columns_to_escape = hpandas.get_string_columns(dataframe) + \
        hpandas.get_categorical_columns(dataframe)
    for column in columns_to_escape:
        dataframe[column] = dataframe[column].apply(__escape)

    if isinstance(dataframe.index, pd.MultiIndex):
        index_tuples = [tuple([__escape(x) for x in index]) for index in dataframe.index]
        dataframe.index = pd.MultiIndex.from_tuples(index_tuples)
    else:
        dataframe.index = [__escape(x) for x in dataframe.index.to_numpy()]

    if isinstance(dataframe.columns, pd.MultiIndex):
        index_tuples = [tuple([__escape(x) for x in columns]) for columns in dataframe.columns]
        dataframe.columns = pd.MultiIndex.from_tuples(index_tuples)
    else:
        dataframe.columns = [__escape(x) for x in dataframe.columns.to_numpy()]

    return dataframe
