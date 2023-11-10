"""A collection of functions that assist in validation/comparison of data and conditions."""

from __future__ import annotations
from collections.abc import Sized, Iterable, Callable

import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_categorical_dtype

from helpsk.exceptions import HelpskParamTypeError, HelpskParamValueError
from helpsk.utility import suppress_warnings


def is_none_nan(value: object) -> bool:
    """Returns True if the value is None or various NaN/NA types."""
    if value is None \
            or value is pd.NA \
            or value is pd.NaT \
            or (isinstance(value, (np.float64, float, int)) and np.isnan(value)):
        return True

    return False


def any_none_nan(  # noqa: PLR0911
        values: list | np.ndarray | pd.Series | pd.DataFrame | object) -> bool:
    """
    Can be used with a single value or a collection of values. Returns `True` if any item in
    `values` are `None`, `np.Nan`, `pd.NA`, `pd.NaT` or if the length of `values` is `0`.

    Args:
        values:
            A collection of values to check.

    Returns:
        bool - True if any item in `values` are None/np.NaN
    """
    if is_none_nan(values):
        return True

    if isinstance(values, Sized) and not isinstance(values, str) and len(values) == 0:
        return True

    if isinstance(values, pd.Series):
        return values.isna().any() or values.isna().any()

    if isinstance(values, pd.DataFrame):
        return values.isna().any().any() or values.isna().any().any()

    if isinstance(values, Iterable) and not isinstance(values, str):
        if len(values) == 0:
            return True

        return any(is_none_nan(x) for x in values)

    try:
        if not isinstance(values, str) and None in values:
            return True
    except Exception:
        pass

    try:
        if np.isnan(values).any():
            return True
    except TypeError:
        return False

    return False


def any_missing(values: list | pd.Series | pd.DataFrame | object) -> bool:
    """
    Same as `any_none_nan` but checks for empty strings.

    Args:
        values:
            A collection of values to check.

    Returns:
        bool - True if any item in `values` are None/np.NaN/''
    """
    if any_none_nan(values):
        return True

    if isinstance(values, pd.Series):
        return values.isin(['']).any()

    if isinstance(values, pd.DataFrame):
        return values.isin(['']).any().any()

    if isinstance(values, str) and values.strip() == '':
        return True

    if isinstance(values, Iterable) and '' in values:
        return True

    return False


def any_duplicated(values: Iterable) -> bool:
    """
    Returns `True` if any items in `values` are duplicated.

    Args:
        values: list, np.ndarray, pd.Series
            A collection of values to check.

    Returns:
        bool
    """
    return len(values) != len(set(values))


def iterables_are_equal(iterable_a: Iterable, iterable_b: Iterable) -> bool:
    """
    Compares the equality of the values of two iterables.

    This function will generally give the same result as list equality (e.g.
    `[x, y, z] == [x, y, z]`). However, in some strange scenarios, `==` will return `False` where
    it doesn't seem like it should

    For example:

    ```
    temp = pd.DataFrame({'col_a': [np.nan, 1.0]})
    temp.col_a.tolist() == [np.nan, 1.0]  # returns False. Why??
    iterables_are_equal(temp.col_a.tolist(), [np.nan, 1])  # returns True
    [np.nan, 1.0] == [np.nan, 1.0]  # returns True


    Also, when comparing a series with an ordered Categorical when the values are the same,
    pd.Series.equals() will return False if the categories have different order. But we only care
    if the values are the same, so this function will return True.
    ```

    Args:
        iterable_a:
            an iterable to equate to iterable_b
        iterable_b:
            an iterable to equate to iterable_a

    Returns:
        True if iterable_a is equal to iterable_b
    """
    with suppress_warnings():
        # if either list-like structure is categorical, then we need to convert both to unordered
        # categorical
        if len(iterable_a) != len(iterable_b):
            return False
        if len(iterable_a) == 0 and len(iterable_b) == 0:
            return True

        if is_categorical_dtype(iterable_a) or is_categorical_dtype(iterable_b):
            iterable_a = pd.Categorical(iterable_a, ordered=False)
            iterable_b = pd.Categorical(iterable_b, ordered=False)
        else:
            iterable_a = pd.Series(iterable_a)
            iterable_b = pd.Series(iterable_b)

        return iterable_a.equals(iterable_b)


def dataframes_match(dataframes: list[pd.DataFrame],
                     float_tolerance: int = 6,
                     ignore_indexes: bool = True,
                     ignore_column_names: bool = True) -> bool:
    """
    Because floating point numbers are difficult to accurate represent, when comparing multiple
    DataFrames, this function first rounds any numeric columns to the number of decimal points
    indicated `float_tolerance`.

    Args:
        dataframes:
            Two or more dataframes to compare against each other and test for equality

        float_tolerance:
            numeric columns will be rounded to the number of digits to the right of the decimal
            specified by this parameter.

        ignore_indexes:
            if True, the indexes of each DataFrame will be ignored for considering equality

        ignore_column_names:
            if True, the column names of each DataFrame will be ignored for considering equality

    Returns:
        Returns True if the dataframes match based on the conditions explained above, otherwise
        returns False
    """
    if not isinstance(dataframes, list):
        raise HelpskParamTypeError("Expected list of pd.DataFrame's.")

    if not len(dataframes) >= 2:
        raise HelpskParamValueError("Expected 2 or more pd.DataFrame's in list.")

    first_dataframe = dataframes[0].round(float_tolerance)

    def first_dataframe_equals_other(other_dataframe: pd.DataFrame) -> bool:
        if first_dataframe.shape != other_dataframe.shape:
            return False

        if ignore_indexes or ignore_column_names:
            # if either of these are True, then we are going to change the index and/or columns,
            # but python is pass-by-reference so we don't want to change the original DataFrame
            # object.
            other_dataframe = other_dataframe.copy()

        if ignore_indexes:
            other_dataframe.index = first_dataframe.index

        if ignore_column_names:
            other_dataframe.columns = first_dataframe.columns

        return first_dataframe.equals(other_dataframe.round(float_tolerance))

    # compare the first dataframe to the rest of the dataframes, after rounding each to the
    # tolerance, and performing other modifications
    # check if all results are True
    return all(first_dataframe_equals_other(x) for x in dataframes[1:])


def is_close(value_a: float, value_b: float, tolerance: float = 0.000001) -> bool:
    """
    Tests whether or not value_a and value_b are "close" (i.e. within the `tolerance` after
    subtracting).

    Args:
        value_a:
            numeric value to test
        value_b:
            numeric value to test
        tolerance:
            the maximum difference (absolute value) allowed between value_a and value_b
    Returns:
          True if values are within specified tolerance
    """
    return abs(value_a - value_b) <= tolerance


def raises_exception(function: Callable, exception_type: type | None = None) -> bool:
    """
    Returns True if `function` raises an Exception; returns False if `function` runs without
    raising an Exception.

    Args:
        function:
            the function which does or does not raise an exception.
        exception_type:
            if `exception_type` is provided, `raises_exception` returns true only if the `function`
            argument raises an Exception **and** the exception has type of `exception_type`.
    """
    try:
        function()
        return False
    except Exception as exception:
        if exception_type:
            return isinstance(exception, exception_type)

        return True
