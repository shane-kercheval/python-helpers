"""A collection of functions that assist in validation/comparison of data and conditions.
"""
from collections.abc import Sized
from typing import List, Union, Callable, Type, Iterable

import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_categorical

from helpsk.exceptions import *  # pylint: disable=wildcard-import,unused-wildcard-import
from helpsk.utility import suppress_warnings


def any_none_nan(values: Union[List, np.ndarray, pd.Series, pd.DataFrame, object]) -> bool:
    """Can be used with a single value or a collection of values. Returns `True` if any item in `values` are
    `None`, `np.Nan`, `pd.NA`, `pd.NaT` or if the length of `values` is `0`.

    Args:
        values:
            A collection of values to check.

    Returns:
        bool - True if any item in `values` are None/np.NaN
    """
    # pylint: disable=too-many-return-statements
    if values is None or values is np.NaN or values is pd.NA or values is pd.NaT:  # pylint: disable=nan-comparison
        return True

    if isinstance(values, Sized) and not isinstance(values, str) and len(values) == 0:
        return True

    if isinstance(values, pd.Series):
        return values.isnull().any() or values.isna().any()

    if isinstance(values, pd.DataFrame):
        return values.isnull().any().any() or values.isna().any().any()

    if isinstance(values, Iterable) and not isinstance(values, str):
        if len(values) == 0:
            return True

        return any((any_none_nan(x) for x in values))

    try:
        if not isinstance(values, str) and None in values:
            return True
    except Exception:  # pylint: disable=broad-except # noqa
        pass

    try:
        if np.isnan(values).any():
            return True
    except TypeError:
        return False

    return False


def assert_not_none_nan(values: Union[List, np.ndarray, pd.Series, pd.DataFrame, object]) -> None:
    """Raises an HelpskAssertionError if any item in `values` are `None`, `np.Nan`, or if the length of
    `values` is `0`.

    For numeric types only.

    Args:
        values:
            A collection of values to check.
    """
    assert_false(any_none_nan(values), message='None/NaN Values Found')


def any_missing(values: Union[List, pd.Series, pd.DataFrame, object]) -> bool:
    """Same as `any_none_nan` but checks for empty strings

    Args:
        values:
            A collection of values to check.

    Returns:
        bool - True if any item in `values` are None/np.NaN/''
    """
    if any_none_nan(values):
        return True

    if isinstance(values, pd.Series):
        return values.isin(['']).any()  # noqa

    if isinstance(values, pd.DataFrame):
        return values.isin(['']).any().any()  # noqa

    if isinstance(values, str) and values.strip() == '':
        return True

    if isinstance(values, Iterable) and '' in values:
        return True

    return False


def assert_not_any_missing(values: Union[List, pd.Series, pd.DataFrame, object]) -> None:
    """Raises an HelpskAssertionError if any item in `values` are `None`, `np.Nan`, an empty string (i.e. '')
    or if the length of `values` is `0`.

    Args:
        values:
            A collection of values to check.
    """
    assert_false(any_missing(values), message='Missing Values Found')


def any_duplicated(values: Union[List, np.ndarray, pd.Series]) -> bool:
    """Returns `True` if any items in `values` are duplicated.

    Args:
        values: list, np.ndarray, pd.Series
            A collection of values to check.

    Returns:
        bool
    """
    return len(values) != len(set(values))


def assert_not_duplicated(values: Union[List, np.ndarray, pd.Series]) -> None:
    """Raises an HelpskAssertionError if any items in `values` are duplicated.

    Args:
        values: list, np.ndarray, pd.Series
            A collection of values to check.
    """
    assert_false(any_duplicated(values), message='Duplicate Values Found')


def assert_all(values: Union[List, np.ndarray, pd.Series, pd.DataFrame]) -> None:
    """Raises an `HelpskAssertionError` unless all items in `values` are `True`

    Args:
        values:
            A collection of values to check.
    """
    if isinstance(values, pd.Series):
        if not values.all():  # noqa
            raise HelpskAssertionError('Not All True')
    elif isinstance(values, pd.DataFrame):
        if not values.all().all():  # noqa
            raise HelpskAssertionError('Not All True')
    else:
        if not all(values):
            raise HelpskAssertionError('Not All True')


def assert_not_any(values: Union[List, np.ndarray, pd.Series, pd.DataFrame]) -> None:
    """Raises an `HelpskAssertionError` if any items in `values` are `True`

    Args:
        values:
            A collection of values to check.
    """
    if isinstance(values, pd.Series):
        assert_false(values.any(), message='Found True')  # noqa

    elif isinstance(values, pd.DataFrame):
        assert_false(values.any().any(), message='Found True')  # noqa

    else:
        assert_false(any(values), message='Found True')


def assert_true(condition: bool, message: str = 'Condition Not True') -> None:
    """Raises an HelpskAssertionError if `condition` is not True

    Args:
        condition:
            Something that evaluates to True/False
        message:
            Message passed to the HelpskAssertionError
    """
    if not isinstance(condition, (bool, np.bool_)):
        raise HelpskParamTypeError('condition should be boolean')

    if not condition:
        raise HelpskAssertionError(message)


def assert_false(condition: bool, message: str = 'Condition True') -> None:
    """Raises an HelpskAssertionError if `condition` is not False

    Args:
        condition: bool
            Something that evaluates to True/False
        message:
            Message passed to the HelpskAssertionError
    """
    if not isinstance(condition, (bool, np.bool_)):
        raise HelpskParamTypeError('condition should be boolean')

    if condition:
        raise HelpskAssertionError(message)


def iterables_are_equal(iterable_a: Iterable, iterable_b: Iterable) -> bool:
    """Compares the equality of the values of two iterables.

    This function will generally give the same result as list equality (e.g. `[x, y, z] == [x, y, z]`)
    However, in some strange scenarios, `==` will return `False` where it doesn't seem like it should

    For example:

    ```
    temp = pd.DataFrame({'col_a': [np.nan, 1.0]})
    temp.col_a.tolist() == [np.nan, 1.0]  # returns False. Why??
    iterables_are_equal(temp.col_a.tolist(), [np.nan, 1])  # returns True
    [np.nan, 1.0] == [np.nan, 1.0]  # returns True


    Also, when comparing a series with an ordered Categorical when the values are the same,
    pd.Series.equals() will return False if the categories have different order. But we only care if the
    values are the same, so this function will return True.
    ```

    Args:
        iterable_a:
            an iterable to equate to iterable_b
        iterable_b:
            an iterable to equate to iterable_a

    Returns:
        True if iterable_a is equal to iterable_b
    """
    # seems to be confusion and inconsistencies across stack overflow on how to properly check for category
    # so this might be overkill but not exactly sure
    # def is_categorical(series):
    #     if isinstance(series, (pd.Categorical, pd.CategoricalDtype)):
    #         return True
    #     if isinstance(series, pd.Series):
    #         return series.dtype.name == 'category'
    #     return False

    with suppress_warnings():
        # if either list-like structure is categorical, then we need to convert both to unordered categorical
        if is_categorical(iterable_a) or is_categorical(iterable_b):
            iterable_a = pd.Categorical(iterable_a, ordered=False)
            iterable_b = pd.Categorical(iterable_b, ordered=False)
        else:
            iterable_a = pd.Series(iterable_a)
            iterable_b = pd.Series(iterable_b)

        return iterable_a.equals(iterable_b)


def dataframes_match(dataframes: List[pd.DataFrame],
                     float_tolerance: int = 6,
                     ignore_indexes: bool = True,
                     ignore_column_names: bool = True) -> bool:
    """
    Because floating point numbers are difficult to accurate represent, when comparing multiple DataFrames,
    this function first rounds any numeric columns to the number of decimal points indicated
    `float_tolerance`.

    Args:
        dataframes:
            Two or more dataframes to compare against each other and test for equality

        float_tolerance:
            numeric columns will be rounded to the number of digits to the right of the decimal specified by
            this parameter.

        ignore_indexes:
            if True, the indexes of each DataFrame will be ignored for considering equality

        ignore_column_names:
            if True, the column names of each DataFrame will be ignored for considering equality

    Returns:
        Returns True if the dataframes match based on the conditions explained above, otherwise returns False
    """
    if not isinstance(dataframes, list):
        raise HelpskParamTypeError("Expected list of pd.DataFrame's.")

    if not len(dataframes) >= 2:
        raise HelpskParamValueError("Expected 2 or more pd.DataFrame's in list.")

    first_dataframe = dataframes[0].round(float_tolerance)

    def first_dataframe_equals_other(other_dataframe):
        if first_dataframe.shape != other_dataframe.shape:
            return False

        if ignore_indexes or ignore_column_names:
            # if either of these are True, then we are going to change the index and/or columns, but
            # python is pass-by-reference so we don't want to change the original DataFrame object.
            other_dataframe = other_dataframe.copy()

        if ignore_indexes:
            other_dataframe.index = first_dataframe.index

        if ignore_column_names:
            other_dataframe.columns = first_dataframe.columns

        return first_dataframe.equals(other_dataframe.round(float_tolerance))

    # compare the first dataframe to the rest of the dataframes, after rounding each to the tolerance, and
    # performing other modifications
    # check if all results are True
    return all(first_dataframe_equals_other(x) for x in dataframes[1:])


def assert_dataframes_match(dataframes: List[pd.DataFrame],
                            float_tolerance: int = 6,
                            ignore_indexes: bool = True,
                            ignore_column_names: bool = True,
                            message: str = 'Dataframes do not match') -> None:
    """
    Raises an assertion error if dataframes don't match.

    Args:
        dataframes:
            Two or more dataframes to compare against each other and test for equality

        float_tolerance:
            numeric columns will be rounded to the number of digits to the right of the decimal specified by
            this parameter.

        ignore_indexes:
            if True, the indexes of each DataFrame will be ignored for considering equality

        ignore_column_names:
            if True, the column names of each DataFrame will be ignored for considering equality

        message:
            message to pass to HelpskAssertionError
    """
    if not dataframes_match(dataframes=dataframes,
                            float_tolerance=float_tolerance,
                            ignore_indexes=ignore_indexes,
                            ignore_column_names=ignore_column_names):
        raise HelpskAssertionError(message)


def is_close(value_a: float, value_b: float, tolerance: float = 0.000001) -> bool:
    """Tests whether or not value_a and value_b are "close" (i.e. within the `tolerance` after subtracting)

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


def assert_is_close(value_a: float, value_b: float, tolerance: float = 0.000001):
    """Raises an assert error if value_a and value_b are not "close" (see documentation of `is_close()`
    function).

    Args:
          value_a:
              numeric value to test
          value_b:
              numeric value to test
          tolerance:
              number of digits to round to
    """
    if not is_close(value_a=value_a, value_b=value_b, tolerance=tolerance):
        raise HelpskAssertionError(f"`{value_a}` and `{value_b}` are not within a tolerance of `{tolerance}`")


def raises_exception(function: Callable, exception_type: Type = None) -> bool:
    """Returns True if `function` raises an Exception; returns False if `function` runs without raising an
    Exception.
    Args:
        function:
            the function which does or does not raise an exception.
        exception_type:
            if `exception_type` is provided, `raises_exception` returns true only if the `function` argument
            raises an Exception **and** the exception has type of `exception_type`.
    """
    try:
        function()
        return False
    except Exception as exception:  # pylint: disable=broad-except
        if exception_type:
            return isinstance(exception, exception_type)

        return True
