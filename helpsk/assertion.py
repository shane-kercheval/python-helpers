from typing import List, Union, Callable
import numpy as np
import pandas as pd


def any_none_nan(values: Union[List, np.ndarray, pd.Series, pd.DataFrame]) -> bool:
    """
    Returns `True` if any item in `values` are `None`, `np.Nan`, or if the length of `values` is `0`.
    For numeric types only.

    Parameters
    ----------
    values : list, np.ndarray, pd.Series, pd.DataFrame
        A collection of values to check.

    Returns
    -------
    bool - True if any item in `values` are None/np.NaN
    """
    if values is None or values is np.NaN:
        return True

    if len(values) == 0:
        return True

    if isinstance(values, pd.Series):
        return values.isnull().any() or values.isna().any()

    if isinstance(values, pd.DataFrame):
        return values.isnull().any().any() or values.isna().any().any()

    if None in values:
        return True

    try:
        if np.isnan(values).any():
            return True
    except TypeError:
        return False

    return False


def any_missing(values: Union[List, pd.Series, pd.DataFrame]) -> bool:
    """
    Returns `True` if any item in `values` are `None`, `np.Nan`, an empty string (i.e. '') or if the length of `values` is `0`.

    Parameters
    ----------
    values : list, pd.Series, pd.DataFrame
        A collection of values to check.

    Returns
    -------
    bool - True if any item in `values` are None/np.NaN/''
    """
    if any_none_nan(values):
        return True

    if isinstance(values, pd.Series):
        return values.isin(['']).any()

    if isinstance(values, pd.DataFrame):
        return values.isin(['']).any().any()

    if '' in values:
        return True

    return False


def any_duplicated(values: Union[List, np.ndarray, pd.Series]) -> bool:
    """
    Returns `True` if any items in `values` are duplicated.

    Parameters
    ----------
    values : list, np.ndarray, pd.Series
        A collection of values to check.

    Returns
    -------
    bool
    """
    return len(values) != len(set(values))





def assert_not_any(values):
    """
    Raises Exception if any values are true
    """
    raise NotImplementedError()


def assert_identical(values):
    """
    Raises Exception if xyz is not identical
    """    
    raise NotImplementedError()


def assert_none_missing(values, empty_string_as_missing: bool = True):
    """
    Raises Exception if any items in `values` are missing.

    Keyword arguments:
    empty_string_as_missing -- if True, treats empty string as missing value
    """
    raise NotImplementedError()


def assert_none_duplicated(values, ignore_missing_values: bool = True):
    """
    Raises Exception if any items in `values` are duplicated.

    Keyword arguments:
    ignore_missing_values -- if True, removes missing values before checking if duplicated
    """

    # if ignore_missing_values is False throw exception if more than one are missing?
    raise NotImplementedError()

#  @staticmethod
# def assert_dataframes_equal(data_frame1: pd.DataFrame,
#                             data_frame2: pd.DataFrame,
#                             check_column_types: bool = True):
#     def is_number(s):
#         try:
#             float(s)
#             return True
#         except ValueError:
#             return False
#
#     # check that the types of the columns are all the same
#     if check_column_types:
#         assert all([x == y for x, y in zip(data_frame1.dtypes.values, data_frame2.dtypes.values)])
#     assert all(data_frame1.columns.values == data_frame2.columns.values)
#     assert all(data_frame1.index.values == data_frame2.index.values)
#     numeric_col, cat_cols = OOLearningHelpers.get_columns_by_type(data_dtypes=data_frame1.dtypes)
#
#     for col in numeric_col:
#         # check if the values are close, or if they are both NaN
#         assert all([is_close(x, y) or (math.isnan(x) and math.isnan(y))
#                     for x, y in zip(data_frame1[col].values, data_frame2[col].values)])
#
#     for col in cat_cols:
#         # if the two strings aren't equal, but also aren't 'nan', it will cause a problem because
#         # isnan will try to convert the string to a number, but it will fail with TypeError, so have to
#         # ensure both values are a number before we check that they are nan.
#         assert all([x == y or (is_number(x) and is_number(y) and math.isnan(x) and math.isnan(y))
#                     for x, y in zip(data_frame1[col].values, data_frame2[col].values)])
#
#     return True


def raises_exception(function: Callable) -> bool:
    """Returns True if `function` raises an Exception; returns False if `function` runs without raising an Exception.

    Keyword arguments:
    function -- a function
    """
    try:
        function()
        return False
    except:  # noqa
        return True
