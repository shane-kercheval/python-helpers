from typing import List, Union, Callable, Type
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


def assert_not_none_nan(values: Union[List, np.ndarray, pd.Series, pd.DataFrame]) -> bool:
    """
    Raises an AssertionError if any item in `values` are `None`, `np.Nan`, or if the length of `values` is 
    `0`.
    For numeric types only.

    Parameters
    ----------
    values : list, np.ndarray, pd.Series, pd.DataFrame
        A collection of values to check.

    Returns
    -------
    None
    """
    if any_none_nan(values):
        raise AssertionError('None/NaN Values Found')


def any_missing(values: Union[List, pd.Series, pd.DataFrame]) -> bool:
    """
    Returns `True` if any item in `values` are `None`, `np.Nan`, an empty string (i.e. '') or if the length of
    `values` is `0`.

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


def assert_not_any_missing(values: Union[List, pd.Series, pd.DataFrame]) -> bool:
    """
    Raises an AssertionError if any item in `values` are `None`, `np.Nan`, an empty string (i.e. '') or if the
    length of `values` is `0`.

    Parameters
    ----------
    values : list, pd.Series, pd.DataFrame
        A collection of values to check.

    Returns
    -------
    bool - True if any item in `values` are None/np.NaN/''
    """
    if any_missing(values):
        raise AssertionError('Missing Values Found')


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


def assert_not_duplicated(values: Union[List, np.ndarray, pd.Series]) -> bool:
    """
    Raises an AssertionError if any items in `values` are duplicated.

    Parameters
    ----------
    values : list, np.ndarray, pd.Series
        A collection of values to check.

    Returns
    -------
    """
    if any_duplicated(values):
        raise AssertionError('Duplicate Values Found')


def assert_all(values: Union[List, np.ndarray, pd.Series, pd.DataFrame]):
    """
    Raises an `AssertionError` unless all items in `values` are `True`

    Parameters
    ----------
    values : list, np.ndarray, pd.Series, pd.DataFrame
        A collection of values to check.

    Returns
    -------
    None
    """
    if isinstance(values, pd.Series):
        if not values.all():
            raise AssertionError('Not All True')
    elif isinstance(values, pd.DataFrame):
        if not values.all().all():
            raise AssertionError('Not All True')
    else:
        if not all(values):
            raise AssertionError('Not All True')


def assert_not_any(values: Union[List, np.ndarray, pd.Series, pd.DataFrame]):
    """
    Raises an `AssertionError` if any items in `values` are `True`

    Parameters
    ----------
    values : list, np.ndarray, pd.Series, pd.DataFrame
        A collection of values to check.

    Returns
    -------
    None
    """
    if isinstance(values, pd.Series):
        if values.any():
            raise AssertionError('Found True')
    elif isinstance(values, pd.DataFrame):
        if values.any().any():
            raise AssertionError('Found True')
    else:
        if any(values):
            raise AssertionError('Found True')


def assert_identical(values):
    """
    Raises Exception if xyz is not identical
    """    
    raise NotImplementedError()


def raises_exception(function: Callable, exception_type: Type= None) -> bool:
    """Returns True if `function` raises an Exception; returns False if `function` runs without raising an Exception.

    Keyword arguments:
    function -- a function
    """
    try:
        function()
        return False
    except Exception as e:  # noqa
        if exception_type:
            return isinstance(e, exception_type)
        else:
            return True

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
