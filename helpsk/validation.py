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
        return values.isnull().any() or values.isna().any()  # noqa

    if isinstance(values, pd.DataFrame):
        return values.isnull().any().any() or values.isna().any().any()  # noqa

    if None in values:
        return True

    try:
        if np.isnan(values).any():
            return True
    except TypeError:
        return False

    return False


def assert_not_none_nan(values: Union[List, np.ndarray, pd.Series, pd.DataFrame]) -> None:
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
    assert_false(any_none_nan(values), message='None/NaN Values Found')


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
        return values.isin(['']).any()  # noqa

    if isinstance(values, pd.DataFrame):
        return values.isin(['']).any().any()  # noqa

    if '' in values:
        return True

    return False


def assert_not_any_missing(values: Union[List, pd.Series, pd.DataFrame]) -> None:
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
    assert_false(any_missing(values), message='Missing Values Found')


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


def assert_not_duplicated(values: Union[List, np.ndarray, pd.Series]) -> None:
    """
    Raises an AssertionError if any items in `values` are duplicated.

    Parameters
    ----------
    values : list, np.ndarray, pd.Series
        A collection of values to check.

    Returns
    -------
    """
    assert_false(any_duplicated(values), message='Duplicate Values Found')


def assert_all(values: Union[List, np.ndarray, pd.Series, pd.DataFrame]) -> None:
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
        if not values.all():  # noqa
            raise AssertionError('Not All True')
    elif isinstance(values, pd.DataFrame):
        if not values.all().all():  # noqa
            raise AssertionError('Not All True')
    else:
        if not all(values):
            raise AssertionError('Not All True')


def assert_not_any(values: Union[List, np.ndarray, pd.Series, pd.DataFrame]) -> None:
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
        assert_false(values.any(), message='Found True')  # noqa

    elif isinstance(values, pd.DataFrame):
        assert_false(values.any().any(), message='Found True')  # noqa

    else:
        assert_false(any(values), message='Found True')


def assert_true(condition: bool, message: str = 'Condition Not True') -> None:
    if not (isinstance(condition, bool) or isinstance(condition, np.bool_)):
        raise TypeError('condition should be boolean')

    if not condition:
        raise AssertionError(message)


def assert_false(condition: bool, message: str = 'Condition True') -> None:
    if not (isinstance(condition, bool) or isinstance(condition, np.bool_)):
        raise TypeError('condition should be boolean')

    if condition:
        raise AssertionError(message)


def raises_exception(function: Callable, exception_type: Type = None) -> bool:
    """
    Returns True if `function` raises an Exception; returns False if `function` runs without raising an
    Exception.

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
