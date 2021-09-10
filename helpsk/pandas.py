"""This module contains helper functions when working with pandas objects (e.g. DataFrames, Series)."""

import datetime
import math
from typing import List, Union, Optional, Callable

import numpy as np
import pandas as pd
from pandas.io.formats.style import Styler

from helpsk.exceptions import HelpskParamValueError
import helpsk.pandas_style as pstyle
from helpsk import color


def get_numeric_columns(dataframe: pd.DataFrame) -> List[str]:
    """Returns the column names from the dataframe that are numeric.

    Args:
        dataframe: a pandas dataframe

    Returns:
        list of column names that correspond to numeric types. NOTE: if a column contains all `np.nan` values,
        it will count as numeric and will be returned in the list.
    """
    return [column for column in dataframe.columns if pd.api.types.is_numeric_dtype(dataframe[column])]


def get_non_numeric_columns(dataframe: pd.DataFrame) -> List[str]:
    """Returns the column names from the dataframe that are not numeric.

    Returns:
        list of column names that correspond to numeric types. NOTE: if a column contains all `np.nan` values,
        it will count as numeric and will not be returned in the list.
    """
    return [column for column in dataframe.columns if not pd.api.types.is_numeric_dtype(dataframe[column])]


def is_series_date(series: pd.Series) -> bool:
    """Returns True if the series contains Dates or DateTimes

    Args:
        series: a pandas Series

    Returns:
        a list of columns names that are either Dates or DateTimes
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return True

    valid_index = series.first_valid_index()
    if valid_index:
        return isinstance(series.iloc[valid_index], datetime.date)

    return False


def reorder_categories(categorical: Union[pd.Series, pd.Categorical],
                       weights: Optional[Union[pd.Series, np.ndarray]] = None,
                       weight_function: Callable = np.median,
                       ascending=True,
                       ordered=False) -> pd.Categorical:
    """Returns copy of the `categorical` series, with categories reordered based on the number of occurrences
     of the category (if `weights` is set to None`), or (if `weights` is not None) to numeric values in
     `weights` based on the function `weight_function`.

    similar to `fct_reorder()` in R.

    Args:
        categorical: A collection of categorical values that will be turned into a Categorical object with
            ordered categories.
        weights:
            Numeric values used to reorder the categorical. Must be the same length as `categorical`. If
            `None`, then the categories will be reordered based on number of occurrences for each category.
        weight_function:
            Function that determines the order of the categories. The default is `np.median`, meaning that
            order of the categories in the returned Categorical object is determined by the median value (of
            the corresponding `values`) for each category.
        ascending:
            if True, categories are ordered in ascending order based on result of `weight_function`
        ordered:
            passed to `pd.Categorical` to indicate if returned object should be `ordered`.
    """
    if weights is None:
        ordered_categories = categorical.value_counts(ascending=ascending, dropna=True)
    else:
        if len(categorical) != len(weights):
            message = f'Length of `categorical` ({len(categorical)}) ' \
                      f'must match length of `values ({len(weights)})'
            raise HelpskParamValueError(message)

        if isinstance(weights, pd.Series):
            weights = weights.values

        weights = pd.Series(weights, index=categorical)
        # for each categoric value, calculate the associated value of the categoric, based on the func
        # e.g. if func is np.median, get the median value associated with categoric value
        ordered_categories = weights.groupby(level=0).agg(weight_function)\
            .fillna(0).sort_values(ascending=ascending)

    # check that the unique values in categorical are a subset of the categories that we are setting
    # (subset because for categorical series there might be a category set on the series (i.e.
    # series.cat.categories) that doesn't have any associated values in the series)
    assert set(categorical.dropna()).issubset(set(ordered_categories.index.dropna()))

    results = pd.Categorical(values=categorical,
                             categories=list(ordered_categories.index.values),
                             ordered=ordered)
    return results


# pylint: disable=too-many-arguments
def top_n_categories(categorical: Union[pd.Series, pd.Categorical],
                     top_n: int = 5,
                     other_category: str = 'Other',
                     weights: Optional[Union[pd.Series, np.ndarray]] = None,
                     weight_function: Callable = np.median,
                     ordered: bool = False) -> pd.Categorical:
    """Returns copy of `categorical` series, with the top `n` categories retained based on either the count of
    values (if `weight` is None) or based on the values in `weight` determined by `aggregate_function`, and
    all other categories converted to `other`

    similar to `fct_lump()` in R.

    Args:
        categorical: A collection of categorical values that will be turned into a Categorical object with
            ordered categories.
        top_n:
            the number of categories to retain. All other categories (i.e. the values in the series) will be
            replaced with the value contained in `other_category`.
            Therefore, a total of `top_n + 1` categories will be return.
        other_category:
            the value given to all other categories
        weights:
            Numeric values used to reorder the categorical. Must be the same length as `categorical`.
        weight_function:
            Function that determines the order of the categories. The default is `np.median`, meaning that
            order of the categories in the returned Categorical object is determined by the median value (of
            the corresponding `values`) for each category.
        ordered:
            passed to `pd.Categorical` to indicate if returned object should be `ordered`.
    """

    # if there are less than (or the same amount of) unique values than top_n; then we don't have to do
    # anything
    if len(categorical.unique()) <= top_n:
        return categorical.copy()

    if weights is None:
        top_categories = categorical.value_counts(ascending=False, dropna=True).head(top_n)
        top_categories = list(top_categories.index.values)

    else:
        if len(categorical) != len(weights):
            message = f'Length of `categorical` ({len(categorical)}) ' \
                      f'must match length of `values ({len(weights)})'
            raise HelpskParamValueError(message)

        if isinstance(weights, pd.Series):
            weights = weights.values

        weights = pd.Series(weights, index=categorical)
        # for each categoric value, calculate the associated value of the categoric, based on the func
        # e.g. if func is np.median, get the median value associated with categoric value
        top_categories = weights.groupby(level=0).agg(weight_function).fillna(0).\
            sort_values(ascending=False).head(top_n)
        top_categories = list(top_categories.index.values)

    # TODO: this needs to be clean up; Categorical is causing issues
    final_series = pd.Categorical(categorical)
    if other_category not in final_series.categories:
        final_series = final_series.add_categories(other_category)
    final_series = pd.Series(final_series)
    final_series[final_series.apply(lambda x: x not in top_categories).fillna(False)] = other_category

    if other_category not in top_categories:
        top_categories = top_categories + [other_category]

    final_series = pd.Categorical(final_series,
                                  categories=top_categories,
                                  ordered=ordered)
    return final_series


def convert_integer_series_to_categorical(series: pd.Series, mapping: dict,
                                          ordered: bool = False) -> pd.Series:
    """Converts a Series object from integers to Categorical for a given mapping of integers to strings.

    Args:
        series:
            a pandas series containing integers values
        mapping:
            dictionary containing the unique values of the integers in the current data as the key, and the
            categoric string as the value.

            The mapping dict is required to have each value in the series accounted for. However, the
            series does not have to have all the values in the mapping. (For example, converting a small
            sample of data that doesn't contain all possible values should not fail.)
        ordered:
            a boolean representing whether the resulting Categorical series will be ordered, or not.

    Returns:
        A pandas Categorical Series
    """
    # check that the keys in mapping contains all numbers found in the series
    unique_values = series.unique()
    unique_values = unique_values[~np.isnan(unique_values)]
    missing_keys = [x for x in unique_values if x not in mapping.keys()]
    if missing_keys:
        message = f'The following value(s) were found in `series` but not in `mapping` keys `{missing_keys}`'
        raise HelpskParamValueError(message)

    # Pandas expects the underlying values to be a sequence starting with 0 (i.e. to be `0, 1, 2, ...`),
    # which won't always be the case (e.g. the values in the Series might start with 1, or at any number,
    # or might not be in sequence.
    # We need to first map the actual values to the sequence pandas expects.
    # e.g. if the actual values being mapped are [1, 2, 3] then the actual_to_expected_mapping will be
    # {actual: expected, ...} i.e. {1: 0, 2: 1, 3: 2}
    actual_to_expected_mapping = dict(zip(mapping.keys(), np.arange(len(mapping))))
    converted_series = pd.Series(series).map(actual_to_expected_mapping).fillna(-1)
    converted_series = pd.Categorical.from_codes(converted_series.astype(int),
                                                 mapping.values(),
                                                 ordered=ordered)
    return pd.Series(converted_series)


def numeric_summary(dataframe: pd.DataFrame,
                    round_by: int = 2,
                    return_style: bool = False) -> Union[pd.DataFrame, Styler, None]:
    """Provides a summary of basic stats for the numeric columns of a DataFrame.

    Args:
        dataframe:
            a pandas dataframe
        round_by:
            the number of decimal places to round the results
        return_style:
            If True, returns a pd.DataFrame.style object. This can be used for displaying in Jupyter Notebook.
            If False, returns a pd.DataFrame

    Returns:
        Returns a pandas DataFrame with the following attributes (returned as columns) for each of the
        numeric columns in `dataframe` (which are returned as rows).

        `# of Non-Nulls`: The number of non-null values found for the given column.
        `# of Nulls`: The number of null values found for the given column.
        `% Nulls`: The percent of null values found for a given column.
        `# of Zeros`: The number of values that equal `0`, found for the given column.
        `% Zeros`: The percent of `0`s found.
        `Mean`: The `mean` of all the values for a given column.
        `St Dev.`: The `standard deviation` of all the values for a given column.
        `Coef of Var`: The `coefficient of variation (CV)`, is defined as the standard deviation divided by
            the mean, and describes the variability of the column's values relative to its mean.

            We can use this metric to compare the variation of two different variables (i.e. columns) that
            have different units or scales.
        `Skewness`: "unbiased skew"; utilizes `pandas` DataFrame underlying `.skew()` function
            https://pythontic.com/pandas/dataframe-computations/skew
        `Kurtosis`: "unbiased kurtosis ... using Fisher’s definition of kurtosis"; utilizes `pandas` DataFrame
            underlying `.skew()` function
        `Min`: minimum value found
        `10%`: the value found at the 10th percentile of data
        `25%`: the value found at the 25th percentile of data
        `50%`: the value found at the 50th percentile of data
        `75%`: the value found at the 75th percentile of data
        `90%`: the value found at the 90th percentile of data
        `Max`: maximum value found
    """
    # if there aren't any numeric columns and the target variable is not numeric, we don't have anything
    # to display, return None
    numeric_columns = get_numeric_columns(dataframe=dataframe)

    if not numeric_columns:
        return None

    # column, number of nulls in column, percent of nulls in column
    null_data = [(column,
                  dataframe[column].isnull().sum(),
                  round(dataframe[column].isnull().sum() / len(dataframe), round_by))
                 for column in numeric_columns]
    columns, num_nulls, perc_nulls = zip(*null_data)

    # column, number of 0's, percent of 0's
    zeros_data = [(sum(dataframe[column] == 0),
                   round(sum(dataframe[column] == 0) / len(dataframe), round_by))
                  for column in numeric_columns]
    num_zeros, perc_zeros = zip(*zeros_data)
    results = pd.DataFrame(
        {'# of Non-Nulls': [dataframe[x].count() for x in numeric_columns],
         '# of Nulls': num_nulls,
         '% Nulls': perc_nulls,
         '# of Zeros': num_zeros,
         '% Zeros': perc_zeros,
         'Mean': [round(dataframe[x].mean(), round_by) for x in numeric_columns],
         'St Dev.': [round(dataframe[x].std(), round_by) for x in numeric_columns],
         'Coef of Var': [round(dataframe[x].std() / dataframe[x].mean(), round_by)
                         if dataframe[x].mean() != 0 else np.nan
                         for x in numeric_columns],
         'Skewness': [round(dataframe[x].skew(), round_by) for x in numeric_columns],
         'Kurtosis': [round(dataframe[x].kurt(), round_by) for x in numeric_columns],
         'Min': [round(dataframe[x].min(), round_by) for x in numeric_columns],
         '10%': [round(dataframe[x].quantile(q=0.10), round_by) for x in numeric_columns],
         '25%': [round(dataframe[x].quantile(q=0.25), round_by) for x in numeric_columns],
         '50%': [round(dataframe[x].quantile(q=0.50), round_by) for x in numeric_columns],
         '75%': [round(dataframe[x].quantile(q=0.75), round_by) for x in numeric_columns],
         '90%': [round(dataframe[x].quantile(q=0.90), round_by) for x in numeric_columns],
         'Max': [round(dataframe[x].max(), round_by) for x in numeric_columns]},
        index=columns)

    if return_style:
        columns_to_format = [x for x in results.columns
                             if x not in ['# of Non-Nulls', '# of Nulls', '% Nulls', '# of Zeros', '% Zeros']]
        results = results.style. \
            format({
                '% Nulls': '{:,.1%}'.format,
                '% Zeros': '{:,.1%}'.format,
            }). \
            pipe(pstyle.format,
                 subset=['# of Non-Nulls', '# of Nulls', '# of Zeros'],
                 round_by=0).\
            pipe(pstyle.format, subset=columns_to_format, round_by=1). \
            highlight_between(left=0.00000001, right=math.inf, subset=['# of Nulls', '% Nulls', '# of Zeros',
                                                                       '% Zeros'],
                              color=color.WARNING). \
            bar(subset=['Coef of Var'], color=color.GRAY, vmin=0, vmax=1). \
            bar(subset=['Skewness'], color=color.GRAY,  align='mid', vmin=-2, vmax=2)

    return results


def non_numeric_summary(dataframe: pd.DataFrame, return_style: bool = False) -> Union[pd.DataFrame, None]:
    """Provides a summary of basic stats for the non-numeric columns of a DataFrame.

    Args:
        dataframe:
            a pandas dataframe
        return_style:
            If True, returns a pd.DataFrame.style object. This can be used for displaying in Jupyter Notebook.
            If False, returns a pd.DataFrame

    Returns:
        Returns a pandas DataFrame with the following attributes (returned as columns) for each of the
        non-numeric columns in `dataframe` (which are returned as rows).

        `# of Non-Nulls`: The number of non-null values found for the given column.
        `# of Nulls`: The number of null values found for the given column.
        `% Nulls`: The percent of null values found (i.e. `nulls / (count + nulls)`) for a given column.
        `Most Freq. Value`: The most frequent value found for a given column.
        `# of Unique`: The number of unique values found for a given column.
        `% Unique`: The percent of unique values found (i.e. `unique` divided by the total number of values
            (null or non-null) for a given column.
    """
    # if there aren't any non-numeric columns and the target variable is numeric, we don't have anything
    # to display, return None

    non_numeric_columns = get_non_numeric_columns(dataframe=dataframe)
    if not non_numeric_columns:
        return None

    # column, number of nulls in column, percent of nulls in column
    null_data = [(column,
                  dataframe[column].isnull().sum(),
                  round(dataframe[column].isnull().sum() / len(dataframe), 3))
                 for column in non_numeric_columns]
    columns, num_nulls, perc_nulls = zip(*null_data)
    results = pd.DataFrame({'# of Non-Nulls': [dataframe[x].count() for x in non_numeric_columns],
                            '# of Nulls': num_nulls,
                            '% Nulls': perc_nulls,
                            'Most Freq. Value': [dataframe[x].value_counts().index[0]
                                                 for x in non_numeric_columns],
                            '# of Unique': [len(dataframe[x].dropna().unique()) for x in non_numeric_columns],
                            '% Unique': [round(len(dataframe[x].dropna().unique()) / dataframe[x].count(),
                                               3) for x in non_numeric_columns]},
                           index=columns)

    if return_style:
        results = results.style.format({
                '% Unique': '{:,.1%}'.format
            }). \
            pipe(pstyle.format,
                 subset=['# of Non-Nulls', '# of Nulls'],
                 round_by=0). \
            pipe(pstyle.format, subset=['# of Unique'], round_by=1). \
            highlight_between(left=0.00000001, right=math.inf, subset=['# of Nulls', '% Nulls'],
                              color=color.WARNING). \
            bar(subset=['% Unique'], color=color.GRAY, vmin=0, vmax=1)

    return results


def print_dataframe(dataframe: pd.DataFrame) -> None:
    """Helper function that prints a DataFrame without replacing columns with `...` when there are many
    columns.

    Args:
        dataframe: pandas DataFrame to print
    """
    with pd.option_context('display.max_columns', None), \
            pd.option_context('display.width', 20000):
        print(dataframe)


def value_frequency(series: pd.Series, sort_by_frequency=True) -> pd.DataFrame:
    """Shows the unique values and corresponding frequencies.

    Args:
        series:
            a Pandas series, either categorical or with integers.
        sort_by_frequency:
            if True then sort by frequency desc; otherwise sort by index (either numerically ascending if
            series is numeric, or alphabetically if non-ordered categoric, or by category if ordered categoric

    Returns:
        DataFrame with each of the values in `series` as a single row, and `Frequency` and `Percent` columns
        matching the number of occurrences and percent representation in the series.
    """
    results = pd.DataFrame({'Frequency': series.value_counts(normalize=False, dropna=False),
                            'Percent': series.value_counts(normalize=True, dropna=False)})

    if not sort_by_frequency:
        if series.dtype.name == 'category':
            if series.cat.ordered:
                results.sort_index(inplace=True)
            else:
                # pandas sorts non-ordered categories by whatever order the category shows up, not
                # alphabetically
                # so change the categories to be alphabetical and sort

                values = results.index.values.dropna().unique().tolist()  # noqa
                values.sort()
                results['temp'] = results.index
                results['temp'] = results['temp'].cat.reorder_categories(values, ordered=True)
                results = results.sort_values(['temp']).drop(columns='temp')
        else:
            results.sort_index(inplace=True)
    return results
