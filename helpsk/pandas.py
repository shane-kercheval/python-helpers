"""This module contains helper functions when working with pandas objects (e.g. DataFrames, Series)."""

import datetime
from typing import List, Union

import numpy as np
import pandas as pd


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


def numeric_summary(dataframe: pd.DataFrame) -> Union[pd.DataFrame, None]:
    """
    Returns the following attributes (as columns) for each of the numeric features (as rows).

    Returns:
        a pandas Dataframe with the following row indexes:

        `count`: The number of non-null values found for the given feature.
        `nulls`: The number of null values found for the given feature.
        `perc_nulls`: The percent of null values found (i.e. `nulls / (count + nulls)`) for a given feature.
        `num_zeros`: The number of values that equal `0`, found for the given feature.
        `perc_zeros`: The percent of `0`s found (i.e. `num_zeros / number of values in series`) for a given
            feature. Note: `number of values in series` is `count` + `nulls`, so this shows the percent of
            zeros found considering all of the values in the series, not just the non-null values.
        `mean`: The `mean` of all the values for a given feature.
        `st_dev`: The `standard deviation` of all the values for a given feature.
        `Coef of var`: The `coefficient of variation (CV)`, is defined as the standard deviation divided by
            the mean, and describes the variability of the feature's values relative to its mean.

            We can use this metric to compare the variation of two different variables (i.e. features) that
            have different units or scales.
        `skewness`: "unbiased skew"; utilizes `pandas` DataFrame underlying `.skew()` function
        `kurtosis`: "unbiased kurtosis ... using Fisher’s definition of kurtosis"; utilizes `pandas` DataFrame
            underlying `.skew()` function
        `min`: minimum value found
        `10%`: the value found at the 10th percentile of data
        `25%`: the value found at the 25th percentile of data
        `50%`: the value found at the 50th percentile of data
        `75%`: the value found at the 75th percentile of data
        `90%`: the value found at the 90th percentile of data
        `max`: maximum value found
    """
    # if there aren't any numeric features and the target variable is not numeric, we don't have anything
    # to display, return None
    numeric_columns = get_numeric_columns(dataframe=dataframe)

    if not numeric_columns:
        return None

    # column, number of nulls in column, percent of nulls in column
    null_data = [(column,
                  dataframe[column].isnull().sum(),
                  round(dataframe[column].isnull().sum() / len(dataframe), 3))
                 for column in numeric_columns]
    columns, num_nulls, perc_nulls = zip(*null_data)

    # column, number of 0's, percent of 0's
    zeros_data = [(column, sum(dataframe[column] == 0),
                   round(sum(dataframe[column] == 0) / len(dataframe), 3))
                  for column in numeric_columns]
    columns, num_zeros, perc_zeros = zip(*zeros_data)
    results = pd.DataFrame(
        {'# of Non-Nulls': [dataframe[x].count() for x in numeric_columns],
         '# of Nulls': num_nulls,
         '% Nulls': perc_nulls,
         '# of Zeros': num_zeros,
         '% Zeros': perc_zeros,
         'Mean': [round(dataframe[x].mean(), 3) for x in numeric_columns],
         'St Dev.': [round(dataframe[x].std(), 3) for x in numeric_columns],
         'Coef of Var': [round(dataframe[x].std() / dataframe[x].mean(), 3)
                         if dataframe[x].mean() != 0 else np.nan
                         for x in numeric_columns],
         'Skewness': [round(dataframe[x].skew(), 3) for x in numeric_columns],
         'Kurtosis': [round(dataframe[x].kurt(), 3) for x in numeric_columns],
         'Min': [round(dataframe[x].min(), 3) for x in numeric_columns],
         '10%': [round(dataframe[x].quantile(q=0.10), 3) for x in numeric_columns],
         '25%': [round(dataframe[x].quantile(q=0.25), 3) for x in numeric_columns],
         '50%': [round(dataframe[x].quantile(q=0.50), 3) for x in numeric_columns],
         '75%': [round(dataframe[x].quantile(q=0.75), 3) for x in numeric_columns],
         '90%': [round(dataframe[x].quantile(q=0.90), 3) for x in numeric_columns],
         'Max': [round(dataframe[x].max(), 3) for x in numeric_columns]},
        index=columns)
    return results


def non_numeric_summary(dataframe: pd.DataFrame) -> Union[pd.DataFrame, None]:
    """
    Returns the following attributes (as columns) for each of the categoric features (as rows).

    Returns:
        `count`: The number of non-null values found for the given feature.
        `nulls`: The number of null values found for the given feature.
        `perc_nulls`: The percent of null values found (i.e. `nulls / (count + nulls)`) for a given feature.
        `top`: The most frequent value found for a given feature.
        `unique`: The number of unique values found for a given feature.
        `perc_unique`: The percent of unique values found (i.e. `unique` divided by the total number of values
            (null or non-null) for a given feature.
    """
    # if there aren't any non-numeric features and the target variable is numeric, we don't have anything
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
                            '% Null': perc_nulls,
                            'Most Freq. Value': [dataframe[x].value_counts().index[0] for x in non_numeric_columns],
                            '# of Unique': [len(dataframe[x].dropna().unique()) for x in non_numeric_columns],
                            '% Unique': [round(len(dataframe[x].dropna().unique()) / dataframe[x].count(),
                                                  3) for x in non_numeric_columns]},
                           index=columns)
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
