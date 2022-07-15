"""Contains a collection of date related helper functions.
"""
from typing import Union
from enum import unique, Enum, auto
import datetime

import pandas as pd
from dateutil import relativedelta
import numpy as np
from helpsk.validation import any_none_nan


@unique
class Granularity(Enum):
    """
    Valid values for date granularity
    """
    DAY = auto()
    MONTH = auto()
    QUARTER = auto()


def floor(value: Union[datetime.datetime, datetime.date, pd.Series],
          granularity: Granularity = Granularity.DAY,
          fiscal_start: int = 1) -> Union[datetime.date, pd.Series]:

    """"Rounds" the datetime value down (i.e. floor) to the the nearest granularity.

    For example, if `date` is `2021-03-20` and granularity is `Granularity.QUARTER` then `floor()` will return
    `2021-01-01`.

    If the fiscal year starts on November (`fiscal_start is `11`; i.e. the quarter is November,
    December, January) and the date is `2022-01-28` and granularity is `Granularity.QUARTER` then `floor()`
    will return `2021-11-01`.

    Args:
        value:
            a datetime or pd.Series of datetimes

        granularity:
            the granularity to round down to (i.e. DAY, MONTH, QUARTER)

        fiscal_start:
            Only applicable for `Granularity.QUARTER`. The value should be the month index (e.g. Jan is 1,
            Feb, 2, etc.) that corresponds to the first month of the fiscal year.

    Returns:
        date - the date rounded down to the nearest granularity
    """
    if isinstance(value, pd.Series):
        return pd.Series(
            [floor(x, granularity=granularity, fiscal_start=fiscal_start) for x in value],
            name=value.name,
            index=value.index
        )

    if any_none_nan(value):
        return value

    if granularity == Granularity.DAY:
        if isinstance(value, datetime.datetime):
            return value.date()

        return value

    if granularity == Granularity.MONTH:
        if isinstance(value, datetime.datetime):
            return value.replace(day=1).date()

        return value.replace(day=1)

    if granularity == Granularity.QUARTER:
        relative_start_index = ((fiscal_start - 1) % 3) + 1
        current_month_index = ((value.month - 1) % 3) + 1
        months_to_subtract = (((relative_start_index * -1) + current_month_index) % 3)
        months_to_subtract = relativedelta.relativedelta(months=months_to_subtract)
        return floor(value, granularity=Granularity.MONTH) - months_to_subtract

    raise ValueError("Unknown Granularity type")


def fiscal_quarter(value: Union[datetime.datetime, datetime.date],
                   include_year: bool = False,
                   fiscal_start: int = 1) -> float:
    """Returns the fiscal quarter (or year and quarter numeric value) for a given date.

    For example:
        from dateutil.parser import parse

        date.fiscal_quarter(parse('2021-01-15')) == 1
        date.fiscal_quarter(parse('2021-01-15'), include_year=True) == 2021.1


        date.fiscal_quarter(parse('2021-01-15'), fiscal_start=2) == 4
        date.fiscal_quarter(parse('2021-01-15'), include_year=True, fiscal_start=2) == 2021.4
        date.fiscal_quarter(parse('2020-11-15'), include_year=True, fiscal_start=2) == 2021.4

    Logic converted from R's Lubridate package:
        https://github.com/tidyverse/lubridate/blob/master/R/accessors-quarter.r

    Args:
        value:
            a date or datetime

        include_year:
            logical indicating whether or not to include the year
            if `True` then returns a float in the form of year.quarter.
            For example, Q4 of 2021 would be returned as `2021.4`

        fiscal_start:
            numeric indicating the starting month of a fiscal year.
            A value of 1 indicates standard quarters (i.e. starting in January).

    Returns:
        date - the date rounded down to the nearest granularity
    """

    assert 1 <= fiscal_start <= 12

    fiscal_start = (fiscal_start - 1) % 12
    shifted = np.arange(fiscal_start, 11 + fiscal_start + 1) % 12 + 1
    quarters = np.repeat([1, 2, 3, 4], 3)
    match_index = np.where(value.month == shifted)
    assert len(match_index) == 1
    match_index = int(match_index[0])
    quarter = quarters[match_index]

    if include_year:
        if fiscal_start == 0:
            return value.year + (quarter / 10)

        next_year_months = np.arange(fiscal_start + 1, 12 + 1)
        return value.year + (value.month in next_year_months) + (quarter / 10)

    return quarter


def to_string(value: Union[datetime.datetime, datetime.date],
              granularity: Granularity = Granularity.DAY,
              fiscal_start: int = 1) -> str:
    """Converts the date to a string.

    Examples:

        ```
        from dateutil.parser import parse

        to_string(value=parse('2021-01-15'), granularity=Granularity.DAY) == "2021-01-15"
        to_string(value=parse('2021-01-15'), granularity=Granularity.Month) == "2021-Jan"
        to_string(value=parse('2021-01-15'), granularity=Granularity.QUARTER) == "2021-Q1"
        to_string(value=parse('2021-01-15'),
                  granularity=Granularity.QUARTER,
                  fiscal_start=2) == "2021-FQ4"
        ```

    If the fiscal year starts on November (`fiscal_start is `11`; i.e. the quarter is November,
    December, January) and the date is `2022-01-28` and granularity is `Granularity.QUARTER` then `floor()`
    will return `2021-11-01`.

    Args:
        value:
            a datetime

        granularity:
            the granularity to round down to (i.e. DAY, MONTH, QUARTER)

        fiscal_start:
            Only applicable for `Granularity.QUARTER`. The value should be the month index (e.g. Jan is 1,
            Feb, 2, etc.) that corresponds to the first month of the fiscal year.

            If fiscal_start start is any other value than `1` quarters will be abbreviated with `F` to denote
            non-standard / fiscal quarters. For example, "2021-FQ4" is the 4th fiscal quarter of 2021.

    Returns:
        date - the date rounded down to the nearest granularity
    """
    if granularity == Granularity.DAY:
        return value.strftime("%Y-%m-%d")

    if granularity == Granularity.MONTH:
        return value.strftime("%Y-%b")

    if granularity == Granularity.QUARTER:
        if fiscal_start == 1:
            return str(fiscal_quarter(value,
                                      include_year=True,
                                      fiscal_start=fiscal_start)).replace('.', '-Q')

        return str(fiscal_quarter(value,
                                  include_year=True,
                                  fiscal_start=fiscal_start)).replace('.', '-FQ')

    raise TypeError('Unrecognized Granularity')
