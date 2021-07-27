import datetime
from dateutil import relativedelta
from typing import Union

from enum import unique, Enum, auto


@unique
class Granularity(Enum):
    DAY = auto()
    MONTH = auto()
    QUARTER = auto()


def ymd(date_string: str) -> datetime.date:
    return datetime.datetime.strptime(date_string, "%Y-%m-%d").date()


def format_as_month(date: datetime.date) -> str:
    return date.strftime("%Y-%b")


def format_as_quarter(dates: datetime.date, first_fiscal_month: int = 1) -> str:
    pass


def floor(value: Union[datetime.datetime, datetime.date],
          granularity: Granularity = Granularity.DAY,
          first_fiscal_month: int = 1) -> datetime.date:

    """
    "Rounds" the datetime value down (i.e. floor) to the the nearest granularity.

    For example, if `date` is `2021-03-20` then `floor_quarter` will return `2021-01-01`.

    If the fiscal year starts on November (`first_fiscal_month is `11`; i.e. the quarter is November,
    December, January) and the date is `2022-01-28` then `floor_quarter` will return `2021-11-01`.

    Parameters
    ----------
    value : datetime.datetime
        a datetime

    first_fiscal_month: int
        Only applicable for `Granularity.QUARTER`. The value should be the month index (e.g. Jan is 1, Feb, 2,
        etc.) that corresponds to the first month of the fiscal year.

    Returns
    -------
    date - the date rounded down to the naerest granularity
    """
    if granularity == Granularity.DAY:
        if isinstance(value, datetime.datetime):
            return value.date()
        else :
            return value

    if granularity == Granularity.MONTH:
        if isinstance(value, datetime.datetime):
            return value.replace(day=1).date()
        else :
            return value.replace(day=1)

    if granularity == Granularity.QUARTER:
        return ''

    raise ValueError("Unknown Granularity type")





def floor_quarter(date: datetime.date, first_fiscal_month: int = 1) -> datetime.date:
    """
    "Rounds" the date down (i.e. floor) to the start of the current quarter. 

    For example, if `date` is `2021-03-20` then `floor_quarter` will return `2021-01-01`.

    If the fiscal year starts on November (`first_fiscal_month is `11`; i.e. the quarter is November,
    December, January) and the date is `2022-01-28` then `floor_quarter` will return `2021-11-01`.

    Parameters
    ----------
    date : datetime.date
        a date

    first_fiscal_month: int
        the month index (e.g. Jan is 1, Feb, 2, etc.) that corresponds to the first month of the
        fiscal year.

    Returns
    -------
    date - the start of the current quarter
    """
    relative_start_index = ((first_fiscal_month - 1) % 3) + 1
    current_month_index = ((date.month - 1) % 3) + 1
    months_to_subtract = (((relative_start_index * -1) + current_month_index) % 3)

    return floor_month(date) - relativedelta.relativedelta(months=months_to_subtract)


def to_string(date: datetime.date) -> str:
    return date.strftime("%Y-%m-%d")
