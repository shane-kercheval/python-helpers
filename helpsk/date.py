import datetime
from dateutil import relativedelta


def ymd(date_string: str) -> datetime.date:
    return datetime.datetime.strptime(date_string, "%Y-%m-%d").date()


def format_as_month(date: datetime.date) -> str:
    return date.strftime("%Y-%b")


def format_as_quarter(dates: datetime.date, first_fiscal_month: int = 1) -> str:
    pass


def floor_month(date: datetime.date) -> datetime.date:
    return date.replace(day=1)


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


def date_to_string(date: datetime.date) -> str:
    return date.strftime("%Y-%m-%d")
