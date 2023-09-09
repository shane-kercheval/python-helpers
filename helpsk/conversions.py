"""Functions to help with calculating conversation/retension rates."""

import datetime
import pandas as pd


def _percent_converted(seconds_to_convert: pd.Series, period_in_seconds: int) -> float:
    return ((seconds_to_convert > 0) & (seconds_to_convert <= period_in_seconds)).mean()


def cohorted_conversion_rate(
        df: pd.DataFrame,
        base_timestamp: str,
        conversion_timestamp: str,
        cohort: str,
        intervals: list[tuple[int, str]],
        groups: str | None = None,
        current_datetime: str | None = None) -> pd.DataFrame:
    """
    Calculate the cohorted conversion rate for a given base timestamp and conversion timestamp

    Update the function so that it takes a `current_datetime` value that represents the current date and time. For a cohort where any record in the cohort hasn't had enough time elapse between the base_timestamp and the time to convert based on the intervals, that cohort should have a nan value. So for example, if the cohort for `2020-01-01` has a record where the base_timestamp is `2020-01-25 00:00:00` and current_datime is `2020-02-02 00:00:00` and there intervals [(1, 'days'), (10, 'days')] then the first interval is valid because everyone in the cohort has had enough time to convert (i.e. everyone has had at least 1 day in the cohort), but not everyone has had 5 days (e.g. the timestamp for `2020-01-25 00:00:00` has only been in the cohort for 8 days which is not enough time to know if it will convert in 10 days. So the conversion rate is valid for the first interval but not the second.
    
    Args:
        base_timestamp:
            The column name for the base timestamp to use for the calculation.
        conversion_timestamp:
            The column name for the conversion timestamp to use for the calculation.
        cohort:
            The column name for the cohort (e.g. day, week, month) to use for the calculation.
        intervals:
            A list of intervals to calculate the conversion rate for in the form of (value, unit)
            which represents the number of units to calculate the conversion rate for.
        groups:
            The column name for the groups/segments to calculate the conversion rate for.
        current_datetime:
            The current datetime to use for calculating whether an interval is valid.
    """
    def to_seconds(value, unit):
        unit_to_seconds = {
            'seconds': 1,
            'minutes': 60,
            'hours': 3600,
            'days': 86400,
        }
        return value * unit_to_seconds[unit]

    if not current_datetime:
        current_datetime = datetime.datetime.utcnow()
    current_datetime = pd.to_datetime(current_datetime)

    group_by = [cohort]
    if groups:
        group_by.append(groups)

    def f(x):
        d = dict()
        d['# of records'] = len(x)
        # the number of seconds from the last base timestamp in the cohort to the current datetime
        # if any record in the cohort has not had enough time elapse between the base_timestamp and
        # the time to convert based on the intervals, that cohort should have a nan value
        seconds_from_max_base = (current_datetime - x[base_timestamp].max()).total_seconds()
        for value, unit in intervals:
            interval_seconds = to_seconds(value, unit)
            if seconds_from_max_base >= interval_seconds:
                d[f'{value} {unit}'] = _percent_converted(
                    x['seconds_to_conversion'],
                    interval_seconds
                )
            else:
                d[f'{value} {unit}'] = None

        return pd.Series(d)

    conversions = (
        df
        .assign(seconds_to_conversion=lambda x: (x[conversion_timestamp] - x[base_timestamp]).dt.total_seconds())
        .groupby(group_by)
        .apply(f)
        .reset_index()
        .sort_values(group_by, ascending=True)
    )
    return conversions
