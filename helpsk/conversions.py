"""Functions to help with calculating conversation/retension rates."""

import pandas as pd


def _percent_converted(seconds_to_convert: pd.Series, period_in_seconds: dict) -> float:
    return ((seconds_to_convert > 0) & (seconds_to_convert <= period_in_seconds)).mean()


def cohorted_conversion_rates(
        base_timestamp: pd.Series,
        conversion_timestamp: pd.Series,
        cohort: pd.Series,
        intervals: list[tuple[int, str]]):
    """
    Calculate cohorted conversion rates given base timestamp and conversion timestamp.

    Update the function so that it takes a `current_datetime` value that represents the current date and time. For a cohort where any record in the cohort hasn't had enough time elapse between the base_timestamp and the time to convert based on the intervals, that cohort should have a nan value. So for example, if the cohort for `2020-01-01` has a record where the base_timestamp is `2020-01-25 00:00:00` and current_datime is `2020-02-02 00:00:00` and there intervals [(1, 'days'), (10, 'days')] then the first interval is valid because everyone in the cohort has had enough time to convert (i.e. everyone has had at least 1 day in the cohort), but not everyone has had 5 days (e.g. the timestamp for `2020-01-25 00:00:00` has only been in the cohort for 8 days which is not enough time to know if it will convert in 10 days. So the conversion rate is valid for the first interval but not the second.

    Args:
        base_timestamp:
            The base timestamp to use for the calculation.
        conversion_timestamp:
            The conversion timestamp to use for the calculation.
        cohort:
            The cohort to use/group-by for the calculation.
        intervals:
            A list of intervals to calculate the conversion rate for in the form of (value, unit)
            which represents the number of units to calculate the conversion rate for. Valid values
            for unit are 'seconds', 'minutes', 'hours', and 'days'.
    """
    def to_seconds(value, unit):
        unit_to_seconds = {
            'seconds': 1,
            'minutes': 60,
            'hours': 3600,
            'days': 86400,
        }
        return value * unit_to_seconds[unit]

    agg_dict = {
        f'{value}_{unit}': (
            'seconds_to_conversion', lambda x: _percent_converted(x, to_seconds(value, unit))
        )
        for value, unit in intervals
    }
    df = pd.DataFrame({
        'base_timestamp': base_timestamp,
        'conversion_timestamp': conversion_timestamp,
        'cohort': cohort,
    })
    conversions = (
        df
        .assign(seconds_to_conversion=lambda x: (x['conversion_timestamp'] - x['base_timestamp']).dt.total_seconds())
        .groupby('cohort')
        .agg(**agg_dict)
        .reset_index()
    )
    return conversions
