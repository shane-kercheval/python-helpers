"""Functions to help with calculating conversation/retension rates."""
from __future__ import annotations
import datetime
import numpy as np
import pandas as pd
from helpsk.pandas import relocate


def _percent_converted(seconds_to_convert: pd.Series, period_in_seconds: int) -> float:
    return ((seconds_to_convert > 0) & (seconds_to_convert <= period_in_seconds)).mean()


def cohorted_conversion_rates(
        df: pd.DataFrame,
        base_timestamp: str,
        conversion_timestamp: str,
        cohort: str,
        intervals: list[tuple[int, str]],
        groups: str | None = None,
        current_datetime: str | None = None) -> pd.DataFrame:
    """
    Calculate the cohorted conversion rate for a given base timestamp and conversion timestamp

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
        .assign(seconds_to_conversion=lambda x: (x[conversion_timestamp] - x[base_timestamp]).dt.total_seconds())  # noqa
        .groupby(group_by)
        .apply(f)
        .reset_index()
        .sort_values(group_by, ascending=True)
    )
    return conversions


def _sort_intervals(durations):
    """This is used to display the intervals/durations in the correct order on the graph."""
    unit_multipliers = {
        'days': 86400,  # 24 * 60 * 60
        'hours': 3600,  # 60 * 60
        'minutes': 60,
        'seconds': 1,
    }

    def to_seconds(duration_tuple):
        value, unit = duration_tuple
        return value * unit_multipliers[unit]

    sorted_durations = sorted(durations, key=to_seconds, reverse=True)
    return sorted_durations


def plot_cohorted_conversion_rates(
        df: pd.DataFrame,
        base_timestamp: str,
        conversion_timestamp: str,
        cohort: str,
        intervals: list[tuple[int, str]],
        groups: str | None = None,
        current_datetime: str | None = None,
        graph_type: str = 'bar',
        title: str | None = None,
        subtitle: str | None = None,
        x_axis_label: str | None = None,
        y_axis_label: str | None = None,
        legend_label: str | None = None,
        facet_col_wrap: int = 2,
        category_orders: dict | None = None,
        bar_mode: str = 'overlay',
        opacity: float = 0.9,
        height: int = 700,
        width: int | None = None,
        free_y_axis: bool = True) -> pd.DataFrame:
    """
    Calculate the cohorted converssion rate for a given base timestamp and conversion timestamp

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
        graph_type:
            The type of chart to use for the visualization. Either 'bar' or 'line'
        bar_mode:
            Valid options are `overlay` and `group`. Default is `overlay`. If `relative` is passed
            in it is converted to `overlay`.
    """
    import plotly_express as px
    conversions = cohorted_conversion_rates(
        df=df,
        base_timestamp=base_timestamp,
        conversion_timestamp=conversion_timestamp,
        cohort=cohort,
        intervals=intervals,
        groups=groups,
        current_datetime=current_datetime,
    )
    if not title and not subtitle:
        title = '<br><sub>This graph shows the cohorted conversion rates over time at various durations relative to the base timestamp.</sub>'  # noqa
    else:
        title = title or ''
        if subtitle:
            title += f'<br><sub>{subtitle}</sub>'

    columns = [f'{x} {y}' for x, y in _sort_intervals(intervals)]
    labels = {
        'value': y_axis_label or 'Conversion Rate',
        'variable': legend_label or 'Allowed Duration',
        cohort: x_axis_label or 'Cohort',
    }
    if category_orders:
        category_orders['variable'] = columns
    else:
        category_orders = {'variable': columns}
    hover_data = {
        'value': ':.1%',
        '# of records': ':,',
    }
    if bar_mode is None or bar_mode == 'relative':
        bar_mode = 'overlay'

    if graph_type == 'bar':
        fig = px.bar(
            conversions,
            x=cohort,
            y=columns,
            title=title,
            labels=labels,
            category_orders=category_orders,
            facet_col=groups,
            facet_col_wrap=facet_col_wrap,
            hover_data=hover_data,
            barmode=bar_mode,
            opacity=opacity,
            height=height,
            width=width,
        )
    elif graph_type == 'line':
        fig = px.line(
            conversions,
            x=cohort,
            y=columns,
            title=title,
            labels=labels,
            category_orders=category_orders,
            facet_col=groups,
            facet_col_wrap=facet_col_wrap,
            hover_data=hover_data,
            height=height,
            width=width,
        )
    else:
        raise ValueError(f'Invalid graph_type: {graph_type}')

    if groups and free_y_axis:
        fig.update_yaxes(matches=None)
        fig.for_each_yaxis(lambda yaxis: yaxis.update(showticklabels=True))

    fig.update_yaxes(tickformat=',.0%')
    return fig


def retention_matrix(
        df: pd.DataFrame,
        timestamp: str,
        unique_id: str,
        min_events: int = 1,
        cohort_interval: str = 'week',
        current_datetime: str | None = None) -> pd.DataFrame:
    """
    Calculate the retention matrix for a given timestamp column and unique ID column.

    Args:
        df:
            The dataframe containing the data to calculate the retention matrix for. The data is
            expected to be 'events', meaning that each row represents a single event for a single
            user.
        timestamp:
            The column name for the timestamp to use for the calculation. The column must not
            contain any missing values.
        unique_id:
            The column name for the unique ID to use for the calculation. The column must not
            contain any missing values.
        min_events:
            The minimum number of events a user must have in order to be considered retained in
            subsequent weeks.
        cohort_interval:
            The interval to use for the cohort. Valid options are 'week' and 'day'. This value
            also determines the interval to use for the event period.
        current_datetime:
            The current datetime to use to filter out any events that occurred after the current
            datetime. If None, the current UTC datetime is used.

    """
    # get the number of weeks between the event week and the cohort week
    if cohort_interval == 'week':
        num_days_in_internal = 7
    elif cohort_interval == 'day':
        num_days_in_internal = 1
    else:
        raise ValueError(f'cohort_interval must be either "week" or "day", not {cohort_interval}')

    cohort_interval = cohort_interval[0].upper()

    if current_datetime is None:
        current_datetime = datetime.datetime.utcnow()

    # Step 1: Determine the cohort of each user (the week of their first event)
    df = df[[timestamp, unique_id]].copy()
    assert not df.isna().any().any()

    # filter out any events that occurred after the current datetime
    df = df[df[timestamp] <= current_datetime]
    df['event_period'] = df[timestamp].dt.to_period(cohort_interval).dt.start_time
    df['cohort'] = (
        df.groupby(unique_id)[timestamp]
        .transform('min')
        .dt.to_period(cohort_interval)
        .dt.start_time
    )
    df['period'] = (df['event_period'] - df['cohort']).dt.days // num_days_in_internal
    df['period'] = df['period'].astype(int)

    # Group by cohort_week, event_week and unique_id, then filter by min_visits
    user_events_by_period = (
        df
        .groupby(['cohort', 'period', unique_id])
        .size()
        .reset_index(name='event_count')
    )
    assert not user_events_by_period[[unique_id, 'period']].duplicated().any()
    # for records where period == 0, the user is considered retained by definition so we don't
    # need to filter
    retained_users = user_events_by_period[
        (user_events_by_period['event_count'] >= min_events) | (user_events_by_period['period'] == 0)  # noqa
    ]

    # Step 2: Create a crosstab of users by cohort week and event week
    cohort_matrix = pd.crosstab(
        retained_users['cohort'],
        retained_users['period'],
    )

    # Step 3: Calculate the retention rate by dividing the number of active users each week by the
    # cohort size
    cohort_size = cohort_matrix.iloc[:, 0]
    matrix = cohort_matrix.divide(cohort_size, axis=0).fillna(0)
    assert (matrix <= 1).all().all()
    for row in matrix.index:
        for col in matrix.columns:
            if row + datetime.timedelta(days=col * num_days_in_internal) > current_datetime:
                matrix.loc[row, col] = np.nan

    # Step 4: Reset index and column names for better readability
    matrix.reset_index(inplace=True)
    matrix.columns.name = None
    matrix.columns = matrix.columns.astype(str)
    matrix['# of records'] = cohort_size.astype(int).values
    matrix = relocate(matrix, column='# of records', after='cohort')
    assert (matrix['0'] == 1).all().all()
    return matrix
