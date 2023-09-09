"""Functions to help with calculating conversation/retension rates."""
from __future__ import annotations
import datetime
import pandas as pd


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
        title += f'<br><sub>{subtitle}</sub>'

    columns = [f'{x} {y}' for x, y in intervals]
    labels = {
        'value': y_axis_label or 'Conversion Rate',
        'variable': legend_label or 'Allowed Duration',
        'cohort': x_axis_label or 'Cohort',
    }
    category_orders = {'variable': sorted(columns, reverse=True)}
    hover_data = {
        'value': ':.1%',
        '# of records': ':,',
    }
    if bar_mode is None or bar_mode == 'relative':
        bar_mode = 'overlay'

    if graph_type == 'bar':
        fig = px.bar(
            conversions,
            x='cohort',
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
            x='cohort',
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
