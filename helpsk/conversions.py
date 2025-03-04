"""Functions to help with calculating conversation/retension rates."""
from __future__ import annotations
import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset
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
    Calculate the cohorted conversion rate for a given base timestamp and conversion timestamp.

    Args:
        df:
            The dataframe containing the data to calculate the conversion rate for.
        base_timestamp:
            The column name for the base timestamp to use for the calculation.
        conversion_timestamp:
            The column name for the conversion timestamp to use for the calculation.
        cohort:
            The column name for the cohort (e.g. day, week, month) to use for the calculation.
            This is probably the same as the base_timestamp, except converted to e.g. week, month,
            quarter, etc. For example, if the base_timestamp is 'created_at' then cohort might be
            created with:
                df['cohort'] = pd.to_datetime(df['created_at']).\
                    dt.to_period('W').dt.to_timestamp()
        intervals:
            A list of intervals to calculate the conversion rate for in the form of (value, unit)
            which represents the number of units to calculate the conversion rate for. For example,
            [(1, 'day'), (7, 'days')] would calculate the conversion rate at 1 day and 7 days for
            each cohort.
        groups:
            The column name for the groups/segments to calculate the conversion rate for.
        current_datetime:
            The current datetime to use for calculating whether an interval is valid.
    """
    def to_seconds(value: int, unit: str) -> int:
        unit_to_seconds = {
            'seconds': 1,
            'minutes': 60,
            'hours': 3600,
            'days': 86400,
        }
        return value * unit_to_seconds[unit]

    # Only copy the columns we need
    if groups:
        df_copy = df[[base_timestamp, conversion_timestamp, cohort, groups]].copy()
    else:
        df_copy = df[[base_timestamp, conversion_timestamp, cohort]].copy()

    # Normalize current_datetime
    if not current_datetime:
        current_datetime = pd.Timestamp.now(tz='UTC')
    else:
        current_datetime = pd.to_datetime(current_datetime)

    # Make current_datetime timezone-naive if it has timezone info
    if hasattr(current_datetime, 'tzinfo') and current_datetime.tzinfo is not None:
        current_datetime = current_datetime.tz_localize(None)

    # Make all datetime columns timezone-naive
    for col in [base_timestamp, conversion_timestamp, cohort]:
        if pd.api.types.is_datetime64_any_dtype(df_copy[col]):
            # Check if any timestamps in the column have timezone info
            has_tz = False
            for ts in df_copy[col]:
                if pd.notna(ts) and hasattr(ts, 'tzinfo') and ts.tzinfo is not None:
                    has_tz = True
                    break
            if has_tz:
                # Remove timezone info from all timestamps in the column using apply
                # This is more robust than df.dt when dealing with mixed timezone data
                df_copy[col] = df_copy[col].apply(
                    lambda x: x.tz_localize(None) if pd.notna(x) and hasattr(x, 'tzinfo') and x.tzinfo is not None else x,  # noqa: E501
                )

    # Keep the original behavior from here on
    group_by = [cohort]
    if groups:
        group_by.append(groups)

    def f(x: pd.Series) -> pd.Series:
        d = {}
        d['# of records'] = len(x)
        seconds_from_max_base = (current_datetime - x[base_timestamp].max()).total_seconds()
        for value, unit in intervals:
            interval_seconds = to_seconds(value, unit)
            if seconds_from_max_base >= interval_seconds:
                d[f'{value} {unit}'] = _percent_converted(
                    x['seconds_to_conversion'],
                    interval_seconds,
                )
            else:
                d[f'{value} {unit}'] = None
        return pd.Series(d)

    # Calculate seconds_to_conversion more safely using apply
    def calculate_time_diff(row):  # noqa: ANN001, ANN202
        if pd.isna(row[conversion_timestamp]) or pd.isna(row[base_timestamp]):
            return pd.NA
        return (row[conversion_timestamp] - row[base_timestamp]).total_seconds()

    df_copy['seconds_to_conversion'] = df_copy.apply(calculate_time_diff, axis=1)

    return (
        df_copy
        .groupby(group_by)
        .apply(f, include_groups=False)
        .reset_index()
        .sort_values(group_by, ascending=True)
    )


def _sort_intervals(durations: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Used to display the intervals/durations in the correct order on the graph."""
    unit_multipliers = {
        'days': 86400,  # 24 * 60 * 60
        'hours': 3600,  # 60 * 60
        'minutes': 60,
        'seconds': 1,
    }

    def to_seconds(duration_tuple: tuple[int, str]) -> int:
        value, unit = duration_tuple
        return value * unit_multipliers[unit]

    return sorted(durations, key=to_seconds, reverse=True)


def plot_cohorted_conversion_rates(
        df: pd.DataFrame,
        base_timestamp: str,
        conversion_timestamp: str,
        cohort: str,
        intervals: list[tuple[int, str]],
        groups: str | None = None,
        current_datetime: str | None = None,
        graph_type: str = 'bar',
        show_num_records: bool = True,
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
    Calculate the cohorted converssion rate for a given base timestamp and conversion timestamp.

    Args:
        df:
            The dataframe containing the data to calculate the conversion rate for.
        base_timestamp:
            The column name for the base timestamp to use for the calculation.
        conversion_timestamp:
            The column name for the conversion timestamp to use for the calculation.
        cohort:
            The column name for the cohort (e.g. day, week, month) to use for the calculation.
            This is probably the same as the base_timestamp, except converted to e.g. week, month,
            quarter, etc. For example, if the base_timestamp is 'created_at' then cohort might be
            created with:
                df['cohort'] = pd.to_datetime(df['created_at']).\
                    dt.to_period('W').dt.to_timestamp()
        intervals:
            A list of intervals to calculate the conversion rate for in the form of (value, unit)
            which represents the number of units to calculate the conversion rate for. For example,
            [(1, 'day'), (7, 'days')] would calculate the conversion rate at 1 day and 7 days for
            each cohort.
        groups:
            The column name for the groups/segments to calculate the conversion rate for.
        current_datetime:
            The current datetime to use for calculating whether an interval is valid.
        graph_type:
            The type of chart to use for the visualization. Either 'bar' or 'line'
        show_num_records:
            Only applicable when graph_type is 'line'. Whether to show the number of records for
            each cohort/interval. This is useful for determining the sample size for each cohort.
            The number of records is shown as the size of the dots on the line chart.
        title:
            The title of the plot.
        subtitle:
            The subtitle of the plot.
        x_axis_label:
            The label for the x-axis.
        y_axis_label:
            The label for the y-axis.
        legend_label:
            The label for the legend.
        facet_col_wrap:
            The number of columns in the facet grid.
        category_orders:
            A dictionary of category orders for the plot.
        bar_mode:
            Valid options are `overlay` and `group`. Default is `overlay`. If `relative` is passed
            in it is converted to `overlay`.
        opacity:
            The opacity of the bars or lines.
        height:
            The height of the plot in pixels.
        width:
            The width of the plot in pixels.
        free_y_axis:
            Whether to allow the y-axis to be free for each facet.
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
        'value': ':.2%',
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
        size_column = '# of records' if show_num_records else None
        scatter_traces = px.scatter(
            conversions,
            x=cohort,
            y=columns,
            size=size_column,
            facet_col=groups,
            facet_col_wrap=facet_col_wrap,
            hover_data=hover_data,
        )
        scatter_traces.update_traces(showlegend=False)
        for trace in scatter_traces.data:
            fig.add_trace(trace)
    else:
        raise ValueError(f'Invalid graph_type: {graph_type}')

    if groups and free_y_axis:
        fig.update_yaxes(matches=None)
        fig.for_each_yaxis(lambda yaxis: yaxis.update(showticklabels=True))

    fig.update_yaxes(tickformat=',.1%')
    return fig


def cohorted_adoption_rates(  # noqa: PLR0912, PLR0915
        df: pd.DataFrame,
        base_timestamp: str,
        conversion_timestamp: str,
        cohort: str,
        n_units: int,
        units: str,
        last_x_cohorts: int = 15,
        groups: str | None = None,
        current_datetime: str | None = None) -> pd.DataFrame:
    """
    Calculate the cohorted adoption rate for a given base timestamp and conversion timestamp.
    Adtopion rate is defined as the percentage of users who converted within a given number of
    units (e.g. days, weeks, months) of the base timestamp. This shows the growth rate and
    corresponding plateau of the adoption rate over time.

    Args:
        df:
            The dataframe containing the data to calculate the conversion rate for.
        base_timestamp:
            The column name for the base timestamp to use for the calculation.
        conversion_timestamp:
            The column name for the conversion timestamp to use for the calculation.
        cohort:
            The column name for the cohort (e.g. day, week, month) to use for the calculation.
            This is probably the same as the base_timestamp, except converted to e.g. week, month,
            quarter, etc. For example, if the base_timestamp is 'created_at' then cohort might be
            created with:
                df['cohort'] = pd.to_datetime(df['created_at']).\
                    dt.to_period('W').dt.to_timestamp()
        n_units:
            The first n units (e.g. 30 days) to calculate the adoption rate for.
        units:
            The units to use for the calculation. Valid options are 'seconds', 'minutes', 'hours',
            'days', 'weeks'.
        last_x_cohorts:
            The last x cohorts to calculate the adoption rate for. This is useful to filter out
            older cohorts that are no longer relevant and/or to reduce the noise in teh data.
        groups:
            The column name for the groups/segments to calculate the conversion rate for.
        current_datetime:
            The current datetime to use to filter out any events that occurred after the current
            datetime. If None, the current UTC datetime is used.
    """
    if current_datetime is None:
        current_datetime = pd.Timestamp.now(tz='UTC').tz_localize(None)
    elif isinstance(current_datetime, str):
        current_datetime = pd.to_datetime(current_datetime)

    # Make current_datetime timezone-naive if it has timezone info
    if hasattr(current_datetime, 'tzinfo') and current_datetime.tzinfo is not None:
        current_datetime = current_datetime.tz_localize(None)

    all_columns = [cohort, base_timestamp, conversion_timestamp]
    if groups:
        all_columns.append(groups)

    data = df[all_columns].copy()

    # Make all datetime columns timezone-naive to handle mixed timezone data
    for col in [base_timestamp, conversion_timestamp, cohort]:
        if pd.api.types.is_datetime64_any_dtype(data[col]):
            # Check if any timestamps in the column have timezone info
            has_tz = False
            for ts in data[col]:
                if pd.notna(ts) and hasattr(ts, 'tzinfo') and ts.tzinfo is not None:
                    has_tz = True
                    break

            if has_tz:
                # Remove timezone info from all timestamps in the column
                data[col] = data[col].apply(
                    lambda x: x.tz_localize(None) if pd.notna(x) and hasattr(x, 'tzinfo') and x.tzinfo is not None else x,  # noqa: E501
                )

    # get most recent x cohorts; need to remove NAs
    cohorts = data[cohort].unique()
    cohorts = cohorts[~pd.isna(cohorts)]

    # Convert all cohorts to timezone-naive before sorting to handle mixed timezone data
    cohorts_naive = []
    cohorts_original = {}  # Keep mapping of naive to original for filtering

    for timestamp in cohorts:
        if pd.notna(timestamp):
            if hasattr(timestamp, 'tzinfo') and timestamp.tzinfo is not None:
                naive_ts = timestamp.tz_localize(None)
                cohorts_naive.append(naive_ts)
                cohorts_original[naive_ts] = timestamp
            else:
                cohorts_naive.append(timestamp)
                cohorts_original[timestamp] = timestamp

    # Sort the naive timestamps and get the last X
    sorted_naive_cohorts = sorted(cohorts_naive)[-last_x_cohorts:]

    # Map back to original timestamps for filtering
    cohorts = [cohorts_original[ts] for ts in sorted_naive_cohorts]
    data = data[data[cohort].isin(cohorts)]

    # calculate the difference between the base timestamp and the conversion timestamp
    # handle cases where conversion_timestamp is None using row-wise approach
    def calculate_diff(row):  # noqa: ANN001, ANN202
        if pd.isna(row[conversion_timestamp]):
            return pd.NA
        diff_seconds = (row[conversion_timestamp] - row[base_timestamp]).total_seconds()
        if units == 'seconds':
            return diff_seconds
        if units == 'minutes':
            return diff_seconds / 60
        if units == 'hours':
            return diff_seconds / 3600
        if units == 'days':
            return diff_seconds / 86400
        if units == 'weeks':
            return diff_seconds / 604800
        raise ValueError(f'Invalid units: {units}')

    diff = data.apply(calculate_diff, axis=1)

    columns = [cohort, base_timestamp]
    if groups:
        columns.append(groups)

    adoption = pd.DataFrame()
    for i in range(0, n_units + 1):
        temp = data[columns].copy()
        # Handle NaN values properly in conversion calculation
        diff_valid = diff.copy()
        diff_valid = pd.Series(
            [(d <= i) and (d > 0) if pd.notna(d) else False for d in diff],
            index=diff.index,
        )
        temp['conversion'] = diff_valid
        temp['index'] = i
        time_delta = pd.Timedelta(
            i * 7 if units == 'weeks' else i,
            'days' if units == 'weeks' else units,
        )
        cutoff_datetime = temp[base_timestamp] + time_delta

        # Ensure cutoff_datetime has the same tzinfo status as current_datetime
        if hasattr(current_datetime, 'tzinfo') and current_datetime.tzinfo is None:
            # If current_datetime is naive, we need naive comparison
            has_tz = False
            for ts in cutoff_datetime:
                if pd.notna(ts) and hasattr(ts, 'tzinfo') and ts.tzinfo is not None:
                    has_tz = True
                    break

            if has_tz:
                # Convert any tz-aware timestamps to naive for comparison
                cutoff_datetime = cutoff_datetime.apply(
                    lambda x: x.tz_localize(None) if pd.notna(x) and hasattr(x, 'tzinfo') and x.tzinfo is not None else x,  # noqa: E501
                )

        temp['is_valid'] = cutoff_datetime < current_datetime
        adoption = pd.concat([adoption, temp], axis=0)

    def f(x: pd.Series) -> pd.Series:
        d = {}
        d['# of records'] = len(x)
        d['Is Finished'] = x['is_valid'].all()
        d['Converted'] = pd.NA if not x['is_valid'].any() else x['conversion'].sum()
        d['Conversion Rate'] = pd.NA if not x['is_valid'].any() else x['conversion'].mean()
        return pd.Series(d)

    adoption_by_cohort = (
        adoption
        .groupby([cohort, 'index', groups] if groups else [cohort, 'index'])
        .apply(f, include_groups=False)
        .reset_index()
        .sort_values(by=[cohort, 'index'])
    )
    return adoption_by_cohort[adoption_by_cohort['Conversion Rate'].notna()]


def plot_cohorted_adoption_rates(
        df: pd.DataFrame,
        base_timestamp: str,
        conversion_timestamp: str,
        cohort: str,
        n_units: int,
        units: str,
        last_x_cohorts: int = 15,
        groups: str | None = None,
        current_datetime: str | None = None,
        show_unfinished_cohorts: bool = True,
        title: str | None = None,
        subtitle: str | None = None,
        x_axis_label: str | None = None,
        y_axis_label: str | None = None,
        facet_col_wrap: int = 2,
        category_orders: dict | None = None,
        height: int = 700,
        width: int | None = None,
        free_y_axis: bool = True) -> pd.DataFrame:
    """
    Calculate the cohorted converssion rate for a given base timestamp and conversion timestamp.

    Args:
        df:
            The dataframe containing the data to calculate the conversion rate for.
        base_timestamp:
            The column name for the base timestamp to use for the calculation.
        conversion_timestamp:
            The column name for the conversion timestamp to use for the calculation.
        cohort:
            The column name for the cohort (e.g. day, week, month) to use for the calculation.
            This is probably the same as the base_timestamp, except converted to e.g. week, month,
            quarter, etc. For example, if the base_timestamp is 'created_at' then cohort might be
            created with:
                df['cohort'] = pd.to_datetime(df['created_at']).\
                    dt.to_period('W').dt.to_timestamp()
        n_units:
            The first n units (e.g. 30 days) to calculate the adoption rate for.
        units:
            The units to use for the calculation. Valid options are 'seconds', 'minutes', 'hours',
            'days', 'weeks'.
        last_x_cohorts:
            The last x cohorts to calculate the adoption rate for. This is useful to filter out
            older cohorts that are no longer relevant and/or to reduce the noise in teh data.
        groups:
            The column name for the groups/segments to calculate the conversion rate for.
        current_datetime:
            The current datetime to use to filter out any events that occurred after the current
            datetime. If None, the current UTC datetime is used.
        show_unfinished_cohorts:
            If True, show cohorts that have not yet finished (i.e. not all records in the cohort
            have had enough time to convert).
        title:
            The title of the plot.
        subtitle:
            The subtitle of the plot.
        x_axis_label:
            The label for the x-axis.
        y_axis_label:
            The label for the y-axis.
        facet_col_wrap:
            The number of columns in the facet grid.
        category_orders:
            A dictionary of category orders for the plot.
        height:
            The height of the plot in pixels.
        width:
            The width of the plot in pixels.
        free_y_axis:
            Whether to allow the y-axis to be free for each facet.
    """
    import plotly_express as px
    conversions = cohorted_adoption_rates(
        df=df,
        base_timestamp=base_timestamp,
        conversion_timestamp=conversion_timestamp,
        cohort=cohort,
        n_units=n_units,
        units=units,
        last_x_cohorts=last_x_cohorts,
        groups=groups,
        current_datetime=current_datetime,
    )
    if not title and not subtitle:
        title = '<br><sub>This graph shows the cohorted adoption rates over time.</sub>'
    else:
        title = title or ''
        if subtitle:
            title += f'<br><sub>{subtitle}</sub>'

    labels = {
        'index': x_axis_label or f'# of {units} since `{base_timestamp}`',
        'Conversion Rate': y_axis_label or 'Cumulative Conversion Rate',
    }
    hover_data = {
        '# of records': ':,',
        'Converted': ':,',
        'Conversion Rate': ':.2%',
    }
    if show_unfinished_cohorts:
        symbol = 'Is Finished'
        symbol_map = {True: 'hash', False: 'diamond'}
    else:
        symbol = None
        symbol_map = None
    fig = px.line(
        conversions,
        x='index',
        y='Conversion Rate',
        color=cohort,
        symbol=symbol,
        symbol_map=symbol_map,
        title=title,
        labels=labels,
        category_orders=category_orders,
        facet_col=groups,
        facet_col_wrap=facet_col_wrap,
        hover_data=hover_data,
        height=height,
        width=width,
    )
    if groups and free_y_axis:
        fig.update_yaxes(matches=None)
        fig.for_each_yaxis(lambda yaxis: yaxis.update(showticklabels=True))

    fig.update_yaxes(tickformat=',.1%')
    return fig


def retention_matrix(  # noqa: PLR0912, PLR0915
        df: pd.DataFrame,
        timestamp: str,
        unique_id: str,
        min_events: int = 1,
        intervals: str = 'week',
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
        intervals:
            The intervals to use for the cohort. Valid options are 'month', 'week' and 'day'.
            This value determines the interval to use for both the cohort and the event period.
            Meaning, columns returned will be '0', '1', '2', etc. for the number of intervals. If
            'months', then these numbers represent the number of months since the cohort. If 'week'
            then these numbers represent the number of weeks since the cohort. And so on.
        current_datetime:
            The current datetime to use to filter out any events that occurred after the current
            datetime. If None, the current UTC datetime is used.

    """
    # Copy needed columns
    df_copy = df[[timestamp, unique_id]].copy()
    assert not df_copy.isna().any().any()

    # Normalize current_datetime
    if current_datetime is None:
        current_datetime = pd.Timestamp.now(tz='UTC')
    elif isinstance(current_datetime, str):
        current_datetime = pd.to_datetime(current_datetime)

    # Make current_datetime timezone-naive if it has timezone info
    if hasattr(current_datetime, 'tzinfo') and current_datetime.tzinfo is not None:
        current_datetime = current_datetime.tz_localize(None)

    # Make timestamp column timezone-naive if it has timezone info
    if pd.api.types.is_datetime64_any_dtype(df_copy[timestamp]):
        # Check if any timestamps in the column have timezone info
        has_tz = False
        for ts in df_copy[timestamp]:
            if pd.notna(ts) and hasattr(ts, 'tzinfo') and ts.tzinfo is not None:
                has_tz = True
                break

        if has_tz:
            # Remove timezone info from all timestamps in the column
            df_copy[timestamp] = df_copy[timestamp].apply(
                lambda x: x.tz_localize(None) if pd.notna(x) and hasattr(x, 'tzinfo') and x.tzinfo is not None else x,  # noqa: E501
            )

    # get the number of weeks between the event week and the cohort week
    is_month_interval = intervals == 'month'
    if intervals == 'month':
        num_days_in_internal = None
    elif intervals == 'week':
        num_days_in_internal = 7
    elif intervals == 'day':
        num_days_in_internal = 1
    else:
        raise ValueError(f'interval must be either "month", "week" or "day", not {intervals}')

    intervals = intervals[0].upper()

    if current_datetime is None:
        current_datetime = pd.Timestamp.now(tz='UTC').tz_localize(None)

    df_copy = df_copy[[timestamp, unique_id]].copy()
    assert not df_copy.isna().any().any()
    # Filter out any events that occurred after the current datetime, handling mixed timezone data
    # Create a filtered copy to avoid the mixed timezone comparison issue
    filtered_indices = []
    for idx, row in df_copy.iterrows():
        ts = row[timestamp]
        # If timestamps have different timezone properties, convert to naive for comparison
        if pd.notna(ts):
            if (hasattr(ts, 'tzinfo') and ts.tzinfo is not None and
                current_datetime.tzinfo is None):
                ts = ts.tz_localize(None)
            elif (ts.tzinfo is None and hasattr(current_datetime, 'tzinfo') and
                  current_datetime.tzinfo is not None):
                ts_compare = current_datetime.tz_localize(None)
                if ts <= ts_compare:
                    filtered_indices.append(idx)
                continue

            if ts <= current_datetime:
                filtered_indices.append(idx)

    df_copy = df_copy.loc[filtered_indices]
    # Step 1: Determine the cohort of each user (the day/week/month of their first event)
    # Check if we have data after filtering
    if df_copy.empty:
        # Return an empty dataframe with appropriate columns
        return pd.DataFrame(columns=['cohort', '# of unique ids', '0'])

    # First, make all timestamps timezone-naive to ensure consistent processing
    df_copy['timestamp_naive'] = df_copy[timestamp].apply(
        lambda x: x.tz_localize(None) if pd.notna(x) and hasattr(x, 'tzinfo') and x.tzinfo is not None else x,  # noqa: E501
    )

    # Now all timestamps are naive, we can safely convert to periods
    df_copy['event_period'] = pd.to_datetime(df_copy['timestamp_naive']).dt.to_period(intervals).dt.start_time  # noqa: E501

    # Get min timestamp per user, then convert to period start - using the naive timestamps
    min_timestamps = df_copy.groupby(unique_id, observed=True)['timestamp_naive'].transform('min')
    df_copy['cohort'] = pd.to_datetime(min_timestamps).dt.to_period(intervals).dt.start_time
    if is_month_interval:
        # calculate the number of months between the cohort and the event
        df_copy['period'] = (df_copy['event_period'].dt.year - df_copy['cohort'].dt.year) * 12 + \
            (df_copy['event_period'].dt.month - df_copy['cohort'].dt.month)
    else:
        df_copy['period'] = (df_copy['event_period'] - df_copy['cohort']).dt.days // num_days_in_internal  # noqa: E501

    df_copy['period'] = df_copy['period'].astype(int)

    # Group by cohort_week, event_week and unique_id, then filter by min_visits
    user_events_by_period = (
        df_copy
        .groupby(['cohort', 'period', unique_id], observed=True)
        .size()
        .reset_index(name='event_count')
    )
    # weird bug(?) where in some cases pandas returns records with event_count == 0 (i.e.
    # users with no events in a given period which duplicates the cohort/unique_id)
    user_events_by_period = user_events_by_period.query('event_count > 0')
    assert not user_events_by_period[[unique_id, 'period']].duplicated().any()
    # for records where period == 0, the user is considered retained by definition (regardless of
    # number of events) so we don't need to filter
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
            offset = {'months': col} if is_month_interval else {'days': col * num_days_in_internal}

            # Ensure consistent timezone handling for comparison
            cutoff_date = row + DateOffset(**offset)
            if hasattr(cutoff_date, 'tzinfo') and cutoff_date.tzinfo is not None and current_datetime.tzinfo is None:  # noqa: E501
                cutoff_date = cutoff_date.tz_localize(None)

            if cutoff_date > current_datetime:
                matrix.loc[row, col] = np.nan

    # Step 4: Reset index and column names for better readability
    matrix = matrix.reset_index()
    matrix.columns.name = None
    matrix.columns = matrix.columns.astype(str)
    matrix['# of unique ids'] = cohort_size.astype(int).to_numpy()
    matrix = relocate(matrix, column='# of unique ids', after='cohort')
    assert (matrix['0'] == 1).all().all()
    return matrix
