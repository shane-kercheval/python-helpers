
"""Tests for the conversions.py module."""
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import pytest
from helpsk.conversions import (
    _sort_intervals,
    cohorted_conversion_rates,
    plot_cohorted_conversion_rates,
    retention_matrix
)


def test_cohorted_conversion_rate__day_cohort(conversions):
    df = conversions.copy()
    # Define your intervals and current datetime
    intervals = [(1, 'seconds'), (1, 'minutes'), (2, 'hours'), (1, 'days'), (30, 'days')]
    current_datetime = '2023-02-02 00:00:00'

    # Create cohorts based on created_at date
    df['cohort'] = df['created_at'].dt.to_period('D').dt.to_timestamp()

    df_copy = df.copy()
    result = cohorted_conversion_rates(
        df=df,
        base_timestamp='created_at',
        conversion_timestamp='conversion_1',
        cohort='cohort',
        intervals=intervals,
        current_datetime=current_datetime,
    )
    assert df.equals(df_copy)
    assert not result['cohort'].duplicated().any()
    assert set(result['cohort']) == set(df['cohort'].unique())
    assert result['# of records'].tolist() == [3, 1, 1, 1, 1, 1]
    # all records have had a chance to convert
    # for 2023-01-01, one converts exactly at the created_at time (which doesn't count);
    # another happens before the created_at time, and one happens 1 second after)
    assert result['1 seconds'].round(3).tolist() == [0.333, 0, 0, 0, 0, 0]
    assert result['1 minutes'].round(3).tolist() == [0.333, 0, 0, 0, 0, 1]
    assert result['2 hours'].round(3).tolist() == [0.333, 0, 0, 0, 0, 1]
    assert result['1 days'].round(3).tolist() == [0.333, 0, 1, 0, 0, 1]
    assert result['30 days'].iloc[0].round(3) == 0.333
    assert result['30 days'].iloc[1] == 0
    assert pd.isna(result['30 days'].iloc[2])
    assert pd.isna(result['30 days'].iloc[3])
    assert pd.isna(result['30 days'].iloc[4])
    assert pd.isna(result['30 days'].iloc[5])

    current_datetime = '2023-01-25 23:59:50'

    # Call the conversion_rate function
    result = cohorted_conversion_rates(
        df=df,
        base_timestamp='created_at',
        conversion_timestamp='conversion_1',
        cohort='cohort',
        intervals=intervals,
        current_datetime=current_datetime,
    )
    assert df.equals(df_copy)
    assert not result['cohort'].duplicated().any()
    assert set(result['cohort']) == set(df['cohort'].unique())
    assert result['# of records'].tolist() == [3, 1, 1, 1, 1, 1]

    assert result['1 seconds'].iloc[0].round(3) == 0.333
    assert result['1 seconds'].iloc[1] == 0
    assert result['1 seconds'].iloc[2] == 0
    assert result['1 seconds'].iloc[3] == 0
    assert pd.isna(result['1 seconds'].iloc[4])
    assert pd.isna(result['1 seconds'].iloc[5])

    assert result['1 minutes'].iloc[0].round(3) == 0.333
    assert result['1 minutes'].iloc[1] == 0
    assert result['1 minutes'].iloc[2] == 0
    assert result['1 minutes'].iloc[3] == 0
    assert pd.isna(result['1 minutes'].iloc[4])
    assert pd.isna(result['1 minutes'].iloc[5])

    assert result['2 hours'].iloc[0].round(3) == 0.333
    assert result['2 hours'].iloc[1] == 0
    assert result['2 hours'].iloc[2] == 0
    assert result['2 hours'].iloc[3] == 0
    assert pd.isna(result['2 hours'].iloc[4])
    assert pd.isna(result['2 hours'].iloc[5])

    assert result['1 days'].iloc[0].round(3) == 0.333
    assert result['1 days'].iloc[1] == 0
    assert result['1 days'].iloc[2] == 1
    assert pd.isna(result['1 days'].iloc[3])
    assert pd.isna(result['1 days'].iloc[4])
    assert pd.isna(result['1 days'].iloc[5])

    assert pd.isna(result['30 days'].iloc[0])
    assert pd.isna(result['30 days'].iloc[1])
    assert pd.isna(result['30 days'].iloc[2])
    assert pd.isna(result['30 days'].iloc[3])
    assert pd.isna(result['30 days'].iloc[4])
    assert pd.isna(result['30 days'].iloc[5])


def test_cohorted_conversion_rate__week_cohort(conversions):
    df = conversions.copy()
    # Define your intervals and current datetime
    intervals = [(1, 'seconds'), (1, 'minutes'), (2, 'hours'), (1, 'days'), (30, 'days')]
    current_datetime = '2023-02-02 00:00:00'

    # Create cohorts based on created_at date
    df['cohort'] = df['created_at'].dt.to_period('W').dt.to_timestamp()

    df_copy = df.copy()
    result = cohorted_conversion_rates(
        df=df,
        base_timestamp='created_at',
        conversion_timestamp='conversion_1',
        cohort='cohort',
        intervals=intervals,
        current_datetime=current_datetime,
    )
    assert df.equals(df_copy)
    assert not result['cohort'].duplicated().any()
    assert set(result['cohort']) == set(df['cohort'].unique())
    assert result['# of records'].tolist() == [3, 1, 1, 1, 2]
    # all records have had a chance to convert
    # for 2023-01-01, one converts exactly at the created_at time (which doesn't count);
    # another happens before the created_at time, and one happens 1 second after)
    assert result['1 seconds'].round(3).tolist() == [0.333, 0, 0, 0, 0]
    assert result['1 minutes'].round(3).tolist() == [0.333, 0, 0, 0, 0.5]
    assert result['2 hours'].round(3).tolist() == [0.333, 0, 0, 0, 0.5]
    assert result['1 days'].round(3).tolist() == [0.333, 0, 1, 0, 0.5]
    assert result['30 days'].iloc[0].round(3) == 0.333
    assert result['30 days'].iloc[1] == 0
    assert pd.isna(result['30 days'].iloc[2])
    assert pd.isna(result['30 days'].iloc[3])
    assert pd.isna(result['30 days'].iloc[4])

    current_datetime = '2023-01-25 23:59:50'

    # Call the conversion_rate function
    result = cohorted_conversion_rates(
        df=df,
        base_timestamp='created_at',
        conversion_timestamp='conversion_1',
        cohort='cohort',
        intervals=intervals,
        current_datetime=current_datetime,
    )
    assert df.equals(df_copy)
    assert not result['cohort'].duplicated().any()
    assert set(result['cohort']) == set(df['cohort'].unique())
    assert result['# of records'].tolist() == [3, 1, 1, 1, 2]

    assert result['1 seconds'].iloc[0].round(3) == 0.333
    assert result['1 seconds'].iloc[1] == 0
    assert result['1 seconds'].iloc[2] == 0
    assert result['1 seconds'].iloc[3] == 0
    assert pd.isna(result['1 seconds'].iloc[4])

    assert result['1 minutes'].iloc[0].round(3) == 0.333
    assert result['1 minutes'].iloc[1] == 0
    assert result['1 minutes'].iloc[2] == 0
    assert result['1 minutes'].iloc[3] == 0
    assert pd.isna(result['1 minutes'].iloc[4])

    assert result['2 hours'].iloc[0].round(3) == 0.333
    assert result['2 hours'].iloc[1] == 0
    assert result['2 hours'].iloc[2] == 0
    assert result['2 hours'].iloc[3] == 0
    assert pd.isna(result['2 hours'].iloc[4])

    assert result['1 days'].iloc[0].round(3) == 0.333
    assert result['1 days'].iloc[1] == 0
    assert result['1 days'].iloc[2] == 1
    assert pd.isna(result['1 days'].iloc[3])
    assert pd.isna(result['1 days'].iloc[4])

    assert pd.isna(result['30 days'].iloc[0])
    assert pd.isna(result['30 days'].iloc[1])
    assert pd.isna(result['30 days'].iloc[2])
    assert pd.isna(result['30 days'].iloc[3])
    assert pd.isna(result['30 days'].iloc[4])


def test_cohorted_conversion_rate__groups(conversions):
    df = conversions.copy()
    # Define your intervals and current datetime
    intervals = [(1, 'seconds'), (1, 'minutes'), (2, 'hours'), (1, 'days'), (30, 'days')]
    current_datetime = '2023-01-25 23:59:50'
    df['cohort'] = df['created_at'].dt.to_period('W').dt.to_timestamp()

    df_copy = df.copy()
    result = cohorted_conversion_rates(
        df=df,
        base_timestamp='created_at',
        conversion_timestamp='conversion_1',
        cohort='cohort',
        intervals=intervals,
        groups='segments',
        current_datetime=current_datetime,
    )
    assert df.equals(df_copy)
    assert not result[['cohort', 'segments']].duplicated().any()
    assert set(result['cohort']) == set(df['cohort'].unique())
    assert result['# of records'].tolist() == [2, 1, 1, 1, 1, 2]

    assert result['1 seconds'].iloc[0] == 0.5
    assert result['1 seconds'].iloc[1] == 0
    assert result['1 seconds'].iloc[2] == 0
    assert result['1 seconds'].iloc[3] == 0
    assert result['1 seconds'].iloc[4] == 0
    assert pd.isna(result['1 seconds'].iloc[5])

    assert result['1 minutes'].iloc[0] == 0.5
    assert result['1 minutes'].iloc[1] == 0
    assert result['1 minutes'].iloc[2] == 0
    assert result['1 minutes'].iloc[3] == 0
    assert result['1 minutes'].iloc[4] == 0
    assert pd.isna(result['1 minutes'].iloc[5])

    assert result['2 hours'].iloc[0] == 0.5
    assert result['2 hours'].iloc[1] == 0
    assert result['2 hours'].iloc[2] == 0
    assert result['2 hours'].iloc[3] == 0
    assert result['2 hours'].iloc[4] == 0
    assert pd.isna(result['2 hours'].iloc[5])

    assert result['1 days'].iloc[0] == 0.5
    assert result['1 days'].iloc[1] == 0
    assert result['1 days'].iloc[2] == 0
    assert result['1 days'].iloc[3] == 1
    assert pd.isna(result['1 days'].iloc[4])
    assert pd.isna(result['1 days'].iloc[5])

    assert pd.isna(result['30 days'].iloc[0])
    assert pd.isna(result['30 days'].iloc[1])
    assert pd.isna(result['30 days'].iloc[2])
    assert pd.isna(result['30 days'].iloc[3])
    assert pd.isna(result['30 days'].iloc[4])
    assert pd.isna(result['30 days'].iloc[5])


def test_plot_cohorted_conversion_rates(conversions):
    df = conversions.copy()
    df['cohort'] = df['created_at'].dt.to_period('W').dt.to_timestamp()
    _ = plot_cohorted_conversion_rates(
        df=df,
        base_timestamp='created_at',
        conversion_timestamp='conversion_1',
        cohort='cohort',
        intervals=[(1, 'seconds'), (1, 'minutes'), (2, 'hours'), (1, 'days'), (30, 'days')],
        groups=None,
        current_datetime='2023-01-25 23:59:50',
    )
    _ = plot_cohorted_conversion_rates(
        df=df,
        base_timestamp='created_at',
        conversion_timestamp='conversion_1',
        cohort='cohort',
        intervals=[(1, 'seconds'), (1, 'minutes'), (2, 'hours'), (1, 'days'), (30, 'days')],
        groups='segments',
        category_orders={},
        current_datetime='2023-01-25 23:59:50',
    )
    _ = plot_cohorted_conversion_rates(
        df=df,
        base_timestamp='created_at',
        conversion_timestamp='conversion_1',
        cohort='cohort',
        intervals=[(1, 'seconds'), (1, 'minutes'), (2, 'hours'), (1, 'days'), (30, 'days')],
        groups='segments',
        current_datetime='2023-01-25 23:59:50',
        graph_type='bar',
        title='Title',
        subtitle='Subtitle',
        x_axis_label='X Axis Label',
        y_axis_label='Y Axis Label',
        legend_label='Legend Label',
        facet_col_wrap=6,
        bar_mode='relative',
        category_orders={'abc': ['a', 'b', 'c']},
        opacity=0.9,
        height=800,
        width=100,
        free_y_axis=False,
    )
    _ = plot_cohorted_conversion_rates(
        df=df,
        base_timestamp='created_at',
        conversion_timestamp='conversion_1',
        cohort='cohort',
        intervals=[(1, 'seconds'), (1, 'minutes'), (2, 'hours'), (1, 'days'), (30, 'days')],
        groups='segments',
        current_datetime='2023-01-25 23:59:50',
        graph_type='line',
        title='Title',
        subtitle='Subtitle',
        x_axis_label='X Axis Label',
        y_axis_label='Y Axis Label',
        legend_label='Legend Label',
        facet_col_wrap=6,
        bar_mode='relative',
        opacity=0.9,
        height=800,
        width=100,
        free_y_axis=True,
    )


def test__sort_intervals():
    intervals = [
        (120, 'seconds'),
        (30, 'minutes'),
        (2, 'hours'),
        (1, 'days'),
    ]
    sorted_intervals = _sort_intervals(intervals)
    assert sorted_intervals == [
        (1, 'days'),
        (2, 'hours'),
        (30, 'minutes'),
        (120, 'seconds'),
    ]

    intervals = [
        (45, 'seconds'),
        (1, 'hours'),
        (5, 'minutes'),
    ]
    sorted_intervals = _sort_intervals(intervals)
    assert sorted_intervals == [
        (1, 'hours'),
        (5, 'minutes'),
        (45, 'seconds'),
    ]

    intervals = [
        (1, 'days'),
        (14, 'days'),
        (7, 'days'),
    ]
    sorted_intervals = _sort_intervals(intervals)
    assert sorted_intervals == [
        (14, 'days'),
        (7, 'days'),
        (1, 'days'),
    ]


def generate_fake_data():
    base_date = datetime(2023, 1, 1)
    data = []
    for user_id in range(100):
        for days in range(0, 70, 1):
            event_datetime = base_date + timedelta(days=days + user_id)
            if event_datetime > base_date + timedelta(days=100):
                continue
            if np.random.rand() < 0.5:
                # print('append', user_id, days)
                data.append({
                    'user_id': user_id,
                    'datetime': event_datetime
                })
            if np.random.rand() < 0.5:
                # print('append', user_id, days)
                data.append({
                    'user_id': user_id,
                    'datetime': event_datetime
                })
    return pd.DataFrame(data, columns=['user_id', 'datetime'])


def test_retention_matrix():  # noqa
    df = generate_fake_data()
    copy_df = df.copy()
    retention_week = retention_matrix(
        df,
        timestamp='datetime',
        unique_id='user_id',
        intervals='week',
        current_datetime=df['datetime'].max(),
    )
    assert df.equals(copy_df)
    assert retention_week['cohort'].notna().all()
    assert retention_week['# of unique ids'].notna().all()
    assert (retention_week.drop(columns=['cohort', '# of unique ids']).fillna(0) <= 1).all().all()
    assert (retention_week['# of unique ids'] > 1).any()
    assert (retention_week['0'] == 1).all().all()
    assert retention_week['# of unique ids'].sum() == df['user_id'].nunique()
    expected_cohort_sizes = (
        df
        .groupby('user_id')
        .agg(min_date=('datetime', 'min'))
        .assign(cohort=lambda x: x['min_date'].dt.to_period('W').dt.to_timestamp())
        .groupby('cohort')
        .size()
        .sort_index()
        .values
        .tolist()
    )
    assert retention_week['# of unique ids'].tolist() == expected_cohort_sizes

    retention_day = retention_matrix(
        df,
        timestamp='datetime',
        unique_id='user_id',
        intervals='day',
    )
    assert df.equals(copy_df)
    assert retention_day['cohort'].notna().all()
    assert retention_day['# of unique ids'].notna().all()
    assert (retention_day.drop(columns=['cohort', '# of unique ids']).fillna(0) <= 1).all().all()
    assert (retention_day['# of unique ids'] > 1).any()
    assert (retention_day['0'] == 1).all().all()
    assert retention_day['# of unique ids'].sum() == df['user_id'].nunique()
    expected_cohort_sizes = (
        df
        .groupby('user_id')
        .agg(min_date=('datetime', 'min'))
        .assign(cohort=lambda x: x['min_date'].dt.to_period('D').dt.to_timestamp())
        .groupby('cohort')
        .size()
        .sort_index()
        .values
        .tolist()
    )
    assert retention_day['# of unique ids'].tolist() == expected_cohort_sizes

    # test with min_events == 2
    retention2 = retention_matrix(
        df,
        timestamp='datetime',
        unique_id='user_id',
        intervals='week',
        current_datetime=df['datetime'].max(),
        min_events=2,
    )
    assert retention2['cohort'].equals(retention_week['cohort'])
    assert retention2['# of unique ids'].equals(retention_week['# of unique ids'])
    assert retention2['0'].equals(retention_week['0'])
    _retention_2 = retention2.drop(columns=['cohort', '# of unique ids', '0']).fillna(0)
    _retention_week = retention_week.drop(columns=['cohort', '# of unique ids', '0']).fillna(0)
    assert (_retention_2 < _retention_week).any().any()
    assert retention2['cohort'].notna().all()
    assert retention2['# of unique ids'].notna().all()
    assert (retention2.drop(columns=['cohort', '# of unique ids']).fillna(0) <= 1).all().all()
    assert (retention2['# of unique ids'] > 1).any()
    assert (retention2['0'] == 1).all().all()
    assert retention2['# of unique ids'].sum() == df['user_id'].nunique()
    expected_cohort_sizes = (
        df
        .groupby('user_id')
        .agg(min_date=('datetime', 'min'))
        .assign(cohort=lambda x: x['min_date'].dt.to_period('W').dt.to_timestamp())
        .groupby('cohort')
        .size()
        .sort_index()
        .values
        .tolist()
    )
    assert retention2['# of unique ids'].tolist() == expected_cohort_sizes

    df.iloc[0, 0] = np.nan
    with pytest.raises(AssertionError):
        retention_matrix(
            df,
            timestamp='datetime',
            unique_id='user_id',
            intervals='week',
        )


def test_retention_matrix_by_month():  # noqa
    df = generate_fake_data()
    copy_df = df.copy()
    retention_month = retention_matrix(
        df,
        timestamp='datetime',
        unique_id='user_id',
        intervals='month',
        current_datetime=df['datetime'].max(),
    )
    assert df.equals(copy_df)
    assert retention_month['cohort'].notna().all()
    assert retention_month['# of unique ids'].notna().all()
    assert (retention_month.drop(columns=['cohort', '# of unique ids']).fillna(0) <= 1).all().all()
    assert (retention_month['# of unique ids'] > 1).any()
    assert (retention_month['0'] == 1).all().all()
    assert retention_month['# of unique ids'].sum() == df['user_id'].nunique()
    expected_cohort_sizes = (
        df
        .groupby('user_id')
        .agg(min_date=('datetime', 'min'))
        .assign(cohort=lambda x: x['min_date'].dt.to_period('M').dt.to_timestamp())
        .groupby('cohort')
        .size()
        .sort_index()
        .values
        .tolist()
    )
    assert retention_month['# of unique ids'].tolist() == expected_cohort_sizes

    retention_month_28 = retention_matrix(
        df,
        timestamp='datetime',
        unique_id='user_id',
        intervals='month',
        current_datetime=df['datetime'].max(),
        min_events=5,
    )
    # this could randomly fail if we don't have any retained users in the last month
    assert retention_month.columns.tolist() == retention_month_28.columns.tolist()
    assert df.equals(copy_df)
    assert retention_month_28['cohort'].notna().all()
    assert retention_month_28['# of unique ids'].notna().all()
    assert (retention_month_28.drop(columns=['cohort', '# of unique ids']).fillna(0) <= 1).all().all()
    assert (retention_month_28['# of unique ids'] > 1).any()
    assert (retention_month_28['0'] == 1).all().all()
    assert retention_month_28['# of unique ids'].sum() == df['user_id'].nunique()
    expected_cohort_sizes = (
        df
        .groupby('user_id')
        .agg(min_date=('datetime', 'min'))
        .assign(cohort=lambda x: x['min_date'].dt.to_period('M').dt.to_timestamp())
        .groupby('cohort')
        .size()
        .sort_index()
        .values
        .tolist()
    )
    assert retention_month_28['# of unique ids'].tolist() == expected_cohort_sizes
    _retention_month = retention_month.drop(columns=['cohort', '# of unique ids', '0']).fillna(0)
    _retention_month_28 = retention_month_28.drop(columns=['cohort', '# of unique ids', '0']).fillna(0)
    assert (_retention_month > _retention_month_28).any().any()


def test_retention__duplicate_user_cohort_bug():  # noqa
    df = pd.DataFrame({
        'categories': pd.Categorical(['a', 'b', 'c', 'a', 'b']),
        'dates': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05']),  # noqa
        'datetimes': pd.to_datetime(['2023-01-01 01:01:01', '2023-01-02 02:02:02', '2023-01-03 03:03:03', '2023-01-04 04:04:04', '2023-01-05 05:05:05']),  # noqa
    })
    copy_df = df.copy()
    retention = retention_matrix(
        df,
        timestamp='datetimes',
        unique_id='categories',
        intervals='week',
        current_datetime=None,
    )
    assert df.equals(copy_df)
    assert retention['cohort'].notna().all()
    assert retention['# of unique ids'].notna().all()
    assert (retention.drop(columns=['cohort', '# of unique ids']).fillna(0) <= 1).all().all()
    assert (retention['# of unique ids'] > 1).any()
    assert (retention['0'] == 1).all().all()
    assert retention['# of unique ids'].sum() == df['categories'].nunique()
    expected_cohort_sizes = (
        df
        .groupby('categories')
        .agg(min_date=('datetimes', 'min'))
        .assign(cohort=lambda x: x['min_date'].dt.to_period('W').dt.to_timestamp())
        .groupby('cohort')
        .size()
        .sort_index()
        .values
        .tolist()
    )
    assert retention['# of unique ids'].tolist() == expected_cohort_sizes

    retention = retention_matrix(
        df,
        timestamp='datetimes',
        unique_id='categories',
        intervals='week',
        current_datetime=None,
        min_events=2,
    )
    assert df.equals(copy_df)
    assert retention['cohort'].notna().all()
    assert retention['# of unique ids'].notna().all()
    assert (retention.drop(columns=['cohort', '# of unique ids']).fillna(0) <= 1).all().all()
    assert (retention['# of unique ids'] > 1).any()
    assert (retention['0'] == 1).all().all()
    assert retention['# of unique ids'].sum() == df['categories'].nunique()
    expected_cohort_sizes = (
        df
        .groupby('categories')
        .agg(min_date=('datetimes', 'min'))
        .assign(cohort=lambda x: x['min_date'].dt.to_period('W').dt.to_timestamp())
        .groupby('cohort')
        .size()
        .sort_index()
        .values
        .tolist()
    )
    assert retention['# of unique ids'].tolist() == expected_cohort_sizes
