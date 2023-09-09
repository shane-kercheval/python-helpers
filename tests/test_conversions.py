
"""Tests for the conversions.py module."""
import pandas as pd
from helpsk.conversions import cohorted_conversion_rates, plot_cohorted_conversion_rates


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
