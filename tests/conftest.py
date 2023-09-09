import pytest
import pandas as pd


@pytest.fixture
def conversions():
    data = {
        'created_at': [
            '2023-01-01 00:00:00',  # Base date for the cohort
            '2023-01-01 01:00:00',  # Created an hour later than base date
            '2023-01-02 00:00:00',  # Created a day later
            '2023-01-15 00:00:00',  # Mid of the month, checking longer intervals
            '2023-01-25 00:00:00',  # Near the end of the month
            '2023-01-30 23:59:59',  # Edge case: Just before the end of the month
            '2023-01-01 00:00:01',  # Edge case: Created just after the base date
            '2023-01-31 00:00:00'   # Last day of the month
        ],
        'conversion_1': [
            '2023-01-01 00:00:01',  # Conversion happens in one second
            '2023-01-01 01:00:00',  # Conversion happens exactly at creation time (edge case)
            '2023-01-01 23:59:59',  # Conversion happens before the creation date
            '2023-01-16 00:00:00',  # Conversion happens the next day
            '2023-02-01 00:00:00',  # Conversion happens in the next month (checking longer intervals)  # noqa
            None,  # No conversion (edge case)
            '2023-01-01 00:00:00',  # Conversion happens before creation (logical error, edge case)
            '2023-01-31 00:01:00'   # Conversion happens shortly after creation
        ],
        'conversion_2': [
            '2023-01-01 00:01:00',  # Normal case, conversion after a minute
            '2023-01-01 01:01:00',  # Conversion happens after an hour
            '2023-01-02 00:01:00',  # Conversion happens after a minute
            '2023-02-01 00:00:00',  # Conversion happens in the next month
            '2023-02-24 00:00:00',  # Conversion nearly a month later
            '2023-01-31 23:59:59',  # Conversion at the very end of the month
            None,  # No conversion (edge case)
            '2023-01-31 00:01:00'   # Conversion happens shortly after creation
        ],
        'conversion_3': [
            '2023-01-01 00:00:02',  # Normal case, conversion in two seconds
            '2023-01-01 00:59:59',  # Conversion happens just before an hour completes
            '2023-01-03 00:00:00',  # Conversion after more than a day
            None,  # No conversion (edge case)
            '2023-02-24 23:59:59',  # Conversion nearly a month later, at the end of the day
            '2023-02-01 00:00:00',  # Conversion in the next month
            '2023-01-01 00:00:00',  # Conversion happens at the same time as creation (edge case)
            '2023-01-31 00:00:02'   # Conversion happens shortly after creation
        ],
        'segments': [
            'A',
            'B',
            'A',
            'B',
            'A',
            'A',
            'A',
            'A',
        ]
    }

    df = pd.DataFrame(data)
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['conversion_1'] = pd.to_datetime(df['conversion_1'])
    df['conversion_2'] = pd.to_datetime(df['conversion_2'])
    df['conversion_3'] = pd.to_datetime(df['conversion_3'])
    return df
