import unittest
import datetime
from enum import Enum, auto

import pandas as pd
import numpy as np

import helpsk.diff as diff
from helpsk import color
from tests.helpers import get_test_path


# noinspection PyMethodMayBeStatic
class TestDiff(unittest.TestCase):

    def test__create_html_change_span(self):
        self.assertEqual(
            diff._create_html_change_span(value='x', is_change=True),
            '<span style="background:#F1948A";>x</span>'
        )
        self.assertEqual(
            diff._create_html_change_span(value='x', is_change=False),
            '<span>x</span>'
        )

    def test__create_html_difference_list(self):
        self.assertEqual(
            diff._create_html_difference_list(value_a='TesT', value_b='test'),
            ['-T', '+t', ' e', ' s', '-T', '+t']
        )

    def test__create_html_cell(self):
        diff_list = diff._create_html_difference_list(value_a='TesT', value_b='test')
        self.assertEqual(
            diff._create_html_cell(difference_list=diff_list, is_first_value=True),
            '<span style="background:#F1948A";>T</span><span>e</span><span>s</span><span style="background:'  # noqa
            '#F1948A";>T</span>'
        )
        self.assertEqual(
            diff._create_html_cell(difference_list=diff_list, is_first_value=False),
            '<span style="background:#F1948A";>t</span><span>e</span><span>s</span><span style="background:'  # noqa
            '#F1948A";>t</span>'
        )

    def test__diff_dataframes(self):
        class TestEnum(Enum):
            VALUE_A = auto()
            VALUE_B = auto()

        df_a = pd.DataFrame({
            'col_1': [1, 1, 1, 1],
            'col_a': [np.nan, 2, 3, 4],
            'col_b': [np.nan, 'b', 'd', 'd'],
            'col_c': pd.date_range('2021-01-01', '2021-01-04'),
            'col_d': [None, np.nan, datetime.date(2021, 4, 2), datetime.date(2021, 4, 2)],
            'col_e': np.nan,
            'col_f': [1.0, 2.0, 3.0, 4.0],
            'col_h': [np.nan, TestEnum.VALUE_A, TestEnum.VALUE_B, TestEnum.VALUE_B],
            'col_i': [
                None,
                np.nan,
                datetime.datetime(2021, 4, 2, 0, 0, 0),
                datetime.datetime(2021, 4, 2, 0, 0, 0)
            ],
        })
        df_b = pd.DataFrame({
            'col_2': [2, 2, 2, 2],
            'col_a': [np.nan, 22, 13, 4],
            'col_b': [np.nan, 'b', 'ad', 'dz'],
            'col_c': pd.date_range('2022-01-02', '2022-01-05'),
            'col_d': [None, np.nan, datetime.date(2021, 5, 2), datetime.date(2021, 4, 6)],
            'col_e': np.nan,
            'col_f': [1.1, 22.0, 3.0, 4.0],
            'col_h': [np.nan, TestEnum.VALUE_B, TestEnum.VALUE_A, TestEnum.VALUE_B],
            'col_i': [
                None,
                np.nan,
                datetime.datetime(2021, 4, 2, 1, 1, 1),
                datetime.datetime(2021, 4, 2, 0, 0, 0)
            ],
        })
        html_a = diff.diff_dataframes(dataframe_a=df_a, dataframe_b=df_b)
        with open(get_test_path('diff/diff_dataframes.html'), 'w') as file:
            file.write(html_a)

        html_b = diff.diff_dataframes(dataframe_a=df_a, dataframe_b=df_b, change_color='#85C1E9')
        with open(get_test_path('diff/diff_dataframes_blue.html'), 'w') as file:
            file.write(html_b)

        # set index values to be different, shouldn't matter
        df_b.index = ['a', 'b', 'c', 'd']
        html_c = diff.diff_dataframes(dataframe_a=df_a, dataframe_b=df_b)
        self.assertEqual(html_a, html_c)

    def test__diff_test__strings(self):
        results = diff.diff_text(text_a='Hello', text_b='hello')
        with open(get_test_path('diff/diff_text__hello.html'), 'w') as file:
            file.write(results)

        results = diff.diff_text(
            text_a='Hello,\nthis is a message. Thanks! :)',
            text_b='hello, this is not a message. No Thanks :('
        )
        with open(get_test_path('diff/diff_text__new_line.html'), 'w') as file:
            file.write(results)

        results = diff.diff_text(
            text_a='Hello,\nthis is a message. Thanks! :)',
            text_b='hello, this is not a message. No Thanks :(',
            change_color=color.WARNING
        )
        with open(get_test_path('diff/diff_text__color.html'), 'w') as file:
            file.write(results)

    def test__diff_test__lists(self):
        results = diff.diff_text(
            text_a=[
                'hey',
                'Hello,\nthis is a message. Thanks! :)',
            ],
            text_b=[
                'Hello',
                'hello, this is not a message. No Thanks :(',
            ],
            change_color=color.WARNING
        )
        with open(get_test_path('diff/diff_text__lists.html'), 'w') as file:
            file.write(results)

    def test__diff_test__empty_strings(self):
        results = diff.diff_text(text_a='', text_b='')
        with open(get_test_path('diff/diff_text__empty_both.html'), 'w') as file:
            file.write(results)

        results = diff.diff_text(text_a='', text_b='Not Empty')
        with open(get_test_path('diff/diff_text__empty_a.html'), 'w') as file:
            file.write(results)

        results = diff.diff_text(text_a='Not Empty', text_b='')
        with open(get_test_path('diff/diff_text__empty_b.html'), 'w') as file:
            file.write(results)

        results = diff.diff_text(text_a=[], text_b=[])
        with open(get_test_path('diff/diff_text__empty_lists.html'), 'w') as file:
            file.write(results)

        results = diff.diff_text(text_a=[''], text_b=[''])
        with open(get_test_path('diff/diff_text__empty_list_values.html'), 'w') as file:
            file.write(results)

        results = diff.diff_text(text_a=[''], text_b=['Not Empty'])
        with open(get_test_path('diff/diff_text__empty_list_values_a.html'), 'w') as file:
            file.write(results)


if __name__ == '__main__':
    unittest.main()
