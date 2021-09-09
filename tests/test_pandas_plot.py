import os
import unittest
from enum import Enum, unique, auto

import numpy as np

from helpsk import validation as hv
from helpsk.pandas import *
from helpsk.utility import redirect_stdout_to_file
from helpsk.validation import iterables_are_equal
from tests.helpers import get_data_credit, get_test_path

import helpsk.pandas_style as pstyle


@unique
class TestEnum(Enum):
    VALUE_A = auto()
    VALUE_B = auto()


class TestPandasStyle(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.credit_data = get_data_credit()
        sample_data = pd.DataFrame({'col_a': [np.nan, 2, 3, 4],
                                    'col_b': [np.nan, 'b', 'd', 'd'],
                                    'col_c': pd.date_range('2021-01-01', '2021-01-04'),
                                    'col_d': [None, np.nan,
                                              datetime.date(2021, 4, 2), datetime.date(2021, 4, 2)],
                                    'col_e': np.nan,
                                    'col_f': [1.0, 2.0, 3.0, 4.0],
                                    'col_h': [np.nan, TestEnum.VALUE_A, TestEnum.VALUE_B, TestEnum.VALUE_B],
                                    'col_i': [None, np.nan, datetime.datetime(2021, 4, 2, 0, 0, 0),
                                              datetime.datetime(2021, 4, 2, 0, 0, 0)],
                                    })
        sample_data.loc[0, 'col_c'] = np.nan
        sample_data['col_g'] = sample_data['col_b'].astype('category')
        cls.sample_data = sample_data

    # def test_format(self):
    #     with open(get_test_path() + '/test_files/pandas_style/format__default.html', 'w') as file:
    #         file.write(pstyle.format(self.sample_data).render())

    def test_bar_inverse(self):
        # found a bug when doing `value_frequency(series, sort_by_frequency=False)` with a series that had
        # a count of `0` for a category (i.e. category existed but not any values)
        test_data = self.sample_data
        test_data.insert(loc=1,
                         column='col_a_copy',
                         value=test_data['col_a'].copy())
        test_data.insert(loc=7,
                         column='bar_inverse',
                         value=test_data['col_f'].copy())

        with open(get_test_path() + '/test_files/pandas_style/inverse_bar.html', 'w') as file:
            table_html = test_data.style. \
                bar(subset=['col_a'], color='pink', vmin=2). \
                pipe(pstyle.bar_inverse, subset='col_a_copy', color='pink', min_value=2). \
                bar(subset=['col_f'], color='green'). \
                pipe(pstyle.bar_inverse, subset='bar_inverse', color='green'). \
                render()

            file.write(table_html)

    def test_all_styles(self):
        # found a bug when doing `value_frequency(series, sort_by_frequency=False)` with a series that had
        # a count of `0` for a category (i.e. category existed but not any values)
        test_data = self.credit_data
        test_data.loc[0:5, 'credit_amount'] = np.nan
        test_data.insert(loc=5,
                         column='credit_amount_copy',
                         value=test_data['credit_amount'].copy())
        with open(get_test_path() + '/test_files/pandas_style/all_styles.html', 'w') as file:
            table_html = test_data. \
                head(20). \
                pipe(pstyle.bar_inverse, subset='credit_amount_copy', color='gray'). \
                bar(subset=['credit_amount'], color='grey'). \
                render()
            file.write(table_html)
