import datetime
import os
import unittest


# noinspection PyMethodMayBeStatic
from enum import unique, Enum, auto

import numpy as np
import pandas as pd

from helpsk.plot import plot_value_frequency
from tests.helpers import get_data_credit, check_plot, get_test_path


@unique
class TestEnum(Enum):
    VALUE_A = auto()
    VALUE_B = auto()


class TestPlot(unittest.TestCase):

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

    def test_plot_value_frequency(self):
        test_series = self.credit_data['purpose'].copy()
        test_series[0:10] = np.nan

        check_plot(file_name=get_test_path() + '/plot/test_plot_value_frequency.png',
                   plot_function=lambda: plot_value_frequency(series=test_series, sort_by_frequency=True, figure_size=(10, 7)))
