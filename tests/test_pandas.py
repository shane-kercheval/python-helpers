import os
import unittest
from enum import Enum, unique, auto

from helpsk.pandas import *
from helpsk.utility import redirect_stdout_to_file
from tests.helpers import get_data_credit, get_test_path


@unique
class TestEnum(Enum):
    VALUE_A = auto()
    VALUE_B = auto()



# noinspection PyMethodMayBeStatic
class TestPandas(unittest.TestCase):

    def helper_test_summary(self, file_name, summary_results):
        if os.path.isfile(file_name):
            os.remove(file_name)
        with redirect_stdout_to_file(file_name):
            print_dataframe(summary_results)
        self.assertTrue(os.path.isfile(file_name))

    @classmethod
    def setUpClass(cls):
        cls.credit_data = get_data_credit()
        sample_data = pd.DataFrame({'col_a': [np.nan, 2, 3, 4],
                                    'col_b': [np.nan, 'b', 'c', 'd'],
                                    'col_c': pd.date_range('2021-01-01', '2021-01-04'),
                                    'col_d': [None, np.nan, datetime.date.today(), datetime.date.today()],
                                    'col_e': np.nan,
                                    'col_f': [1.0, 2.0, 3.0, 4.0],
                                    'col_h': [np.nan, TestEnum.VALUE_A, TestEnum.VALUE_B, np.nan],
                                    'col_i': [None, np.nan, datetime.datetime.now(), datetime.datetime.now()],
                                    })
        sample_data.loc[0, 'col_c'] = np.nan
        sample_data['col_g'] = sample_data['col_b'].astype('category')
        cls.sample_data = sample_data

    def test_get_numeric_columns(self):
        self.assertEqual(get_numeric_columns(self.sample_data), ['col_a', 'col_e', 'col_f'])
        self.assertEqual(get_numeric_columns(self.sample_data[['col_e']]), ['col_e'])
        self.assertEqual(get_numeric_columns(self.sample_data[['col_d']]), [])
        self.assertEqual(get_numeric_columns(pd.DataFrame()), [])

    def test_get_non_numeric_columns(self):
        non_numeric_columns = get_non_numeric_columns(self.sample_data)
        non_numeric_columns.sort()
        self.assertEqual(non_numeric_columns,
                         ['col_b', 'col_c', 'col_d', 'col_g', 'col_h', 'col_i'])
        self.assertEqual(get_non_numeric_columns(self.sample_data[['col_d']]), ['col_d'])
        self.assertEqual(get_non_numeric_columns(self.sample_data[['col_e']]), [])
        self.assertEqual(get_non_numeric_columns(pd.DataFrame()), [])

    def test_is_series_date(self):
        self.assertFalse(is_series_date(self.sample_data['col_a']))
        self.assertFalse(is_series_date(self.sample_data['col_b']))
        self.assertTrue(is_series_date(self.sample_data['col_c']))
        self.assertTrue(is_series_date(self.sample_data['col_d']))
        self.assertFalse(is_series_date(self.sample_data['col_e']))
        self.assertFalse(is_series_date(self.sample_data['col_f']))
        self.assertFalse(is_series_date(self.sample_data['col_g']))
        self.assertFalse(is_series_date(self.sample_data['col_h']))
        self.assertTrue(is_series_date(self.sample_data['col_i']))
        self.assertFalse(is_series_date(pd.Series()))

    def test_numeric_summary(self):
        self.helper_test_summary(get_test_path() + '/test_files/test_numeric_summary__credit.txt',
                                 numeric_summary(self.credit_data))

        self.helper_test_summary(get_test_path() + '/test_files/test_numeric_summary__sample.txt',
                                 numeric_summary(self.sample_data))

    def test_non_numeric_summary(self):
        self.helper_test_summary(get_test_path() + '/test_files/test_non_numeric_summary__credit.txt',
                                 non_numeric_summary(self.credit_data))

        self.helper_test_summary(get_test_path() + '/test_files/test_non_numeric_summary__sample.txt',
                                 non_numeric_summary(self.sample_data))
