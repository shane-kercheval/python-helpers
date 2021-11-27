import datetime
import unittest
# noinspection PyMethodMayBeStatic
from enum import unique, Enum, auto

import numpy as np
import pandas as pd

import helpsk.plot as hplot
import helpsk.validation as hval
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

        from helpsk.pandas import value_frequency
        value_frequency(series=test_series)

        check_plot(file_name=get_test_path() + '/test_files/plot/test_plot_value_frequency__sort_true.png',
                   plot_function=lambda: hplot.plot_value_frequency(series=test_series,
                                                                    sort_by_frequency=True))

        check_plot(file_name=get_test_path() + '/test_files/plot/test_plot_value_frequency__sort_false.png',
                   plot_function=lambda: hplot.plot_value_frequency(series=test_series,
                                                                    sort_by_frequency=False))

    def test_plot_correlation_heatmap(self):
        test_series = self.credit_data.copy()
        test_series.loc[0:10, 'credit_amount'] = np.nan
        test_series.loc[0:10, 'num_dependents'] = np.nan

        check_plot(file_name=get_test_path() + '/test_files/plot/test_plot_correlation_heatmap__default.png',
                   plot_function=lambda: hplot.plot_correlation_heatmap(test_series))

        check_plot(file_name=get_test_path() + '/test_files/plot/test_plot_correlation_heatmap__params.png',
                   plot_function=lambda: hplot.plot_correlation_heatmap(test_series, threshold=0.2,
                                                                        title='Credit Data: Correlations Above 0.2',
                                                                        figure_size=(6, 6),
                                                                        round_by=3,
                                                                        features_to_highlight=['credit_amount',
                                                                                               'num_dependents']))

    def test_plot_dodged_barchart(self):
        test_series = self.credit_data.copy()
        test_series.loc[0:10, 'credit_history'] = np.nan
        test_series.loc[5:15, 'own_telephone'] = np.nan

        original = test_series.copy()

        check_plot(file_name=get_test_path() + '/test_files/plot/plot_dodged_barchart__default.png',
                   plot_function=lambda: hplot.plot_dodged_barchart(dataframe=test_series,
                                                                    outer_column='own_telephone',
                                                                    inner_column='credit_history'))

        self.assertTrue(hval.dataframes_match([original, test_series]))

    def test_plot_dodged_barchart__boolean(self):
        """There is a bug where using a boolean series fails"""
        test_series = self.credit_data.copy()
        test_series['all_paid'] = test_series['credit_history'] == 'all paid'
        test_series.loc[0:10, 'all_paid'] = np.nan

        test_series['own_telephone'] = test_series['own_telephone'] == 'yes'
        test_series.loc[5:15, 'own_telephone'] = None

        original = test_series.copy()

        check_plot(file_name=get_test_path() + '/test_files/plot/plot_dodged_barchart__booleans.png',
                   plot_function=lambda: hplot.plot_dodged_barchart(dataframe=test_series,
                                                                    outer_column='own_telephone',
                                                                    inner_column='all_paid'))

        self.assertTrue(hval.dataframes_match([original, test_series]))

    def test_plot_histogram_with_categorical(self):
        test_series = self.credit_data.copy()
        test_series.loc[0:10, 'credit_history'] = np.nan
        test_series.loc[5:15, 'credit_amount'] = np.nan

        original = test_series.copy()

        check_plot(file_name=get_test_path() + '/test_files/plot/test_plot_histogram_with_categorical__default.png',
                   plot_function=lambda: hplot.plot_histogram_with_categorical(dataframe=test_series,
                                                                               numeric_column='credit_amount',
                                                                               categorical_column='credit_history'))

        self.assertTrue(hval.dataframes_match([original, test_series]))

    def test_plot_histogram_with_categorical(self):
        # bug found when no missing values are found in target because we added a <missing> category but
        # we get an error when calling reorder_categories with extra categories
        credit_data = self.credit_data.copy()
        hplot.plot_histogram_with_categorical(dataframe=credit_data,
                                              numeric_column='credit_amount',
                                              categorical_column='target')
