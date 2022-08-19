import os
import unittest
from enum import Enum, unique, auto
import datetime

import pandas as pd
import numpy as np

from helpsk import pandas as hp
from helpsk import validation as hv
from helpsk.utility import redirect_stdout_to_file
from helpsk.validation import iterables_are_equal, dataframes_match
from tests.helpers import get_data_credit, get_test_path, clean_formatted_dataframe


@unique
class TestEnum(Enum):
    __test__ = False
    VALUE_A = auto()
    VALUE_B = auto()


# noinspection PyMethodMayBeStatic
class TestPandas(unittest.TestCase):

    def helper_test_summary(self, file_name, summary_results):
        if os.path.isfile(file_name):
            os.remove(file_name)
        with redirect_stdout_to_file(file_name):
            hp.print_dataframe(summary_results)
        self.assertTrue(os.path.isfile(file_name))

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
                                    'col_j': [False, False, True, False],
                                    'col_k': [None, None, None, None],
                                    'col_l': [np.nan, np.nan, np.nan, np.nan]
                                    })
        sample_data.loc[0, 'col_c'] = np.nan
        sample_data['col_g'] = sample_data['col_b'].astype('category')
        cls.sample_data = sample_data

    def test_is_series_numeric(self):
        actual = self.sample_data.apply(hp.is_series_numeric)
        expected = {
            'col_a': True,
            'col_b': False,
            'col_c': False,
            'col_d': False,
            'col_e': True,
            'col_f': True,
            'col_h': False,
            'col_i': False,
            'col_j': False,
            'col_k': False,
            'col_l': True,
            'col_g': False
        }
        self.assertEqual(expected, actual.to_dict())

    def test_is_series_bool(self):
        self.assertTrue(hp.is_series_bool(series=pd.Series([True])))
        self.assertTrue(hp.is_series_bool(series=pd.Series([False])))
        self.assertTrue(hp.is_series_bool(series=pd.Series([True, False])))
        self.assertTrue(hp.is_series_bool(series=pd.Series([True, False, None])))
        self.assertTrue(hp.is_series_bool(series=pd.Series([True, False, np.nan])))
        self.assertTrue(hp.is_series_bool(series=pd.Series([None, True, False, None])))
        self.assertTrue(hp.is_series_bool(series=pd.Series([np.nan, True, False, np.nan])))

        self.assertFalse(hp.is_series_bool(series=pd.Series([], dtype=object)))
        self.assertFalse(hp.is_series_bool(series=pd.Series(['Whatever'])))
        self.assertFalse(hp.is_series_bool(series=pd.Series([True, False, 'Whatever'])))
        self.assertFalse(hp.is_series_bool(series=pd.Series([True, False, TestEnum.VALUE_A])))
        self.assertFalse(hp.is_series_bool(series=pd.Series([None])))
        self.assertFalse(hp.is_series_bool(series=pd.Series([np.nan])))

    def test_is_series_date(self):
        self.assertFalse(hp.is_series_date(self.sample_data['col_a']))
        self.assertFalse(hp.is_series_date(self.sample_data['col_b']))
        self.assertTrue(hp.is_series_date(self.sample_data['col_c']))
        self.assertTrue(hp.is_series_date(self.sample_data['col_d']))
        self.assertFalse(hp.is_series_date(self.sample_data['col_e']))
        self.assertFalse(hp.is_series_date(self.sample_data['col_f']))
        self.assertFalse(hp.is_series_date(self.sample_data['col_g']))
        self.assertFalse(hp.is_series_date(self.sample_data['col_h']))
        self.assertTrue(hp.is_series_date(self.sample_data['col_i']))
        self.assertFalse(hp.is_series_date(pd.Series(dtype=np.float64)))

    def test_is_series_date__non_numeric_index(self):
        data = self.sample_data.copy()
        data.index = ['A', 'B', 'C', 'D']
        self.assertEqual(hp.get_date_columns(data), ['col_c', 'col_d', 'col_i'])

    def test_is_series_string(self):
        actual = self.sample_data.apply(hp.is_series_string)
        expected = {
            'col_a': False,
            'col_b': True,
            'col_c': False,
            'col_d': False,
            'col_e': False,
            'col_f': False,
            'col_h': False,
            'col_i': False,
            'col_j': False,
            'col_k': False,
            'col_l': False,
            'col_g': False
        }
        self.assertEqual(expected, actual.to_dict())
        self.assertEqual(hp.get_string_columns(self.sample_data), ['col_b'])

        data = self.sample_data.copy()
        data.index = ['A', 'B', 'C', 'D']
        self.assertEqual(hp.get_string_columns(data), ['col_b'])

    def test_is_series_categorical(self):
        actual = self.sample_data.apply(hp.is_series_categorical)
        expected = {
            'col_a': False,
            'col_b': False,
            'col_c': False,
            'col_d': False,
            'col_e': False,
            'col_f': False,
            'col_h': False,
            'col_i': False,
            'col_j': False,
            'col_k': False,
            'col_l': False,
            'col_g': True
        }
        self.assertEqual(expected, actual.to_dict())
        self.assertEqual(hp.get_categorical_columns(self.sample_data), ['col_g'])

    def test_replace_all_bools_with_strings(self):
        results = hp.replace_all_bools_with_strings(series=pd.Series([True]))
        self.assertEqual(list(results), ['True'])
        results = hp.replace_all_bools_with_strings(series=pd.Series([False]))
        self.assertEqual(list(results), ['False'])
        results = hp.replace_all_bools_with_strings(series=pd.Series([True, False]))
        self.assertEqual(list(results), ['True', 'False'])
        results = hp.replace_all_bools_with_strings(series=pd.Series([True, False]).astype('category'))
        self.assertEqual(list(results), ['True', 'False'])
        results = hp.replace_all_bools_with_strings(series=pd.Series([True, False, None]))
        self.assertEqual(list(results), ['True', 'False', None])
        results = hp.replace_all_bools_with_strings(series=pd.Series([True, False, np.nan]))
        self.assertEqual(list(results), ['True', 'False', np.nan])
        results = hp.replace_all_bools_with_strings(series=pd.Series([None, True, False, None]))
        self.assertEqual(list(results), [None, 'True', 'False', None])
        results = hp.replace_all_bools_with_strings(series=pd.Series([np.nan, True, False, np.nan]))
        self.assertEqual(list(results), [np.nan, 'True', 'False', np.nan])

        results = hp.replace_all_bools_with_strings(series=pd.Series(['True']))
        self.assertEqual(list(results), ['True'])
        results = hp.replace_all_bools_with_strings(series=pd.Series(['False']))
        self.assertEqual(list(results), ['False'])
        results = hp.replace_all_bools_with_strings(series=pd.Series(['True', 'False']))
        self.assertEqual(list(results), ['True', 'False'])
        results = hp.replace_all_bools_with_strings(series=pd.Series(['True', 'False']).astype('category'))
        self.assertEqual(list(results), ['True', 'False'])
        results = hp.replace_all_bools_with_strings(series=pd.Series(['True', 'False', None]))
        self.assertEqual(list(results), ['True', 'False', None])
        results = hp.replace_all_bools_with_strings(series=pd.Series(['True', 'False', np.nan]))
        self.assertEqual(list(results), ['True', 'False', np.nan])
        results = hp.replace_all_bools_with_strings(series=pd.Series([None, 'True', 'False', None]))
        self.assertEqual(list(results), [None, 'True', 'False', None])
        results = hp.replace_all_bools_with_strings(series=pd.Series([np.nan, 'True', 'False', np.nan]))
        self.assertEqual(list(results), [np.nan, 'True', 'False', np.nan])

        results = hp.replace_all_bools_with_strings(series=pd.Series([], dtype=object))
        self.assertEqual(list(results), [])
        results = hp.replace_all_bools_with_strings(series=pd.Series(['Whatever']))
        self.assertEqual(list(results), ['Whatever'])
        results = hp.replace_all_bools_with_strings(series=pd.Series([True, False, 'Whatever']))
        self.assertEqual(list(results), ['True', 'False', 'Whatever'])
        results = hp.replace_all_bools_with_strings(series=pd.Series([True, False, TestEnum.VALUE_A]))
        self.assertEqual(list(results), ['True', 'False', TestEnum.VALUE_A])
        results = hp.replace_all_bools_with_strings(series=pd.Series([None]))
        self.assertEqual(list(results), [None])
        results = hp.replace_all_bools_with_strings(series=pd.Series([np.nan]))
        self.assertEqual(len(results), 1)

    def test_fill_na(self):
        # test non-categorical
        self.assertEqual(hp.fill_na(pd.Series([], dtype=object)).values.tolist(), [])
        self.assertEqual(hp.fill_na(pd.Series([None])).values.tolist(), ['<Missing>'])
        self.assertEqual(hp.fill_na(pd.Series([np.nan])).values.tolist(), ['<Missing>'])

        results = hp.fill_na(pd.Series([True, False, np.nan]))
        self.assertEqual(results.tolist(), [True, False, '<Missing>'])
        self.assertIsInstance(results[0], bool)

        # this fails if not converted to object
        results = hp.fill_na(pd.Series([True, False, np.nan]).astype('boolean'))
        self.assertEqual(results.tolist(), [True, False, '<Missing>'])
        self.assertIsInstance(results[0], bool)

        results = hp.fill_na(pd.Series([0, 1, np.nan]))
        self.assertEqual(results.tolist(), [0, 1, '<Missing>'])
        self.assertIsInstance(results[0], float)

        # this fails if not converted to object
        results = hp.fill_na(pd.Series([0, 1, np.nan]).astype('float'))
        self.assertEqual(results.tolist(), [0, 1, '<Missing>'])
        self.assertIsInstance(results[0], float)

        results = hp.fill_na(pd.Series(['A', 'B', np.nan]))
        self.assertEqual(results.tolist(), ['A', 'B', '<Missing>'])
        self.assertIsInstance(results[0], str)

        self.assertEqual(hp.fill_na(pd.Series(['A', 'B'])).values.tolist(), ['A', 'B'])

        # test categorical
        results = hp.fill_na(pd.Series([], dtype=object).astype('category'))
        self.assertEqual(results.tolist(), [])
        self.assertTrue(hp.is_series_categorical(results))
        self.assertTrue('<Missing>' in results.cat.categories)

        results = hp.fill_na(pd.Series([None]).astype('category'))
        self.assertEqual(results.tolist(), ['<Missing>'])
        self.assertTrue(hp.is_series_categorical(results))
        self.assertTrue('<Missing>' in results.cat.categories)

        results = hp.fill_na(pd.Series([np.nan]).astype('category'))
        self.assertEqual(results.tolist(), ['<Missing>'])
        self.assertTrue(hp.is_series_categorical(results))
        self.assertTrue('<Missing>' in results.cat.categories)

        results = hp.fill_na(pd.Series([True, False, np.nan]).astype('category'))
        self.assertEqual(results.tolist(), [True, False, '<Missing>'])
        self.assertTrue(hp.is_series_categorical(results))
        self.assertTrue('<Missing>' in results.cat.categories)

        results = hp.fill_na(pd.Series([0, 1, np.nan]).astype('category'))
        self.assertEqual(results.tolist(), [0, 1, '<Missing>'])
        self.assertTrue(hp.is_series_categorical(results))
        self.assertTrue('<Missing>' in results.cat.categories)

        results = hp.fill_na(pd.Series(['A', 'B', '<Missing>', np.nan]).astype('category'))
        self.assertEqual(results.tolist(), ['A', 'B', '<Missing>', '<Missing>'])
        self.assertTrue(hp.is_series_categorical(results))
        self.assertTrue('<Missing>' in results.cat.categories)

        results = hp.fill_na(pd.Series(['A', 'B', np.nan]).astype('category'))
        self.assertEqual(results.tolist(), ['A', 'B', '<Missing>'])
        self.assertTrue(hp.is_series_categorical(results))
        self.assertTrue('<Missing>' in results.cat.categories)

        results = hp.fill_na(pd.Series(['A', 'B']).astype('category'))
        self.assertEqual(results.tolist(), ['A', 'B'])
        self.assertTrue(hp.is_series_categorical(results))
        self.assertTrue('<Missing>' in results.cat.categories)

    def test_relocate(self):
        df = pd.DataFrame({
            'A': ['a', 'b', 'c'],
            'X': ['x', 'y', 'z'],
            '1': [1, 2, 3],
        })
        df_original = df.copy()

        self.assertRaises(AssertionError, lambda: hp.relocate(df=df, column='A'))
        self.assertRaises(AssertionError, lambda: hp.relocate(df=df, column='A', before='Z'))
        self.assertRaises(AssertionError, lambda: hp.relocate(df=df, column='Z', before='A'))

        def subtest_relocate(column, before, after, expected_order):
            new = hp.relocate(df=df, column=column, before=before, after=after)
            self.assertTrue(hv.dataframes_match([df_original, df]))
            self.assertEqual(new.columns.tolist(), expected_order)
            self.assertTrue((new['A'] == df_original['A']).all())
            self.assertTrue((new['X'] == df_original['X']).all())
            self.assertTrue((new['1'] == df_original['1']).all())

        subtest_relocate(column='A', before='X', after=None, expected_order=['A', 'X', '1'])
        subtest_relocate(column='A', before='1', after=None, expected_order=['X', 'A', '1'])
        subtest_relocate(column='X', before='A', after=None, expected_order=['X', 'A', '1'])
        subtest_relocate(column='X', before='1', after=None, expected_order=['A', 'X', '1'])
        subtest_relocate(column='1', before='A', after=None, expected_order=['1', 'A', 'X'])
        subtest_relocate(column='1', before='X', after=None, expected_order=['A', '1', 'X'])

        subtest_relocate(column='A', before=None, after='X', expected_order=['X', 'A', '1'])
        subtest_relocate(column='A', before=None, after='1', expected_order=['X', '1', 'A'])
        subtest_relocate(column='X', before=None, after='A', expected_order=['A', 'X', '1'])
        subtest_relocate(column='X', before=None, after='1', expected_order=['A', '1', 'X'])
        subtest_relocate(column='1', before=None, after='A', expected_order=['A', '1', 'X'])
        subtest_relocate(column='1', before=None, after='X', expected_order=['A', 'X', '1'])

    def test_get_numeric_columns(self):
        self.assertEqual(hp.get_numeric_columns(self.sample_data), ['col_a', 'col_e', 'col_f', 'col_l'])
        self.assertEqual(hp.get_numeric_columns(self.sample_data[['col_e']]), ['col_e'])
        self.assertEqual(hp.get_numeric_columns(self.sample_data[['col_d']]), [])
        self.assertEqual(hp.get_numeric_columns(pd.DataFrame()), [])

    def test_get_non_numeric_columns(self):
        non_numeric_columns = hp.get_non_numeric_columns(self.sample_data)
        non_numeric_columns.sort()
        self.assertEqual(non_numeric_columns,
                         ['col_b', 'col_c', 'col_d', 'col_g', 'col_h', 'col_i', 'col_j', 'col_k'])
        self.assertEqual(hp.get_non_numeric_columns(self.sample_data[['col_d']]), ['col_d'])
        self.assertEqual(hp.get_non_numeric_columns(self.sample_data[['col_e']]), [])
        self.assertEqual(hp.get_non_numeric_columns(pd.DataFrame()), [])

    def test_reorder_categories__occurrences(self):
        categorical = self.credit_data['purpose'].copy()
        categorical[0:50] = np.nan
        original_categorical = categorical.copy()

        original_categories = ['new car', 'used car', 'furniture/equipment', 'radio/tv',
                               'domestic appliance', 'repairs', 'education', 'vacation', 'retraining',
                               'business', 'other']

        expected_categories = list(categorical.value_counts(ascending=True, dropna=True).index.values)
        expected_categories_rev = expected_categories.copy()
        expected_categories_rev.reverse()

        # check that the purpose column hasn't changed and that it is not ordered and has original categories
        self.assertFalse(categorical.cat.ordered)
        self.assertEqual(list(categorical.cat.categories), original_categories)

        # test default weights
        # also, we are testing a Categorical object; verify it is categorical, then later test non-categorical
        self.assertEqual(categorical.dtype.name, 'category')
        results = hp.reorder_categories(categorical=categorical, ascending=True, ordered=False)
        self.assertTrue(results is not categorical)  # check that a new object was returned
        self.assertTrue(iterables_are_equal(results, original_categorical))
        self.assertEqual(list(results.categories), expected_categories)
        self.assertFalse(results.ordered)

        # check no side effects
        results[0] = 'new car'
        self.assertEqual(list(original_categorical.cat.categories), original_categories)
        self.assertTrue(pd.isna(original_categorical[0]))

        # test ascending=False & ordered=True
        results = hp.reorder_categories(categorical=categorical, ascending=False, ordered=True)
        self.assertTrue(results is not categorical)  # check that a new object was returned
        self.assertTrue(iterables_are_equal(results, original_categorical))
        self.assertEqual(list(results.categories), expected_categories_rev)
        self.assertTrue(results.ordered)

        # check no side effects
        results[0] = 'new car'
        self.assertEqual(list(categorical.cat.categories), list(original_categorical.cat.categories))
        self.assertTrue(iterables_are_equal(categorical, original_categorical))
        self.assertTrue(pd.isna(categorical[0]))

        series = pd.Series(['a', 'b', 'b', 'c', 'b', 'c'])
        results = hp.reorder_categories(categorical=series, ascending=False, ordered=True)
        self.assertTrue(results is not categorical)  # check that a new object was returned
        self.assertTrue(iterables_are_equal(results, series))
        self.assertEqual(list(results.categories), ['b', 'c', 'a'])
        self.assertTrue(results.ordered)

    def test_reorder_categories__weights(self):
        categorical = self.credit_data['purpose'].copy()
        categorical[0:50] = np.nan
        original_categorical = categorical.copy()
        weights = self.credit_data['credit_amount'].copy()
        weights[25:75] = np.nan

        original_categories = ['new car', 'used car', 'furniture/equipment', 'radio/tv',
                               'domestic appliance', 'repairs', 'education', 'vacation', 'retraining',
                               'business', 'other']

        expected_categories = ['vacation', 'retraining', 'domestic appliance', 'repairs', 'education',
                               'radio/tv', 'new car', 'furniture/equipment', 'business', 'used car', 'other']
        expected_categories_rev = expected_categories.copy()
        expected_categories_rev.reverse()

        with self.assertRaises(hv.HelpskParamValueError):
            hp.reorder_categories(categorical=categorical[0:-1], weights=weights)

        with self.assertRaises(hv.HelpskParamValueError):
            hp.reorder_categories(categorical=categorical, weights=weights[0:-1])

        # check that the purpose column hasn't changed and that it is not ordered and has original categories
        self.assertFalse(categorical.cat.ordered)
        self.assertEqual(list(categorical.cat.categories), original_categories)

        # test default weights
        # also, we are testing a Categorical object; verify it is categorical, then later test non-categorical
        self.assertEqual(categorical.dtype.name, 'category')
        results = hp.reorder_categories(
            categorical=categorical, weights=weights, ascending=True, ordered=False
        )
        self.assertTrue(results is not categorical)  # check that a new object was returned
        self.assertTrue(iterables_are_equal(results, original_categorical))
        self.assertEqual(list(results.categories), expected_categories)
        self.assertFalse(results.ordered)

        # check no side effects
        results[0] = 'new car'
        self.assertEqual(list(categorical.cat.categories), list(original_categorical.cat.categories))
        self.assertTrue(iterables_are_equal(categorical, original_categorical))
        self.assertTrue(pd.isna(categorical[0]))

        # test ascending=False & ordered=True
        results = hp.reorder_categories(
            categorical=categorical, weights=weights, ascending=False, ordered=True
        )
        self.assertTrue(results is not categorical)  # check that a new object was returned
        self.assertTrue(iterables_are_equal(results, original_categorical))
        self.assertEqual(list(results.categories), expected_categories_rev)
        self.assertTrue(results.ordered)

        # check no side effects
        results[0] = 'new car'
        self.assertEqual(list(categorical.cat.categories), list(original_categorical.cat.categories))
        self.assertTrue(iterables_are_equal(categorical, original_categorical))
        self.assertTrue(pd.isna(categorical[0]))

        series = pd.Series(['a', 'b', 'c'] * 2)
        weights = pd.Series([1, 3, 2] * 2)
        results = hp.reorder_categories(categorical=series, weights=weights, ascending=False, ordered=True)
        self.assertTrue(results is not categorical)  # check that a new object was returned
        self.assertTrue(iterables_are_equal(results, series))
        self.assertEqual(list(results.categories), ['b', 'c', 'a'])
        self.assertTrue(results.ordered)

    def test_top_n_categories__occurrences(self):
        categorical = self.credit_data['purpose'].copy()
        categorical[0:50] = np.nan
        original_categorical = categorical.copy()
        categorical_counts = categorical.value_counts(dropna=False)

        results = hp.top_n_categories(categorical=categorical)
        self.assertTrue(results is not categorical)  # check that a new object was returned
        self.assertEqual(list(results.categories), list(categorical_counts.index.values[0:5]) + ['Other'])
        self.assertEqual(results.value_counts(dropna=False).sum(), 1000)

        result_counts = results.value_counts().head(-1)
        expected_counts = categorical_counts.head(5)
        self.assertEqual(list(result_counts.index), list(expected_counts.index))
        self.assertEqual(list(result_counts.values), list(expected_counts.values))

        # check no side effects
        results[0] = 'new car'
        self.assertEqual(list(categorical.cat.categories), list(original_categorical.cat.categories))
        self.assertTrue(iterables_are_equal(categorical, original_categorical))
        self.assertTrue(pd.isna(categorical[0]))

        # if top_n is set to a higher number than there are categories, then we should get back an object
        # with the same values
        results = hp.top_n_categories(categorical=categorical, top_n=20)
        self.assertTrue(results is not categorical)  # check that a new object was returned
        self.assertTrue(set(results.cat.categories) == set(categorical.cat.categories))  # noqa
        self.assertTrue(iterables_are_equal(results, categorical))

        # check no side effects
        results[0] = 'new car'
        self.assertEqual(list(categorical.cat.categories), list(original_categorical.cat.categories))
        self.assertTrue(iterables_are_equal(categorical, original_categorical))
        self.assertTrue(pd.isna(categorical[0]))

        results = hp.top_n_categories(
            categorical=categorical,
            top_n=3,
            other_category='Some others',
            ordered=True)
        self.assertTrue(results is not categorical)  # check that a new object was returned
        self.assertEqual(list(results.categories),
                         list(categorical_counts.index.values[0:3]) + ['Some others'])
        self.assertEqual(results.value_counts(dropna=False).sum(), 1000)

        result_counts = results.value_counts().head(-1)
        expected_counts = categorical_counts.head(3)
        self.assertEqual(list(result_counts.index), list(expected_counts.index))
        self.assertEqual(list(result_counts.values), list(expected_counts.values))

        # check no side effects
        results[0] = 'new car'
        self.assertEqual(list(categorical.cat.categories), list(original_categorical.cat.categories))
        self.assertTrue(iterables_are_equal(categorical, original_categorical))
        self.assertTrue(pd.isna(categorical[0]))

    def test_top_n_categories__weights(self):
        categorical = self.credit_data['purpose'].copy()
        categorical[0:50] = np.nan
        original_categorical = categorical.copy()
        weights = pd.Series([-1] * len(categorical))
        # ascending True because we are going sum by negative -1 so categories with least amount of
        # occurrences will be in the "top" categories i.e. -1 > -100
        # now we expect our top_n to be the lowest frequency of categories
        categorical_counts = categorical.value_counts(dropna=False, ascending=True)

        with self.assertRaises(hv.HelpskParamValueError):
            hp.top_n_categories(categorical=categorical[0:-1], weights=weights)

        with self.assertRaises(hv.HelpskParamValueError):
            hp.top_n_categories(categorical=categorical, weights=weights[0:-1])

        results = hp.top_n_categories(categorical=categorical, weights=weights, weight_function=np.sum)
        self.assertTrue(results is not categorical)  # check that a new object was returned
        self.assertEqual(list(results.categories), list(categorical_counts.index.values[0:5]) + ['Other'])
        self.assertEqual(results.value_counts(dropna=False).sum(), 1000)

        result_counts = results.value_counts().head(-1)
        expected_counts = categorical_counts.head(5)
        self.assertEqual(list(result_counts.index), list(expected_counts.index))
        self.assertEqual(list(result_counts.values), list(expected_counts.values))

        # check no side effects
        results[0] = 'vacation'
        self.assertEqual(list(categorical.cat.categories), list(original_categorical.cat.categories))
        self.assertTrue(iterables_are_equal(categorical, original_categorical))
        self.assertTrue(pd.isna(categorical[0]))

        # if top_n is set to a higher number than there are categories, then we should get back an object
        # with the same values
        results = hp.top_n_categories(
            categorical=categorical,
            top_n=20,
            weights=weights,
            weight_function=np.sum
        )
        self.assertTrue(results is not categorical)  # check that a new object was returned
        self.assertTrue(set(results.cat.categories) == set(categorical.cat.categories))  # noqa
        self.assertTrue(iterables_are_equal(results, categorical))

        # check no side effects
        results[0] = 'vacation'
        self.assertEqual(list(categorical.cat.categories), list(original_categorical.cat.categories))
        self.assertTrue(iterables_are_equal(categorical, original_categorical))
        self.assertTrue(pd.isna(categorical[0]))

        results = hp.top_n_categories(
            categorical=categorical, top_n=3, other_category='Some others',
            ordered=True, weights=weights, weight_function=np.sum
        )
        self.assertTrue(results is not categorical)  # check that a new object was returned
        self.assertEqual(list(results.categories),
                         list(categorical_counts.index.values[0:3]) + ['Some others'])
        self.assertEqual(results.value_counts(dropna=False).sum(), 1000)

        result_counts = results.value_counts().head(-1)
        expected_counts = categorical_counts.head(3)
        self.assertEqual(list(result_counts.index), list(expected_counts.index))
        self.assertEqual(list(result_counts.values), list(expected_counts.values))

        # check no side effects
        results[0] = 'vacation'
        self.assertEqual(list(categorical.cat.categories), list(original_categorical.cat.categories))
        self.assertTrue(iterables_are_equal(categorical, original_categorical))
        self.assertTrue(pd.isna(categorical[0]))

    def test_numeric_summary(self):
        self.helper_test_summary(
            get_test_path('pandas/test_numeric_summary__credit.txt'),
            hp.numeric_summary(self.credit_data, return_style=False, sort_by_columns=False)
        )

        self.helper_test_summary(
            get_test_path('pandas/test_numeric_summary__credit__sorted.txt'),
            hp.numeric_summary(self.credit_data, return_style=False, sort_by_columns=True)
        )

        self.helper_test_summary(
            get_test_path('pandas/test_numeric_summary__sample.txt'),
            hp.numeric_summary(self.sample_data, return_style=False, sort_by_columns=False)
        )

        self.helper_test_summary(
            get_test_path('pandas/test_numeric_summary__sample__sorted.txt'),
            hp.numeric_summary(self.sample_data, return_style=False, sort_by_columns=True)
        )

    def test_numeric_summary__nan_column(self):
        test_data = self.credit_data.copy()
        test_data['all_missing'] = np.nan
        self.helper_test_summary(
            get_test_path('pandas/test_numeric_summary__credit__all_missing.txt'),
            hp.numeric_summary(test_data, return_style=False)
        )

        self.helper_test_summary(
            get_test_path('pandas/test_numeric_summary__style__credit__all_missing.html'),
            clean_formatted_dataframe(hp.numeric_summary(test_data, return_style=True, sort_by_columns=False).to_html())  # noqa
        )

        self.helper_test_summary(
            get_test_path('pandas/test_numeric_summary__style__credit__all_missing__sorted.html'),  # noqa
            clean_formatted_dataframe(hp.numeric_summary(test_data, return_style=True, sort_by_columns=True).to_html())  # noqa
        )

    def test_numeric_summary_style(self):
        test_data = self.credit_data.copy()
        test_data.loc[0:46, ['duration']] = np.nan
        test_data.loc[10:54, ['credit_amount']] = 0

        self.helper_test_summary(
            get_test_path('pandas/test_numeric_summary__style__credit.html'),
            clean_formatted_dataframe(hp.numeric_summary(test_data, return_style=True, sort_by_columns=False).to_html())  # noqa
        )

        self.helper_test_summary(
            get_test_path('pandas/test_numeric_summary__style__credit__sorted.html'),
            clean_formatted_dataframe(hp.numeric_summary(test_data, return_style=True, sort_by_columns=True).to_html())  # noqa
        )

        self.helper_test_summary(
            get_test_path('pandas/test_numeric_summary__style__sample.html'),
            clean_formatted_dataframe(hp.numeric_summary(self.sample_data, return_style=True).to_html())  # noqa
        )

    def test_non_numeric_summary(self):
        credit_data = self.credit_data.copy()
        credit_data['purpose'] = credit_data['purpose']. \
            replace({'radio/tv': '1111111111222222222233333333334444444444'})

        self.helper_test_summary(
            get_test_path('pandas/test_non_numeric_summary__credit.txt'),
            hp.non_numeric_summary(credit_data, return_style=False, sort_by_columns=False)
        )

        self.helper_test_summary(
            get_test_path('pandas/test_non_numeric_summary__credit__sorted.txt'),
            hp.non_numeric_summary(credit_data, return_style=False, sort_by_columns=True)
        )

        self.helper_test_summary(
            get_test_path('pandas/test_non_numeric_summary__sample.txt'),
            hp.non_numeric_summary(self.sample_data, return_style=False, sort_by_columns=False)
        )

        self.helper_test_summary(
            get_test_path('pandas/test_non_numeric_summary__sample__sorted.txt'),
            hp.non_numeric_summary(self.sample_data, return_style=False, sort_by_columns=True)
        )

    def test_non_numeric_summary__nan_column(self):
        test_data = self.credit_data.copy()
        test_data['purpose'] = test_data['purpose']. \
            replace({'radio/tv': '1111111111222222222233333333334444444444'})
        test_data['all_missing'] = None
        self.helper_test_summary(
            get_test_path('pandas/test_non_numeric_summary__credit__all_missing.txt'),
            hp.non_numeric_summary(test_data, return_style=False, sort_by_columns=False)
        )

        self.helper_test_summary(
            get_test_path('pandas/test_non_numeric_summary__credit__all_missing__sorted.txt'),
            hp.non_numeric_summary(test_data, return_style=False, sort_by_columns=True)
        )

        self.helper_test_summary(
            get_test_path('pandas/test_non_numeric_summary__style__credit__all_missing.html'),
            clean_formatted_dataframe(hp.non_numeric_summary(test_data, return_style=True, sort_by_columns=False).to_html())  # noqa
        )

        self.helper_test_summary(
            get_test_path('pandas/test_non_numeric_summary__style__credit__all_missing__sorted.html'),  # noqa
            clean_formatted_dataframe(hp.non_numeric_summary(test_data, return_style=True, sort_by_columns=True).to_html())  # noqa
        )

    def test_non_numeric_summary__list_column(self):
        test_data = pd.DataFrame({
            'list_column': [['a', 'b'], [1, 2], []],
            'list_column2': [['a', 'b'], [1, 2], np.nan],
        })

        self.helper_test_summary(
            get_test_path('pandas/test_non_numeric_summary__list_column.txt'),
            hp.non_numeric_summary(test_data, return_style=False, sort_by_columns=False)
        )

    def test_non_numeric_summary2(self):
        test_data = self.credit_data.copy()
        test_data['purpose'] = test_data['purpose']. \
            replace({'radio/tv': '1111111111222222222233333333334444444444'})
        test_data.loc[25:75, ['checking_status']] = np.nan

        self.helper_test_summary(
            get_test_path('pandas/test_non_numeric_summary__style__credit.html'),
            clean_formatted_dataframe(hp.non_numeric_summary(test_data, return_style=True, sort_by_columns=False).to_html())  # noqa
        )

        self.helper_test_summary(
            get_test_path('pandas/test_non_numeric_summary__style__credit__sorted.html'),
            clean_formatted_dataframe(hp.non_numeric_summary(test_data, return_style=True, sort_by_columns=True).to_html())  # noqa
        )

        self.helper_test_summary(
            get_test_path('pandas/test_non_numeric_summary__style__sample.html'),
            clean_formatted_dataframe(hp.non_numeric_summary(self.sample_data, return_style=True).to_html())
        )

    def test_convert_integer_series_to_categorical(self):
        # check that function fails if mapping doesn't contains all numbers in data
        with self.assertRaises(hv.HelpskParamValueError):
            mapping = {2: 'a', 3: 'b'}
            hp.convert_integer_series_to_categorical(series=self.sample_data.col_a, mapping=mapping)

        with self.assertRaises(hv.HelpskParamValueError):
            mapping = {2: 'a'}
            hp.convert_integer_series_to_categorical(series=self.sample_data.col_a, mapping=mapping)

        with self.assertRaises(hv.HelpskParamValueError):
            mapping = {}
            hp.convert_integer_series_to_categorical(series=self.sample_data.col_a, mapping=mapping)

        # check that series will work even if it doesn't have all the values in mapping
        mapping = {1: 'value not found', 2: 'a', 3: 'b', 4: 'c'}
        result = hp.convert_integer_series_to_categorical(
            series=self.sample_data.col_a,
            mapping=mapping,
            ordered=False
        )
        self.assertTrue(hv.iterables_are_equal(result.cat.categories, mapping.values()))
        self.assertTrue(hv.iterables_are_equal(list(result.values), [np.nan, 'a', 'b', 'c']))
        # ensure no side effects on list passed in
        self.assertTrue(hv.iterables_are_equal(self.sample_data.col_a, [np.nan, 2, 3, 4]))
        self.assertFalse(result.cat.ordered)

        # same thing but with ordered Categorical
        mapping = {1: 'not found', 2: 'a', 3: 'b', 4: 'c'}
        result = hp.convert_integer_series_to_categorical(
            series=self.sample_data.col_a,
            mapping=mapping,
            ordered=True
        )
        self.assertTrue(hv.iterables_are_equal(result.cat.categories, mapping.values()))
        self.assertTrue(hv.iterables_are_equal(list(result.values), [np.nan, 'a', 'b', 'c']))
        # ensure no side effects on list passed in
        self.assertTrue(hv.iterables_are_equal(self.sample_data.col_a, [np.nan, 2, 3, 4]))
        self.assertTrue(result.cat.ordered)

    def test_value_frequency(self):
        credit_history_order = ['delayed previously',
                                'critical/other existing credit',
                                'existing paid',
                                'no credits/all paid',
                                'all paid']
        credit_history_alphabetical = credit_history_order.copy()
        credit_history_alphabetical.sort()

        test_series_unordered = self.credit_data['credit_history'].copy()
        self.assertFalse(test_series_unordered.cat.ordered)
        test_series_unordered[0:10] = np.nan

        test_series_ordered = test_series_unordered.cat.reorder_categories(credit_history_order, ordered=True)
        self.assertTrue(test_series_ordered.cat.ordered)

        results = hp.value_frequency(test_series_unordered, sort_by_frequency=True)
        expected_value_order = test_series_unordered.value_counts(dropna=False).index.tolist()
        self.assertEqual(results.index.tolist(), expected_value_order)
        self.assertTrue((results['Frequency'] == test_series_unordered.value_counts(dropna=False).values).all())  # noqa
        self.assertTrue((results['Percent'] == test_series_unordered.value_counts(normalize=True, dropna=False).values).all())  # noqa

        # cache the verified results
        cached_results = results

        # test unordered categorical - sort_by_frequency=False
        results = hp.value_frequency(test_series_unordered, sort_by_frequency=False)
        self.assertEqual(results.index.tolist(), credit_history_alphabetical + [np.nan])
        self.assertEqual(
            cached_results.loc[results.index.tolist(), 'Frequency'].tolist(),
            results['Frequency'].tolist()
        )
        self.assertTrue(cached_results.loc[results.index.tolist(), 'Percent'].tolist() == results['Percent'].tolist())  # noqa

        # test ordered categorical - sort_by_frequency=False
        results = hp.value_frequency(test_series_ordered, sort_by_frequency=False)
        self.assertEqual(results.index.tolist(), credit_history_order + [np.nan])
        self.assertTrue(cached_results.loc[results.index.tolist(), 'Frequency'].tolist() == results['Frequency'].tolist())  # noqa
        self.assertTrue(cached_results.loc[results.index.tolist(), 'Percent'].tolist() == results['Percent'].tolist())  # noqa

        # test ordered categorical - sort_by_frequency=True
        results = hp.value_frequency(test_series_ordered, sort_by_frequency=True)
        self.assertEqual(results.index.tolist(), cached_results.index.tolist())
        self.assertTrue(cached_results['Frequency'].tolist() == results['Frequency'].tolist())
        self.assertTrue(cached_results['Percent'].tolist() == results['Percent'].tolist())
        self.assertTrue(hv.dataframes_match([cached_results, results]))

        # test numeric
        test_series_float = self.credit_data['duration'].copy()
        test_series_float[0:10] = np.nan

        results = hp.value_frequency(test_series_float, sort_by_frequency=True)
        expected_value_counts = test_series_float.value_counts(normalize=False, dropna=False)
        self.assertTrue(hv.iterables_are_equal(results.index.values, expected_value_counts.index.values))
        self.assertTrue(hv.iterables_are_equal(results['Frequency'].values, expected_value_counts.values))

        cached_results = results

        results = hp.value_frequency(test_series_float, sort_by_frequency=False)
        expected_indexes = test_series_float.dropna().unique()
        expected_indexes.sort()
        expected_indexes = expected_indexes.tolist() + [np.nan]

        self.assertEqual(results.index.values.tolist()[0:-1], expected_indexes[0:-1])
        self.assertTrue(hv.iterables_are_equal(results.index.values.tolist(), expected_indexes))  # noqa
        self.assertTrue(hv.iterables_are_equal(results['Frequency'].values, cached_results.loc[results.index.values, 'Frequency'].values))  # noqa
        self.assertTrue(hv.iterables_are_equal(results['Percent'].values, cached_results.loc[results.index.values, 'Percent'].values))  # noqa

    def test_value_frequency__missing_category(self):
        # found a bug when doing `value_frequency(series, sort_by_frequency=False)` with a series that had
        # a count of `0` for a category (i.e. category existed but not any values)
        test_series = self.credit_data['checking_status'].copy()
        test_series[test_series == 'no checking'] = np.nan

        results = hp.value_frequency(test_series, sort_by_frequency=True)
        expected_value_counts = test_series.value_counts(normalize=False, dropna=False)

        self.assertTrue(hv.iterables_are_equal(results.index.values, expected_value_counts.index.values))
        self.assertTrue(hv.iterables_are_equal(results['Frequency'].values, expected_value_counts.values))

        cached_results = results

        results = hp.value_frequency(test_series, sort_by_frequency=False)
        expected_indexes = ['0<=X<200', '<0', '>=200', 'no checking', np.nan]

        self.assertEqual(results.index.values.tolist()[0:-1], expected_indexes[0:-1])
        self.assertTrue(hv.iterables_are_equal(results.index.values.tolist(), expected_indexes))  # noqa
        self.assertTrue(hv.iterables_are_equal(results['Frequency'].values, cached_results.loc[results.index.values, 'Frequency'].values))  # noqa
        self.assertTrue(hv.iterables_are_equal(results['Percent'].values, cached_results.loc[results.index.values, 'Percent'].values))  # noqa

    def test_count_groups(self):
        data = self.credit_data.copy()
        data.loc[:, 'target'] = data['target'] == 'good'
        data.loc[0:10, 'target'] = np.nan
        data.loc[:, 'target'] = data['target'].astype('boolean')
        data.loc[9:20, 'checking_status'] = np.nan
        data.loc[19:30, 'credit_amount'] = np.nan

        # change dataset so that there is an entire category missing (e.g. no target == True and checking
        # status `<Missing>`
        indexes_to_blank = (data['target'] == True) & (data['checking_status'] == '0<=X<200')  # noqa
        data.loc[indexes_to_blank, 'checking_status'] = np.nan
        # change dataset so there is an entire category with no sum_by
        indexes_to_blank = (data['target'] == False) & (data['checking_status'] == 'no checking')  # noqa
        data.loc[indexes_to_blank, 'credit_amount'] = np.nan

        original = self.credit_data.copy()
        result = hp.count_groups(
            dataframe=original,
            group_1='target',
            group_2='housing',
            group_sum='credit_amount',
            return_style=False
        )
        self.assertTrue(dataframes_match([original, self.credit_data]))
        file_name = get_test_path('pandas/count_groups__credit.txt')
        with redirect_stdout_to_file(file_name):
            hp.print_dataframe(result)

        self.assertEqual(result[('target', 'target')].dropna().tolist(), ['good', 'bad'])
        self.assertEqual(result[('target', 'Count')].sum(), 1000)
        self.assertEqual(result[('target', 'Count Perc')].sum(), 1)
        self.assertEqual(result[('target', 'Sum')].sum(), self.credit_data['credit_amount'].sum())
        self.assertEqual(result[('target', 'Sum Perc')].sum(), 1)
        self.assertEqual(result[('housing', 'Count')].sum(), 1000)
        self.assertEqual(result[('housing', 'Count Perc')].sum(), 2)
        self.assertEqual(result[('housing', 'Sum')].sum(), self.credit_data['credit_amount'].sum())
        self.assertTrue(hv.is_close(result[('housing', 'Sum Perc')].sum(), 2))

        def test_count_groups(group_1, group_2=None, group_sum=None, remove_first_level_duplicates=False):
            counts = hp.count_groups(
                dataframe=data,
                group_1=group_1,
                group_2=group_2,
                group_sum=group_sum,
                remove_first_level_duplicates=remove_first_level_duplicates,
                return_style=False
            )

            name = get_test_path(
                f'pandas/count_groups_{group_1}_{group_2}_'
                f'{group_sum}_{remove_first_level_duplicates}.txt'
            )
            with redirect_stdout_to_file(name):
                hp.print_dataframe(counts)
            return counts

        results_1 = test_count_groups(group_1='target', remove_first_level_duplicates=False)
        self.assertEqual(results_1[('target', 'Count')].sum(),  data.shape[0])
        self.assertTrue(hv.is_close(results_1[('target', 'Count Perc')].sum(), 1))

        results = test_count_groups(group_1='target', remove_first_level_duplicates=True)
        self.assertTrue(dataframes_match([results, results_1]))

        results_2 = test_count_groups(group_1='checking_status', remove_first_level_duplicates=False)
        self.assertEqual(results_2[('checking_status', 'Count')].sum(), data.shape[0])
        self.assertTrue(hv.is_close(results_2[('checking_status', 'Count Perc')].sum(), 1))

        results = test_count_groups(group_1='checking_status', remove_first_level_duplicates=True)
        self.assertTrue(dataframes_match([results, results_2]))

        results_3 = test_count_groups(
            group_1='target',
            group_2='checking_status',
            remove_first_level_duplicates=False
        )
        self.assertTrue(dataframes_match([
            results_1,
            results_3[[('target', 'target'), ('target', 'Count'), ('target', 'Count Perc')]].drop_duplicates()  # noqa
        ]))

        self.assertEqual(results_3[('checking_status', 'Count')].sum(), data.shape[0])

        group_2_count_sum = results_3.groupby(results_3[('target', 'target')]). \
            agg({('checking_status', 'Count'): sum})

        self.assertTrue((results_1[('target', 'Count')].values ==   # noqa
                         group_2_count_sum[('checking_status', 'Count')].values).all())
        self.assertEqual(len(group_2_count_sum), 3)

        group_2_count_perc_sum = results_3.groupby(results_3[('target', 'target')]).\
            agg({('checking_status', 'Count Perc'): sum})
        self.assertEqual(len(group_2_count_perc_sum), 3)
        self.assertTrue((group_2_count_perc_sum[('checking_status', 'Count Perc')] == 1).all())

        results = test_count_groups(
            group_1='target',
            group_2='checking_status',
            remove_first_level_duplicates=True
        )
        self.assertTrue(dataframes_match([
            results_3[[('target', 'target'),
                       ('target', 'Count'),
                       ('target', 'Count Perc')]].astype(object).drop_duplicates(),
            results[[('target', 'target'),
                     ('target', 'Count'),
                     ('target', 'Count Perc')]].dropna().astype(object)
        ]))
        self.assertTrue(dataframes_match([
            results_3[[('checking_status', 'checking_status'),
                       ('checking_status', 'Count'),
                       ('checking_status', 'Count Perc')]],
            results[[('checking_status', 'checking_status'),
                     ('checking_status', 'Count'),
                     ('checking_status', 'Count Perc')]]
        ]))

        results = test_count_groups(
            group_1='target',
            group_sum='credit_amount',
            remove_first_level_duplicates=False
        )
        self.assertTrue(dataframes_match([
            results[[('target', 'target'),
                     ('target', 'Count'),
                     ('target', 'Count Perc')]].astype(object),
            results_1.astype(object)
        ]))
        self.assertEqual(results[('target', 'Sum')].sum(), data['credit_amount'].sum())
        self.assertEqual(results[('target', 'Sum Perc')].sum(), 1)

        results_x = test_count_groups(
            group_1='target',
            group_sum='credit_amount',
            remove_first_level_duplicates=True
        )
        self.assertTrue(dataframes_match([results_x, results]))

        results = test_count_groups(
            group_1='checking_status',
            group_sum='credit_amount',
            remove_first_level_duplicates=False
        )
        self.assertTrue(dataframes_match([
            results[[('checking_status', 'checking_status'),
                     ('checking_status', 'Count'),
                     ('checking_status', 'Count Perc')]].astype(object),
            results_2.astype(object)
        ]))
        self.assertEqual(results[('checking_status', 'Sum')].sum(), data['credit_amount'].sum())
        self.assertEqual(results[('checking_status', 'Sum Perc')].sum(), 1)

        results_y = test_count_groups(
            group_1='checking_status',
            group_sum='credit_amount',
            remove_first_level_duplicates=True
        )
        self.assertTrue(dataframes_match([results_y, results]))

        results = test_count_groups(
            group_1='target',
            group_2='checking_status',
            group_sum='credit_amount',
            remove_first_level_duplicates=False
        )
        self.assertTrue(dataframes_match([
            results_3.astype(object),
            results[[
                ('target', 'target'),
                ('target', 'Count'),
                ('target', 'Count Perc'),
                ('checking_status', 'checking_status'),
                ('checking_status', 'Count'),
                ('checking_status', 'Count Perc')
            ]].astype(object)
        ]))
        group_2_sum_perc_sum = results.groupby(results[('target', 'target')]). \
            agg({('checking_status', 'Sum Perc'): sum})
        self.assertEqual(len(group_2_sum_perc_sum), 3)
        self.assertTrue((group_2_sum_perc_sum[('checking_status', 'Sum Perc')].values == 1).all())

        results = test_count_groups(
            group_1='target',
            group_2='checking_status',
            group_sum='credit_amount',
            remove_first_level_duplicates=True
        )
        self.assertEqual(results[('checking_status', 'Sum')].sum(), data['credit_amount'].sum())
        self.assertEqual(results[('checking_status', 'Sum Perc')].sum(), 3)

        results = hp.count_groups(
            dataframe=data,
            group_1='target',
            group_2='checking_status',
            group_sum='credit_amount',
            return_style=True
        )
        with open(get_test_path('pandas/count_groups.html'), 'w') as file:
            file.write(clean_formatted_dataframe(results.to_html()))

        # test all of group 1 missing group_sum
        data.loc[data['target'] == True, 'credit_amount'] = np.nan  # noqa
        results = hp.count_groups(
            dataframe=data,
            group_1='target',
            group_2='checking_status',
            group_sum='credit_amount',
            return_style=False
        )
        target_counts = results[[('target', 'target'),
                                 ('target', 'Count'),
                                 ('target', 'Count Perc')]].dropna()
        self.assertTrue(dataframes_match([target_counts.astype(object), results_1.astype(object)]))
        self.assertTrue(hv.is_close(results[('target', 'Sum')].sum(), data['credit_amount'].sum()))
        self.assertTrue(hv.is_close(results[('target', 'Sum Perc')].sum(), 1))
        self.assertTrue((results.loc[results[('target', 'target')] == True, ('target', 'Sum Perc')] == 0).all())  # noqa

        results = hp.count_groups(
            dataframe=data,
            group_1='target',
            group_2='checking_status',
            group_sum='credit_amount',
            return_style=True
        )
        with open(get_test_path('pandas/count_groups_group_1_na_sum.html'), 'w') as file:
            file.write(clean_formatted_dataframe(results.to_html()))
