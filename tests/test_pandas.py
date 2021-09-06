import os
import unittest
from enum import Enum, unique, auto

from helpsk import validation as hv
from helpsk.pandas import *
from helpsk.utility import redirect_stdout_to_file
from helpsk.validation import iterables_are_equal
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
        self.assertFalse(is_series_date(pd.Series(dtype=np.float64)))

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
        results = reorder_categories(categorical=categorical, ascending=True, ordered=False)
        self.assertTrue(iterables_are_equal(results, original_categorical))
        self.assertEqual(list(results.categories), expected_categories)
        self.assertFalse(results.ordered)

        # check no side effects
        results[0] = 'new car'
        self.assertEqual(list(original_categorical.cat.categories), original_categories)
        self.assertTrue(pd.isna(original_categorical[0]))

        # test ascending=False & ordered=True
        results = reorder_categories(categorical=categorical, ascending=False, ordered=True)
        self.assertTrue(iterables_are_equal(results, original_categorical))
        self.assertEqual(list(results.categories), expected_categories_rev)
        self.assertTrue(results.ordered)

        # check no side effects
        results[0] = 'new car'
        self.assertEqual(list(categorical.cat.categories), list(original_categorical.cat.categories))
        self.assertTrue(iterables_are_equal(categorical, original_categorical))
        self.assertTrue(pd.isna(categorical[0]))

        series = pd.Series(['a', 'b', 'b', 'c', 'b', 'c'])
        results = reorder_categories(categorical=series, ascending=False, ordered=True)
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

        with self.assertRaises(HelpskParamValueError):
            reorder_categories(categorical=categorical[0:-1], weights=weights)

        with self.assertRaises(HelpskParamValueError):
            reorder_categories(categorical=categorical, weights=weights[0:-1])

        # check that the purpose column hasn't changed and that it is not ordered and has original categories
        self.assertFalse(categorical.cat.ordered)
        self.assertEqual(list(categorical.cat.categories), original_categories)

        # test default weights
        # also, we are testing a Categorical object; verify it is categorical, then later test non-categorical
        self.assertEqual(categorical.dtype.name, 'category')
        results = reorder_categories(categorical=categorical, weights=weights, ascending=True, ordered=False)
        self.assertTrue(iterables_are_equal(results, original_categorical))
        self.assertEqual(list(results.categories), expected_categories)
        self.assertFalse(results.ordered)

        # check no side effects
        results[0] = 'new car'
        self.assertEqual(list(categorical.cat.categories), list(original_categorical.cat.categories))
        self.assertTrue(iterables_are_equal(categorical, original_categorical))
        self.assertTrue(pd.isna(categorical[0]))

        # test ascending=False & ordered=True
        results = reorder_categories(categorical=categorical, weights=weights, ascending=False, ordered=True)
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
        results = reorder_categories(categorical=series, weights=weights, ascending=False, ordered=True)
        self.assertTrue(iterables_are_equal(results, series))
        self.assertEqual(list(results.categories), ['b', 'c', 'a'])
        self.assertTrue(results.ordered)

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

    def test_convert_integer_series_to_categorical(self):
        # check that function fails if mapping doesn't contains all numbers in data
        with self.assertRaises(HelpskParamValueError):
            mapping = {2: 'a', 3: 'b'}
            convert_integer_series_to_categorical(series=self.sample_data.col_a, mapping=mapping)

        with self.assertRaises(HelpskParamValueError):
            mapping = {2: 'a'}
            convert_integer_series_to_categorical(series=self.sample_data.col_a, mapping=mapping)

        with self.assertRaises(HelpskParamValueError):
            mapping = {}
            convert_integer_series_to_categorical(series=self.sample_data.col_a, mapping=mapping)

        # check that series will work even if it doesn't have all the values in mapping
        mapping = {1: 'value not found', 2: 'a', 3: 'b', 4: 'c'}
        result = convert_integer_series_to_categorical(series=self.sample_data.col_a,
                                                       mapping=mapping,
                                                       ordered=False)
        self.assertTrue(hv.iterables_are_equal(result.cat.categories, mapping.values()))
        self.assertTrue(hv.iterables_are_equal(list(result.values), [np.nan, 'a', 'b', 'c']))
        # ensure no side effects on list passed in
        self.assertTrue(hv.iterables_are_equal(self.sample_data.col_a, [np.nan, 2, 3, 4]))
        self.assertFalse(result.cat.ordered)

        # same thing but with ordered Categorical
        mapping = {1: 'not found', 2: 'a', 3: 'b', 4: 'c'}
        result = convert_integer_series_to_categorical(series=self.sample_data.col_a,
                                                       mapping=mapping,
                                                       ordered=True)
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

        test_series_ordered = test_series_unordered.cat.reorder_categories(credit_history_order,
                                                                           ordered=True, inplace=False)
        self.assertTrue(test_series_ordered.cat.ordered)

        results = value_frequency(test_series_unordered, sort_by_frequency=True)
        expected_value_order = test_series_unordered.value_counts(dropna=False).index.tolist()
        self.assertEqual(results.index.tolist(), expected_value_order)
        self.assertTrue((results['Frequency'] == test_series_unordered.value_counts(dropna=False).values).all())  # noqa
        self.assertTrue((results['Percent'] == test_series_unordered.value_counts(normalize=True, dropna=False).values).all())  # noqa

        # cache the verified results
        cached_results = results

        # test unordered categorical - sort_by_frequency=False
        results = value_frequency(test_series_unordered, sort_by_frequency=False)
        self.assertEqual(results.index.tolist(), credit_history_alphabetical + [np.nan])
        self.assertEqual(cached_results.loc[results.index.tolist(), 'Frequency'].tolist(), results['Frequency'].tolist())
        self.assertTrue(cached_results.loc[results.index.tolist(), 'Percent'].tolist() == results['Percent'].tolist())

        # test ordered categorical - sort_by_frequency=False
        results = value_frequency(test_series_ordered, sort_by_frequency=False)
        self.assertEqual(results.index.tolist(), credit_history_order + [np.nan])
        self.assertTrue(cached_results.loc[results.index.tolist(), 'Frequency'].tolist() == results['Frequency'].tolist())
        self.assertTrue(cached_results.loc[results.index.tolist(), 'Percent'].tolist() == results['Percent'].tolist())

        # test ordered categorical - sort_by_frequency=True
        results = value_frequency(test_series_ordered, sort_by_frequency=True)
        self.assertEqual(results.index.tolist(), cached_results.index.tolist())
        self.assertTrue(cached_results['Frequency'].tolist() == results['Frequency'].tolist())
        self.assertTrue(cached_results['Percent'].tolist() == results['Percent'].tolist())
        self.assertTrue(hv.dataframes_match([cached_results, results]))

        # test numeric
        test_series_float = self.credit_data['duration'].copy()
        test_series_float[0:10] = np.nan

        results = value_frequency(test_series_float, sort_by_frequency=True)
        expected_value_counts = test_series_float.value_counts(normalize=False, dropna=False)
        self.assertTrue(hv.iterables_are_equal(results.index.values, expected_value_counts.index.values))
        self.assertTrue(hv.iterables_are_equal(results['Frequency'].values, expected_value_counts.values))

        cached_results = results

        results = value_frequency(test_series_float, sort_by_frequency=False)
        expected_indexes = test_series_float.dropna().unique()
        expected_indexes.sort()
        expected_indexes = expected_indexes.tolist() + [np.nan]

        self.assertEqual(results.index.values.tolist()[0:-1], expected_indexes[0:-1])
        self.assertTrue(hv.iterables_are_equal(results.index.values.tolist(), expected_indexes))  # noqa
        self.assertTrue(hv.iterables_are_equal(results['Frequency'].values, cached_results.loc[results.index.values, 'Frequency'].values))
        self.assertTrue(hv.iterables_are_equal(results['Percent'].values, cached_results.loc[results.index.values, 'Percent'].values))

    def test_value_frequency__missing_category(self):
        # found a bug when doing `value_frequency(series, sort_by_frequency=False)` with a series that had
        # a count of `0` for a category (i.e. category existed but not any values)
        test_series = self.credit_data['checking_status']
        test_series[test_series == 'no checking'] = np.nan

        results = value_frequency(test_series, sort_by_frequency=True)
        expected_value_counts = test_series.value_counts(normalize=False, dropna=False)

        self.assertTrue(hv.iterables_are_equal(results.index.values, expected_value_counts.index.values))
        self.assertTrue(hv.iterables_are_equal(results['Frequency'].values, expected_value_counts.values))

        cached_results = results

        results = value_frequency(test_series, sort_by_frequency=False)
        expected_indexes = ['0<=X<200', '<0', '>=200', 'no checking', np.nan]

        self.assertEqual(results.index.values.tolist()[0:-1], expected_indexes[0:-1])
        self.assertTrue(hv.iterables_are_equal(results.index.values.tolist(), expected_indexes))  # noqa
        self.assertTrue(hv.iterables_are_equal(results['Frequency'].values, cached_results.loc[results.index.values, 'Frequency'].values))
        self.assertTrue(hv.iterables_are_equal(results['Percent'].values, cached_results.loc[results.index.values, 'Percent'].values))
