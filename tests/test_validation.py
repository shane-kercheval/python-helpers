import unittest
from datetime import timedelta

import numpy as np
import pandas as pd

from helpsk import string as hs
from helpsk import validation as hv
from helpsk import exceptions as he


# noinspection PyMethodMayBeStatic
class TestValidation(unittest.TestCase):

    def test_any_none_nan(self):
        self.assertTrue(hv.any_none_nan(None))
        self.assertTrue(hv.any_none_nan(np.NaN))
        self.assertTrue(hv.any_none_nan(pd.NaT))
        self.assertTrue(hv.any_none_nan(pd.NA))

        self.assertFalse(hv.any_none_nan(0))
        self.assertFalse(hv.any_none_nan(1))
        self.assertFalse(hv.any_none_nan(''))
        self.assertFalse(hv.any_none_nan('a'))

        # test list
        self.assertTrue(hv.any_none_nan([1, np.nan, None]))
        self.assertTrue(hv.any_none_nan([1, np.nan]))
        self.assertTrue(hv.any_none_nan([1, pd.NA]))
        self.assertTrue(hv.any_none_nan([1, pd.NaT]))
        self.assertTrue(hv.any_none_nan([1, None]))
        self.assertTrue(hv.any_none_nan([np.nan]))
        self.assertTrue(hv.any_none_nan([pd.NA]))
        self.assertTrue(hv.any_none_nan([pd.NaT]))
        self.assertTrue(hv.any_none_nan([None]))
        self.assertTrue(hv.any_none_nan([]))
        self.assertFalse(hv.any_none_nan([1]))
        self.assertFalse(hv.any_none_nan(['']))

        # test numpy array
        self.assertTrue(hv.any_none_nan(np.array([1, np.nan, None])))
        self.assertTrue(hv.any_none_nan(np.array([1, np.nan])))
        self.assertTrue(hv.any_none_nan(np.array([1, pd.NA])))
        self.assertTrue(hv.any_none_nan(np.array([1, pd.NaT])))
        self.assertTrue(hv.any_none_nan(np.array([1, None])))
        self.assertTrue(hv.any_none_nan(np.array([np.nan])))
        self.assertTrue(hv.any_none_nan(np.array([pd.NA])))
        self.assertTrue(hv.any_none_nan(np.array([pd.NaT])))
        self.assertTrue(hv.any_none_nan(np.array([None])))
        self.assertTrue(hv.any_none_nan(np.array([])))
        self.assertFalse(hv.any_none_nan(np.array([1])))
        self.assertFalse(hv.any_none_nan(np.array([''])))

        # test pandas series
        self.assertTrue(hv.any_none_nan(pd.Series([1, np.nan, None])))
        self.assertTrue(hv.any_none_nan(pd.Series([1, np.nan])))
        self.assertTrue(hv.any_none_nan(pd.Series([1, pd.NA])))
        self.assertTrue(hv.any_none_nan(pd.Series([1, pd.NaT])))
        self.assertTrue(hv.any_none_nan(pd.Series([1, None])))
        self.assertTrue(hv.any_none_nan(pd.Series([np.nan])))
        self.assertTrue(hv.any_none_nan(pd.Series([pd.NA])))
        self.assertTrue(hv.any_none_nan(pd.Series([pd.NaT])))
        self.assertTrue(hv.any_none_nan(pd.Series([None])))
        self.assertTrue(hv.any_none_nan(pd.Series([], dtype=float)))
        self.assertFalse(hv.any_none_nan(pd.Series([1])))
        self.assertFalse(hv.any_none_nan(pd.Series([''])))

        # test pandas data.frame
        self.assertTrue(hv.any_none_nan(pd.DataFrame([[1, np.nan, None], [1, 2, 3]])))
        self.assertTrue(hv.any_none_nan(pd.DataFrame([[1, np.nan], [1, 2]])))
        self.assertTrue(hv.any_none_nan(pd.DataFrame([[1, pd.NA], [1, 2]])))
        self.assertTrue(hv.any_none_nan(pd.DataFrame([[1, pd.NaT], [1, 2]])))
        self.assertTrue(hv.any_none_nan(pd.DataFrame([[1, None], [1, 2]])))
        self.assertTrue(hv.any_none_nan(pd.DataFrame([[np.nan], [1]])))
        self.assertTrue(hv.any_none_nan(pd.DataFrame([[pd.NA], [1]])))
        self.assertTrue(hv.any_none_nan(pd.DataFrame([[pd.NaT], [1]])))
        self.assertTrue(hv.any_none_nan(pd.DataFrame([[None], [1]])))
        self.assertTrue(hv.any_none_nan(pd.DataFrame([], dtype=float)))
        self.assertFalse(hv.any_none_nan(pd.DataFrame([1])))
        self.assertFalse(hv.any_none_nan(pd.DataFrame([[1], [1]])))
        self.assertFalse(hv.any_none_nan(pd.DataFrame([[''], [1]])))

    def test_any_missing(self):
        self.assertTrue(hv.any_missing(None))
        self.assertTrue(hv.any_missing(np.NaN))
        self.assertTrue(hv.any_missing(pd.NA))
        self.assertTrue(hv.any_missing(pd.NaT))
        self.assertTrue(hv.any_missing(''))

        self.assertFalse(hv.any_missing(0))

        # test list
        self.assertTrue(hv.any_missing([1, np.nan, None]))
        self.assertTrue(hv.any_missing([1, np.nan]))
        self.assertTrue(hv.any_missing([1, pd.NA]))
        self.assertTrue(hv.any_missing([1, pd.NaT]))
        self.assertTrue(hv.any_missing([1, None]))
        self.assertTrue(hv.any_missing([np.nan]))
        self.assertTrue(hv.any_missing([pd.NA]))
        self.assertTrue(hv.any_missing([pd.NaT]))
        self.assertTrue(hv.any_missing([None]))
        self.assertTrue(hv.any_missing([]))
        self.assertTrue(hv.any_missing(['']))
        self.assertTrue(hv.any_missing(['abc', '']))
        self.assertTrue(hv.any_missing([1, '']))
        self.assertFalse(hv.any_missing([1]))
        self.assertFalse(hv.any_missing(['a']))

        # test pandas series
        self.assertTrue(hv.any_missing(pd.Series([1, np.nan, None])))
        self.assertTrue(hv.any_missing(pd.Series([1, np.nan])))
        self.assertTrue(hv.any_missing(pd.Series([1, pd.NA])))
        self.assertTrue(hv.any_missing(pd.Series([1, pd.NaT])))
        self.assertTrue(hv.any_missing(pd.Series([1, None])))
        self.assertTrue(hv.any_missing(pd.Series([np.nan])))
        self.assertTrue(hv.any_missing(pd.Series([pd.NA])))
        self.assertTrue(hv.any_missing(pd.Series([pd.NaT])))
        self.assertTrue(hv.any_missing(pd.Series([None])))
        self.assertTrue(hv.any_missing(pd.Series([], dtype=float)))
        self.assertTrue(hv.any_missing(pd.Series([''])))
        self.assertTrue(hv.any_missing(pd.Series(['abc', ''])))
        self.assertTrue(hv.any_missing(pd.Series([1, ''])))
        self.assertFalse(hv.any_missing(pd.Series([1])))
        self.assertFalse(hv.any_missing(pd.Series(['a'])))

        # test pandas data.frame
        self.assertTrue(hv.any_missing(pd.DataFrame([[1, np.nan, None], [1, 2, 3]])))
        self.assertTrue(hv.any_missing(pd.DataFrame([[1, np.nan], [1, 2]])))
        self.assertTrue(hv.any_missing(pd.DataFrame([[1, pd.NA], [1, 2]])))
        self.assertTrue(hv.any_missing(pd.DataFrame([[1, pd.NaT], [1, 2]])))
        self.assertTrue(hv.any_missing(pd.DataFrame([[1, None], [1, 2]])))
        self.assertTrue(hv.any_missing(pd.DataFrame([[np.nan], [1]])))
        self.assertTrue(hv.any_missing(pd.DataFrame([[pd.NA], [1]])))
        self.assertTrue(hv.any_missing(pd.DataFrame([[pd.NaT], [1]])))
        self.assertTrue(hv.any_missing(pd.DataFrame([[None], [1]])))
        self.assertTrue(hv.any_missing(pd.DataFrame([], dtype=float)))
        self.assertTrue(hv.any_missing(pd.DataFrame([['abc', ''], ['abc', 'abc']])))
        self.assertTrue(hv.any_missing(pd.DataFrame([''])))
        self.assertFalse(hv.any_missing(pd.DataFrame([1])))
        self.assertFalse(hv.any_missing(pd.DataFrame([[1], [1]])))

    def test_any_duplicated(self):
        # test list
        self.assertFalse(hv.any_duplicated(['']))
        self.assertFalse(hv.any_duplicated(['', 1]))
        self.assertFalse(hv.any_duplicated(['', 1, None]))

        self.assertTrue(hv.any_duplicated(['', 1, '']))
        self.assertTrue(hv.any_duplicated(['', 1, 1]))
        self.assertTrue(hv.any_duplicated(['', 1, None, None]))

        # test pd.Series
        self.assertFalse(hv.any_duplicated(pd.Series([''])))
        self.assertFalse(hv.any_duplicated(pd.Series(['', 1])))
        self.assertFalse(hv.any_duplicated(pd.Series(['', 1, None])))

        self.assertTrue(hv.any_duplicated(pd.Series(['', 1, ''])))
        self.assertTrue(hv.any_duplicated(pd.Series(['', 1, 1])))
        self.assertTrue(hv.any_duplicated(pd.Series(['', 1, None, None])))

    def test_iterables_are_equal(self):

        def subtest_iterables_are_equal(iterable_a, iterable_b, dtype):
            assert hv.iterables_are_equal(iterable_a, iterable_b)
            if dtype != str:
                assert hv.iterables_are_equal(np.array(iterable_a), np.array(iterable_b))
                assert hv.iterables_are_equal(np.array(iterable_a), iterable_b)
            assert hv.iterables_are_equal(
                pd.Series(iterable_a, dtype=dtype),
                pd.Series(iterable_b, dtype=dtype)
            )

        subtest_iterables_are_equal([], [], dtype=np.float64)
        subtest_iterables_are_equal([np.nan], [np.nan], np.float64)
        subtest_iterables_are_equal([1, 2, 3, 4], [1, 2, 3, 4], np.float64)
        subtest_iterables_are_equal([np.nan, 1, 2, 3, 4], [np.nan, 1, 2, 3, 4], np.float64)
        subtest_iterables_are_equal([np.nan, 1, 2, 3, 4, np.nan], [np.nan, 1, 2, 3, 4, np.nan], np.float64)  # noqa

        subtest_iterables_are_equal(iterable_a=['a', 'b', 'c', 'd'], iterable_b=['a', 'b', 'c', 'd'], dtype=str)  # noqa
        subtest_iterables_are_equal([np.nan, 'a', 'b', 'c', 'd'], [np.nan, 'a', 'b', 'c', 'd'], str)  # noqa
        subtest_iterables_are_equal([np.nan, 'a', 'b', 'c', 'd', np.nan], [np.nan, 'a', 'b', 'c', 'd', np.nan], str)  # noqa

        subtest_iterables_are_equal(['a', 'b', 'c', 4], ['a', 'b', 'c', 4], str)
        subtest_iterables_are_equal([np.nan, 'a', 'b', 'c', 4], [np.nan, 'a', 'b', 'c', 4], str)
        subtest_iterables_are_equal([np.nan, 'a', 'b', 'c', 4, np.nan], [np.nan, 'a', 'b', 'c', 4, np.nan], str)  # noqa

        def subtest_iterables_are_not_equal(iterable_a, iterable_b, dtype):
            assert not hv.iterables_are_equal(iterable_a, iterable_b)
            assert not hv.iterables_are_equal(np.array(iterable_a), np.array(iterable_b))
            assert not hv.iterables_are_equal(pd.Series(iterable_a, dtype=dtype), pd.Series(iterable_b, dtype=dtype))  # noqa

        subtest_iterables_are_not_equal(iterable_a=[], iterable_b=[1], dtype=np.float64)
        subtest_iterables_are_not_equal(iterable_a=[1], iterable_b=[], dtype=np.float64)
        subtest_iterables_are_not_equal(iterable_a=[], iterable_b=[np.nan], dtype=np.float64)
        subtest_iterables_are_not_equal(iterable_a=[np.nan], iterable_b=[], dtype=np.float64)
        subtest_iterables_are_not_equal(iterable_a=[], iterable_b=[''], dtype=str)
        subtest_iterables_are_not_equal(iterable_a=[''], iterable_b=[], dtype=str)

        subtest_iterables_are_not_equal(iterable_a=[1, 2, 3], iterable_b=[1, 2, 3, 4], dtype=np.float64)  # noqa
        subtest_iterables_are_not_equal(iterable_a=[0, 1, 2, 3], iterable_b=[1, 2, 3], dtype=np.float64)  # noqa
        subtest_iterables_are_not_equal(iterable_a=[1.000001, 2, 3], iterable_b=[1, 2, 3], dtype=np.float64)  # noqa
        subtest_iterables_are_not_equal(iterable_a=[1, 2, 3], iterable_b=[1.000001, 2, 3], dtype=np.float64)  # noqa
        subtest_iterables_are_not_equal(iterable_a=[1, 2, 3], iterable_b=[1.000001, 2, 3], dtype=np.float64)  # noqa
        subtest_iterables_are_not_equal(iterable_a=[1, 2, 3], iterable_b=[np.nan, 2, 3], dtype=np.float64)  # noqa
        subtest_iterables_are_not_equal(iterable_a=[np.nan, 2, 3], iterable_b=[1, 2, 3], dtype=np.float64)  # noqa
        subtest_iterables_are_not_equal(iterable_a=[1, 2, 3], iterable_b=[None, 2, 3], dtype=np.float64)  # noqa

    def test_iterables_are_equal__ordered(self):
        # found a bug where two pd.Categorical series will not return True even if they have the
        # same values, but with different ordered; but we only care about the values
        values = [np.nan, 'a', 'b', 'c', 'd']
        series = pd.Series(values)
        unordered_categorical = pd.Categorical(values=values, ordered=False)
        ordered_categorical = pd.Categorical(values=values, categories=['d', 'c', 'a', 'b'], ordered=True)  # noqa

        self.assertTrue(hv.iterables_are_equal(values, series))
        self.assertTrue(hv.iterables_are_equal(values, unordered_categorical))
        self.assertTrue(hv.iterables_are_equal(values, ordered_categorical))
        self.assertTrue(hv.iterables_are_equal(series, unordered_categorical))
        self.assertTrue(hv.iterables_are_equal(series, ordered_categorical))
        self.assertTrue(hv.iterables_are_equal(unordered_categorical, ordered_categorical))

        different_values = values.copy()
        different_values[-1] = 'z'

        self.assertFalse(hv.iterables_are_equal(different_values, series))
        self.assertFalse(hv.iterables_are_equal(different_values, unordered_categorical))
        self.assertFalse(hv.iterables_are_equal(different_values, ordered_categorical))
        self.assertFalse(hv.iterables_are_equal(pd.Categorical(different_values, ordered=False),
                                                unordered_categorical))
        self.assertFalse(hv.iterables_are_equal(pd.Categorical(different_values, ordered=True),
                                                ordered_categorical))

    def test_dataframes_match(self):

        dataframe_1 = pd.DataFrame({
            'col_floats': [1.123456789, 2.123456789, 3.123456789, np.nan],
            'col_strings': [np.nan, 'a', 'b', 'c'],
            'col_enums': [np.nan, hs.RoundTo.NONE, hs.RoundTo.AUTO, hs.RoundTo.THOUSANDS],
            'col_dates': pd.date_range('2021-01-01', periods=4),
            'col_missing': [np.nan, np.nan, np.nan, np.nan]}
        )
        dataframe_1.loc[0, 'col_dates'] = np.nan

        with self.assertRaises(he.HelpskParamTypeError):
            hv.dataframes_match(dataframes=None)  # noqa

        with self.assertRaises(he.HelpskParamTypeError):
            hv.dataframes_match(dataframes=dataframe_1)  # noqa

        with self.assertRaises(he.HelpskParamValueError):
            hv.dataframes_match(dataframes=[])

        with self.assertRaises(he.HelpskParamValueError):
            hv.dataframes_match(dataframes=[dataframe_1])

        # test that there are no side effects; e.g. we set index/column values if we ignore them
        dataframe_1_original = dataframe_1.copy()
        dataframe_2 = dataframe_1.copy()
        dataframe_2.index = ['a', 'b', 'c', 'd']
        dataframe_2.columns = ['a', 'b', 'c', 'd', 'e']
        dataframe_2_original = dataframe_2.copy()
        self.assertTrue(hv.dataframes_match(dataframes=[dataframe_1, dataframe_2],
                                            float_tolerance=1,
                                            ignore_indexes=True,
                                            ignore_column_names=True))
        self.assertTrue(dataframe_1.equals(dataframe_1_original))
        self.assertTrue((dataframe_1.index == dataframe_1_original.index).all())
        self.assertTrue((dataframe_1.columns == dataframe_1_original.columns).all())
        self.assertTrue(dataframe_2.equals(dataframe_2_original))
        self.assertTrue((dataframe_2.index == dataframe_2_original.index).all())
        self.assertTrue((dataframe_2.columns == dataframe_2_original.columns).all())

        self.assertTrue(hv.dataframes_match(dataframes=[dataframe_1, dataframe_1]))
        self.assertTrue(hv.dataframes_match(dataframes=[dataframe_1, dataframe_1, dataframe_1]))
        self.assertTrue(hv.dataframes_match(dataframes=[dataframe_1, dataframe_1.copy(), dataframe_1.copy()]))  # noqa

        self.assertTrue(hv.dataframes_match(dataframes=[dataframe_1, dataframe_1.round(6)]))
        self.assertFalse(hv.dataframes_match(dataframes=[dataframe_1, dataframe_1.round(5)]))
        self.assertFalse(hv.dataframes_match(dataframes=[dataframe_1, dataframe_1, dataframe_1.round(5)]))  # noqa
        self.assertTrue(hv.dataframes_match(dataframes=[dataframe_1, dataframe_1.round(5)],
                                            float_tolerance=5))
        self.assertTrue(hv.dataframes_match(dataframes=[dataframe_1, dataframe_1.round(6),
                                                        dataframe_1.round(5)],
                                            float_tolerance=5))

        dataframe_2 = dataframe_1.copy()
        self.assertFalse(pd.isna(dataframe_2.iat[0, 0]))
        dataframe_2.iat[0, 0] = dataframe_2.iat[0, 0] - 0.000001
        self.assertFalse(hv.dataframes_match(dataframes=[dataframe_1, dataframe_2]))
        self.assertFalse(hv.dataframes_match(dataframes=[dataframe_2, dataframe_1]))
        self.assertFalse(hv.dataframes_match(dataframes=[dataframe_1, dataframe_1.copy(), dataframe_2]))  # noqa

        dataframe_2 = dataframe_1.copy()
        self.assertFalse(pd.isna(dataframe_2.iat[1, 1]))
        dataframe_2.iat[1, 1] = 'c'
        self.assertFalse(hv.dataframes_match(dataframes=[dataframe_1, dataframe_2]))
        self.assertFalse(hv.dataframes_match(dataframes=[dataframe_2, dataframe_1]))
        self.assertFalse(hv.dataframes_match(dataframes=[dataframe_1, dataframe_1.copy(), dataframe_2]))  # noqa

        dataframe_2 = dataframe_1.copy()
        self.assertFalse(pd.isna(dataframe_2.iat[2, 2]))
        dataframe_2.iat[2, 2] = hs.RoundTo.MILLIONS
        self.assertFalse(hv.dataframes_match(dataframes=[dataframe_1, dataframe_2]))
        self.assertFalse(hv.dataframes_match(dataframes=[dataframe_2, dataframe_1]))
        self.assertFalse(hv.dataframes_match(dataframes=[dataframe_1, dataframe_1.copy(), dataframe_2]))  # noqa

        dataframe_2 = dataframe_1.copy()
        self.assertFalse(pd.isna(dataframe_2.iat[2, 3]))
        dataframe_2.iat[2, 3] = dataframe_2.iat[2, 3] + timedelta(seconds=1)
        self.assertFalse(hv.dataframes_match(dataframes=[dataframe_1, dataframe_2]))
        self.assertFalse(hv.dataframes_match(dataframes=[dataframe_2, dataframe_1]))
        self.assertFalse(hv.dataframes_match(dataframes=[dataframe_1, dataframe_1.copy(), dataframe_2]))  # noqa

        dataframe_2 = dataframe_1.copy()
        self.assertFalse(pd.isna(dataframe_2.iat[2, 3]))
        dataframe_2.iat[2, 3] = dataframe_2.iat[2, 3] + timedelta(seconds=1) - timedelta(seconds=1)
        self.assertTrue(hv.dataframes_match(dataframes=[dataframe_1, dataframe_2]))
        self.assertTrue(hv.dataframes_match(dataframes=[dataframe_1, dataframe_1.copy(), dataframe_2]))  # noqa

        dataframe_2 = dataframe_1.copy()
        self.assertFalse(pd.isna(dataframe_2.iat[0, 0]))
        dataframe_2.iat[0, 0] = np.NaN
        self.assertFalse(hv.dataframes_match(dataframes=[dataframe_1, dataframe_2]))
        self.assertFalse(hv.dataframes_match(dataframes=[dataframe_2, dataframe_1]))
        self.assertFalse(hv.dataframes_match(dataframes=[dataframe_1, dataframe_1.copy(), dataframe_2]))  # noqa

        dataframe_2 = dataframe_1.copy()
        self.assertFalse(pd.isna(dataframe_2.iat[1, 1]))
        dataframe_2.iat[1, 1] = np.NaN
        self.assertFalse(hv.dataframes_match(dataframes=[dataframe_1, dataframe_2]))
        self.assertFalse(hv.dataframes_match(dataframes=[dataframe_2, dataframe_1]))
        self.assertFalse(hv.dataframes_match(dataframes=[dataframe_1, dataframe_1.copy(), dataframe_2]))  # noqa

        dataframe_2 = dataframe_1.copy()
        self.assertFalse(pd.isna(dataframe_2.iat[1, 2]))
        dataframe_2.iat[1, 2] = np.NaN
        self.assertFalse(hv.dataframes_match(dataframes=[dataframe_1, dataframe_2]))
        self.assertFalse(hv.dataframes_match(dataframes=[dataframe_2, dataframe_1]))
        self.assertFalse(hv.dataframes_match(dataframes=[dataframe_1, dataframe_1.copy(), dataframe_2]))  # noqa

        dataframe_2 = dataframe_1.copy()
        self.assertFalse(pd.isna(dataframe_2.iat[1, 3]))
        dataframe_2.iat[1, 3] = np.NaN
        self.assertFalse(hv.dataframes_match(dataframes=[dataframe_1, dataframe_2]))
        self.assertFalse(hv.dataframes_match(dataframes=[dataframe_2, dataframe_1]))
        self.assertFalse(hv.dataframes_match(dataframes=[dataframe_1, dataframe_1.copy(), dataframe_2]))  # noqa

        dataframe_2 = dataframe_1.copy()
        dataframe_2.index = ['a', 'b', 'c', 'd']
        dataframe_2.columns = ['a', 'b', 'c', 'd', 'e']
        self.assertTrue(hv.dataframes_match(dataframes=[dataframe_1, dataframe_2]))
        self.assertTrue(hv.dataframes_match(dataframes=[dataframe_1, dataframe_2, dataframe_2]))
        self.assertFalse(hv.dataframes_match(dataframes=[dataframe_1, dataframe_2], ignore_indexes=False))  # noqa
        self.assertFalse(hv.dataframes_match(dataframes=[dataframe_1, dataframe_2],
                                             ignore_column_names=False))
        self.assertFalse(hv.dataframes_match(dataframes=[dataframe_2, dataframe_1],
                                             ignore_column_names=False))
        self.assertFalse(hv.dataframes_match(dataframes=[dataframe_1, dataframe_2],
                                             ignore_indexes=False,
                                             ignore_column_names=False))
        self.assertFalse(hv.dataframes_match(dataframes=[dataframe_2, dataframe_1],
                                             ignore_indexes=False,
                                             ignore_column_names=False))

    def test_is_close(self):
        self.assertFalse(hv.is_close(np.nan, 1))
        self.assertFalse(hv.is_close(1, np.nan))
        self.assertFalse(hv.is_close(np.nan, np.nan))
        with self.assertRaises(Exception):
            hv.is_close(None, 1)  # noqa
        with self.assertRaises(Exception):
            hv.is_close(1, None)  # noqa
        with self.assertRaises(Exception):
            hv.is_close(None, None)  # noqa

        def sub_test(value, tolerance, failure_tolerance):
            self.assertTrue(hv.is_close(value, value))
            self.assertTrue(hv.is_close(value, value + tolerance))
            self.assertTrue(hv.is_close(value + tolerance, value))
            self.assertTrue(hv.is_close(-value, -value - tolerance))
            self.assertTrue(hv.is_close(-value - tolerance, -value))

        sub_test(value=0, tolerance=0.000001, failure_tolerance=0.0000011)
        sub_test(value=1, tolerance=0.000001, failure_tolerance=0.0000011)
        sub_test(value=100, tolerance=0.000001, failure_tolerance=0.0000011)
        sub_test(value=-100, tolerance=0.000001, failure_tolerance=0.0000011)

    def test_raises_exception(self):

        def my_function_exception():
            raise ValueError()

        def my_function_runs():
            return True

        self.assertTrue(hv.raises_exception(my_function_exception))
        # should return True since my_function_exception raises ValueError
        self.assertTrue(hv.raises_exception(my_function_exception, ValueError))
        # should return False since my_function_exception raises ValueError, not TypeError
        self.assertFalse(hv.raises_exception(my_function_exception, TypeError))
        self.assertFalse(hv.raises_exception(my_function_runs))
