import unittest
from datetime import timedelta

import numpy as np
import pandas as pd

from helpsk import string as hs
from helpsk import validation as hv
from helpsk.exceptions import *


# noinspection PyMethodMayBeStatic
class TestValidation(unittest.TestCase):

    def test_any_none_nan(self):
        self.assertTrue(hv.any_none_nan(None))  # noqa
        self.assertTrue(hv.any_none_nan(np.NaN))

        # test list
        self.assertTrue(hv.any_none_nan([1, np.nan, None]))
        self.assertTrue(hv.any_none_nan([1, np.nan]))
        self.assertTrue(hv.any_none_nan([1, None]))
        self.assertTrue(hv.any_none_nan([np.nan]))
        self.assertTrue(hv.any_none_nan([None]))
        self.assertTrue(hv.any_none_nan([]))
        self.assertFalse(hv.any_none_nan([1]))
        self.assertFalse(hv.any_none_nan(['']))

        # test numpy array
        self.assertTrue(hv.any_none_nan(np.array([1, np.nan, None])))
        self.assertTrue(hv.any_none_nan(np.array([1, np.nan])))
        self.assertTrue(hv.any_none_nan(np.array([1, None])))
        self.assertTrue(hv.any_none_nan(np.array([np.nan])))
        self.assertTrue(hv.any_none_nan(np.array([None])))
        self.assertTrue(hv.any_none_nan(np.array([])))
        self.assertFalse(hv.any_none_nan(np.array([1])))
        self.assertFalse(hv.any_none_nan(np.array([''])))

        # test pandas series
        self.assertTrue(hv.any_none_nan(pd.Series([1, np.nan, None])))
        self.assertTrue(hv.any_none_nan(pd.Series([1, np.nan])))
        self.assertTrue(hv.any_none_nan(pd.Series([1, None])))
        self.assertTrue(hv.any_none_nan(pd.Series([np.nan])))
        self.assertTrue(hv.any_none_nan(pd.Series([None])))
        self.assertTrue(hv.any_none_nan(pd.Series([], dtype=float)))
        self.assertFalse(hv.any_none_nan(pd.Series([1])))
        self.assertFalse(hv.any_none_nan(pd.Series([''])))

        # test pandas data.frame
        self.assertTrue(hv.any_none_nan(pd.DataFrame([[1, np.nan, None], [1, 2, 3]])))
        self.assertTrue(hv.any_none_nan(pd.DataFrame([[1, np.nan], [1, 2]])))
        self.assertTrue(hv.any_none_nan(pd.DataFrame([[1, None], [1, 2]])))
        self.assertTrue(hv.any_none_nan(pd.DataFrame([[np.nan], [1]])))
        self.assertTrue(hv.any_none_nan(pd.DataFrame([[None], [1]])))
        self.assertTrue(hv.any_none_nan(pd.DataFrame([], dtype=float)))
        self.assertFalse(hv.any_none_nan(pd.DataFrame([1])))
        self.assertFalse(hv.any_none_nan(pd.DataFrame([[1], [1]])))
        self.assertFalse(hv.any_none_nan(pd.DataFrame([[''], [1]])))

    def test_assert_not_none_nan(self):

        with self.assertRaises(HelpskAssertionError):
            hv.assert_not_none_nan(None)  # noqa
        with self.assertRaises(HelpskAssertionError):
            hv.assert_not_none_nan(np.NaN)

        # test list
        with self.assertRaises(HelpskAssertionError):
            hv.assert_not_none_nan([1, np.nan, None])
        with self.assertRaises(HelpskAssertionError):
            hv.assert_not_none_nan([1, np.nan])
        with self.assertRaises(HelpskAssertionError):
            hv.assert_not_none_nan([1, None])
        with self.assertRaises(HelpskAssertionError):
            hv.assert_not_none_nan([np.nan])
        with self.assertRaises(HelpskAssertionError):
            hv.assert_not_none_nan([None])
        with self.assertRaises(HelpskAssertionError):
            hv.assert_not_none_nan([])
        hv.assert_not_none_nan([1])
        hv.assert_not_none_nan([''])

        # test numpy array
        with self.assertRaises(HelpskAssertionError):
            hv.assert_not_none_nan(np.array([1, np.nan, None]))
        with self.assertRaises(HelpskAssertionError):
            hv.assert_not_none_nan(np.array([1, np.nan]))
        with self.assertRaises(HelpskAssertionError):
            hv.assert_not_none_nan(np.array([1, None]))
        with self.assertRaises(HelpskAssertionError):
            hv.assert_not_none_nan(np.array([np.nan]))
        with self.assertRaises(HelpskAssertionError):
            hv.assert_not_none_nan(np.array([None]))
        with self.assertRaises(HelpskAssertionError):
            hv.assert_not_none_nan(np.array([]))
        hv.assert_not_none_nan(np.array([1]))
        hv.assert_not_none_nan(np.array(['']))

        # test pandas series
        with self.assertRaises(HelpskAssertionError):
            hv.assert_not_none_nan(pd.Series([1, np.nan, None]))
        with self.assertRaises(HelpskAssertionError):
            hv.assert_not_none_nan(pd.Series([1, np.nan]))
        with self.assertRaises(HelpskAssertionError):
            hv.assert_not_none_nan(pd.Series([1, None]))
        with self.assertRaises(HelpskAssertionError):
            hv.assert_not_none_nan(pd.Series([np.nan]))
        with self.assertRaises(HelpskAssertionError):
            hv.assert_not_none_nan(pd.Series([None]))
        with self.assertRaises(HelpskAssertionError):
            hv.assert_not_none_nan(pd.Series([], dtype=float))
        hv.assert_not_none_nan(pd.Series([1]))
        hv.assert_not_none_nan(pd.Series(['']))

        # test pandas data.frame
        with self.assertRaises(HelpskAssertionError):
            hv.assert_not_none_nan(pd.DataFrame([[1, np.nan, None], [1, 2, 3]]))  # noqa
        with self.assertRaises(HelpskAssertionError):
            hv.assert_not_none_nan(pd.DataFrame([[1, np.nan], [1, 2]]))
        with self.assertRaises(HelpskAssertionError):
            hv.assert_not_none_nan(pd.DataFrame([[1, None], [1, 2]]))
        with self.assertRaises(HelpskAssertionError):
            hv.assert_not_none_nan(pd.DataFrame([[np.nan], [1]]))
        with self.assertRaises(HelpskAssertionError):
            hv.assert_not_none_nan(pd.DataFrame([[None], [1]]))
        with self.assertRaises(HelpskAssertionError):
            hv.assert_not_none_nan(pd.DataFrame([], dtype=float))
        hv.assert_not_none_nan(pd.DataFrame([1]))
        hv.assert_not_none_nan(pd.DataFrame([[1], [1]]))
        hv.assert_not_none_nan(pd.DataFrame([[''], [1]]))

    def test_any_missing(self):
        self.assertTrue(hv.any_missing(None))  # noqa
        self.assertTrue(hv.any_missing(np.NaN))
        self.assertTrue(hv.any_missing(''))  # noqa

        # test list
        self.assertTrue(hv.any_missing([1, np.nan, None]))
        self.assertTrue(hv.any_missing([1, np.nan]))
        self.assertTrue(hv.any_missing([1, None]))
        self.assertTrue(hv.any_missing([np.nan]))
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
        self.assertTrue(hv.any_missing(pd.Series([1, None])))
        self.assertTrue(hv.any_missing(pd.Series([np.nan])))
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
        self.assertTrue(hv.any_missing(pd.DataFrame([[1, None], [1, 2]])))
        self.assertTrue(hv.any_missing(pd.DataFrame([[np.nan], [1]])))
        self.assertTrue(hv.any_missing(pd.DataFrame([[None], [1]])))
        self.assertTrue(hv.any_missing(pd.DataFrame([], dtype=float)))
        self.assertTrue(hv.any_missing(pd.DataFrame([['abc', ''], ['abc', 'abc']])))
        self.assertTrue(hv.any_missing(pd.DataFrame([''])))
        self.assertFalse(hv.any_missing(pd.DataFrame([1])))
        self.assertFalse(hv.any_missing(pd.DataFrame([[1], [1]])))

    def test_assert_not_any_missing(self):

        with self.assertRaises(HelpskAssertionError):
            hv.assert_not_any_missing(None)  # noqa
        with self.assertRaises(HelpskAssertionError):
            hv.assert_not_any_missing(np.NaN)
        with self.assertRaises(HelpskAssertionError):
            hv.assert_not_any_missing('')  # noqa

        # test list
        with self.assertRaises(HelpskAssertionError):
            hv.assert_not_any_missing([1, np.nan, None])
        with self.assertRaises(HelpskAssertionError):
            hv.assert_not_any_missing([1, np.nan])
        with self.assertRaises(HelpskAssertionError):
            hv.assert_not_any_missing([1, None])
        with self.assertRaises(HelpskAssertionError):
            hv.assert_not_any_missing([np.nan])
        with self.assertRaises(HelpskAssertionError):
            hv.assert_not_any_missing([None])
        with self.assertRaises(HelpskAssertionError):
            hv.assert_not_any_missing([])
        with self.assertRaises(HelpskAssertionError):
            hv.assert_not_any_missing([''])
        with self.assertRaises(HelpskAssertionError):
            hv.assert_not_any_missing(['abc', ''])
        with self.assertRaises(HelpskAssertionError):
            hv.assert_not_any_missing([1, ''])
        hv.assert_not_any_missing([1])
        hv.assert_not_any_missing(['a'])

        # test pandas series
        with self.assertRaises(HelpskAssertionError):
            hv.assert_not_any_missing(pd.Series([1, np.nan, None]))
        with self.assertRaises(HelpskAssertionError):
            hv.assert_not_any_missing(pd.Series([1, np.nan]))
        with self.assertRaises(HelpskAssertionError):
            hv.assert_not_any_missing(pd.Series([1, None]))
        with self.assertRaises(HelpskAssertionError):
            hv.assert_not_any_missing(pd.Series([np.nan]))
        with self.assertRaises(HelpskAssertionError):
            hv.assert_not_any_missing(pd.Series([None]))
        with self.assertRaises(HelpskAssertionError):
            hv.assert_not_any_missing(pd.Series([], dtype=float))
        with self.assertRaises(HelpskAssertionError):
            hv.assert_not_any_missing(pd.Series(['']))
        with self.assertRaises(HelpskAssertionError):
            hv.assert_not_any_missing(pd.Series(['abc', '']))
        with self.assertRaises(HelpskAssertionError):
            hv.assert_not_any_missing(pd.Series([1, '']))
        hv.assert_not_any_missing(pd.Series([1]))
        hv.assert_not_any_missing(pd.Series(['a']))

        # test pandas data.frame
        with self.assertRaises(HelpskAssertionError):
            hv.assert_not_any_missing(pd.DataFrame([[1, np.nan, None], [1, 2, 3]]))
        with self.assertRaises(HelpskAssertionError):
            hv.assert_not_any_missing(pd.DataFrame([[1, np.nan], [1, 2]]))
        with self.assertRaises(HelpskAssertionError):
            hv.assert_not_any_missing(pd.DataFrame([[1, None], [1, 2]]))
        with self.assertRaises(HelpskAssertionError):
            hv.assert_not_any_missing(pd.DataFrame([[np.nan], [1]]))
        with self.assertRaises(HelpskAssertionError):
            hv.assert_not_any_missing(pd.DataFrame([[None], [1]]))
        with self.assertRaises(HelpskAssertionError):
            hv.assert_not_any_missing(pd.DataFrame([], dtype=float))
        with self.assertRaises(HelpskAssertionError):
            hv.assert_not_any_missing(pd.DataFrame([['abc', ''], ['abc', 'abc']]))
        with self.assertRaises(HelpskAssertionError):
            hv.assert_not_any_missing(pd.DataFrame(['']))
        hv.assert_not_any_missing(pd.DataFrame([1]))
        hv.assert_not_any_missing(pd.DataFrame([[1], [1]]))

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

    def test_assert_not_duplicated(self):
        # test list
        hv.assert_not_duplicated([''])
        hv.assert_not_duplicated(['', 1])
        hv.assert_not_duplicated(['', 1, None])

        with self.assertRaises(HelpskAssertionError):
            hv.assert_not_duplicated(['', 1, ''])
        with self.assertRaises(HelpskAssertionError):
            hv.assert_not_duplicated(['', 1, 1])
        with self.assertRaises(HelpskAssertionError):
            hv.assert_not_duplicated(['', 1, None, None])

        # test pd.Series
        hv.assert_not_duplicated(pd.Series(['']))
        hv.assert_not_duplicated(pd.Series(['', 1]))
        hv.assert_not_duplicated(pd.Series(['', 1, None]))

        with self.assertRaises(HelpskAssertionError):
            hv.assert_not_duplicated(pd.Series(['', 1, '']))
        with self.assertRaises(HelpskAssertionError):
            hv.assert_not_duplicated(pd.Series(['', 1, 1]))
        with self.assertRaises(HelpskAssertionError):
            hv.assert_not_duplicated(pd.Series(['', 1, None, None]))

    def test_assert_true(self):
        hv.assert_true(True)
        self.assertTrue(isinstance(np.bool_(True), np.bool_))
        hv.assert_true(np.bool_(True))

        with self.assertRaises(HelpskParamTypeError):
            hv.assert_true([])  # noqa
        with self.assertRaises(HelpskParamTypeError):
            hv.assert_true([True])  # noqa

        with self.assertRaises(HelpskAssertionError):
            hv.assert_true(False)

        with self.assertRaises(HelpskAssertionError) as cm:
            hv.assert_true(False, message='my message')
        self.assertEqual(cm.exception.args[0], 'my message')

    def test_assert_false(self):
        hv.assert_false(False)
        self.assertTrue(isinstance(np.bool_(False), np.bool_))
        hv.assert_false(np.bool_(False))

        with self.assertRaises(HelpskParamTypeError):
            hv.assert_false([])  # noqa
        with self.assertRaises(HelpskParamTypeError):
            hv.assert_false([False])  # noqa

        with self.assertRaises(HelpskAssertionError):
            hv.assert_false(True)

        with self.assertRaises(HelpskAssertionError) as cm:
            hv.assert_false(True, message='my message')
        self.assertEqual(cm.exception.args[0], 'my message')

    def test_assert_all(self):
        cases = [
            [True],
            [True, True],
            np.array([True]),
            np.array([True, True]),
            pd.Series([True]),
            pd.Series([True, True]),
            pd.DataFrame([True]),
            pd.DataFrame([True, True]),
            pd.DataFrame([[True, True], [True, True]]),
        ]
        # running assert all on these cases should not result in any assertion errors
        for case in cases:
            hv.assert_all(case)

        cases = [
            [False],
            [True, False],
            np.array([False]),
            np.array([True, False]),
            pd.Series([False]),
            pd.Series([True, False]),
            pd.DataFrame([False]),
            pd.DataFrame([True, False]),
            pd.DataFrame([[True, True], [True, False]]),
        ]
        for index, case in enumerate(cases):
            with self.subTest(index=index):
                with self.assertRaises(HelpskAssertionError):
                    hv.assert_all(case)


    def test_assert_not_any(self):
        cases = [
            [False],
            [False, False],
            np.array([False]),
            np.array([False, False]),
            pd.Series([False]),
            pd.Series([False, False]),
            pd.DataFrame([False]),
            pd.DataFrame([False, False]),
            pd.DataFrame([[False, False], [False, False]]),
        ]
        # running assert all on these cases should not result in any assertion errors
        for case in cases:
            hv.assert_not_any(case)

        cases = [
            [True],
            [True, False],
            np.array([True]),
            np.array([True, False]),
            pd.Series([True]),
            pd.Series([True, False]),
            pd.DataFrame([True]),
            pd.DataFrame([True, False]),
            pd.DataFrame([[False, False], [False, True]]),
        ]
        for index, case in enumerate(cases):
            with self.subTest(index=index):
                with self.assertRaises(HelpskAssertionError):
                    hv.assert_not_any(case)

    def test_dataframes_match(self):

        dataframe_1 = pd.DataFrame({'col_floats': [1.123456789, 2.123456789, 3.123456789],
                                    'col_strings': ['a', 'b', 'c'],
                                    'col_enums': [hs.RoundTo.NONE, hs.RoundTo.AUTO, hs.RoundTo.THOUSANDS],
                                    'col_dates': pd.date_range('2021-01-01', periods=3)})

        with self.assertRaises(HelpskParamTypeError):
            hv.dataframes_match(dataframes=None)  # noqa

        with self.assertRaises(HelpskParamTypeError):
            hv.dataframes_match(dataframes=dataframe_1)  # noqa

        with self.assertRaises(HelpskParamValueError):
            hv.dataframes_match(dataframes=[])

        with self.assertRaises(HelpskParamValueError):
            hv.dataframes_match(dataframes=[dataframe_1])

        # test that there are no side effects; e.g. we set the index/column values if we ignore them
        dataframe_1_original = dataframe_1.copy()
        dataframe_2 = dataframe_1.copy()
        dataframe_2.index = ['a', 'b', 'c']
        dataframe_2.columns = ['a', 'b', 'c', 'd']
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
        self.assertTrue(hv.dataframes_match(dataframes=[dataframe_1, dataframe_1.copy(), dataframe_1.copy()]))

        self.assertTrue(hv.dataframes_match(dataframes=[dataframe_1, dataframe_1.round(6)]))
        self.assertFalse(hv.dataframes_match(dataframes=[dataframe_1, dataframe_1.round(5)]))
        self.assertFalse(hv.dataframes_match(dataframes=[dataframe_1, dataframe_1, dataframe_1.round(5)]))
        self.assertTrue(hv.dataframes_match(dataframes=[dataframe_1, dataframe_1.round(5)],
                                            float_tolerance=5))
        self.assertTrue(hv.dataframes_match(dataframes=[dataframe_1, dataframe_1.round(6),
                                                        dataframe_1.round(5)],
                                            float_tolerance=5))

        dataframe_2 = dataframe_1.copy()
        dataframe_2.iat[0, 0] = dataframe_2.iat[0, 0] - 0.000001
        self.assertFalse(hv.dataframes_match(dataframes=[dataframe_1, dataframe_2]))
        self.assertFalse(hv.dataframes_match(dataframes=[dataframe_1, dataframe_1.copy(), dataframe_2]))

        dataframe_2 = dataframe_1.copy()
        dataframe_2.iat[1, 1] = 'c'
        self.assertFalse(hv.dataframes_match(dataframes=[dataframe_1, dataframe_2]))
        self.assertFalse(hv.dataframes_match(dataframes=[dataframe_1, dataframe_1.copy(), dataframe_2]))

        dataframe_2 = dataframe_1.copy()
        dataframe_2.iat[2, 2] = hs.RoundTo.MILLIONS
        self.assertFalse(hv.dataframes_match(dataframes=[dataframe_1, dataframe_2]))
        self.assertFalse(hv.dataframes_match(dataframes=[dataframe_1, dataframe_1.copy(), dataframe_2]))

        dataframe_2 = dataframe_1.copy()
        dataframe_2.iat[2, 3] = dataframe_2.iat[2, 3] + timedelta(seconds=1)
        self.assertFalse(hv.dataframes_match(dataframes=[dataframe_1, dataframe_2]))
        self.assertFalse(hv.dataframes_match(dataframes=[dataframe_1, dataframe_1.copy(), dataframe_2]))

        dataframe_2 = dataframe_1.copy()
        dataframe_2.iat[2, 3] = dataframe_2.iat[2, 3] + timedelta(seconds=1) - timedelta(seconds=1)
        self.assertTrue(hv.dataframes_match(dataframes=[dataframe_1, dataframe_2]))
        self.assertTrue(hv.dataframes_match(dataframes=[dataframe_1, dataframe_1.copy(), dataframe_2]))

        dataframe_2 = dataframe_1.copy()
        dataframe_2.iat[0, 0] = np.NaN
        self.assertFalse(hv.dataframes_match(dataframes=[dataframe_1, dataframe_2]))
        self.assertFalse(hv.dataframes_match(dataframes=[dataframe_1, dataframe_1.copy(), dataframe_2]))

        dataframe_2 = dataframe_1.copy()
        dataframe_2.iat[1, 1] = np.NaN
        self.assertFalse(hv.dataframes_match(dataframes=[dataframe_1, dataframe_2]))
        self.assertFalse(hv.dataframes_match(dataframes=[dataframe_1, dataframe_1.copy(), dataframe_2]))

        dataframe_2 = dataframe_1.copy()
        dataframe_2.iat[1, 2] = np.NaN
        self.assertFalse(hv.dataframes_match(dataframes=[dataframe_1, dataframe_2]))
        self.assertFalse(hv.dataframes_match(dataframes=[dataframe_1, dataframe_1.copy(), dataframe_2]))

        dataframe_2 = dataframe_1.copy()
        dataframe_2.iat[1, 3] = np.NaN
        self.assertFalse(hv.dataframes_match(dataframes=[dataframe_1, dataframe_2]))
        self.assertFalse(hv.dataframes_match(dataframes=[dataframe_1, dataframe_1.copy(), dataframe_2]))

        dataframe_2 = dataframe_1.copy()
        dataframe_2.columns = ['a', 'b', 'c', 'd']
        dataframe_2.index = ['a', 'b', 'c']
        self.assertTrue(hv.dataframes_match(dataframes=[dataframe_1, dataframe_2]))
        self.assertTrue(hv.dataframes_match(dataframes=[dataframe_1, dataframe_2, dataframe_2]))
        self.assertFalse(hv.dataframes_match(dataframes=[dataframe_1, dataframe_2], ignore_indexes=False))
        self.assertFalse(hv.dataframes_match(dataframes=[dataframe_1, dataframe_2],
                                             ignore_column_names=False))
        self.assertFalse(hv.dataframes_match(dataframes=[dataframe_1, dataframe_2],
                                             ignore_indexes=False,
                                             ignore_column_names=False))

    def test_assert_dataframes_match(self):
        dataframe_1 = pd.DataFrame({'col': [1.123456789, 2.123456789, 3.123456789]})

        # test assertion errors when passing in invalid data

        with self.assertRaises(HelpskParamTypeError):
            hv.assert_dataframes_match(dataframes=None)  # noqa

        with self.assertRaises(HelpskParamTypeError):
            hv.assert_dataframes_match(dataframes=dataframe_1)  # noqa

        with self.assertRaises(HelpskParamValueError):
            hv.assert_dataframes_match(dataframes=[])

        with self.assertRaises(HelpskParamValueError):
            hv.assert_dataframes_match(dataframes=[dataframe_1])

        self.assertTrue(hv.assert_dataframes_match(dataframes=[dataframe_1, dataframe_1]) is None)

        dataframe_2 = dataframe_1.copy()
        dataframe_2.iat[0, 0] = np.nan

        with self.assertRaises(HelpskAssertionError):
            hv.assert_dataframes_match(dataframes=[dataframe_1, dataframe_2])
