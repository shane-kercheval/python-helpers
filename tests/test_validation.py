import unittest
import numpy as np
import pandas as pd

from helpsk import validation as vld


# noinspection PyMethodMayBeStatic
class TestValidation(unittest.TestCase):

    def test_any_none_nan(self):
        assert vld.any_none_nan(None)
        assert vld.any_none_nan(np.NaN)

        # test list
        assert vld.any_none_nan([1, np.nan, None])
        assert vld.any_none_nan([1, np.nan])
        assert vld.any_none_nan([1, None])
        assert vld.any_none_nan([np.nan])
        assert vld.any_none_nan([None])
        assert vld.any_none_nan([])
        assert not vld.any_none_nan([1])
        assert not vld.any_none_nan([''])

        # test numpy array
        assert vld.any_none_nan(np.array([1, np.nan, None]))
        assert vld.any_none_nan(np.array([1, np.nan]))
        assert vld.any_none_nan(np.array([1, None]))
        assert vld.any_none_nan(np.array([np.nan]))
        assert vld.any_none_nan(np.array([None]))
        assert vld.any_none_nan(np.array([]))
        assert not vld.any_none_nan(np.array([1]))
        assert not vld.any_none_nan(np.array(['']))

        # test pandas series
        assert vld.any_none_nan(pd.Series([1, np.nan, None]))
        assert vld.any_none_nan(pd.Series([1, np.nan]))
        assert vld.any_none_nan(pd.Series([1, None]))
        assert vld.any_none_nan(pd.Series([np.nan]))
        assert vld.any_none_nan(pd.Series([None]))
        assert vld.any_none_nan(pd.Series([], dtype=float))
        assert not vld.any_none_nan(pd.Series([1]))
        assert not vld.any_none_nan(pd.Series(['']))

        # test pandas data.frame
        assert vld.any_none_nan(pd.DataFrame([[1, np.nan, None], [1, 2, 3]]))
        assert vld.any_none_nan(pd.DataFrame([[1, np.nan], [1, 2]]))
        assert vld.any_none_nan(pd.DataFrame([[1, None], [1, 2]]))
        assert vld.any_none_nan(pd.DataFrame([[np.nan], [1]]))
        assert vld.any_none_nan(pd.DataFrame([[None], [1]]))
        assert vld.any_none_nan(pd.DataFrame([], dtype=float))
        assert not vld.any_none_nan(pd.DataFrame([1]))
        assert not vld.any_none_nan(pd.DataFrame([[1], [1]]))
        assert not vld.any_none_nan(pd.DataFrame([[''], [1]]))

    def test_assert_not_none_nan(self):
        assert vld.raises_exception(lambda: vld.assert_not_none_nan(None))
        assert vld.raises_exception(lambda: vld.assert_not_none_nan(np.NaN))

        # test list
        assert vld.raises_exception(lambda: vld.assert_not_none_nan([1, np.nan, None]))
        assert vld.raises_exception(lambda: vld.assert_not_none_nan([1, np.nan]))
        assert vld.raises_exception(lambda: vld.assert_not_none_nan([1, None]))
        assert vld.raises_exception(lambda: vld.assert_not_none_nan([np.nan]))
        assert vld.raises_exception(lambda: vld.assert_not_none_nan([None]))
        assert vld.raises_exception(lambda: vld.assert_not_none_nan([]))
        assert not vld.raises_exception(lambda: vld.assert_not_none_nan([1]))
        assert not vld.raises_exception(lambda: vld.assert_not_none_nan(['']))

        # test numpy array
        assert vld.raises_exception(lambda: vld.assert_not_none_nan(np.array([1, np.nan, None])))
        assert vld.raises_exception(lambda: vld.assert_not_none_nan(np.array([1, np.nan])))
        assert vld.raises_exception(lambda: vld.assert_not_none_nan(np.array([1, None])))
        assert vld.raises_exception(lambda: vld.assert_not_none_nan(np.array([np.nan])))
        assert vld.raises_exception(lambda: vld.assert_not_none_nan(np.array([None])))
        assert vld.raises_exception(lambda: vld.assert_not_none_nan(np.array([])))
        assert not vld.raises_exception(lambda: vld.assert_not_none_nan(np.array([1])))
        assert not vld.raises_exception(lambda: vld.assert_not_none_nan(np.array([''])))

        # test pandas series
        assert vld.raises_exception(lambda: vld.assert_not_none_nan(pd.Series([1, np.nan, None])))
        assert vld.raises_exception(lambda: vld.assert_not_none_nan(pd.Series([1, np.nan])))
        assert vld.raises_exception(lambda: vld.assert_not_none_nan(pd.Series([1, None])))
        assert vld.raises_exception(lambda: vld.assert_not_none_nan(pd.Series([np.nan])))
        assert vld.raises_exception(lambda: vld.assert_not_none_nan(pd.Series([None])))
        assert vld.raises_exception(lambda: vld.assert_not_none_nan(pd.Series([], dtype=float)))
        assert not vld.raises_exception(lambda: vld.assert_not_none_nan(pd.Series([1])))
        assert not vld.raises_exception(lambda: vld.assert_not_none_nan(pd.Series([''])))

        # test pandas data.frame
        assert vld.raises_exception(lambda: vld.assert_not_none_nan(pd.DataFrame([[1, np.nan, None], [1, 2, 3]])))  # noqa
        assert vld.raises_exception(lambda: vld.assert_not_none_nan(pd.DataFrame([[1, np.nan], [1, 2]])))
        assert vld.raises_exception(lambda: vld.assert_not_none_nan(pd.DataFrame([[1, None], [1, 2]])))
        assert vld.raises_exception(lambda: vld.assert_not_none_nan(pd.DataFrame([[np.nan], [1]])))
        assert vld.raises_exception(lambda: vld.assert_not_none_nan(pd.DataFrame([[None], [1]])))
        assert vld.raises_exception(lambda: vld.assert_not_none_nan(pd.DataFrame([], dtype=float)))
        assert not vld.raises_exception(lambda: vld.assert_not_none_nan(pd.DataFrame([1])))
        assert not vld.raises_exception(lambda: vld.assert_not_none_nan(pd.DataFrame([[1], [1]])))
        assert not vld.raises_exception(lambda: vld.assert_not_none_nan(pd.DataFrame([[''], [1]])))

    def test_any_missing(self):
        assert vld.any_missing(None)
        assert vld.any_missing(np.NaN)
        assert vld.any_missing('')

        # test list
        assert vld.any_missing([1, np.nan, None])
        assert vld.any_missing([1, np.nan])
        assert vld.any_missing([1, None])
        assert vld.any_missing([np.nan])
        assert vld.any_missing([None])
        assert vld.any_missing([])
        assert vld.any_missing([''])
        assert vld.any_missing(['abc', ''])
        assert vld.any_missing([1, ''])
        assert not vld.any_missing([1])
        assert not vld.any_missing(['a'])

        # test pandas series
        assert vld.any_missing(pd.Series([1, np.nan, None]))
        assert vld.any_missing(pd.Series([1, np.nan]))
        assert vld.any_missing(pd.Series([1, None]))
        assert vld.any_missing(pd.Series([np.nan]))
        assert vld.any_missing(pd.Series([None]))
        assert vld.any_missing(pd.Series([], dtype=float))
        assert vld.any_missing(pd.Series(['']))
        assert vld.any_missing(pd.Series(['abc', '']))
        assert vld.any_missing(pd.Series([1, '']))
        assert not vld.any_missing(pd.Series([1]))
        assert not vld.any_missing(pd.Series(['a']))

        # test pandas data.frame
        assert vld.any_missing(pd.DataFrame([[1, np.nan, None], [1, 2, 3]]))
        assert vld.any_missing(pd.DataFrame([[1, np.nan], [1, 2]]))
        assert vld.any_missing(pd.DataFrame([[1, None], [1, 2]]))
        assert vld.any_missing(pd.DataFrame([[np.nan], [1]]))
        assert vld.any_missing(pd.DataFrame([[None], [1]]))
        assert vld.any_missing(pd.DataFrame([], dtype=float))
        assert vld.any_missing(pd.DataFrame([['abc', ''], ['abc', 'abc']]))
        assert vld.any_missing(pd.DataFrame(['']))
        assert not vld.any_missing(pd.DataFrame([1]))
        assert not vld.any_missing(pd.DataFrame([[1], [1]]))

    def test_any_duplicated(self):
        # test list
        assert not vld.any_duplicated([''])
        assert not vld.any_duplicated(['', 1])
        assert not vld.any_duplicated(['', 1, None])

        assert vld.any_duplicated(['', 1, ''])
        assert vld.any_duplicated(['', 1, 1])
        assert vld.any_duplicated(['', 1, None, None])

        # test pd.Series
        assert not vld.any_duplicated(pd.Series(['']))
        assert not vld.any_duplicated(pd.Series(['', 1]))
        assert not vld.any_duplicated(pd.Series(['', 1, None]))

        assert vld.any_duplicated(pd.Series(['', 1, '']))
        assert vld.any_duplicated(pd.Series(['', 1, 1]))
        assert vld.any_duplicated(pd.Series(['', 1, None, None]))

    def test_assert_not_duplicated(self):
        # test list
        assert not vld.raises_exception(lambda: vld.assert_not_duplicated(['']))
        assert not vld.raises_exception(lambda: vld.assert_not_duplicated(['', 1]))
        assert not vld.raises_exception(lambda: vld.assert_not_duplicated(['', 1, None]))

        assert vld.raises_exception(lambda: vld.assert_not_duplicated(['', 1, '']),
                                    type(AssertionError()))
        assert vld.raises_exception(lambda: vld.assert_not_duplicated(['', 1, 1]),
                                    type(AssertionError()))
        assert vld.raises_exception(lambda: vld.assert_not_duplicated(['', 1, None, None]),
                                    type(AssertionError()))

        # test pd.Series
        assert not vld.raises_exception(lambda: vld.assert_not_duplicated(pd.Series([''])))
        assert not vld.raises_exception(lambda: vld.assert_not_duplicated(pd.Series(['', 1])))
        assert not vld.raises_exception(lambda: vld.assert_not_duplicated(pd.Series(['', 1, None])))

        assert vld.raises_exception(lambda: vld.assert_not_duplicated(pd.Series(['', 1, ''])),
                                    type(AssertionError()))
        assert vld.raises_exception(lambda: vld.assert_not_duplicated(pd.Series(['', 1, 1])),
                                    type(AssertionError()))
        assert vld.raises_exception(lambda: vld.assert_not_duplicated(pd.Series(['', 1, None, None])),
                                    type(AssertionError()))

    def test_raises_exception(self):

        def my_function_exception():
            raise ValueError()

        def my_function_runs():
            return True

        assert vld.raises_exception(my_function_exception)
        # should return True since my_function_exception raises ValueError
        assert vld.raises_exception(my_function_exception, type(ValueError()))
        # should return False since my_function_exception raises ValueError, not TypeError
        assert not vld.raises_exception(my_function_exception, type(TypeError()))
        assert not vld.raises_exception(my_function_runs)

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
        passing_cases = [not vld.raises_exception(lambda: vld.assert_all(case),
                                                        type(AssertionError()))
                         for case in cases]
        assert all(passing_cases)

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
        passing_cases = [vld.raises_exception(lambda: vld.assert_all(case),
                                                    type(AssertionError()))
                         for case in cases]
        assert all(passing_cases)

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
        passing_cases = [not vld.raises_exception(lambda: vld.assert_not_any(case),
                                                        type(AssertionError()))
                         for case in cases]
        assert all(passing_cases)

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
        passing_cases = [vld.raises_exception(lambda: vld.assert_not_any(case),
                                                    type(AssertionError()))
                         for case in cases]
        assert all(passing_cases)
