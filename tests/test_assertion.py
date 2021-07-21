import unittest
import numpy as np
import pandas as pd

from helpsk import assertion


# noinspection PyMethodMayBeStatic
class TestAssertion(unittest.TestCase):

    def test_any_none_nan(self):
        assert assertion.any_none_nan(None)
        assert assertion.any_none_nan(np.NaN)

        # test list
        assert assertion.any_none_nan([1, np.nan, None])
        assert assertion.any_none_nan([1, np.nan])
        assert assertion.any_none_nan([1, None])
        assert assertion.any_none_nan([np.nan])
        assert assertion.any_none_nan([None])
        assert assertion.any_none_nan([])
        assert not assertion.any_none_nan([1])
        assert not assertion.any_none_nan([''])

        # test numpy array
        assert assertion.any_none_nan(np.array([1, np.nan, None]))
        assert assertion.any_none_nan(np.array([1, np.nan]))
        assert assertion.any_none_nan(np.array([1, None]))
        assert assertion.any_none_nan(np.array([np.nan]))
        assert assertion.any_none_nan(np.array([None]))
        assert assertion.any_none_nan(np.array([]))
        assert not assertion.any_none_nan(np.array([1]))
        assert not assertion.any_none_nan(np.array(['']))

        # test pandas series
        assert assertion.any_none_nan(pd.Series([1, np.nan, None]))
        assert assertion.any_none_nan(pd.Series([1, np.nan]))
        assert assertion.any_none_nan(pd.Series([1, None]))
        assert assertion.any_none_nan(pd.Series([np.nan]))
        assert assertion.any_none_nan(pd.Series([None]))
        assert assertion.any_none_nan(pd.Series([], dtype=float))
        assert not assertion.any_none_nan(pd.Series([1]))
        assert not assertion.any_none_nan(pd.Series(['']))

        # test pandas data.frame
        assert assertion.any_none_nan(pd.DataFrame([[1, np.nan, None], [1, 2, 3]]))
        assert assertion.any_none_nan(pd.DataFrame([[1, np.nan], [1, 2]]))
        assert assertion.any_none_nan(pd.DataFrame([[1, None], [1, 2]]))
        assert assertion.any_none_nan(pd.DataFrame([[np.nan], [1]]))
        assert assertion.any_none_nan(pd.DataFrame([[None], [1]]))
        assert assertion.any_none_nan(pd.DataFrame([], dtype=float))
        assert not assertion.any_none_nan(pd.DataFrame([1]))
        assert not assertion.any_none_nan(pd.DataFrame([[1], [1]]))
        assert not assertion.any_none_nan(pd.DataFrame([[''], [1]]))

    def test_any_missing(self):
        assert assertion.any_missing(None)
        assert assertion.any_missing(np.NaN)
        assert assertion.any_missing('')

        # test list
        assert assertion.any_missing([1, np.nan, None])
        assert assertion.any_missing([1, np.nan])
        assert assertion.any_missing([1, None])
        assert assertion.any_missing([np.nan])
        assert assertion.any_missing([None])
        assert assertion.any_missing([])
        assert assertion.any_missing([''])
        assert assertion.any_missing(['abc', ''])
        assert assertion.any_missing([1, ''])
        assert not assertion.any_missing([1])
        assert not assertion.any_missing(['a'])

        # test pandas series
        assert assertion.any_missing(pd.Series([1, np.nan, None]))
        assert assertion.any_missing(pd.Series([1, np.nan]))
        assert assertion.any_missing(pd.Series([1, None]))
        assert assertion.any_missing(pd.Series([np.nan]))
        assert assertion.any_missing(pd.Series([None]))
        assert assertion.any_missing(pd.Series([], dtype=float))
        assert assertion.any_missing(pd.Series(['']))
        assert assertion.any_missing(pd.Series(['abc', '']))
        assert assertion.any_missing(pd.Series([1, '']))
        assert not assertion.any_missing(pd.Series([1]))
        assert not assertion.any_missing(pd.Series(['a']))

        # test pandas data.frame
        assert assertion.any_missing(pd.DataFrame([[1, np.nan, None], [1, 2, 3]]))
        assert assertion.any_missing(pd.DataFrame([[1, np.nan], [1, 2]]))
        assert assertion.any_missing(pd.DataFrame([[1, None], [1, 2]]))
        assert assertion.any_missing(pd.DataFrame([[np.nan], [1]]))
        assert assertion.any_missing(pd.DataFrame([[None], [1]]))
        assert assertion.any_missing(pd.DataFrame([], dtype=float))
        assert assertion.any_missing(pd.DataFrame([['abc', ''], ['abc', 'abc']]))
        assert assertion.any_missing(pd.DataFrame(['']))
        assert not assertion.any_missing(pd.DataFrame([1]))
        assert not assertion.any_missing(pd.DataFrame([[1], [1]]))

    def test_any_duplicated(self):
        # test list
        assert not assertion.any_duplicated([''])
        assert not assertion.any_duplicated(['', 1])
        assert not assertion.any_duplicated(['', 1, None])

        assert assertion.any_duplicated(['', 1, ''])
        assert assertion.any_duplicated(['', 1, 1])
        assert assertion.any_duplicated(['', 1, None, None])

        # test pd.Series
        assert not assertion.any_duplicated(pd.Series(['']))
        assert not assertion.any_duplicated(pd.Series(['', 1]))
        assert not assertion.any_duplicated(pd.Series(['', 1, None]))

        assert assertion.any_duplicated(pd.Series(['', 1, '']))
        assert assertion.any_duplicated(pd.Series(['', 1, 1]))
        assert assertion.any_duplicated(pd.Series(['', 1, None, None]))

    def test_raises_exception(self):

        def my_function_exception():
            raise Exception()

        def my_function_runs():
            return True

        assert assertion.raises_exception(my_function_exception)
        assert not assertion.raises_exception(my_function_runs)




