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

        # test numpy array
        assert assertion.any_none_nan(np.array([1, np.nan, None]))
        assert assertion.any_none_nan(np.array([1, np.nan]))
        assert assertion.any_none_nan(np.array([1, None]))
        assert assertion.any_none_nan(np.array([np.nan]))
        assert assertion.any_none_nan(np.array([None]))
        assert assertion.any_none_nan(np.array([]))
        assert not assertion.any_none_nan(np.array([1]))

        # test pandas series
        assert assertion.any_none_nan(pd.Series([1, np.nan, None]))
        assert assertion.any_none_nan(pd.Series([1, np.nan]))
        assert assertion.any_none_nan(pd.Series([1, None]))
        assert assertion.any_none_nan(pd.Series([np.nan]))
        assert assertion.any_none_nan(pd.Series([None]))
        assert assertion.any_none_nan(pd.Series([], dtype=float))
        assert not assertion.any_none_nan(pd.Series([1]))

        # test pandas data.frame
        assert assertion.any_none_nan(pd.DataFrame([[1, np.nan, None], [1, 2, 3]]))
        assert assertion.any_none_nan(pd.DataFrame([[1, np.nan], [1, 2]]))
        assert assertion.any_none_nan(pd.DataFrame([[1, None], [1, 2]]))
        assert assertion.any_none_nan(pd.DataFrame([[np.nan], [1]]))
        assert assertion.any_none_nan(pd.DataFrame([[None], [1]]))
        assert assertion.any_none_nan(pd.DataFrame([], dtype=float))
        assert not assertion.any_none_nan(pd.DataFrame([1]))
        assert not assertion.any_none_nan(pd.DataFrame([[1], [1]]))

    def test_any_empty(self):
        assert True

    def test_raises_exception(self):

        def my_function_exception():
            raise Exception()

        def my_function_runs():
            return True

        assert assertion.raises_exception(my_function_exception)
        assert not assertion.raises_exception(my_function_runs)

    # def test_any_duplicated(self):
    #
    #     values = [1, 2, 3, 4, None]
    #
    #     values = np.array([1, 2, 3, 4, None])
    #     values = np.array([1, 2, 3, 4, np.nan, None, ''])
    #
    #     np.nan is np.nan
    #
    #     [x for x in values if x is not None]
    #
    #
    #     values_2 = values
    #     values.append(5)
    #
    #     values = list[1]
    #
    #     values
    #     values_2
    #
    #     values = np.array(['a', 'b', 'c', 'b'])
    #
    #     assertion.any_duplicated(['a', 'b', 'c'])
    #     assertion.any_duplicated(['a', 'b', 'c', 'b'])
