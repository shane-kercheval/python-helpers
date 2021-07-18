import unittest
from helpsk import Assertion


# noinspection PyMethodMayBeStatic
class TestAssertion(unittest.TestCase):

    # def test_any_duplicated(self):
    #     Assertion.any_duplicated(['a', 'b'])

    def test_raises_exception(self):

        def my_function_exception():
            raise BaseException()

        def my_function_runs():
            return True

        assert Assertion.raises_exception(my_function_exception)
        assert not Assertion.raises_exception(my_function_runs)
