from helpsk.utility import is_debugging
from unittest import TestCase
from os import getcwd


def get_test_path() -> str:
    """Returns the path to /tests folder, adjusting for the difference in the current working directory when
    debugging vs not debugging.
    """
    path = getcwd()
    if not is_debugging():
        path = path + '/tests'

    return path


def subtests_expected_vs_actual(test_case: TestCase,
                                expected_values: list,
                                actual_values: list,
                                **kwargs) -> None:
    """ runs a TestCase.subTest() for each item in expected_values and actual_values and tests that the
    expected_values are equal to the actual_values via TestCase.assertEqual()

    Example (ran within `test_example_file.py -> TestExampleClass -> test_example_function()`):

        ```
        expected_values = [1, 2, 3, 4]
        actual_values = [1, 0, 3, -1]

        subTests_actual_expected(test_case=self, expected_values=expected_values, actual_values=actual_values,
                                 my_param1=True, my_param2='value')

        ```

        The output will be:

        ```
        FAIL: test_example_function(test_example_file.TestExampleClass)
            (index=1, expected=2, actual=0, my_param1=True, my_param2='value')
        AssertionError: 2 != 0

        FAIL: test_example_function(test_example_file.TestExampleClass)
            (index=3, expected=4, actual=-1, my_param1=True, my_param2='value')
        AssertionError: 4 != -1
        ```

    Args:
        test_case:
            the current TestCase object (i.e. pass `self` from within a test function)
        expected_values:
            a list of expected values
        actual_values:
            a list of actual values
        **kwargs:
            a variable list of param/value items that will get passed to TestCase.subTest and printed out for
            failing sub-tests.
    """
    assert len(expected_values) == len(actual_values)

    for index, (expected, actual) in enumerate(zip(expected_values, actual_values)):
        with test_case.subTest(index=index, expected=expected, actual=actual, **kwargs):
            test_case.assertEqual(expected, actual)
