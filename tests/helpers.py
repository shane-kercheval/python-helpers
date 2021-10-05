import os
from importlib import reload
from os import getcwd
from typing import Callable, Union
from unittest import TestCase

import pandas as pd
from matplotlib import pyplot as plt

from helpsk.pandas import print_dataframe
from helpsk.utility import is_debugging, redirect_stdout_to_file


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


def check_plot(file_name: str, plot_function: Callable, set_size_w_h: Union[tuple, None] = (10, 6)):
    reload(plt)  # necessary because matplotlib throws strange errors about alpha values

    def clear():
        plt.gcf().clear()
        plt.cla()
        plt.clf()
        plt.close()

    clear()
    if os.path.isfile(file_name):
        os.remove(file_name)
    assert os.path.isfile(file_name) is False
    plot_function()
    if set_size_w_h is not None:
        fig = plt.gcf()
        fig.set_size_inches(set_size_w_h[0], set_size_w_h[1])
    plt.savefig(file_name)
    clear()
    assert os.path.isfile(file_name)


def get_data_credit() -> pd.DataFrame:
    # https://www.openml.org/d/31
    # save locally in case dataset changes or is removed
    # from sklearn.datasets import fetch_openml
    # credit_g = fetch_openml('credit-g', version=1)
    # data = credit_g['data']
    # data['target'] = credit_g['target']
    # data.to_pickle(get_test_path() + '/test_data/credit.pkl')
    return pd.read_pickle(get_test_path() + '/test_data/credit.pkl')


def get_data_titanic() -> pd.DataFrame:
    # save locally in case dataset changes or is removed
    # from sklearn.datasets import fetch_openml
    # titanic = fetch_openml('titanic', version=1)
    # data = titanic['data']
    # data['survived'] = titanic['target']
    # data.to_pickle(get_test_path() + '/test_data/titanic.pkl')
    return pd.read_pickle(get_test_path() + '/test_data/titanic.pkl')


def get_data_housing() -> pd.DataFrame:
    # https://www.openml.org/d/537
    # save locally in case dataset changes or is removed
    # from sklearn.datasets import fetch_openml
    # housing = fetch_openml('houses', version=1)
    # data = housing['data']
    # data['target'] = housing['target']
    # data.to_pickle(get_test_path() + '/test_data/housing.pkl')
    return pd.read_pickle(get_test_path() + '/test_data/housing.pkl')


def helper_test_dataframe(file_name, dataframe):
    if os.path.isfile(file_name):
        os.remove(file_name)
    with redirect_stdout_to_file(file_name):
        print_dataframe(dataframe)
    assert os.path.isfile(file_name)
