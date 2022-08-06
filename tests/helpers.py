from __future__ import annotations
import os
import re
from importlib import reload
from unittest import TestCase

import pandas as pd
from matplotlib import pyplot as plt

from collections.abc import Iterable, Callable
from helpsk.pandas import print_dataframe
from helpsk.utility import redirect_stdout_to_file


def get_test_path(file) -> str:
    """Returns the path to /tests folder, adjusting for the difference in the current working
    directory when debugging vs not debugging.
    """
    path = os.getcwd()
    return os.path.join(path, 'tests/test_files', file)


def subtests_expected_vs_actual(test_case: TestCase,
                                expected_values: Iterable,
                                actual_values: Iterable,
                                **kwargs) -> None:
    """ runs a TestCase.subTest() for each item in expected_values and actual_values and tests that
    the expected_values are equal to the actual_values via TestCase.assertEqual()

    Example (ran within `test_example_file.py -> TestExampleClass -> test_example_function()`):

        ```
        expected_values = [1, 2, 3, 4]
        actual_values = [1, 0, 3, -1]

        subTests_actual_expected(
            test_case=self,
            expected_values=expected_values,
            actual_values=actual_values,
            my_param1=True,
            my_param2='value'
        )

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
            a variable list of param/value items that will get passed to TestCase.subTest and
            printed out for failing sub-tests.
    """
    assert len(expected_values) == len(actual_values)

    for index, (expected, actual) in enumerate(zip(expected_values, actual_values)):
        with test_case.subTest(index=index, expected=expected, actual=actual, **kwargs):
            test_case.assertEqual(expected, actual)


def check_plot(
        file_name: str,
        plot_function: Callable,
        set_size_w_h: tuple | None = (10, 6)):
    reload(plt)  # necessary because matplotlib throws strange errors about alpha values

    def clear():
        plt.gcf().clear()
        plt.cla()
        plt.clf()
        plt.close()

    clear()
    if os.path.isfile(file_name):
        os.remove(file_name)
    assert not os.path.isfile(file_name)
    plot_function()
    if set_size_w_h is not None:
        fig = plt.gcf()
        fig.set_size_inches(set_size_w_h[0], set_size_w_h[1])
    plt.savefig(file_name)
    clear()
    assert os.path.isfile(file_name)


def clean_formatted_dataframe(rendered):
    """`This dataframe changes the random code generated when saving formatted dataframes
    (i.e df.style.to_html()). This is necessary because it each time unit tests are ran, the html
    changes, and it is difficult to know if the change/diff occurred because of the random code, or
    because something actually changed.

    For example, it changes the `1ef9B` in

    ```
    '<style type="text/css">\n#T_1ef9B_row0_col0, #T_1ef9B_row0_col1, ...
    ```

    to `99999`:

    ```
    '<style type="text/css">\n#T_99999_row0_col0, #T_99999_row0_col1, ...
    ```
    `"""
    temp = rendered.replace('\n', '')
    if temp.startswith('<style type="text/css"></style>'):
        code = re.\
            sub('">  <.*', '', temp).\
            replace('<style type="text/css"></style><table id="T', '')
    else:
        code = re.sub('_row.*', '_', temp).replace('<style type="text/css">#T', '')

    if code.endswith('_'):
        code = code[:-1]

    return rendered.replace(code, '_99999')


def get_data_credit() -> pd.DataFrame:
    # https://www.openml.org/d/31
    # save locally in case dataset changes or is removed
    # from sklearn.datasets import fetch_openml
    # credit_g = fetch_openml('credit-g', version=1)
    # data = credit_g['data']
    # data['target'] = credit_g['target']
    # data.to_pickle(get_test_path('../test_data/credit.pkl'))
    return pd.read_pickle(get_test_path('../test_data/credit.pkl'))


def get_data_titanic() -> pd.DataFrame:
    # save locally in case dataset changes or is removed
    # from sklearn.datasets import fetch_openml
    # titanic = fetch_openml('titanic', version=1)
    # data = titanic['data']
    # data['survived'] = titanic['target']
    # data.to_pickle(get_test_path('../test_data/titanic.pkl'))
    return pd.read_pickle(get_test_path('../test_data/titanic.pkl'))


def get_data_housing() -> pd.DataFrame:
    # https://www.openml.org/d/537
    # save locally in case dataset changes or is removed
    # from sklearn.datasets import fetch_openml
    # housing = fetch_openml('houses', version=1)
    # data = housing['data']
    # data['target'] = housing['target']
    # data.to_pickle(get_test_path('../test_data/housing.pkl'))
    return pd.read_pickle(get_test_path('../test_data/housing.pkl'))


def helper_test_dataframe(file_name, dataframe):
    if os.path.isfile(file_name):
        os.remove(file_name)

    pd.set_option('display.max_rows', None)

    with redirect_stdout_to_file(file_name):
        print_dataframe(dataframe)
    assert os.path.isfile(file_name)
