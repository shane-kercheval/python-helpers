from functools import lru_cache
import functools
import logging.config
import time
import unittest

from helpsk.logging import Timer, log_timer, log_function_call
from tests.helpers import get_test_path

logging.config.fileConfig(get_test_path("logging/test_logging.conf"),
                          defaults={'logfilename': get_test_path("logging/log.log")},
                          disable_existing_loggers=False)

class TestLogging(unittest.TestCase):

    def test__timer(self):
        print('\n')

        with Timer("testing timer"):
            time.sleep(0.05)

        with Timer("testing another timer") as timer:
            time.sleep(0.1)

        self.assertIsNotNone(timer._interval)

    def test__timer_decorator(self):
        print('\n')

        @log_timer
        def my_function_1(param_1, param_2):
            time.sleep(0.1)
            return param_1, param_2

        @log_timer
        def my_function_2():
            time.sleep(0.1)

        # make sure we call the @wraps function
        self.assertEqual(my_function_1.__name__, 'my_function_1')
        self.assertEqual(my_function_2.__name__, 'my_function_2')

        my_function_2()

        a, b = my_function_1(1, param_2=2)
        self.assertEqual(a, 1)
        self.assertEqual(b, 2)
        a, b = my_function_1(param_1='value-1', param_2='value-2')
        self.assertEqual(a, 'value-1')
        self.assertEqual(b, 'value-2')

    def test__log_function__without_params(self):
        @log_function_call
        def my_function_1_no_params(param_1, param_2):
            return param_1, param_2

        @log_function_call
        def my_function_2_no_params():
            return 'value'

        # make sure we call the @wraps function
        self.assertEqual(my_function_1_no_params.__name__, 'my_function_1_no_params')
        self.assertEqual(my_function_2_no_params.__name__, 'my_function_2_no_params')

        value = my_function_2_no_params()
        self.assertEqual(value, 'value')

        a, b = my_function_1_no_params(1, param_2=2)
        self.assertEqual(a, 1)
        self.assertEqual(b, 2)
        a, b = my_function_1_no_params(param_1='value-1', param_2='value-2')
        self.assertEqual(a, 'value-1')
        self.assertEqual(b, 'value-2')

    def test__log_function__with_params(self):
        @log_function_call()
        def my_function_1_params(param_1, param_2):
            return param_1, param_2

        @log_function_call()
        def my_function_2_params():
            return 'value'

        # make sure we call the @wraps function
        self.assertEqual(my_function_1_params.__name__, 'my_function_1_params')
        self.assertEqual(my_function_2_params.__name__, 'my_function_2_params')

        value = my_function_2_params()
        self.assertEqual(value, 'value')

        a, b = my_function_1_params(1, param_2=2)
        self.assertEqual(a, 1)
        self.assertEqual(b, 2)
        a, b = my_function_1_params(param_1='value-1', param_2='value-2')
        self.assertEqual(a, 'value-1')
        self.assertEqual(b, 'value-2')

    def test__log_function__long_value(self):
        """
        This function tests the log_function_call decorator when the parameter values are longer
        than the maximum value that is set to log for each param.
        """

        # test without parentheses
        @log_function_call
        def my_function_long_value(param_1):
            return param_1

        value = 'test' * 10_000
        result = my_function_long_value(param_1=value)
        assert value == result

        # test with parentheses
        @log_function_call()
        def my_function_long_value(param_1):
            return param_1

        value = 'test' * 10_000
        result = my_function_long_value(param_1=value)
        assert value == result

        # test with max_param_chars value
        @log_function_call(max_param_chars=10)
        def my_function_long_value(param_1):
            return param_1

        value = 'test' * 10_000
        result = my_function_long_value(param_1=value)
        assert value == result
