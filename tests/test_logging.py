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

    def test__log_function(self):
        @log_function_call
        def my_function_1(param_1, param_2):
            return param_1, param_2

        @log_function_call
        def my_function_2():
            return 'value'

        # make sure we call the @wraps function
        self.assertEqual(my_function_1.__name__, 'my_function_1')
        self.assertEqual(my_function_2.__name__, 'my_function_2')

        value = my_function_2()
        self.assertEqual(value, 'value')

        a, b = my_function_1(1, param_2=2)
        self.assertEqual(a, 1)
        self.assertEqual(b, 2)
        a, b = my_function_1(param_1='value-1', param_2='value-2')
        self.assertEqual(a, 'value-1')
        self.assertEqual(b, 'value-2')
