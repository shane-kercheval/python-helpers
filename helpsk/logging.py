from __future__ import annotations
import logging
import datetime
from collections.abc import Callable
from functools import wraps


def log_function_call(function: Callable) -> Callable:
    """
    This function should be used as a decorator to log the function name and paramters of the
    function when called.

    Args: function that is decorated
    """
    @wraps(function)
    def wrapper(*args, **kwargs):
        function.__name__
        if len(args) == 0 and len(kwargs) == 0:
            _log_function(function_name=function.__name__, params=None)
        else:
            parameters = dict()
            if len(args) > 0:
                parameters['args'] = args
            if len(kwargs) > 0:
                parameters.update(kwargs)
            _log_function(function_name=function.__name__, params=parameters)

        return function(*args, **kwargs)
    return wrapper


def _log_function(function_name: str, params: dict | None = None):
    """
    This function is meant to be used at the start of the calling function; calls logging.info and
    passes the name of the function and optional parameter names/values.

    Args:
        func_name:
            the name of the function
        params:
            a dictionary containing the names of the function parameters (as dictionary keys) and
            the parameter values (as dictionary values).
    """
    logging.info(f"FUNCTION: {function_name.upper()}")
    if params is not None:
        logging.info("PARAMS:")
        for key, value in params.items():
            logging.info(f"    {key}: {value}")


class Timer:
    """
    This class provides way to time the duration of code within the context manager.
    """
    def __init__(self, message: str, post_message: bool = False):
        """
        Args:
            message: message to show when timer starts
            post_message: if True, include same message when timer ends.
        """
        self._message = message
        self._post_message = post_message

    def __enter__(self):
        logging.basicConfig()
        logging.info(f'Timer Started: {self._message}')
        self._start = datetime.datetime.now()
        return self

    def __exit__(self, *args):
        self._end = datetime.datetime.now()
        self._interval = self._end - self._start
        message = ''
        if self._post_message:
            message = self._message + " "
        logging.info(f'Timer Finished: {message}({self._interval.total_seconds():.2f} seconds)')


def log_timer(function: Callable) -> Callable:
    """
    This function should be used as a decorator to log the time/duration of a function call.

    Args: function that is decorated
    """
    @wraps(function)
    def wrapper(*args, **kwargs):
        with Timer(f"FUNCTION={function.__module__}:{function.__name__}", post_message=True):
            results = function(*args, **kwargs)
        return results

    return wrapper
