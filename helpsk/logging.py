import logging
import datetime
from typing import Callable, Union
from functools import wraps


def log_info(message: str):
    """
    Calls logging.info. Use this function rather than logging.info directly in case a production
    environment requires a different library/setup.

    Args:
        message: the message to log
    """
    logging.info(message)


def log_function_call(function: Callable) -> Callable:
    """
    This function should be used as a decorator to log the function name and paramters of the function when
    called.

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


def _log_function(function_name: str, params: Union[dict, None] = None):
    """
    This function is meant to be used at the start of the calling function; calls log_info and passes the
    name of the function and optional parameter names/values.

    Args:
        func_name:
            the name of the function
        params:
            a dictionary containing the names of the function parameters (as dictionary keys) and the
            parameter values (as dictionary values).
    """
    log_info(f"FUNCTION: {function_name.upper()}")
    if params is not None:
        log_info("PARAMS:")
        for key, value in params.items():
            log_info(f"    {key}: {value}")


class Timer:
    """
    This class provides way to time the duration of code within the context manager.
    """
    def __init__(self, message, include_message_at_finish=False):
        self._message = message
        self._include_message_at_finish = include_message_at_finish

    def __enter__(self):
        logging.basicConfig()
        log_info(f'Timer Started: {self._message}')
        self._start = datetime.datetime.now()

        return self

    def __exit__(self, *args):
        self._end = datetime.datetime.now()
        self._interval = self._end - self._start
        message = ''
        if self._include_message_at_finish:
            message = self._message + " "

        log_info(f'Timer Finished: {message}({self._interval.total_seconds():.2f} seconds)')


def log_timer(function: Callable) -> Callable:
    @wraps(function)
    def wrapper(*args, **kwargs):
        with Timer(f"FUNCTION={function.__module__}:{function.__name__}", include_message_at_finish=True):
            results = function(*args, **kwargs)
        return results

    return wrapper
