"""Contains functions that are useful for logging."""

from __future__ import annotations
import logging
import datetime
from collections.abc import Callable
from functools import wraps


def log_function_call(max_param_chars: int = 100) -> Callable:
    """
    Decorator to log the function name and paramters of the function when called.

    Args:
        max_param_chars: the maximum number of characters to log for each param.
    """
    if isinstance(max_param_chars, int):
        def decorating_function(func: Callable) -> Callable:
            return _log_function_call_decorate(func=func, max_param_chars=max_param_chars)
        return decorating_function
    if callable(max_param_chars):
        # The func was passed in directly via the max_param_chars argument
        # This happens when the decorator is called without parentheses
        func, max_param_chars = max_param_chars, 100
        return _log_function_call_decorate(func=func, max_param_chars=max_param_chars)

    raise ValueError(f"Expected type of `int` or `callable`, received {type(max_param_chars)}")


def _log_function_call_decorate(func: Callable, max_param_chars: int) -> Callable:
    """
    Helper function that contains the actual decorator for `log_function_call`. It exists
    because log_function_call allows arguments and using the decorator without parentheses results
    in max_param_chars being set to the function we weant to wrap. This function allows
    log_function_call to the the correct parameters and pass them in to this function accordingly.
    I copied a similar pattern from the lru_cache decorator in
    https://github.com/python/cpython/blob/main/Lib/functools.py.
    """
    @wraps(func)
    def wrapper(*args: tuple, **kwargs: dict) -> Callable:
        func.__name__
        if len(args) == 0 and len(kwargs) == 0:
            _log_function(
                function_name=func.__name__,
                params=None,
                max_param_chars=max_param_chars,
            )
        else:
            parameters = {}
            if len(args) > 0:
                parameters['args'] = args
            if len(kwargs) > 0:
                parameters.update(kwargs)
            _log_function(
                function_name=func.__name__,
                params=parameters,
                max_param_chars=max_param_chars,
            )
        return func(*args, **kwargs)
    return wrapper


def _log_function(
        function_name: str,
        params: dict | None = None,
        max_param_chars: int = 100) -> None:
    """
    Meant to be used at the start of the calling function; calls logging.info and
    passes the name of the function and optional parameter names/values.

    Args:
        function_name:
            the name of the function
        params:
            a dictionary containing the names of the function parameters (as dictionary keys) and
            the parameter values (as dictionary values).
        max_param_chars:
            the maximum number of characters to log for each param.
    """
    logging.info(f"FUNCTION: {function_name.upper()}")
    if params is not None:
        logging.info("PARAMS:")
        for key, value in params.items():
            value = str(value)  # noqa: PLW2901
            if len(value) > max_param_chars:
                value = value[0:max_param_chars] + '...'  # noqa: PLW2901
            logging.info(f"    {key}: {value}")


class Timer:
    """Provides way to time the duration of code within the context manager."""

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

    def __exit__(self, *args: tuple):
        self._end = datetime.datetime.now()
        self._interval = self._end - self._start
        message = ''
        if self._post_message:
            message = self._message + " "
        logging.info(f'Timer Finished: {message}({self._interval.total_seconds():.2f} seconds)')


def log_timer(function: Callable) -> Callable:
    """
    Function should be used as a decorator to log the time/duration of a function call.

    Args:
        function: function that is decorated
    """
    @wraps(function)
    def wrapper(*args: tuple, **kwargs: dict) -> Callable:
        with Timer(f"FUNCTION={function.__module__}:{function.__name__}", post_message=True):
            return function(*args, **kwargs)
    return wrapper
