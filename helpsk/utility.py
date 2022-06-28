"""A collection of system and environment related helper functions.
"""
import inspect
import warnings
import logging
import datetime
import sys
import os
import pickle
from contextlib import contextmanager, redirect_stdout


def read_pickle(path):
    """
    Simple helper function that read's from a pickled object.

    Args:
        path:
            File path where the pickled object will be stored.
    """
    with open(path, 'rb') as handle:
        unpickled_object = pickle.load(handle)
    return unpickled_object


def to_pickle(obj, path):
    """
    Simple helper function that saves a pickled object.

    Args:
        obj:
            the object to save
        path:
            File path where the pickled object will be read from.
    """
    with open(path, 'wb') as handle:
        pickle.dump(obj, handle)


@contextmanager
def redirect_stdout_to_file(file, mode='w'):
    """ Helper context manager that opens a file and redirects standard output to that file.

    Example:

    ```
    with redirect_stdout_to_file(file_name):
        print_dataframe(dataframe)
    ```

    Args:
        file:
            the name and path of the file to open (argument is passed to `open()`)
        mode:
            the mode of the file e.g. `w` (argument is passed to `open()`)
    """
    with open(file, mode) as handle:
        with redirect_stdout(handle):
            yield


@contextmanager
def suppress_stdout():
    """Suppress Output


    ```
    print("Now you see it")
    with suppress_stdout():
        print("Now you don't")
    ```

    code from: https://stackoverflow.com/questions/2125702/how-to-suppress-console-output-in-python
    """
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


@contextmanager
def suppress_warnings():
    """Simple Wrapper around warnings.catch_warnings()"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def is_debugging():
    """
    Returns:
        Returns True if the environment is currently debugging (e.g. set a breakpoint in PyCharm),
        False if not
    """
    for frame in inspect.stack():
        if frame[1].endswith("pydevd.py"):
            return True

    return False


class Timer:
    """
    This class provides way to time the duration of code within the context manager.
    """
    def __init__(self, message, use_logging=False):
        self._message = message
        self._use_logging = use_logging

    def __enter__(self):
        message = f'Timer Started: {self._message}'

        if self._use_logging:
            logging.basicConfig()
            logging.info(message)
        else:
            print(message)
        self._start = datetime.datetime.now()

        return self

    def __exit__(self, *args):
        self._end = datetime.datetime.now()
        self._interval = self._end - self._start
        message = f'Timer Finished ({self._interval.total_seconds():.2f} seconds)'
        if self._use_logging:
            logging.info(message)
        else:
            print(message)
