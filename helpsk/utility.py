"""A collection of system and environment related helper functions.
"""
import inspect
import warnings
import sys
import os
from contextlib import contextmanager, redirect_stdout


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
