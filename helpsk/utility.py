"""A collection of system and environment related helper functions."""

import inspect
import warnings
import datetime
import sys
import os
import pickle
from pathlib import Path
import pandas as pd

from yaml import safe_load
from contextlib import contextmanager, redirect_stdout


def open_yaml(file_path: str) -> dict[str, any]:
    """Open a yaml file via yaml.safe_load."""
    with open(file_path) as f:
        return safe_load(f)


def read_pickle(path: str) -> object:
    """
    Simple helper function that read's from a pickled object.

    Args:
        path:
            File path where the pickled object will be stored.
    """
    with open(path, 'rb') as handle:
        return pickle.load(handle)


def to_pickle(obj: object, path: str) -> None:
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


def dataframe_to_pickle(df: pd.DataFrame, output_directory: str, file_name: str) -> str:
    """
    Takes a Pandas DataFrame and saves it as a pickled object to the directory with
    the file name specified. The output directory is created if it does not yet exist.

    Args:
        df: the Pandas DataFrame to pickle
        output_directory: the directory to save the pickled object
        file_name: the name of the file
    """
    Path(output_directory).mkdir(exist_ok=True)
    file_path = os.path.join(output_directory, file_name)
    df.to_pickle(file_path)
    return file_path


def dataframe_to_csv(df: pd.DataFrame, output_directory: str, file_name: str) -> str:
    """
    Takes a Pandas DataFrame and saves it as a csv file to the directory with the
    file name specified. The output directory is created if it does not yet exist.

    Args:
        df: the Pandas DataFrame to pickle
        output_directory: the directory to save the csv file
        file_name: the name of the file
    """
    Path(output_directory).mkdir(exist_ok=True)
    file_path = os.path.join(output_directory, file_name)
    df.to_csv(file_path, index=False)
    return file_path


def object_to_pickle(obj: object, output_directory: str, file_name: str) -> str:
    """
    Takes a generic object and saves it as a pickled object to the directory with the
    file name specified. The output directory is created if it does not yet exist.

    Args:
        obj: the object to pickle
        output_directory: the directory to save the pickled object
        file_name: the name of the file
    """
    Path(output_directory).mkdir(exist_ok=True)
    file_path = os.path.join(output_directory, file_name)
    to_pickle(obj=obj, path=file_path)
    return file_path


@contextmanager
def redirect_stdout_to_file(file: str, mode: str = 'w') -> None:
    """
    Helper context manager that opens a file and redirects standard output to that file.

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
    with open(file, mode) as handle, redirect_stdout(handle):
        yield


@contextmanager
def suppress_stdout() -> None:
    """
    Suppress Output.

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
def suppress_warnings() -> None:
    """Simple Wrapper around warnings.catch_warnings()."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def is_debugging() -> bool:
    """
    Returns
        Returns True if the environment is currently debugging (e.g. set a breakpoint in PyCharm),
        False if not.
    """
    return any(frame[1].endswith('pydevd.py') for frame in inspect.stack())


def repr(instance: object) -> str:  # noqa: A001
    """
    Method can be used to build a standard __repr__ function from within a class.

    This function is modified from:
        Fluent Python, 2nd ed., by Luciano Ramalho (O'Reilly). Pg. 189
        Copyright 2022 Luciano Ramalho, 978-1-492-05635-5

    Examples:
    >>> class Example:
    ...     def __init__(self, x: int, y: int):
    ...         self.x = x
    ...         self.y = y
    ...     def __repr__(self) -> str:
    ...         return repr(self)
    >>> print(f"{Example(1, 2)!r}")
    Example(
        x = 1,
        y = 2,
    )
    >>>

    Args:
        instance: the instance e.g. self
    """
    cls = instance.__class__
    cls_name = cls.__name__
    indent = ' ' * 4
    rep = [f'{cls_name}(']
    for field in instance.__dict__.items():
        field_name = field[0]
        field_value = field[1]
        rep += [f'{indent}{field_name} = {field_value!r},']
    rep += [')']
    return '\n'.join(rep)


class Timer:
    """Class provides way to time the duration of code within the context manager."""

    def __init__(self, message: str):
        self._message = message

    def __enter__(self):
        print(f'Timer Started: {self._message}')
        self._start = datetime.datetime.now()
        return self

    def __exit__(self, *args: tuple):
        self._end = datetime.datetime.now()
        self._interval = self._end - self._start
        print(f'Timer Finished ({self._interval.total_seconds():.2f} seconds)')
