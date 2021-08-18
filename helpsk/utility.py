"""A collection of system and environment related helper functions.
"""
import inspect


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
