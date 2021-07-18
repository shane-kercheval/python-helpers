from typing import List, Union, Callable


def raises_exception(function: Callable):
    try:
        function()
        return False
    except:  # noqa
        return True


def any_duplicated(thing):
    raise NotImplementedError()


def assert_none_missing(thing):
    raise NotImplementedError()


def assert_none_duplicated(thing):
    raise NotImplementedError()

