from typing import List, Union, Callable


def raises_exception(function: Callable):
    """Returns True if `function` raises an Exception; returns False if `function` runs without raising an Exzception.

    Keyword arguments:
    function -- a function
    """
    try:
        function()
        return False
    except:  # noqa
        return True


def any_duplicated(thing):
    raise NotImplementedError()


def assert_not_any(thing):
    """Raises Exception if any values are true
    """
    raise NotImplementedError()


def assert_identical(thing):
    """Raises Exception if xyz is not identical
    """    
    raise NotImplementedError()


def assert_none_missing(thing, empty_string_as_missing: bool = True):
    """Raises Exception if any values are missing.

    Keyword arguments:
    empty_string_as_missing -- if True, treats empty string as missing value
    """
    raise NotImplementedError()


def assert_none_duplicated(thing, remove_missing_values: bool = True):
    """Raises Exception if any values are missing.

    Keyword arguments:
    remove_missing_values -- if True, removes missing values before checking if duplicated
    """

    # if remove_missing_values is False throw exception if more than one are missing?
    raise NotImplementedError()
