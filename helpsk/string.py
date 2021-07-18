from typing import List, Union


def collapse(*args: Union[str, List[str]], separate: str = '', surround: str = '') -> str:
    """Takes a list of strings and concatenates them, separating each string with the value of `separate` and
    surrounding each string with the value of `surround`.

    Keyword arguments:
    *args -- list of strings to concatenate
    separate -- string used to separate each string (default '')
    surround -- string used to surround each string (default '')
    """
    if len(args) == 1 and isinstance(args[0], list):
        args = args[0]

    return separate.join([surround + x + surround for x in args])
