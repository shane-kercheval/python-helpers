from typing import List


def collapse(*args: List[str], separate: str = '', surround: str = '') -> str:
    """Takes a list of string and concatenates them, separating each string with `separate` and surrounding each string
    with `surround`.

    Keyword arguments:
    *args -- list of strings to concatenate
    separate -- string used to separate each string (default '')
    surround -- string used to surround each string (default '')
    """
    if len(args) == 1 and isinstance(args[0], list):
        args = args[0]
    return separate.join([surround + x + surround for x in args])
