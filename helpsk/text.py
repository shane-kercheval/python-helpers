"""
This module provides helper functions when working with text.

This module is more concerned with bytes/unicode and text processing, compared with string.py which is used
for string manipulation.
"""
from unicodedata import normalize


def normalize_equal(str1: str, str2: str, case_fold: bool = True, method: str = 'NFC') -> bool:
    """
    This function compares two strings for equality after normalizing by the given `method`.

    "NFC is the normalization form recommended by the W3C [... and] make comparisons work as expected."
    (Fluent Python, pg 140.)

    "NFKC [normalziation ...] may lose or distort information but [it] can produce convenient intermediate
    representations for searching and indexes". (Fluent Python, pg 140.)

    This function is copied from:
        Fluent Python, 2nd ed., by Luciano Ramalho (O'Reilly). Pg. 144
        Copyright 2022 Luciano Ramalho, 978-1-492-05635-5

    >>> (normalize('NFC', 'café'), normalize('NFC', 'cafe\u0301'))
    ('café', 'café')
    >>> (len('café'), len('cafe\u0301'))
    (4, 5)
    >>> 'café' == 'cafe\u0301'
    False
    >>> normalize_equal('café', 'cafe\u0301')
    True
    >>> normalize_equal('café', 'Cafe\u0301')
    True
    >>> normalize_equal('café', 'Cafe\u0301', case_fold=False)
    False

    >>> ohm = '\u2126'
    >>> omega = 'Ω'
    >>> (ohm, omega)
    ('Ω', 'Ω')
    >>> (normalize('NFC', ohm), normalize('NFC', omega))
    ('Ω', 'Ω')
    >>> ohm == omega
    False
    >>> normalize_equal(ohm, omega)
    True
    >>> normalize_equal(ohm, omega, case_fold=False)
    True

    >>> normalize('NFC', '½')
    '½'
    >>> normalize('NFKC', '½')
    '1⁄2'
    >>> '1⁄2' == '1/2'
    False

    >>> normalize('NFC', 'Straße').casefold()
    'strasse'
    >>> normalize_equal('Straße', 'strasse', case_fold=False)
    False
    >>> normalize_equal('Straße', 'strasse')
    True

    Args:
        str1: the first string value
        str2: the second string value
        case_fold: if True, apply case folding (essentially converting text to lower case)
        method: the method; valid values are 'NFC', 'NFD', 'NFKC', 'NFKD'
    """
    if case_fold:
        return (normalize(method, str1).casefold() == normalize(method, str2).casefold())

    return normalize(method, str1) == normalize(method, str2)
