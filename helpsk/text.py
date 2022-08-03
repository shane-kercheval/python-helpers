"""
This module provides helper functions when working with text.

This module is more concerned with bytes/unicode and text processing, compared with string.py which
is used for string manipulation.

Functions in this file are copied or modified from Chapter 4 of:
        Fluent Python, 2nd ed., by Luciano Ramalho (O'Reilly).
        Copyright 2022 Luciano Ramalho, 978-1-492-05635-5
"""
from unicodedata import normalize, combining
import string


def normalize_equal(str1: str, str2: str, case_fold: bool = True, method: str = 'NFC') -> bool:
    """
    This function compares two strings for equality after normalizing by the given `method`.

    "NFC is the normalization form recommended by the W3C [... and] make comparisons work as
    expected." (Fluent Python, pg 140.)

    "NFKC [normalziation ...] may lose or distort information but [it] can produce convenient
    intermediate representations for searching and indexes". (Fluent Python, pg 140.)

    This function is modified from:
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


def remove_marks(text: str, latin_only: bool = True) -> str:
    """
    Remove all diacritic marks.

    This function and the examples below are copied/modified from:
        Fluent Python, 2nd ed., by Luciano Ramalho (O'Reilly). Pg. 145-146
        Copyright 2022 Luciano Ramalho, 978-1-492-05635-5

    >>> order = '“Herr Voß: • ½ cup of Œtker™ caffè latte • bowl of açaí.”'
    >>> remove_marks(order, latin_only=False)
    '“Herr Voß: • ½ cup of Œtker™ caffe latte • bowl of acai.”'
    >>> remove_marks(order, latin_only=True)
    '“Herr Voß: • ½ cup of Œtker™ caffe latte • bowl of acai.”'

    >>> greek = 'Ζέφυρος, Zéfiro'
    >>> remove_marks(greek, latin_only=False)
    'Ζεφυρος, Zefiro'
    >>> remove_marks(greek, latin_only=True)
    'Ζέφυρος, Zefiro'

    Args:
        text:
            text to remove marks from
        latin_only:
            only remove marks from latin characters.

            "Often the reason to remove diacritics is to change Latin text to pure ASCII, but this
            function [when using `latin_only=False`] also changes non-Latin characters - like Greek
            letters - which will never become ASCII just by losing their accents. So it [often]
            makes sense to analyze each base character and to remove attached marks only if the
            base character is a letter from the Latin alphabet". (Fluent Python pg. 145-146).

    """
    if latin_only:
        norm_text = normalize('NFD', text)
        latin_base = False
        preserve = []
        for c in norm_text:
            if combining(c) and latin_base:
                continue
            preserve.append(c)

            if not combining(c):
                latin_base = c in string.ascii_letters
        shaved = ''.join(preserve)
    else:
        norm_text = normalize('NFD', text)
        shaved = ''.join(c for c in norm_text if not combining(c))

    return normalize('NFC', shaved)


_single_map = str.maketrans(
    """‚ƒ„ˆ‹‘’“”•–—˜›""",
    """'f"^<''""---~>"""
)
_multi_map = str.maketrans({
    '€': 'EUR',
    '…': '...',
    'Æ': 'AE',
    'æ': 'ae',
    'Œ': 'OE',
    'œ': 'oe',
    '™': '(TM)',
    '‰': '<per mille>',
    '†': '**',
    '‡': '***',
})
_multi_map.update(_single_map)


def _dewinize(text: str) -> str:
    """
    Replace Win1252 symbols with ASCII chars or sequences

    This function is copied from:
        Fluent Python, 2nd ed., by Luciano Ramalho (O'Reilly). Pg. 146-147
        Copyright 2022 Luciano Ramalho, 978-1-492-05635-5
    """
    return text.translate(_multi_map)


def asciize(text: str) -> str:
    """
    This function:
        - removes all diacritic marks from latin base characters
        - replaces Win1252 symbols with ASCII chars
        - replaces common symbols in Westrn text (e.g., curly quotes, em dashes, bullets, etc.)
            into ASCII equivalents.

    This function and the examples below are copied/modified from:
        Fluent Python, 2nd ed., by Luciano Ramalho (O'Reilly). Pg. 147
        Copyright 2022 Luciano Ramalho, 978-1-492-05635-5

    >>> order = '“Herr Voß: • ½ cup of Œtker™ caffè latte • bowl of açaí.”'
    >>> _dewinize(order)
    '"Herr Voß: - ½ cup of OEtker(TM) caffè latte - bowl of açaí."'
    >>> asciize(order)
    '"Herr Voss: - 1⁄2 cup of OEtker(TM) caffe latte - bowl of acai."'

Handling a string with Greek and Latin accented characters:

    >>> greek = 'Ζέφυρος, Zéfiro'
    >>> remove_marks(greek, latin_only=False)
    'Ζεφυρος, Zefiro'
    >>> remove_marks(greek, latin_only=True)
    'Ζέφυρος, Zefiro'
    >>> _dewinize(greek)
    'Ζέφυρος, Zéfiro'
    >>> asciize(greek)
    'Ζέφυρος, Zefiro'
    """
    no_marks = remove_marks(_dewinize(text), latin_only=True)
    no_marks = no_marks.replace('ß', 'ss')
    return normalize('NFKC', no_marks)
