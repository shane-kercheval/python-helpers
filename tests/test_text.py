from unicodedata import normalize
import unittest

from helpsk.text import normalize_equal


# noinspection PyMethodMayBeStatic
class TestValidation(unittest.TestCase):

    def test_normalize_equal(self):
        self.assertTrue(normalize_equal('café', 'cafe\u0301'))
        self.assertTrue(normalize_equal('café', 'Cafe\u0301'))
        self.assertFalse(normalize_equal('café', 'Cafe\u0301', case_fold=False))
        ohm = '\u2126'
        omega = 'Ω'
        self.assertNotEqual(ohm, omega)
        self.assertTrue(normalize_equal(ohm, omega))
        self.assertTrue(normalize_equal(ohm, omega, case_fold=False))
        self.assertFalse(normalize_equal('Straße', 'strasse', case_fold=False))
        self.assertTrue(normalize_equal('Straße', 'strasse'))
        self.assertTrue(normalize_equal('a', 'A'))
        self.assertFalse(normalize_equal('a', 'A', case_fold=False))
