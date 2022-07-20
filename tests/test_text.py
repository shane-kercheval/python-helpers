import unittest

from helpsk.text import normalize_equal, remove_marks, asciize


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

    def test_remove_marks(self):
        order = '“Herr Voß: • ½ cup of Œtker™ caffè latte • bowl of açaí.”'
        expected = '“Herr Voß: • ½ cup of Œtker™ caffe latte • bowl of acai.”'
        self.assertEqual(expected, remove_marks(order, latin_only=False))
        self.assertEqual(expected, remove_marks(order, latin_only=True))
        greek = 'Ζέφυρος, Zéfiro'
        self.assertEqual('Ζεφυρος, Zefiro', remove_marks(greek, latin_only=False))
        self.assertEqual('Ζέφυρος, Zefiro', remove_marks(greek, latin_only=True))

    def test_asciize(self):
        order = '“Herr Voß: • ½ cup of Œtker™ caffè latte • bowl of açaí.”'
        self.assertEqual('"Herr Voss: - 1⁄2 cup of OEtker(TM) caffe latte - bowl of acai."', asciize(order))
        greek = 'Ζέφυρος, Zéfiro'
        self.assertEqual('Ζέφυρος, Zefiro', asciize(greek))
