import unittest
from os import remove
import os.path
from helpsk import utility as hu
from tests.helpers import get_test_path


# noinspection PyMethodMayBeStatic
class TestValidation(unittest.TestCase):

    def test_pickle(self):
        obj = "this is an object"

        path = get_test_path() + '/test_files/pickled_object.pkl'
        self.assertFalse(os.path.isfile(path))
        hu.to_pickle(obj=obj, path=path)
        self.assertTrue(os.path.isfile(path))
        unpickled_object = hu.read_pickle(path=path)
        self.assertEqual(obj, unpickled_object)
        remove(path)
        self.assertFalse(os.path.isfile(path))

    def test_is_debugging(self):
        # not sure how to test is_debugging is true, except manually
        self.assertFalse(hu.is_debugging())
