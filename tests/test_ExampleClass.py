import unittest
from helpsk import ExampleClass


# noinspection PyMethodMayBeStatic
class TestExampleClass(unittest.TestCase):
    def test_something(self):
        example = ExampleClass()
        assert example.my_variable == 0
