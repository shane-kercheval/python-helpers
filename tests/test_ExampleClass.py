import unittest
from helpsk.example_class import ExampleClass


# noinspection PyMethodMayBeStatic
class TestExampleClass(unittest.TestCase):
    def test_something(self):
        example = ExampleClass()
        self.assertEqual(example.my_variable, 0)
