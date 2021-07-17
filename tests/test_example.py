import unittest
from helpsk import example


# noinspection PyMethodMayBeStatic
class MyTestCase(unittest.TestCase):
    def test_something(self):
        assert example.add_one(14) == 15


if __name__ == '__main__':
    unittest.main()
