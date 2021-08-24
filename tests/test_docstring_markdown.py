import unittest

from docstring_markdown import build_markdown
from tests.helpers import get_test_path


# noinspection PyMethodMayBeStatic
class TestExampleClass(unittest.TestCase):
    def test_something(self):
        build_markdown.execute_build(input_path=get_test_path() + '/../helpsk/',
                                     output_path=get_test_path() + '/test_files')

