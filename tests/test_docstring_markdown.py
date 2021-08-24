import unittest
from os import remove
from os.path import isfile
from docstring_markdown import build_markdown
from tests.helpers import get_test_path


# noinspection PyMethodMayBeStatic
class TestDocstringMarkdown(unittest.TestCase):
    def test_execute_build(self):

        with open(get_test_path() + '/test_files/execute_build__expected_documentation.md', 'r') as file:
            expected_contents = file.read()

        def test_helper(function):
            test_file = get_test_path() + '/test_files/documentation.md'
            function()
            assert isfile(test_file)
            with open(test_file, 'r') as file:
                test_file_contents = file.read()
            remove(test_file)
            assert expected_contents == test_file_contents

        test_helper(lambda: build_markdown.execute_build(input_path=get_test_path() + '/../helpsk/',
                                                         output_path=get_test_path() + '/test_files'))
