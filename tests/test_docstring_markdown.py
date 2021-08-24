import unittest
from os import remove
from os.path import isfile
from docstring_markdown import build_markdown
from tests.helpers import get_test_path


# noinspection PyMethodMayBeStatic
class TestDocstringMarkdown(unittest.TestCase):
    def test_execute_build(self):

        with open(get_test_path() + '/test_files/execute_build__expected_documentation.md', 'r') as file:
            expected_contents_with_toc = file.read()

        with open(get_test_path() + '/test_files/execute_build__expected_documentation__no_toc.md',
                  'r') as file:
            expected_contents_without_toc = file.read()

        def test_helper(function,
                        test_file_name,
                        expected_contents):
            test_file = get_test_path() + test_file_name
            function()
            assert isfile(test_file)
            with open(test_file, 'r') as file_pointer:
                test_file_contents = file_pointer.read()
            remove(test_file)
            assert expected_contents == test_file_contents

        test_helper(lambda: build_markdown.execute_build(input_path=get_test_path() + '/../helpsk/',
                                                         output_path=get_test_path() + '/test_files'),
                    test_file_name='/test_files/documentation.md',
                    expected_contents=expected_contents_with_toc)

        test_helper(lambda: build_markdown.execute_build(input_path=get_test_path() + '/../helpsk/',
                                                         output_path=get_test_path() + '/test_files',
                                                         output_filename='another_file_name.md'),
                    test_file_name='/test_files/another_file_name.md',
                    expected_contents=expected_contents_with_toc)

        test_helper(lambda: build_markdown.execute_build(input_path=get_test_path() + '/../helpsk/',
                                                         output_path=get_test_path() + '/test_files',
                                                         toc_off=True),
                    test_file_name='/test_files/documentation.md',
                    expected_contents=expected_contents_without_toc)

        test_helper(lambda: build_markdown.execute_build(input_path=get_test_path() + '/../helpsk/',
                                                         output_path=get_test_path() + '/test_files',
                                                         output_filename='another_file_name.md',
                                                         toc_off=True),
                    test_file_name='/test_files/another_file_name.md',
                    expected_contents=expected_contents_without_toc)
