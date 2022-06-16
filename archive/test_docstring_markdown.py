import unittest
from os import remove
from os.path import isfile
from docstring_markdown import build_markdown
from docstring_markdown.build_markdown import is_line_docstring, \
    is_single_line_docstring, is_line_function_definition, is_line_class_definition,\
    calculate_indentation_levels
from tests.helpers import get_test_path


# noinspection PyMethodMayBeStatic
class TestDocstringMarkdown(unittest.TestCase):

    @unittest.skip
    def test_execute_build(self):

        with open(get_test_path() + '/test_files/docstring_markdown/execute_build__expected_documentation.md', 'r') as file:
            expected_contents_with_toc = file.read()

        with open(get_test_path() + '/test_files/docstring_markdown/execute_build__expected_documentation__no_toc.md',
                  'r') as file:
            expected_contents_without_toc = file.read()

        def test_helper(function,
                        test_file_name,
                        expected_contents):
            test_file = get_test_path() + test_file_name
            function()
            self.assertTrue(isfile(test_file))
            with open(test_file, 'r') as file_pointer:
                test_file_contents = file_pointer.read()
            remove(test_file)
            self.assertEqual(expected_contents, test_file_contents)

        test_helper(lambda: build_markdown.execute_build(input_path=get_test_path() + '/../helpsk/',
                                                         output_path=get_test_path() + '/test_files'),
                    test_file_name='/test_files/docstring_markdown/documentation.md',
                    expected_contents=expected_contents_with_toc)

        test_helper(lambda: build_markdown.execute_build(input_path=get_test_path() + '/../helpsk/',
                                                         output_path=get_test_path() + '/test_files',
                                                         output_filename='another_file_name.md'),
                    test_file_name='/test_files/docstring_markdown/another_file_name.md',
                    expected_contents=expected_contents_with_toc)

        test_helper(lambda: build_markdown.execute_build(input_path=get_test_path() + '/../helpsk/',
                                                         output_path=get_test_path() + '/test_files',
                                                         toc_off=True),
                    test_file_name='/test_files/docstring_markdown/documentation.md',
                    expected_contents=expected_contents_without_toc)

        test_helper(lambda: build_markdown.execute_build(input_path=get_test_path() + '/../helpsk/',
                                                         output_path=get_test_path() + '/test_files',
                                                         output_filename='another_file_name.md',
                                                         toc_off=True),
                    test_file_name='/test_files/docstring_markdown/another_file_name.md',
                    expected_contents=expected_contents_without_toc)

    def test_is_line_docstring(self):
        self.assertTrue(is_line_docstring('"""'))
        self.assertTrue(is_line_docstring('"""\n'))
        self.assertTrue(is_line_docstring('"""asdf\n'))

    def test_is_single_line_docstring(self):
        self.assertTrue(is_single_line_docstring('"""doc"""'))
        self.assertFalse(is_single_line_docstring('"""'))
        self.assertFalse(is_single_line_docstring('"""doc'))

    def test_is_line_function_definition(self):
        self.assertTrue(is_line_function_definition('def xxx'))
        self.assertTrue(is_line_function_definition('    def xxx('))
        self.assertTrue(is_line_function_definition('        def xxx(adsf...'))

        self.assertFalse(is_line_function_definition('def __xxx'))
        self.assertFalse(is_line_function_definition('    def __xxx('))
        self.assertFalse(is_line_function_definition('        def __xxx(adsf...'))

        self.assertFalse(is_line_function_definition('class xxx'))
        self.assertFalse(is_line_function_definition('    class xxx('))
        self.assertFalse(is_line_function_definition('        class xxx(adsf...'))

    def test_is_line_class_definition(self):
        self.assertFalse(is_line_class_definition('def xxx'))
        self.assertFalse(is_line_class_definition('    def xxx('))
        self.assertFalse(is_line_class_definition('        def xxx(adsf...'))

        self.assertFalse(is_line_class_definition('def __xxx'))
        self.assertFalse(is_line_class_definition('    def __xxx('))
        self.assertFalse(is_line_class_definition('        def __xxx(adsf...'))

        self.assertTrue(is_line_class_definition('class xxx'))
        self.assertTrue(is_line_class_definition('    class xxx('))
        self.assertTrue(is_line_class_definition('        class xxx(adsf...'))

    def test_calculate_indentation_levels(self):
        self.assertEqual(calculate_indentation_levels(leading_spaces=''), 0)
        self.assertEqual(calculate_indentation_levels(leading_spaces='    '), 1)
        self.assertEqual(calculate_indentation_levels(leading_spaces='        '), 2)
        self.assertEqual(calculate_indentation_levels(leading_spaces='            '), 3)
        self.assertEqual(calculate_indentation_levels(leading_spaces='                '), 4)

        with self.assertRaises(AssertionError):
            calculate_indentation_levels(leading_spaces=' ')
        with self.assertRaises(AssertionError):
            calculate_indentation_levels(leading_spaces='  ')
        with self.assertRaises(AssertionError):
            calculate_indentation_levels(leading_spaces='   ')

        with self.assertRaises(AssertionError):
            calculate_indentation_levels(leading_spaces='     ')
        with self.assertRaises(AssertionError):
            calculate_indentation_levels(leading_spaces='      ')
        with self.assertRaises(AssertionError):
            calculate_indentation_levels(leading_spaces='       ')
