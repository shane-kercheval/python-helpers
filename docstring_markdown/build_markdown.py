"""
This script is a command line program that extract's docstrings from Python files and generates documentation
in the form of a markdown file.

See `build` method for example usages.

This was more of a proof of concept than it is actually meant to be used in the package. I'll keep here until
a later time and then refactor accordingly, perhaps into a different repo.
"""

import click
import glob
import re

__author__ = "Shane Kercheval"


def is_line_docstring(line: str) -> bool:
    """Returns True if the line starts with 0 or more spaces and \"\"\"
    """
    return bool(re.search(r'^ *?"""', line))


def is_single_line_docstring(line: str) -> bool:
    """Returns True if the line starts with 0 or more spaces and begins and ends with \"\"\"
    """
    line = line.strip()
    return line.startswith('"""') and line.endswith('"""') and line.count('"""') == 2


def is_line_class_definition(line: str) -> bool:
    return bool(re.search('^( *)class ', line))


def is_line_function_definition(line: str) -> bool:
    """Returns true if the corresponding line (of a python file) is the start of a function definition.

    Excludes functions that start with `__` which indicates a private function.

    Args:
        line: a line in a python file
    """
    return bool(re.search('^( *)def ', line)) and 'def __' not in line


def calculate_indentation_levels(leading_spaces: str) -> int:
    """Returns the number of indentation levels, where a level is 4 spaces. So 0 leading_spaces has a level of
    0; 4 leading spaces has a level of 1, and so on. 


    Args:
        leading_spaces:
            A string containing the leading spaces e.g. '', or '    ', and so on.

    """
    assert leading_spaces.strip() == ''
    assert len(leading_spaces) % 4 == 0
    return int(len(leading_spaces) / 4)


def create_table_of_content_item(leading_spaces, line):
    """Helper function to recreate a Table of Content line item. 
    """
    return f'{leading_spaces}- [{line.strip()}](#{line.strip().replace(" ", "-")})\n'


def execute_build(input_path: str,
                  output_path: str = './',
                  output_filename: str = 'documentation.md',
                  toc_off: bool = False):
    """Main logic for extracting dostrings from python files
    """

    top_level_header = '##'

    if not re.compile(r'/$').search(input_path):
        input_path += '/'

    generated_markdown = []
    generated_table_of_contents = []

    for filename in glob.iglob(input_path + '[!_]*.py', recursive=True):

        def remove(items, items_to_remove):
            """Remove `items_to_remove` from list of `items`"""
            return [item for item in items if item not in items_to_remove]

        friendly_filename = '/'.join(remove(filename.split('/'), ['.', '..']))
        generated_markdown.append(f'{top_level_header} {friendly_filename}\n\n')
        generated_table_of_contents.append(create_table_of_content_item(leading_spaces='',
                                                                        line=friendly_filename))

        with open(filename) as f:
            file_contents = f.readlines()

        # cycle through each line in the Python file
        # if certain patterns are detected (e.g. docstring, class, function) then there may be a nested-while
        # that loops through that section of the file.
        line_number = 0
        while line_number < len(file_contents):
            line = file_contents[line_number]

            if is_line_docstring(line):

                docstring_leading_spaces = re.search(r'(^ *)"""', line).group(1)

                if is_single_line_docstring(line):
                    line = line.removeprefix(docstring_leading_spaces + '"""').removesuffix('"""\n')
                    generated_markdown.append(line + "\n")
                else:
                    line = line.removeprefix(docstring_leading_spaces + '"""')
                    line = line.strip()
                    if line:
                        generated_markdown.append(line + "\n")

                    # until next line is docstring
                    while not is_line_docstring(file_contents[line_number + 1]):
                        line_number += 1
                        line = file_contents[line_number]
                        line = line.removeprefix(docstring_leading_spaces)

                        if line.strip().lower() in ['args:', 'returns:']:
                            line = line.strip().replace(':', '').capitalize()
                            line = f'##### {line}\n\n'

                        generated_markdown.append(line)

                    generated_markdown.append('\n')
                    line_number += 1

            if is_line_class_definition(line):
                leading_spaces = re.search("^( *)(?:def|class)", line).group(1)
                levels = calculate_indentation_levels(leading_spaces)

                line = line.removeprefix(leading_spaces)

                # header
                friendly_name = re.sub(r'([(:]).*', '', line).strip()  # r'(\(|:).*'
                generated_markdown.append(f'\n{top_level_header}{"#" * (levels + 1)} {friendly_name}\n\n')
                toc_line_item = create_table_of_content_item(leading_spaces=leading_spaces + '    ',
                                                             line=friendly_name)
                generated_table_of_contents.append(toc_line_item)

            if is_line_function_definition(line):
                leading_spaces = re.search('^( *)(?:def|class)', line).group(1)
                levels = calculate_indentation_levels(leading_spaces)
                line = line.removeprefix(leading_spaces)

                # header
                friendly_name = re.sub(r'([(:]).*', '', line).strip()
                generated_markdown.append(f'\n{top_level_header}{"#" * (levels + 1)} {friendly_name}\n\n')
                
                generated_markdown.append(f"\n```python\n{line}")

                # until next line is docstring
                while not is_line_docstring(file_contents[line_number + 1]):
                    line_number += 1
                    line = file_contents[line_number]
                    line = line.removeprefix(docstring_leading_spaces)  # noqa
                    generated_markdown.append(line)

                generated_markdown.append("```\n\n")  # end definition code-block

                toc_line_item = create_table_of_content_item(leading_spaces=leading_spaces + '    ',
                                                             line=friendly_name)
                generated_table_of_contents.append(toc_line_item)

            line_number += 1
        generated_markdown.append('\n---\n\n')

    generated_table_of_contents.append('\n\n')

    if not output_path.endswith('/'):
        output_path = output_path + '/'


    next_file_type = 'w'
    if not toc_off:
        with open(output_path + output_filename, 'w') as the_file:
            the_file.writelines(generated_table_of_contents)
        next_file_type = 'a'

    with open(output_path + output_filename, next_file_type) as the_file:
        the_file.writelines(generated_markdown)


@click.group()
def main():
    """
    Converts Python docstrings to a single markdown file (documentation.md)
    """
    pass


@main.command()
@click.argument('input_path')
@click.argument('output_path')
@click.option('-output_filename', help="The name of the output (markdown) file.")
@click.option('-toc_off',
              is_flag=True,
              help='Use this flag to exclude the table of contents from the output.')
def build(input_path, output_path, output_filename, toc_off):
    """
    Command that extract's docstrings from Python files and generates documentation in the form of a markdown
    file.

    Example Usage:

        python3 build_markdown.py build ../helpsk . -output_filename my_docs.md

        python3 build_markdown.py build --help
        python3 build_markdown.py build ../path_to_files
        python3 build_markdown.py build ../path_to_files . -toc_off
        python3 build_markdown.py build ../path_to_files ../path_to_output -output_filename my_docs.md
            -toc_off
    """

    if not output_filename:
        output_filename = 'documentation.md'

    print(f"Path of Python files:        '{input_path}'")
    print(f"Path to place markdown file: '{output_path}'")
    print(f"Markdown file Name:          '{output_filename}'")
    print(f"Include Table of Contents:   {str(not toc_off)}")
    execute_build(input_path, output_path, output_filename, toc_off)


if __name__ == "__main__":
    main()
