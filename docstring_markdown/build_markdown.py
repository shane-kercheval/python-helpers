"""
Script that extract's docstrings from Python files and generates documentation in the form of a markdown file.

See `build` method for example usages. 
"""

import click
import glob
import re

__author__ = "Shane Kercheval"

top_level_header = '##'


def is_line_docstring(line: str) -> bool:
    """Returns True if the line starts with 0 or more spaces and \"\"\"
    """
    return bool(re.search(r'^ *?"""', line))


assert is_line_docstring('"""')
assert is_line_docstring('"""\n')
assert is_line_docstring('"""asdf\n')


def is_single_line_docstring(line: str) -> bool:
    """Returns True if the line starts with 0 or more spaces and begins and ends with \"\"\"
    """
    line = line.strip()
    return line.startswith('"""') and line.endswith('"""') and line.count('"""') == 2


assert is_single_line_docstring('"""doc"""')
assert not is_single_line_docstring('"""')
assert not is_single_line_docstring('"""doc')


def is_line_class_definition(line: str) -> bool:
    return bool(re.search('^( *)(?:class )', line))


def is_line_function_definition(line: str) -> bool:
    """Returns true if the corresponding line (of a python file) is the start of a function definition.

    Excludes functions that start with `__` which indicates a private function.

    Args:
        line: a line in a python file
    """
    return bool(re.search('^( *)(?:def )', line)) and 'def __' not in line


assert is_line_function_definition('def xxx')
assert is_line_function_definition('    def xxx(')
assert is_line_function_definition('        def xxx(adsf...')

assert not is_line_function_definition('def __xxx')
assert not is_line_function_definition('    def __xxx(')
assert not is_line_function_definition('        def __xxx(adsf...')

assert not is_line_function_definition('class xxx')
assert not is_line_function_definition('    class xxx(')
assert not is_line_function_definition('        class xxx(adsf...')


def calculate_levels(leading_spaces: str) -> int:
    assert leading_spaces.strip() == ''
    assert len(leading_spaces) % 4 == 0
    return int(len(leading_spaces) / 4)


def table_of_content_item(leading_spaces, line):
    return f'{leading_spaces}- [{line.strip()}](#{line.strip().replace(" ", "-")})\n'


def execute_build(input_path: str,
                  output_path: str = './',
                  output_filename: str = 'documentation.md',
                  toc_off: bool = False):

    if not re.compile(r'/$').search(input_path):
        input_path += '/'

    output = []
    table_of_contents = []

    for filename in glob.iglob(input_path + '[!_]*.py', recursive=True):

        def remove(list, items_to_remove):
            return [item for item in list if item not in items_to_remove]

        friendly_filename = '/'.join(remove(filename.split('/'), ['.', '..']))
        output.append(f'{top_level_header} {friendly_filename}\n\n')

        table_of_contents.append(table_of_content_item(leading_spaces='', line=friendly_filename))

        with open(filename) as f:
            file_contents = f.readlines()

        line_number = 0
        while line_number < len(file_contents):
            line = file_contents[line_number]

            if is_line_docstring(line):

                docstring_leading_spaces = re.search(r'(^ *)"""', line).group(1)

                if is_single_line_docstring(line):
                    line = line.removeprefix(docstring_leading_spaces + '"""').removesuffix('"""\n')
                    output.append(line + "\n")
                else:
                    line = line.removeprefix(docstring_leading_spaces + '"""')
                    line = line.strip()
                    if line:
                        output.append(line + "\n")

                    # until next line is docstring
                    while not is_line_docstring(file_contents[line_number + 1]):
                        line_number += 1
                        line = file_contents[line_number]
                        line = line.removeprefix(docstring_leading_spaces)

                        if line.strip().lower() in ['args:', 'returns:']:
                            line = line.strip().replace(':', '').capitalize()
                            line = f'##### {line}\n\n'

                        output.append(line)

                    output.append('\n')
                    line_number += 1

            if is_line_class_definition(line):
                leading_spaces = re.search("^( *)(?:def|class)", line).group(1)
                levels = calculate_levels(leading_spaces)

                line = line.removeprefix(leading_spaces)

                # header
                friendly_name = re.sub(r'(\(|:).*', '', line).strip()
                output.append(f'\n{top_level_header}{"#" * (levels + 1)} {friendly_name}\n\n')
                # output.append("```\n\n")  # end definition code-block
                # output.append(f"\n```\n{line}")

                table_of_contents.append(table_of_content_item(leading_spaces=leading_spaces + '    ',
                                                               line=friendly_name))

            if is_line_function_definition(line):
                leading_spaces = re.search('^( *)(?:def|class)', line).group(1)
                levels = calculate_levels(leading_spaces)
                line = line.removeprefix(leading_spaces)

                # header
                friendly_name = re.sub(r'(\(|:).*', '', line).strip()
                output.append(f'\n{top_level_header}{"#" * (levels + 1)} {friendly_name}\n\n')
                
                output.append(f"\n```python\n{line}")

                # until next line is docstring
                while not is_line_docstring(file_contents[line_number + 1]):
                    line_number += 1
                    line = file_contents[line_number]
                    line = line.removeprefix(docstring_leading_spaces)
                    output.append(line)

                output.append("```\n\n")  # end definition code-block

                table_of_contents.append(table_of_content_item(leading_spaces=leading_spaces + '    ',
                                                               line=friendly_name))

            line_number += 1
        output.append('\n---\n\n')

    table_of_contents.append('\n\n')

    if not output_path.endswith('/'):
        output_path = output_path + '/'

    if not toc_off:
        with open(output_path + output_filename, 'w') as the_file:
            the_file.writelines(table_of_contents)

        file_type = 'a'
    else:
        file_type = 'w'

    with open(output_path + output_filename, file_type) as the_file:
        the_file.writelines(output)


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
@click.option('-toc_off', is_flag=True, help='Use this flag to exclude the table of contents from the output.')
def build(input_path, output_path, output_filename, toc_off):
    """
    Command that extract's docstrings from Python files and generates documentation in the form of a markdown
    file.

    Example Usage:

        python3 build_markdown.py build --help
        python3 build_markdown.py build ../path_to_files
        python3 build_markdown.py build ../path_to_files . -toc_off
        python3 build_markdown.py build ../path_to_files ../path_to_output -output_filename my_docs.md -toc_off
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