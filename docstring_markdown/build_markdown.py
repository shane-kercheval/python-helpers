import click
import glob
import re

__author__ = "Shane Kercheval"

top_level_header = '##'


def is_line_docstring(line: str) -> bool:
    """Returns True if the line starts with 0 or more spaces and \"\"\"
    """
    return bool(re.search(r'^ *?"""', line))


def is_single_line_docstring(line: str) -> bool:
    """Returns True if the line starts with 0 or more spaces and begins and ends with \"\"\"
    """
    line = line.strip()
    return line.startswith('"""') and line.endswith('"""')

def is_line_class_definition(line: str) -> bool:
    return bool(re.search('^( *)(?:class )', line))


def is_line_function_definition(line: str) -> bool:
    return bool(re.search('^( *)(?:def )', line))


def calculate_levels(leading_spaces: str) -> int:
    assert leading_spaces.strip() == ''
    assert len(leading_spaces) % 4 == 0
    return int(len(leading_spaces) / 4)


def execute_build(path):
    if not re.compile(r'/$').search(path):
        path += '/'

    output = []

    for filename in glob.iglob(path + '[!_]*.py', recursive=True):

        def remove(list, items_to_remove):
            return [item for item in list if item not in items_to_remove]

        friendly_filename = '/'.join(remove(filename.split('/'), ['.', '..']))
        output.append(f'{top_level_header} {friendly_filename}\n\n')

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
                        output.append(line)

                    output.append('\n')
                    line_number += 1

            if is_line_class_definition(line):
                leading_spaces = re.search('^( *)(?:def|class)', line).group(1)
                levels = calculate_levels(leading_spaces)

                line = line.removeprefix(leading_spaces)

                # header
                friendly_name = re.sub(r'(\(|:).*', '', line).strip()
                output.append(f'\n{top_level_header}{"#" * (levels + 1)} {friendly_name}\n\n')
                # output.append("```\n\n")  # end definition code-block
                # output.append(f"\n```\n{line}")

            if is_line_function_definition(line):
                leading_spaces = re.search('^( *)(?:def|class)', line).group(1)
                levels = calculate_levels(leading_spaces)

                line = line.removeprefix(leading_spaces)

                # header
                friendly_name = re.sub(r'(\(|:).*', '', line).strip()
                output.append(f'\n{top_level_header}{"#" * (levels + 1)} {friendly_name}\n\n')
                
                output.append(f"\n```\n{line}")

                # until next line is docstring
                while not is_line_docstring(file_contents[line_number + 1]):
                    line_number += 1
                    line = file_contents[line_number]
                    line = line.removeprefix(docstring_leading_spaces)
                    output.append(line)

                output.append("```\n\n")  # end definition code-block


            #
            #
            #
            #     if docstring_leading_spaces:
            #         line = line.removeprefix(docstring_leading_spaces)
            #
            #     if is_single_line_docstring(line):
            #         line = line.removesuffix('"""\n')
            #         in_docstring = False
            #
            #     line = re.sub('^"""\n?', '', line)
            #     line = line.removesuffix('\n')
            #
            #     if line:
            #         output.append(line + "\n")
            # elif in_function_or_class:
            #     # use the previous leading_spaces and remove accordingly, save the rest
            #     output.append(line.removeprefix(leading_spaces))
            # else:
            #     function_or_class = re.search('^( *)(?:def|class)', line)
            #     if function_or_class:
            #         in_function_or_class = True
            #         # we are in the first line of the function/class (otherwise we'd be in the elif above)
            #         # let's first add the header, then
            #         # let's wrap the entire function/class def in a code-block
            #         leading_spaces = function_or_class.group(1)
            #         levels = calculate_levels(leading_spaces)
            #
            #         line = line.removeprefix(leading_spaces)
            #
            #         # header
            #         friendly_name = re.sub(r'(\(|:).*', '', line).strip()
            #         output.append(f'\n{top_level_header}{"#" * (levels + 1)} {friendly_name}\n')
            #
            #         output.append("```\n\n")  # end definition code-block
            #
            #         output.append(f"\n```\n{line}")


            line_number += 1
        output.append('\n---\n\n')

    with open('documentation.md', 'w') as the_file:
        the_file.writelines(output)

@click.group()
def main():
    """
    Converts Python docstrings to a single markdown file (documentation.md)
    """
    pass


@main.command()
@click.argument('path')
@click.option('-option_a', help='This is option_a')
#@click.option('-option_a', help='This is option_a', required=True)
def build(path, option_a):
    """path to files/folders of python files"""
    #click.echo(option_a is None)
    execute_build(path)

#    click.echo(path) 


if __name__ == "__main__":
    main()
