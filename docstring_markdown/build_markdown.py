import click
import glob
import re

__author__ = "Shane Kercheval"

top_level_header = '##'


def is_single_line_docstring(line):
    return bool(re.search('^""".*"""\n$', line))


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

        # file_contents[7]
        # file_contents[1]
        # file_contents[2]

        in_docstring = False
        in_function_or_class = False

        for line in file_contents:
            #line = file_contents[7]
            #print(line)

            if '"""' in line:
                # if we are ending an docstring then add newline
                # if in_docstring:
                #     output.append('\n')

                in_docstring = not in_docstring
                # if we have entered into a docstring, we are no longer in function/class
                if in_docstring and in_function_or_class:
                    in_function_or_class = False
                    output.append("```\n\n")  # end definition code-block

                docstring_leading_spaces = re.search(r'(^ *)"""', line).group(1)

            if in_docstring:
                if docstring_leading_spaces:
                    line = line.removeprefix(docstring_leading_spaces)

                if is_single_line_docstring(line):
                    line = line.removesuffix('"""\n')
                    in_docstring = False

                line = re.sub('^"""\n?', '', line)
                line = line.removesuffix('\n')

                if line:
                    output.append(line + "\n")
            elif in_function_or_class:
                # use the previous leading_spaces and remove accordingly, save the rest
                output.append(line.removeprefix(leading_spaces))
            else:
                function_or_class = re.search('^( *)(?:def|class)', line)
                if function_or_class:
                    in_function_or_class = True
                    # we are in the first line of the function/class (otherwise we'd be in the elif above)
                    # let's first add the header, then
                    # let's wrap the entire function/class def in a code-block
                    leading_spaces = function_or_class.group(1)
                    levels = calculate_levels(leading_spaces)

                    line = line.removeprefix(leading_spaces)

                    # header
                    friendly_name = re.sub(r'(\(|:).*', '', line).strip()
                    output.append(f'\n{top_level_header}{"#" * (levels + 1)} {friendly_name}\n')


                    output.append(f"\n```\n{line}")

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
