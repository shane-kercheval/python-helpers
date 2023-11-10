"""Contains functions to help with comparing two strings or two dataframes."""

import difflib
from typing import Union
import pandas as pd


def _create_html_difference_list(value_a: str, value_b: str) -> str:
    """
    Given value_a and value_b return unified_diff, after cleaning up what unified_diff returns.

    Args:
        value_a: string to compare
        value_b: another string to compare
    """
    diff = list(difflib.unified_diff(a=value_a, b=value_b, n=1000))
    return [" " + char for char in value_a] if len(diff) == 0 else diff[3:]


def _create_html_change_span(value: str, is_change: bool, change_color: str = '#F1948A') -> str:
    """
    Creates a single character from one company's differences.

    Args:
        value: a single character
        is_change: if True, highlight the character in red
        change_color: color of background to highlight differences.

    Returns:
        e.g. "<span style="background:#ffe6e6";>value</span>"
    """
    background_color = ''
    if is_change:
        background_color = f' style="background:{change_color}";'
    return f'<span{background_color}>{value}</span>'


def _create_html_cell(
        difference_list: list,
        is_first_value: bool,
        change_color: str = '#F1948A') -> str:
    """
    Creates a single cell (e.g. name, domain, etc.) from one company's differences.

    Args:
        difference_list: list returned from create_difference_list
        is_first_value: if True, treats difference_list according to first value.
        change_color: color of background to highlight differences.
    """
    if is_first_value:
        diff = [(x[1:], x[0] != ' ') for x in difference_list if x[0] in [' ', '-']]
    else:
        diff = [(x[1:], x[0] != ' ') for x in difference_list if x[0] in [' ', '+']]

    html = [
        _create_html_change_span(value=x[0], is_change=x[1], change_color=change_color)
        for x in diff
    ]
    return ''.join(html)


def diff_text(
        text_a: Union[str, list[str]],
        text_b: Union[str, list[str]],
        change_color: str = '#F1948A') -> str:
    """
    Returns string as HTML containing highlighted differences between `text_a` and
    `text_b`.

    The HTML will contain a table with a single column that that contains `text_a` on top and
    `text_b` on the bottom.

    All `new line` characters are removed and replaced with a space.

    Args:
        text_a: this text will be represented on the top of each html cell.
        text_b: this text will be represented on the bottom of each html cell.
        change_color: color of background to highlight differences.
    """
    html = '<html><head><style> table, th, td { border: 1px solid black; border-collapse: ' \
        'collapse; white-space: normal;} </style></head><body style="font-family: monospace">'
    html += '<table><tr>'
    html += '<th>index</th>'
    html += '<th>diff</th>'
    html += '</tr>'

    line_break = '<hr style="width:25%; text-align:left; margin-left:10px; height:1px; ' \
        'border-width:0;' \
        'color:blue; background-color:blue">'

    def create_inline_change(diff_list: list[str]) -> str:
        diff_a = _create_html_cell(
            difference_list=diff_list, is_first_value=True, change_color=change_color,
        )
        diff_b = _create_html_cell(
            difference_list=diff_list, is_first_value=False, change_color=change_color,
        )
        return f"<td>{diff_a}<br>{line_break}{diff_b}</td>"

    if isinstance(text_a, str):
        assert isinstance(text_b, str)
        text_a = [text_a]
        text_b = [text_b]
    else:
        assert len(text_a) == len(text_b)

    for index in range(len(text_a)):
        html += '<tr>'
        html += f"<td>{index}</td>"
        difference_list = _create_html_difference_list(
            value_a=text_a[index],
            value_b=text_b[index],
        )
        html += create_inline_change(diff_list=difference_list)
        html += '</tr>'
    html += "</table></body></html>"
    return html


def diff_dataframes(
        dataframe_a: pd.DataFrame,
        dataframe_b: pd.DataFrame,
        change_color: str = '#F1948A') -> str:
    """
    Returns string as HTML containing highlighted differences between `dataframe_a` and
    `dataframe_b`.

    The HTML will contain a table showing columns that are found in both `dataframe_a` and
    `dataframe_b`.

    The DataFrames should be equal in length.

    Args:
        dataframe_a: this dataframe will be represented with values on the top of each html cell.
        dataframe_b: this dataframe will be represented with values on the bottom of each html
            cell.
        change_color: color of background to highlight differences.
    """
    assert len(dataframe_a) == len(dataframe_b)
    joint_columns = [x for x in dataframe_a.columns if x in dataframe_b.columns]
    assert len(joint_columns) > 0

    html = '<html><head><style> table, th, td { border: 1px solid black; border-collapse: ' \
        'collapse; white-space: normal;} </style></head><body style="font-family: monospace">'
    html += '<table><tr>'
    html += '<th>index</th>'
    for column in joint_columns:
        html += f'<th>{column}</th>'
    html += '</tr>'

    line_break = '<hr style="width:25%; text-align:left; margin-left:10px; height:1px; ' \
        'border-width:0;' \
        'color:blue; background-color:blue">'

    def create_inline_change(diff_list: list[str]) -> str:
        diff_a = _create_html_cell(
            difference_list=diff_list, is_first_value=True, change_color=change_color,
        )
        diff_b = _create_html_cell(
            difference_list=diff_list, is_first_value=False, change_color=change_color,
        )
        return f"<td>{diff_a}<br>{line_break}{diff_b}</td>"

    for index in range(len(dataframe_a)):
        html += '<tr>'
        html += f"<td>{index}</td>"
        for column in joint_columns:
            # print(f"{index}:{column}")
            value_a = str(dataframe_a[column].iloc[index])
            value_b = str(dataframe_b[column].iloc[index])
            difference_list = _create_html_difference_list(value_a=value_a, value_b=value_b)
            html += create_inline_change(diff_list=difference_list)

        html += '</tr>'
    html += "</table></body></html>"
    return html
