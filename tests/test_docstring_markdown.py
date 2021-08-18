import os
import unittest
from docstring_markdown import build_markdown

os.getcwd()
# noinspection PyMethodMayBeStatic
class TestExampleClass(unittest.TestCase):
    def test_something(self):
        build_markdown.execute_build(path='../helpsk')
