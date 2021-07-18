import unittest
from helpsk import string


# noinspection PyMethodMayBeStatic
class TestStrings(unittest.TestCase):
    def test_collapse(self):

        # test *args as different parameters
        assert string.collapse('a') == 'a'
        assert string.collapse('a', 'b') == 'ab'
        assert string.collapse('a', 'b', 'c') == 'abc'

        assert string.collapse('a', surround="'") == "'a'"
        assert string.collapse('a', 'b', surround="'") == "'a''b'"
        assert string.collapse('a', 'b', 'c', surround="'") == "'a''b''c'"

        assert string.collapse('a', separate=', ') == "a"
        assert string.collapse('a', 'b', separate=', ') == "a, b"
        assert string.collapse('a', 'b', 'c', separate=', ') == "a, b, c"

        assert string.collapse('a', separate=', ', surround="'") == "'a'"
        assert string.collapse('a', 'b', separate=', ', surround="'") == "'a', 'b'"
        assert string.collapse('a', 'b', 'c', separate=', ', surround="'") == "'a', 'b', 'c'"

        # test *args as list of strings
        assert string.collapse(['a']) == 'a'
        assert string.collapse(['a', 'b']) == 'ab'
        assert string.collapse(['a', 'b', 'c']) == 'abc'

        assert string.collapse(['a'], surround="'") == "'a'"
        assert string.collapse(['a', 'b'], surround="'") == "'a''b'"
        assert string.collapse(['a', 'b', 'c'], surround="'") == "'a''b''c'"

        assert string.collapse(['a'], separate=', ') == "a"
        assert string.collapse(['a', 'b'], separate=', ') == "a, b"
        assert string.collapse(['a', 'b', 'c'], separate=', ') == "a, b, c"

        assert string.collapse(['a'], separate=', ', surround="'") == "'a'"
        assert string.collapse(['a', 'b'], separate=', ', surround="'") == "'a', 'b'"
        assert string.collapse(['a', 'b', 'c'], separate=', ', surround="'") == "'a', 'b', 'c'"
