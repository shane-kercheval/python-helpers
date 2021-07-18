import unittest
from helpsk import Strings


# noinspection PyMethodMayBeStatic
class TestStrings(unittest.TestCase):
    def test_collapse(self):

        # test *args as different parameters
        assert Strings.collapse('a') == 'a'
        assert Strings.collapse('a', 'b') == 'ab'
        assert Strings.collapse('a', 'b', 'c') == 'abc'

        assert Strings.collapse('a', surround="'") == "'a'"
        assert Strings.collapse('a', 'b', surround="'") == "'a''b'"
        assert Strings.collapse('a', 'b', 'c', surround="'") == "'a''b''c'"

        assert Strings.collapse('a', separate=', ') == "a"
        assert Strings.collapse('a', 'b', separate=', ') == "a, b"
        assert Strings.collapse('a', 'b', 'c', separate=', ') == "a, b, c"

        assert Strings.collapse('a', separate=', ', surround="'") == "'a'"
        assert Strings.collapse('a', 'b', separate=', ', surround="'") == "'a', 'b'"
        assert Strings.collapse('a', 'b', 'c', separate=', ', surround="'") == "'a', 'b', 'c'"

        # test *args as list of strings
        assert Strings.collapse(['a']) == 'a'
        assert Strings.collapse(['a', 'b']) == 'ab'
        assert Strings.collapse(['a', 'b', 'c']) == 'abc'

        assert Strings.collapse(['a'], surround="'") == "'a'"
        assert Strings.collapse(['a', 'b'], surround="'") == "'a''b'"
        assert Strings.collapse(['a', 'b', 'c'], surround="'") == "'a''b''c'"

        assert Strings.collapse(['a'], separate=', ') == "a"
        assert Strings.collapse(['a', 'b'], separate=', ') == "a, b"
        assert Strings.collapse(['a', 'b', 'c'], separate=', ') == "a, b, c"

        assert Strings.collapse(['a'], separate=', ', surround="'") == "'a'"
        assert Strings.collapse(['a', 'b'], separate=', ', surround="'") == "'a', 'b'"
        assert Strings.collapse(['a', 'b', 'c'], separate=', ', surround="'") == "'a', 'b', 'c'"
