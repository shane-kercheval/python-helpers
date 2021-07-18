import unittest
from helpsk import String


# noinspection PyMethodMayBeStatic
class TestStrings(unittest.TestCase):
    def test_collapse(self):

        # test *args as different parameters
        assert String.collapse('a') == 'a'
        assert String.collapse('a', 'b') == 'ab'
        assert String.collapse('a', 'b', 'c') == 'abc'

        assert String.collapse('a', surround="'") == "'a'"
        assert String.collapse('a', 'b', surround="'") == "'a''b'"
        assert String.collapse('a', 'b', 'c', surround="'") == "'a''b''c'"

        assert String.collapse('a', separate=', ') == "a"
        assert String.collapse('a', 'b', separate=', ') == "a, b"
        assert String.collapse('a', 'b', 'c', separate=', ') == "a, b, c"

        assert String.collapse('a', separate=', ', surround="'") == "'a'"
        assert String.collapse('a', 'b', separate=', ', surround="'") == "'a', 'b'"
        assert String.collapse('a', 'b', 'c', separate=', ', surround="'") == "'a', 'b', 'c'"

        # test *args as list of strings
        assert String.collapse(['a']) == 'a'
        assert String.collapse(['a', 'b']) == 'ab'
        assert String.collapse(['a', 'b', 'c']) == 'abc'

        assert String.collapse(['a'], surround="'") == "'a'"
        assert String.collapse(['a', 'b'], surround="'") == "'a''b'"
        assert String.collapse(['a', 'b', 'c'], surround="'") == "'a''b''c'"

        assert String.collapse(['a'], separate=', ') == "a"
        assert String.collapse(['a', 'b'], separate=', ') == "a, b"
        assert String.collapse(['a', 'b', 'c'], separate=', ') == "a, b, c"

        assert String.collapse(['a'], separate=', ', surround="'") == "'a'"
        assert String.collapse(['a', 'b'], separate=', ', surround="'") == "'a', 'b'"
        assert String.collapse(['a', 'b', 'c'], separate=', ', surround="'") == "'a', 'b', 'c'"
