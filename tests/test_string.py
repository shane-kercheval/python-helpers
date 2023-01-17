import unittest
import pandas as pd
from itertools import product
import helpsk.string as hs
import helpsk.validation as hv
from tests.helpers import get_test_path


# noinspection PyMethodMayBeStatic
class TestStrings(unittest.TestCase):
    def test_collapse(self):
        # test *args as different parameters
        self.assertEqual(hs.collapse('a'), 'a')
        self.assertEqual(hs.collapse('a', 'b'), 'ab')
        self.assertEqual(hs.collapse('a', 'b', 'c'), 'abc')

        self.assertEqual(hs.collapse('abc', surround="'"), "'a''b''c'")
        self.assertEqual(hs.collapse('abc', surround="'", separate="-"), "'a'-'b'-'c'")

        self.assertEqual(hs.collapse('a', surround="'"), "'a'")
        self.assertEqual(hs.collapse('a', 'b', surround="'"), "'a''b'")
        self.assertEqual(hs.collapse('a', 'b', 'c', surround="'"), "'a''b''c'")

        self.assertEqual(hs.collapse('a', separate=', '), "a")
        self.assertEqual(hs.collapse('a', 'b', separate=', '), "a, b")
        self.assertEqual(hs.collapse('a', 'b', 'c', separate=', '), "a, b, c")

        self.assertEqual(hs.collapse('a', separate=', ', surround="'"), "'a'")
        self.assertEqual(hs.collapse('a', 'b', separate=', ', surround="'"), "'a', 'b'")
        self.assertEqual(hs.collapse('a', 'b', 'c', separate=', ', surround="'"), "'a', 'b', 'c'")

        # test *args as list of strings
        self.assertEqual(hs.collapse(['a']), 'a')
        self.assertEqual(hs.collapse(['a', 'b']), 'ab')
        self.assertEqual(hs.collapse(['a', 'b', 'c']), 'abc')

        self.assertEqual(hs.collapse(['a'], surround="'"), "'a'")
        self.assertEqual(hs.collapse(['a', 'b'], surround="'"), "'a''b'")
        self.assertEqual(hs.collapse(['a', 'b', 'c'], surround="'"), "'a''b''c'")

        self.assertEqual(hs.collapse(['a'], separate=', '), "a")
        self.assertEqual(hs.collapse(['a', 'b'], separate=', '), "a, b")
        self.assertEqual(hs.collapse(['a', 'b', 'c'], separate=', '), "a, b, c")

        self.assertEqual(hs.collapse(['a'], separate=', ', surround="'"), "'a'")
        self.assertEqual(hs.collapse(['a', 'b'], separate=', ', surround="'"), "'a', 'b'")
        self.assertEqual(
            hs.collapse(['a', 'b', 'c'], separate=', ', surround="'"),
            "'a', 'b', 'c'"
        )

    def test_format_number(self):
        granularity_options = [
            hs.RoundTo.AUTO, hs.RoundTo.NONE, hs.RoundTo.THOUSANDS, hs.RoundTo.MILLIONS,
            hs.RoundTo.BILLIONS, hs.RoundTo.TRILLIONS
        ]
        places = [0, 1, 2, 3, 4, 5]

        parameter_combos = list(product(granularity_options, places))
        test_values = [
            1234567890000000,  # 1,000 Trillion
            1000000000000000,  # 1000 Trillion
            100000000000000,  # 100 Trillion
            1000000000000,  # 1 Trillion
            1000000000000,  # 1 Trillion
            100000000000,  # 100 Billion
            10000000000,  # 10 Billion
            1000000000,  # 1 Billion
            100000000,  # 100 Million
            10000000,  # 10 Million
            1000000,  # 1 Million
            100000,
            10000,
            1000,
            100,
            10,
            1,
            0,
            0.01,
            0.01,
            0.001,
            0.0001,
            0.00001,
            0.000001,
            0.0000001,
            0.00000001,
            0.000000001,
            # 0.0000000001,
            # 0.00000000001,
            # 0.000000000001,
        ]

        def create_dataframe_combinations(value):
            return pd.DataFrame({'value': value,
                                 'granularity': [x[0] for x in parameter_combos],
                                 'places': [x[1] for x in parameter_combos]})

        def apply_format(row):
            return hs.format_number(value=row['value'],
                                    granularity=row['granularity'],
                                    places=row['places'])

        def run_combinations_for_value(value, positive=True):
            result = create_dataframe_combinations(value if positive else value * -1)
            result['result'] = result.apply(apply_format, axis=1)
            return result

        # Positive Values
        results = [run_combinations_for_value(value) for value in test_values]
        results = pd.concat(results)
        results['granularity'] = results['granularity'].transform(lambda x: str(x))
        expected_results = pd.read_csv(get_test_path('string/string__format_number__expected_values_positive.csv'))  # noqa
        self.assertTrue(hv.dataframes_match([results, expected_results]))

        # Negative Values
        results = [run_combinations_for_value(value, positive=False) for value in test_values]
        results = pd.concat(results)
        results['granularity'] = results['granularity'].transform(lambda x: str(x))
        expected_results = pd.read_csv(get_test_path('string/string__format_number__expected_values_negative.csv'))  # noqa
        self.assertTrue(hv.dataframes_match([results, expected_results]))

    def test_format_number__0(self):
        assert hs.format_number(value=0, granularity=hs.RoundTo.AUTO, places=2) == '0'
        assert hs.format_number(value=0, granularity=hs.RoundTo.NONE, places=0) == '0'
        assert hs.format_number(value=0, granularity=hs.RoundTo.THOUSANDS, places=1) == '0'
        assert hs.format_number(value=0, granularity=hs.RoundTo.MILLIONS, places=2) == '0'
        assert hs.format_number(value=0, granularity=hs.RoundTo.BILLIONS, places=3) == '0'
        assert hs.format_number(value=0, granularity=hs.RoundTo.TRILLIONS, places=4) == '0'
