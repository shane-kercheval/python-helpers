import unittest
from os import remove
import os.path
import time
import pandas as pd
from dataclasses import dataclass

from helpsk.utility import to_pickle, read_pickle, open_yaml, Timer, is_debugging, dataframe_to_pickle, \
    dataframe_to_csv, object_to_pickle, repr
from tests.helpers import get_test_path


# noinspection PyMethodMayBeStatic
class TestValidation(unittest.TestCase):

    def test_pickle(self):
        obj = "this is an object"
        path = get_test_path('pickled_object.pkl')
        self.assertFalse(os.path.isfile(path))
        to_pickle(obj=obj, path=path)
        self.assertTrue(os.path.isfile(path))
        unpickled_object = read_pickle(path=path)
        self.assertEqual(obj, unpickled_object)
        remove(path)
        self.assertFalse(os.path.isfile(path))

    def test__open_yaml(self):
        config_yaml = open_yaml(get_test_path('utility/config.yaml'))
        self.assertIsInstance(config_yaml, dict)
        self.assertIsNotNone(config_yaml['CONFIG']['DIRECTORY'])

    def test_is_debugging(self):
        # not sure how to test is_debugging is true, except manually
        self.assertFalse(is_debugging())

    def test__timer(self):
        print('\n')

        with Timer("testing timer"):
            time.sleep(0.05)

        with Timer("testing another timer") as timer:
            time.sleep(0.1)

        self.assertIsNotNone(timer._interval)

    def test__dataframe_to_pickle(self):
        df = pd.DataFrame(['test'])
        output_directory = 'temp_test'
        file_name = 'test.pkl'
        self.assertFalse(os.path.isdir(output_directory))
        file_path = dataframe_to_pickle(df=df, output_directory=output_directory, file_name=file_name)
        self.assertTrue(os.path.isfile(file_path))
        self.assertEqual(file_path, 'temp_test/test.pkl')
        df_unpickled = pd.read_pickle(file_path)
        self.assertTrue((df_unpickled.iloc[0] == df.iloc[0]).all())  # noqa
        os.remove(file_path)
        os.removedirs(output_directory)
        self.assertFalse(os.path.isdir(output_directory))

    def test__dataframe_to_csv(self):
        df = pd.DataFrame(['test'])
        output_directory = 'temp_test'
        file_name = 'test.csv'
        self.assertFalse(os.path.isdir(output_directory))
        file_path = dataframe_to_csv(df=df, output_directory=output_directory, file_name=file_name)
        self.assertTrue(os.path.isfile(file_path))
        self.assertEqual(file_path, 'temp_test/test.csv')
        df_unpickled = pd.read_csv(file_path)
        self.assertEqual(df_unpickled.loc[0, '0'], df.iloc[0, 0])
        os.remove(file_path)
        os.removedirs(output_directory)
        self.assertFalse(os.path.isdir(output_directory))

    def test__object_to_pickle(self):
        df = pd.DataFrame(['test'])
        output_directory = 'temp_test'
        file_name = 'test.pkl'
        self.assertFalse(os.path.isdir(output_directory))
        file_path = object_to_pickle(obj=df, output_directory=output_directory, file_name=file_name)
        self.assertTrue(os.path.isfile(file_path))
        self.assertEqual(file_path, 'temp_test/test.pkl')
        df_unpickled = pd.read_pickle(file_path)
        self.assertTrue((df_unpickled.iloc[0] == df.iloc[0]).all())  # noqa
        os.remove(file_path)
        os.removedirs(output_directory)
        self.assertFalse(os.path.isdir(output_directory))

    def test_repr(self):
        class Example:
            def __init__(self, x: int, y: int):
                self.x = x
                self.y = y

            def __repr__(self) -> str:
                return repr(self)

        self.assertEqual(
            f"{Example(1, 2)!r}",
            'Example(\n    x = 1,\n    y = 2,\n)'
        )

        @dataclass
        class Example:
            x: int
            y: int

            def __repr__(self) -> str:
                return repr(self)

        self.assertEqual(
            f"{Example(1, 2)!r}",
            'Example(\n    x = 1,\n    y = 2,\n)'
        )


if __name__ == '__main__':
    unittest.main()
