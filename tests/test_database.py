import datetime
import unittest
from contextlib import contextmanager
import configparser
from enum import unique, Enum, auto
from os import path
from unittest.mock import patch

import numpy as np
import pandas as pd

from helpsk.database import Redshift, Snowflake
from tests.helpers import get_test_path


@unique
class TestEnum(Enum):
    __test__ = False
    VALUE_A = auto()
    VALUE_B = auto()


@contextmanager
def mock_redshift():
    """This is basically a wrapper around mock.patch to hide the mock logic. i.e. This is a context manager
    that wraps the mock context wrapper
    """
    with patch('psycopg2.connect'), patch('pandas.read_sql_query') as pandas_read_mock:
        pandas_read_mock.return_value = pd.DataFrame({'test': ['test']})
        yield


@contextmanager
def mock_snowflake():
    """This is basically a wrapper around mock.patch to hide the mock logic. i.e. This is a context manager
    that wraps the mock context wrapper
    """
    with patch('snowflake.connector.connect') as mock_snowflake_connector:
        # mock snowflake logic:
        # connect() -> cursor() -> ... -> fetch_pandas_all()
        # actual code
        # self.connection_object = snowflake.connector.connect()
        # cursor = self.connection_object.cursor()
        # cursor.execute(sql)
        # dataframe = cursor.fetch_pandas_all()
        mock_con = mock_snowflake_connector.return_value
        mock_cur = mock_con.cursor.return_value
        mock_cur.fetch_pandas_all.return_value = pd.DataFrame({'test': ['test']})
        yield


# noinspection PyMethodMayBeStatic
class TestDatabase(unittest.TestCase):

    def setUp(self) -> None:
        self.sample_redshift_file = get_test_path('database/sample_redshift.config')
        self.assertTrue(path.isfile(self.sample_redshift_file))

        self.sample_snowflake_file = get_test_path('database/sample_snowflake.config')
        self.assertTrue(path.isfile(self.sample_snowflake_file))

        self.sample_snowflake_autocommit_true_file = get_test_path('database/sample_snowflake_autocommit_true.config')  # noqa
        self.assertTrue(path.isfile(self.sample_snowflake_autocommit_true_file))

        self.sample_snowflake_autocommit_false_file = get_test_path('database/sample_snowflake_autocommit_false.config')  # noqa
        self.assertTrue(path.isfile(self.sample_snowflake_autocommit_false_file))

    def test_redshift_settings(self):
        redshift = Redshift(test='value', test2='value2')
        self.assertEqual({'test': 'value', 'test2': 'value2'}, redshift._kwargs)
        self.assertEqual('test=value test2=value2', redshift._connection_string)

        redshift = Redshift.from_config(config_path=self.sample_redshift_file, config_key='redshift')
        expected_kwargs = {
            'user': 'my_username',
            'password': 'my-password-123',
            'dbname': 'the_database',
            'host': 'host.address.redshift.amazonaws.com',
            'port': '1234'
        }
        self.assertEqual(expected_kwargs, redshift._kwargs)

        expected_connection_string = 'user=my_username password=my-password-123 dbname=the_database ' \
            'host=host.address.redshift.amazonaws.com port=1234'
        self.assertEqual(expected_connection_string, redshift._connection_string)

    def test_snowflake_settings(self):
        snowflake = Snowflake(test='value', test2='value2')
        self.assertEqual({'test': 'value', 'test2': 'value2'}, snowflake._kwargs)
        snowflake = Snowflake(test='value', test2='value2', autocommit='FALSE')
        self.assertEqual({'test': 'value', 'test2': 'value2', 'autocommit': False}, snowflake._kwargs)
        snowflake = Snowflake(test='value', test2='value2', autocommit='FALSE')
        self.assertEqual({'test': 'value', 'test2': 'value2', 'autocommit': False}, snowflake._kwargs)

        snowflake = Snowflake.from_config(config_path=self.sample_snowflake_file, config_key='snowflake')
        expected_kwargs = {
           'user': 'my.email@address.com',
           'account': 'account.id',
           'authenticator': 'externalbrowser',
           'warehouse': 'WAREHOUSE_NAME',
           'database': 'DATABASE_NAME',
        }
        self.assertEqual(expected_kwargs, snowflake._kwargs)

        snowflake = Snowflake.from_config(
            config_path=self.sample_snowflake_autocommit_false_file,
            config_key='snowflake'
        )
        expected_kwargs = {
           'user': 'my.email@address.com',
           'account': 'account.id',
           'authenticator': 'externalbrowser',
           'warehouse': 'WAREHOUSE_NAME',
           'database': 'DATABASE_NAME',
           'autocommit': False,
        }
        self.assertEqual(expected_kwargs, snowflake._kwargs)

        snowflake = Snowflake.from_config(
            config_path=self.sample_snowflake_autocommit_true_file,
            config_key='snowflake'
        )
        expected_kwargs = {
           'user': 'my.email@address.com',
           'account': 'account.id',
           'authenticator': 'externalbrowser',
           'warehouse': 'WAREHOUSE_NAME',
           'database': 'DATABASE_NAME',
           'autocommit': True,
        }
        self.assertEqual(expected_kwargs, snowflake._kwargs)

    def test_Snowflake_generate_sql_create_table(self):
        sample_data = pd.DataFrame({'col_a': [np.nan, 2, 3, 4],
                                    'col_b': [np.nan, 'b', 'd', 'd'],
                                    'col_bb': ['a', 'b', 'd', 'd'],
                                    'col_c': pd.date_range('2021-01-01', '2021-01-04'),
                                    'col_d': [datetime.date(2021, 4, 2), datetime.date(2021, 4, 2),
                                              datetime.date(2021, 4, 2), datetime.date(2021, 4, 2)],
                                    'col_e': np.nan,
                                    'col_f': [1.0, 2.0, 3.0, 4.0],
                                    'col_h': [TestEnum.VALUE_A, TestEnum.VALUE_A,
                                              TestEnum.VALUE_B, TestEnum.VALUE_B],
                                    'col_j': [False, False, True, False],
                                    'col_k': [None, None, None, None],
                                    'col_l': [np.nan, np.nan, np.nan, np.nan]
                                    })
        sample_data['col_g'] = sample_data['col_b'].astype('category')

        sql = Snowflake._generate_sql_create_table(
            table='table_name',
            dataframe=sample_data,
            database='database_name',
            schema='schema_name',
        )
        expected_value = """CREATE OR REPLACE TABLE DATABASE_NAME.SCHEMA_NAME.TABLE_NAME (
            col_a float8,
            col_b varchar,
            col_bb varchar,
            col_c datetime,
            col_d varchar,
            col_e float8,
            col_f float8,
            col_h varchar,
            col_j boolean,
            col_k varchar,
            col_l float8,
            col_g varchar
        )""".replace("        ", "")
        self.assertEqual(sql, expected_value)

        sql = Snowflake._generate_sql_create_table(
            table='table_name',
            dataframe=sample_data,
            database='database_name',
            # schema='schema_name',
        )
        expected_value = """CREATE OR REPLACE TABLE DATABASE_NAME.TABLE_NAME (
            col_a float8,
            col_b varchar,
            col_bb varchar,
            col_c datetime,
            col_d varchar,
            col_e float8,
            col_f float8,
            col_h varchar,
            col_j boolean,
            col_k varchar,
            col_l float8,
            col_g varchar
        )""".replace("        ", "")
        self.assertEqual(sql, expected_value)

        sql = Snowflake._generate_sql_create_table(
            table='table_name',
            dataframe=sample_data,
            # database='database_name',
            schema='schema_name',
        )
        expected_value = """CREATE OR REPLACE TABLE SCHEMA_NAME.TABLE_NAME (
                    col_a float8,
                    col_b varchar,
                    col_bb varchar,
                    col_c datetime,
                    col_d varchar,
                    col_e float8,
                    col_f float8,
                    col_h varchar,
                    col_j boolean,
                    col_k varchar,
                    col_l float8,
                    col_g varchar
                )""".replace("        ", "")
        self.assertEqual(sql, expected_value)

        sql = Snowflake._generate_sql_create_table(
            table='table_name',
            dataframe=sample_data,
            # database='database_name',
            # schema='schema_name',
        )
        expected_value = """CREATE OR REPLACE TABLE TABLE_NAME (
                            col_a float8,
                            col_b varchar,
                            col_bb varchar,
                            col_c datetime,
                            col_d varchar,
                            col_e float8,
                            col_f float8,
                            col_h varchar,
                            col_j boolean,
                            col_k varchar,
                            col_l float8,
                            col_g varchar
                        )""".replace("        ", "")
        self.assertEqual(sql, expected_value)

    def test_Database(self):
        redshift_config = configparser.ConfigParser()
        redshift_config.read(self.sample_redshift_file)
        redshift_config = dict(redshift_config['redshift'].items())

        snowflake_config = configparser.ConfigParser()
        snowflake_config.read(self.sample_snowflake_file)
        snowflake_config = dict(snowflake_config['snowflake'].items())

        def test_via_constructor(db_obj, db_mock):
            self.assertFalse(db_obj.is_connected())
            self.assertIsNone(db_obj.connection_object)

            # mock connection method so that we can "open" the connection to the database
            with db_mock():
                """db_obj is a Database object that is in a closed state and has the mock objects set up"""
                for _ in range(2):  # test that the same object can be opened/closed multiple times
                    # connection should be closed
                    self.assertFalse(db_obj.is_connected())
                    self.assertIsNone(db_obj.connection_object)

                    # test connecting
                    db_obj.connect()
                    self.assertTrue(db_obj.is_connected())
                    self.assertIsNotNone(db_obj.connection_object)
                    self.assertIsInstance(db_obj.connection_object, unittest.mock.MagicMock)
                    db_obj.connect()  # test that calling open again doesn't not fail

                    # test querying
                    results = db_obj.query("SELECT * FROM doesnt_exist LIMIT 100", show_elapsed_time=False)
                    self.assertIsInstance(results, pd.DataFrame)
                    self.assertEqual(results.iloc[0, 0], 'test')
                    # test connection is still open after querying
                    self.assertTrue(db_obj.is_connected())
                    self.assertIsNotNone(db_obj.connection_object)
                    # test subsequent query
                    results = db_obj.query("SELECT * FROM doesnt_exist LIMIT 100", show_elapsed_time=False)
                    self.assertIsInstance(results, pd.DataFrame)
                    self.assertEqual(results.iloc[0, 0], 'test')

                    # test closing
                    db_obj.close()
                    self.assertFalse(db_obj.is_connected())
                    self.assertIsNone(db_obj.connection_object)

        test_via_constructor(db_obj=Redshift(**redshift_config), db_mock=mock_redshift)
        test_via_constructor(
            db_obj=Redshift.from_config(self.sample_redshift_file, 'redshift'),
            db_mock=mock_redshift
        )
        test_via_constructor(db_obj=Snowflake(**snowflake_config), db_mock=mock_snowflake)
        test_via_constructor(
            db_obj=Snowflake.from_config(self.sample_snowflake_file, 'snowflake'),
            db_mock=mock_snowflake
        )

        def test_context_manager(db_class, db_config_path, db_config_key, db_mock):
            # test context manager
            with db_mock():
                with db_class.from_config(config_path=db_config_path, config_key=db_config_key) as db_object:
                    self.assertTrue(db_object.is_connected())
                    self.assertIsNotNone(db_object.connection_object)
                    self.assertIsInstance(db_object.connection_object, unittest.mock.MagicMock)

                    # test querying
                    results = db_object.query("SELECT * FROM doesnt_exist LIMIT 100", show_elapsed_time=False)
                    self.assertIsInstance(results, pd.DataFrame)
                    self.assertEqual(results.iloc[0, 0], 'test')
                    # test connection is still open after querying
                    self.assertTrue(db_object.is_connected())
                    self.assertIsNotNone(db_object.connection_object)

                self.assertFalse(db_object.is_connected())
                self.assertIsNone(db_object.connection_object)

        test_context_manager(
            db_class=Redshift,
            db_config_path=self.sample_redshift_file,
            db_config_key='redshift',
            db_mock=mock_redshift
        )
        test_context_manager(
            db_class=Snowflake,
            db_config_path=self.sample_snowflake_file,
            db_config_key='snowflake',
            db_mock=mock_snowflake
        )

        # tests that if failure to connect (i.e. no mock and connection failure) that the object is not in an
        # open state
        database = Redshift.from_config(config_path=self.sample_redshift_file, config_key='redshift')
        with self.assertRaises(Exception):
            database.connect()
        self.assertFalse(database.is_connected())
        self.assertIsNone(database.connection_object)


if __name__ == '__main__':
    unittest.main()
