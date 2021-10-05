import unittest
from contextlib import contextmanager
from os import path
from unittest.mock import patch

import pandas as pd

from tests.helpers import get_test_path
from helpsk.database import GenericConfigFile, RedshiftConfigFile, SnowflakeConfigFile, Redshift, Snowflake


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
        self.sample_redshift_file = get_test_path() + '/test_files/database/sample_redshift.config'
        self.assertTrue(path.isfile(self.sample_redshift_file))

        self.sample_snowflake_file = get_test_path() + '/test_files/database/sample_snowflake.config'
        self.assertTrue(path.isfile(self.sample_snowflake_file))

    def test_GenericConfigFile(self):
        config_mapping = {'user_param': 'username',
                          'password_param': 'password',
                          'database_param': 'database',
                          'port_param': 'port',
                          'host_param': 'host'}
        config = GenericConfigFile(config_file=self.sample_redshift_file,
                                   config_key='redshift',
                                   config_mapping=config_mapping)
        expected_value = {'user_param': 'my_username',
                          'password_param': 'my-password-123',
                          'database_param': 'the_database',
                          'port_param': '1234',
                          'host_param': 'host.address.redshift.amazonaws.com'}
        self.assertEqual(config.get_dictionary(), expected_value)

    def test_RedshiftConfigFile(self):
        config = RedshiftConfigFile(config_file=self.sample_redshift_file)
        config_dict = config.get_dictionary()
        expected_value = {'user': 'my_username',
                          'password': 'my-password-123',
                          'database': 'the_database',
                          'port': '1234',
                          'host': 'host.address.redshift.amazonaws.com'}
        self.assertEqual(expected_value, config_dict)

        # test that this can be passed into the redshift database object, which implies that config_dict
        # contains the proper keyboard arguments
        Redshift(**config_dict)

    def test_SnowflakeConfigFile(self):
        config = SnowflakeConfigFile(config_file=self.sample_snowflake_file)
        config_dict = config.get_dictionary()
        expected_value = {'user': 'my.email@address.com',
                          'account': 'account.id',
                          'authenticator': 'externalbrowser',
                          'warehouse': 'WAREHOUSE_NAME',
                          'database': 'DATABASE_NAME'}
        self.assertEqual(expected_value, config_dict)

        # test that this can be passed into the snowflake database object, which implies that config_dict
        # contains the proper keyboard arguments
        Snowflake(**config_dict)

    def test_Database(self):
        redshift_config = RedshiftConfigFile(config_file=self.sample_redshift_file)
        snowflake_config = SnowflakeConfigFile(config_file=self.sample_snowflake_file)

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

        test_via_constructor(db_obj=Redshift(**redshift_config.get_dictionary()), db_mock=mock_redshift)
        test_via_constructor(db_obj=Redshift.from_config(redshift_config), db_mock=mock_redshift)
        test_via_constructor(db_obj=Snowflake(**snowflake_config.get_dictionary()), db_mock=mock_snowflake)
        test_via_constructor(db_obj=Snowflake.from_config(snowflake_config), db_mock=mock_snowflake)

        def test_context_manager(db_class, db_config, db_mock):
            # test context manager
            with db_mock():
                with db_class.from_config(db_config) as db_object:
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

        test_context_manager(db_class=Redshift, db_config=redshift_config, db_mock=mock_redshift)
        test_context_manager(db_class=Snowflake, db_config=snowflake_config, db_mock=mock_snowflake)

        # tests that if failure to connect (i.e. no mock and connection failure) that the object is not in an
        # open state
        database = Redshift.from_config(redshift_config)
        with self.assertRaises(Exception):
            database.connect()
        self.assertFalse(database.is_connected())
        self.assertIsNone(database.connection_object)
