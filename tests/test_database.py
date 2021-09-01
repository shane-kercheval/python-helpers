import unittest
from os import path
from unittest.mock import patch

import pandas as pd

from tests.helpers import get_test_path
from helpsk.database import GenericConfigFile, RedshiftConfigFile, SnowflakeConfigFile, Redshift, Snowflake


# noinspection PyMethodMayBeStatic
class TestDatabase(unittest.TestCase):

    def setUp(self) -> None:
        self.sample_redshift_file = get_test_path() + '/test_files/sample_redshift.config'
        self.assertTrue(path.isfile(self.sample_redshift_file))

        self.sample_snowflake_file = get_test_path() + '/test_files/sample_snowflake.config'
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
                          'database': 'DATEBASE_NAME'}
        self.assertEqual(expected_value, config_dict)

        # test that this can be passed into the snowflake database object, which implies that config_dict
        # contains the proper keyboard arguments
        Snowflake(**config_dict)

    def test_Database(self):
        redshift_config = RedshiftConfigFile(config_file=self.sample_redshift_file)
        snowflake_config = SnowflakeConfigFile(config_file=self.sample_snowflake_file)

        database_objects = [Redshift(**redshift_config.get_dictionary()),
                            Redshift.from_config(redshift_config),
                            Snowflake(**snowflake_config.get_dictionary()),
                            Snowflake.from_config(snowflake_config),
                            ]

        def sub_test(db_obj):
            """db_obj is a Database object that is in a closed state and has the mock objects set up"""
            # connection should be closed
            self.assertFalse(db_object.is_open())
            self.assertIsNone(db_object.connection_object)

            # test connecting
            db_obj.open()
            self.assertTrue(db_obj.is_open())
            self.assertIsNotNone(db_obj.connection_object)
            self.assertIsInstance(db_obj.connection_object, unittest.mock.MagicMock)
            db_obj.open()  # test that calling open again doesn't not fail

            # test querying
            results = db_obj.query("SELECT * FROM doesnt_exist LIMIT 100")
            self.assertIsInstance(results, pd.DataFrame)
            self.assertEqual(results.iloc[0, 0], 'test')
            # test connection is still open after querying
            self.assertTrue(db_obj.is_open())
            self.assertIsNotNone(db_obj.connection_object)
            # test subsequent query
            results = db_obj.query("SELECT * FROM doesnt_exist LIMIT 100")
            self.assertIsInstance(results, pd.DataFrame)
            self.assertEqual(results.iloc[0, 0], 'test')

            # test closing
            db_obj.close()
            self.assertFalse(db_obj.is_open())
            self.assertIsNone(db_obj.connection_object)

            # text context manager
            # also tests that the same object can be open/closed multiple times 

        for index, db_object in enumerate(database_objects):
            with self.subTest(index=index, database=type(db_object)):
                self.assertFalse(db_object.is_open())
                self.assertIsNone(db_object.connection_object)

                # mock connection method so that we can "open" the connection to the database
                if isinstance(db_object, Redshift):
                    with patch('psycopg2.connect'), patch('pandas.read_sql_query') as pandas_read_mock:
                        pandas_read_mock.return_value = pd.DataFrame({'test': ['test']})
                        sub_test(db_obj=db_object)
                else:
                    with patch('snowflake.connector.connect') as mock_snowflake_connector:
                        # mock this logic out:
                        # connect() -> cursor() -> ... -> fetch_pandas_all()
                        # actual code
                        # cursor = self.connection_object.cursor()
                        # cursor.execute(sql)
                        # dataframe = cursor.fetch_pandas_all()
                        mock_con = mock_snowflake_connector.return_value
                        mock_cur = mock_con.cursor.return_value
                        mock_cur.fetch_pandas_all.return_value = pd.DataFrame({'test': ['test']})
                        sub_test(db_obj=db_object)

# test open connections after closing connection
# test context manager "with asdf"
# tests that if failure to connect (i.e. no mock and connection failure) that the db_boject is not in an open state open
