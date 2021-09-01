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

    def test_Redshift(self):
        redshift_config = RedshiftConfigFile(config_file=self.sample_redshift_file)
        database_objects = [Redshift(**redshift_config.get_dictionary()),
                            Redshift.from_config(redshift_config),
                            ]
        for index, db_object in enumerate(database_objects):
            with self.subTest(index=index, database=type(db_object)):
                self.assertFalse(db_object.is_open())
                self.assertIsNone(db_object.connection_object)

                # mock psycopg2.connect so that we can "open" the connection to the database
                with patch('psycopg2.connect'):
                    db_object.open()

                self.assertTrue(db_object.is_open())
                self.assertIsNotNone(db_object.connection_object)
                db_object.open()  # test that calling open again doesn't not fail

                results = db_object.query("SELECT * FROM doesnt_exist LIMIT 100")
                self.assertIsInstance(results, pd.DataFrame)
                results = db_object.query("SELECT * FROM doesnt_exist LIMIT 100")
                self.assertIsInstance(results, pd.DataFrame)

                db_object.close()
                self.assertFalse(db_object.is_open())
                self.assertIsNone(db_object.connection_object)

    def test_Snowflake(self):
        pass
