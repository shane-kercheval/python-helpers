from typing import Union, Optional

import pandas as pd
import configparser

from helpsk.database_base import Configuration, Database


class GenericConfigFile(Configuration):

    def __init__(self, config_file: str, config_key: str, config_mapping: dict):
        self._config_file = config_file
        self._config_key = config_key
        self._config_mapping = config_mapping

    def get_dictionary(self) -> dict:
        config = configparser.ConfigParser()
        config.read(self._config_file)
        return {key: config[self._config_key][value] for key, value in self._config_mapping.items()}


class RedshiftConfigFile(GenericConfigFile):
    """Expects a configuration file in the format of:

        [config_key]
        user=[user]
        password=[password]
        database=[database]
        port=[port]
        host=[host]
    """

    def __init__(self, config_file: str, config_key: str = 'redshift', config_mapping: Optional[dict] = None):

        if not config_mapping:
            config_mapping = {'user': 'username',
                              'password': 'password',
                              'database': 'database',
                              'port': 'port',
                              'host': 'host'}

        super().__init__(config_file=config_file, config_key=config_key, config_mapping=config_mapping)


class SnowflakeConfigFile(GenericConfigFile):
    """Expects a configuration file in the format of:

        [config_key]
        user=[user]
        account=[account]
        authenticator=[authenticator]
        warehouse=[warehouse]
        database=[database]
    """
    def __init__(self, config_file: str, config_key: str = 'snowflake',
                 config_mapping: Optional[dict] = None):

        if not config_mapping:
            config_mapping = {'user': 'username',
                              'account': 'account',
                              'authenticator': 'authenticator',
                              'warehouse': 'warehouse',
                              'database': 'database'}

        super().__init__(config_file=config_file, config_key=config_key, config_mapping=config_mapping)


class Redshift(Database):

    def __init__(self, user: str, password: str, database: str, host: str, port: Union[str, int]):
        super().__init__()
        self._connection_string = "dbname={} host={} port={} user={} password={}". \
            format(database, host, port, user, password)

    def _open_connection_object(self) -> object:
        from psycopg2 import connect
        return connect(self._connection_string)

    def _close_connection_object(self):
        self.connection_object.close()

    def query(self, sql: str) -> pd.DataFrame:
        assert self.is_open()
        return pd.read_sql_query(sql, self.connection_object)


class Snowflake(Database):
    """
    Instructions for installing snowflake dependencies:
        https://docs.snowflake.com/en/user-guide/python-connector-install.html

    I used v2.5.0, and python v3.9 (requirements_39)

    pip install -r https://raw.githubusercontent.com/snowflakedb/snowflake-connector-python/v2.5.0/tested_requirements/requirements_39.reqs
    pip install snowflake-connector-python==v2.5.0


    I also had to do:

    pip install snowflake-connector-python[pandas]
    https://docs.snowflake.com/en/user-guide/python-connector-pandas.html#installation
    """

    def __init__(self, user: str, account: str, authenticator: str,
                 warehouse: str, database: str, autocommit: bool = False):
        super().__init__()
        self.user = user
        self.account = account
        self.authenticator = authenticator
        self.warehouse = warehouse
        self.database = database
        self.autocommit = autocommit

    def _open_connection_object(self) -> object:
        from snowflake.connector import connect
        return connect(
            user=self.user,
            account=self.account,
            authenticator=self.authenticator,
            warehouse=self.warehouse,
            database=self.database,
            autocommit=self.autocommit,
        )

    def _close_connection_object(self):
        self.connection_object.close()

    def query(self, sql: str) -> pd.DataFrame:
        assert self.is_open()
        cursor = self.connection_object.cursor()
        cursor.execute(sql)
        dataframe = cursor.fetch_pandas_all()

        return dataframe
