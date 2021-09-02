"""This model contains classes that wrap the connection/querying logic of various databases. Database objects
can be created from the __init__ function or from the .from_config class function by passing in an instance
of the corresponding configuration class.

Examples:

    with Redshift(user='username', password='asdf', ....) as redshift:
        redshift.query("SELECT * FROM table LIMIT 100")

    or

    redshift_config = RedshiftConfigFile('/path/to/redshift.config')
    with Redshift.from_config(redshift_config) as redshift:
        redshift.query("SELECT * FROM table LIMIT 100")
"""
import configparser
from typing import Union

import pandas as pd

from helpsk.database_base import Configuration, Database


class GenericConfigFile(Configuration):  # pylint: disable=too-few-public-methods
    """Class that is used to map a configuration file in the format below, to a dictionary that will be passed
    into the corresponding Database object.

        [config_key]
        user=[user]
        password=[password]
        database=[database]
        port=[port]
        host=[host]
    """
    def __init__(self, config_file: str, config_key: str, config_mapping: dict):
        """
        Args:
             config_file:
                the path to the configuration file

             config_key:
                the configuration key i.e. keyword of configuration items

             config_mapping:
                a dictionary containing the mapping to go from the configuration item names to the keys of the
                dictionary created. The keys should be the same as the Database object's constructor.

                So if the Database constructor is `__init__(user, password, database)`

                and the configuration file is in the format of:

                    [database_config]
                    user_item=robert
                    password_item=123
                    database_item=my_database

                then the config_mapping dict should be

                    {'user': 'user_item',
                     'password': 'password_item',
                     'database': 'database_item'}

                and the resulting dict from calling get_dictionary would be

                    {'user': 'robert',
                     'password': '123',
                     'database': 'my_database'}

                which would be passed into the Database object, either as

                    config_file = GenericConfigFile(...)
                    Database(**config_file.get_dictionary())

                or

                    config_file = GenericConfigFile(...)
                    Database.from_config(config_file)
        """
        self._config_file = config_file
        self._config_key = config_key
        self._config_mapping = config_mapping

    def get_dictionary(self) -> dict:
        """ logic that builds the dictionary described in the __init__ docstring

        Returns:
             a dictionary to be passed to the Database object
        """
        config = configparser.ConfigParser()
        config.read(self._config_file)
        return {key: config[self._config_key][value] for key, value in self._config_mapping.items()}


class RedshiftConfigFile(GenericConfigFile):  # pylint: disable=too-few-public-methods
    """Supplies a standard config_key and config_mapping to the GenericConfigFile object.

    Corresponds to a configuration file in the format of:

        [redshift]
        user=[user]
        password=[password]
        database=[database]
        port=[port]
        host=[host]
    """
    def __init__(self, config_file: str, config_key: str = 'redshift'):
        """
        Args:
            config_file:
                the path to the configuration file

            config_key:
                the configuration key, i.e. the `[xxx]` part of the configuration file.
                The default value is `redshift`.

        """
        config_mapping = {'user': 'username',
                          'password': 'password',
                          'database': 'database',
                          'port': 'port',
                          'host': 'host'}

        super().__init__(config_file=config_file, config_key=config_key, config_mapping=config_mapping)


class SnowflakeConfigFile(GenericConfigFile):  # pylint: disable=too-few-public-methods
    """Supplies a standard config_key and config_mapping to the GenericConfigFile object.

    Corresponds to a configuration file in the format of:

        [snowflake]
        user=[user]
        account=[account]
        authenticator=[authenticator]
        warehouse=[warehouse]
        database=[database]
    """
    def __init__(self, config_file: str, config_key: str = 'snowflake'):
        """
        Args:
            config_file:
                the path to the configuration file

            config_key:
                the configuration key, i.e. the `[xxx]` part of the configuration file.
                The default value is `snowflake`.

        """
        config_mapping = {'user': 'username',
                          'account': 'account',
                          'authenticator': 'authenticator',
                          'warehouse': 'warehouse',
                          'database': 'database'}

        super().__init__(config_file=config_file, config_key=config_key, config_mapping=config_mapping)


class Redshift(Database):
    """Wraps logic for connecting to Redshift and querying.

    Example:
        config = RedshiftConfigFile('/path/to/redshift.config')
        with Redshift.from_config(config) as redshift:
            redshift.query("SELECT * FROM table LIMIT 100")
    """
    def __init__(self, user: str, password: str, database: str, host: str, port: Union[str, int]):  # pylint: disable=too-many-arguments
        """Initialization"""
        super().__init__()
        self._connection_string = "dbname={} host={} port={} user={} password={}". \
            format(database, host, port, user, password)

    def _open_connection_object(self) -> object:
        """Wraps logic for connecting to redshift
        """
        from psycopg2 import connect  # pylint: disable=import-outside-toplevel
        return connect(self._connection_string)

    def _close_connection_object(self):
        """Wraps logic for closing the connection to redshift
        """
        self.connection_object.close()

    def query(self, sql: str) -> pd.DataFrame:
        """Wraps logic for querying redshift.

        Args:
            sql:
                SQL to execute e.g. "SELECT * FROM table LIMIT 100"

        Returns:
            a pandas Dataframe with the results from the query
        """
        assert self.is_connected()
        return pd.read_sql_query(sql, self.connection_object)


class Snowflake(Database):
    """Wraps logic for connecting to Snowflake and querying.

        Example:
            config = SnowflakeConfigFile('/path/to/snowflake.config')
            with Snowflake.from_config(config) as snowflake:
                snowflake.query("SELECT * FROM table LIMIT 100")

    Instructions for installing snowflake dependencies:
        https://docs.snowflake.com/en/user-guide/python-connector-install.html

    I used v2.5.0, and python v3.9 (requirements_39):

    ```
    pip install -r ......./snowflake-connector-python/v2.5.0/tested_requirements/requirements_39.reqs
    pip install snowflake-connector-python==v2.5.0
    ```

    Additionally:
        https://docs.snowflake.com/en/user-guide/python-connector-pandas.html#installation

    ```
    pip install snowflake-connector-python[pandas]
    ```
    """
    # pylint: disable=too-many-arguments
    def __init__(self, user: str, account: str, authenticator: str,
                 warehouse: str, database: str, autocommit: bool = False):
        """Initialization"""
        super().__init__()
        self.user = user
        self.account = account
        self.authenticator = authenticator
        self.warehouse = warehouse
        self.database = database
        self.autocommit = autocommit

    def _open_connection_object(self) -> object:
        """Wraps logic for connecting to snowflake
        """
        from snowflake.connector import connect  # pylint: disable=import-outside-toplevel
        return connect(
            user=self.user,
            account=self.account,
            authenticator=self.authenticator,
            warehouse=self.warehouse,
            database=self.database,
            autocommit=self.autocommit,
        )

    def _close_connection_object(self):
        """Wraps logic for closing the connection to snowflake
        """
        self.connection_object.close()

    def query(self, sql: str) -> pd.DataFrame:
        """Wraps logic for querying snowflake.

        Args:
            sql:
                SQL to execute e.g. "SELECT * FROM table LIMIT 100"

        Returns:
            a pandas Dataframe with the results from the query
        """
        assert self.is_connected()
        cursor = self.connection_object.cursor()
        cursor.execute(sql)
        dataframe = cursor.fetch_pandas_all()

        return dataframe
