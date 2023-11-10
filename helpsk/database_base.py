"""
Contains the base classes that wraps the connection/querying logic of
various databases.

See documentation in database.py
"""

from abc import ABC, abstractmethod
from typing import TypeVar
import configparser
import time
import pandas as pd
from helpsk.utility import suppress_stdout, suppress_warnings


ConnectionObject = TypeVar('ConnectionObject')


class Database(ABC):
    """Wraps the connection/querying logic of various databases."""

    def __init__(self, **kwargs: dict):
        """
        **kwargs should contain the names of the underlying connection object and corresponding
        values.

        For example:
            - For Redshift, typical keyword arguments might be:
                - `dbname`, `password`, `database`, `port`, `host`
            - For Snowflake, typical keyword arguments might be:
                - 'user', 'account', 'authenticator', 'warehouse', 'database', 'autocommit'
        """
        self._kwargs = kwargs
        self.connection_object: ConnectionObject = None

    @classmethod
    def from_config(cls, config_path: str, config_key: str) -> 'Database':  # noqa: ANN102
        """
        Passes key/value pairs found in configuration file into underlying connection. Keys must
        match corresponding connection string or connection method arguments.

        For example:
            - For Redshift, typical keys might be:
                - `dbname`, `password`, `database`, `port`, `host`

            ```
            [redshift]
            user=my_username
            password=my-password-123
            dbname=the_database
            host=host.address.redshift.amazonaws.com
            port=1234
            ```

            - For Snowflake, typical keys might be:
                - 'user', 'account', 'authenticator', 'warehouse', 'database', 'autocommit'

            ```
            [snowflake]
            user=my.email@address.com
            account=account.id
            authenticator=externalbrowser
            warehouse=WAREHOUSE_NAME
            database=DATABASE_NAME
            ```

        Args:
            config_path:
                path to the configuration file containing key/value pairs.
            config_key:
                name of the configuration key in configuration file
                e.g. `[redshift]`
        """
        config = configparser.ConfigParser()
        config.read(config_path)
        config_dict = dict(config[config_key].items())
        return cls(**config_dict)

    def is_connected(self) -> bool:
        """Returns True if the database is connected, otherwise False."""
        return self.connection_object is not None

    @abstractmethod
    def _open_connection_object(self) -> ConnectionObject:
        """
        Child classes will implement the logic to connect to the database and return the
        connection object.

        The returning value will be stored in self.connection_object

        Returns the connection object (usually returned from the underlying `connect()` method.
        For example, the returning value from `snowflake.connection.connect()`
        """

    @abstractmethod
    def _close_connection_object(self) -> None:
        """
        Child classes will implement the logic to close the connection to the database,
        typically by calling `close()` on the connection object returned from
        _open_connection_object.

        For example:
            self.connection_object.close()
        """

    def connect(self) -> None:
        """
        Call this method to open the connection to the database.

        Alternatively you can use the context manager:

        ```
        with Database(...) as database:
            database.query("SELECT * FROM table LIMIT 100")
        ```
        """
        if not self.is_connected():
            self.connection_object = self._open_connection_object()
            assert self.is_connected()

    def close(self) -> None:
        """
        Call this method to close the connection to the database.

        Alternatively you can use the context manager:

        ```
        with Database(...) as database:
            database.query("SELECT * FROM table LIMIT 100")
        ```
        """
        if self.is_connected():
            self._close_connection_object()
            self.connection_object = None
            assert not self.is_connected()

    def __enter__(self):
        """Enables the context manager."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):  # noqa: ANN001
        """Enables the context manager."""
        self.close()

    @abstractmethod
    def _query(self, sql: str) -> pd.DataFrame:
        """
        Method for child classes to override that contains logic to query.

        Args:
            sql:
                SQL to execute e.g. "SELECT * FROM table LIMIT 100"

        Returns:
            a pandas Dataframe with the results from the query
        """

    def query(self, sql: str, *,
              show_messages: bool = False,
              show_warnings: bool = False,
              show_elapsed_time: bool = True) -> pd.DataFrame:
        """
        Queries the database and returns the results as a pandas Dataframe.

        Args:
            sql:
                SQL to execute e.g. "SELECT * FROM table LIMIT 100"
            show_messages:
                if True, shows messages generated from the underlying query
            show_warnings:
                if True, shows warnings generated from the underlying query
            show_elapsed_time:
                if True, shows the elapsed execution time for the query

        Returns:
            a pandas Dataframe with the results from the query.
        """
        assert self.is_connected()
        start_time = time.time()
        if show_messages and show_warnings:
            with suppress_stdout(), suppress_warnings():
                results = self._query(sql=sql)
        elif show_messages:
            with suppress_stdout():
                results = self._query(sql=sql)
        elif show_warnings:
            with suppress_warnings():
                results = self._query(sql=sql)
        else:
            results = self._query(sql=sql)

        end_time = time.time()

        if show_elapsed_time:
            elapsed_time = end_time - start_time
            time_units = 'minutes' if elapsed_time > 60 else 'seconds'
            elapsed_time = elapsed_time / 60 if elapsed_time > 60 else elapsed_time
            elapsed_time = round(elapsed_time, 1)
            print(f'Elapsed Time: {elapsed_time} {time_units}')

        return results

    @abstractmethod
    def execute_statement(self, statement: str) -> None:
        """Executes a statement without any data returned."""

    @abstractmethod
    def insert_records(
        self,
        dataframe: pd.DataFrame,
        table: str,
        create_table: bool = False,
        overwrite: bool = False,
        schema: str | None = None,
        database: str | None = None) -> None:
        """
        Inserts rows into a table from a pandas DataFrame.

        Args:
            dataframe:
                the pandas dataframe to insert
            table:
                the name of the table
            create_table:
                if True, creates the table before inserting
            overwrite:
                if True, drops all records before inserting
            schema:
                the name of the schema
            database:
                the name of the database
        """
