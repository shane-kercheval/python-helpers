"""Contains the base/abstract classes that wraps the connection/querying logic of various databases.

See documentation in database.py
"""
from abc import ABCMeta, abstractmethod
import time
import pandas as pd
from helpsk.utility import suppress_stdout, suppress_warnings


class Configuration(metaclass=ABCMeta):  # pylint: disable=too-few-public-methods
    """A basic configuration object will product a dictionary representing the keyword parameters that can
    be passed to a Database object.
    """
    @abstractmethod
    def get_dictionary(self) -> dict:
        """
        Returns:
             a dictionary to be passed to the Database object
        """


class Database(metaclass=ABCMeta):
    """
    Base class that wraps the connection/querying logic of various databases.
    """
    def __init__(self, **kwargs):  # pylint: disable=unused-argument
        self.connection_object = None

    @classmethod
    def from_config(cls, config: Configuration):
        """Takes a Configuration object that contains the connection details to the database.
        """
        return cls(**config.get_dictionary())

    def is_connected(self):
        """
        Returns:
            True if the database is connected, otherwise False
        """
        return self.connection_object is not None

    @abstractmethod
    def _open_connection_object(self) -> object:
        """Child classes will implement the logic to connect to the database and return the connection object.

        The returning value will be stored in self.connection_object

        Returns:
            the connection object (usually returned from the underlying `connect()` method. For example, the
            returning value from `snowflake.connection.connect()`
        """

    @abstractmethod
    def _close_connection_object(self):
        """Child classes will implement the logic to close the connection to the database, typically by
        calling `close()` on the connection object returned from _open_connection_object.

        For example:
            self.connection_object.close()
        """

    def connect(self):
        """Call this method to open the connection to the database.

        Alternatively you can use the context manager:

        ```
        with Database(...) as database:
            database.query("SELECT * FROM table LIMIT 100")
        ```
        """
        if not self.is_connected():
            self.connection_object = self._open_connection_object()
            assert self.is_connected()

    def close(self):
        """Call this method to close the connection to the database.

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
        """Enables the context manager"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Enables the context manager"""
        self.close()

    @abstractmethod
    def _query(self, sql: str) -> pd.DataFrame:
        """
        Method for child classes to override that contains logic to query

        Args:
            sql:
                SQL to execute e.g. "SELECT * FROM table LIMIT 100"

        Returns:
            a pandas Dataframe with the results from the query
        """

    def query(self, sql: str, *,
              show_messages: bool = False, show_warnings: bool = False,
              show_elapsed_time: bool = True) -> pd.DataFrame:
        """Queries the database and returns the results as a pandas Dataframe

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
            a pandas Dataframe with the results from the query
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
    def execute_statement(self, statement: str):
        """This method executes a statement without any data returned."""

    @abstractmethod
    def insert_records(self,
                        dataframe: pd.DataFrame,
                        table: str,
                        create_table: bool = False,
                        overwrite: bool = True,
                        schema: str = None,
                        database: str = None):
        """
        This method inserts rows into a table from a pandas DataFrame.

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
