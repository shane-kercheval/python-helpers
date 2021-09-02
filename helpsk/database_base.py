"""Contains the base/abstract classes that wraps the connection/querying logic of various databases.

See documentation in database.py
"""
from abc import ABCMeta, abstractmethod

import pandas as pd


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

        Return:
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
    def query(self, sql: str) -> pd.DataFrame:
        """Queries the database and returns the results as a pandas Dataframe

        Args:
            sql:
                SQL to execute e.g. "SELECT * FROM table LIMIT 100"

        Returns:
            a pandas Dataframe with the results from the query
        """
