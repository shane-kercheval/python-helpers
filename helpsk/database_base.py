import os
from abc import ABCMeta, abstractmethod
import configparser
import pandas as pd


class Configuration(metaclass=ABCMeta):
    @abstractmethod
    def get_dictionary(self) -> dict:
        pass



class Database(metaclass=ABCMeta):

    def __init__(self, **kwargs):
        self.connection_object = None

    @classmethod
    def from_config(cls, config: Configuration):
        return cls(**config.get_dictionary())

    def is_connected(self):
        return self.connection_object is not None

    @abstractmethod
    def _open_connection_object(self) -> object:
        pass

    @abstractmethod
    def _close_connection_object(self):
        pass

    def connect(self):
        if not self.is_connected():
            self.connection_object = self._open_connection_object()
            assert self.is_connected()

    def close(self):
        if self.is_connected():
            self._close_connection_object()
            self.connection_object = None
            assert not self.is_connected()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()

    @abstractmethod
    def query(self, sql: str) -> pd.DataFrame:
        pass
