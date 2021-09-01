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

    def is_open(self):
        return self.connection_object is not None

    @abstractmethod
    def _open_connection_object(self) -> object:
        pass

    @abstractmethod
    def _close_connection_object(self):
        pass

    def open(self):
        if not self.is_open():
            self.connection_object = self._open_connection_object()
            assert self.is_open()

    def close(self):
        if self.is_open():
            self._close_connection_object()
            self.connection_object = None
            assert not self.is_open()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()

    @abstractmethod
    def query(self, sql: str) -> pd.DataFrame:
        pass
