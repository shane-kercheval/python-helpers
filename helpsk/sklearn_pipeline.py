"""This module contains helper functions when working with sklearn (scikit-learn) objects;
in particular, for building pipelines"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder


class TransformerChooser(BaseEstimator, TransformerMixin):
    """Transformer that wraps another Transformer. This allows different transformer objects to be tuned.
    """
    def __init__(self, transformer=None):
        """
        Args:
            transformer:
                Transformer object (e.g. StandardScaler, MinMaxScaler)
        """
        self.transformer = transformer

    def fit(self, X, y=None):  # pylint: disable=invalid-name # noqa
        """fit implementation
        """
        if self.transformer is None:
            return self

        return self.transformer.fit(X, y)

    def transform(self, X):  # pylint: disable=invalid-name # noqa
        """transform implementation
        """
        if self.transformer is None:
            return X

        return self.transformer.transform(X)


class CustomOrdinalEncoder(BaseEstimator, TransformerMixin):
    """First replaces missing values with '<missing>' then applies OrdinalEncoder
    """

    def __init__(self):
        self._ordinal_encoder = OrdinalEncoder()  # unknown_value=-1,
        # handle_unknown='use_encoded_value')
        self._missing_value = '<missing>'

    def _fill_na(self, X):  # pylint: disable=invalid-name # noqa
        """Helper function that fills missing values with strings before calling OrdinalEncoder"""
        for column in X.columns.values:
            if X[column].dtype.name == 'category':
                if self._missing_value not in X[column].cat.categories:
                    X[column] = X[column].cat.add_categories(self._missing_value)
                X[column] = X[column].fillna(self._missing_value)

        return X

    def fit(self, X, y=None):  # pylint: disable=invalid-name,unused-argument # noqa
        """fit implementation"""
        X = self._fill_na(X)  # pylint: disable=invalid-name # noqa
        self._ordinal_encoder.fit(X)
        return self

    def transform(self, X):  # pylint: disable=invalid-name # noqa
        """transform implementation"""
        X = self._fill_na(X)  # pylint: disable=invalid-name # noqa
        return self._ordinal_encoder.transform(X)
