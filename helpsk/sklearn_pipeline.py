"""
Contains helper functions when working with sklearn (scikit-learn) objects; in particular, for
building pipelines.
"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder


class TransformerChooser(BaseEstimator, TransformerMixin):
    """
    Transformer that wraps another Transformer. This allows different transformer objects to be
    tuned.
    """

    def __init__(self, transformer: BaseEstimator | None = None):
        """
        Args:
            transformer:
                Transformer object (e.g. StandardScaler, MinMaxScaler).
        """
        self.transformer = transformer

    def fit(self, X, y=None):  # noqa
        """Fit implementation."""
        if self.transformer is None:
            return self

        return self.transformer.fit(X, y)

    def transform(self, X):  # noqa
        """Transform implementation."""
        if self.transformer is None:
            return X

        return self.transformer.transform(X)


class CustomOrdinalEncoder(BaseEstimator, TransformerMixin):
    """First replaces missing values with '<missing>' then applies OrdinalEncoder."""

    def __init__(self, handle_unknown: str = 'use_encoded_value'):
        self._ordinal_encoder = OrdinalEncoder(handle_unknown=handle_unknown, unknown_value=-1)
        # unknown_value=-1,
        # handle_unknown='use_encoded_value')
        self.handle_unknown = handle_unknown
        self._missing_value = '<missing>'

    def _fill_na(self, X):  # noqa
        """Helper function that fills missing values with strings before calling OrdinalEncoder."""
        for column in X.columns.to_numpy():
            if X[column].dtype.name == 'category':
                if self._missing_value not in X[column].cat.categories:
                    X[column] = X[column].cat.add_categories(self._missing_value)
                X[column] = X[column].fillna(self._missing_value)

        return X

    def fit(self, X, y=None):  # noqa
        """Fit implementation."""
        X = self._fill_na(X)  # noqa: N806
        self._ordinal_encoder.fit(X)
        return self

    def transform(self, X):  # noqa
        """Transform implementation."""
        X = self._fill_na(X)  # noqa: N806
        return self._ordinal_encoder.transform(X)
