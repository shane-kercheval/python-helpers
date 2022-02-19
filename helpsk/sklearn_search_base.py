"""
This module contains classes that define search spaces compatible with GridSearchCV, RandomSearchCV, or 
BayesSearchCV for classification models.
"""
from abc import abstractmethod, ABC
from typing import List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder

import helpsk as hlp
from helpsk.sklearn_pipeline import CustomOrdinalEncoder


class SearchSpaceBase(ABC):
    """
    Base class for defining a hyper-parameter "search space" for e.g. GridSearchCV, BayesianSearchCV, etc.

    This class defines the default pipeline to search over (e.g. imputing/scaling values, encoding, etc.) that
    will need to happen regardless of search space type (e.g. Grid vs Random vs Bayesian) or model type (e.g.
    classification vs regression).
    """
    def __init__(self, random_state: int = None):
        """initialization"""
        self._random_state = random_state

    @staticmethod
    def pipeline(data: pd.DataFrame) -> Pipeline:
        """
        This function defines the default pipeline to search over (e.g. imputing/scaling values, encoding,
        etc.) that will need to happen regardless of search space type (e.g. Grid vs Random vs Bayesian) or
        model type (e.g. classification vs regression).

        Args:
            data:
                a dataset (pd.DataFrame) that is going to be used to train the mode. This is used, for
                example, to determine which columns are numeric vs non-numeric, in order to build and return
                the pipeline.
        """
        numeric_column_names = hlp.pandas.get_numeric_columns(data)
        non_numeric_column_names = hlp.pandas.get_non_numeric_columns(data)

        numeric_pipeline = Pipeline([
            # tune how we want to impute values
            # e.g. whether or not we want to impute (and how) or simply remove rows with missing values
            ('imputer', hlp.sklearn_pipeline.TransformerChooser()),
            # tune how we want to scale values
            # e.g. MinMax/Normalization/None
            ('scaler', hlp.sklearn_pipeline.TransformerChooser()),
            ('pca', hlp.sklearn_pipeline.TransformerChooser()),
        ])
        non_numeric_pipeline = Pipeline([
            # tune how we handle categoric values
            # e.g. One Hot, Custom-OrdinalEncoder
            ('encoder', hlp.sklearn_pipeline.TransformerChooser()),
        ])
        # associate numeric/non-numeric columns with corresponding pipeline
        transformations_pipeline = ColumnTransformer([
            ('numeric', numeric_pipeline, numeric_column_names),
            ('non_numeric', non_numeric_pipeline, non_numeric_column_names)
        ])
        # add model to create the full pipeline
        full_pipeline = Pipeline([
            ('prep', transformations_pipeline),
            ('model', DummyClassifier())
        ])

        return full_pipeline

    @abstractmethod
    def search_spaces(self) -> list:
        """
        This method should return the search spaces based into the SearchCV object. For example, it should
        return a list of dictionaries if being passed to the `param_grid` parameter in the GridSearchCV's
        `__init__` function, or should return a list of tuples if being passed to the `search_spaces`
        parameter in the BayesSearchCV's `__init__` function.
        """

    def param_name_mappings(self) -> dict:
        """
        This function returns a dictionary, with the keys being the paths from the `sklearn.pipeline.Pipeline`
        returned by the `pipeline()` function (e.g. "prep__numeric__imputer") and transforms the
        path into a 'friendlier' value (e.g. "imputer"), returned as the value in the dictionary.

        The dictionary returned by this function can be used, for example, by passing it to the
        `parameter_name_mappings` parameter in the `MLExperimentResults.from_sklearn_search_cv()` function.
        This will allow the `MLExperimentResults` to use the friendlier names in the output (e.g. tables and
        graphs) and will make the output more readable.
        """
        mappings = {}
        for space in self.search_spaces():
            params = list(space[0].keys())
            for param in params:
                if param not in mappings:
                    if param == 'model':
                        mappings[param] = param
                    elif param.startswith('model__'):
                        mappings[param] = param.removeprefix('model__')
                    elif param.endswith('__transformer'):
                        mappings[param] = param.\
                            removesuffix('__transformer').\
                            removeprefix('prep__numeric__').\
                            removeprefix('prep__non_numeric__')
                    else:
                        mappings[param] = param

        ordered_mappings = {key: value for key, value in mappings.items() if not key.startswith('prep__')}
        ordered_mappings.update({key: value for key, value in mappings.items() if key.startswith('prep__')})
        return ordered_mappings


class ModelSearchSpaceBase(SearchSpaceBase, ABC):
    """
    Base class for defining model specific search spaces, regardless of search type (e.g. GridSearchCV,
    BayesSearchCV, etc.) or model type (e.g. classification, regression)

    See ModelBayesianSearchSpaceBase and e.g. LogisticBayesianSearchSpace for examples of how to inherit.
    """
    def __init__(self,
                 iterations: int = 50,
                 include_default_model: bool = True,
                 imputers=None,
                 scalers=None,
                 pca=None,
                 encoders=None,
                 random_state: int = None):
        """
        Args:
            iterations:

            include_default_model:

            imputers:

            scalers:

            encoders:

            random_state:
        """
        super().__init__(random_state)

        self._iterations = iterations
        self._include_default_model = include_default_model
        self._imputers = imputers
        self._scalers = scalers
        self._pca = pca
        self._encoders = encoders
        self._model_parameters = None

    @staticmethod
    def _create_default_imputers() -> list:
        """Return default imputers to be used in the search space"""
        return [
            SimpleImputer(strategy='mean'),
            SimpleImputer(strategy='median'),
            SimpleImputer(strategy='most_frequent')
        ]

    @staticmethod
    def _create_single_imputer() -> list:
        """Return a single imputer to be used, for example, when searching default hyper-param values"""
        return [SimpleImputer(strategy='mean')]

    @staticmethod
    def _create_default_scalers() -> list:
        """Return default scalers to be used in the search space"""
        return [
            StandardScaler(),
            MinMaxScaler(),
        ]

    @staticmethod
    def _create_single_scaler() -> list:
        """Return a single scaler to be used, for example, when searching default hyper-param values"""
        return [StandardScaler()]

    @staticmethod
    def _create_default_pca() -> list:
        """Return default PCA to be used in the search space"""
        return [
            None,
            PCA(n_components='mle')
        ]

    @staticmethod
    def _create_default_encoders() -> list:
        """Return default encoders to be used in the search space"""
        return [
            OneHotEncoder(handle_unknown='ignore'),
            CustomOrdinalEncoder(),
        ]

    @staticmethod
    def _create_single_encoder() -> list:
        """Return a single encoder to be used, for example, when searching default hyper-param values"""
        return [OneHotEncoder(handle_unknown='ignore')]

    @staticmethod
    def _build_transformer_search_space(imputers,
                                        scalers,
                                        pca,
                                        encoders) -> dict:
        """Takes imputers, scalers, etc., and constructions the transformation search space."""
        return {
            'prep__numeric__imputer__transformer': imputers,
            'prep__numeric__scaler__transformer': scalers,
            'prep__numeric__pca__transformer': pca,
            'prep__non_numeric__encoder__transformer': encoders,
        }

    @abstractmethod
    def _create_model(self):
        """This method returns a model object with whatever default values should be set."""

    @abstractmethod
    def _default_model_transformer_search_space(self) -> dict:
        """This method returns a dictionary of the default transformations to apply if including the default
        model in the search spaces (i.e. if `_include_default_model` is True). This can be accomplished by
        calling the `_build_transformer_search_space` function."""

    def _transformer_search_space(self) -> dict:
        """This method returns a dictionary of the transformations to tune. This can be accomplished by
        calling the `_build_transformer_search_space` function."""

        return self._build_transformer_search_space(
            imputers=self._imputers,
            scalers=self._scalers,
            pca=self._pca,
            encoders=self._encoders,
        )

    def _model_search_space(self) -> dict:
        """This method builds up the model's search space based on the `self._model_parameters` field that
        should be set by the inheriting class."""
        def add_param(param_dict, param_name, param_value):
            if param_value:
                param_dict['model__' + param_name] = param_value

        parameters = dict()
        for name, value in self._model_parameters.items():
            add_param(parameters, name, value)

        return parameters

    @abstractmethod
    def search_spaces(self) -> List[tuple]:
        """Returns a list of search spaces (e.g. 2 items if `include_default_model` is True; one for the
        model with default params, and one for searching across all params.)
        Each space is a tuple with a dictionary (hyper-param search space) as the first item and an integer
        (number of iterations) as second item."""
