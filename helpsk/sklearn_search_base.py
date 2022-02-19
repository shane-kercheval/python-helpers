from abc import abstractmethod, ABC
from typing import List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

import helpsk as hlp


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
