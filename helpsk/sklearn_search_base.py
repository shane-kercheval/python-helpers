"""
This module contains classes that define search spaces compatible with GridSearchCV, RandomSearchCV, or 
BayesSearchCV for classification models.
"""
from abc import abstractmethod, ABC
from typing import List, Union, Callable

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from skopt.space import Categorical

import helpsk as hlp
from helpsk.sklearn_pipeline import CustomOrdinalEncoder


# search space has to return a pipeline (transformations and model) and a parameter space (transformations and model)

# You have a transfomrer search space which defines which transformations are going to happen
# and a model search space.  the model pipeline
# we need to differentiate between the **pipeline** which is the skeleton/recipe and the search space,
# which is the range of values for each item in the recipe
# recipe / map / pipeline
# ingredients / pipeline-space / param_grid / parameter_space
# search space

# both the transformations

class StandardTransformationPipelineBuilder:

    @staticmethod
    def default_transformation_pipeline(data: pd.DataFrame):
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

        return transformations_pipeline


class ModelSearchSpaceBase(ABC):
    """
    Base class for defining a hyper-parameter "search space" for e.g. GridSearchCV, BayesianSearchCV, etc.

    This class defines the interface/requirements for tuning a hyper-parameter space. Specifically, it
    combines the concepts of a `pipeline` (which is the outline/recipe of the transformations and model) and
    `parameter space` (which is the range of values to tune).
    """
    def __init__(self,
                 transformation_pipeline: Callable,
                 transformation_space: dict,
                 model_space: dict,
                 include_default_space: bool = True,
                 random_state: int = None):
        """initialization
        
            Args:
                transformation_pipeline:
                    This is a function that returns a Pipeline object that defines the components of the
                    transformations. A function needs to be passed to build the Pipeline object because
                    the pipeline might depend on information not available when we want to create the space,
                    e.g. we might need to know the numeric/non-numeric column names.
                transformation_space:
                    A dictionary that contain keys that correspond to the path of the
                    transformation pipeline and values that correspond to the range of values to search.

                    Note that the transformation and model pipelines will be combined, and the transformation
                    paths should be prefixed with `transformations__` and the model paths should be prefixed
                    with `model__`. For example, for GridSearchCV:

                        'transformations__non_numeric_pipeline__encoder_chooser__transformer': [
                            OneHotEncoder(),
                            CustomOrdinalEncoder()
                        ],

                model_space:
                    A dictionary that contain keys that correspond to the path of the
                    model parameters to tune..

                    Note that the transformation and model pipelines will be combined, and the transformation
                    paths should be prefixed with `transformations__` and the model paths should be prefixed
                    with `model__`. For example, for GridSearchCV:

                        'model__max_features': [100, 'auto'],

                include_default_space:
                    if True, `parameter_space` property returns an additional search space (that only contains
                    a single combination of parameters) corresponding to the default transformation space
                    defined for that model
                    default transformation space (single combination) and no
                random_state:
                    a random seed
        """
        self._transformation_pipeline = transformation_pipeline
        self._transformation_space = transformation_space
        self._model_space = model_space
        self._include_default_space = include_default_space
        self._random_state = random_state

    @abstractmethod
    def create_model_object(self) -> BaseEstimator:
        """Define the model object (BaseEstimator)."""

    @property
    @abstractmethod
    def default_transformation_space(self) -> dict:
        """Define the transformation space (single combination/iteration) corresponding to the
        default/standard transformations to apply to the model. Used to gauge baseline performance of the
        model."""

    @property
    @abstractmethod
    def parameter_space(self) -> Union[dict, list]:
        """
        This function combines the transformation and model search space and returns the full parameter
        spaces passed into the SearchCV object. It either returns dictionary if searching one search space
        or a list of dictionaries if searching multiple search spaces (e.g. include_default_model is True).
        Each search space corresponds to the single pipeline object return by `pipeline()`.
        """

    @property
    def pipeline(self, data: pd.DataFrame) -> Pipeline:

        full_pipeline = Pipeline([
            ('transformations', self._transformation_pipeline(data)),
            ('model', self.create_model_object)
        ])
        return full_pipeline

    @property
    def param_name_mappings(self) -> dict:
        """
        This function returns a dictionary, with the keys being the paths from the `sklearn.pipeline.Pipeline`
        returned by the `pipeline()` function (e.g. "transformations__numeric__imputer") and transforms the
        path into a 'friendlier' value (e.g. "imputer"), returned as the value in the dictionary.

        The dictionary returned by this function can be used, for example, by passing it to the
        `parameter_name_mappings` parameter in the `MLExperimentResults.from_sklearn_search_cv()` function.
        This will allow the `MLExperimentResults` to use the friendlier names in the output (e.g. tables and
        graphs) and will make the output more readable.
        """
        mappings = {}
        for space in self.parameter_space:
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
                            removeprefix('transformations__numeric__').\
                            removeprefix('transformations__non_numeric__')
                    else:
                        mappings[param] = param

        ordered_mappings = {key: value for key, value in mappings.items() if not key.startswith('transformations__')}
        ordered_mappings.update({key: value for key, value in mappings.items() if key.startswith('transformations__')})
        return ordered_mappings


class BayesianModelSearchSpaceBase(ModelSearchSpaceBase, ABC):

    def __init__(self,
                 transformation_pipeline: Pipeline = None,
                 transformation_space: dict = None,
                 model_space: dict = None,
                 include_default_space: bool = True,
                 iterations: int = 50,
                 random_state: int = None):
        super().__init__(
            transformation_pipeline=transformation_pipeline,
            transformation_space=transformation_space,
            model_space=model_space,
            include_default_space=include_default_space,
            random_state=random_state,
        )
        self._iterations = iterations

    @property
    def parameter_space(self) -> Union[dict, list]:
        # combine transformation and model space dictionaries and then create tuple with number of iterations
        param_space = {key: value for key, value in self._model_space.items()}
        param_space.update(self._transformation_space)
        search_spaces = (param_space, self._iterations)
        if self._include_default_space:
            default_space = {'model': Categorical([self.create_model_object])}
            default_space.update(self.default_transformation_space)
            search_spaces = [search_spaces,  (default_space, 1)]

        return search_spaces


class ModelSearchSpaceBaseArchive(ABC):
    """
    Base class for defining model specific search spaces, regardless of search type (e.g. GridSearchCV,
    BayesSearchCV, etc.) or model type (e.g. classification, regression)

    See ModelBayesianSearchSpaceBase and e.g. LogisticBayesianSearchSpace for examples of how to inherit.
    """
    def __init__(self,
                 iterations: int = 50,
                 include_default_model: bool = True,
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
            'transformations__numeric__imputer__transformer': imputers,
            'transformations__numeric__scaler__transformer': scalers,
            'transformations__numeric__pca__transformer': pca,
            'transformations__non_numeric__encoder__transformer': encoders,
        }

    @abstractmethod
    def _create_model(self):
        """This method returns a model object with whatever default values should be set."""

    @abstractmethod
    def _default_transformation_space(self) -> dict:
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
    def parameter_space(self) -> List[tuple]:
        """Returns a list of search spaces (e.g. 2 items if `include_default_model` is True; one for the
        model with default params, and one for searching across all params.)
        Each space is a tuple with a dictionary (hyper-param search space) as the first item and an integer
        (number of iterations) as second item."""
