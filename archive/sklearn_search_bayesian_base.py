"""
This module contains base classes that define search spaces compatible with BayesSearchCV for either
classification or regression models.
"""
from skopt.space import Real, Integer, Categorical, Dimension
from helpsk.sklearn_search_base import *


class DefaultValue(Dimension):
    """
    A class to indicate a default value (for a model's parameter) to be used in a
    `ModelBayesianSearchSpaceBase` class.
    """
    def __init__(self):
        """this method is not intended to be used directly, the class itself is used to indicate state."""
        pass

    def set_transformer(self):
        """this method is not intended to be used directly, the class itself is used to indicate state."""
        pass

    @property
    def bounds(self):
        """this method is not intended to be used directly, the class itself is used to indicate state."""
        return None

    @property
    def is_constant(self):
        """this method is not intended to be used directly, the class itself is used to indicate state."""
        return None

    @property
    def transformed_bounds(self):
        """this method is not intended to be used directly, the class itself is used to indicate state."""
        return None


class DefaultReal(DefaultValue, Real):
    """
    A class to indicate a default value of type Real (for a model's parameter) to be used in a
    `ModelBayesianSearchSpaceBase` class.
    """
    pass


class DefaultInteger(DefaultValue, Integer):
    """
    A class to indicate a default value of type Integer (for a model's parameter) to be used in a
    `ModelBayesianSearchSpaceBase` class.
    """
    pass


class DefaultCategorical(DefaultValue, Categorical):
    """
    A class to indicate a default value of type Categorical (for a model's parameter) to be used in a
    `ModelBayesianSearchSpaceBase` class.
    """
    pass


class ModelBayesianSearchSpaceBase(ModelSearchSpaceBase, ABC):
    """
    Base class for defining model specific search spaces for BayesSearchCV, regardless of model type (e.g.
    classification, regression)
    """
    @staticmethod
    def _create_default_imputers() -> Categorical:
        """Return default imputers to be used in the search space"""
        return Categorical(categories=ModelSearchSpaceBase._create_default_imputers(),
            prior=[0.5, 0.25, 0.25]
        )

    @staticmethod
    def _create_single_imputer() -> Categorical:
        """Return a single imputer to be used, for example, when searching default hyper-param values"""
        return Categorical(categories=ModelSearchSpaceBase._create_single_imputer())

    @staticmethod
    def _create_default_scalers() -> Categorical:
        """Return default scalers to be used in the search space"""
        return Categorical(categories=ModelSearchSpaceBase._create_default_scalers(),
            prior=[0.65, 0.35]
        )

    @staticmethod
    def _create_single_scaler() -> Categorical:
        """Return a single scaler to be used, for example, when searching default hyper-param values"""
        return Categorical(categories=ModelSearchSpaceBase._create_single_scaler())

    @staticmethod
    def _create_default_pca() -> Categorical:
        """Return default PCA to be used in the search space"""
        return Categorical(categories=ModelSearchSpaceBase._create_default_pca())

    @staticmethod
    def _create_default_encoders() -> Categorical:
        """Return default encoders to be used in the search space"""
        return Categorical(categories=ModelSearchSpaceBase._create_default_encoders(),
            prior=[0.65, 0.35]
        )

    @staticmethod
    def _create_single_encoder() -> Categorical:
        """Return a single encoder to be used, for example, when searching default hyper-param values"""
        return Categorical(categories=ModelSearchSpaceBase._create_single_encoder())

    @staticmethod
    def _create_empty_categorical() -> Categorical:
        """Return empty categorical to be used in the search space"""
        return Categorical([None])

    def search_spaces(self) -> List[tuple]:
        """Returns a list of search spaces (e.g. 2 items if `include_default_model` is True; one for the
        model with default params, and one for searching across all params.)
        Each space is a tuple with a dictionary (hyper-param search space) as the first item and an integer
        (number of iterations) as second item."""
        from skopt.space import Categorical

        model_search_space = {'model': Categorical([self._create_model()])}
        model_search_space.update(self._model_search_space())
        model_search_space.update(self._transformer_search_space())
        search_spaces = [(
            model_search_space,
            self._iterations
        )]

        if self._include_default_model:
            default_space = {'model': Categorical([self._create_model()])}
            default_space.update(self._default_model_transformer_search_space())
            search_spaces = search_spaces + [(default_space, 1)]

        return search_spaces
