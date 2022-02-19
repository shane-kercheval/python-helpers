from skopt.space import Real, Integer, Categorical, Dimension

from helpsk.sklearn_search_base import *


class DefaultValue(Dimension):
    def __init__(self):
        pass

    def set_transformer(self):
        pass

    @property
    def bounds(self):
        return None

    @property
    def is_constant(self):
        return None

    @property
    def transformed_bounds(self):
        return None


class DefaultReal(DefaultValue, Real):
    pass


class DefaultInteger(DefaultValue, Integer):
    pass


class DefaultCategorical(DefaultValue, Categorical):
    pass


class ModelBayesianSearchSpaceBase(ModelSearchSpaceBase, ABC):

    @staticmethod
    def _create_default_imputers() -> Categorical:
        return Categorical(categories=ModelSearchSpaceBase._create_default_imputers(),
            prior=[0.5, 0.25, 0.25]
        )

    @staticmethod
    def _create_single_imputer() -> Categorical:
        return Categorical(categories=ModelSearchSpaceBase._create_single_imputer())

    @staticmethod
    def _create_default_scalers() -> Categorical:
        return Categorical(categories=ModelSearchSpaceBase._create_default_scalers(),
            prior=[0.65, 0.35]
        )

    @staticmethod
    def _create_single_scaler() -> Categorical:
        return Categorical(categories=ModelSearchSpaceBase._create_single_scaler())

    @staticmethod
    def _create_default_pca() -> Categorical:
        return Categorical(categories=ModelSearchSpaceBase._create_default_pca())

    @staticmethod
    def _create_default_encoders() -> Categorical:
        return Categorical(categories=ModelSearchSpaceBase._create_default_encoders(),
            prior=[0.65, 0.35]
        )

    @staticmethod
    def _create_single_encoder() -> Categorical:
        return Categorical(categories=ModelSearchSpaceBase._create_single_encoder())

    @staticmethod
    def _create_empty_categorical() -> Categorical:
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
