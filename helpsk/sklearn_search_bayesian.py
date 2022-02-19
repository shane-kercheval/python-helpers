"""
This module contains classes that define search spaces compatible with BayesSearchCV for classification 
models.
"""
from helpsk.sklearn_search_bayesian_classification import *


class BayesianSearchSpace(SearchSpaceBase):
    """
    Wrapper/container that is used for specifying multiple models/spaces for the BayesSearchCV search space.
    The user can pass a list of ModelBayesianSearchSpaceBase objects, or can allow the class to define the
    search spaces. Can be used for classification or regression search spaces.
    """
    def __init__(self,
                 data: pd.DataFrame,
                 model_search_spaces: List[ModelBayesianSearchSpaceBase] = None,
                 iterations: int = 50,
                 include_default_model: bool = True,
                 model_type: str = 'classification',
                 random_state: int = None):

        assert model_type in ['classification', 'regression']

        super().__init__(random_state=random_state)
        self._data = data
        # self._numeric_column_names = hlp.pandas.get_numeric_columns(data)
        # self._non_numeric_column_names = hlp.pandas.get_non_numeric_columns(data)
        if model_search_spaces:
            self._model_search_spaces = model_search_spaces
        else:

            if model_type == 'classification':
                self._model_search_spaces = [
                    LogisticBayesianSearchSpace(iterations=iterations,
                                                include_default_model=include_default_model,
                                                random_state=random_state),
                    LinearSVCBayesianSearchSpace(iterations=iterations,
                                                 include_default_model=include_default_model,
                                                 random_state=random_state),
                    ExtraTreesBayesianSearchSpace(iterations=iterations,
                                                  include_default_model=include_default_model,
                                                  random_state=random_state),
                    RandomForestBayesianSearchSpace(iterations=iterations,
                                                    include_default_model=include_default_model,
                                                    random_state=random_state),
                    XGBoostBayesianSearchSpace(iterations=iterations,
                                               include_default_model=include_default_model,
                                               random_state=random_state),
                ]
            elif model_type == 'regression':
                raise NotImplementedError()
            else:
                raise NotImplementedError()

    def pipeline(self):
        return super().pipeline(self._data)

    def search_spaces(self) -> List[tuple]:
        all_spaces = []
        for space in self._model_search_spaces:  # each `space.search_spaces()` is a list of tuples
            all_spaces = all_spaces + space.search_spaces()

        return all_spaces
