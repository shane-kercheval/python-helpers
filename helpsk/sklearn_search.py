from abc import abstractmethod, ABC
from typing import List, Union

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.svm import LinearSVC

from skopt.space import Real, Integer, Categorical, Dimension

import helpsk as hlp
from helpsk.sklearn_pipeline import CustomOrdinalEncoder


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


class BayesianSearchSpaceBase(ABC):
    def __init__(self, random_state: int = None):
        self._random_state = random_state

    @staticmethod
    def pipeline(data: pd.DataFrame):
        numeric_column_names = hlp.pandas.get_numeric_columns(data)
        non_numeric_column_names = hlp.pandas.get_non_numeric_columns(data)

        numeric_pipeline = Pipeline([
            # tune how we want to impute values
            # e.g. whether or not we want to impute (and how) or simply remove rows with missing values
            ('imputer', hlp.sklearn_pipeline.TransformerChooser()),
            # tune how we want to scale values
            # e.g. MinMax/Normalization/None
            ('scaler', hlp.sklearn_pipeline.TransformerChooser()),
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
    def search_spaces(self) -> List[tuple]:
        """"""

    def param_name_mappings(self) -> dict:
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


class ModelBayesianSearchSpaceBase(BayesianSearchSpaceBase, ABC):
    def __init__(self,
                 iterations: int = 50,
                 include_default_model: bool = True,
                 imputers: Categorical = DefaultCategorical(),
                 scalers: Categorical = DefaultCategorical(),
                 encoders: Categorical = DefaultCategorical(),
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

        self._iterations = iterations
        self._include_default_model = include_default_model
        self._imputers = imputers
        self._scalers = scalers
        self._encoders = encoders
        self._model_parameters = None

        super().__init__(random_state)

    @staticmethod
    def _get_default_imputers() -> Categorical:
        return Categorical(categories=[
                SimpleImputer(strategy='mean'),
                SimpleImputer(strategy='median'),
                SimpleImputer(strategy='most_frequent')
            ],
            prior=[0.5, 0.25, 0.25]
        )

    @staticmethod
    def _get_single_imputer() -> Categorical:
        return Categorical(categories=[SimpleImputer(strategy='mean')])

    @staticmethod
    def _get_default_scalers() -> Categorical:
        return Categorical(categories=[
                StandardScaler(),
                MinMaxScaler(),
            ],
            prior=[0.65, 0.35]
        )

    @staticmethod
    def _get_single_scaler() -> Categorical:
        return Categorical(categories=[StandardScaler()])

    @staticmethod
    def _get_default_encoders() -> Categorical:
        return Categorical(categories=[
                OneHotEncoder(handle_unknown='ignore'),
                CustomOrdinalEncoder(),
            ],
            prior=[0.65, 0.35]
        )

    @staticmethod
    def _get_single_encoder() -> Categorical:
        return Categorical(categories=[OneHotEncoder(handle_unknown='ignore')])

    @staticmethod
    def _get_empty_categorical() -> Categorical:
        return Categorical([None])

    @staticmethod
    def _build_transformer_search_space(imputers: Categorical,
                                        scalers: Categorical,
                                        encoders: Categorical) -> dict:

        return {
            'prep__numeric__imputer__transformer': imputers,
            'prep__numeric__scaler__transformer': scalers,
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
            encoders=self._encoders,
        )

    def _model_search_space(self) -> dict:
        def add_param(param_dict, param_name, param_value):
            if param_value:
                param_dict['model__' + param_name] = param_value

        parameters = dict()
        for name, value in self._model_parameters.items():
            add_param(parameters, name, value)

        return parameters

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

    def param_name_mappings(self):
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


class LogisticBayesianSearchSpace(ModelBayesianSearchSpaceBase):
    def __init__(self,
                 C: Union[Real, None] = DefaultReal(),  # noqa
                 # model default value options
                 solver: str = 'lbfgs',
                 max_iter: int = 1000,
                 # search space options
                 iterations: int = 50,
                 include_default_model: bool = True,
                 # transformation search space options
                 imputers: Union[Categorical, None] = DefaultCategorical(),
                 scalers: Union[Categorical, None] = DefaultCategorical(),
                 encoders: Union[Categorical, None] = DefaultCategorical(),
                 random_state: int = None):

        if isinstance(imputers, DefaultValue):
            imputers = self._get_default_imputers()
        if isinstance(scalers, DefaultValue):
            scalers = self._get_default_scalers()
        if isinstance(encoders, DefaultValue):
            encoders = self._get_default_encoders()

        super().__init__(iterations=iterations,
                         include_default_model=include_default_model,
                         imputers=imputers,
                         scalers=scalers,
                         encoders=encoders,
                         random_state=random_state)
        self._model_parameters = dict(
            C=Real(low=1e-6, high=100, prior='log-uniform') if isinstance(C, DefaultValue) else C,
        )

        self._solver = solver
        self._max_iter = max_iter

    def _create_model(self):
        return LogisticRegression(
            solver=self._solver,
            max_iter=self._max_iter,
            random_state=self._random_state
        )

    def _default_model_transformer_search_space(self) -> dict:
        return self._build_transformer_search_space(
            imputers=self._get_single_imputer(),
            scalers=self._get_single_scaler(),
            encoders=self._get_single_encoder(),
        )


class LinearSVCBayesianSearchSpace(ModelBayesianSearchSpaceBase):
    def __init__(self,
                 # hyper-params search space
                 C: Union[Real, None] = DefaultReal(),  # noqa
                 # search space options
                 iterations: int = 50,
                 include_default_model: bool = True,
                 # transformation search space options
                 imputers: Union[Categorical, None] = DefaultCategorical(),
                 scalers: Union[Categorical, None] = DefaultCategorical(),
                 encoders: Union[Categorical, None] = DefaultCategorical(),
                 random_state: int = None):

        if isinstance(imputers, DefaultValue):
            imputers = self._get_default_imputers()
        if isinstance(scalers, DefaultValue):
            scalers = self._get_default_scalers()
        if isinstance(encoders, DefaultValue):
            encoders = self._get_default_encoders()

        super().__init__(iterations=iterations,
                         include_default_model=include_default_model,
                         imputers=imputers,
                         scalers=scalers,
                         encoders=encoders,
                         random_state=random_state)
        self._model_parameters = dict(
            C=Real(low=1e-6, high=100, prior='log-uniform') if isinstance(C, DefaultValue) else C,
        )

    def _create_model(self):
        return LinearSVC(
            random_state=self._random_state
        )

    def _default_model_transformer_search_space(self) -> dict:
        return self._build_transformer_search_space(
            imputers=self._get_single_imputer(),
            scalers=self._get_single_scaler(),
            encoders=self._get_single_encoder(),
        )


class TreesBayesianSearchSpaceBase(ModelBayesianSearchSpaceBase, ABC):
    def __init__(self,
                 # hyper-params search space
                 max_features: Union[Real, None] = DefaultReal(),
                 max_depth: Union[Integer, None] = DefaultInteger(),
                 min_samples_split: Union[Integer, None] = DefaultInteger(),
                 min_samples_leaf: Union[Integer, None] = DefaultInteger(),
                 max_samples: Union[Real, None] = DefaultReal(),
                 criterion: Union[Categorical, None] = DefaultCategorical(),
                 # search space options
                 iterations: int = 50,
                 include_default_model: bool = True,
                 # transformation search space options
                 imputers: Union[Categorical, None] = DefaultCategorical(),
                 scalers: Union[Categorical, None] = DefaultCategorical(),
                 encoders: Union[Categorical, None] = DefaultCategorical(),
                 random_state: int = None):

        if isinstance(imputers, DefaultValue):
            imputers = self._get_default_imputers()
        if isinstance(scalers, DefaultValue):
            scalers = self._get_empty_categorical()
        if isinstance(encoders, DefaultValue):
            encoders = self._get_default_encoders()

        super().__init__(iterations=iterations,
                         include_default_model=include_default_model,
                         imputers=imputers,
                         scalers=scalers,
                         encoders=encoders,
                         random_state=random_state)

        self._model_parameters = dict(
            max_features=Real(low=0.01, high=0.95, prior='uniform') if isinstance(max_features, DefaultValue) else max_features,
            max_depth=Integer(low=1, high=100, prior='uniform') if isinstance(max_depth, DefaultValue) else max_depth,
            min_samples_split=Integer(low=2, high=50, prior='uniform') if isinstance(min_samples_split, DefaultValue) else min_samples_split,
            min_samples_leaf=Integer(low=1, high=50, prior='uniform') if isinstance(min_samples_leaf, DefaultValue) else min_samples_leaf,
            max_samples=Real(low=0.5, high=1.0, prior='uniform') if isinstance(max_samples, DefaultValue) else max_samples,
            criterion=Categorical(['gini', 'entropy']) if isinstance(criterion, DefaultValue) else criterion,
        )

    def _default_model_transformer_search_space(self) -> dict:
        return self._build_transformer_search_space(
            imputers=self._get_single_imputer(),
            scalers=self._get_empty_categorical(),
            encoders=self._get_single_encoder(),
        )


class ExtraTreesBayesianSearchSpace(TreesBayesianSearchSpaceBase):
    def _create_model(self):
        return ExtraTreesClassifier(
            n_estimators=500,
            bootstrap=True,
            random_state=self._random_state
        )


class RandomForestBayesianSearchSpace(TreesBayesianSearchSpaceBase):
    def _create_model(self):
        return RandomForestClassifier(
            n_estimators=500,
            random_state=self._random_state
        )


class XGBoostBayesianSearchSpace(ModelBayesianSearchSpaceBase):
    from skopt.space import Real, Integer, Categorical
    # temp = Real(100, 2000, prior='uniform')
    # temp = Real(100, 2000, prior='log-uniform')
    # import matplotlib.pyplot as plt
    # plt.hist(temp.rvs(n_samples=10000), density=True, bins=30)
    # plt.axvline(x=0, color='red')
    # plt.axvline(x=100, color='black')

    def __init__(self,
                 # hyper-params search space
                 max_depth: Union[Integer, None] = DefaultInteger(),
                 learning_rate: Union[Real, None] = DefaultReal(),
                 n_estimators: Union[Integer, None] = DefaultInteger(),
                 min_child_weight: Union[Integer, None] = DefaultInteger(),
                 subsample: Union[Real, None] = DefaultReal(),
                 colsample_bytree: Union[Real, None] = DefaultReal(),
                 colsample_bylevel: Union[Real, None] = DefaultReal(),
                 reg_alpha: Union[Real, None] = DefaultReal(),
                 reg_lambda: Union[Real, None] = DefaultReal(),
                 # model options
                 eval_metric='logloss',
                 # search space options
                 iterations: int = 50,
                 include_default_model: bool = True,
                 # transformation search space options
                 imputers: Union[Categorical, None] = DefaultCategorical(),
                 scalers: Union[Categorical, None] = DefaultCategorical(),
                 encoders: Union[Categorical, None] = DefaultCategorical(),
                 random_state: int = None):

        if isinstance(imputers, DefaultValue):
            imputers = self._get_default_imputers()
        if isinstance(scalers, DefaultValue):
            scalers = self._get_empty_categorical()  # do not scale for XGBoost
        if isinstance(encoders, DefaultValue):
            encoders = self._get_default_encoders()

        super().__init__(iterations=iterations,
                         include_default_model=include_default_model,
                         imputers=imputers,
                         scalers=scalers,
                         encoders=encoders,
                         random_state=random_state)

        self._model_parameters = dict(
            max_depth=Integer(low=1, high=10, prior='log-uniform') if isinstance(max_depth, DefaultValue) else max_depth,  # noqa
            learning_rate=Real(0.01, 0.3, prior='log-uniform') if isinstance(learning_rate, DefaultValue) else learning_rate,  # noqa
            n_estimators=Integer(100, 2000, prior='uniform') if isinstance(n_estimators, DefaultValue) else n_estimators,  # noqa
            min_child_weight=Integer(1, 50, prior='log-uniform') if isinstance(min_child_weight, DefaultValue) else min_child_weight,  # noqa
            subsample=Real(0.5, 1, prior='uniform') if isinstance(subsample, DefaultValue) else subsample,
            colsample_bytree=Real(0.5, 1, prior='uniform') if isinstance(colsample_bytree, DefaultValue) else colsample_bytree,  # noqa
            colsample_bylevel=Real(0.5, 1, prior='uniform') if isinstance(colsample_bylevel, DefaultValue) else colsample_bylevel,  # noqa
            reg_alpha=Real(0.0001, 1, prior='log-uniform') if isinstance(reg_alpha, DefaultValue) else reg_alpha,  # noqa
            reg_lambda=Real(1, 4, prior='log-uniform') if isinstance(reg_lambda, DefaultValue) else reg_lambda,  # noqa
        )

        self._eval_metric = eval_metric

    def _create_model(self):
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=500,
            eval_metric=self._eval_metric,
            use_label_encoder=False,
            random_state=self._random_state,
        )

    def _default_model_transformer_search_space(self) -> dict:
        return self._build_transformer_search_space(
            imputers=self._get_single_imputer(),
            scalers=self._get_empty_categorical(),
            encoders=self._get_single_encoder(),
        )


class BayesianSearchSpace(BayesianSearchSpaceBase):

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
