from abc import abstractmethod, ABC
from typing import List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.svm import LinearSVC

import helpsk as hlp


class BayesianSearchSpaceBase(ABC):
    def __init__(self, random_state: int = None):
        self._random_state = random_state
        self._numeric_column_names = None
        self._non_numeric_column_names = None

    def pipeline(self):
        if self._numeric_column_names is None and self._non_numeric_column_names is None:
            raise RuntimeError('You must call `prime()` before using this function()')

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
            ('numeric', numeric_pipeline, self._numeric_column_names),
            ('non_numeric', non_numeric_pipeline, self._non_numeric_column_names)
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


class ModelBayesianSearchSpaceBase(BayesianSearchSpaceBase, ABC):
    def __init__(self,
                 iterations: int = 50,
                 include_default_model: bool = True,
                 imputer_strategies=['mean', 'median', 'most_frequent'],  # noqa
                 scaler_min_max: bool = True,
                 scaler_standard: bool = True,
                 scaler_none: bool = True,
                 encoder_one_hot: bool = True,
                 encoder_ordinal: bool = True,
                 random_state: int = None):
        """
        Args:
           imputer_strategies:

           scaler_min_max:

           scaler_standard:

           scaler_none:
                If True, include `None` as a hyper-param value. If False, only do min/max or standard (either
                scaler_min_max or scaler_standard has to be True). Setting this to True does *not* imply no
                scaling at all, only if we want to tune not scaling.

                For example, for models we always want to scale (e.g. Logistic Regression, LinearSVC), then
                setting `scaler_none=False` (and either/both `scaler_min_max=True` and/or
                `scaler_standard=True` will accomplish this.
           encoder_one_hot:

           encoder_ordinal:
        """

        self._iterations = iterations
        self._include_default_model = include_default_model
        self._imputer_strategies = imputer_strategies
        self._scaler_min_max = scaler_min_max
        self._scaler_standard = scaler_standard
        self._scaler_none = scaler_none
        self._encoder_one_hot = encoder_one_hot
        self._encoder_ordinal = encoder_ordinal
        super().__init__(random_state)

    @staticmethod
    def _build_transformer_search_space(imputer_strategies: List[str],
                                        scaler_min_max: bool,
                                        scaler_standard: bool,
                                        scaler_none: bool,
                                        encoder_one_hot: bool,
                                        encoder_ordinal: bool) -> dict:
        from skopt.space import Categorical

        assert isinstance(imputer_strategies, list)

        if imputer_strategies:
            imputers = [SimpleImputer(strategy=x) if x else None for x in imputer_strategies]
        else:
            imputers = [None]

        assert scaler_none or scaler_min_max or scaler_standard
        scalers = [None]
        if scaler_min_max:
            scalers += [MinMaxScaler()]
        if scaler_standard:
            scalers += [StandardScaler()]

        if not scaler_none:
            scalers = [x for x in scalers if x is not None]

        assert encoder_one_hot or encoder_ordinal
        encoders = []
        if encoder_one_hot:
            encoders += [OneHotEncoder(handle_unknown='ignore')]
        if encoder_ordinal:
            encoders += [hlp.sklearn_pipeline.CustomOrdinalEncoder()]

        return {
            'prep__numeric__imputer__transformer': Categorical(imputers),
            'prep__numeric__scaler__transformer': Categorical(scalers),
            'prep__non_numeric__encoder__transformer': Categorical(encoders),
        }

    def prime(self, data: pd.DataFrame):
        self._numeric_column_names = hlp.pandas.get_numeric_columns(data)
        self._non_numeric_column_names = hlp.pandas.get_non_numeric_columns(data)

    @abstractmethod
    def _create_model(self):
        """This method returns a model object with whatever default values should be set."""

    @abstractmethod
    def _model_search_space(self) -> dict:
        """This method returns a dictionary of model hyper-params to tune. The key should be the name of the
        parameter, prefixed with 'model__' and the value is the skopt.space to search."""

    @abstractmethod
    def _transformer_search_space(self) -> dict:
        """This method returns a dictionary of the transformations to tune. This can be accomplished by
        calling the `_build_transformer_search_space` function."""

    @abstractmethod
    def _default_model_transformer_search_space(self) -> dict:
        """This method returns a dictionary of the default transformations to apply if including the default
        model in the search spaces (i.e. if `_include_default_model` is True). This can be accomplished by
        calling the `_build_transformer_search_space` function."""

    def search_spaces(self) -> List[tuple]:
        """Returns a list of search spaces (e.g. 2 items if `include_default_model` is True; one for the
        model with default params, and one for searching across all params.)
        Each space is a tuple with a dictionary (hyper-param search space) as the first item and an integer
        (number of iterations) as second item."""
        from skopt.space import Categorical

        model_search_space = {'model': self._create_model()}
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
                 # model search space options
                 C: tuple = (1e-6, 100),  # noqa
                 # model default value options
                 solver: str = 'lbfgs',
                 max_iter: int = 1000,
                 # search space options
                 iterations: int = 50,
                 include_default_model: bool = True,
                 # transformation search space options
                 imputer_strategies=['mean', 'median', 'most_frequent'],  # noqa
                 scaler_min_max: bool = True,
                 scaler_standard: bool = True,
                 scaler_none: bool = False,  # we always want to scale
                 encoder_one_hot: bool = True,
                 encoder_ordinal: bool = True,
                 random_state: int = None):
        super().__init__(iterations=iterations,
                         include_default_model=include_default_model,
                         imputer_strategies=imputer_strategies,
                         scaler_min_max=scaler_min_max,
                         scaler_standard=scaler_standard,
                         scaler_none=scaler_none,
                         encoder_one_hot=encoder_one_hot,
                         encoder_ordinal=encoder_ordinal,
                         random_state=random_state)
        self._C = C
        self._solver = solver
        self._max_iter = max_iter

    def _create_model(self):
        return LogisticRegression(
            solver=self._solver,
            max_iter=self._max_iter,
            random_state=self._random_state
        )

    def _model_search_space(self) -> dict:
        from skopt.space import Real
        return {
            'model__C': Real(self._C[0], self._C[1], prior='log-uniform'),
        }

    def _transformer_search_space(self) -> dict:
        return self._build_transformer_search_space(
            imputer_strategies=self._imputer_strategies,
            scaler_min_max=self._scaler_min_max,
            scaler_standard=self._scaler_standard,
            scaler_none=self._scaler_none,
            encoder_one_hot=self._encoder_one_hot,
            encoder_ordinal=self._encoder_ordinal,
        )

    def _default_model_transformer_search_space(self) -> dict:
        return self._build_transformer_search_space(
            imputer_strategies=['mean'],
            scaler_min_max=False,
            scaler_standard=True,
            scaler_none=False,
            encoder_one_hot=True,
            encoder_ordinal=False,
        )


class LinearSVCBayesianSearchSpace(ModelBayesianSearchSpaceBase):
    def __init__(self,
                 # model search space options
                 C: tuple = (1e-6, 100),  # noqa
                 # search space options
                 iterations: int = 50,
                 include_default_model: bool = True,
                 # transformation search space options
                 imputer_strategies=['mean', 'median', 'most_frequent'],  # noqa
                 scaler_min_max: bool = True,
                 scaler_standard: bool = True,
                 scaler_none: bool = False,  # we always want to scale
                 encoder_one_hot: bool = True,
                 encoder_ordinal: bool = True,
                 random_state: int = None):
        super().__init__(iterations=iterations,
                         include_default_model=include_default_model,
                         imputer_strategies=imputer_strategies,
                         scaler_min_max=scaler_min_max,
                         scaler_standard=scaler_standard,
                         scaler_none=scaler_none,
                         encoder_one_hot=encoder_one_hot,
                         encoder_ordinal=encoder_ordinal,
                         random_state=random_state)
        self._C = C

    def _create_model(self):
        return LinearSVC(
            random_state=self._random_state
        )

    def _model_search_space(self) -> dict:
        from skopt.space import Real
        return {
            'model__C': Real(self._C[0], self._C[1], prior='log-uniform'),
        }

    def _transformer_search_space(self) -> dict:
        return self._build_transformer_search_space(
            imputer_strategies=self._imputer_strategies,
            scaler_min_max=self._scaler_min_max,
            scaler_standard=self._scaler_standard,
            scaler_none=self._scaler_none,
            encoder_one_hot=self._encoder_one_hot,
            encoder_ordinal=self._encoder_ordinal,
        )

    def _default_model_transformer_search_space(self) -> dict:
        return self._build_transformer_search_space(
            imputer_strategies=['mean'],
            scaler_min_max=False,
            scaler_standard=True,
            scaler_none=False,
            encoder_one_hot=True,
            encoder_ordinal=False,
        )


class TreesBayesianSearchSpaceBase(ModelBayesianSearchSpaceBase, ABC):
    def __init__(self,
                 # model search space options
                 max_features=(0.01, 0.95),
                 max_depth=(1, 100),
                 min_samples_split=(2, 50),
                 min_samples_leaf=(1, 50),
                 max_samples=(0.5, 1.0),
                 criterion=['gini', 'entropy'],  # noqa
                 # search space options
                 iterations: int = 50,
                 include_default_model: bool = True,
                 # transformation search space options
                 imputer_strategies=['mean', 'median', 'most_frequent'],  # noqa
                 scaler_min_max: bool = False,
                 scaler_standard: bool = False,
                 scaler_none: bool = True,
                 encoder_one_hot: bool = True,
                 encoder_ordinal: bool = True,
                 random_state: int = None):
        super().__init__(iterations=iterations,
                         include_default_model=include_default_model,
                         imputer_strategies=imputer_strategies,
                         scaler_min_max=scaler_min_max,
                         scaler_standard=scaler_standard,
                         scaler_none=scaler_none,
                         encoder_one_hot=encoder_one_hot,
                         encoder_ordinal=encoder_ordinal,
                         random_state=random_state)
        self._max_features = max_features
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf
        self._max_samples = max_samples
        self._criterion = criterion

    def _model_search_space(self) -> dict:
        from skopt.space import Real, Integer, Categorical
        return {
            'model__max_features': Real(self._max_features[0], self._max_features[1], prior='uniform'),
            'model__max_depth': Integer(self._max_depth[0], self._max_depth[1], prior='uniform'),
            'model__min_samples_split': Integer(self._min_samples_split[0], self._min_samples_split[1],
                                                prior='uniform'),
            'model__min_samples_leaf': Integer(self._min_samples_leaf[0], self._min_samples_leaf[1],
                                               prior='uniform'),
            'model__max_samples': Real(self._max_samples[0], self._max_samples[1], prior='uniform'),
            'model__criterion': Categorical(self._criterion),
        }

    def _transformer_search_space(self) -> dict:
        return self._build_transformer_search_space(
            imputer_strategies=self._imputer_strategies,
            scaler_min_max=self._scaler_min_max,
            scaler_standard=self._scaler_standard,
            scaler_none=self._scaler_none,
            encoder_one_hot=self._encoder_one_hot,
            encoder_ordinal=self._encoder_ordinal,
        )

    def _default_model_transformer_search_space(self) -> dict:
        return self._build_transformer_search_space(
            imputer_strategies=['mean'],
            scaler_min_max=False,
            scaler_standard=False,
            scaler_none=True,
            encoder_one_hot=True,
            encoder_ordinal=False,
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
    def __init__(self,
                 # model search space options
                 max_depth=(1, 50),
                 learning_rate=(0.0001, 0.5),
                 n_estimators=(100, 2000),
                 min_child_weight=(1, 50),
                 subsample=(0.5, 1),
                 colsample_bytree=(0.5, 1),
                 colsample_bylevel=(0.5, 1),
                 reg_alpha=(0.0001, 1),
                 reg_lambda=(1, 4),
                 # model options
                 eval_metric='logloss',
                 # search space options
                 iterations: int = 50,
                 include_default_model: bool = True,
                 # transformation search space options
                 imputer_strategies=['mean', 'median', 'most_frequent'],  # noqa
                 scaler_min_max: bool = False,
                 scaler_standard: bool = False,
                 scaler_none: bool = True,
                 encoder_one_hot: bool = True,
                 encoder_ordinal: bool = True,
                 random_state: int = None):
        super().__init__(iterations=iterations,
                         include_default_model=include_default_model,
                         imputer_strategies=imputer_strategies,
                         scaler_min_max=scaler_min_max,
                         scaler_standard=scaler_standard,
                         scaler_none=scaler_none,
                         encoder_one_hot=encoder_one_hot,
                         encoder_ordinal=encoder_ordinal,
                         random_state=random_state)
        self._max_depth = max_depth
        self._learning_rate = learning_rate
        self._n_estimators = n_estimators
        self._min_child_weight = min_child_weight
        self._subsample = subsample
        self._colsample_bytree = colsample_bytree
        self._colsample_bylevel = colsample_bylevel
        self._reg_alpha = reg_alpha
        self._reg_lambda = reg_lambda
        self._eval_metric = eval_metric

    def _create_model(self):
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=500,
            eval_metric=self._eval_metric,
            use_label_encoder=False,
            random_state=self._random_state,
        )

    def _model_search_space(self) -> dict:
        from skopt.space import Real, Integer
        return {
            'model__max_depth': Integer(self._max_depth[0], self._max_depth[1], prior='log-uniform'),
            'model__learning_rate': Real(self._learning_rate[0], self._learning_rate[1], prior='uniform'),
            'model__n_estimators': Integer(self._n_estimators[0], self._n_estimators[1], prior='log-uniform'),
            'model__min_child_weight': Integer(self._min_child_weight[0], self._min_child_weight[1],
                                               prior='log-uniform'),
            'model__subsample': Real(self._subsample[0], self._subsample[1], prior='uniform'),
            'model__colsample_bytree': Real(self._colsample_bytree[0], self._colsample_bytree[1],
                                            prior='uniform'),
            'model__colsample_bylevel': Real(self._colsample_bylevel[0], self._colsample_bylevel[1],
                                             prior='uniform'),
            'model__reg_alpha': Real(self._reg_alpha[0], self._reg_alpha[1], prior='log-uniform'),
            'model__reg_lambda': Real(self._reg_lambda[0], self._reg_lambda[1], prior='log-uniform'),
        }

    def _transformer_search_space(self) -> dict:
        return self._build_transformer_search_space(
            imputer_strategies=self._imputer_strategies,
            scaler_min_max=self._scaler_min_max,
            scaler_standard=self._scaler_standard,
            scaler_none=self._scaler_none,
            encoder_one_hot=self._encoder_one_hot,
            encoder_ordinal=self._encoder_ordinal,
        )

    def _default_model_transformer_search_space(self) -> dict:
        return self._build_transformer_search_space(
            imputer_strategies=['mean'],
            scaler_min_max=False,
            scaler_standard=False,
            scaler_none=True,
            encoder_one_hot=True,
            encoder_ordinal=False,
        )


class ClassificationBayesianSearchSpace(BayesianSearchSpaceBase):

    def __init__(self,
                 data: pd.DataFrame,
                 model_search_spaces: List[ModelBayesianSearchSpaceBase] = None,
                 random_state: int = None):
        super().__init__(random_state=random_state)
        self._numeric_column_names = hlp.pandas.get_numeric_columns(data)
        self._non_numeric_column_names = hlp.pandas.get_non_numeric_columns(data)
        if model_search_spaces:
            self._model_search_spaces = model_search_spaces
        else:
            self._model_search_spaces = [
                LogisticBayesianSearchSpace(),
                LinearSVCBayesianSearchSpace(),
                ExtraTreesBayesianSearchSpace(),
                RandomForestBayesianSearchSpace(),
                XGBoostBayesianSearchSpace(),
            ]

    def search_spaces(self) -> List[tuple]:
        all_spaces = []
        for space in self._model_search_spaces:  # each `space.search_spaces()` is a list of tuples
            all_spaces = all_spaces + space.search_spaces()

        return all_spaces
