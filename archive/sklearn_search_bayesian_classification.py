"""
This module contains classes that define search spaces compatible with BayesSearchCV for classification 
models.
"""
from typing import Union

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from helpsk.sklearn_search_bayesian_base import *


class LogisticBayesianSearchSpace(ModelBayesianSearchSpaceBase):
    """Defines the BayesSearchCV search space for Logistic Regression."""
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
                 pca: Union[Categorical, None] = DefaultCategorical(),
                 encoders: Union[Categorical, None] = DefaultCategorical(),
                 random_state: int = None):
        """Defines the model/transformation parameters that can be tuned."""

        if isinstance(imputers, DefaultValue):
            imputers = self._create_default_imputers()
        if isinstance(scalers, DefaultValue):
            scalers = self._create_default_scalers()
        if isinstance(pca, DefaultValue):
            pca = self._create_default_pca()
        if isinstance(encoders, DefaultValue):
            encoders = self._create_default_encoders()

        super().__init__(iterations=iterations,
                         include_default_model=include_default_model,
                         imputers=imputers,
                         scalers=scalers,
                         pca=pca,
                         encoders=encoders,
                         random_state=random_state)
        self._model_parameters = dict(
            C=Real(low=1e-6, high=100, prior='log-uniform') if isinstance(C, DefaultValue) else C,
        )

        self._solver = solver
        self._max_iter = max_iter

    def _create_model(self):
        """Defines the model that will be trained/tuned."""
        return LogisticRegression(
            solver=self._solver,
            max_iter=self._max_iter,
            random_state=self._random_state
        )

    def _default_model_transformer_search_space(self) -> dict:
        """Defines the default transformation search space for a model with default parameters."""
        return self._build_transformer_search_space(
            imputers=self._create_single_imputer(),
            scalers=self._create_single_scaler(),
            pca=self._create_empty_categorical(),
            encoders=self._create_single_encoder(),
        )


class LinearSVCBayesianSearchSpace(ModelBayesianSearchSpaceBase):
    """Defines the BayesSearchCV search space for LinearSVC."""
    def __init__(self,
                 # hyper-params search space
                 C: Union[Real, None] = DefaultReal(),  # noqa
                 # search space options
                 iterations: int = 50,
                 include_default_model: bool = True,
                 # transformation search space options
                 imputers: Union[Categorical, None] = DefaultCategorical(),
                 scalers: Union[Categorical, None] = DefaultCategorical(),
                 pca: Union[Categorical, None] = DefaultCategorical(),
                 encoders: Union[Categorical, None] = DefaultCategorical(),
                 random_state: int = None):
        """Defines the model/transformation parameters that can be tuned."""

        if isinstance(imputers, DefaultValue):
            imputers = self._create_default_imputers()
        if isinstance(scalers, DefaultValue):
            scalers = self._create_default_scalers()
        if isinstance(pca, DefaultValue):
            pca = self._create_default_pca()
        if isinstance(encoders, DefaultValue):
            encoders = self._create_default_encoders()

        super().__init__(iterations=iterations,
                         include_default_model=include_default_model,
                         imputers=imputers,
                         scalers=scalers,
                         pca=pca,
                         encoders=encoders,
                         random_state=random_state)
        self._model_parameters = dict(
            C=Real(low=1e-6, high=100, prior='log-uniform') if isinstance(C, DefaultValue) else C,
        )

    def _create_model(self):
        """Defines the model that will be trained/tuned."""
        return LinearSVC(
            random_state=self._random_state
        )

    def _default_model_transformer_search_space(self) -> dict:
        """Defines the default transformation search space for a model with default parameters."""
        return self._build_transformer_search_space(
            imputers=self._create_single_imputer(),
            scalers=self._create_single_scaler(),
            pca=self._create_empty_categorical(),
            encoders=self._create_single_encoder(),
        )


class TreesBayesianSearchSpaceBase(ModelBayesianSearchSpaceBase, ABC):
    """Base class that defines the BayesSearchCV search space for Tree-based models."""
    def __init__(self,
                 # hyper-params search space
                 max_features: Union[Real, None] = DefaultReal(),
                 max_depth: Union[Integer, None] = DefaultInteger(),
                 n_estimators: Union[Integer, None] = DefaultInteger(),
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
                 pca: Union[Categorical, None] = DefaultCategorical(),
                 encoders: Union[Categorical, None] = DefaultCategorical(),
                 random_state: int = None):
        """Defines the model/transformation parameters that can be tuned."""

        if isinstance(imputers, DefaultValue):
            imputers = self._create_default_imputers()
        if isinstance(scalers, DefaultValue):
            scalers = self._create_empty_categorical()
        if isinstance(pca, DefaultValue):
            pca = self._create_default_pca()
        if isinstance(encoders, DefaultValue):
            encoders = self._create_default_encoders()

        super().__init__(iterations=iterations,
                         include_default_model=include_default_model,
                         imputers=imputers,
                         scalers=scalers,
                         pca=pca,
                         encoders=encoders,
                         random_state=random_state)

        self._model_parameters = dict(
            max_features=Real(low=0.01, high=0.95, prior='uniform') if isinstance(max_features, DefaultValue) else max_features,
            max_depth=Integer(low=1, high=100, prior='uniform') if isinstance(max_depth, DefaultValue) else max_depth,
            n_estimators=Integer(500, 2000, prior='uniform') if isinstance(n_estimators, DefaultValue) else n_estimators,
            min_samples_split=Integer(low=2, high=50, prior='uniform') if isinstance(min_samples_split, DefaultValue) else min_samples_split,
            min_samples_leaf=Integer(low=1, high=50, prior='uniform') if isinstance(min_samples_leaf, DefaultValue) else min_samples_leaf,
            max_samples=Real(low=0.5, high=1.0, prior='uniform') if isinstance(max_samples, DefaultValue) else max_samples,
            criterion=Categorical(['gini', 'entropy']) if isinstance(criterion, DefaultValue) else criterion,
        )

    def _default_model_transformer_search_space(self) -> dict:
        """Defines the default transformation search space for a model with default parameters."""
        return self._build_transformer_search_space(
            imputers=self._create_single_imputer(),
            scalers=self._create_empty_categorical(),
            pca=self._create_empty_categorical(),
            encoders=self._create_single_encoder(),
        )


class ExtraTreesBayesianSearchSpace(TreesBayesianSearchSpaceBase):
    """Defines the BayesSearchCV search space for ExtraTreesClassifier models."""
    def _create_model(self):
        """Defines the model that will be trained/tuned."""
        return ExtraTreesClassifier(
            n_estimators=500,
            bootstrap=True,
            random_state=self._random_state
        )


class RandomForestBayesianSearchSpace(TreesBayesianSearchSpaceBase):
    """Defines the BayesSearchCV search space for RandomForestClassifier models."""
    def _create_model(self):
        """Defines the model that will be trained/tuned."""
        return RandomForestClassifier(
            n_estimators=500,
            random_state=self._random_state
        )


class XGBoostBayesianSearchSpace(ModelBayesianSearchSpaceBase):
    """Defines the BayesSearchCV search space for XGBClassifier models."""
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
                 eval_metric: str = 'logloss',
                 # search space options
                 iterations: int = 50,
                 include_default_model: bool = True,
                 # transformation search space options
                 imputers: Union[Categorical, None] = DefaultCategorical(),
                 scalers: Union[Categorical, None] = DefaultCategorical(),
                 pca: Union[Categorical, None] = DefaultCategorical(),
                 encoders: Union[Categorical, None] = DefaultCategorical(),
                 random_state: int = None):
        """Defines the model/transformation parameters that can be tuned."""

        if isinstance(imputers, DefaultValue):
            imputers = self._create_default_imputers()
        if isinstance(scalers, DefaultValue):
            scalers = self._create_empty_categorical()  # do not scale for XGBoost
        if isinstance(pca, DefaultValue):
            pca = self._create_default_pca()
        if isinstance(encoders, DefaultValue):
            encoders = self._create_default_encoders()

        super().__init__(iterations=iterations,
                         include_default_model=include_default_model,
                         imputers=imputers,
                         scalers=scalers,
                         pca=pca,
                         encoders=encoders,
                         random_state=random_state)

        self._model_parameters = dict(
            max_depth=Integer(low=1, high=20, prior='log-uniform') if isinstance(max_depth, DefaultValue) else max_depth,  # noqa
            learning_rate=Real(0.01, 0.3, prior='log-uniform') if isinstance(learning_rate, DefaultValue) else learning_rate,  # noqa
            n_estimators=Integer(500, 2000, prior='uniform') if isinstance(n_estimators, DefaultValue) else n_estimators,  # noqa
            min_child_weight=Integer(1, 50, prior='log-uniform') if isinstance(min_child_weight, DefaultValue) else min_child_weight,  # noqa
            subsample=Real(0.5, 1, prior='uniform') if isinstance(subsample, DefaultValue) else subsample,
            colsample_bytree=Real(0.5, 1, prior='uniform') if isinstance(colsample_bytree, DefaultValue) else colsample_bytree,  # noqa
            colsample_bylevel=Real(0.5, 1, prior='uniform') if isinstance(colsample_bylevel, DefaultValue) else colsample_bylevel,  # noqa
            reg_alpha=Real(0.0001, 1, prior='log-uniform') if isinstance(reg_alpha, DefaultValue) else reg_alpha,  # noqa
            reg_lambda=Real(1, 4, prior='log-uniform') if isinstance(reg_lambda, DefaultValue) else reg_lambda,  # noqa
        )

        self._eval_metric = eval_metric

    def _create_model(self):
        """Defines the model that will be trained/tuned."""
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=500,
            eval_metric=self._eval_metric,
            use_label_encoder=False,
            random_state=self._random_state,
        )

    def _default_model_transformer_search_space(self) -> dict:
        """Defines the default transformation search space for a model with default parameters."""
        return self._build_transformer_search_space(
            imputers=self._create_single_imputer(),
            scalers=self._create_empty_categorical(),
            pca=self._create_empty_categorical(),
            encoders=self._create_single_encoder(),
        )
