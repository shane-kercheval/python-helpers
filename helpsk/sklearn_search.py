from enum import unique, Enum, auto

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder

import helpsk as hlp


@unique
class ClassifierSearchSpaceModels(Enum):
    LogisticRegression = auto()
    XGBoost = auto()


# need to make sure this works for a single model or multiple models
class ClassifierSearchSpace:
    def __init__(self,
                 # remove these and pass data and get column names directly?
                 data,
                 models=[  # noqa
                     ClassifierSearchSpaceModels.XGBoost,
                     ClassifierSearchSpaceModels.LogisticRegression,
                 ],
                 iterations=[50, 50]):  # noqa
        assert len(models) == len(iterations)

        self._numeric_column_names = hlp.pandas.get_numeric_columns(data)
        self._non_numeric_column_names = hlp.pandas.get_non_numeric_columns(data)
        self._models = models
        self._iterations = iterations

    def pipeline(self):
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

    @staticmethod
    def _build_transformer_search_space(imputer_strategies=['mean', 'median', 'most_frequent'],  # noqa
                                        scaler_min_max=True,
                                        scaler_standard=True,
                                        encoder_one_hot=True,
                                        encoder_ordinal=True):
        from skopt.space import Categorical

        if imputer_strategies:
            imputers = [SimpleImputer(strategy=x) if x else None for x in imputer_strategies]
        else:
            imputers = [None]

        scalers = [None]
        if scaler_min_max:
            scalers += [MinMaxScaler()]
        if scaler_standard:
            scalers += [StandardScaler()]

        assert encoder_one_hot or encoder_ordinal
        encoders = []
        if encoder_one_hot:
            encoders += [OneHotEncoder()]
        if encoder_ordinal:
            encoders += [hlp.sklearn_pipeline.CustomOrdinalEncoder()]

        return {
            'prep__numeric__imputer__transformer': Categorical(imputers),
            'prep__numeric__scaler__transformer': Categorical(scalers),
            'prep__non_numeric__encoder__transformer': Categorical(encoders),
        }
    
    @staticmethod
    def _search_space_logistic(solver='lbfgs',
                               max_iter=1000,
                               C=(1e-6, 1e+2),  # noqa
                               C_prior='log-uniform',  # noqa
                               imputer_strategies=['mean', 'median', 'most_frequent'],  # noqa
                               random_state=None):
        from skopt.space import Real, Categorical

        model = LogisticRegression(
            solver=solver,
            max_iter=max_iter,
            random_state=random_state
        )

        logistic_search_space = {
            'model': Categorical([model]),
            'model__C': Real(C[0], C[1], prior=C_prior),
        }
        # these steps correspond to the pipeline built in `build_classifier_search_pipeline()`
        logistic_search_space.update(ClassifierSearchSpace._build_transformer_search_space(
            imputer_strategies=imputer_strategies,
        ))

        return logistic_search_space

    @staticmethod
    def _search_space_xgboost(eval_metric='logloss',
                              use_label_encoder=False,
                              max_depth=(3, 10),
                              n_estimators=(50, 1000),
                              learning_rate=(0.01, 0.3),
                              colsample_bytree=(0.01, 1),
                              subsample=(0.1, 1),
                              imputer_strategies=['mean', 'median', 'most_frequent'],  # noqa
                              random_state=None):
        """
        #`XGBoostError: XGBoost Library (libxgboost.dylib) could not be loaded on Apple Silicon (ARM)`
        #https://github.com/dmlc/xgboost/issues/6909
        #```
        #pip install --upgrade --force-reinstall xgboost --no-binary xgboost -v
        #```
        """
        from skopt.space import Real, Categorical, Integer
        from xgboost import XGBClassifier

        model = XGBClassifier(
            eval_metric=eval_metric,
            use_label_encoder=use_label_encoder,
            random_state=random_state,
        )
        # https://towardsdatascience.com/xgboost-fine-tune-and-optimize-your-model-23d996fab663
        # max_depth: 3–10
        # n_estimators: 100 (lots of observations) to 1000 (few observations)
        # learning_rate: 0.01–0.3
        # colsample_bytree: 0.5–1
        # subsample: 0.6–1
        # Then, you can focus on optimizing max_depth and n_estimators.
        # You can then play along with the learning_rate, and increase it to speed up the model without
        # decreasing the performances. If it becomes faster without losing in performances, you can increase
        # the number of estimators to try to increase the performances.
        xgb_search_space = {
            'model': Categorical([model]),
            'model__max_depth': Integer(max_depth[0], max_depth[1]),
            'model__n_estimators':  Integer(n_estimators[0], n_estimators[1]),
            'model__learning_rate': Real(learning_rate[0], learning_rate[1]),
            'model__colsample_bytree': Real(colsample_bytree[0], colsample_bytree[1]),
            'model__subsample': Real(subsample[0], subsample[1]),
        }
        # these steps correspond to the pipeline built in `build_classifier_search_pipeline()`
        xgb_search_space.update(ClassifierSearchSpace._build_transformer_search_space(
            imputer_strategies=imputer_strategies,
            scaler_min_max=False,
            scaler_standard=False,
        ))

        return xgb_search_space

    def search_spaces(self):
        search_spaces = []
        for model_enum, num_iterations in zip(self._models, self._iterations):
            if model_enum == ClassifierSearchSpaceModels.LogisticRegression:
                space = ClassifierSearchSpace._search_space_logistic()
            elif model_enum == ClassifierSearchSpaceModels.XGBoost:
                space = ClassifierSearchSpace._search_space_xgboost()
            else:
                assert False

            search_spaces = search_spaces + [(space, num_iterations)]
        
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
