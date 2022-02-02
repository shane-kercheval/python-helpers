import unittest

import numpy as np

import helpsk as hlp
from helpsk.exceptions import HelpskAssertionError
from helpsk.sklearn_eval import MLExperimentResults, TwoClassEvaluator, RegressionEvaluator
from helpsk.sklearn_pipeline import CustomOrdinalEncoder
from helpsk.sklearn_search import ClassifierSearchSpace, ClassifierSearchSpaceModels
from sklearn.model_selection import train_test_split

from helpsk.utility import redirect_stdout_to_file
from tests.helpers import get_data_credit, get_test_path, check_plot, helper_test_dataframe, get_data_housing
from sklearn.preprocessing import label_binarize, MinMaxScaler, StandardScaler, OneHotEncoder


# noinspection PyMethodMayBeStatic
class TestSklearnEval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ####
        # set up grid-search on classification model for credit data
        ####

        credit_data = get_data_credit()
        credit_data.loc[0:46, ['duration']] = np.nan
        credit_data.loc[25:75, ['checking_status']] = np.nan
        credit_data.loc[10:54, ['credit_amount']] = 0
        y_full = credit_data['target']
        X_full = credit_data.drop(columns='target')  # noqa
        y_full = label_binarize(y_full, classes=['good', 'bad']).flatten()
        X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=42)  # noqa
        del y_full, X_full
        cls.search_space = ClassifierSearchSpace(data=X_train)

    def test_ClassifierSearchSpace(self):

        transformer_search_space = ClassifierSearchSpace._build_transformer_search_space()
        self.assertEqual(list(transformer_search_space.keys()),
                         ['prep__numeric__imputer__transformer',
                          'prep__numeric__scaler__transformer',
                          'prep__non_numeric__encoder__transformer'])

        categorical = transformer_search_space['prep__numeric__imputer__transformer']
        self.assertEqual(len(categorical.categories), 3)
        self.assertEqual(categorical.categories[0].strategy, 'mean')
        self.assertEqual(categorical.categories[1].strategy, 'median')
        self.assertEqual(categorical.categories[2].strategy, 'most_frequent')
        del categorical

        categorical = transformer_search_space['prep__numeric__scaler__transformer']
        self.assertEqual(len(categorical.categories), 3)
        self.assertIsNone(categorical.categories[0])
        self.assertIsInstance(categorical.categories[1], MinMaxScaler)
        self.assertIsInstance(categorical.categories[2], StandardScaler)
        del categorical

        categorical = transformer_search_space['prep__non_numeric__encoder__transformer']
        self.assertEqual(len(categorical.categories), 2)
        self.assertIsInstance(categorical.categories[0], OneHotEncoder)
        self.assertIsInstance(categorical.categories[1], CustomOrdinalEncoder)
        del categorical

        transformer_search_space = ClassifierSearchSpace._build_transformer_search_space(
            imputer_strategies=['most_frequent'],
            scaler_min_max=False,
            scaler_standard=False,
            encoder_one_hot=True,
            encoder_ordinal=False
        )
        self.assertEqual(list(transformer_search_space.keys()),
                         ['prep__numeric__imputer__transformer',
                          'prep__numeric__scaler__transformer',
                          'prep__non_numeric__encoder__transformer'])

        categorical = transformer_search_space['prep__numeric__imputer__transformer']
        self.assertEqual(len(categorical.categories), 1)
        self.assertEqual(categorical.categories[0].strategy, 'most_frequent')
        del categorical

        categorical = transformer_search_space['prep__numeric__scaler__transformer']
        self.assertEqual(len(categorical.categories), 1)
        self.assertIsNone(categorical.categories[0])
        del categorical

        categorical = transformer_search_space['prep__non_numeric__encoder__transformer']
        self.assertEqual(len(categorical.categories), 1)
        self.assertIsInstance(categorical.categories[0], OneHotEncoder)
        del categorical

        # default space for logistic regression
        logistic_space = ClassifierSearchSpace._search_space_logistic()
        self.assertEqual(list(logistic_space.keys()),
                         ['model',
                          'model__C',
                          'prep__numeric__imputer__transformer',
                          'prep__numeric__scaler__transformer',
                          'prep__non_numeric__encoder__transformer'])
        with redirect_stdout_to_file(get_test_path() + '/test_files/sklearn_search/logistic_space_default.txt'):
            print(logistic_space)
        del logistic_space

        # search space for logistic regression with modified params
        logistic_space = ClassifierSearchSpace._search_space_logistic(
            solver='sag',
            max_iter=999,
            C=(1e-5, 1e+3),  # noqa
            C_prior='uniform',  # noqa
            imputer_strategies=['most_frequent'],  # noqa
            random_state=42
        )
        self.assertEqual(list(logistic_space.keys()),
                         ['model',
                          'model__C',
                          'prep__numeric__imputer__transformer',
                          'prep__numeric__scaler__transformer',
                          'prep__non_numeric__encoder__transformer'])
        with redirect_stdout_to_file(get_test_path() + '/test_files/sklearn_search/logistic_space_modified.txt'):
            print(logistic_space)
        del logistic_space

        # default space for xgboost
        xgboost_space = ClassifierSearchSpace._search_space_xgboost()
        self.assertEqual(list(xgboost_space.keys()),
                         ['model',
                          'model__max_depth',
                          'model__n_estimators',
                          'model__learning_rate',
                          'model__colsample_bytree',
                          'model__subsample',
                          'prep__numeric__imputer__transformer',
                          'prep__numeric__scaler__transformer',
                          'prep__non_numeric__encoder__transformer'])
        with redirect_stdout_to_file(get_test_path() + '/test_files/sklearn_search/xgboost_space_default.txt'):
            print(xgboost_space)
        del xgboost_space

        # search space for xgboost with modified params
        xgboost_space = ClassifierSearchSpace._search_space_xgboost(
            eval_metric='logloss',
            use_label_encoder=False,
            max_depth=(2, 30),
            n_estimators=(10, 10000),
            learning_rate=(0.01111, 0.3333),
            colsample_bytree=(0.01234, 1123),
            subsample=(0.1111, 0.999),
            imputer_strategies=['median'],
            random_state=42
        )
        self.assertEqual(list(xgboost_space.keys()),
                         ['model',
                          'model__max_depth',
                          'model__n_estimators',
                          'model__learning_rate',
                          'model__colsample_bytree',
                          'model__subsample',
                          'prep__numeric__imputer__transformer',
                          'prep__numeric__scaler__transformer',
                          'prep__non_numeric__encoder__transformer'])
        with redirect_stdout_to_file(get_test_path() + '/test_files/sklearn_search/xgboost_space_modified.txt'):
            print(xgboost_space)
        del xgboost_space

        pipeline = self.search_space.pipeline()
        with redirect_stdout_to_file(get_test_path() + '/test_files/sklearn_search/pipeline_default.txt'):
            print(pipeline)
        del pipeline

        search_spaces = self.search_space.search_spaces()
        with redirect_stdout_to_file(get_test_path() + '/test_files/sklearn_search/search_spaces_default.txt'):
            print(search_spaces)
        del search_spaces



        self.search_space.search_spaces()

        self.search_space.param_name_mappings()
