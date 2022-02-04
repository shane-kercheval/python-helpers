import os
import unittest
import warnings  # noqa

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score, SCORERS, roc_auc_score, fbeta_score, cohen_kappa_score, \
    confusion_matrix, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize, StandardScaler, OneHotEncoder

import helpsk as hlp
from helpsk.exceptions import HelpskAssertionError
from helpsk.pandas import print_dataframe
from helpsk.sklearn_eval import MLExperimentResults, TwoClassEvaluator, RegressionEvaluator
from helpsk.sklearn_pipeline import CustomOrdinalEncoder
from helpsk.utility import redirect_stdout_to_file
from tests.helpers import get_data_credit, get_test_path, check_plot, helper_test_dataframe, get_data_housing, clean_formatted_dataframe


def warn(*args, **kwargs):  # noqa
    pass
warnings.warn = warn  # noqa


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
        numeric_columns = hlp.pandas.get_numeric_columns(X_train)
        non_numeric_columns = hlp.pandas.get_non_numeric_columns(X_train)
        numeric_pipeline = Pipeline([
            ('imputing', SimpleImputer(strategy='mean')),
            ('scaling', StandardScaler()),
        ])
        non_numeric_pipeline = Pipeline([
            ('encoder_chooser', hlp.sklearn_pipeline.TransformerChooser()),
        ])
        transformations_pipeline = ColumnTransformer([
            ('numeric_pipeline', numeric_pipeline, numeric_columns),
            ('non_numeric_pipeline', non_numeric_pipeline, non_numeric_columns)
        ])
        random_forest_model = RandomForestClassifier(random_state=42)
        full_pipeline = Pipeline([
            ('preparation', transformations_pipeline),
            ('model', random_forest_model)
        ])
        param_grad = [
            {
                'preparation__non_numeric_pipeline__encoder_chooser__transformer': [OneHotEncoder(),
                                                                                    CustomOrdinalEncoder()],
                'model__min_samples_split': [2],  # test zero-variance params
                'model__max_features': [100, 'auto'],
                'model__n_estimators': [10, 50],
            },
        ]
        # https://github.com/scikit-learn/scikit-learn/blob/2beed55847ee70d363bdbfe14ee4401438fba057/sklearn_eval/metrics/_scorer.py#L702
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html
        # https://stackoverflow.com/questions/60615281/different-result-roc-auc-score-and-plot-roc-curve
        scores = {
            # https://github.com/scikit-learn/scikit-learn/blob/2beed5584/sklearn_eval/metrics/_scorer.py#L537
            'ROC/AUC': SCORERS['roc_auc'],
            'F1': make_scorer(f1_score, greater_is_better=True),
            'Pos. Pred. Val': make_scorer(precision_score, greater_is_better=True),
            'True Pos. Rate': make_scorer(recall_score, greater_is_better=True),
        }
        grid_search = GridSearchCV(full_pipeline,
                                   param_grid=param_grad,
                                   cv=RepeatedKFold(n_splits=3, n_repeats=1, random_state=42),
                                   scoring=scores,
                                   refit=False,
                                   return_train_score=True)
        grid_search.fit(X_train, y_train)
        cls.credit_data__grid_search = grid_search

        param_grad = [
            {
                'preparation__non_numeric_pipeline__encoder_chooser__transformer': [OneHotEncoder(),
                                                                                    CustomOrdinalEncoder()],
                'model__max_features': [100, 'auto'],
                'model__n_estimators': [10, 50],
            },
        ]
        grid_search = GridSearchCV(full_pipeline,
                                   param_grid=param_grad,
                                   cv=RepeatedKFold(n_splits=3, n_repeats=1, random_state=42),
                                   scoring='roc_auc',
                                   refit=True,
                                   return_train_score=False)
        grid_search.fit(X_train, y_train)
        cls.credit_data__grid_search__roc_auc = grid_search

        best_model = grid_search.best_estimator_
        predicted_scores = best_model.predict_proba(X_test)[:, 1]
        cls.credit_data__y_test = y_test
        cls.credit_data__y_scores = predicted_scores

        ####
        # set up grid-search on regression model for housing data
        ####
        housing_data = get_data_housing()
        housing_data.loc[0:46, ['median_income']] = np.nan
        housing_data.loc[25:75, ['housing_median_age']] = np.nan
        y_full = housing_data['target']
        X_full = housing_data.drop(columns='target')  # noqa
        X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=42)  # noqa
        del y_full, X_full
        numeric_pipeline = Pipeline([
            ('imputing', SimpleImputer(strategy='mean')),
            ('scaling', StandardScaler()),
        ])
        random_forest_model = RandomForestRegressor(random_state=42)
        full_pipeline = Pipeline([
            ('preparation', numeric_pipeline),
            ('model', random_forest_model)
        ])
        param_grad = [
            {
                'model__max_features': [2, 'auto'],
                'model__n_estimators': [10, 50]
            },
        ]
        scores = {
            'RMSE': make_scorer(mean_squared_error, greater_is_better=False, squared=False),
            'MAE': make_scorer(mean_absolute_error, greater_is_better=False),
        }
        grid_search = GridSearchCV(full_pipeline,
                                   param_grid=param_grad,
                                   cv=RepeatedKFold(n_splits=3, n_repeats=1, random_state=42),
                                   scoring=scores,
                                   refit='RMSE',
                                   return_train_score=True)
        grid_search.fit(X_train, y_train)
        cls.housing_data__grid_search = grid_search
        best_model = grid_search.best_estimator_
        predicted_values = best_model.predict(X_test)
        cls.housing_data__y_test = y_test
        cls.housing_data__y_predictions = predicted_values

    def test_MLExperimentResults_invalid_param_name_mapping(self):
        # the keys passed to parameter_name_mappings should match the parameters founds
        self.assertRaises(HelpskAssertionError,
                          lambda: MLExperimentResults.
                          from_sklearn_search_cv(searcher=self.credit_data__grid_search,
                                                 higher_score_is_better=True,
                                                 description="test description",
                                                 parameter_name_mappings={'this_should_fail': 'value'}))

    def test_MLExperimentResults_gridsearch_classification(self):
        new_param_column_names = {'model__max_features': 'max_features',
                                  'model__n_estimators': 'n_estimators',
                                  'model__min_samples_split': 'min_samples_split',
                                  'preparation__non_numeric_pipeline__encoder_chooser__transformer':
                                      'encoder'}

        # test grid search object that has multiple scores (classification)
        # passing in parameter mappings
        grid_search_credit = self.credit_data__grid_search

        parser = MLExperimentResults.from_sklearn_search_cv(searcher=grid_search_credit,
                                                            higher_score_is_better=True,
                                                            description="test description",
                                                            parameter_name_mappings=new_param_column_names)

        yaml_file = get_test_path() + '/test_files/sklearn_eval/credit_data__grid_search.yaml'
        os.remove(yaml_file)
        parser.to_yaml_file(yaml_file)
        parser_from_dict = MLExperimentResults(parser._dict)
        parser_from_yaml = MLExperimentResults.from_yaml_file(yaml_file)

        self.assertEqual(str(parser._dict), str(parser_from_dict._dict))
        self.assertEqual(str(parser._dict), str(parser_from_yaml._dict))

        self.assertEqual(parser.description, "test description")
        self.assertEqual(parser.description, parser_from_dict.description)
        self.assertEqual(parser.description, parser_from_yaml.description)
        self.assertEqual(parser.higher_score_is_better, True)
        self.assertEqual(parser.higher_score_is_better, parser_from_dict.higher_score_is_better)
        self.assertEqual(parser.higher_score_is_better, parser_from_yaml.higher_score_is_better)
        self.assertEqual(parser.cross_validation_type,
                         "<class 'sklearn.model_selection._search.GridSearchCV'>")
        self.assertEqual(parser.cross_validation_type, parser_from_dict.cross_validation_type)
        self.assertEqual(parser.cross_validation_type, parser_from_yaml.cross_validation_type)
        self.assertEqual(parser.number_of_splits,
                         grid_search_credit.cv.n_repeats * grid_search_credit.n_splits_)
        self.assertEqual(parser.number_of_splits, parser_from_dict.number_of_splits)
        self.assertEqual(parser.number_of_splits, parser_from_yaml.number_of_splits)
        self.assertEqual(parser.score_names, ['ROC/AUC', 'F1', 'Pos. Pred. Val', 'True Pos. Rate'])
        self.assertEqual(parser.score_names, parser_from_dict.score_names)
        self.assertEqual(parser.score_names, parser_from_yaml.score_names)
        self.assertEqual(parser.parameter_names_original, list(new_param_column_names.keys()))
        self.assertEqual(parser.parameter_names_original, parser_from_dict.parameter_names_original)
        self.assertEqual(parser.parameter_names_original, parser_from_yaml.parameter_names_original)
        self.assertEqual(parser.parameter_names, list(new_param_column_names.values()))
        self.assertEqual(parser.parameter_names, parser_from_dict.parameter_names)
        self.assertEqual(parser.parameter_names, parser_from_yaml.parameter_names)
        self.assertEqual(parser.parameter_names_mapping, new_param_column_names)
        self.assertEqual(parser.parameter_names_mapping, parser_from_dict.parameter_names_mapping)
        self.assertEqual(parser.parameter_names_mapping, parser_from_yaml.parameter_names_mapping)

        self.assertTrue(isinstance(parser.test_score_rankings, dict))
        self.assertEqual(list(parser.test_score_rankings.keys()), parser.score_names)
        self.assertEqual(parser.test_score_rankings, parser_from_dict.test_score_rankings)
        self.assertEqual(parser.test_score_rankings, parser_from_yaml.test_score_rankings)
        for score in parser.score_names:
            self.assertTrue(all(np.array(parser.test_score_rankings[score]) == grid_search_credit.cv_results_[f'rank_test_{score}']))

        self.assertEqual(
            parser.trial_labels(order_from_best_to_worst=True),
            [
                '{max_features: auto, n_estimators: 50, min_samples_split: 2, encoder: OneHotEncoder()}',
                '{max_features: auto, n_estimators: 50, min_samples_split: 2, encoder: CustomOrdinalEncoder()}',
                '{max_features: auto, n_estimators: 10, min_samples_split: 2, encoder: OneHotEncoder()}',
                '{max_features: auto, n_estimators: 10, min_samples_split: 2, encoder: CustomOrdinalEncoder()}',
                '{max_features: 100, n_estimators: 10, min_samples_split: 2, encoder: OneHotEncoder()}',
                '{max_features: 100, n_estimators: 10, min_samples_split: 2, encoder: CustomOrdinalEncoder()}',
                '{max_features: 100, n_estimators: 50, min_samples_split: 2, encoder: OneHotEncoder()}',
                '{max_features: 100, n_estimators: 50, min_samples_split: 2, encoder: CustomOrdinalEncoder()}'
            ]
        )
        self.assertEqual(
            parser.trial_labels(order_from_best_to_worst=False),
            [
                '{max_features: 100, n_estimators: 10, min_samples_split: 2, encoder: OneHotEncoder()}',
                '{max_features: 100, n_estimators: 10, min_samples_split: 2, encoder: CustomOrdinalEncoder()}',
                '{max_features: 100, n_estimators: 50, min_samples_split: 2, encoder: OneHotEncoder()}',
                '{max_features: 100, n_estimators: 50, min_samples_split: 2, encoder: CustomOrdinalEncoder()}',
                '{max_features: auto, n_estimators: 10, min_samples_split: 2, encoder: OneHotEncoder()}',
                '{max_features: auto, n_estimators: 10, min_samples_split: 2, encoder: CustomOrdinalEncoder()}',
                '{max_features: auto, n_estimators: 50, min_samples_split: 2, encoder: OneHotEncoder()}',
                '{max_features: auto, n_estimators: 50, min_samples_split: 2, encoder: CustomOrdinalEncoder()}'
             ]
        )

        def assert_np_arrays_are_close(array1, array2):
            self.assertEqual(len(array1), len(array2))
            for index in range(len(array1)):
                is_close = hlp.validation.is_close(array1[index], array2[index])
                both_nan = np.isnan(array1[index]) and np.isnan(array2[index])
                self.assertTrue(is_close or both_nan)

        self.assertTrue(isinstance(parser.test_score_averages, dict))
        self.assertEqual(list(parser.test_score_averages.keys()), parser.score_names)

        assert_np_arrays_are_close(np.array([1, 2, 3]), np.array([1, 2, 3]))
        assert_np_arrays_are_close(np.array([1, 2, np.nan]), np.array([1, 2, np.nan]))
        self.assertRaises(AssertionError,
                          lambda: assert_np_arrays_are_close(np.array([1, 2, 3]), np.array([1, 2, 3.001])))
        self.assertRaises(AssertionError,
                          lambda: assert_np_arrays_are_close(np.array([1, 2, 3]), np.array([1, 2, np.nan])))

        cv_dataframe = parser.to_dataframe(exclude_zero_variance_params=False)
        with redirect_stdout_to_file(get_test_path() + '/test_files/sklearn_eval/credit__grid_search__all_scores__dataframe__no_exclude.txt'):
            print_dataframe(cv_dataframe)
        # ensure the hyper-param columns (last 4 columns) are in the same order as the mapping.
        self.assertEqual(list(cv_dataframe.columns[-4:]), list(new_param_column_names.values()))

        self.assertTrue('min_samples_split' in cv_dataframe.columns)
        self.assertTrue(all(cv_dataframe['min_samples_split'] == [2, 2, 2, 2, 2, 2, 2, 2]))
        del cv_dataframe

        with redirect_stdout_to_file(get_test_path() + '/test_files/sklearn_eval/credit__grid_search__all_scores__dataframe__exclude.txt'):
            print_dataframe(parser.to_dataframe(exclude_zero_variance_params=True))

        self.assertEqual(list(parser.best_trial_indexes), list(parser.to_dataframe().index))
        cv_dataframe = parser.to_dataframe().sort_index()
        # ensure the hyper-param columns (now last 3 columns) are in the same order as the mapping
        self.assertEqual(list(cv_dataframe.columns[-3:]), ['max_features', 'n_estimators', 'encoder'])
        self.assertFalse('min_samples_split' in cv_dataframe.columns)
        hlp.validation.assert_dataframes_match([parser.to_dataframe(sort_by_score=False), cv_dataframe])

        labeled_dataframe = parser.to_labeled_dataframe()
        self.assertTrue(all(labeled_dataframe['Trial Index'] == list(range(1, parser.number_of_trials + 1))))
        self.assertIsNotNone(labeled_dataframe['label'])
        hlp.validation.assert_dataframes_match(dataframes=[parser.to_dataframe(sort_by_score=False),
                                                           labeled_dataframe.drop(columns=['Trial Index',
                                                                                           'label'])])

        assert_np_arrays_are_close(cv_dataframe[f'{parser.score_names[0]} Mean'],
                                   grid_search_credit.cv_results_[f'mean_test_{parser.score_names[0]}'])
        assert_np_arrays_are_close(cv_dataframe[f'{parser.score_names[1]} Mean'],
                                   grid_search_credit.cv_results_[f'mean_test_{parser.score_names[1]}'])
        assert_np_arrays_are_close(cv_dataframe[f'{parser.score_names[2]} Mean'],
                                   grid_search_credit.cv_results_[f'mean_test_{parser.score_names[2]}'])
        assert_np_arrays_are_close(cv_dataframe[f'{parser.score_names[3]} Mean'],
                                   grid_search_credit.cv_results_[f'mean_test_{parser.score_names[3]}'])

        self.assertEqual(list(grid_search_credit.cv_results_['param_model__max_features'].data),
                         cv_dataframe[parser.parameter_names[0]].tolist())
        self.assertEqual(list(grid_search_credit.cv_results_['param_model__n_estimators'].data),
                         cv_dataframe[parser.parameter_names[1]].tolist())
        encoder_param_name = 'param_preparation__non_numeric_pipeline__encoder_chooser__transformer'
        encoder_values = [str(x) for x in list(grid_search_credit.cv_results_[encoder_param_name].data)]
        self.assertEqual(encoder_values, cv_dataframe[parser.parameter_names[3]].tolist())

        with open(get_test_path() + '/test_files/sklearn_eval/credit__grid_search__all_scores.html', 'w') as file:
            file.write(clean_formatted_dataframe(parser.to_formatted_dataframe().render()))

        with open(get_test_path() + '/test_files/sklearn_eval/credit__grid_search__all_scores_2.html', 'w') as file:
            file.write(clean_formatted_dataframe(parser.to_formatted_dataframe(exclude_zero_variance_params=False).render()))

        with open(get_test_path() + '/test_files/sklearn_eval/credit__grid_search__primary_score_only.html', 'w') as file:
            file.write(clean_formatted_dataframe(parser.to_formatted_dataframe(primary_score_only=True).render()))

        assert_np_arrays_are_close(parser.primary_score_averages,
                                   np.array(parser.test_score_averages[parser.primary_score_name]))
        assert_np_arrays_are_close(parser.primary_score_averages,
                                   grid_search_credit.cv_results_['mean_test_ROC/AUC'])
        assert_np_arrays_are_close(parser.primary_score_averages,
                                   parser_from_dict.primary_score_averages)
        assert_np_arrays_are_close(parser.primary_score_averages,
                                   parser_from_yaml.primary_score_averages)

        self.assertEqual(len(parser.score_standard_errors(score_name=parser.score_names[0])),
                         parser.number_of_trials)
        self.assertEqual(len(parser.score_standard_errors(score_name=parser.score_names[1])),
                         parser.number_of_trials)
        self.assertEqual(len(parser.score_standard_errors(score_name=parser.score_names[2])),
                         parser.number_of_trials)
        self.assertEqual(len(parser.score_standard_errors(score_name=parser.score_names[3])),
                         parser.number_of_trials)

        self.assertEqual(parser.numeric_parameters, ['n_estimators'])
        self.assertEqual(parser.non_numeric_parameters, ['max_features', 'encoder'])

        for score in parser.score_names:
            assert_np_arrays_are_close(np.array(parser.test_score_averages[score]),
                                       grid_search_credit.cv_results_[f'mean_test_{score}'])
            assert_np_arrays_are_close(np.array(parser.test_score_averages[score]),
                                       np.array(parser_from_dict.test_score_averages[score]))
            assert_np_arrays_are_close(np.array(parser.test_score_averages[score]),
                                       np.array(parser_from_yaml.test_score_averages[score]))

        self.assertTrue(isinstance(parser.test_score_standard_deviations, dict))
        self.assertEqual(list(parser.test_score_standard_deviations.keys()), parser.score_names)

        for score in parser.score_names:
            assert_np_arrays_are_close(np.array(parser.test_score_standard_deviations[score]),
                                       grid_search_credit.cv_results_[f'std_test_{score}'])
            assert_np_arrays_are_close(np.array(parser.test_score_standard_deviations[score]),
                                       np.array(parser_from_dict.test_score_standard_deviations[score]))
            assert_np_arrays_are_close(np.array(parser.test_score_standard_deviations[score]),
                                       np.array(parser_from_yaml.test_score_standard_deviations[score]))

        self.assertTrue(isinstance(parser.train_score_averages, dict))
        self.assertEqual(list(parser.train_score_averages.keys()), parser.score_names)

        for score in parser.score_names:
            assert_np_arrays_are_close(np.array(parser.train_score_averages[score]),
                                       grid_search_credit.cv_results_[f'mean_train_{score}'])
            assert_np_arrays_are_close(np.array(parser.train_score_averages[score]),
                                       np.array(parser_from_dict.train_score_averages[score]))
            assert_np_arrays_are_close(np.array(parser.train_score_averages[score]),
                                       np.array(parser_from_yaml.train_score_averages[score]))

        self.assertTrue(isinstance(parser.train_score_standard_deviations, dict))
        self.assertEqual(list(parser.train_score_standard_deviations.keys()), parser.score_names)

        for score in parser.score_names:
            assert_np_arrays_are_close(np.array(parser.train_score_standard_deviations[score]),
                                       grid_search_credit.cv_results_[f'std_train_{score}'])
            assert_np_arrays_are_close(np.array(parser.train_score_standard_deviations[score]),
                                       np.array(parser_from_dict.train_score_standard_deviations[score]))
            assert_np_arrays_are_close(np.array(parser.train_score_standard_deviations[score]),
                                       np.array(parser_from_yaml.train_score_standard_deviations[score]))

        self.assertTrue(isinstance(parser.trials, list))
        self.assertEqual(len(parser.trials), parser.number_of_trials)
        self.assertEqual(parser.trials, parser_from_dict.trials)
        self.assertEqual(parser.trials, parser_from_yaml.trials)

        self.assertTrue(isinstance(parser.timings, dict))
        self.assertEqual(list(parser.timings.keys()), ['fit time averages',
                                                       'fit time standard deviations',
                                                       'score time averages',
                                                       'score time standard deviations'])

        assert_np_arrays_are_close(np.array(parser.timings[list(parser.timings.keys())[0]]),
                                   grid_search_credit.cv_results_['mean_fit_time'])
        assert_np_arrays_are_close(np.array(parser.timings[list(parser.timings.keys())[1]]),
                                   grid_search_credit.cv_results_['std_fit_time'])
        assert_np_arrays_are_close(np.array(parser.timings[list(parser.timings.keys())[2]]),
                                   grid_search_credit.cv_results_['mean_score_time'])
        assert_np_arrays_are_close(np.array(parser.timings[list(parser.timings.keys())[3]]),
                                   grid_search_credit.cv_results_['std_score_time'])

        for timing in list(parser.timings.keys()):
            assert_np_arrays_are_close(np.array(parser.timings[timing]),
                                       np.array(parser_from_dict.timings[timing]))
            assert_np_arrays_are_close(np.array(parser.timings[timing]),
                                       np.array(parser_from_yaml.timings[timing]))

        self.assertEqual(parser.number_of_trials, 8)
        self.assertEqual(parser.number_of_trials, parser_from_dict.number_of_trials)
        self.assertEqual(parser.number_of_trials, parser_from_yaml.number_of_trials)

        self.assertEqual(parser.number_of_scores, 4)
        self.assertEqual(parser.number_of_scores, parser_from_dict.number_of_scores)
        self.assertEqual(parser.number_of_scores, parser_from_yaml.number_of_scores)

        self.assertEqual(parser.primary_score_name, 'ROC/AUC')
        self.assertEqual(parser.primary_score_name, parser_from_dict.primary_score_name)
        self.assertEqual(parser.primary_score_name, parser_from_yaml.primary_score_name)

        self.assertTrue(all(parser.trial_rankings == grid_search_credit.cv_results_['rank_test_ROC/AUC']))  # noqa
        self.assertEqual(parser.best_score_index, 6)
        self.assertEqual(parser.best_score,
                         grid_search_credit.cv_results_['mean_test_ROC/AUC'][6])
        self.assertEqual(parser.best_score,
                         np.nanmax(grid_search_credit.cv_results_['mean_test_ROC/AUC']))

        self.assertEqual(parser.parameter_names_original,
                         ['model__max_features',
                          'model__n_estimators',
                          'model__min_samples_split',
                          'preparation__non_numeric_pipeline__encoder_chooser__transformer'])
        self.assertEqual(parser.parameter_names,
                         ['max_features', 'n_estimators', 'min_samples_split', 'encoder'])
        self.assertEqual(parser.best_params,
                         {'max_features': 'auto',
                          'n_estimators': 50,
                          'min_samples_split': 2,
                          'encoder': 'OneHotEncoder()'})

        self.assertTrue(all([parser.to_dataframe(exclude_zero_variance_params=False).loc[parser.best_score_index, key] == value
                             for key, value in parser.best_params.items()]))

        self.assertEqual(parser.best_score, parser_from_dict.best_score)
        self.assertEqual(parser.best_score, parser_from_yaml.best_score)

    def test_MLExperimentResults_gridsearch_classification_single_score(self):
        # test grid search object that has one score (classification)
        # not passing in parameter mappings
        grid_search_credit = self.credit_data__grid_search__roc_auc
        parser = MLExperimentResults.from_sklearn_search_cv(searcher=grid_search_credit,
                                                            higher_score_is_better=True,
                                                            description="test description",
                                                            parameter_name_mappings=None)
        yaml_file = get_test_path() + '/test_files/sklearn_eval/credit_data__grid_search_roc.yaml'
        os.remove(yaml_file)
        parser.to_yaml_file(yaml_file)
        parser_from_dict = MLExperimentResults(parser._dict)
        parser_from_yaml = MLExperimentResults.from_yaml_file(yaml_file)

        self.assertEqual(str(parser._dict), str(parser_from_dict._dict))
        self.assertEqual(str(parser._dict), str(parser_from_yaml._dict))

        self.assertEqual(parser.description, "test description")
        self.assertEqual(parser.description, parser_from_dict.description)
        self.assertEqual(parser.description, parser_from_yaml.description)
        self.assertEqual(parser.higher_score_is_better, True)
        self.assertEqual(parser.higher_score_is_better, parser_from_dict.higher_score_is_better)
        self.assertEqual(parser.higher_score_is_better, parser_from_yaml.higher_score_is_better)
        self.assertEqual(parser.cross_validation_type,
                         "<class 'sklearn.model_selection._search.GridSearchCV'>")
        self.assertEqual(parser.cross_validation_type, parser_from_dict.cross_validation_type)
        self.assertEqual(parser.cross_validation_type, parser_from_yaml.cross_validation_type)
        self.assertEqual(parser.number_of_splits,
                         grid_search_credit.cv.n_repeats * grid_search_credit.n_splits_)
        self.assertEqual(parser.number_of_splits, parser_from_dict.number_of_splits)
        self.assertEqual(parser.number_of_splits, parser_from_yaml.number_of_splits)
        self.assertEqual(parser.score_names, ['roc_auc'])
        self.assertEqual(parser.score_names, parser_from_dict.score_names)
        self.assertEqual(parser.score_names, parser_from_yaml.score_names)
        self.assertEqual(parser.parameter_names_original,
                         ['model__max_features', 'model__n_estimators',
                          'preparation__non_numeric_pipeline__encoder_chooser__transformer'])
        self.assertEqual(parser.parameter_names_original, parser_from_dict.parameter_names_original)
        self.assertEqual(parser.parameter_names_original, parser_from_yaml.parameter_names_original)
        self.assertEqual(parser.parameter_names, parser.parameter_names_original)
        self.assertEqual(parser.parameter_names, parser_from_dict.parameter_names)
        self.assertEqual(parser.parameter_names, parser_from_yaml.parameter_names)
        self.assertIsNone(parser.parameter_names_mapping)
        self.assertIsNone(parser_from_dict.parameter_names_mapping)
        self.assertIsNone(parser_from_yaml.parameter_names_mapping)

        self.assertTrue(isinstance(parser.test_score_rankings, dict))
        self.assertEqual(list(parser.test_score_rankings.keys()), parser.score_names)
        self.assertTrue(all(np.array(parser.test_score_rankings['roc_auc']) == grid_search_credit.cv_results_['rank_test_score']))
        self.assertEqual(parser.test_score_rankings, parser_from_dict.test_score_rankings)
        self.assertEqual(parser.test_score_rankings, parser_from_yaml.test_score_rankings)

        self.assertEqual(parser.trial_labels(order_from_best_to_worst=True),
                         ['{model__max_features: auto, model__n_estimators: 50, preparation__non_numeric_pipeline__encoder_chooser__transformer: OneHotEncoder()}',  # noqa
                          '{model__max_features: auto, model__n_estimators: 50, preparation__non_numeric_pipeline__encoder_chooser__transformer: CustomOrdinalEncoder()}',  # noqa
                          '{model__max_features: auto, model__n_estimators: 10, preparation__non_numeric_pipeline__encoder_chooser__transformer: OneHotEncoder()}',  # noqa
                          '{model__max_features: auto, model__n_estimators: 10, preparation__non_numeric_pipeline__encoder_chooser__transformer: CustomOrdinalEncoder()}',  # noqa
                          '{model__max_features: 100, model__n_estimators: 10, preparation__non_numeric_pipeline__encoder_chooser__transformer: OneHotEncoder()}',  # noqa
                          '{model__max_features: 100, model__n_estimators: 10, preparation__non_numeric_pipeline__encoder_chooser__transformer: CustomOrdinalEncoder()}',  # noqa
                          '{model__max_features: 100, model__n_estimators: 50, preparation__non_numeric_pipeline__encoder_chooser__transformer: OneHotEncoder()}',  # noqa
                          '{model__max_features: 100, model__n_estimators: 50, preparation__non_numeric_pipeline__encoder_chooser__transformer: CustomOrdinalEncoder()}'])  # noqa
        self.assertEqual(parser.trial_labels(order_from_best_to_worst=False),
                         ['{model__max_features: 100, model__n_estimators: 10, preparation__non_numeric_pipeline__encoder_chooser__transformer: OneHotEncoder()}',  # noqa
                          '{model__max_features: 100, model__n_estimators: 10, preparation__non_numeric_pipeline__encoder_chooser__transformer: CustomOrdinalEncoder()}',  # noqa
                          '{model__max_features: 100, model__n_estimators: 50, preparation__non_numeric_pipeline__encoder_chooser__transformer: OneHotEncoder()}',  # noqa
                          '{model__max_features: 100, model__n_estimators: 50, preparation__non_numeric_pipeline__encoder_chooser__transformer: CustomOrdinalEncoder()}',  # noqa
                          '{model__max_features: auto, model__n_estimators: 10, preparation__non_numeric_pipeline__encoder_chooser__transformer: OneHotEncoder()}',  # noqa
                          '{model__max_features: auto, model__n_estimators: 10, preparation__non_numeric_pipeline__encoder_chooser__transformer: CustomOrdinalEncoder()}',  # noqa
                          '{model__max_features: auto, model__n_estimators: 50, preparation__non_numeric_pipeline__encoder_chooser__transformer: OneHotEncoder()}',  # noqa
                          '{model__max_features: auto, model__n_estimators: 50, preparation__non_numeric_pipeline__encoder_chooser__transformer: CustomOrdinalEncoder()}'])  # noqa

        def assert_np_arrays_are_close(array1, array2):
            self.assertEqual(len(array1), len(array2))
            for index in range(len(array1)):
                is_close = hlp.validation.is_close(array1[index], array2[index])
                both_nan = np.isnan(array1[index]) and np.isnan(array2[index])
                self.assertTrue(is_close or both_nan)

        assert_np_arrays_are_close(np.array(parser.test_score_averages[parser.primary_score_name]),
                                   grid_search_credit.cv_results_['mean_test_score'])
        assert_np_arrays_are_close(np.array(parser.test_score_averages[parser.primary_score_name]),
                                   np.array(parser_from_dict.test_score_averages[parser.primary_score_name]))
        assert_np_arrays_are_close(np.array(parser.test_score_averages[parser.primary_score_name]),
                                   np.array(parser_from_yaml.test_score_averages[parser.primary_score_name]))

        self.assertEqual(list(parser.best_trial_indexes), list(parser.to_dataframe().index))
        with redirect_stdout_to_file(get_test_path() + '/test_files/sklearn_eval/credit__grid_search__single_score__dataframe.txt'):
            print_dataframe(parser.to_dataframe())
        cv_dataframe = parser.to_dataframe().sort_index()
        hlp.validation.assert_dataframes_match([parser.to_dataframe(sort_by_score=False), cv_dataframe])
        assert_np_arrays_are_close(cv_dataframe[f'{parser.score_names[0]} Mean'],
                                   grid_search_credit.cv_results_['mean_test_score'])

        labeled_dataframe = parser.to_labeled_dataframe()
        self.assertTrue(all(labeled_dataframe['Trial Index'] == list(range(1, parser.number_of_trials + 1))))
        self.assertIsNotNone(labeled_dataframe['label'])
        hlp.validation.assert_dataframes_match(dataframes=[parser.to_dataframe(sort_by_score=False),
                                                           labeled_dataframe.drop(columns=['Trial Index',
                                                                                           'label'])])

        self.assertEqual(list(grid_search_credit.cv_results_['param_model__max_features'].data), cv_dataframe[parser.parameter_names[0]].tolist())
        self.assertEqual(list(grid_search_credit.cv_results_['param_model__n_estimators'].data), cv_dataframe[parser.parameter_names[1]].tolist())
        encoder_values = [str(x) for x in
                          list(grid_search_credit.cv_results_['param_preparation__non_numeric_pipeline__encoder_chooser__transformer'].data)]
        self.assertEqual(encoder_values, cv_dataframe[parser.parameter_names[2]].tolist())

        with open(get_test_path() + '/test_files/sklearn_eval/credit__grid_search__single_score_all_scores.html', 'w') as file:
            file.write(clean_formatted_dataframe(parser.to_formatted_dataframe().render()))

        with open(get_test_path() + '/test_files/sklearn_eval/credit__grid_search__single_score_primary_score_only.html', 'w') as file:
            file.write(clean_formatted_dataframe(parser.to_formatted_dataframe(primary_score_only=True).render()))

        self.assertTrue(isinstance(parser.test_score_averages, dict))
        self.assertEqual(list(parser.test_score_averages.keys()), parser.score_names)

        assert_np_arrays_are_close(np.array([1, 2, 3]), np.array([1, 2, 3]))
        assert_np_arrays_are_close(np.array([1, 2, np.nan]), np.array([1, 2, np.nan]))
        self.assertRaises(AssertionError,
                          lambda: assert_np_arrays_are_close(np.array([1, 2, 3]), np.array([1, 2, 3.001])))
        self.assertRaises(AssertionError,
                          lambda: assert_np_arrays_are_close(np.array([1, 2, 3]), np.array([1, 2, np.nan])))

        assert_np_arrays_are_close(parser.primary_score_averages,
                                   np.array(parser.test_score_averages[parser.primary_score_name]))
        assert_np_arrays_are_close(parser.primary_score_averages,
                                   grid_search_credit.cv_results_['mean_test_score'])
        assert_np_arrays_are_close(parser.primary_score_averages,
                                   parser_from_dict.primary_score_averages)
        assert_np_arrays_are_close(parser.primary_score_averages,
                                   parser_from_yaml.primary_score_averages)

        self.assertEqual(len(parser.score_standard_errors(score_name=parser.score_names[0])),
                         parser.number_of_trials)

        self.assertEqual(parser.numeric_parameters, ['model__n_estimators'])
        self.assertEqual(parser.non_numeric_parameters, ['model__max_features',
                                                         'preparation__non_numeric_pipeline__encoder_chooser__transformer'])

        assert_np_arrays_are_close(np.array(parser.test_score_averages[parser.primary_score_name]),
                                   grid_search_credit.cv_results_['mean_test_score'])
        assert_np_arrays_are_close(np.array(parser.test_score_averages[parser.primary_score_name]),
                                   np.array(parser_from_dict.test_score_averages[parser.primary_score_name]))
        assert_np_arrays_are_close(np.array(parser.test_score_averages[parser.primary_score_name]),
                                   np.array(parser_from_yaml.test_score_averages[parser.primary_score_name]))

        self.assertTrue(isinstance(parser.test_score_standard_deviations, dict))
        self.assertEqual(list(parser.test_score_standard_deviations.keys()), parser.score_names)

        assert_np_arrays_are_close(np.array(parser.test_score_standard_deviations[parser.primary_score_name]),
                                   grid_search_credit.cv_results_['std_test_score'])
        assert_np_arrays_are_close(np.array(parser.test_score_standard_deviations[parser.primary_score_name]),
                                   np.array(parser_from_dict.test_score_standard_deviations[parser.primary_score_name]))
        assert_np_arrays_are_close(np.array(parser.test_score_standard_deviations[parser.primary_score_name]),
                                   np.array(parser_from_yaml.test_score_standard_deviations[parser.primary_score_name]))

        self.assertIsNone(parser.train_score_averages)
        self.assertIsNone(parser.train_score_standard_deviations)

        self.assertTrue(isinstance(parser.trials, list))
        self.assertEqual(len(parser.trials), parser.number_of_trials)
        self.assertEqual(parser.trials, parser_from_dict.trials)
        self.assertEqual(parser.trials, parser_from_yaml.trials)

        self.assertTrue(isinstance(parser.timings, dict))
        self.assertEqual(list(parser.timings.keys()), ['fit time averages',
                                                       'fit time standard deviations',
                                                       'score time averages',
                                                       'score time standard deviations'])

        assert_np_arrays_are_close(np.array(parser.timings[list(parser.timings.keys())[0]]),
                                   grid_search_credit.cv_results_['mean_fit_time'])
        assert_np_arrays_are_close(np.array(parser.timings[list(parser.timings.keys())[1]]),
                                   grid_search_credit.cv_results_['std_fit_time'])
        assert_np_arrays_are_close(np.array(parser.timings[list(parser.timings.keys())[2]]),
                                   grid_search_credit.cv_results_['mean_score_time'])
        assert_np_arrays_are_close(np.array(parser.timings[list(parser.timings.keys())[3]]),
                                   grid_search_credit.cv_results_['std_score_time'])

        for timing in list(parser.timings.keys()):
            assert_np_arrays_are_close(np.array(parser.timings[timing]),
                                       np.array(parser_from_dict.timings[timing]))
            assert_np_arrays_are_close(np.array(parser.timings[timing]),
                                       np.array(parser_from_yaml.timings[timing]))

        self.assertEqual(parser.number_of_trials, 8)
        self.assertEqual(parser.number_of_trials, parser_from_dict.number_of_trials)
        self.assertEqual(parser.number_of_trials, parser_from_yaml.number_of_trials)

        self.assertEqual(parser.number_of_scores, 1)
        self.assertEqual(parser.number_of_scores, parser_from_dict.number_of_scores)
        self.assertEqual(parser.number_of_scores, parser_from_yaml.number_of_scores)

        self.assertEqual(parser.primary_score_name, 'roc_auc')
        self.assertEqual(parser.primary_score_name, parser_from_dict.primary_score_name)
        self.assertEqual(parser.primary_score_name, parser_from_yaml.primary_score_name)

        self.assertTrue(all(parser.trial_rankings == grid_search_credit.cv_results_['rank_test_score']))  # noqa
        self.assertEqual(parser.best_score_index, 6)
        self.assertEqual(parser.best_score,
                         grid_search_credit.cv_results_['mean_test_score'][6])
        self.assertEqual(parser.best_score,
                         np.nanmax(grid_search_credit.cv_results_['mean_test_score']))

        self.assertTrue(all([parser.to_dataframe().loc[parser.best_score_index, key] == value
                             for key, value in parser.best_params.items()]))

        self.assertEqual(parser.best_params['model__max_features'],
                         grid_search_credit.best_params_['model__max_features'])
        self.assertEqual(parser.best_params['model__n_estimators'],
                         grid_search_credit.best_params_['model__n_estimators'])
        self.assertEqual(parser.best_params['preparation__non_numeric_pipeline__encoder_chooser__transformer'],
                         str(grid_search_credit.best_params_['preparation__non_numeric_pipeline__encoder_chooser__transformer']))

        self.assertEqual(parser.best_score, parser_from_dict.best_score)
        self.assertEqual(parser.best_score, parser_from_yaml.best_score)

    def test_MLExperimentResults_gridsearch_regression(self):
        # test grid search object that has multiple scores (regression)
        # not passing in parameter mappings
        grid_search_housing = self.housing_data__grid_search
        parser = MLExperimentResults.from_sklearn_search_cv(searcher=grid_search_housing,
                                                            higher_score_is_better=False,
                                                            description="test description")
        yaml_file = get_test_path() + '/test_files/sklearn_eval/housing_data__grid_search.yaml'
        os.remove(yaml_file)
        parser.to_yaml_file(yaml_file)
        parser_from_dict = MLExperimentResults(parser._dict)
        parser_from_yaml = MLExperimentResults.from_yaml_file(yaml_file)

        self.assertEqual(str(parser._dict), str(parser_from_dict._dict))
        self.assertEqual(str(parser._dict), str(parser_from_yaml._dict))

        self.assertEqual(parser.description, "test description")
        self.assertEqual(parser.description, parser_from_dict.description)
        self.assertEqual(parser.description, parser_from_yaml.description)
        self.assertEqual(parser.higher_score_is_better, False)
        self.assertEqual(parser.higher_score_is_better, parser_from_dict.higher_score_is_better)
        self.assertEqual(parser.higher_score_is_better, parser_from_yaml.higher_score_is_better)
        self.assertEqual(parser.cross_validation_type,
                         "<class 'sklearn.model_selection._search.GridSearchCV'>")
        self.assertEqual(parser.cross_validation_type, parser_from_dict.cross_validation_type)
        self.assertEqual(parser.cross_validation_type, parser_from_yaml.cross_validation_type)
        self.assertEqual(parser.number_of_splits,
                         grid_search_housing.cv.n_repeats * grid_search_housing.n_splits_)
        self.assertEqual(parser.number_of_splits, parser_from_dict.number_of_splits)
        self.assertEqual(parser.number_of_splits, parser_from_yaml.number_of_splits)
        self.assertEqual(parser.score_names, ['RMSE', 'MAE'])
        self.assertEqual(parser.score_names, parser_from_dict.score_names)
        self.assertEqual(parser.score_names, parser_from_yaml.score_names)

        self.assertEqual(parser.parameter_names_original, parser_from_dict.parameter_names_original)
        self.assertEqual(parser.parameter_names_original, parser_from_yaml.parameter_names_original)
        self.assertEqual(parser.parameter_names, parser.parameter_names_original)
        self.assertEqual(parser.parameter_names, parser_from_dict.parameter_names)
        self.assertEqual(parser.parameter_names, parser_from_yaml.parameter_names)
        self.assertIsNone(parser.parameter_names_mapping)
        self.assertIsNone(parser_from_dict.parameter_names_mapping)
        self.assertIsNone(parser_from_yaml.parameter_names_mapping)

        self.assertTrue(isinstance(parser.test_score_rankings, dict))
        self.assertEqual(list(parser.test_score_rankings.keys()), parser.score_names)
        self.assertEqual(parser.test_score_rankings, parser_from_dict.test_score_rankings)
        self.assertEqual(parser.test_score_rankings, parser_from_yaml.test_score_rankings)
        for score in parser.score_names:
            self.assertTrue(all(np.array(parser.test_score_rankings[score]) == grid_search_housing.cv_results_[f'rank_test_{score}']))

        self.assertEqual(parser.trial_labels(order_from_best_to_worst=True),
                         ['{model__max_features: auto, model__n_estimators: 50}',
                          '{model__max_features: auto, model__n_estimators: 10}',
                          '{model__max_features: 2, model__n_estimators: 50}',
                          '{model__max_features: 2, model__n_estimators: 10}'])
        self.assertEqual(parser.trial_labels(order_from_best_to_worst=False),
                         ['{model__max_features: 2, model__n_estimators: 10}',
                          '{model__max_features: 2, model__n_estimators: 50}',
                          '{model__max_features: auto, model__n_estimators: 10}',
                          '{model__max_features: auto, model__n_estimators: 50}'])

        def assert_np_arrays_are_close(array1, array2):
            self.assertEqual(len(array1), len(array2))
            for index in range(len(array1)):
                is_close = hlp.validation.is_close(array1[index], array2[index])
                both_nan = np.isnan(array1[index]) and np.isnan(array2[index])
                self.assertTrue(is_close or both_nan)

        self.assertTrue(isinstance(parser.test_score_averages, dict))
        self.assertEqual(list(parser.test_score_averages.keys()), parser.score_names)

        assert_np_arrays_are_close(np.array([1, 2, 3]), np.array([1, 2, 3]))
        assert_np_arrays_are_close(np.array([1, 2, np.nan]), np.array([1, 2, np.nan]))
        self.assertRaises(AssertionError,
                          lambda: assert_np_arrays_are_close(np.array([1, 2, 3]), np.array([1, 2, 3.001])))
        self.assertRaises(AssertionError,
                          lambda: assert_np_arrays_are_close(np.array([1, 2, 3]), np.array([1, 2, np.nan])))

        self.assertEqual(list(parser.best_trial_indexes), list(parser.to_dataframe().index))
        with redirect_stdout_to_file(get_test_path() + '/test_files/sklearn_eval/housing__grid_search__dataframe.txt'):
            print_dataframe(parser.to_dataframe())
        cv_dataframe = parser.to_dataframe().sort_index()
        hlp.validation.assert_dataframes_match([parser.to_dataframe(sort_by_score=False), cv_dataframe])
        assert_np_arrays_are_close(cv_dataframe[f'{parser.score_names[0]} Mean'],
                                   grid_search_housing.cv_results_[f'mean_test_{parser.score_names[0]}'] * -1)
        assert_np_arrays_are_close(cv_dataframe[f'{parser.score_names[1]} Mean'],
                                   grid_search_housing.cv_results_[f'mean_test_{parser.score_names[1]}'] * -1)

        labeled_dataframe = parser.to_labeled_dataframe()
        self.assertTrue(all(labeled_dataframe['Trial Index'] == list(range(1, parser.number_of_trials + 1))))
        self.assertIsNotNone(labeled_dataframe['label'])
        hlp.validation.assert_dataframes_match(dataframes=[parser.to_dataframe(sort_by_score=False),
                                                           labeled_dataframe.drop(columns=['Trial Index',
                                                                                           'label'])])

        self.assertEqual(list(grid_search_housing.cv_results_['param_model__max_features'].data), cv_dataframe[parser.parameter_names[0]].tolist())
        self.assertEqual(list(grid_search_housing.cv_results_['param_model__n_estimators'].data), cv_dataframe[parser.parameter_names[1]].tolist())

        with open(get_test_path() + '/test_files/sklearn_eval/housing__grid_search__all_scores.html', 'w') as file:
            file.write(clean_formatted_dataframe(parser.to_formatted_dataframe().render()))

        with open(get_test_path() + '/test_files/sklearn_eval/housing__grid_search__primary_score_only.html', 'w') as file:
            file.write(clean_formatted_dataframe(parser.to_formatted_dataframe(primary_score_only=True).render()))

        self.assertTrue(isinstance(parser.test_score_averages, dict))
        self.assertEqual(list(parser.test_score_averages.keys()), parser.score_names)

        assert_np_arrays_are_close(parser.primary_score_averages,
                                   np.array(parser.test_score_averages[parser.primary_score_name]))
        assert_np_arrays_are_close(parser.primary_score_averages,
                                   grid_search_housing.cv_results_['mean_test_RMSE'] * -1)
        assert_np_arrays_are_close(parser.primary_score_averages,
                                   parser_from_dict.primary_score_averages)
        assert_np_arrays_are_close(parser.primary_score_averages,
                                   parser_from_yaml.primary_score_averages)

        self.assertEqual(len(parser.score_standard_errors(score_name=parser.score_names[0])),
                         parser.number_of_trials)
        self.assertEqual(len(parser.score_standard_errors(score_name=parser.score_names[1])),
                         parser.number_of_trials)

        self.assertEqual(parser.numeric_parameters, ['model__n_estimators'])
        self.assertEqual(parser.non_numeric_parameters, ['model__max_features'])

        for score in parser.score_names:
            assert_np_arrays_are_close(np.array(parser.test_score_averages[score]),
                                       grid_search_housing.cv_results_[f'mean_test_{score}'] * -1)
            assert_np_arrays_are_close(np.array(parser.test_score_averages[score]),
                                       np.array(parser_from_dict.test_score_averages[score]))
            assert_np_arrays_are_close(np.array(parser.test_score_averages[score]),
                                       np.array(parser_from_yaml.test_score_averages[score]))

        self.assertTrue(isinstance(parser.test_score_standard_deviations, dict))
        self.assertEqual(list(parser.test_score_standard_deviations.keys()), parser.score_names)

        for score in parser.score_names:
            assert_np_arrays_are_close(np.array(parser.test_score_standard_deviations[score]),
                                       grid_search_housing.cv_results_[f'std_test_{score}'])
            assert_np_arrays_are_close(np.array(parser.test_score_standard_deviations[score]),
                                       np.array(parser_from_dict.test_score_standard_deviations[score]))
            assert_np_arrays_are_close(np.array(parser.test_score_standard_deviations[score]),
                                       np.array(parser_from_yaml.test_score_standard_deviations[score]))

        self.assertTrue(isinstance(parser.train_score_averages, dict))
        self.assertEqual(list(parser.train_score_averages.keys()), parser.score_names)

        for score in parser.score_names:
            assert_np_arrays_are_close(np.array(parser.train_score_averages[score]),
                                       grid_search_housing.cv_results_[f'mean_train_{score}'] * -1)
            assert_np_arrays_are_close(np.array(parser.train_score_averages[score]),
                                       np.array(parser_from_dict.train_score_averages[score]))
            assert_np_arrays_are_close(np.array(parser.train_score_averages[score]),
                                       np.array(parser_from_yaml.train_score_averages[score]))

        self.assertTrue(isinstance(parser.train_score_standard_deviations, dict))
        self.assertEqual(list(parser.train_score_standard_deviations.keys()), parser.score_names)

        for score in parser.score_names:
            assert_np_arrays_are_close(np.array(parser.train_score_standard_deviations[score]),
                                       grid_search_housing.cv_results_[f'std_train_{score}'])
            assert_np_arrays_are_close(np.array(parser.train_score_standard_deviations[score]),
                                       np.array(parser_from_dict.train_score_standard_deviations[score]))
            assert_np_arrays_are_close(np.array(parser.train_score_standard_deviations[score]),
                                       np.array(parser_from_yaml.train_score_standard_deviations[score]))

        self.assertTrue(isinstance(parser.trials, list))
        self.assertEqual(len(parser.trials), parser.number_of_trials)
        self.assertEqual(parser.trials, parser_from_dict.trials)
        self.assertEqual(parser.trials, parser_from_yaml.trials)

        self.assertTrue(isinstance(parser.timings, dict))
        self.assertEqual(list(parser.timings.keys()), ['fit time averages',
                                                       'fit time standard deviations',
                                                       'score time averages',
                                                       'score time standard deviations'])

        assert_np_arrays_are_close(np.array(parser.timings[list(parser.timings.keys())[0]]),
                                   grid_search_housing.cv_results_['mean_fit_time'])
        assert_np_arrays_are_close(np.array(parser.timings[list(parser.timings.keys())[1]]),
                                   grid_search_housing.cv_results_['std_fit_time'])
        assert_np_arrays_are_close(np.array(parser.timings[list(parser.timings.keys())[2]]),
                                   grid_search_housing.cv_results_['mean_score_time'])
        assert_np_arrays_are_close(np.array(parser.timings[list(parser.timings.keys())[3]]),
                                   grid_search_housing.cv_results_['std_score_time'])

        for timing in list(parser.timings.keys()):
            assert_np_arrays_are_close(np.array(parser.timings[timing]),
                                       np.array(parser_from_dict.timings[timing]))
            assert_np_arrays_are_close(np.array(parser.timings[timing]),
                                       np.array(parser_from_yaml.timings[timing]))

        self.assertEqual(parser.number_of_trials, 4)
        self.assertEqual(parser.number_of_trials, parser_from_dict.number_of_trials)
        self.assertEqual(parser.number_of_trials, parser_from_yaml.number_of_trials)

        self.assertEqual(parser.number_of_scores, 2)
        self.assertEqual(parser.number_of_scores, parser_from_dict.number_of_scores)
        self.assertEqual(parser.number_of_scores, parser_from_yaml.number_of_scores)

        self.assertEqual(parser.primary_score_name, 'RMSE')
        self.assertEqual(parser.primary_score_name, parser_from_dict.primary_score_name)
        self.assertEqual(parser.primary_score_name, parser_from_yaml.primary_score_name)

        self.assertTrue(all(parser.trial_rankings == grid_search_housing.cv_results_['rank_test_RMSE']))  # noqa
        self.assertEqual(parser.best_score_index, 3)
        self.assertEqual(parser.best_score,
                         grid_search_housing.cv_results_['mean_test_RMSE'][3] * -1)
        self.assertEqual(parser.best_score,
                         np.nanmin(grid_search_housing.cv_results_['mean_test_RMSE'] * -1))

        self.assertTrue(all([parser.to_dataframe().loc[parser.best_score_index, key] == value
                             for key, value in parser.best_params.items()]))

        self.assertEqual(parser.best_score, parser_from_dict.best_score)
        self.assertEqual(parser.best_score, parser_from_yaml.best_score)

    def test_MLExperimentResults_plots(self):
        new_param_column_names = {'model__max_features': 'max_features',
                                  'model__n_estimators': 'n_estimators',
                                  'model__min_samples_split': 'min_samples_split',
                                  'preparation__non_numeric_pipeline__encoder_chooser__transformer':
                                      'encoder'}
        parser = MLExperimentResults.from_sklearn_search_cv(searcher=self.credit_data__grid_search,
                                                            higher_score_is_better=True,
                                                            description="test description",
                                                            parameter_name_mappings=new_param_column_names)
        _ = parser.plot_performance_across_trials()
        _ = parser.plot_parameter_values_across_trials()
        _ = parser.plot_parallel_coordinates()
        _ = parser.plot_scatter_matrix()

        parser = MLExperimentResults.from_sklearn_search_cv(searcher=self.credit_data__grid_search__roc_auc,
                                                            higher_score_is_better=True,
                                                            description="test description",
                                                            parameter_name_mappings=None)
        _ = parser.plot_performance_across_trials()
        _ = parser.plot_parameter_values_across_trials()
        _ = parser.plot_parallel_coordinates()
        _ = parser.plot_scatter_matrix()

        parser = MLExperimentResults.from_sklearn_search_cv(searcher=self.housing_data__grid_search,
                                                            higher_score_is_better=False,
                                                            description="test description")
        _ = parser.plot_performance_across_trials()
        _ = parser.plot_parameter_values_across_trials()
        _ = parser.plot_parallel_coordinates()
        _ = parser.plot_scatter_matrix()
        # plotly.offline.plot(_, filename=get_test_path() + '/test_files/sklearn_eval/temp.html', auto_open=True)

    def test_TwoClassEvaluator(self):
        y_true = self.credit_data__y_test
        y_score = self.credit_data__y_scores
        score_threshold = 0.5
        y_pred = [1 if x > score_threshold else 0 for x in y_score]
        evaluator = TwoClassEvaluator(actual_values=self.credit_data__y_test,
                                      predicted_scores=self.credit_data__y_scores,
                                      positive_class='Defaulted',
                                      negative_class='Not Defaulted',
                                      score_threshold=0.5)

        con_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)
        self.assertEqual(evaluator._true_negatives, con_matrix[0, 0])
        self.assertEqual(evaluator._false_positives, con_matrix[0, 1])
        self.assertEqual(evaluator._false_negatives, con_matrix[1, 0])
        self.assertEqual(evaluator._true_positives, con_matrix[1, 1])
        self.assertEqual(evaluator.sample_size, len(self.credit_data__y_test))
        self.assertTrue(evaluator.true_positive_rate == evaluator.sensitivity == evaluator.recall)
        self.assertEqual(evaluator.recall, recall_score(y_true=y_true, y_pred=y_pred))
        self.assertEqual(evaluator.true_negative_rate, evaluator.specificity)
        self.assertEqual(round(evaluator.false_positive_rate, 9), round(1 - evaluator.specificity, 9))
        self.assertEqual(evaluator.positive_predictive_value, evaluator.precision)
        self.assertEqual(evaluator.precision, precision_score(y_true=y_true, y_pred=y_pred))
        self.assertEqual(evaluator.f1_score, f1_score(y_true=y_true, y_pred=y_pred))
        self.assertEqual(evaluator.auc, roc_auc_score(y_true=y_true, y_score=y_score))
        self.assertEqual(evaluator.fbeta_score(beta=0), fbeta_score(y_true=y_true, y_pred=y_pred, beta=0))
        self.assertEqual(evaluator.fbeta_score(beta=1), fbeta_score(y_true=y_true, y_pred=y_pred, beta=1))
        self.assertEqual(evaluator.fbeta_score(beta=1), evaluator.f1_score)
        self.assertEqual(evaluator.fbeta_score(beta=2), fbeta_score(y_true=y_true, y_pred=y_pred, beta=2))
        self.assertEqual(round(evaluator.kappa, 9), round(cohen_kappa_score(y1=y_true, y2=y_pred), 9))

        helper_test_dataframe(file_name=get_test_path() + '/test_files/sklearn_eval/get_auc_curve_dataframe.txt',
                              dataframe=evaluator._get_auc_curve_dataframe())

        helper_test_dataframe(file_name=get_test_path() + '/test_files/sklearn_eval/get_threshold_curve_dataframe.txt',
                              dataframe=evaluator._get_threshold_curve_dataframe())

        check_plot(file_name=get_test_path() + '/test_files/sklearn_eval/plot_predicted_scores_histogram.png',
                   plot_function=lambda: evaluator.plot_predicted_scores_histogram())

        check_plot(file_name=get_test_path() + '/test_files/sklearn_eval/plot_actual_vs_predict_histogram.png',
                   plot_function=lambda: evaluator.plot_actual_vs_predict_histogram())

        self.assertIsInstance(evaluator.all_metrics, dict)
        self.assertIsInstance(evaluator.all_metrics_df(return_style=False), pd.DataFrame)

        with open(get_test_path() + '/test_files/sklearn_eval/all_metrics_df.html', 'w') as file:
            table_html = evaluator.all_metrics_df(return_explanations=False,
                                                  dummy_classifier_strategy=None,
                                                  return_style=True).render()
            file.write(clean_formatted_dataframe(table_html))

        with open(get_test_path() + '/test_files/sklearn_eval/all_metrics_df__dummy.html', 'w') as file:
            table_html = evaluator.all_metrics_df(return_explanations=False,
                                                  dummy_classifier_strategy='prior',
                                                  return_style=True).render()
            file.write(clean_formatted_dataframe(table_html))

        with open(get_test_path() + '/test_files/sklearn_eval/all_metrics_df__dummies.html', 'w') as file:
            table_html = evaluator.all_metrics_df(return_explanations=False,
                                                  dummy_classifier_strategy=['prior', 'constant'],
                                                  return_style=True).render()
            file.write(clean_formatted_dataframe(table_html))

        with open(get_test_path() + '/test_files/sklearn_eval/all_metrics_df__round_3.html', 'w') as file:
            table_html = evaluator.all_metrics_df(return_explanations=False,
                                                  return_style=True,
                                                  round_by=3).render()
            file.write(clean_formatted_dataframe(table_html))

        with open(get_test_path() + '/test_files/sklearn_eval/all_metrics_df__with_details.html', 'w') as file:
            table_html = evaluator.all_metrics_df(return_explanations=True, return_style=True).render()
            file.write(clean_formatted_dataframe(table_html))

        with open(get_test_path() + '/test_files/sklearn_eval/all_metrics_df__with_details__round_3.html', 'w') as file:
            table_html = evaluator.all_metrics_df(return_explanations=True,
                                                  return_style=True,
                                                  round_by=3).render()
            file.write(clean_formatted_dataframe(table_html))

    def test_RegressionEvaluator(self):
        evaluator = RegressionEvaluator(actual_values=self.housing_data__y_test,
                                        predicted_values=self.housing_data__y_predictions)

        self.assertEqual(evaluator.mean_absolute_error,
                         mean_absolute_error(y_true=self.housing_data__y_test,
                                             y_pred=self.housing_data__y_predictions))
        self.assertEqual(round(mean_squared_error(y_true=self.housing_data__y_test,
                                                  y_pred=self.housing_data__y_predictions), 4),
                         round(evaluator.mean_squared_error, 4))
        self.assertEqual(np.sqrt(evaluator.mean_squared_error), evaluator.root_mean_squared_error)
        self.assertEqual(evaluator.total_observations, len(self.housing_data__y_test))
        self.assertIsInstance(evaluator.all_metrics, dict)
        self.assertIsInstance(evaluator.all_metrics_df(), pd.DataFrame)

        with open(get_test_path() + '/test_files/sklearn_eval/reg_eval__all_metrics_df.html', 'w') as file:
            table_html = evaluator.all_metrics_df(return_style=True,
                                                  dummy_regressor_strategy=None).render()
            file.write(clean_formatted_dataframe(table_html))

        with open(get_test_path() + '/test_files/sklearn_eval/reg_eval__all_metrics_df__dummy.html', 'w') as file:
            table_html = evaluator.all_metrics_df(return_style=True,
                                                  dummy_regressor_strategy='mean').render()
            file.write(clean_formatted_dataframe(table_html))

        with open(get_test_path() + '/test_files/sklearn_eval/reg_eval__all_metrics_df__dummies.html', 'w') as file:
            table_html = evaluator.all_metrics_df(return_style=True,
                                                  dummy_regressor_strategy=['mean', 'median']).render()
            file.write(clean_formatted_dataframe(table_html))

        with open(get_test_path() + '/test_files/sklearn_eval/reg_eval__all_metrics_df__round_3.html', 'w') as file:
            table_html = evaluator.all_metrics_df(return_style=True, round_by=3).render()
            file.write(clean_formatted_dataframe(table_html))

        with open(get_test_path() + '/test_files/sklearn_eval/reg_eval__all_metrics_df__round_0.html', 'w') as file:
            table_html = evaluator.all_metrics_df(return_style=True, round_by=0).render()
            file.write(clean_formatted_dataframe(table_html))

        check_plot(file_name=get_test_path() + '/test_files/sklearn_eval/reg_eval__plot_residuals_vs_fits.png',
                   plot_function=lambda: evaluator.plot_residuals_vs_fits())

        check_plot(file_name=get_test_path() + '/test_files/sklearn_eval/reg_eval__plot_residuals_vs_actuals.png',
                   plot_function=lambda: evaluator.plot_residuals_vs_actuals())

        check_plot(file_name=get_test_path() + '/test_files/sklearn_eval/reg_eval__plot_predictions_vs_actuals.png',
                   plot_function=lambda: evaluator.plot_predictions_vs_actuals())

    def test_plot_confusion_matrix(self):
        evaluator = TwoClassEvaluator(actual_values=self.credit_data__y_test,
                                      predicted_scores=self.credit_data__y_scores,
                                      positive_class='Defaulted',
                                      negative_class='Not Defaulted',
                                      score_threshold=0.5)

        check_plot(file_name=get_test_path() + '/test_files/sklearn_eval/plot_confusion_matrix.png',
                   plot_function=lambda: evaluator.plot_confusion_matrix())

    def test_plot_auc_curve(self):
        evaluator = TwoClassEvaluator(actual_values=self.credit_data__y_test,
                                      predicted_scores=self.credit_data__y_scores,
                                      positive_class='Defaulted',
                                      negative_class='Not Defaulted',
                                      score_threshold=0.5)

        check_plot(file_name=get_test_path() + '/test_files/sklearn_eval/plot_auc_curve.png',
                   plot_function=lambda: evaluator.plot_auc_curve())

    def test_plot_threshold_curves(self):
        evaluator = TwoClassEvaluator(actual_values=self.credit_data__y_test,
                                      predicted_scores=self.credit_data__y_scores,
                                      positive_class='Defaulted',
                                      negative_class='Not Defaulted',
                                      score_threshold=0.5)

        check_plot(file_name=get_test_path() + '/test_files/sklearn_eval/plot_threshold_curves.png',
                   plot_function=lambda: evaluator.plot_threshold_curves())

    def test_plot_precision_recall_tradeoff(self):
        evaluator = TwoClassEvaluator(actual_values=self.credit_data__y_test,
                                      predicted_scores=self.credit_data__y_scores,
                                      positive_class='Defaulted',
                                      negative_class='Not Defaulted',
                                      score_threshold=0.5)

        check_plot(file_name=get_test_path() + '/test_files/sklearn_eval/plot_precision_recall_tradeoff.png',
                   plot_function=lambda: evaluator.plot_precision_recall_tradeoff())

    def test_calculate_lift_gain(self):
        evaluator = TwoClassEvaluator(actual_values=self.credit_data__y_test,
                                      predicted_scores=self.credit_data__y_scores,
                                      positive_class='Defaulted',
                                      negative_class='Not Defaulted',
                                      score_threshold=0.5)

        helper_test_dataframe(file_name=get_test_path() + '/test_files/sklearn_eval/calculate_lift_gain.txt',
                              dataframe=evaluator.calculate_lift_gain())

        helper_test_dataframe(file_name=get_test_path() + '/test_files/sklearn_eval/calculate_lift_gain__10_buckets.txt',
                              dataframe=evaluator.calculate_lift_gain(num_buckets=10))

        with open(get_test_path() + '/test_files/sklearn_eval/calculate_lift_gain.html', 'w') as file:
            table_html = evaluator.calculate_lift_gain(return_style=True).render()
            file.write(clean_formatted_dataframe(table_html))

        with open(get_test_path() + '/test_files/sklearn_eval/calculate_lift_gain__10_buckets.html', 'w') as file:
            table_html = evaluator.calculate_lift_gain(return_style=True, num_buckets=10).render()
            file.write(clean_formatted_dataframe(table_html))
