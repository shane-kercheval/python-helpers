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
from helpsk.utility import suppress_warnings
from helpsk.sklearn_eval import cv_results_to_dataframe, TwoClassEvaluator, RegressionEvaluator
from helpsk.sklearn_pipeline import CustomOrdinalEncoder
from tests.helpers import get_data_credit, get_test_path, check_plot, helper_test_dataframe, get_data_housing


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
        y_full = label_binarize(y_full, classes=['bad', 'good']).flatten()
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
                'preparation__non_numeric_pipeline__encoder_chooser__base_transformer': [OneHotEncoder(),
                                                                                         CustomOrdinalEncoder()],
                'model__max_features': [100, 'auto'],
                'model__n_estimators': [10, 50]
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

        grid_search = GridSearchCV(full_pipeline,
                                   param_grid=param_grad,
                                   cv=RepeatedKFold(n_splits=3, n_repeats=1, random_state=42),
                                   scoring='roc_auc',
                                   refit=True,
                                   return_train_score=True)
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

    def test_cv_results_to_dataframe(self):
        grid_search = self.credit_data__grid_search
        results = cv_results_to_dataframe(searcher=grid_search,
                                          num_folds=3,
                                          num_repeats=1,
                                          return_train_score=True,
                                          return_style=False)

        self.assertIsInstance(results, pd.DataFrame)
        self.assertIsInstance(results['preparation | non_numeric_pipeline | encoder_chooser | base_transformer'].iloc[0],
                              str)
        equal = results.columns == ['ROC/AUC Mean', 'ROC/AUC 95CI.LO', 'ROC/AUC 95CI.HI',
                                    'ROC/AUC Training Mean',
                                    'F1 Mean', 'F1 95CI.LO', 'F1 95CI.HI',
                                    'F1 Training Mean',
                                    'Pos. Pred. Val Mean', 'Pos. Pred. Val 95CI.LO', 'Pos. Pred. Val 95CI.HI',
                                    'Pos. Pred. Val Training Mean',
                                    'True Pos. Rate Mean', 'True Pos. Rate 95CI.LO', 'True Pos. Rate 95CI.HI',
                                    'True Pos. Rate Training Mean',
                                    'model | max_features', 'model | n_estimators',
                                    'preparation | non_numeric_pipeline | encoder_chooser | base_transformer']
        self.assertTrue(all(equal))

        results = cv_results_to_dataframe(searcher=grid_search,
                                          num_folds=3,
                                          num_repeats=1,
                                          return_train_score=False,
                                          return_style=False)

        self.assertIsInstance(results, pd.DataFrame)
        self.assertIsInstance(results['preparation | non_numeric_pipeline | encoder_chooser | base_transformer'].iloc[0],
                              str)
        equal = results.columns == ['ROC/AUC Mean', 'ROC/AUC 95CI.LO', 'ROC/AUC 95CI.HI',
                                    'F1 Mean', 'F1 95CI.LO', 'F1 95CI.HI',
                                    'Pos. Pred. Val Mean', 'Pos. Pred. Val 95CI.LO', 'Pos. Pred. Val 95CI.HI',
                                    'True Pos. Rate Mean', 'True Pos. Rate 95CI.LO', 'True Pos. Rate 95CI.HI',
                                    'model | max_features', 'model | n_estimators',
                                    'preparation | non_numeric_pipeline | encoder_chooser | base_transformer']
        self.assertTrue(all(equal))

        with suppress_warnings():
            results = cv_results_to_dataframe(searcher=grid_search,
                                              num_folds=3,
                                              num_repeats=1,
                                              return_train_score=True,
                                              return_style=True)
        with open(get_test_path() + '/test_files/sklearn_eval/credit__grid_search__with_training.html', 'w') as file:
            file.write(results.render())

        with suppress_warnings():
            results = cv_results_to_dataframe(searcher=grid_search,
                                              num_folds=3,
                                              num_repeats=1,
                                              return_train_score=False,
                                              return_style=True)
        with open(get_test_path() + '/test_files/sklearn_eval/credit__grid_search__without_training.html', 'w') as file:
            file.write(results.render())

        grid_search = self.credit_data__grid_search__roc_auc
        with suppress_warnings():
            results = cv_results_to_dataframe(searcher=grid_search,
                                              num_folds=3,
                                              num_repeats=1,
                                              return_train_score=True,
                                              return_style=True)
        test_file = get_test_path() + '/test_files/sklearn_eval/credit__grid_search__default_scores__with_training.html'
        with open(test_file, 'w') as file:
            file.write(results.render())

        with suppress_warnings():
            results = cv_results_to_dataframe(searcher=grid_search,
                                              num_folds=3,
                                              num_repeats=1,
                                              return_train_score=False,
                                              return_style=True)
        test_file = get_test_path() + '/test_files/sklearn_eval/credit__grid_search__default_scores__without_training.html'
        with open(test_file, 'w') as file:
            file.write(results.render())

    def test_cv_results_to_dataframe_regression(self):
        grid_search = self.housing_data__grid_search
        results = cv_results_to_dataframe(searcher=grid_search,
                                          num_folds=3,
                                          num_repeats=1,
                                          return_train_score=True,
                                          return_style=False)

        self.assertIsInstance(results, pd.DataFrame)
        self.assertIsInstance(results['model | max_features'].iloc[0], str)
        equal = results.columns == ['RMSE Mean', 'RMSE 95CI.LO', 'RMSE 95CI.HI', 'RMSE Training Mean',
                                    'MAE Mean', 'MAE 95CI.LO', 'MAE 95CI.HI', 'MAE Training Mean',
                                    'model | max_features', 'model | n_estimators']
        self.assertTrue(all(equal))

        results = cv_results_to_dataframe(searcher=grid_search,
                                          num_folds=3,
                                          num_repeats=1,
                                          return_train_score=False,
                                          return_style=False)

        self.assertIsInstance(results, pd.DataFrame)
        self.assertIsInstance(results['model | max_features'].iloc[0], str)
        equal = results.columns == ['RMSE Mean', 'RMSE 95CI.LO', 'RMSE 95CI.HI',
                                    'MAE Mean', 'MAE 95CI.LO', 'MAE 95CI.HI',
                                    'model | max_features', 'model | n_estimators']
        self.assertTrue(all(equal))

        with suppress_warnings():
            results = cv_results_to_dataframe(searcher=grid_search,
                                              num_folds=3,
                                              num_repeats=1,
                                              greater_is_better=False,
                                              return_train_score=True,
                                              return_style=True)
        with open(get_test_path() + '/test_files/sklearn_eval/housing__grid_search__with_training.html', 'w') as file:
            file.write(results.render())

        with suppress_warnings():
            results = cv_results_to_dataframe(searcher=grid_search,
                                              num_folds=3,
                                              num_repeats=1,
                                              greater_is_better=False,
                                              return_train_score=False,
                                              return_style=True)
        with open(get_test_path() + '/test_files/sklearn_eval/housing__grid_search__without_training.html', 'w') as file:
            file.write(results.render())

    def test_TwoClassEvaluator(self):
        y_true = self.credit_data__y_test
        y_score = self.credit_data__y_scores
        score_threshold = 0.5
        y_pred = [1 if x > score_threshold else 0 for x in y_score]
        evaluator = TwoClassEvaluator(actual_values=self.credit_data__y_test,
                                      predicted_scores=self.credit_data__y_scores,
                                      labels=('Bad', 'Good'),
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

        self.assertIsInstance(evaluator.all_metrics, dict)
        self.assertIsInstance(evaluator.all_metrics_df(return_style=False), pd.DataFrame)

        with open(get_test_path() + '/test_files/sklearn_eval/all_metrics_df.html', 'w') as file:
            table_html = evaluator.all_metrics_df(return_details=False, return_style=True).render()
            file.write(table_html)

        with open(get_test_path() + '/test_files/sklearn_eval/all_metrics_df__round_3.html', 'w') as file:
            table_html = evaluator.all_metrics_df(return_details=False,
                                                  return_style=True,
                                                  round_by=3).render()
            file.write(table_html)

        with open(get_test_path() + '/test_files/sklearn_eval/all_metrics_df__with_details.html', 'w') as file:
            table_html = evaluator.all_metrics_df(return_details=True, return_style=True).render()
            file.write(table_html)

        with open(get_test_path() + '/test_files/sklearn_eval/all_metrics_df__with_details__round_3.html', 'w') as file:
            table_html = evaluator.all_metrics_df(return_details=True,
                                                  return_style=True,
                                                  round_by=3).render()
            file.write(table_html)

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
            table_html = evaluator.all_metrics_df(return_style=True).render()
            file.write(table_html)

        with open(get_test_path() + '/test_files/sklearn_eval/reg_eval__all_metrics_df__round_3.html', 'w') as file:
            table_html = evaluator.all_metrics_df(return_style=True, round_by=3).render()
            file.write(table_html)

        with open(get_test_path() + '/test_files/sklearn_eval/reg_eval__all_metrics_df__round_0.html', 'w') as file:
            table_html = evaluator.all_metrics_df(return_style=True, round_by=0).render()
            file.write(table_html)

        check_plot(file_name=get_test_path() + '/test_files/sklearn_eval/reg_eval__plot_residuals_vs_fits.png',
                   plot_function=lambda: evaluator.plot_residuals_vs_fits())

        check_plot(file_name=get_test_path() + '/test_files/sklearn_eval/reg_eval__plot_residuals_vs_actuals.png',
                   plot_function=lambda: evaluator.plot_residuals_vs_actuals())

        check_plot(file_name=get_test_path() + '/test_files/sklearn_eval/reg_eval__plot_predictions_vs_actuals.png',
                   plot_function=lambda: evaluator.plot_predictions_vs_actuals())

    def test_plot_confusion_matrix(self):
        evaluator = TwoClassEvaluator(actual_values=self.credit_data__y_test,
                                      predicted_scores=self.credit_data__y_scores,
                                      labels=('Bad', 'Good'),
                                      score_threshold=0.5)

        check_plot(file_name=get_test_path() + '/test_files/sklearn_eval/plot_confusion_matrix.png',
                   plot_function=lambda: evaluator.plot_confusion_matrix())

    def test_plot_auc_curve(self):
        evaluator = TwoClassEvaluator(actual_values=self.credit_data__y_test,
                                      predicted_scores=self.credit_data__y_scores,
                                      labels=('Bad', 'Good'),
                                      score_threshold=0.5)

        check_plot(file_name=get_test_path() + '/test_files/sklearn_eval/plot_auc_curve.png',
                   plot_function=lambda: evaluator.plot_auc_curve())

    def test_plot_threshold_curves(self):
        evaluator = TwoClassEvaluator(actual_values=self.credit_data__y_test,
                                      predicted_scores=self.credit_data__y_scores,
                                      labels=('Bad', 'Good'),
                                      score_threshold=0.5)

        check_plot(file_name=get_test_path() + '/test_files/sklearn_eval/plot_threshold_curves.png',
                   plot_function=lambda: evaluator.plot_threshold_curves())

    def test_plot_precision_recall_tradeoff(self):
        evaluator = TwoClassEvaluator(actual_values=self.credit_data__y_test,
                                      predicted_scores=self.credit_data__y_scores,
                                      labels=('Bad', 'Good'),
                                      score_threshold=0.5)

        check_plot(file_name=get_test_path() + '/test_files/sklearn_eval/plot_precision_recall_tradeoff.png',
                   plot_function=lambda: evaluator.plot_precision_recall_tradeoff())

    def test_calculate_lift_gain(self):
        evaluator = TwoClassEvaluator(actual_values=self.credit_data__y_test,
                                      predicted_scores=self.credit_data__y_scores,
                                      labels=('Bad', 'Good'),
                                      score_threshold=0.5)

        helper_test_dataframe(file_name=get_test_path() + '/test_files/sklearn_eval/calculate_lift_gain.txt',
                              dataframe=evaluator.calculate_lift_gain())

        helper_test_dataframe(file_name=get_test_path() + '/test_files/sklearn_eval/calculate_lift_gain__10_buckets.txt',
                              dataframe=evaluator.calculate_lift_gain(num_buckets=10))

        with open(get_test_path() + '/test_files/sklearn_eval/calculate_lift_gain.html', 'w') as file:
            table_html = evaluator.calculate_lift_gain(return_style=True).render()
            file.write(table_html)

        with open(get_test_path() + '/test_files/sklearn_eval/calculate_lift_gain__10_buckets.html', 'w') as file:
            table_html = evaluator.calculate_lift_gain(return_style=True, num_buckets=10).render()
            file.write(table_html)
