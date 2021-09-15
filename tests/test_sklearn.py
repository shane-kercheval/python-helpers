import unittest
import warnings  # noqa

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score, SCORERS, roc_auc_score, fbeta_score, cohen_kappa_score, \
    confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize, StandardScaler, OneHotEncoder

import helpsk as hlp
from helpsk.sklearn import CustomOrdinalEncoder, cv_results_to_dataframe, TwoClassEvaluator
from tests.helpers import get_data_credit, get_test_path, check_plot


def warn(*args, **kwargs):  # noqa
    pass
warnings.warn = warn  # noqa


# noinspection PyMethodMayBeStatic
class TestSklearn(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
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
            ('encoder_chooser', hlp.sklearn.TransformerChooser()),
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
        # https://github.com/scikit-learn/scikit-learn/blob/2beed55847ee70d363bdbfe14ee4401438fba057/sklearn/metrics/_scorer.py#L702
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html
        # https://stackoverflow.com/questions/60615281/different-result-roc-auc-score-and-plot-roc-curve
        scores = {
            # https://github.com/scikit-learn/scikit-learn/blob/2beed5584/sklearn/metrics/_scorer.py#L537
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

    def test_cv_results_to_dataframe(self):
        grid_search = self.credit_data__grid_search
        results = cv_results_to_dataframe(searcher=grid_search,
                                          num_folds=3,
                                          num_repeats=1,
                                          return_style=False)

        self.assertIsInstance(results, pd.DataFrame)
        self.assertIsInstance(results['preparation | non_numeric_pipeline | encoder_chooser | base_transformer'].iloc[0],
                              str)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = cv_results_to_dataframe(searcher=grid_search,
                                              num_folds=3,
                                              num_repeats=1,
                                              return_style=True)
        with open(get_test_path() + '/test_files/sklearn/credit__grid_search.html', 'w') as file:
            file.write(results.render())

        grid_search = self.credit_data__grid_search__roc_auc
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = cv_results_to_dataframe(searcher=grid_search,
                                              num_folds=3,
                                              num_repeats=1,
                                              return_style=True)
        test_file = get_test_path() + '/test_files/sklearn/credit__grid_search__default_scores.html'
        with open(test_file, 'w') as file:
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
        self.assertTrue(evaluator.true_negative_rate == evaluator.specificity)
        self.assertTrue(evaluator.positive_predictive_value == evaluator.precision)
        self.assertEqual(evaluator.precision, precision_score(y_true=y_true, y_pred=y_pred))
        self.assertEqual(evaluator.f1_score, f1_score(y_true=y_true, y_pred=y_pred))
        self.assertEqual(evaluator.roc_auc, roc_auc_score(y_true=y_true, y_score=y_score))
        self.assertEqual(evaluator.fbeta_score(beta=0), fbeta_score(y_true=y_true, y_pred=y_pred, beta=0))
        self.assertEqual(evaluator.fbeta_score(beta=1), fbeta_score(y_true=y_true, y_pred=y_pred, beta=1))
        self.assertEqual(evaluator.fbeta_score(beta=1), evaluator.f1_score)
        self.assertEqual(evaluator.fbeta_score(beta=2), fbeta_score(y_true=y_true, y_pred=y_pred, beta=2))
        self.assertEqual(round(evaluator.kappa, 9), round(cohen_kappa_score(y1=y_true, y2=y_pred), 9))

        self.assertIsInstance(evaluator.all_metrics, dict)
        self.assertIsInstance(evaluator.all_metrics_df(return_style=False), pd.DataFrame)

        with open(get_test_path() + '/test_files/sklearn/all_metrics_df.html', 'w') as file:
            table_html = evaluator.all_metrics_df(return_style=True).render()
            file.write(table_html)

        with open(get_test_path() + '/test_files/sklearn/all_metrics_df__round_3.html', 'w') as file:
            table_html = evaluator.all_metrics_df(return_style=True,
                                                  round_by=3).render()

            evaluator.all_metrics_df()
            evaluator.all_metrics_df().style.format(precision=3).render()
            file.write(table_html)

    def test_plot_confusion_matrix(self):
        evaluator = TwoClassEvaluator(actual_values=self.credit_data__y_test,
                                      predicted_scores=self.credit_data__y_scores,
                                      labels=('Bad', 'Good'),
                                      score_threshold=0.5)

        check_plot(file_name=get_test_path() + '/test_files/sklearn/plot_confusion_matrix.png',
                   plot_function=lambda: evaluator.plot_confusion_matrix())
