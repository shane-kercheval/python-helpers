import unittest
import warnings  # noqa

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score, SCORERS
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize, StandardScaler, OneHotEncoder

import helpsk as hlp
from helpsk.sklearn import CustomOrdinalEncoder, cv_results_to_dataframe
from tests.helpers import get_data_credit, get_test_path


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
        X_full = credit_data.drop(columns='target')
        y_full = label_binarize(y_full, classes=['bad', 'good']).flatten()
        X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=42)
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
        random_forest_model = RandomForestClassifier()
        full_pipeline = Pipeline([
            ('preparation', transformations_pipeline),
            ('model', random_forest_model)
        ])
        param_grad = [
            {
                'preparation__non_numeric_pipeline__encoder_chooser__base_transformer': [OneHotEncoder(),
                                                                                         CustomOrdinalEncoder()],
                'model__max_features': [2, 100, 'auto'],
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

    def test_plot_value_frequency(self):
        grid_search = self.credit_data__grid_search
        score_names = ['ROC/AUC', 'F1', 'Pos. Pred. Val', 'True Pos. Rate']

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = cv_results_to_dataframe(cv_results=grid_search.cv_results_,
                                              num_folds=3,
                                              num_repeats=1,
                                              score_names=score_names,
                                              return_styler=True)
        with open(get_test_path() + '/test_files/sklearn/credit__grid_search.html', 'w') as file:
            file.write(results.render())
