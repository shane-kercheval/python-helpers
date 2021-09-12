"""This module contains helper functions when working with sklearn (scikit-learn) objects"""
from typing import Optional
import math
import scipy.stats as st
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder

import helpsk.color as hcolor
import helpsk.pandas_style as hstyle


class TransformerChooser(BaseEstimator, TransformerMixin):
    """Transformer that wraps another Transformer. This allows different transformer objects to be tuned.
    """
    def __init__(self, base_transformer=None):
        """
        Args:
            base_transformer:
                Transformer object (e.g. StandardScaler, MinMaxScaler)
        """
        self.base_transformer = base_transformer

    def fit(self, X, y=None):  # pylint: disable=invalid-name # noqa
        """fit implementation
        """
        if self.base_transformer is None:
            return self

        return self.base_transformer.fit(X, y)

    def transform(self, X):  # pylint: disable=invalid-name # noqa
        """transform implementation
        """
        if self.base_transformer is None:
            return X

        return self.base_transformer.transform(X)


class CustomOrdinalEncoder(BaseEstimator, TransformerMixin):
    """First replaces missing values with '<missing>' then applies OrdinalEncoder
    """

    def __init__(self):
        self._ordinal_encoder = OrdinalEncoder()  # unknown_value=-1,
        # handle_unknown='use_encoded_value')
        self._missing_value = '<missing>'

    def _fill_na(self, X):  # pylint: disable=invalid-name # noqa
        """Helper function that fills missing values with strings before calling OrdinalEncoder"""
        for column in X.columns.values:
            if X[column].dtype.name == 'category':
                if self._missing_value not in X[column].cat.categories:
                    X[column] = X[column].cat.add_categories(self._missing_value)
                X[column] = X[column].fillna(self._missing_value)

        return X

    def fit(self, X, y=None):  # pylint: disable=invalid-name,unused-argument # noqa
        """fit implementation"""
        X = self._fill_na(X)  # pylint: disable=invalid-name # noqa
        self._ordinal_encoder.fit(X)
        return self

    def transform(self, X):  # pylint: disable=invalid-name # noqa
        """transform implementation"""
        X = self._fill_na(X)  # pylint: disable=invalid-name # noqa
        return self._ordinal_encoder.transform(X)


# pylint: disable=too-many-locals
def cv_results_to_dataframe(cv_results: dict,
                            num_folds: int,
                            num_repeats: int,
                            score_names: Optional[list] = None,
                            return_styler: bool = True):
    """

    Args:
        cv_results:
            a dictionary returned by `.cv_results_` property of a `BaseSearchCV` object

            ```
            grid_search = GridSearchCV(...)
            grid_search.fit(...)
            cv_results_to_dataframe(grid_search.cv_results, ...)
            ```
        num_folds:
            the number of folds used for the cross validation; used to calculate the standard error of the
            mean for each score
        num_repeats:
            the number of repeats used for the cross validation; used to calculate the standard error of the
            mean for each score
        return_styler:
            if True, return pd.DataFrame().style object after being styled
        score_names:
            If custom scores are passed to `BaseSearchCV` object's __init__ `scoring` parameter via
            dictionary, then this list should be the keys of that dictionary.
            If a string is passed to `scoring` parameter, then `score_names` should be None

            Example:

            ```
            scores = {
                'ROC/AUC':  SCORERS['roc_auc'],
                'F1': make_scorer(f1_score, greater_is_better=True),
                'Pos. Pred. Val': make_scorer(precision_score, greater_is_better=True),
                'True Pos. Rate': make_scorer(recall_score, greater_is_better=True),
            }
            grid_search = GridSearchCV(full_pipeline,
                                       param_grid=param_grad,
                                       cv=RepeatedKFold(n_splits=5, n_repeats=2),
                                       scoring=scores,
                                       refit='ROC/AUC',
                                       ...
            )
            cv_results_to_dataframe(grid_search.cv_results, score_names=list(scores.keys()))
            ```
    """
    results = None

    if score_names:

        # extract mean and standard deviation for each score
        for score in score_names:
            results = pd.concat([
                results,
                pd.DataFrame({
                    score + " Mean": cv_results['mean_test_' + score],
                    score + " St. Dev": cv_results['std_test_' + score],
                })
            ], axis=1)

        # for each score, calculate the 95% confidence interval for the mean

        sample_size = num_folds * num_repeats

        for score in score_names:
            mean_key = score + ' Mean'
            st_dev_key = score + ' St. Dev'

            score_means = results[mean_key]
            score_standard_errors = results[st_dev_key] / math.sqrt(sample_size)

            confidence_intervals = st.t.interval(alpha=0.95,  # confidence interval
                                                 df=sample_size - 1,  # degrees of freedom
                                                 loc=score_means,
                                                 scale=score_standard_errors)

            results = results.drop(columns=st_dev_key)

            insertion_index = results.columns.get_loc(mean_key) + 1
            results.insert(loc=insertion_index, column=score + ' 95CI.HI', value=confidence_intervals[1])
            results.insert(loc=insertion_index, column=score + ' 95CI.LO', value=confidence_intervals[0])

        results = pd.concat([
            results,
            pd.DataFrame(cv_results["params"]),
        ], axis=1)

        results = results.sort_values(by=str(list(score_names)[0]) + ' Mean', ascending=False)

        if return_styler:
            results = results.style

            for score in score_names:
                mean_key = score + ' Mean'
                ci_low_key = score + ' 95CI.LO'
                ci_high_key = score + ' 95CI.HI'

                results. \
                    bar(subset=[mean_key], color=hcolor.Colors.PIGMENT_GREEN.value). \
                    bar(subset=[ci_high_key], color=hcolor.GRAY). \
                    pipe(hstyle.bar_inverse, subset=[ci_low_key], color=hcolor.GRAY)

            results = hstyle.format(styler=results, round_by=3, hide_index=True)

    else:
        results = None

    return results
