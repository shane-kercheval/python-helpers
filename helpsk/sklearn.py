"""This module contains helper functions when working with sklearn (scikit-learn) objects"""
import math

import pandas as pd
import scipy.stats as st
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection._search import BaseSearchCV  # noqa
from sklearn.preprocessing import OrdinalEncoder

import helpsk.color as hcolor
import helpsk.pandas_style as hstyle
# pylint: disable=too-many-locals
from helpsk.exceptions import HelpskParamValueError


def cv_results_to_dataframe(searcher: BaseSearchCV,
                            num_folds: int,
                            num_repeats: int,
                            return_style: bool = True):
    """
    Args:
        searcher:
            A `BaseSearchCV` object that has either used a string passed to the `scoring` parameter of the
            constructor (e.g. `GridSearchCV(..., scoring='roc_auc', ...)` or a dictionary with metric names as
            keys and callables a values.

            An example of the dictionary option:

                scores = {
                    'ROC/AUC': SCORERS['roc_auc'],
                    'F1': make_scorer(f1_score, greater_is_better=True),
                    'Pos. Pred. Val': make_scorer(precision_score, greater_is_better=True),
                    'True Pos. Rate': make_scorer(recall_score, greater_is_better=True),
                }
                grid_search = GridSearchCV(..., scoring=scores, ...)
        num_folds:
            the number of folds used for the cross validation; used to calculate the standard error of the
            mean for each score
        num_repeats:
            the number of repeats used for the cross validation; used to calculate the standard error of the
            mean for each score
        return_style:
            if True, return pd.DataFrame().style object after being styled
    """
    sample_size = num_folds * num_repeats
    results = None
    cv_results = searcher.cv_results_

    str_score_name = None
    if isinstance(searcher.scoring, dict):
        score_names = list(searcher.scoring.keys())
    elif isinstance(searcher.scoring, str):
        score_names = ['score']
        str_score_name = searcher.scoring
    else:
        mess = 'The `searcher` does not have a string or dictionary .scoring property. Cannot extract scores.'
        raise HelpskParamValueError(mess)

    # extract mean and standard deviation for each score
    # if a string was passed into the searcher `scoring` parameter (e.g. 'roc/auc') then the score name will
    # just be 'mean_test_score' but we will convert it to `roc/auc Mean`. Same for standard deviation.
    # In that case, then after we are done, we need to change `score_names = ['score']` to
    # `score_names = [str_score_name]`
    for score in score_names:
        score_name = score if str_score_name is None else str_score_name
        results = pd.concat([
            results,
            pd.DataFrame({
                score_name + " Mean": cv_results['mean_test_' + score],
                score_name + " St. Dev": cv_results['std_test_' + score],
            })
        ], axis=1)

    # see comment above
    if str_score_name is not None:
        score_names = [str_score_name]

    # for each score, calculate the 95% confidence interval for the mean
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

    parameter_dataframe = pd.DataFrame(cv_results["params"])
    parameter_dataframe.columns = [x.replace('__', ' | ') for x in parameter_dataframe.columns]

    results = pd.concat([
        results,
        parameter_dataframe,
    ], axis=1)

    results = results.sort_values(by=str(list(score_names)[0]) + ' Mean', ascending=False)

    if return_style:
        results = results.style

        for score in score_names:
            mean_key = score + ' Mean'
            ci_low_key = score + ' 95CI.LO'
            ci_high_key = score + ' 95CI.HI'

            results. \
                bar(subset=[mean_key], color=hcolor.Colors.PIGMENT_GREEN.value). \
                bar(subset=[ci_high_key], color=hcolor.GRAY). \
                pipe(hstyle.bar_inverse, subset=[ci_low_key], color=hcolor.GRAY). \
                pipe(hstyle.format, round_by=3, hide_index=True)

    return results


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
