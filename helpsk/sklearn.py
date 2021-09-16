"""This module contains helper functions when working with sklearn (scikit-learn) objects"""
import math
from typing import Tuple, Union, Optional

import numpy as np
import pandas as pd
import scipy.stats as st
import seaborn as sns
from matplotlib import pyplot as plt
from pandas.io.formats.style import Styler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection._search import BaseSearchCV  # noqa
from sklearn.preprocessing import OrdinalEncoder

import helpsk.color as hcolor
import helpsk.pandas_style as hstyle
# pylint: disable=too-many-locals
from helpsk.exceptions import HelpskParamValueError


def cv_results_to_dataframe(searcher: BaseSearchCV,
                            num_folds: int,
                            num_repeats: int,
                            return_style: bool = True) -> Union[pd.DataFrame, Styler]:
    """
    Args:
        searcher:
            A `BaseSearchCV` object that has either used a string passed to the `scoring` parameter of the
            constructor (e.g. `GridSearchCV(..., scoring='auc', ...)` or a dictionary with metric names as
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

    for column in parameter_dataframe.columns:
        # this will convert objects (like Transformers) into strings, which makes further operations on the
        # column (e.g. aggregations/group_by/etc) much easier.
        if parameter_dataframe[column].dtype.name == 'object':
            parameter_dataframe[column] = parameter_dataframe[column].astype(str)

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


# pylint: disable=too-many-instance-attributes,too-many-public-methods
class TwoClassEvaluator:
    """This class calculates various metrics for Two Class (i.e. 0's/1's) prediction scenarios."""
    def __init__(self,
                 actual_values: np.ndarray,
                 predicted_scores: np.ndarray,
                 labels: Tuple[str, str],
                 score_threshold: float = 0.5
                 ):
        """
        Args:
            actual_values:
                array of 0's and 1's
            predicted_scores:
                array of values from `predict_proba()`; NOT the actual labels
            labels:
                tuple containing the label of the negative class in the first index and the positive class
                in the second index
            score_threshold:
                the score/probability threshold for turning scores into 0's and 1's and corresponding labels
        """
        if not all(np.unique(actual_values) == [0, 1]):
            message = f"Values of `actual_values` should 0 or 1. Found `{np.unique(actual_values)}`"
            raise HelpskParamValueError(message)

        if not all(np.logical_and(predicted_scores >= 0, predicted_scores <= 1)):
            message = "Values of `predicted_scores` should be between 0 and 1."
            raise HelpskParamValueError(message)

        if not any(np.logical_and(predicted_scores > 0, predicted_scores < 1)):
            message = "Values of `predicted_scores` should not all be 0's and 1's."
            raise HelpskParamValueError(message)

        self._labels = labels
        self._actual_values = actual_values
        self._predicted_scores = predicted_scores
        self.score_threshold = score_threshold
        predicted_values = np.where(predicted_scores > self.score_threshold, 1, 0)
        self._confusion_matrix = confusion_matrix(y_true=actual_values, y_pred=predicted_values)

        self.sample_size = len(actual_values)
        assert self.sample_size == self._confusion_matrix.sum()

        true_negatives, false_positives, false_negatives, true_positives = self._confusion_matrix.ravel()

        self._actual_positives = true_positives + false_negatives
        assert self._actual_positives == sum(self._actual_values == 1)

        self._actual_negatives = true_negatives + false_positives

        self._true_negatives = true_negatives
        self._false_positives = false_positives
        self._false_negatives = false_negatives
        self._true_positives = true_positives

        self.auc = roc_auc_score(y_true=actual_values, y_score=predicted_scores)

    @property
    def true_positive_rate(self) -> float:
        """True Positive Rate"""
        return 0 if self._actual_positives == 0 else self._true_positives / self._actual_positives

    @property
    def true_negative_rate(self) -> float:
        """True Negative Rate i.e. Specificity"""
        return 0 if self._actual_negatives == 0 else self._true_negatives / self._actual_negatives

    @property
    def false_negative_rate(self) -> float:
        """False Negative Rate"""
        return 0 if self._actual_positives == 0 else self._false_negatives / self._actual_positives

    @property
    def false_positive_rate(self) -> float:
        """False Positive Rate"""
        return 0 if self._actual_negatives == 0 else self._false_positives / self._actual_negatives

    @property
    def accuracy(self) -> Union[float, None]:
        """accuracy"""
        return None if self.sample_size == 0 else \
            (self._true_negatives + self._true_positives) / self.sample_size

    @property
    def error_rate(self) -> Union[float, None]:
        """error_rate"""
        return None if self.sample_size == 0 else \
            (self._false_positives + self._false_negatives) / self.sample_size

    @property
    def positive_predictive_value(self) -> float:
        """Positive Predictive Value i.e. Precision"""
        return 0 if (self._true_positives + self._false_positives) == 0 else \
            self._true_positives / (self._true_positives + self._false_positives)

    @property
    def negative_predictive_value(self) -> float:
        """Negative Predictive Value"""
        return 0 if (self._true_negatives + self._false_negatives) == 0 else \
            self._true_negatives / (self._true_negatives + self._false_negatives)

    @property
    def prevalence(self) -> Union[float, None]:
        """Prevalence"""
        return None if self.sample_size == 0 else \
            (self._true_positives + self._false_negatives) / self.sample_size

    @property
    def kappa(self) -> Union[float, None]:
        """Kappa"""
        if self.sample_size == 0 or \
                ((self._true_negatives + self._false_negatives) / self.sample_size) == 0:
            return None
        # proportion of the actual agreements
        # add the proportion of all instances where the predicted type and actual type agree
        pr_a = (self._true_negatives + self._true_positives) / self.sample_size
        # probability of both predicted and actual being negative
        p_negative_prediction_and_actual = \
            ((self._true_negatives + self._false_positives) / self.sample_size) * \
            ((self._true_negatives + self._false_negatives) / self.sample_size)
        # probability of both predicted and actual being positive
        p_positive_prediction_and_actual = \
            self.prevalence * ((self._false_positives + self._true_positives) / self.sample_size)
        # probability that chance alone would lead the predicted and actual values to match, under the
        # assumption that both are selected randomly (i.e. implies independence) according to the observed
        # proportions (probability of independent events = P(A & B) == P(A) * P(B)
        pr_e = p_negative_prediction_and_actual + p_positive_prediction_and_actual
        return (pr_a - pr_e) / (1 - pr_e)

    @property
    def f1_score(self) -> float:
        """F1 Score
        https://en.wikipedia.org/wiki/F-score
        """
        return self.fbeta_score(beta=1)

    def fbeta_score(self, beta: float) -> float:
        """
        :param beta: The `beta` parameter determines the weight of precision in the combined score.
            `beta < 1` lends more weight to precision (i.e. positive predictive value), while
            `beta > 1` favors recall (i.e. true positive rate)
            (`beta -> 0` considers only precision, `beta -> inf` only recall).
            http://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html
        :return:
        """
        if self.positive_predictive_value is None or self.sensitivity is None or \
                (self.positive_predictive_value + self.sensitivity) == 0:
            return 0

        return (1 + (beta ** 2)) * (self.positive_predictive_value * self.sensitivity) / \
               (((beta ** 2) * self.positive_predictive_value) + self.sensitivity)

    @property
    def sensitivity(self) -> float:
        """Sensitivity i.e. True Positive Rate"""
        return self.true_positive_rate

    @property
    def specificity(self) -> float:
        """Specificity i.e. True Negative Rate"""
        return self.true_negative_rate

    @property
    def precision(self) -> float:
        """Precision i.e. Positive Predictive Value"""
        return self.positive_predictive_value

    @property
    def recall(self):
        """Recall i.e. True Positive Rate"""
        return self.true_positive_rate

    @property
    def all_metrics(self) -> dict:
        """All of the metrics are returned as a dictionary."""
        return {'AUC': self.auc,
                'True Positive Rate': self.sensitivity,
                'True Negative Rate': self.specificity,
                'False Positive Rate': self.false_positive_rate,
                'False Negative Rate': self.false_negative_rate,
                'Positive Predictive Value': self.positive_predictive_value,
                'Negative Predictive Value': self.negative_predictive_value,
                'F1 Score': self.f1_score,
                'Kappa': self.kappa,
                'Two-Class Accuracy': self.accuracy,
                'Error Rate': self.error_rate,
                'Prevalence': self.prevalence,
                'No Information Rate': max(self.prevalence, 1 - self.prevalence),  # i.e. largest class %
                'Total Observations': self.sample_size}

    def all_metrics_df(self,
                       return_style: bool = False,
                       round_by: Optional[int] = None) -> Union[pd.DataFrame, Styler]:
        """All of the metrics are returned as a DataFrame.

        Args:
            return_style:
                if True, return styler object; else return dataframe
            round_by:
                the number of digits to round by; if None, then don't round
        """
        result = pd.DataFrame.from_dict(self.all_metrics, orient='index', columns=['Scores'])

        if round_by:
            result['Scores'] = result['Scores'].round(round_by)

        if return_style:
            subset_scores = [x for x in result.index.values if x != 'Total Observations']

            subset_scores = pd.IndexSlice[result.loc[subset_scores, :].index, :]
            subset_negative_bad = pd.IndexSlice[result.loc[['False Positive Rate',
                                                            'False Negative Rate'], :].index, :]
            subset_secondary = pd.IndexSlice[result.loc[['Two-Class Accuracy', 'Error Rate',
                                                         'Prevalence', 'No Information Rate'], :].index, :]
            subset_total_observations = pd.IndexSlice[result.loc[['Total Observations'], :].index, :]

            result = result.style

            if round_by:
                result = result.format(precision=round_by)

            result = result.format(subset=subset_total_observations,
                                   thousands=',',
                                   precision=0)

            result = result. \
                bar(subset=subset_scores, color=hcolor.Colors.PIGMENT_GREEN.value, vmin=0, vmax=1). \
                bar(subset=subset_negative_bad, color=hcolor.Colors.POPPY.value, vmin=0, vmax=1). \
                bar(subset=subset_secondary, color=hcolor.GRAY, vmin=0, vmax=1)

        return result

    def plot_confusion_matrix(self):
        """Plots a heatmap of the confusion matrix."""
        labels = np.array([[f'True Negatives\n{self._true_negatives}\n{self._true_negatives / self.sample_size:.1%}',  # pylint: disable=line-too-long  # noqa
                            f'False Positives\n{self._false_positives}\n{self._false_positives / self.sample_size:.1%}'],  # pylint: disable=line-too-long  # noqa
                           [f'False Negatives\n{self._false_negatives}\n{self._false_negatives / self.sample_size:.1%}',  # pylint: disable=line-too-long  # noqa
                            f'True Positives\n{self._true_positives}\n{self._true_positives / self.sample_size:.1%}']])  # pylint: disable=line-too-long  # noqa

        axis = plt.subplot()
        sns.heatmap(self._confusion_matrix, annot=labels, cmap='Blues', ax=axis, fmt='')
        # labels, title and ticks
        axis.set_xlabel('Predicted')
        axis.set_ylabel('Actual')
        # axis.set_title('Confusion Matrix');
        axis.xaxis.set_ticklabels([self._labels[0], self._labels[1]])
        axis.yaxis.set_ticklabels([self._labels[0], self._labels[1]])
        plt.tight_layout()

    def plot_auc_curve(self):
        """Plots the ROC AUC"""
        def get_true_pos_false_pos(threshold):
            temp_eval = TwoClassEvaluator(actual_values=self._actual_values,
                                          predicted_scores=self._predicted_scores,
                                          labels=('x', 'y'),
                                          score_threshold=threshold)

            return threshold, temp_eval.true_positive_rate, temp_eval.false_positive_rate

        auc_curve = [get_true_pos_false_pos(threshold=x) for x in np.arange(0.0, 1.01, 0.01)]
        auc_curve = pd.DataFrame(auc_curve,
                                 columns=['threshold', 'True Positive Rate', 'False Positive Rate'])

        axis = sns.lineplot(data=auc_curve, x='False Positive Rate', y='True Positive Rate', ci=None)
        axis.set_title(f"AUC: {round(self.auc, 3)}")
        for i, (x, y, s) in enumerate(zip(auc_curve['False Positive Rate'],  # pylint: disable=invalid-name
                                          auc_curve['True Positive Rate'],
                                          auc_curve['threshold'])):
            if i % 5 == 0:
                axis.text(x, y, f'{s:.3}')
        axis.set_xticks(np.arange(0, 1.1, .1))
        axis.set_yticks(np.arange(0, 1.1, .1))
        plt.grid()
        plt.tight_layout()

    def plot_threshold_curves(self, score_threshold_range: Tuple[int, int] = (0.3, 0.9)):
        """Plots various scores (e.g. True Positive Rate, False Positive Rate, etc.) for various score
        thresholds. (A score threshold is the value for which you would predict a positive label if the
        value of the score is above the threshold (e.g. usually 0.5).

        Args:
            score_threshold_range:
                range of score thresholds to plot (x-axis); tuple with minimum threshold in first index and
                maximum threshold in second index.
        """
        def get_threshold_scores(threshold):
            temp_eval = TwoClassEvaluator(actual_values=self._actual_values,
                                          predicted_scores=self._predicted_scores,
                                          labels=('x', 'y'),
                                          score_threshold=threshold)

            return threshold,\
                temp_eval.true_positive_rate,\
                temp_eval.false_positive_rate,\
                temp_eval.positive_predictive_value,\
                temp_eval.false_negative_rate,\
                temp_eval.true_negative_rate

        threshold_curves = [get_threshold_scores(threshold=x) for x in np.arange(score_threshold_range[0],
                                                                                 score_threshold_range[1],
                                                                                 0.025)]
        threshold_curves = pd.DataFrame(threshold_curves,
                                        columns=['Score Threshold',
                                                 'True Pos. Rate (Recall)',
                                                 'False Pos. Rate',
                                                 'Pos. Predictive Value (Precision)',
                                                 'False Neg. Rate',
                                                 'True Neg. Rate (Specificity)'])

        axis = sns.lineplot(x='Score Threshold', y='value', hue='variable',
                            data=pd.melt(frame=threshold_curves, id_vars=['Score Threshold']))
        axis.set_xticks(np.arange(score_threshold_range[0], score_threshold_range[1] + 0.1, 0.1))
        axis.set_yticks(np.arange(0, 1.1, .1))
        plt.vlines(x=0.5, ymin=0, ymax=1, colors='black')
        plt.grid()
        plt.tight_layout()

    def calculate_lift_gain(self,
                            num_buckets: int = 20,
                            return_style: bool = False,
                            include_all_info: bool = False) -> Union[pd.DataFrame, Styler]:
        """
        https://www.listendata.com/2014/08/excel-template-gain-and-lift-charts.html

        Gain is the % of positive (actual) events we have 'captured' i.e. located by looking at the
        top x% of predicted scores, such that the highest scores are looked at first.
        For example, if the percentile is `5%` and the gain value is `0.3`, we can say that if we randomly
        searched `5%` of the data, we would expect to uncover about `5%` of the positive events;
        however, we have uncovered 30% of events by searching the highest 5% of scores.
        Lift is simply the ratio of the percent of events that what we have uncovered for a given percentile
        of data (i.e. gain) divided by what we would have expected by random chance (i.e. the percentile).
        So in the previous example, we uncovered 30% of positive events by searching the top 5% of scores;
        whereas if we took a random sample, we would have only expected to find 5% of the positive events.
        The lift is `0.3 / 0.05` which is `6`; meaning we found `6` times the amount of positive events by
        searching the top 5% of scores, than if we were to randomly sample the data.
        """
        data = pd.DataFrame({
            'predicted_scores': self._predicted_scores,
            'actual_values': self._actual_values,
        })
        data.sort_values(['predicted_scores'], ascending=False, inplace=True)

        # .qcut gets percentiles
        bins = pd.qcut(x=data['predicted_scores'],
                       q=num_buckets,
                       labels=list(range(100, 0, round(-100 / num_buckets))))

        data['Percentile'] = bins

        def gain_grouping(group):
            results = {
                '# of Obs.': len(group.actual_values),
                '# of Pos. Events': sum(group.actual_values == 1)
            }
            return pd.Series(results, index=['# of Obs.', '# of Pos. Events'])

        gain_lift_data = data.groupby('Percentile').apply(gain_grouping)
        temp = pd.DataFrame({'# of Obs.': 0, '# of Pos. Events': 0}, index=[0])
        temp.index.names = ['Percentile']
        gain_lift_data = pd.concat([gain_lift_data, temp])
        gain_lift_data.sort_index(ascending=True, inplace=True)

        gain_lift_data['Cumul. Pos. Events'] = gain_lift_data['# of Pos. Events'].cumsum()
        gain_lift_data['Gain'] = gain_lift_data['Cumul. Pos. Events'] / self._actual_positives
        gain_lift_data = gain_lift_data.loc[~(gain_lift_data.index == 0), :]
        gain_lift_data['Lift'] = gain_lift_data['Gain'] / (gain_lift_data.index.values / 100)

        if not include_all_info:
            gain_lift_data = gain_lift_data[['Gain', 'Lift']]

        gain_lift_data = gain_lift_data.round(2)

        if return_style:
            gain_lift_data = gain_lift_data.style
            gain_lift_data.format(precision=2). \
                bar(subset='Gain', color=hcolor.Colors.PASTEL_BLUE.value,
                    vmin=0, vmax=1). \
                bar(subset='Lift', color=hcolor.Colors.PASTEL_BLUE.value)

        return gain_lift_data


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
