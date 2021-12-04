"""This module contains helper functions when working with sklearn (scikit-learn) objects;
in particular, for evaluating models"""
# pylint: disable=too-many-lines
import math
import warnings
from typing import Tuple, Union, Optional

import numpy as np
import pandas as pd
import scipy.stats as st
import seaborn as sns
from matplotlib import pyplot as plt
from pandas.io.formats.style import Styler
from sklearn.metrics import confusion_matrix, roc_auc_score, r2_score
from sklearn.model_selection._search import BaseSearchCV  # noqa

import helpsk.color as hcolor
import helpsk.pandas_style as hstyle
import helpsk.string as hstring
# pylint: disable=too-many-locals
from helpsk.exceptions import HelpskParamValueError
from helpsk.plot import STANDARD_WIDTH_HEIGHT
from helpsk.validation import assert_is_close, assert_true

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=Warning)
    from statsmodels import api as sm  # https://github.com/statsmodels/statsmodels/issues/3814


# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-public-methods
class SearchCVParser:
    """
    This class contains the logic to parse and extract information from a BaseSearchCV object (e.g.
    GridSearchCV, RandomizedSearchCV, BayesSearchCV)
    """
    def __init__(self,
                 searcher: BaseSearchCV,
                 higher_score_is_better: bool = True,
                 new_param_column_names: Union[dict, None] = None):

        self._higher_score_is_better = higher_score_is_better
        self.num_repeats = searcher.cv.n_repeats
        self.num_folds = searcher.cv.cvargs['n_splits']

        self._results, self._primary_score_standard_errors = \
            SearchCVParser.cv_results_to_dataframe(searcher=searcher,
                                                   higher_score_is_better=higher_score_is_better)

        if new_param_column_names:
            self._results.rename(columns=new_param_column_names, inplace=True)

        # mean/std times per trial; i.e. to get total time, multiple by number_of_runs_per_trial
        self.mean_fit_time = searcher.cv_results_['mean_fit_time']
        self.std_fit_time = searcher.cv_results_['std_fit_time']
        self.mean_score_time = searcher.cv_results_['mean_score_time']
        self.std_score_time = searcher.cv_results_['std_score_time']

        assert_true(len(self.mean_fit_time) == self.number_of_trials)
        assert_is_close(self.total_time, self.mean_time_per_trial * self.number_of_trials)

    @property
    def number_of_trials(self) -> int:
        """"A single trial contains the cross validation runs for a single set of hyper-parameters. The
        'number of trials' is basically the number of combinations of different hyper-parameters that were
        cross validated."""
        return self.results.shape[0]

    @property
    def number_of_runs_per_trial(self) -> int:
        """This is the number of CV folds multiplied by the number of CV repeats."""
        return self.num_folds * self.num_repeats

    @property
    def fit_time_per_trial(self) -> np.array:
        """Average fit time for each trial multiplied by the number of runs per trial

        Returns:
            array containing the fit time for each trail
        """
        return self.mean_fit_time * self.number_of_runs_per_trial

    @property
    def fit_time_total(self) -> float:
        """Total fit time across all trials."""
        return float(np.sum(self.fit_time_per_trial))

    @property
    def score_time_per_trial(self):
        """Average score time for each trial multiplied by the number of runs per trial

        Returns:
            array containing the score time for each trail
        """
        return self.mean_score_time * self.number_of_runs_per_trial

    @property
    def score_time_total(self) -> float:
        """Total score time across all trials."""
        return float(np.sum(self.score_time_per_trial))

    @property
    def mean_time_per_trial(self) -> float:
        """Average time per trial"""
        return float(np.mean(self.fit_time_per_trial + self.score_time_per_trial))

    @property
    def total_time(self) -> float:
        """Total time it took across all trials"""
        return self.fit_time_total + self.score_time_total

    @property
    def score_names(self) -> list:
        """Returns a list of the names of the scores"""
        def replace_type(value):
            return value.replace(' Mean', '').replace(' 95CI.LO', '').replace(' 95CI.HI', '')
        names = [replace_type(x) for x in self.score_columns]
        # use .fromkeys to dedupe
        return list(dict.fromkeys(names))
    
    @property
    def trial_labels(self) -> pd.Series:
        """A trial is a set of hyper-parameters that were cross validated. (Each row of the `results`
         DataFrame corresponds to a single trial. The corresponding label for each trial is a single string
         containing all of the hyper-parameter names and values in the format of
         `{param1: value1, param2: value2}`.

        Returns:
            a pd.Series the same length as `number_of_trials` containing a str
        """
        def create_hyper_param_labels(data_row) -> list:
            """Creates a list of strings that represent the name/value pair for each hyper-parameter."""
            return [f"{x}: {data_row[x]}" for x in self.parameter_columns]

        def create_trial_label(data_row) -> str:
            return f"{{{hstring.collapse(create_hyper_param_labels(data_row), separate=', ')}}}"

        return self.results.apply(create_trial_label, axis=1)

    @property
    def number_of_scores(self) -> int:
        """The number of scores passed to the SearchCV object"""
        return len(self.score_names)

    @property
    def primary_score_name(self) -> str:
        """The first scorer passed to the SearchCV will be treated as the primary score."""
        return self.score_names[0]

    @property
    def primary_score_standard_errors(self) -> pd.Series:
        """The standard errors associated with the scores. Sorted to correspond to `results` DataFrame."""
        return self._primary_score_standard_errors

    @property
    def top_score(self):
        """The top 'primary' score (i.e. the average cross validation score associated with the best set of
        hyper-params."""
        return self.results.iloc[0, 0]

    @property
    def result_indexes_within_1_standard_error(self):
        """Returns the indexes of the `results` DataFrame where the primary scores (i.e. first scorer
        passed to SearchCV object; i.e. first column of the results DataFrame) are within 1 standard error of
        the highest primary score."""
        if self._higher_score_is_better:
            return self.results.index[self.results.iloc[:, 0] >=
                                      self.top_score - self.top_score_standard_error]

        return self.results.index[self.results.iloc[:, 0] <= self.top_score + self.top_score_standard_error]

    @property
    def top_score_standard_error(self) -> float:
        """The standard error associated with the top 'primary' score."""
        return self.primary_score_standard_errors.iloc[0]  # index of highest score

    @property
    def score_columns(self) -> list:
        """Returns a list of the names of the score columns"""
        return [x for x in self.results.columns[self.results.columns.str.endswith((' Mean',
                                                                                   ' 95CI.LO',
                                                                                   ' 95CI.HI'))]
                if x not in self.training_score_columns]

    @property
    def training_score_columns(self) -> list:
        """Returns a list of the names of the training score columns"""
        return self.results.columns[self.results.columns.str.contains(' Training ')].tolist()

    @property
    def parameter_columns(self) -> list:
        """Returns a list of the names of the columns associated with the hyper-parameters"""
        return [x for x in self.results.columns
                if x not in self.score_columns + self.training_score_columns]

    @property
    def results(self):
        """Main DataFrame summarizing the results of the SearchCV object."""
        return self._results.copy()

    def formatted_results(self,
                          round_by: int = 3,
                          return_train_score: bool = True,
                          exclude_no_variance_params: bool = True,
                          return_style: bool = True) -> Union[pd.DataFrame, Styler]:
        """Returns a summary of the results, either as a Data.Frame, or Data.Frame.Styler.

        If a Styler is returned, then Hyper-Parameter columns will be highlighted in blue where the primary
        score (i.e. first column) for the iteration (i.e. the row i.e. the combination of hyper-parameters
        that were cross validated) is within 1 standard error of the top primary score (i.e. first column
        first row).

        Args:
            round_by:
                the number of digits to round by for the score columns (does not round the parameter columns)
            return_train_score:
                if True, return the columns associated with the training scores, if any.
            exclude_no_variance_params
                if True, exclude columns that only have 1 unique value
            return_style:
                If True, return Styler object

        Returns:
            either a DataFrame or Styler object.
        """
        results = self.results

        if exclude_no_variance_params:
            columns_to_drop = [x for x in self.parameter_columns if len(self.results[x].unique()) == 1]
            results = results.drop(columns=columns_to_drop)

        results = results.round(dict(zip(self.score_columns + self.training_score_columns,
                                         [round_by] * len(self.score_columns + self.training_score_columns))))

        if not return_train_score and len(self.training_score_columns) > 0:
            results = results.drop(columns=self.training_score_columns)

        final_columns = results.columns  # save for style logic

        if return_style:
            results = results.style

            # note that we are not adding styles to the training scores; training scores are secondary info
            for score in self.score_names:
                mean_key = score + ' Mean'
                ci_low_key = score + ' 95CI.LO'
                ci_high_key = score + ' 95CI.HI'

                results. \
                    bar(subset=[mean_key], color=hcolor.Colors.PIGMENT_GREEN.value). \
                    bar(subset=[ci_high_key], color=hcolor.GRAY). \
                    pipe(hstyle.bar_inverse, subset=[ci_low_key], color=hcolor.GRAY). \
                    pipe(hstyle.format, round_by=round_by, hide_index=True)

            # highlight iterations whose primary score (i.e. first column of `results` dataframe) is within
            # 1 standard error of the top primary score (i.e. first column first row).
            def highlight_cols(s):
                return 'background-color: %s' % hcolor.Colors.PASTEL_BLUE.value

            # we might have removed columns (e.g. that don't have any variance) so check that the columns
            # were in the final set
            columns_to_highlight = [x for x in self.parameter_columns if x in final_columns]
            results.applymap(highlight_cols,
                             subset=pd.IndexSlice[self.result_indexes_within_1_standard_error,
                                                  columns_to_highlight])

        return results

    # pylint: disable=too-many-statements
    @staticmethod
    def cv_results_to_dataframe(searcher: BaseSearchCV,
                                higher_score_is_better: bool = True) -> tuple[pd.DataFrame, pd.Series]:
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
            higher_score_is_better:
                if True, higher scores are better; if False, lower scores are better
                False assumes that the scores returned from sklearn are negative and will multiple the values
                by 1.

        Returns:
            DataFrame with a row corresponding to each hyper-parameter combination a.k.a 'trial'
        """
        num_folds = searcher.cv.cvargs['n_splits']
        num_repeats = searcher.cv.n_repeats
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
            mess = 'The `searcher` does not have a string or dictionary .scoring property. Cannot extract ' \
                   'scores.'
            raise HelpskParamValueError(mess)

        # extract mean and standard deviation for each score
        # if a string was passed into the searcher `scoring` parameter (e.g. 'roc/auc') then the score name
        # will just be 'mean_test_score' but we will convert it to `roc/auc Mean`. Same for standard
        # deviation. In that case, then after we are done, we need to change `score_names = ['score']` to
        # `score_names = [str_score_name]`
        for score in score_names:
            score_name = score if str_score_name is None else str_score_name
            mean_scores = cv_results['mean_test_' + score]
            mean_scores = mean_scores if higher_score_is_better else mean_scores * -1

            results = pd.concat([
                results,
                pd.DataFrame({
                    score_name + " Mean": mean_scores,
                    score_name + " St. Dev": cv_results['std_test_' + score],
                })
            ], axis=1)

            if 'mean_train_' + score in cv_results:
                mean_training_scores = cv_results['mean_train_' + score]
                mean_training_scores = mean_training_scores \
                    if higher_score_is_better else mean_training_scores * -1
                results = pd.concat([
                    results,
                    pd.DataFrame({
                        score_name + " Training Mean": mean_training_scores,
                        score_name + " Training St. Dev": cv_results['std_train_' + score],
                    })
                ], axis=1)

        # see comment above
        if str_score_name is not None:
            score_names = [str_score_name]

        def add_confidence_interval(score_name, dataframe):  # noqa
            mean_column_name = score_name + ' Mean'
            st_dev_column_name = score_name + ' St. Dev'
            score_means = dataframe[mean_column_name]
            score_standard_errors = dataframe[st_dev_column_name] / math.sqrt(sample_size)

            confidence_intervals = st.t.interval(alpha=0.95,  # confidence interval
                                                 df=sample_size - 1,  # degrees of freedom
                                                 loc=score_means,
                                                 scale=score_standard_errors)

            dataframe = dataframe.drop(columns=st_dev_column_name)

            insertion_index = dataframe.columns.get_loc(mean_column_name) + 1
            dataframe.insert(loc=insertion_index, column=score_name + ' 95CI.HI',
                             value=confidence_intervals[1])
            dataframe.insert(loc=insertion_index, column=score_name + ' 95CI.LO',
                             value=confidence_intervals[0])

            return dataframe, score_standard_errors

        primary_score_standard_errors = None
        # for each score, calculate the 95% confidence interval for the mean
        for score in score_names:
            (results, standard_errors) = add_confidence_interval(score_name=score,
                                                                 dataframe=results)

            if score == score_names[0]:  # if score is the first/primary score (or only)
                primary_score_standard_errors = standard_errors

            results[f'{score} Mean'] = results[f'{score} Mean']
            results[f'{score} 95CI.LO'] = results[f'{score} 95CI.LO']
            results[f'{score} 95CI.HI'] = results[f'{score} 95CI.HI']

            # if there are training scores, then drop the St. Dev, and don't add confidence intervals,
            # which would add too many columns (noise) and not that interesting.
            if score + ' Training Mean' in results.columns:
                results[f'{score} Training Mean'] = results[f'{score} Training Mean']
                results = results.drop(columns=score + ' Training St. Dev')

        assert_true(primary_score_standard_errors is not None)

        parameter_dataframe = pd.DataFrame(cv_results["params"])
        parameter_dataframe.columns = [x.replace('__', ' | ') for x in parameter_dataframe.columns]

        for column in parameter_dataframe.columns:
            # this will convert objects (like Transformers) into strings, which makes further operations on
            # the column (e.g. aggregations/group_by/etc) much easier.
            if parameter_dataframe[column].dtype.name == 'object':
                parameter_dataframe[column] = parameter_dataframe[column].astype(str)

        results = pd.concat([
            results,
            parameter_dataframe,
        ], axis=1)

        results = results.sort_values(by=str(list(score_names)[0]) + ' Mean',
                                      ascending=not higher_score_is_better)

        primary_score_standard_errors = primary_score_standard_errors.reindex(results.index)

        return results, primary_score_standard_errors


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

        assert len(actual_values) == len(predicted_scores)

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
            self._actual_positives / self.sample_size

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

        auc_message = 'Area under the ROC curve (true pos. rate vs false pos. rate); ' \
                      'ranges from 0.5 (purely random classifier) to 1.0 (perfect classifier)'

        tpr_message = f'{self.true_positive_rate:.1%} of positive instances were correctly identified.; ' \
                      f'i.e. {self._true_positives} "{self._labels[1]}" labels were correctly identified ' \
                      f'out of {self._actual_positives} instances; a.k.a Sensitivity/Recall'

        tnr_message = f'{self.true_negative_rate:.1%} of negative instances were correctly identified.; ' \
                      f'i.e. {self._true_negatives} "{self._labels[0]}" labels were correctly identified ' \
                      f'out of {self._actual_negatives} instances'

        fpr_message = f'{self.false_positive_rate:.1%} of negative instances were incorrectly identified ' \
                      f'as positive; ' \
                      f'i.e. {self._false_positives} "{self._labels[0]}" labels were incorrectly ' \
                      f'identified as "{self._labels[1]}", out of {self._actual_negatives} instances'

        fnr_message = f'{self.false_negative_rate:.1%} of positive instances were incorrectly identified ' \
                      f'as negative; ' \
                      f'i.e. {self._false_negatives} "{self._labels[1]}" labels were incorrectly ' \
                      f'identified as "{self._labels[0]}", out of {self._actual_positives} instances'

        ppv_message = f'When the model claims an instance is positive, it is correct ' \
                      f'{self.positive_predictive_value:.1%} of the time; ' \
                      f'i.e. out of the {self._true_positives + self._false_positives} times the model ' \
                      f'predicted "{self._labels[1]}", it was correct {self._true_positives} ' \
                      f'times; a.k.a precision'

        npv_message = f'When the model claims an instance is negative, it is correct ' \
                      f'{self.negative_predictive_value:.1%} of the time; ' \
                      f'i.e. out of the {self._true_negatives + self._false_negatives} times the model ' \
                      f'predicted "{self._labels[0]}", it was correct {self._true_negatives} times'

        f1_message = 'The F1 score can be interpreted as a weighted average of the precision and recall, ' \
                     'where an F1 score reaches its best value at 1 and worst score at 0.'

        accuracy_message = f'{self.accuracy:.1%} of instances were correctly identified'
        error_message = f'{self.error_rate:.1%} of instances were incorrectly identified'
        prevalence_message = f'{self.prevalence:.1%} of the data are positive; i.e. out of ' \
                             f'{self.sample_size} total observations; {self._actual_positives} are labeled ' \
                             f'as "{self._labels[1]}"'
        total_obs_message = f'There are {self.sample_size} total observations; i.e. sample size'

        return {'AUC': (self.auc, auc_message),
                'True Positive Rate': (self.true_positive_rate, tpr_message),
                'True Negative Rate': (self.true_negative_rate, tnr_message),
                'False Positive Rate': (self.false_positive_rate, fpr_message),
                'False Negative Rate': (self.false_negative_rate, fnr_message),
                'Positive Predictive Value': (self.positive_predictive_value, ppv_message),
                'Negative Predictive Value': (self.negative_predictive_value, npv_message),
                'F1 Score': (self.f1_score, f1_message),
                'Accuracy': (self.accuracy, accuracy_message),
                'Error Rate': (self.error_rate, error_message),
                '% Positive': (self.prevalence, prevalence_message),
                'Total Observations': (self.sample_size, total_obs_message)}

    def all_metrics_df(self,
                       return_details: bool = True,
                       return_style: bool = False,
                       round_by: Optional[int] = None) -> Union[pd.DataFrame, Styler]:
        """All of the metrics are returned as a DataFrame.

        Args:
            return_details:
                if True, then return descriptions of score and more information in an additional column
            return_style:
                if True, return styler object; else return dataframe
            round_by:
                the number of digits to round by; if None, then don't round
        """
        if return_details:
            result = pd.DataFrame.from_dict(self.all_metrics,
                                            orient='index',
                                            columns=['Scores', 'Details'])
        else:
            result = pd.DataFrame.from_dict({key: value[0] for key, value in self.all_metrics.items()},
                                            orient='index',
                                            columns=['Scores'])

        if round_by:
            result['Scores'] = result['Scores'].round(round_by)

        if return_style:
            subset_scores = [x for x in result.index.values if x != 'Total Observations']

            subset_scores = pd.IndexSlice[result.loc[subset_scores, :].index, 'Scores']
            subset_negative_bad = pd.IndexSlice[result.loc[['False Positive Rate',
                                                            'False Negative Rate'], 'Scores'].index, 'Scores']
            subset_secondary = pd.IndexSlice[result.loc[['Accuracy', 'Error Rate', '% Positive'],
                                                        'Scores'].index, 'Scores']
            subset_total_observations = pd.IndexSlice[result.loc[['Total Observations'],
                                                                 'Scores'].index, 'Scores']

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

    def plot_auc_curve(self, figure_size: tuple = STANDARD_WIDTH_HEIGHT):
        """Plots the ROC AUC

        Args:
            figure_size:
                tuple containing `(width, height)` of plot. The default height is defined by
                `helpsk.plot.STANDARD_HEIGHT`, and the default width is
                `helpsk.plot.STANDARD_HEIGHT / helpsk.plot.GOLDEN_RATIO`
        """
        plt.figure(figsize=figure_size)

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

    def plot_threshold_curves(self,
                              score_threshold_range: Tuple[float, float] = (0.3, 0.9),
                              figure_size: tuple = STANDARD_WIDTH_HEIGHT):
        """Plots various scores (e.g. True Positive Rate, False Positive Rate, etc.) for various score
        thresholds. (A score threshold is the value for which you would predict a positive label if the
        value of the score is above the threshold (e.g. usually 0.5).

        Args:
            score_threshold_range:
                range of score thresholds to plot (x-axis); tuple with minimum threshold in first index and
                maximum threshold in second index.
            figure_size:
                tuple containing `(width, height)` of plot. The default height is defined by
                `helpsk.plot.STANDARD_HEIGHT`, and the default width is
                `helpsk.plot.STANDARD_HEIGHT / helpsk.plot.GOLDEN_RATIO`
        """
        plt.figure(figsize=figure_size)

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
        plt.vlines(x=self.score_threshold, ymin=0, ymax=1, colors='black')
        plt.grid()
        plt.tight_layout()

    def plot_precision_recall_tradeoff(self,
                                       score_threshold_range: Tuple[float, float] = (0, 1),
                                       figure_size: tuple = STANDARD_WIDTH_HEIGHT):
        """Plots the tradeoff between precision (i.e. positive predict value) and recall (i.e. True Positive
        Rate) for various score thresholds. (A score threshold is the value for which you would predict a
        positive label if the value of the score is above the threshold (e.g. usually 0.5).

        Args:
            score_threshold_range:
                range of score thresholds to plot (x-axis); tuple with minimum threshold in first index and
                maximum threshold in second index.
            figure_size:
                tuple containing `(width, height)` of plot. The default height is defined by
                `helpsk.plot.STANDARD_HEIGHT`, and the default width is
                `helpsk.plot.STANDARD_HEIGHT / helpsk.plot.GOLDEN_RATIO`
        """
        plt.figure(figsize=figure_size)

        def get_threshold_scores(threshold):
            temp_eval = TwoClassEvaluator(actual_values=self._actual_values,
                                          predicted_scores=self._predicted_scores,
                                          labels=('x', 'y'),
                                          score_threshold=threshold)

            return threshold,\
                temp_eval.true_positive_rate,\
                temp_eval.positive_predictive_value

        threshold_curves = [get_threshold_scores(threshold=x) for x in np.arange(score_threshold_range[0],
                                                                                 score_threshold_range[1],
                                                                                 0.025)]
        threshold_curves = pd.DataFrame(threshold_curves,
                                        columns=['Score Threshold',
                                                 'True Pos. Rate (Recall)',
                                                 'Pos. Predictive Value (Precision)'])

        axis = sns.lineplot(x='Score Threshold', y='value', hue='variable',
                            data=pd.melt(frame=threshold_curves, id_vars=['Score Threshold']))
        axis.set_xticks(np.arange(score_threshold_range[0], score_threshold_range[1] + 0.1, 0.1))
        axis.set_yticks(np.arange(0, 1.1, .1))
        plt.vlines(x=self.score_threshold, ymin=0, ymax=1, colors='black')
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


class RegressionEvaluator:
    """
    Evaluates models for regression (i.e. numeric outcome) problems.
    """
    def __init__(self,
                 actual_values: np.ndarray,
                 predicted_values: np.ndarray):
        """
        Args:
            actual_values:
                the actual values
            predicted_values:
                the predicted values
        """

        assert len(actual_values) == len(predicted_values)

        self._actual_values = actual_values
        self._predicted_values = predicted_values
        self._residuals = actual_values - predicted_values
        self._standard_deviation = np.std(actual_values)
        self._mean_squared_error = float(np.mean(np.square(actual_values - predicted_values)))
        self._mean_absolute_error = float(np.mean(np.abs(actual_values - predicted_values)))
        self._r_squared = r2_score(y_true=actual_values, y_pred=predicted_values)

    @property
    def mean_absolute_error(self) -> float:
        """Mean Absolute Error"""
        return self._mean_absolute_error

    @property
    def mean_squared_error(self) -> float:
        """Mean Squared Error"""
        return self._mean_squared_error

    @property
    def root_mean_squared_error(self) -> float:
        """Root Mean Squared Error"""
        return np.sqrt(self.mean_squared_error)

    @property
    def rmse_to_st_dev(self) -> float:
        """The ratio of RMSE to the standard deviation of the actual values.
        Gives an indication of how large the errors are to the actual data.
        """
        return self.root_mean_squared_error / self._standard_deviation

    @property
    def r_squared(self) -> float:
        """R Squared"""
        return self._r_squared

    @property
    def total_observations(self):
        """The total number of observations i.e. sample size."""
        return len(self._actual_values)

    @property
    def all_metrics(self) -> dict:
        """Returns a dictionary of the most common error metrics for regression problems."""
        return {'Mean Absolute Error (MAE)': self.mean_absolute_error,
                'Root Mean Squared Error (RMSE)': self.root_mean_squared_error,
                'RMSE to Standard Deviation of Target': self.rmse_to_st_dev,
                'R Squared': self.r_squared,
                'Total Observations': self.total_observations}

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
        result = pd.DataFrame.from_dict({key: value for key, value in self.all_metrics.items()},  # pylint: disable=unnecessary-comprehension  # noqa
                                        orient='index',
                                        columns=['Scores'])

        if round_by is not None:
            result.iloc[0:2] = result.iloc[0:2].round(round_by)

        if return_style:
            subset_scores = pd.IndexSlice[result.loc[['Mean Absolute Error (MAE)',
                                                      'Root Mean Squared Error (RMSE)'],
                                                     'Scores'].index, 'Scores']
            subset_secondary = pd.IndexSlice[result.loc[['RMSE to Standard Deviation of Target',
                                                         'R Squared'],
                                                        'Scores'].index, 'Scores']
            subset_total_observations = pd.IndexSlice[result.loc[['Total Observations'],
                                                                 'Scores'].index, 'Scores']
            result = result.style
            if round_by is not None:
                result = result.format(subset=subset_scores, thousands=',', precision=round_by)
            else:
                result = result.format(subset=subset_scores, thousands=',')
            result = result.format(subset=subset_secondary, precision=3)
            result = result.format(subset=subset_total_observations, thousands=',', precision=0)

        return result

    def plot_residuals_vs_fits(self, figure_size: tuple = STANDARD_WIDTH_HEIGHT):
        """Plots residuals vs fitted values

        Args:
            figure_size:
                tuple containing `(width, height)` of plot. The default height is defined by
                `helpsk.plot.STANDARD_HEIGHT`, and the default width is
                `helpsk.plot.STANDARD_HEIGHT / helpsk.plot.GOLDEN_RATIO`
        """
        lowess = sm.nonparametric.lowess
        loess_points = lowess(self._residuals, self._predicted_values)
        loess_x, loess_y = zip(*loess_points)

        plt.figure(figsize=figure_size)
        plt.plot(loess_x, loess_y, color='r')
        plt.scatter(x=self._predicted_values, y=self._residuals, s=8, alpha=0.5)
        plt.title('Residuals vs. Fitted Values')
        plt.xlabel('Fitted Values')
        plt.ylabel('Residuals (Actual - Predicted)')

    def plot_predictions_vs_actuals(self, figure_size: tuple = STANDARD_WIDTH_HEIGHT):
        """Plots predictions vs actual values

        Args:
            figure_size:
                tuple containing `(width, height)` of plot. The default height is defined by
                `helpsk.plot.STANDARD_HEIGHT`, and the default width is
                `helpsk.plot.STANDARD_HEIGHT / helpsk.plot.GOLDEN_RATIO`
        """
        lowess = sm.nonparametric.lowess
        loess_points = lowess(self._predicted_values, self._actual_values)
        loess_x, loess_y = zip(*loess_points)

        plt.figure(figsize=figure_size)
        plt.plot(loess_x, loess_y, color='r', alpha=0.5, label='Loess (Predictions vs Actuals)')
        plt.plot(self._actual_values, self._actual_values, color='b', alpha=0.5, label='Perfect Prediction')
        plt.scatter(x=self._actual_values, y=self._predicted_values, s=8, alpha=0.5)
        plt.title('Predicted Values vs. Actual Values')
        plt.xlabel('Actuals')
        plt.ylabel('Predicted')
        axis = plt.gca()
        handles, labels = axis.get_legend_handles_labels()
        axis.legend(handles, labels)
        plt.figtext(0.99, 0.01,
                    'Note: observations above blue line mean model is over-predicting; below means under-'
                    'predicting.',  # noqa
                    horizontalalignment='right')
        return axis

    def plot_residuals_vs_actuals(self, figure_size: tuple = STANDARD_WIDTH_HEIGHT):
        """Plots residuals vs actuals values

        Args:
            figure_size:
                tuple containing `(width, height)` of plot. The default height is defined by
                `helpsk.plot.STANDARD_HEIGHT`, and the default width is
                `helpsk.plot.STANDARD_HEIGHT / helpsk.plot.GOLDEN_RATIO`
        """
        lowess = sm.nonparametric.lowess
        loess_points = lowess(self._residuals, self._actual_values)
        loess_x, loess_y = zip(*loess_points)

        plt.figure(figsize=figure_size)
        plt.plot(loess_x, loess_y, color='r')
        plt.scatter(x=self._actual_values, y=self._residuals, s=8, alpha=0.5)
        plt.title('Residuals vs. Actual Values')
        plt.xlabel('Actual')
        plt.ylabel('Residuals (Actual - Predicted)')
        plt.figtext(0.99, 0.01,
                    'Note: Actual > Predicted => Under-predicting (positive residual); negative residuals '
                    'mean over-predicting',  # noqa
                    horizontalalignment='right')
