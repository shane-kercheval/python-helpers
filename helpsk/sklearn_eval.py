"""This module contains helper functions when working with sklearn (scikit-learn) objects;
in particular, for evaluating models"""
# pylint: disable=too-many-lines
import math
import warnings
from re import match
from typing import Tuple, Union, Optional, List

import numpy as np
import pandas as pd
import scipy.stats as st
import seaborn as sns
import yaml
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
from helpsk.validation import assert_true

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

    # pylint: disable=too-many-arguments
    def __init__(self,
                 searcher: BaseSearchCV,
                 higher_score_is_better: bool = True,
                 run_description: str = "",
                 parameter_name_mappings: Union[dict, None] = None):
        """
        This object encapsulates the results from a SearchCV object (e.g.
        sklearn.model_selection.GridSearch/RandomSearch, skopt.BayesSearchCV). The results can then be
        converted to a dictionary, in a specific format with the intent to write the contents to a
        yaml file.

        At this time, this function does not capture the individual fold scores from the individual splits.

        Params:
            searcher:
                A `BaseSearchCV` object that has either used a string passed to the `scoring` parameter of the
                constructor (e.g. `GridSearchCV(..., scoring='auc', ...)` or a dictionary with metric
                names as keys and callables as values.

                An example of the dictionary option:

                    scores = {
                        'ROC/AUC': SCORERS['roc_auc'],
                        'F1': make_scorer(f1_score, greater_is_better=True),
                        'Pos. Pred. Val': make_scorer(precision_score, greater_is_better=True),
                        'True Pos. Rate': make_scorer(recall_score, greater_is_better=True),
                    }
                    grid_search = GridSearchCV(..., scoring=scores, ...)

            higher_score_is_better:
                If True, higher scores are better; if False, lower scores are better.
                A value of False assumes that the scores returned from sklearn are negative and will multiple
                the values by -1.

            run_description:
                An optional string to save in the dictionary

            parameter_name_mappings:
                A dictionary containing the parameter names returned by the searchCV object as keys (which
                should correspond to the path of the pipeline(s) corresponding to the parameter) and the new,
                friendlier, names that can be displayed in graphs and tables.

                For example:

                    {'model__max_features': 'max_features',
                     'model__n_estimators': 'n_estimators',
                     'prep__non_numeric__encoder__transformer': 'encoder',
                     'prep__numeric__impute__transformer': 'imputer',
                     'prep__numeric__scaling__transformer': 'scaler'}
        """
        if searcher is not None:  # check for None in the case that __init__ is being called from `from_dict`
            self._cv_dict = SearchCVParser.\
                __search_cv_to_dict(searcher=searcher,
                                    higher_score_is_better=higher_score_is_better,
                                    run_description=run_description,
                                    parameter_name_mappings=parameter_name_mappings)
        else:
            self._cv_dict = None

        self._cv_dataframe = None

    @classmethod
    def from_dict(cls, cv_dict):
        """This method creates a SearchCVParser from the dictionary previously created by
        `__search_cv_to_dict()`"""
        parser = cls(searcher=None, higher_score_is_better=None, run_description=None,  # noqa
                     parameter_name_mappings=None)
        parser._cv_dict = cv_dict
        return parser

    @classmethod
    def from_yaml_file(cls, yaml_file_name):
        """This method creates a SearchCVParser from a yaml file created by `to_yaml_file()`"""
        with open(yaml_file_name, 'r') as file:
            cv_dict = yaml.safe_load(file)

        return SearchCVParser.from_dict(cv_dict=cv_dict)

    def to_yaml_file(self, yaml_file_name: str):
        """This method saves the self._cv_dict dictionary to a yaml file."""
        with open(yaml_file_name, 'w') as file:
            yaml.dump(self._cv_dict, file, default_flow_style=False, sort_keys=False)

    # pylint: disable=too-many-branches
    # pylint: disable=too-many-statements
    @staticmethod
    def __search_cv_to_dict(searcher: BaseSearchCV,
                            higher_score_is_better: bool = True,
                            run_description: str = "",
                            parameter_name_mappings: Union[dict, None] = None) -> dict:
        """This extracts the information from a BaseSearchCV object and converts it to a dictionary."""

        def string_if_not_number(obj):
            if isinstance(obj, (int, float, complex)):
                return obj

            return str(obj)

        cv_results_dict = {
            'description': run_description,
            'cross_validation_type': str(type(searcher)),
            'higher_score_is_better': higher_score_is_better
        }

        if isinstance(searcher.scoring, dict):
            score_names = list(searcher.scoring.keys())
        elif isinstance(searcher.scoring, str):
            score_names = [searcher.scoring]
        else:
            message = 'The `searcher` does not have a string or dictionary .scoring property. Cannot ' \
                      'extract scores.'
            raise HelpskParamValueError(message)

        # get number of splits (e.g. 5 fold 2 repeat cross validation has 10 splits)
        # I could check the .cv param of the searcher object but not sure all types of cv objects have the
        # same parameters e.g. searcher.cv.n_repeats
        # if there is only 1 score, we need to look for e.g. "split0_test_score"
        # if there are multiple scores we need to look for e.g. "split0_test_ROC/AUC" but we don't want
        # to duplicate the counts e.g. we don't want to also capture "split0_test_True Pos. Rate"
        if len(score_names) == 1:
            split_score_matching_string = "split\\d_test_score"
        else:
            split_score_matching_string = "split\\d_test_" + score_names[0]

        number_of_splits = len([x for x in searcher.cv_results_.keys()
                                if bool(match(split_score_matching_string, x))])
        cv_results_dict['number_of_splits'] = number_of_splits
        cv_results_dict['score_names'] = score_names
        cv_results_dict['parameter_names'] = [key for key, value in searcher.cv_results_['params'][0].items()]

        if parameter_name_mappings:
            for key in parameter_name_mappings.keys():
                assert_true(key in cv_results_dict['parameter_names'])
            cv_results_dict['parameter_names_mapping'] = parameter_name_mappings

        number_of_iterations = len(searcher.cv_results_['mean_fit_time'])

        # convert test scores to dictionaries
        if len(score_names) == 1:
            test_score_ranking = searcher.cv_results_['rank_test_score'].tolist()
            test_score_averages = searcher.cv_results_['mean_test_score'].tolist()
            test_score_standard_deviations = searcher.cv_results_['std_test_score'].tolist()

            assert_true(len(test_score_ranking) == number_of_iterations)
            assert_true(len(test_score_averages) == number_of_iterations)
            assert_true(len(test_score_standard_deviations) == number_of_iterations)

            cv_results_dict['test_score_rankings'] = {score_names[0]: test_score_ranking}
            cv_results_dict['test_score_averages'] = {score_names[0]: test_score_averages}
            cv_results_dict['test_score_standard_deviations'] = {score_names[0]:
                                                                     test_score_standard_deviations}
        else:
            ranking_dict = {}
            averages_dict = {}
            standard_deviations_dict = {}
            for score in score_names:
                rankings = searcher.cv_results_['rank_test_' + score].tolist()
                averages = searcher.cv_results_['mean_test_' + score].tolist()
                standard_deviations = searcher.cv_results_['std_test_' + score].tolist()

                assert_true(len(rankings) == number_of_iterations)
                assert_true(len(averages) == number_of_iterations)
                assert_true(len(standard_deviations) == number_of_iterations)

                ranking_dict[score] = rankings
                averages_dict[score] = averages
                standard_deviations_dict[score] = standard_deviations

            cv_results_dict['test_score_rankings'] = ranking_dict
            cv_results_dict['test_score_averages'] = averages_dict
            cv_results_dict['test_score_standard_deviations'] = standard_deviations_dict

            # if higher_score_is_better is False, sklearn will return negative numbers; I want actual values
            if not higher_score_is_better:
                averages = cv_results_dict['test_score_averages']
                for key in averages.keys():
                    cv_results_dict['test_score_averages'][key] = [-1 * x for x in averages[key]]

        # convert training scores to dictionaries, if training scores exists
        # i.e. if return_train_score=True for the SearchCV object
        if 'mean_train_score' in searcher.cv_results_ or 'mean_train_'+score_names[0] in searcher.cv_results_:
            if len(score_names) == 1:
                train_score_averages = searcher.cv_results_['mean_train_score'].tolist()
                train_score_standard_deviations = searcher.cv_results_['std_train_score'].tolist()

                assert_true(len(train_score_averages) == number_of_iterations)
                assert_true(len(train_score_standard_deviations) == number_of_iterations)

                cv_results_dict['train_score_averages'] = {score_names[0]: train_score_averages}
                cv_results_dict['train_score_standard_deviations'] = {score_names[0]:
                                                                          train_score_standard_deviations}
            else:
                averages_dict = {}
                standard_deviations_dict = {}
                for score in score_names:
                    averages = searcher.cv_results_['mean_train_' + score].tolist()
                    standard_deviations = searcher.cv_results_['std_train_' + score].tolist()

                    assert_true(len(averages) == number_of_iterations)
                    assert_true(len(standard_deviations) == number_of_iterations)

                    averages_dict[score] = averages
                    standard_deviations_dict[score] = standard_deviations

                cv_results_dict['train_score_averages'] = averages_dict
                cv_results_dict['train_score_standard_deviations'] = standard_deviations_dict

                # if higher_score_is_better is False, sklearn will return negative numbers; I want actual
                # values
                if not higher_score_is_better:
                    averages = cv_results_dict['train_score_averages']
                    for key in averages.keys():
                        cv_results_dict['train_score_averages'][key] = [-1 * x for x in averages[key]]

        assert_true(len(searcher.cv_results_['params']) == number_of_iterations)

        cv_results_dict['parameter_iterations'] = [
            {key: string_if_not_number(value) for key, value in searcher.cv_results_['params'][index].items()}
            for index in range(len(searcher.cv_results_['params']))
        ]

        fit_time_averages = searcher.cv_results_['mean_fit_time'].tolist()
        fit_time_standard_deviations = searcher.cv_results_['std_fit_time'].tolist()
        score_time_averages = searcher.cv_results_['mean_score_time'].tolist()
        score_time_standard_deviations = searcher.cv_results_['std_score_time'].tolist()

        assert_true(len(fit_time_averages) == number_of_iterations)
        assert_true(len(fit_time_standard_deviations) == number_of_iterations)
        assert_true(len(score_time_averages) == number_of_iterations)
        assert_true(len(score_time_standard_deviations) == number_of_iterations)

        cv_results_dict['timings'] = {'fit time averages': fit_time_averages,
                                      'fit time standard deviations': fit_time_standard_deviations,
                                      'score time averages': score_time_averages,
                                      'score time standard deviations': score_time_standard_deviations}

        return cv_results_dict

    def to_dataframe(self, sort_by_score: bool = True):
        """This function converts the score information from the SearchCV object into a pd.DataFrame.

        Params:
            sort_by_score:
                if True, sorts the dataframe starting with the best (primary) score to the worst score.
                Secondary scores are not considered.

        Returns:
            a DataFrame containing score information for each cross-validation iteration. A single row
            corresponds to one iteration (i.e. one set of hyper-parameters that were cross-validated).
        """
        if self._cv_dataframe is None:

            for score_name in self.score_names:
                confidence_intervals = st.t.interval(alpha=0.95,  # confidence interval
                                                     # number_of_splits is sample-size
                                                     df=self.number_of_splits - 1,  # degrees of freedom
                                                     loc=self.test_score_averages[score_name],
                                                     scale=self.score_standard_errors(score_name=score_name))

                # only give confidence intervals for the primary score
                self._cv_dataframe = pd.concat([
                        self._cv_dataframe,
                        pd.DataFrame({score_name + " Mean": self.test_score_averages[score_name],
                                      score_name + " 95CI.LO": confidence_intervals[0],
                                      score_name + " 95CI.HI": confidence_intervals[1]})
                    ],
                    axis=1
                )

            self._cv_dataframe = pd.concat([self._cv_dataframe,
                                            pd.DataFrame.from_dict(self.parameter_iterations)],  # noqa
                                           axis=1)

            if self.parameter_names_mapping:
                self._cv_dataframe = self._cv_dataframe.rename(columns=self.parameter_names_mapping)

        copy = self._cv_dataframe.copy(deep=True)

        if sort_by_score:
            copy = copy.iloc[self.primary_score_best_indexes]

        return copy

    def to_formatted_dataframe(self,
                               round_by: int = 3,
                               num_rows: int = 50,
                               primary_score_only: bool = False,
                               exclude_no_variance_params: bool = True,
                               return_style: bool = True,
                               sort_by_score: bool = True) -> Union[pd.DataFrame, Styler]:
        """This function converts the score information from the SearchCV object into a pd.DataFrame or a
        Styler object, formatted accordingly.

        The Hyper-Parameter columns will be highlighted in blue where the primary
        score (i.e. first column) for the iteration (i.e. the row i.e. the combination of hyper-parameters
        that were cross validated) is within 1 standard error of the top primary score (i.e. first column
        first row).

        Args:
            round_by:
                the number of digits to round by for the score columns (does not round the parameter columns)
            num_rows:
                the number of rows to return in the resulting DataFrame.
            primary_score_only:
                if True, then only include the primary score.
            exclude_no_variance_params:
                if True, exclude columns that only have 1 unique value
            return_style:
                If True, return Styler object, else return pd.DataFrame
            sort_by_score:
                if True, sorts the dataframe starting with the best (primary) score to the worst score.
                Secondary scores are not considered.

        Returns:
            Returns either pd.DataFrame or pd.DataFrame.Styler.
        """
        cv_dataframe = self.to_dataframe(sort_by_score=sort_by_score)
        cv_dataframe = cv_dataframe.head(num_rows)

        if exclude_no_variance_params:
            columns_to_drop = [x for x in self.parameter_names if len(cv_dataframe[x].unique()) == 1]
            cv_dataframe = cv_dataframe.drop(columns=columns_to_drop)

        score_columns = list(cv_dataframe.columns[cv_dataframe.columns.str.endswith((' Mean',
                                                                                     ' 95CI.LO',
                                                                                     ' 95CI.HI'))])
        if primary_score_only:
            columns_to_drop = [x for x in score_columns if not x.startswith(self.primary_score_name)]
            cv_dataframe = cv_dataframe.drop(columns=columns_to_drop)

        cv_dataframe = cv_dataframe.round(dict(zip(score_columns, [round_by] * len(score_columns))))

        final_columns = cv_dataframe.columns  # save for style logic

        if return_style:
            cv_dataframe = cv_dataframe.style

            for score in self.score_names:
                mean_key = score + ' Mean'
                ci_low_key = score + ' 95CI.LO'
                ci_high_key = score + ' 95CI.HI'

                if mean_key in final_columns:
                    cv_dataframe. \
                        bar(subset=[mean_key], color=hcolor.Colors.PIGMENT_GREEN.value)

                if ci_low_key in final_columns:
                    cv_dataframe. \
                        bar(subset=[ci_high_key], color=hcolor.GRAY). \
                        pipe(hstyle.bar_inverse, subset=[ci_low_key], color=hcolor.GRAY)

                cv_dataframe.pipe(hstyle.format, round_by=round_by, hide_index=True)

            # highlight iterations whose primary score (i.e. first column of `results` dataframe) is within
            # 1 standard error of the top primary score (i.e. first column first row).
            # pylint: disable=invalid-name, unused-argument
            def highlight_cols(s):   # noqa
                return 'background-color: %s' % hcolor.Colors.PASTEL_BLUE.value

            # we might have removed columns (e.g. that don't have any variance) so check that the columns
            # were in the final set
            columns_to_highlight = [x for x in self.parameter_names if x in final_columns]
            cv_dataframe.applymap(highlight_cols,
                             subset=pd.IndexSlice[self.indexes_within_1_standard_error,
                                                  columns_to_highlight])

        return cv_dataframe

    ####
    # The following properties expose the highest levels of the underlying dictionary/yaml
    ####
    @property
    def description(self):
        """the description passed to `run_description`."""
        return self._cv_dict['description']

    @property
    def higher_score_is_better(self):
        """The value passed to `higher_score_is_better`."""
        return self._cv_dict['higher_score_is_better']

    @property
    def cross_validation_type(self) -> str:
        """The string representation of the SearchCV object."""
        return self._cv_dict['cross_validation_type']

    @property
    def number_of_splits(self) -> int:
        """This is the number of CV folds. For example, a 5-fold 2-repeat CV has 10 splits."""
        return self._cv_dict['number_of_splits']

    @property
    def score_names(self) -> list:
        """Returns a list of the names of the scores"""
        return self._cv_dict['score_names']

    @property
    def parameter_names_original(self) -> list:
        """Returns the original parameter names (i.e. the path generated by the scikit-learn pipelines."""
        return self._cv_dict['parameter_names']

    @property
    def parameter_names(self) -> list:
        """This property returns either the original parameter names if no `parameter_names_mapping` was
        provided, or it returns the new parameter names (i.e. the values from `parameter_names_mapping`)."""
        if self.parameter_names_mapping:
            return list(self.parameter_names_mapping.values())

        return self.parameter_names_original

    @property
    def parameter_names_mapping(self) -> dict:
        """The dictionary passed to `parameter_name_mappings`."""
        return self._cv_dict.get('parameter_names_mapping')

    @property
    def test_score_rankings(self) -> dict:
        """The rankings of each of the test scores, from the searcher.cv_results_ object."""
        return self._cv_dict['test_score_rankings']

    @property
    def test_score_averages(self) -> dict:
        """The test score averages, from the searcher.cv_results_ object."""
        return self._cv_dict['test_score_averages']

    @property
    def test_score_standard_deviations(self) -> dict:
        """The test score standard deviations, from the searcher.cv_results_ object."""
        return self._cv_dict['test_score_standard_deviations']

    @property
    def train_score_averages(self) -> dict:
        """The training score averages, from the searcher.cv_results_ object, if provided."""
        return self._cv_dict.get('train_score_averages')

    @property
    def train_score_standard_deviations(self) -> dict:
        """The training score standard deviations, from the searcher.cv_results_ object, if provided."""
        return self._cv_dict.get('train_score_standard_deviations')

    @property
    def parameter_iterations(self) -> list:
        """The "iterations" i.e. the hyper-parameter combinations in order of execution."""
        return self._cv_dict['parameter_iterations']

    def iteration_labels(self, order_from_best_to_worst=True) -> List[str]:
        """An iteration is a set of hyper-parameters that were cross validated. The corresponding label for
        each iteration is a single string containing all of the hyper-parameter names and values in the format
        of `{param1: value1, param2: value2}`.

        Params:
            order_from_best_to_worst: if True, returns the labels in order from the best score to the worst
            score, which should match the ordered of .to_dataframe() or .to_formatted_dataframe()`. If False,
            returns the labels in order that they were ran by the cross validation object.

        Returns:
            a pd.Series the same length as `number_of_trials` containing a str
        """
        def create_hyper_param_labels(iteration) -> list:
            """Creates a list of strings that represent the name/value pair for each hyper-parameter."""
            return [f"{self.parameter_names_mapping[x] if self.parameter_names_mapping  else x}: {iteration[x]}"  # pylint: disable=line-too-long  # noqa
                    for x in self.parameter_names_original]
        # create_hyper_param_labels(iteration=self.parameter_iterations[0])

        def create_trial_label(iteration) -> str:
            return f"{{{hstring.collapse(create_hyper_param_labels(iteration), separate=', ')}}}"
        # create_trial_label(iteration=self.parameter_iterations[0])

        labels = [create_trial_label(x) for x in self.parameter_iterations]

        if order_from_best_to_worst:
            labels = [x for _, x in sorted(zip(self.primary_score_iteration_ranking, labels))]

        return labels

    @property
    def timings(self) -> dict:
        """The timings providing by searcher.cv_results_."""
        return self._cv_dict['timings']

    ####
    # The following properties are additional helpers
    ####
    @property
    def number_of_iterations(self) -> int:
        """"A single trial contains the cross validation runs for a single set of hyper-parameters. The
        'number of trials' is basically the number of combinations of different hyper-parameters that were
        cross validated."""
        return len(self.parameter_iterations)

    @property
    def number_of_scores(self) -> int:
        """The number of scores passed to the SearchCV object"""
        return len(self.score_names)

    @property
    def primary_score_name(self) -> str:
        """The first scorer passed to the SearchCV will be treated as the primary score. This property returns
        the name of the score."""
        return self.score_names[0]

    @property
    def primary_score_averages(self) -> np.array:
        """The first scorer passed to the SearchCV will be treated as the primary score. This property returns
        the average score (across all splits) for each iteration. Note that the average scores are
        the weighted averages
        https://stackoverflow.com/questions/44947574/what-is-the-meaning-of-mean-test-score-in-cv-result"""
        return np.array(self.test_score_averages[self.primary_score_name])

    def score_standard_errors(self, score_name: str) -> np.array:
        """The first scorer passed to the SearchCV will be treated as the primary score. This property returns
        the standard error associated with the mean score of each iteration, for the primary score."""
        score_standard_deviations = self.test_score_standard_deviations[score_name]
        return np.array(score_standard_deviations) / math.sqrt(self.number_of_splits)

    @property
    def primary_score_iteration_ranking(self) -> np.array:
        """The ranking of the corresponding index, in terms of best to worst score.

        e.g. [5, 6, 7, 8, 3, 4, 1, 2]
            This means that the 6th index/iteration had the highest ranking (1); and that the 3rd index had
            the worst ranking (8)

        This differs from `primary_score_best_indexes` which returns the order of indexes from best to worst.
        So in the example above, the first value returned in the `primary_score_best_indexes` array would be
        6 because the best score is at index 6. The last value in the array 3, because the worst score is at
        index 3.

        Note that `primary_score_iteration_ranking` starts at 1 while primary_score_best_indexes starts at 0.
        """
        return np.array(self.test_score_rankings[self.primary_score_name])

    @property
    def primary_score_best_indexes(self) -> np.array:
        """The indexes of best to worst primary scores. See documentation for
        `primary_score_iteration_ranking` to understand the differences between the two properties."""
        return np.argsort(self.primary_score_iteration_ranking)

    @property
    def best_primary_score_index(self) -> int:
        """The index of best primary score."""
        return self.primary_score_best_indexes[0]

    def best_primary_score_params(self) -> dict:
        """
        The "best" score (could be the highest or lowest depending on `higher_score_is_better`) associated
        with the primary score.
        """
        best_params = self.parameter_iterations[self.best_primary_score_index]

        if self.parameter_names_mapping:
            best_params = {self.parameter_names_mapping[key]: value for key, value in best_params.items()}

        return best_params

    @property
    def best_primary_score(self) -> float:
        """
        The "best" score (could be the highest or lowest depending on `higher_score_is_better`) associated
        with the primary score.
        """
        return self.primary_score_averages[self.best_primary_score_index]

    @property
    def best_primary_score_standard_error(self) -> float:
        """The standard error associated with the best score of the primary scorer"""
        return self.score_standard_errors(score_name=self.primary_score_name)[self.best_primary_score_index]

    @property
    def indexes_within_1_standard_error(self) -> list:
        """Returns the iteration indexes where the primary scores (i.e. first scorer
        passed to SearchCV object; i.e. first column of the to_dataframe() DataFrame) are within 1 standard
        error of the highest primary score."""
        cv_dataframe = self.to_dataframe(sort_by_score=True)

        if self.higher_score_is_better:
            return list(cv_dataframe.index[cv_dataframe.iloc[:, 0] >=
                                           self.best_primary_score - self.best_primary_score_standard_error])

        return list(cv_dataframe.index[cv_dataframe.iloc[:, 0] <=
                                       self.best_primary_score + self.best_primary_score_standard_error])

    @property
    def fit_time_averages(self) -> np.array:
        """
        Returns a list of floats; one value for each iteration (i.e. a single set of hyper-params).
        Each value is the average number of seconds that the iteration took to fit the model, per split
        (i.e. the average fit time of all splits).
        """
        return np.array(self.timings['fit time averages'])

    @property
    def fit_time_standard_deviations(self) -> np.array:
        """
        Returns a list of floats; one value for each iteration (i.e. a single set of hyper-params).
        Each value is the standard deviation of seconds that the iteration took to fit the model, per split
        (i.e. the standard deviation of fit time across all splits).
        """
        return np.array(self.timings['fit time standard deviations'])

    @property
    def score_time_averages(self) -> np.array:
        """
        Returns a list of floats; one value for each iteration (i.e. a single set of hyper-params).
        Each value is the average number of seconds that the iteration took to score the model, per split
        (i.e. the average score time of all splits).
        """
        return np.array(self.timings['score time averages'])

    @property
    def score_time_standard_deviations(self) -> np.array:
        """
        Returns a list of floats; one value for each iteration (i.e. a single set of hyper-params).
        Each value is the standard deviation of seconds that the iteration took to score the model, per split
        (i.e. the standard deviation of score time across all splits).
        """
        return np.array(self.timings['score time standard deviations'])

    @property
    def iteration_fit_times(self) -> np.array:
        """For each iteration, it is the amount of time it took to fit the model.

        Calculated by Average fit time for each iteration multiplied by the number of splits per iteration.

        self.fit_time_averages * self.number_of_splits

        Returns:
            array containing the fit time for each iteration
        """
        return self.fit_time_averages * self.number_of_splits

    @property
    def fit_time_total(self) -> float:
        """Total fit time across all iterations."""
        return float(np.sum(self.iteration_fit_times))

    @property
    def iteration_score_times(self) -> np.array:
        """For each iteration, it is the amount of time it took to score the model.

        Calculated by Average score time for each iteration multiplied by the number of splits per iteration.

        self.score_time_averages * self.number_of_splits

        Returns:
            array containing the score time for each iteration
        """
        return self.score_time_averages * self.number_of_splits

    @property
    def score_time_total(self) -> float:
        """Total score time across all iterations."""
        return float(np.sum(self.iteration_score_times))

    @property
    def average_time_per_trial(self) -> float:
        """Average time per trial"""
        return float(np.mean(self.iteration_fit_times + self.iteration_score_times))

    @property
    def total_time(self) -> float:
        """Total time it took across all trials"""
        return self.fit_time_total + self.score_time_total


# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-public-methods


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
