"""This module contains helper functions when working with sklearn (scikit-learn) objects;
in particular, for evaluating models"""
# pylint: disable=too-many-lines
import math
import warnings
from re import match
from typing import Tuple, Union, Optional, List, Dict
import re

import numpy as np
import pandas as pd
import plotly
import scipy.stats as st
import seaborn as sns
from plotly.graph_objs import _figure  # noqa
import plotly.express as px
import yaml
from matplotlib import pyplot as plt
from pandas.io.formats.style import Styler
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import confusion_matrix, roc_auc_score, r2_score
from sklearn.model_selection._search import BaseSearchCV  # noqa
from sklearn.preprocessing import MinMaxScaler

import helpsk.color as hcolor
import helpsk.pandas_style as hstyle
import helpsk.string as hstring
# pylint: disable=too-many-locals
from helpsk.pandas import get_numeric_columns, get_non_numeric_columns
from helpsk.exceptions import HelpskParamValueError
from helpsk.plot import STANDARD_WIDTH_HEIGHT, GOLDEN_RATIO
from helpsk.validation import assert_true

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=Warning)
    from statsmodels import api as sm  # https://github.com/statsmodels/statsmodels/issues/3814


# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-public-methods
class MLExperimentResults:
    """
    This class contains the logic to explore the results of a machine learning experiment. There are also
        functions that parse the information from a BaseSearchCV object (e.g. GridSearchCV,
        RandomizedSearchCV, BayesSearchCV).

    The results can be saved/loaded to/from a standardized YAML file via `to_yaml_file()`/`from_yaml_file()`

    A MLExperimentResults object can be instantiated via `from_sklearn_searchCV()`, `__init__()`, or
        `from_yaml_file()`. The `__init__()` function allows people to transform experiment results from
        other sources into a dictionary (see __init__ function documentation for details on expected format)
        so that it can be used with this class.

    Trial: A single training iteration on a specific set of hyper-parameters (e.g. a given set of model
        parameters, transformations, etc.)

    Experiment: a collection of trials. An experiment can be a single run using, for example, GridSearchCV or
        BayesSearchCV. If a GridSearchCV experiment has 30 combinations of hyper-parameters to try, each of
        those combinations is a Trial. The collection of 30 combinations is the experiment.
    """

    def __init__(self, experiment_dict: dict):
        """This method creates a MLExperimentResults object from a dictionary.

        Args:
            experiment_dict:
                a dictionary with the following format, as an example:

            - `description`: any text to provide context into the ML experiment
            - `cross_validation_type`: any text
            - `higher_score_is_better`: boolean indicating whether or not a higher score/metric is better
                e.g. True for AUC, False for RMSE
            - `number_of_splits`: This is the number of times the model is trained in a single trial i.e.
                single cross-validation session. For example, a 5-fold 2-repeat CV has 10 splits.
            - `score_names`: the names of the score(s)
            - `parameter_names`: the names of the parameters
            - `parameter_names_mapping`: a mapping between actual parameters and friendlier names
            - `test_score_rankings`: for each score, a list of rankings. For example, if a list with the
                values of `[5, 6, 7, 8, 3, 4, 1, 2]` means that the first trial (index 0) ranked 5. The
                seventh trial (index 6) was the best score (highest or lowest score depending on the value of
                `higher_score_is_better`)
            - `test_score_averages`: for each score, the average score value among all splits (i.e. across the
                cross-validation run) for a single trial
            - `test_score_standard_deviations`: for each score, the standard deviation score among all splits
                (i.e. across the cross-validation run) for a single trial
            - `train_score_averages`: (optional) same as test_score_xxx above, but for training scores
            - `train_score_standard_deviations`: (optional) same as test_score_xxx above, but for training
                standard deviations
            - `trials`: A list of dictionary. Each item in the list (i.e. the dictionary)
                corresponds to a single trial and a single set of hyper-parameters. The dictionary has an item
                for each of the hyper-parameters, the name of the hyper-parameter as the key and the
                corresponding value
            - `timings`: (optional) A list of timings for fitting/scoring

            ```
            {
                'description': 'test description',
                'cross_validation_type': "<class 'sklearn.model_selection._search.GridSearchCV'>",
                'higher_score_is_better': True,
                'number_of_splits': 3,
                'score_names': ['ROC/AUC', 'F1', 'Pos. Pred. Val', 'True Pos. Rate'],
                'parameter_names': ['model__max_features', 'model__n_estimators',
                                    'preparation__non_numeric_pipeline__encoder_chooser__transformer'],
                'parameter_names_mapping': {
                    'model__max_features': 'max_features',
                    'model__n_estimators': 'n_estimators',
                    'preparation__non_numeric_pipeline__encoder_chooser__transformer': 'encoder'
                },
                'test_score_rankings': {
                    'ROC/AUC': [5, 6, 7, 8, 3, 4, 1, 2],
                    'F1': [5, 6, 7, 8, 3, 4, 2, 1],
                    'Pos. Pred. Val': [5, 6, 7, 8, 1, 4, 3, 2],
                    'True Pos. Rate': [5, 6, 7, 8, 3, 4, 2, 1]
                },
                'test_score_averages': {
                    'ROC/AUC': [nan, nan, nan, nan, 0.7163279567387703, 0.7072491564040325,
                                0.7458632917328131, 0.7451951355826116],
                    'F1': [nan, nan, nan, nan, 0.4039887486499992, 0.3854065126792399, 0.4137381215646683,
                           0.4308356053261986],
                    'Pos. Pred. Val': [nan, nan, nan, nan, 0.6297753300113645, 0.5468751302897644,
                                       0.6015296073435609, 0.6113328664799252],
                    'True Pos. Rate': [nan, nan, nan, nan, 0.2993876710919008, 0.29902233199547185,
                                       0.31574945970978696, 0.3363795410106]
                },
                'test_score_standard_deviations': {
                    'ROC/AUC': [nan, nan, nan, nan, 0.015733457430657828, 0.01795608557324064,
                                0.021358294698372086, 0.0019082263491945234],
                    'F1': [nan, nan, nan, nan, 0.05090125909373751, 0.07559891495722325, 0.04259039056830544,
                           0.041365868651721927],
                    'Pos. Pred. Val': [nan, nan, nan, nan, 0.026117735279684196, 0.06663645906293561,
                                       0.041309705841820844, 0.06827042947757211],
                    'True Pos. Rate': [nan, nan, nan, nan, 0.0497780260716827, 0.07150204558331273,
                                       0.038463218144256496, 0.0460214315468713]
                },
                'train_score_averages': {
                    'ROC/AUC': [nan, nan, nan, nan, 0.9998943476077663, 0.9992031557521965, 1.0, 1.0],
                    'F1': [nan, nan, nan, nan, 0.9765745804947125, 0.9658091912335657, 1.0, 1.0],
                    'Pos. Pred. Val': [nan, nan, nan, nan, 1.0, 0.9956132756132755, 1.0, 1.0],
                    'True Pos. Rate': [nan, nan, nan, nan, 0.9543319834542148, 0.9377688364562852, 1.0, 1.0]
                },
                'train_score_standard_deviations': {
                    'ROC/AUC': [nan, nan, nan, nan, 7.439517969409452e-05, 0.0005401862948682569,
                                6.409875621278546e-17, 0.0],
                    'F1': [nan, nan, nan, nan, 0.007673191451896241, 0.005478380082149172, 0.0, 0.0],
                    'Pos. Pred. Val': [nan, nan, nan, nan, 0.0, 0.0031026880007722375, 0.0, 0.0],
                    'True Pos. Rate': [nan, nan, nan, nan, 0.014730511057304973, 0.00871005201368303, 0.0,
                                       0.0]
                },
                'trials': [
                    {
                        'model__max_features': 100,
                        'model__n_estimators': 10,
                        'preparation__non_numeric_pipeline__encoder_chooser__transformer': 'OneHotEncoder()'
                    },
                    {
                        'model__max_features': 100,
                        'model__n_estimators': 10,
                        'preparation__non_numeric_pipeline__encoder_chooser__transformer':
                            'CustomOrdinalEncoder()'
                    },
                    {
                        'model__max_features': 100,
                        'model__n_estimators': 50,
                        'preparation__non_numeric_pipeline__encoder_chooser__transformer': 'OneHotEncoder()'
                    },
                    {
                        'model__max_features': 100,
                        'model__n_estimators': 50,
                        'preparation__non_numeric_pipeline__encoder_chooser__transformer':
                            'CustomOrdinalEncoder()'
                    },
                    {
                        'model__max_features': 'auto',
                        'model__n_estimators': 10,
                        'preparation__non_numeric_pipeline__encoder_chooser__transformer': 'OneHotEncoder()'
                    },
                    {
                        'model__max_features': 'auto',
                        'model__n_estimators': 10,
                        'preparation__non_numeric_pipeline__encoder_chooser__transformer':
                            'CustomOrdinalEncoder()'
                    },
                    {
                        'model__max_features': 'auto',
                        'model__n_estimators': 50,
                        'preparation__non_numeric_pipeline__encoder_chooser__transformer': 'OneHotEncoder()'
                    },
                    {
                        'model__max_features': 'auto',
                        'model__n_estimators': 50,
                        'preparation__non_numeric_pipeline__encoder_chooser__transformer':
                            'CustomOrdinalEncoder()'
                    }
                ],
                'timings': {
                    'fit time averages': [0.02420210838317871, 0.047980546951293945, 0.053688764572143555,
                                          0.07731231053670247, 0.03327099482218424, 0.05605125427246094,
                                          0.09772396087646484, 0.12112665176391602],
                    'fit time standard deviations': [0.0015644331364039865, 0.0007074507470172489,
                                                     0.00019471963214876084, 0.0006047002838356386,
                                                     0.0014654790113907423, 0.000618793945947827,
                                                     0.00044132423197131406, 0.0007073387039515561],
                    'score time averages': [0.0, 0.0, 0.0, 0.0, 0.020992437998453777, 0.05231904983520508,
                                            0.03004598617553711, 0.06252868970235188],
                    'score time standard deviations': [0.0, 0.0, 0.0, 0.0, 0.0003703884785102133,
                                                       0.0004282967971518099, 0.00014070830842031841,
                                                       0.0004855695604911932]
                }
            }
            ```
        """
        self._dict = experiment_dict
        self._dataframe = None
        self._labeled_dataframe = None

    # pylint: disable=too-many-branches
    # pylint: disable=too-many-statements
    @classmethod
    def from_sklearn_search_cv(cls,
                               searcher: BaseSearchCV,
                               higher_score_is_better: bool = True,
                               description: str = "",
                               parameter_name_mappings: Union[dict, None] = None):
        """
        This function extracts the results from a SearchCV object (e.g.
        sklearn.model_selection.GridSearch/RandomSearch, skopt.BayesSearchCV), which are converted to
        a dictionary with the hierarchy expected by the MLExperimentResults class.

        Args:
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

            description:
                An optional string to save in the dictionary; the intent

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
        def string_if_not_number(obj):
            if isinstance(obj, (int, float, complex)):
                return obj
            # convert to a string, but convert e.g. `XGBoostClassifier(parameter=x, etc)` to
            # `XGBoostClassifier()`
            string_value = str(obj)
            string_value = string_value.replace('\n', '')
            string_value = re.sub(r'Classifier\(.+\)', 'Classifier()', string_value)
            string_value = re.sub(r'Regressor\(.+\)', 'Regressor()', string_value)
            string_value = re.sub(r'Regression\(.+\)', 'Regression()', string_value)
            string_value = string_value.replace("OneHotEncoder(handle_unknown='ignore')", "OneHotEncoder()")
            return string_value

        cv_results_dict = {
            'description': description,
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

        parameter_names = []
        # If there are multiple search spaces, then there might be different, but overlapping, parameters
        # in each search space. We have to collect all of the parameters and then get the unique list (via
        # `set()`).
        # Note that i'm doing this with nested loops (rather than simply using `set()`) so that I can retain
        # the exact order (set() does not retain order)
        for params in searcher.cv_results_['params']:
            for key in params.keys():
                if key not in parameter_names:
                    parameter_names += [key]

        # if we pass in parameter mappings, make sure each parameter is accounted for i.e. the keys of the
        # mappings should be identical (in any order, (via `set()`)) to the parameter names
        if parameter_name_mappings:
            param_mapping_keys = list(parameter_name_mappings.keys())
            assert_true(set(param_mapping_keys) == set(parameter_names))  # ensure equal, unordered
            # use the keys from the mappings as parameter names (rather than just passing param names
            # directly) so that the order of the param names is retained; which will be used to determine
            # order in e.g. to_data_frame()
            cv_results_dict['parameter_names'] = param_mapping_keys
            cv_results_dict['parameter_names_mapping'] = parameter_name_mappings
        else:
            cv_results_dict['parameter_names'] = parameter_names

        number_of_trials = len(searcher.cv_results_['mean_fit_time'])

        # convert test scores to dictionaries
        if len(score_names) == 1:
            test_score_ranking = searcher.cv_results_['rank_test_score'].tolist()
            test_score_averages = searcher.cv_results_['mean_test_score'].tolist()
            test_score_standard_deviations = searcher.cv_results_['std_test_score'].tolist()

            assert_true(len(test_score_ranking) == number_of_trials)
            assert_true(len(test_score_averages) == number_of_trials)
            assert_true(len(test_score_standard_deviations) == number_of_trials)

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

                assert_true(len(rankings) == number_of_trials)
                assert_true(len(averages) == number_of_trials)
                assert_true(len(standard_deviations) == number_of_trials)

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

                assert_true(len(train_score_averages) == number_of_trials)
                assert_true(len(train_score_standard_deviations) == number_of_trials)

                cv_results_dict['train_score_averages'] = {score_names[0]: train_score_averages}
                cv_results_dict['train_score_standard_deviations'] = {score_names[0]:
                                                                          train_score_standard_deviations}
            else:
                averages_dict = {}
                standard_deviations_dict = {}
                for score in score_names:
                    averages = searcher.cv_results_['mean_train_' + score].tolist()
                    standard_deviations = searcher.cv_results_['std_train_' + score].tolist()

                    assert_true(len(averages) == number_of_trials)
                    assert_true(len(standard_deviations) == number_of_trials)

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

        assert_true(len(searcher.cv_results_['params']) == number_of_trials)

        cv_results_dict['trials'] = [
            {key: string_if_not_number(value) for key, value in searcher.cv_results_['params'][index].items()}
            for index in range(len(searcher.cv_results_['params']))
        ]

        fit_time_averages = searcher.cv_results_['mean_fit_time'].tolist()
        fit_time_standard_deviations = searcher.cv_results_['std_fit_time'].tolist()
        score_time_averages = searcher.cv_results_['mean_score_time'].tolist()
        score_time_standard_deviations = searcher.cv_results_['std_score_time'].tolist()

        assert_true(len(fit_time_averages) == number_of_trials)
        assert_true(len(fit_time_standard_deviations) == number_of_trials)
        assert_true(len(score_time_averages) == number_of_trials)
        assert_true(len(score_time_standard_deviations) == number_of_trials)

        cv_results_dict['timings'] = {'fit time averages': fit_time_averages,
                                      'fit time standard deviations': fit_time_standard_deviations,
                                      'score time averages': score_time_averages,
                                      'score time standard deviations': score_time_standard_deviations}

        return MLExperimentResults(experiment_dict=cv_results_dict)

    @classmethod
    def from_yaml_file(cls, yaml_file_name):
        """This method creates a MLExperimentResults object from a yaml file created by `to_yaml_file()`"""
        with open(yaml_file_name, 'r') as file:
            cv_dict = yaml.safe_load(file)

        return MLExperimentResults(experiment_dict=cv_dict)

    def to_yaml_file(self, yaml_file_name: str):
        """This method saves the self._cv_dict dictionary to a yaml file."""
        with open(yaml_file_name, 'w') as file:
            yaml.dump(self._dict, file, default_flow_style=False, sort_keys=False)

    def to_dataframe(self,
                     sort_by_score: bool = True,
                     exclude_zero_variance_params: bool = True,
                     query: str = None) -> pd.DataFrame:
        """This function converts the score information from the SearchCV object into a pd.DataFrame.

        Args:
            sort_by_score:
                if True, sorts the dataframe starting with the best (primary) score to the worst score.
                Secondary scores are not considered. If False, leave the results in the order that the trials
                were executed.

            exclude_zero_variance_params:
                if True, exclude columns that only have 1 unique value

            query:
                a string that queries the resulting pd.DataFrame (passed directly to pandas `.query()`)

                For example, if multiple models are being searched, and `model` is a parameter name (and a
                resulting column), then a query value of `"model == 'LogisticRegression()'"` would return
                only the rows where the value of the `model` column matches `LogisticRegression()`.

        Returns:
            a DataFrame containing score information for each cross-validation trial. A single row
            corresponds to one trial (i.e. one set of hyper-parameters that were cross-validated).
        """
        if self._dataframe is None:
            for score_name in self.score_names:
                confidence_intervals = st.t.interval(alpha=0.95,  # confidence interval
                                                     # number_of_splits is sample-size
                                                     df=self.number_of_splits - 1,  # degrees of freedom
                                                     loc=self.test_score_averages[score_name],
                                                     scale=self.score_standard_errors(score_name=score_name))

                # only give confidence intervals for the primary score
                self._dataframe = pd.concat([
                        self._dataframe,
                        pd.DataFrame({score_name + " Mean": self.test_score_averages[score_name],
                                      score_name + " 95CI.LO": confidence_intervals[0],
                                      score_name + " 95CI.HI": confidence_intervals[1]})
                    ],
                    axis=1
                )

            self._dataframe = pd.concat([self._dataframe,
                                         pd.DataFrame.from_dict(self.trials)[self.parameter_names_original]],  # noqa
                                        axis=1)

            if self.parameter_names_mapping:
                self._dataframe = self._dataframe.rename(columns=self.parameter_names_mapping)

        copy = self._dataframe.copy(deep=True)

        if sort_by_score:
            copy = copy.iloc[self.best_trial_indexes]

        # need to query after we sort so that e.g. best_trial_indexes correspond to same rows
        if query:
            copy = copy.query(query)

        # need to do this after querying because the it's possible a columns could become zero-variance after
        # the query
        if exclude_zero_variance_params:
            zero_variance_columns = [x for x in self.parameter_names if len(copy[x].unique()) == 1]
            copy = copy.drop(columns=zero_variance_columns)

        copy = copy.dropna(axis=1, how='all')  # drop columns that have all NAs

        return copy

    # pylint: disable=too-many-arguments
    def to_formatted_dataframe(self,
                               round_by: int = 3,
                               num_rows: int = 100,
                               primary_score_only: bool = False,
                               exclude_zero_variance_params: bool = True,
                               query: str = None,
                               include_rank=False,
                               return_style: bool = True,
                               sort_by_score: bool = True) -> Union[pd.DataFrame, Styler]:
        """This function converts the score information from the SearchCV object into a pd.DataFrame or a
        Styler object, formatted accordingly.

        The Hyper-Parameter columns will be highlighted in blue where the primary
        score (i.e. first column) for the trial (i.e. the row i.e. the combination of hyper-parameters
        that were cross validated) is within 1 standard error of the top primary score (i.e. first column
        first row).

        Args:
            round_by:
                the number of digits to round by for the score columns (does not round the parameter columns)
            num_rows:
                the number of rows to return in the resulting DataFrame.
            primary_score_only:
                if True, then only include the primary score.
            exclude_zero_variance_params:
                if True, exclude columns that only have 1 unique value
            query:
                a string that queries the resulting pd.DataFrame (passed directly to pandas `.query()`)

                For example, if multiple models are being searched, and `model` is a parameter name (and a
                resulting column), then a query value of `"model == 'LogisticRegression()'"` would return
                only the rows where the value of the `model` column matches `LogisticRegression()`.
            include_rank:
                if True, include a column to show the index of score rank.
            return_style:
                If True, return Styler object, else return pd.DataFrame
            sort_by_score:
                if True, sorts the dataframe starting with the best (primary) score to the worst score.
                Secondary scores are not considered.

        Returns:
            Returns either pd.DataFrame or pd.DataFrame.Styler.
        """
        cv_dataframe = self.to_dataframe(sort_by_score=sort_by_score,
                                         exclude_zero_variance_params=exclude_zero_variance_params,
                                         query=query)

        cv_dataframe = cv_dataframe.head(num_rows)

        # if, for example, we are querying (and returning the style, i.e. highlighting rows within 1 standard
        # error), then we need to only include the indexes that remain after querying
        indexes_within_1_standard_error = [x for x in self.indexes_within_1_standard_error
                                           if x in cv_dataframe.index]

        score_columns = list(cv_dataframe.columns[cv_dataframe.columns.str.endswith((' Mean',
                                                                                     ' 95CI.LO',
                                                                                     ' 95CI.HI'))])
        if primary_score_only:
            columns_to_drop = [x for x in score_columns if not x.startswith(self.primary_score_name)]
            cv_dataframe = cv_dataframe.drop(columns=columns_to_drop)

        cv_dataframe = cv_dataframe.round(dict(zip(score_columns, [round_by] * len(score_columns))))

        final_columns = list(cv_dataframe.columns)  # save for style logic

        if include_rank:
            cv_dataframe['rank'] = list(range(1, cv_dataframe.shape[0] + 1))
            final_columns = ['rank'] + final_columns
            cv_dataframe = cv_dataframe[final_columns]

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

            # highlight trials whose primary score (i.e. first column of `results` dataframe) is within
            # 1 standard error of the top primary score (i.e. first column first row).
            # pylint: disable=invalid-name, unused-argument
            def highlight_cols(s):   # noqa
                return 'background-color: %s' % hcolor.Colors.PASTEL_BLUE.value

            # we might have removed columns (e.g. that don't have any variance) so check that the columns
            # were in the final set
            columns_to_highlight = [x for x in self.parameter_names if x in final_columns]
            cv_dataframe.applymap(highlight_cols,
                             subset=pd.IndexSlice[indexes_within_1_standard_error,
                                                  columns_to_highlight])

        return cv_dataframe

    def to_labeled_dataframe(self, query: str = None) -> pd.DataFrame:
        """Returns a pd.DataFrame similar to `to_dataframe()` with additional columns 'Trial Index' and
        'labels' (which are the labels corresponding to the `trial_label` property and collapse all the
        name/values for the hyper-parameters into a single string)

        This function is mainly used internally to generate graphs, but is useful for users creating custom
        graphs that are not yet implemented by the class.

        Args:
            query:
                a string that queries the resulting pd.DataFrame (passed directly to pandas `.query()`)

                For example, if multiple models are being searched, and `model` is a parameter name (and a
                resulting column), then a query value of `"model == 'LogisticRegression()'"` would return
                only the rows where the value of the `model` column matches `LogisticRegression()`.
        """
        sort_by_score = False  # leave original trial order
        labeled_dataframe = self.to_dataframe(sort_by_score=sort_by_score, query=query)
        columns = labeled_dataframe.columns.to_list()  # cache columns to move Iteration column to front
        labeled_dataframe['Trial Index'] = np.arange(1, labeled_dataframe.shape[0] + 1)
        labeled_dataframe = labeled_dataframe[['Trial Index'] + columns]
        # create the labels that will be used in the plotly hover text
        # only include the labels that correspond to the remaining trials after `query`
        trial_labels = self.trial_labels(order_from_best_to_worst=sort_by_score)
        trial_labels = [trial_labels[x] for x in labeled_dataframe.index]
        labeled_dataframe['label'] = [x.replace('{', '<br>').replace(', ', '<br>').replace('}', '')
                                      for x in trial_labels]

        return labeled_dataframe

    ####
    # The following properties expose the highest levels of the underlying dictionary/yaml
    ####
    @property
    def description(self):
        """the description passed to `description`."""
        return self._dict['description']

    @property
    def higher_score_is_better(self):
        """The value passed to `higher_score_is_better`."""
        return self._dict['higher_score_is_better']

    @property
    def cross_validation_type(self) -> str:
        """The string representation of the SearchCV object."""
        return self._dict['cross_validation_type']

    @property
    def number_of_splits(self) -> int:
        """This is the number of times the model is trained in a single trial i.e. single cross-validation
        session. For example, a 5-fold 2-repeat CV has 10 splits."""
        return self._dict['number_of_splits']

    @property
    def score_names(self) -> list:
        """Returns a list of the names of the scores"""
        return self._dict['score_names']

    @property
    def parameter_names_original(self) -> list:
        """Returns the original parameter names (i.e. the path generated by the scikit-learn pipelines."""
        return self._dict['parameter_names']

    @property
    def parameter_names(self) -> list:
        """This property returns either the original parameter names if no `parameter_names_mapping` was
        provided, or it returns the new parameter names (i.e. the values from `parameter_names_mapping`)."""
        if self.parameter_names_mapping:
            return list(self.parameter_names_mapping.values())

        return self.parameter_names_original

    @property
    def parameter_names_mapping(self) -> dict:
        """The dictionary passed to `parameter_name_mappings` which is used to convert the original names
        to more friendly names, specified as the the values."""
        return self._dict.get('parameter_names_mapping')

    @property
    def test_score_rankings(self) -> dict:
        """The rankings of each of the test scores, from the searcher.cv_results_ object."""
        return self._dict['test_score_rankings']

    @property
    def test_score_averages(self) -> dict:
        """The test score averages, from the searcher.cv_results_ object."""
        return self._dict['test_score_averages']

    @property
    def test_score_standard_deviations(self) -> dict:
        """The test score standard deviations, from the searcher.cv_results_ object."""
        return self._dict['test_score_standard_deviations']

    @property
    def train_score_averages(self) -> dict:
        """The training score averages, from the searcher.cv_results_ object, if provided."""
        return self._dict.get('train_score_averages')

    @property
    def train_score_standard_deviations(self) -> dict:
        """The training score standard deviations, from the searcher.cv_results_ object, if provided."""
        return self._dict.get('train_score_standard_deviations')

    @property
    def trials(self) -> list:
        """The "trials" i.e. the hyper-parameter combinations in order of execution."""
        return self._dict['trials']

    def trial_labels(self, order_from_best_to_worst=True) -> List[str]:
        """An trial is a set of hyper-parameters that were cross validated. The corresponding label for
        each trial is a single string containing all of the hyper-parameter names and values in the format
        of `{param1: value1, param2: value2}`, excluding hyper-parameters that only have a single value.

        Args:
            order_from_best_to_worst: if True, returns the labels in order from the best score to the worst
            score, which should match the ordered of .to_dataframe() or .to_formatted_dataframe()` using the
            default values for those functions. If False, returns the labels in order that they were ran by
            the cross validation object.

        Returns:
            a pd.Series the same length as `number_of_trials` containing a str
        """
        def create_hyper_param_labels(trial) -> list:
            """Creates a list of strings that represent the name/value pair for each hyper-parameter."""
            return [f"{self.parameter_names_mapping[x] if self.parameter_names_mapping and x in self.parameter_names_mapping else x}: {trial[x]}"  # pylint: disable=line-too-long  # noqa
                    # for parameter spaces that have multiple models (and different parameters per model),
                    # we need to make sure that the parameter name is actually in the trial
                    # e.g. the parameter name could correspond to the logistic regression space but we could
                    # be iterating over the xgboost space
                    for x in self.parameter_names_original if x in trial]
        # create_hyper_param_labels(trial=self.trials[0])

        def create_trial_label(trial) -> str:
            return f"{{{hstring.collapse(create_hyper_param_labels(trial), separate=', ')}}}"
        # create_trial_label(trial=self.trials[0])

        labels = [create_trial_label(x) for x in self.trials]

        if order_from_best_to_worst:
            labels = [x for _, x in sorted(zip(self.trial_rankings, labels))]

        return labels

    @property
    def timings(self) -> dict:
        """The timings providing by searcher.cv_results_."""
        return self._dict['timings']

    ####
    # The following properties are additional helpers
    ####
    @property
    def number_of_trials(self) -> int:
        """"A single trial contains the cross validation results for a single set of hyper-parameters. The
        'number of trials' is basically the number of combinations of different hyper-parameters that were
        cross validated."""
        return len(self.trials)

    @property
    def numeric_parameters(self) -> List[str]:
        """Returns a list of parameters names corresponding to numeric columns."""
        return [x for x in get_numeric_columns(dataframe=self.to_dataframe()) if x in self.parameter_names]

    @property
    def non_numeric_parameters(self) -> List[str]:
        """Returns a list of parameters names corresponding to non-numeric columns."""
        return [x for x in get_non_numeric_columns(dataframe=self.to_dataframe())
                if x in self.parameter_names]

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
        the average score (across all splits) for each trial. Note that the average scores are
        the weighted averages
        https://stackoverflow.com/questions/44947574/what-is-the-meaning-of-mean-test-score-in-cv-result"""
        return np.array(self.test_score_averages[self.primary_score_name])

    def score_standard_errors(self, score_name: str) -> np.array:
        """The first scorer passed to the SearchCV will be treated as the primary score. This property returns
        the standard error associated with the mean score of each trial, for the primary score."""
        score_standard_deviations = self.test_score_standard_deviations[score_name]
        return np.array(score_standard_deviations) / math.sqrt(self.number_of_splits)

    @property
    def trial_rankings(self) -> np.array:
        """The ranking of the corresponding index, in terms of best to worst "primary" (i.e. first) score.

        For example, assume this property returned the following list :
            [5, 6, 7, 8, 3, 4, 1, 2]
            This means that the 6th index/trial had the highest ranking (1); and that the 3rd index had
            the worst ranking (8)

        This differs from `best_trial_indexes` which returns the order of indexes (of the trials) from best to
        worst.
        So in the example above, the first value returned in the `best_trial_indexes` array would be
        6 because the best score (across trials) is at index 6 (i.e. the 7th trial).
        The last value in the array returned by `best_trial_indexes` would be 3, because the worst score is at
        index 3 (i.e. the 4th trial).

        Note that `trial_rankings` starts at 1 while best_trial_indexes starts at 0.
        """
        return np.array(self.test_score_rankings[self.primary_score_name])

    @property
    def best_trial_indexes(self) -> np.array:
        """The indexes of best to worst "primary" (i.e. first) scores. See documentation for
        `trial_rankings` to understand the differences between the two properties."""
        return np.argsort(self.trial_rankings)

    @property
    def best_score_index(self) -> int:
        """The index of best "primary" (i.e. first) score."""
        return self.best_trial_indexes[0]

    @property
    def best_params(self) -> dict:
        """
        The *best* score. "Best" could be the highest or lowest depending on `higher_score_is_better`)
        associated with the "primary" (i.e. first) score.
        """
        best_params = self.trials[self.best_score_index].copy()

        if self.parameter_names_mapping:
            best_params = {value: best_params[key] for key, value in self.parameter_names_mapping.items()
                           if key in best_params}

        return best_params

    @property
    def best_score(self) -> float:
        """
        The "best" score (could be the highest or lowest depending on `higher_score_is_better`) associated
        with the primary score.
        """
        return self.primary_score_averages[self.best_score_index]

    @property
    def best_standard_error(self) -> float:
        """The standard error associated with the best of the primary scores"""
        return self.score_standard_errors(score_name=self.primary_score_name)[self.best_score_index]

    @property
    def indexes_within_1_standard_error(self) -> list:
        """Returns the trial indexes where the primary scores (i.e. first scorer
        passed to SearchCV object; i.e. first column of the to_dataframe() DataFrame) are within 1 standard
        error of the highest primary score."""
        cv_dataframe = self.to_dataframe(sort_by_score=True)

        if self.higher_score_is_better:
            return list(cv_dataframe.index[cv_dataframe.iloc[:, 0] >=
                                           self.best_score - self.best_standard_error])

        return list(cv_dataframe.index[cv_dataframe.iloc[:, 0] <=
                                       self.best_score + self.best_standard_error])

    @property
    def fit_time_averages(self) -> np.array:
        """
        Returns a list of floats; one value for each trial (i.e. a single set of hyper-params).
        Each value is the average number of seconds that the trial took to fit the model, per split
        (i.e. the average fit time of all splits).
        """
        return np.array(self.timings['fit time averages'])

    @property
    def fit_time_standard_deviations(self) -> np.array:
        """
        Returns a list of floats; one value for each trial (i.e. a single set of hyper-params).
        Each value is the standard deviation of seconds that the trial took to fit the model, per split
        (i.e. the standard deviation of fit time across all splits).
        """
        return np.array(self.timings['fit time standard deviations'])

    @property
    def score_time_averages(self) -> np.array:
        """
        Returns a list of floats; one value for each trial (i.e. a single set of hyper-params).
        Each value is the average number of seconds that the trial took to score the model, per split
        (i.e. the average score time of all splits).
        """
        return np.array(self.timings['score time averages'])

    @property
    def score_time_standard_deviations(self) -> np.array:
        """
        Returns a list of floats; one value for each trial (i.e. a single set of hyper-params).
        Each value is the standard deviation of seconds that the trial took to score the model, per split
        (i.e. the standard deviation of score time across all splits).
        """
        return np.array(self.timings['score time standard deviations'])

    @property
    def trial_fit_times(self) -> np.array:
        """For each trial, it is the amount of time it took to fit the model.

        Calculated by Average fit time for each trial multiplied by the number of splits per trial.

        self.fit_time_averages * self.number_of_splits

        Returns:
            array containing the fit time for each trial
        """
        return self.fit_time_averages * self.number_of_splits

    @property
    def fit_time_total(self) -> float:
        """Total fit time across all trials."""
        return float(np.sum(self.trial_fit_times))

    @property
    def trial_score_times(self) -> np.array:
        """For each trial, it is the amount of time it took to score the model.

        Calculated by Average score time for each trial multiplied by the number of splits per trial.

        self.score_time_averages * self.number_of_splits

        Returns:
            array containing the score time for each trial
        """
        return self.score_time_averages * self.number_of_splits

    @property
    def score_time_total(self) -> float:
        """Total score time across all trials."""
        return float(np.sum(self.trial_score_times))

    @property
    def average_time_per_trial(self) -> float:
        """Average time per trial"""
        return float(np.mean(self.trial_fit_times + self.trial_score_times))

    @property
    def total_time(self) -> float:
        """Total time it took across all trials"""
        return self.fit_time_total + self.score_time_total

    # pylint: disable=dangerous-default-value
    def plot_performance_across_trials(self,
                                       size: str = None,
                                       color: str = None,
                                       color_continuous_scale: List[str] = px.colors.diverging.balance,
                                       facet_by: str = None,
                                       facet_num_col: int = 3,
                                       query: str = None,
                                       height: float = 600,
                                       width: float = 600 * GOLDEN_RATIO) -> plotly.graph_objects.Figure:
        """
        Returns a Plotly Figure (scatter-plot) of the primary score (y-axis) in order of trial execution
        (x-axis). Especially useful for e.g. BayesSearchCV.

        Args:
            size:
                The name of a hyper-parameter, the values of which will be used to determine the size of the
                corresponding points in the plot. This value is passed to plotly.
            color:
                The name of a hyper-parameter, the values of which will be used to determine the color of the
                corresponding points in the plot. This value is passed to plotly.
            color_continuous_scale:
                A list of strings that should contain valid CSS-colors. This value is passed to plotly.
            facet_by:
                name of the hyper-parameter to facet by.
            facet_num_col:
                the max number of columns in the graph to facet
            query:
                string to be passed to `to_dataframe()`; see documentation for that method.
            height:
                The height of the plot. This value is passed to plotly.
            width:
                The width of the plot. This value is passed to plotly.
        """
        score_column = self.primary_score_name + " Mean"

        title = "Performance Over Time (Across Trials)<br>" \
                "<sup>This graph shows the average CV score across all trials, in order of execution.</sup>"
        if size is not None:
            title = title + f"<br><sup>The size of the point corresponds to the value of <b>'{size}'</b>.</sup>"

        labeled_df = self.to_labeled_dataframe(query=query)
        if facet_by:
            labeled_df['Trial Index'] = labeled_df.groupby(facet_by)["Trial Index"].rank(method="first",
                                                                                          ascending=True)
        # only include the columns we need, so that we don't unnecessarily drop rows with NA (i.e. NAs in
        # columns not used in the graph)
        columns = [x for x in ['Trial Index', score_column, size, color, facet_by, 'label']
                   if x is not None]
        labeled_df = labeled_df[columns]
        labeled_df.dropna(axis=0, how='any', inplace=True)

        fig = px.scatter(
            data_frame=labeled_df,
            x='Trial Index',
            y=score_column,
            size=size,
            color=color,
            color_continuous_scale=color_continuous_scale,
            trendline='lowess',
            facet_col=facet_by,
            facet_col_wrap=facet_num_col,
            labels={
                score_column: f"Average CV Score<br>({self.primary_score_name})",
            },
            title=title,
            custom_data=['label'],
            height=height,
            width=width,
        )
        fig.update_traces(
            hovertemplate="<br>".join([
                "Trial Index: %{x}",
                score_column + ": " + "%{y}",
                "<br>Parameters: %{customdata[0]}",
            ])
        )
        return fig

    def plot_parameter_values_across_trials(self,
                                            query: str = None,
                                            height: float = 600,
                                            width: float = 600 * GOLDEN_RATIO) -> plotly.graph_objects.Figure:
        """
        Returns a Plotly Figure (scatter-plot per numeric parameter) of the parameter's values (y-axis) in
        order of trial execution (x-axis). Especially useful for e.g. BayesSearchCV.

        Args:
            query:
                string to be passed to `to_dataframe()`; see documentation for that method.
            height:
                The height of the plot. This value is passed to plotly.
            width:
                The width of the plot. This value is passed to plotly.
        """
        color_continuous_scale = px.colors.diverging.RdYlGn
        if not self.higher_score_is_better:
            color_continuous_scale = color_continuous_scale.reverse()

        score_column = self.primary_score_name + " Mean"

        labeled_df = self.to_labeled_dataframe(query=query)    
        labeled_long = pd.melt(labeled_df,
                               id_vars=['Trial Index', score_column, 'label'],
                               value_vars=[x for x in self.numeric_parameters if x in labeled_df.columns],
                               var_name='parameter')

        fig = px.scatter(
            data_frame=labeled_long,
            x='Trial Index',
            y='value',
            color=score_column,
            color_continuous_scale=color_continuous_scale,
            facet_col='parameter',
            facet_col_wrap=3,
            trendline='lowess',
            labels={
                'value': 'Parameter Value',
            },
            title="Parameter Values Evaluated Over Time (Across Trials)<br>"
                  "<sup>This graph shows the parameter values evaluated across all trials.<br>"
                  "The color corresponds to the average CV score associated with that trial/point.</sup>",
            custom_data=['label', score_column],
            height=height,
            width=width,
        )
        fig.update_traces(
            hovertemplate="<br>".join([
                "Trial Index: %{x}",
                "Parameter Value: %{y}",
                score_column + ": %{customdata[1]}",
                "<br>Parameters: %{customdata[0]}",
            ])
        )
        fig.update_yaxes(matches=None, showticklabels=True)
        return fig

    def plot_parallel_coordinates(self,
                                  include_all_scores: bool = True,
                                  query: str = None,
                                  height: float = 600,
                                  width: float = 600 * GOLDEN_RATIO) -> plotly.graph_objects.Figure:
        """
        Returns a Plotly Figure (parallel-coordinates) of the numeric parameters' values (y-axis), along with
        the corresponding score averages.

        Args:
            include_all_scores:
                Only applicable if the results have multiple scores.
                If True, includes all of the scores. If False, includes only the primary score.
            query:
                string to be passed to `to_dataframe()`; see documentation for that method.
            height:
                The height of the plot. This value is passed to plotly.
            width:
                The width of the plot. This value is passed to plotly.
        """
        color_continuous_scale = px.colors.diverging.RdYlGn
        if not self.higher_score_is_better:
            color_continuous_scale = color_continuous_scale.reverse()

        primary_score_column = self.primary_score_name + " Mean"

        if include_all_scores:
            score_columns = [score + " Mean" for score in self.score_names]
        else:
            score_columns = [primary_score_column]

        # NOTE: sort_by_score=False because there is a weird bug in plotly such that if the index is
        # not 0-x than the order seems to get messed up
        # https://github.com/plotly/plotly.py/issues/3576
        # https://github.com/plotly/plotly.py/issues/3577
        df = self.to_dataframe(sort_by_score=False, query=query)
        numeric_columns = [x for x in self.numeric_parameters if x in df.columns]
        fig = px.parallel_coordinates(
            df[numeric_columns + score_columns].dropna(axis=0),
            color=primary_score_column,
            color_continuous_scale=color_continuous_scale,
            height=height,
            width=width,
            title="Parallel Coordinates of Hyper-Parameters and Score Averages<br>"
        )
        # plotly.offline.plot(fig, filename='temp.html', auto_open=True)
        return fig

    def plot_scatter_matrix(self,
                            include_all_scores: bool = True,
                            query: str = None,
                            height: float = 600,
                            width: float = 600 * GOLDEN_RATIO) -> plotly.graph_objects.Figure:
        """
        Returns a Plotly Figure (scatter-matrix) of the all parameters and scores.

        Args:
            include_all_scores:
                Only applicable if the results have multiple scores.
                If True, includes all of the scores. If False, includes only the primary score.
            query:
                string to be passed to `to_dataframe()`; see documentation for that method.
            height:
                The height of the plot. This value is passed to plotly.
            width:
                The width of the plot. This value is passed to plotly.
        """
        color_continuous_scale = px.colors.diverging.RdYlGn
        if not self.higher_score_is_better:
            color_continuous_scale = color_continuous_scale.reverse()

        primary_score_column = self.primary_score_name + " Mean"

        if include_all_scores:
            score_columns = [score + " Mean" for score in self.score_names]
        else:
            score_columns = [primary_score_column]

        # NOTE: sort_by_score=False because there is a weird bug in plotly such that if the index is
        # not 0-x than the order seems to get messed up
        # https://github.com/plotly/plotly.py/issues/3576
        # https://github.com/plotly/plotly.py/issues/3577
        df = self.to_dataframe(sort_by_score=False, query=query)
        columns = [x for x in self.parameter_names if x in df.columns]
        fig = px.scatter_matrix(df[score_columns + columns],
                                color=primary_score_column,
                                color_continuous_scale=color_continuous_scale,
                                height=height,
                                width=width)
        return fig

    def plot_performance_numeric_params(self,
                                        query: str = None,
                                        height: float = 600,
                                        width: float = 600 * GOLDEN_RATIO) -> plotly.graph_objects.Figure:
        """
        Returns a Plotly Figure (scatter-matrix) of the each of the numeric parameters' values (x-axis) vs the
        primary score (y-axis).

        Args:
            query:
                string to be passed to `to_dataframe()`; see documentation for that method.
            height:
                The height of the plot. This value is passed to plotly.
            width:
                The width of the plot. This value is passed to plotly.
        """
        color_continuous_scale = px.colors.diverging.RdYlGn
        if not self.higher_score_is_better:
            color_continuous_scale = color_continuous_scale.reverse()

        primary_score_column = self.primary_score_name + " Mean"

        df = self.to_labeled_dataframe(query=query)
        columns = [x for x in self.numeric_parameters if x in df.columns]
        labeled_long = pd.melt(df,
                               id_vars=[primary_score_column, 'label'],
                               value_vars=columns,
                               var_name='parameter')

        fig = px.scatter(
            data_frame=labeled_long,
            x='value',
            y=primary_score_column,
            color=primary_score_column,
            color_continuous_scale=color_continuous_scale,
            facet_col='parameter',
            facet_col_wrap=2,
            trendline='lowess',
            labels={
                'value': 'Parameter Value',
            },
            title="Variable Performance<br><sup>Numeric Parameters</sup>",
            custom_data=['label', primary_score_column],
            height=height,
            width=width
        )
        fig.update_traces(
            hovertemplate="<br>".join([
                "Parameter Value: %{x}",
                primary_score_column + ": %{customdata[1]}",
                "<br>Parameters: %{customdata[0]}",
            ])
        )
        fig.update_xaxes(matches=None, showticklabels=True)
        return fig

    def plot_performance_non_numeric_params(self,
                                            query: str = None,
                                            height: float = 600,
                                            width: float = 600 * GOLDEN_RATIO) -> plotly.graph_objects.Figure:
        """
        Returns a Plotly Figure (box-plot) of the each of the non-numeric parameters' values (x-axis) vs
        the primary score (y-axis).

        Args:
            query:
                string to be passed to `to_dataframe()`; see documentation for that method.
            height:
                The height of the plot. This value is passed to plotly.
            width:
                The width of the plot. This value is passed to plotly.
        """
        # color_continuous_scale = px.colors.diverging.RdYlGn
        # if not self.higher_score_is_better:
        #     color_continuous_scale = color_continuous_scale.reverse()

        primary_score_column = self.primary_score_name + " Mean"

        df = self.to_labeled_dataframe(query=query)
        columns = [x for x in self.non_numeric_parameters if x in df.columns]
        labeled_long = pd.melt(df,
                               id_vars=[primary_score_column, 'label'],
                               value_vars=columns,
                               var_name='parameter')

        scatter = px.scatter(
            data_frame=labeled_long,
            x='value',
            y=primary_score_column,
            # color=primary_score_column,
            # color_continuous_scale=color_continuous_scale,
            facet_col='parameter',
            facet_col_wrap=2,
            labels={
                'value': 'Parameter Value',
            },
            title="Variable Performance<br><sup>Non-Numeric Parameters</sup>",
            custom_data=['label', primary_score_column],
            height=height,
            width=width,
        )
        scatter.update_traces(
            hovertemplate="<br>".join([
                "Parameter Value: %{x}",
                primary_score_column + ": %{customdata[1]}",
                "<br>Parameters: %{customdata[0]}",
            ])
        )
        scatter.update_xaxes(matches=None, showticklabels=True)
        fig = px.box(
            data_frame=labeled_long,
            x='value',
            y=primary_score_column,
            facet_col='parameter',
            facet_col_wrap=2,
            labels={
                'value': 'Parameter Value',
            },
            title="Variable Performance<br><sup>Non-Numeric Parameters</sup>",
            custom_data=['label', primary_score_column],
            height=height,
            width=width
        )
        for x in range(len(scatter.data)):
            fig.add_trace(scatter.data[x])

        fig.update_xaxes(matches=None, showticklabels=True)
        return fig

    def plot_score_vs_parameter(self,
                                parameter,
                                query: str = None,
                                size=None,
                                color=None,
                                height: float = 600,
                                width: float = 600 * GOLDEN_RATIO) -> plotly.graph_objects.Figure:
        """
        Returns a Plotly Figure (scatter-plot) of the primary score (y-axis) vs a given parameter (x-axis).

        Args:
            parameter:
                The name of a hyper-parameter to plot against the score, on the x-axis.
            query:
                string to be passed to `to_dataframe()`; see documentation for that method.
            size:
                The name of a hyper-parameter, the values of which will be used to determine the size of the
                corresponding points in the plot. This value is passed to plotly.
            color:
                The name of a hyper-parameter, the values of which will be used to determine the color of the
                corresponding points in the plot. This value is passed to plotly.
            height:
                The height of the plot. This value is passed to plotly.
            width:
                The width of the plot. This value is passed to plotly.
        """
        color_continuous_scale = px.colors.diverging.balance
        if not self.higher_score_is_better:
            color_continuous_scale = color_continuous_scale.reverse()

        primary_score_column = self.primary_score_name + " Mean"
        title = f"Primary Score ({self.primary_score_name}) vs <b>{parameter}</b>"
        if size:
            title = title + f"<br><sup>The size of the point corresponds to the value of <b>'{size}'</b>.</sup>"

        df = self.to_labeled_dataframe(query=query)
        # only include the columns we need, so that we don't unnecessarily drop rows with NA (i.e. NAs in
        # columns not used in the graph)
        columns = [x for x in [parameter, primary_score_column, size, color, 'label'] if x is not None]
        df = df[columns]
        df.dropna(axis=0, how='any', inplace=True)
        fig = px.scatter(
            data_frame=df,
            x=parameter,
            y=primary_score_column,
            size=size,
            color=color,
            color_continuous_scale=color_continuous_scale,
            trendline='lowess',
            labels={
                primary_score_column: f"Average Cross Validation Score ({self.primary_score_name})",
            },
            title=title,
            custom_data=['label'],
            height=height,
            width=width,
        )
        fig.update_traces(
            hovertemplate="<br>".join([
                "Parameter Value: %{x}",
                primary_score_column + ": " + "%{y}",
                "<br>Parameters: %{customdata[0]}",
            ])
        )
        return fig

    def plot_parameter_vs_parameter(self,
                                    parameter_x,
                                    parameter_y,
                                    query: str = None,
                                    size=None,
                                    height: float = 600,
                                    width: float = 600 * GOLDEN_RATIO) -> plotly.graph_objects.Figure:
        """
        Returns a Plotly Figure (scatter-plot) between two hyper-parameters.

        Args:
            parameter_x:
                The name of a hyper-parameter to plot against another parameter, on the x-axis.
            parameter_y:
                The name of a hyper-parameter to plot against another parameter, on the y-axis.
            query:
                string to be passed to `to_dataframe()`; see documentation for that method.
            size:
                The name of a hyper-parameter, the values of which will be used to determine the size of the
                corresponding points in the plot. This value is passed to plotly.
            height:
                The height of the plot. This value is passed to plotly.
            width:
                The width of the plot. This value is passed to plotly.
        """
        color_continuous_scale = px.colors.diverging.RdYlGn
        if not self.higher_score_is_better:
            color_continuous_scale = color_continuous_scale.reverse()

        primary_score_column = self.primary_score_name + " Mean"
        title = f"<b>{parameter_y}</b> vs <b>{parameter_x}</b>"

        scaled_size = None
        labeled_df = self.to_labeled_dataframe(query=query)

        # only include the columns we need, so that we don't unnecessarily drop rows with NA (i.e. NAs in
        # columns not used in the graph)
        columns = [x for x in [parameter_x, parameter_y, primary_score_column, size, 'label']
                   if x is not None]
        labeled_df = labeled_df[columns]
        labeled_df.dropna(axis=0, how='any', inplace=True)

        if size:
            title = title + f"<br><sup>The size of the point corresponds to the value of <b>'{size}'</b>.</sup>"
            if size in self.numeric_parameters:
                # need to do this or else the points are all the same size
                # but only if size has numeric values
                scaled_size = MinMaxScaler(feature_range=(0.1, 0.9)).fit_transform(labeled_df[[size]]).reshape(1, -1)
                scaled_size = scaled_size.tolist()[0]

        fig = px.scatter(
            data_frame=labeled_df,
            x=parameter_x,
            y=parameter_y,
            size=scaled_size,
            color=primary_score_column,
            color_continuous_scale=color_continuous_scale,
            trendline='lowess',
            title=title,
            custom_data=['label', primary_score_column],
            height=height,
            width=width,
        )
        fig.update_traces(
            hovertemplate="<br>".join([
                parameter_x + ": %{x}",
                parameter_y + ": %{y}",
                primary_score_column + ": " + "%{customdata[1]}",
                "<br>Parameters: %{customdata[0]}",
            ])
        )
        return fig


# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-public-methods
# pylint: disable=too-many-instance-attributes,too-many-public-methods
class TwoClassEvaluator:
    """This class calculates various metrics for Two Class (i.e. 0's/1's) prediction scenarios."""
    # pylint: disable=too-many-arguments
    def __init__(self,
                 actual_values: np.ndarray,
                 predicted_scores: np.ndarray,
                 positive_class: str = 'Positive Class',
                 negative_class: str = 'Negative Class',
                 score_threshold: float = 0.5
                 ):
        """
        Args:
            actual_values:
                array of 0's and 1's
            predicted_scores:
                array of decimal/float values from `predict_proba()`; NOT the actual class
            positive_class:
                string of the name/label of the positive class (i.e. value of 1). In other words, not
                'positive' in the sense of 'good' but 'positive' as in 'True/False Positive'.
            negative_class:
                string of the name/label of the negative class (i.e. value of 0). In other words, not
                'negative' in the sense of 'good' but 'negative' as in 'True/False Negative'.
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

        self._positive_class = positive_class
        self._negative_class = negative_class
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
                      f'i.e. {self._true_positives} "{self._positive_class}" labels were correctly ' \
                      f'identified out of {self._actual_positives} instances; a.k.a Sensitivity/Recall'

        tnr_message = f'{self.true_negative_rate:.1%} of negative instances were correctly identified.; ' \
                      f'i.e. {self._true_negatives} "{self._negative_class}" labels were correctly ' \
                      f'identified out of {self._actual_negatives} instances'

        fpr_message = f'{self.false_positive_rate:.1%} of negative instances were incorrectly identified ' \
                      f'as positive; ' \
                      f'i.e. {self._false_positives} "{self._negative_class}" labels were incorrectly ' \
                      f'identified as "{self._positive_class}", out of {self._actual_negatives} instances'

        fnr_message = f'{self.false_negative_rate:.1%} of positive instances were incorrectly identified ' \
                      f'as negative; ' \
                      f'i.e. {self._false_negatives} "{self._positive_class}" labels were incorrectly ' \
                      f'identified as "{self._negative_class}", out of {self._actual_positives} instances'

        ppv_message = f'When the model claims an instance is positive, it is correct ' \
                      f'{self.positive_predictive_value:.1%} of the time; ' \
                      f'i.e. out of the {self._true_positives + self._false_positives} times the model ' \
                      f'predicted "{self._positive_class}", it was correct {self._true_positives} ' \
                      f'times; a.k.a precision'

        npv_message = f'When the model claims an instance is negative, it is correct ' \
                      f'{self.negative_predictive_value:.1%} of the time; ' \
                      f'i.e. out of the {self._true_negatives + self._false_negatives} times the model ' \
                      f'predicted "{self._negative_class}", it was correct {self._true_negatives} times'

        f1_message = 'The F1 score can be interpreted as a weighted average of the precision and recall, ' \
                     'where an F1 score reaches its best value at 1 and worst score at 0.'

        accuracy_message = f'{self.accuracy:.1%} of instances were correctly identified'
        error_message = f'{self.error_rate:.1%} of instances were incorrectly identified'
        prevalence_message = f'{self.prevalence:.1%} of the data are positive; i.e. out of ' \
                             f'{self.sample_size} total observations; {self._actual_positives} are labeled ' \
                             f'as "{self._positive_class}"'
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

    # pylint: disable=too-many-arguments
    def all_metrics_df(self,
                       return_explanations: bool = True,
                       dummy_classifier_strategy: Union[str, list, None] = 'prior',
                       dummy_classifier_constant: Union[int] = 1,
                       return_style: bool = False,
                       round_by: Optional[int] = None) -> Union[pd.DataFrame, Styler]:
        """All of the metrics are returned as a DataFrame.

        Args:
            return_explanations:
                if True, then return descriptions of score and more information in an additional column
            dummy_classifier_strategy:
                if not None, then returns column(s) corresponding to the scores from predictions of
                sklearn.dummy.DummyClassifier, based on the strategy (or strategies) provided. Valid values
                correspond to values of `strategy` parameter listed
                https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html

                If a list is passed in (e.g. ['prior', 'uniform'], then one score column per value is
                added.

                If None is passed, then no additional columns are added.
            dummy_classifier_constant:
                The explicit constant as predicted by the “constant” strategy for the
                DummyClassifier.
                This parameter is useful only for the “constant” dummy_classifier_strategy.
            return_style:
                if True, return styler object; else return dataframe
            round_by:
                the number of digits to round by; if None, then don't round
        """
        result = pd.DataFrame.from_dict({key: value[0] for key, value in self.all_metrics.items()},
                                        orient='index',
                                        columns=['Score'])

        score_columns = ['Score']

        if dummy_classifier_strategy:
            if isinstance(dummy_classifier_strategy, str):
                dummy_classifier_strategy = [dummy_classifier_strategy]

            for strategy in dummy_classifier_strategy:
                dummy = DummyClassifier(strategy=strategy, constant=dummy_classifier_constant)
                # https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
                # "All strategies make predictions that ignore the input feature values passed as the X
                # argument to fit and predict. The predictions, however, typically depend on values observed
                # in the y parameter passed to fit."
                _ = dummy.fit(X=self._actual_values, y=self._actual_values)
                dummy_probabilities = dummy.predict_proba(X=self._actual_values)
                dummy_probabilities = dummy_probabilities[:, 1]
                dummy_evaluator = TwoClassEvaluator(actual_values=self._actual_values,
                                                    predicted_scores=dummy_probabilities,
                                                    score_threshold=self.score_threshold)

                dummy_scores = dummy_evaluator.all_metrics_df(return_explanations=False,
                                                              dummy_classifier_strategy=None,
                                                              return_style=False)
                column_name = f"Dummy ({strategy})"
                score_columns = score_columns + [column_name]
                dummy_scores = dummy_scores.rename(columns={'Score': column_name})
                result = pd.concat([result, dummy_scores], axis=1)

        if return_explanations:
            explanations = pd.DataFrame.from_dict({key: value[1] for key, value in self.all_metrics.items()},
                                                  orient='index',
                                                  columns=['Explanation'])
            result = pd.concat([result, explanations], axis=1)

        if round_by:
            for column in score_columns:
                result[column] = result[column].round(round_by)

        if return_style:
            subset_scores = [x for x in result.index.values if x != 'Total Observations']

            subset_scores = pd.IndexSlice[result.loc[subset_scores, :].index, score_columns]
            subset_negative_bad = pd.IndexSlice[result.loc[['False Positive Rate',
                                                            'False Negative Rate'], score_columns].index,
                                                score_columns]
            subset_secondary = pd.IndexSlice[result.loc[['Accuracy', 'Error Rate', '% Positive'],
                                                        score_columns].index, score_columns]
            subset_total_observations = pd.IndexSlice[result.loc[['Total Observations'],
                                                                 score_columns].index, score_columns]

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
        axis.xaxis.set_ticklabels([self._negative_class, self._positive_class])
        axis.yaxis.set_ticklabels([self._negative_class, self._positive_class])
        plt.tight_layout()

    def _get_auc_curve_dataframe(self) -> pd.DataFrame:
        """
        Returns a dataframe containing the AUC line (i.e. a column of score thresholds, and the corresponding
        True Positive and False Positive Rate (as columns) for the corresponding score threshold.

        (A score threshold is the value for which you would predict a positive label if the value of the score
        is above the threshold (e.g. usually 0.5).
        """
        def get_true_pos_false_pos(threshold):
            temp_eval = TwoClassEvaluator(actual_values=self._actual_values,
                                          predicted_scores=self._predicted_scores,
                                          score_threshold=threshold)

            return threshold, temp_eval.true_positive_rate, temp_eval.false_positive_rate

        auc_curve = [get_true_pos_false_pos(threshold=x) for x in np.arange(0.0, 1.01, 0.01)]
        auc_curve = pd.DataFrame(auc_curve,
                                 columns=['threshold', 'True Positive Rate', 'False Positive Rate'])
        return auc_curve

    def _get_threshold_curve_dataframe(self, score_threshold_range: Tuple[float, float] = (0.1, 0.9)) \
            -> pd.DataFrame:
        """
        Returns a dataframe containing various score thresholds from 0 to 1 (i.e. cutoff point where score
        will be labeled as a 'positive' event, and various rates (e.g. True Positive Rate, False Positive
        Rate, etc.) for the corresponding score threshold.

        (A score threshold is the value for which you would predict a positive label if the value of the score
        is above the threshold (e.g. usually 0.5).

        Args:
            score_threshold_range:
                range of score thresholds to plot (x-axis); tuple with minimum threshold in first index and
                maximum threshold in second index.
        """
        def get_threshold_scores(threshold):
            temp_eval = TwoClassEvaluator(actual_values=self._actual_values,
                                          predicted_scores=self._predicted_scores,
                                          score_threshold=threshold)

            return threshold,\
                temp_eval.true_positive_rate,\
                temp_eval.false_positive_rate,\
                temp_eval.positive_predictive_value,\
                temp_eval.false_negative_rate,\
                temp_eval.true_negative_rate

        threshold_curves = [get_threshold_scores(threshold=x)
                            for x in np.arange(score_threshold_range[0],
                                               score_threshold_range[1] + 0.025,
                                               0.025)]

        threshold_curves = pd.DataFrame(threshold_curves,
                                        columns=['Score Threshold',
                                                 'True Pos. Rate (Recall)',
                                                 'False Pos. Rate',
                                                 'Pos. Predictive Value (Precision)',
                                                 'False Neg. Rate',
                                                 'True Neg. Rate (Specificity)'])
        return threshold_curves

    # pylint: disable=inconsistent-return-statements
    def plot_auc_curve(self,
                       figure_size: tuple = STANDARD_WIDTH_HEIGHT,
                       return_plotly: bool = False) -> Union[None,
                                                            _figure.Figure]:
        """Plots the ROC AUC

        Args:
            figure_size:
                tuple containing `(width, height)` of plot. The default height is defined by
                `helpsk.plot.STANDARD_HEIGHT`, and the default width is
                `helpsk.plot.STANDARD_HEIGHT / helpsk.plot.GOLDEN_RATIO`
            return_plotly:
                If True, return plotly object. Otherwise, use matplotlib and end function with call:
                `plt.tight_layout()`
        """
        plt.figure(figsize=figure_size)
        auc_curve = self._get_auc_curve_dataframe()

        if return_plotly:
            fig = px.line(
                data_frame=auc_curve,
                x='False Positive Rate',
                y='True Positive Rate',
                color_discrete_sequence=[hcolor.Colors.DOVE_GRAY.value],
                height=550,
                width=550 * GOLDEN_RATIO,
                title=f"AUC: {self.auc:.3f}<br><sub>The threshold of 0.5 is indicated with a large point.</sub>"  # pylint: disable=line-too-long  # noqa
            )
            fig.add_trace(
                px.scatter(
                    data_frame=auc_curve,
                    x='False Positive Rate',
                    y='True Positive Rate',
                    color='threshold',
                ).data[0]
            )
            fig.add_trace(
                px.scatter(
                    data_frame=auc_curve.query('threshold == 0.5'),
                    x='False Positive Rate',
                    y='True Positive Rate',
                    size=[2],
                ).data[0]
            )
            return fig

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

    # pylint: disable=inconsistent-return-statements
    def plot_threshold_curves(self,
                              score_threshold_range: Tuple[float, float] = (0.1, 0.9),
                              figure_size: tuple = STANDARD_WIDTH_HEIGHT,
                              return_plotly: bool = False) -> Union[None,
                                                                    _figure.Figure]:
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
            return_plotly:
                If True, return plotly object. Otherwise, use matplotlib and end function with call:
                `plt.tight_layout()`
        """
        plt.figure(figsize=figure_size)

        threshold_curves = self._get_threshold_curve_dataframe(score_threshold_range=score_threshold_range)

        if return_plotly:
            custom_colors = [
                hcolor.Colors.PASTEL_BLUE.value,
                hcolor.Colors.CUSTOM_GREEN.value,
                hcolor.Colors.YELLOW_PEPPER.value,
                hcolor.Colors.CRAIL.value,
                hcolor.Colors.CADMIUM_ORANGE.value,
            ]
            fig = px.line(
                data_frame=pd.melt(frame=threshold_curves, id_vars=['Score Threshold']),
                x='Score Threshold',
                y='value',
                color='variable',
                color_discrete_sequence=custom_colors,
                labels={
                    'variable': 'Rate Type',
                    'value': 'Rate'
                },
                height=550,
                width=550 * GOLDEN_RATIO,
                title="Tradeoffs Across Various Score Thresholds<br><sub>Black line is default threshold of 0.5.</sub>"  # pylint: disable=line-too-long  # noqa
            )
            fig = fig.add_vline(x=0.5, line_color=hcolor.Colors.BLACK_SHADOW.value)
            return fig

        axis = sns.lineplot(x='Score Threshold', y='value', hue='variable',
                            data=pd.melt(frame=threshold_curves, id_vars=['Score Threshold']))
        axis.set_xticks(np.arange(score_threshold_range[0], score_threshold_range[1] + 0.1, 0.1))
        axis.set_yticks(np.arange(0, 1.1, .1))
        plt.vlines(x=self.score_threshold, ymin=0, ymax=1, colors='black')
        plt.grid()
        plt.tight_layout()

    # pylint: disable=inconsistent-return-statements
    def plot_precision_recall_tradeoff(self,
                                       score_threshold_range: Tuple[float, float] = (0.1, 0.9),
                                       figure_size: tuple = STANDARD_WIDTH_HEIGHT,
                                       return_plotly: bool = False) -> Union[None,
                                                                             _figure.Figure]:
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
            return_plotly:
                If True, return plotly object. Otherwise, use matplotlib and end function with call:
                `plt.tight_layout()`
        """
        plt.figure(figsize=figure_size)

        threshold_curves = self._get_threshold_curve_dataframe(score_threshold_range=score_threshold_range)
        threshold_curves = threshold_curves[['Score Threshold',
                                                 'True Pos. Rate (Recall)',
                                                 'Pos. Predictive Value (Precision)']]

        if return_plotly:
            custom_colors = [
                hcolor.Colors.PASTEL_BLUE.value,
                #     hcolor.Colors.CUSTOM_GREEN.value,
                hcolor.Colors.YELLOW_PEPPER.value,
                #     hcolor.Colors.CRAIL.value,
                #     hcolor.Colors.CADMIUM_ORANGE.value,
            ]

            fig = px.line(
                data_frame=pd.melt(frame=threshold_curves[['Score Threshold',
                                                           'True Pos. Rate (Recall)',
                                                           'Pos. Predictive Value (Precision)']],
                                   id_vars=['Score Threshold']),
                x='Score Threshold',
                y='value',
                color='variable',
                color_discrete_sequence=custom_colors,
                labels={
                    'variable': 'Rate',
                    'value': 'Value'
                },
                height=550,
                width=550 * GOLDEN_RATIO,
                title="Precision Recall Tradeoff<br><sub>Black line is default threshold of 0.5.</sub>"
            )
            fig = fig.add_vline(x=0.5, line_color=hcolor.Colors.BLACK_SHADOW.value)
            return fig
        axis = sns.lineplot(x='Score Threshold', y='value', hue='variable',
                            data=pd.melt(frame=threshold_curves, id_vars=['Score Threshold']))
        axis.set_xticks(np.arange(score_threshold_range[0], score_threshold_range[1] + 0.1, 0.1))
        axis.set_yticks(np.arange(0, 1.1, .1))
        plt.vlines(x=self.score_threshold, ymin=0, ymax=1, colors='black')
        plt.grid()
        plt.tight_layout()

    # pylint: disable=inconsistent-return-statements
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

    # pylint: disable=inconsistent-return-statements
    def plot_predicted_scores_histogram(self):
        """Return a histogram of the predicted scores"""
        sns.histplot(self._predicted_scores)
        plt.tight_layout()

    # pylint: disable=inconsistent-return-statements
    def plot_actual_vs_predict_histogram(self):
        """Return a histogram of the actual vs predicted scores"""
        actual_categories = pd.Series(self._actual_values).\
            replace({0: self._negative_class, 1: self._positive_class})
        axes = sns.displot(
            pd.DataFrame({
                'Predicted Score': self._predicted_scores,
                'Actual Value': actual_categories
            }),
            x='Predicted Score',
            col='Actual Value'
        )
        for axis in axes.axes.flat:
            axis.axvline(x=0.5, ymin=0, ymax=100, color='red')
        plt.tight_layout()


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
                       dummy_regressor_strategy: Union[str, list, None] = 'mean',
                       dummy_regressor_constant: Union[int] = 1,
                       return_style: bool = False,
                       round_by: Optional[int] = None) -> Union[pd.DataFrame, Styler]:
        """All of the metrics are returned as a DataFrame.

        Args:
            dummy_regressor_strategy:
                if not None, then returns column(s) corresponding to the scores from predictions of
                sklearn.dummy.DummyRegressor, based on the strategy (or strategies) provided. Valid values
                correspond to values of `strategy` parameter listed
                https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyRegressor.html

                If a list is passed in (e.g. ['prior', 'uniform'], then one score column per value is
                added.

                If None is passed, then no additional columns are added.
            dummy_regressor_constant:
                The explicit constant as predicted by the “constant” strategy for the
                DummyRegressor.
                This parameter is useful only for the “constant” dummy_regressor_strategy.
            return_style:
                if True, return styler object; else return dataframe
            round_by:
                the number of digits to round by; if None, then don't round
        """
        result = pd.DataFrame.from_dict(self.all_metrics, orient='index', columns=['Score'])

        score_columns = ['Score']

        if dummy_regressor_strategy:
            if isinstance(dummy_regressor_strategy, str):
                dummy_regressor_strategy = [dummy_regressor_strategy]

            for strategy in dummy_regressor_strategy:
                dummy = DummyRegressor(strategy=strategy, constant=dummy_regressor_constant)
                # https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
                # "All strategies make predictions that ignore the input feature values passed as the X
                # argument to fit and predict. The predictions, however, typically depend on values observed
                # in the y parameter passed to fit."
                _ = dummy.fit(X=self._actual_values, y=self._actual_values)
                dummy_predictions = dummy.predict(X=self._actual_values)
                dummy_evaluator = RegressionEvaluator(actual_values=self._actual_values,
                                                      predicted_values=dummy_predictions)

                dummy_scores = dummy_evaluator.all_metrics_df(dummy_regressor_strategy=None,
                                                              return_style=False)
                column_name = f"Dummy ({strategy})"
                score_columns = score_columns + [column_name]
                dummy_scores = dummy_scores.rename(columns={'Score': column_name})
                result = pd.concat([result, dummy_scores], axis=1)

        if round_by is not None:
            result.iloc[0:2] = result.iloc[0:2].round(round_by)

        if return_style:
            subset_scores = pd.IndexSlice[result.loc[['Mean Absolute Error (MAE)',
                                                      'Root Mean Squared Error (RMSE)'],
                                                     score_columns].index,
                                          score_columns]
            subset_secondary = pd.IndexSlice[result.loc[['RMSE to Standard Deviation of Target',
                                                         'R Squared'],
                                                        score_columns].index, score_columns]
            subset_total_observations = pd.IndexSlice[result.loc[['Total Observations'],
                                                                 score_columns].index, score_columns]
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


class TwoClassModelComparison:
    """This class compares multiple models trained on Two Class (i.e. 0's/1's) prediction scenarios."""

    # pylint: disable=too-many-arguments
    def __init__(self,
                 actual_values: np.ndarray,
                 predicted_scores: Dict[str, np.ndarray],
                 positive_class: str = 'Positive Class',
                 negative_class: str = 'Negative Class',
                 score_threshold: float = 0.5
                 ):
        """
        Args:
            actual_values:
                array of 0's and 1's
            predicted_scores:
                dictionary per model with key as the name of the model and value that is an array of
                decimal/float values from `predict_proba()`; NOT the actual class
            positive_class:
                string of the name/label of the positive class (i.e. value of 1). In other words, not
                'positive' in the sense of 'good' but 'positive' as in 'True/False Positive'.
            negative_class:
                string of the name/label of the negative class (i.e. value of 0). In other words, not
                'negative' in the sense of 'good' but 'negative' as in 'True/False Negative'.
            score_threshold:
                the score/probability threshold for turning scores into 0's and 1's and corresponding labels
        """
        assert isinstance(predicted_scores, dict)

        for values in predicted_scores.values():
            assert len(actual_values) == len(values)

        self._positive_class = positive_class
        self._negative_class = negative_class
        self._actual_values = actual_values
        self._predicted_scores = predicted_scores
        self.score_threshold = score_threshold

        self._evaluators = {key: TwoClassEvaluator(actual_values=actual_values,
                                                   predicted_scores=value,
                                                   positive_class=positive_class,
                                                   negative_class=negative_class,
                                                   score_threshold=score_threshold)
                            for key, value in predicted_scores.items()}

    def all_metrics_df(self,
                       dummy_classifier_strategy: Union[str, list, None] = 'prior',
                       dummy_classifier_constant: Union[int] = 1,
                       return_style: bool = False,
                       round_by: Optional[int] = None) -> Union[pd.DataFrame, Styler]:
        """All of the metrics are returned as a DataFrame.

        Args:
            dummy_classifier_strategy:
                if not None, then returns column(s) corresponding to the scores from predictions of
                sklearn.dummy.DummyClassifier, based on the strategy (or strategies) provided. Valid values
                correspond to values of `strategy` parameter listed
                https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html

                If a list is passed in (e.g. ['prior', 'uniform'], then one score column per value is
                added.

                If None is passed, then no additional columns are added.
            dummy_classifier_constant:
                The explicit constant as predicted by the “constant” strategy for the
                DummyClassifier.
                This parameter is useful only for the “constant” dummy_classifier_strategy.
            return_style:
                if True, return styler object; else return dataframe
            round_by:
                the number of digits to round by; if None, then don't round
        """

        result = None
        last_key = list(self._evaluators.keys())[-1]

        for key, value in self._evaluators.items():
            dummy_strategy = dummy_classifier_strategy if key == last_key else None

            scores = value.all_metrics_df(
                return_explanations=False,
                dummy_classifier_strategy=dummy_strategy,
                dummy_classifier_constant=dummy_classifier_constant
            )
            scores = scores.rename(columns={'Score': key})
            result = pd.concat([result, scores], axis=1)

        result = result.loc[[
            'AUC', 'F1 Score',
            'True Positive Rate', 'True Negative Rate',
            'False Positive Rate', 'False Negative Rate',
            'Positive Predictive Value', 'Negative Predictive Value'
        ]]

        result = result.transpose()

        if round_by:
            for column in result.columns:
                result[column] = result[column].round(round_by)

        if return_style:
            positive_scores = [x for x in result.columns if not x.startswith('False')]
            negative_scores = [x for x in result.columns if x.startswith('False')]

            result = result.style

            if round_by:
                result = result.format(precision=round_by)

            result = result. \
                bar(subset=positive_scores, color=hcolor.Colors.PIGMENT_GREEN.value, vmin=0, vmax=1). \
                bar(subset=negative_scores, color=hcolor.Colors.POPPY.value, vmin=0, vmax=1)

        return result

    def plot_metrics_comparison(self,
                                dummy_classifier_strategy: Union[str, list, None] = 'prior',
                                dummy_classifier_constant: Union[int] = 1,
                                ) -> _figure.Figure:
        """
        Returns a Plotly object of a bar-chart of the metrics across all of the models.

        Args:
            dummy_classifier_strategy:
                if not None, then returns column(s) corresponding to the scores from predictions of
                sklearn.dummy.DummyClassifier, based on the strategy (or strategies) provided. Valid values
                correspond to values of `strategy` parameter listed
                https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html

                If a list is passed in (e.g. ['prior', 'uniform'], then one score column per value is
                added.

                If None is passed, then no additional columns are added.
            dummy_classifier_constant:
                The explicit constant as predicted by the “constant” strategy for the
                DummyClassifier.
                This parameter is useful only for the “constant” dummy_classifier_strategy.
        """
        score_df = self.all_metrics_df(
            dummy_classifier_strategy=dummy_classifier_strategy,
            dummy_classifier_constant=dummy_classifier_constant
        ).transpose()

        score_df = score_df.reset_index()

        colors = [e.value for e in hcolor.Colors]
        fig = px.bar(
            data_frame=score_df.melt(id_vars='index'),
            y='variable',
            x='value',
            facet_col='index',
            facet_col_wrap=2,
            color='variable',
            color_discrete_sequence=colors,
            barmode='group',
            height=1000,
            labels={'index': 'Score'},
            title="Model Comparison"
        )
        fig.update_layout(showlegend=False)
        fig.update_yaxes(title=None)
        return fig

    def plot_roc_curves(self) -> _figure.Figure:
        """Returns a plotly object representing the ROC curves across all models."""
        result = None

        for key, value in self._evaluators.items():
            auc_df = value._get_auc_curve_dataframe()  # pylint: disable=protected-access # noqa
            auc_df['Model'] = key
            result = pd.concat([result, auc_df], axis=0)

        colors = [e.value for e in hcolor.Colors]
        fig = px.line(
            data_frame=result,
            x='False Positive Rate',
            y='True Positive Rate',
            color='Model',
            color_discrete_sequence=colors,
            height=550,
            width=550 * GOLDEN_RATIO,
            custom_data=['threshold', 'Model'],
            title="ROC Curve of Models",
        )
        for index in range(len(self._evaluators)):
            scatter_1 = px.scatter(
                data_frame=result,
                x='False Positive Rate',
                y='True Positive Rate',
                color='Model',
                color_discrete_sequence=colors,
                custom_data=['threshold', 'Model'],
            )
            scatter_1.data[index]['showlegend'] = False
            fig.add_trace(
                scatter_1.data[index]
            )
            query = f"threshold == 0.5 & Model == '{list(self._evaluators.keys())[index]}'"
            scatter_2 = px.scatter(
                data_frame=result.query(query),
                x='False Positive Rate',
                y='True Positive Rate',
                color='Model',
                color_discrete_sequence=[colors[index]] + colors,
                custom_data=['threshold', 'Model'],
                size=[2],
            )
            scatter_2.data[0]['showlegend'] = False
            fig.add_trace(
                scatter_2.data[0],
            )

        fig.update_traces(
            hovertemplate="<br>".join([
                "Model: %{customdata[1]}<br><br>"
                "False Positive Rate: %{x}",
                "True Positive Rate: %{y}",
                "Threshold: %{customdata[0]}",
            ])
        )
        return fig
