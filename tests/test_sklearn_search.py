import re
import unittest

import numpy as np
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from skopt import BayesSearchCV

from helpsk import validation
from helpsk.pandas import print_dataframe
from helpsk.sklearn_eval import MLExperimentResults
from helpsk.sklearn_search import *
from helpsk.utility import redirect_stdout_to_file
from tests.helpers import get_data_credit, get_test_path, clean_formatted_dataframe


# noinspection PyMethodMayBeStatic
class TestSklearnSearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass
        credit_data = get_data_credit()
        credit_data.loc[0:46, ['duration']] = np.nan
        credit_data.loc[25:75, ['checking_status']] = np.nan
        credit_data.loc[10:54, ['credit_amount']] = 0
        y_full = credit_data['target']
        X_full = credit_data.drop(columns='target')  # noqa
        y_full = label_binarize(y_full, classes=['good', 'bad']).flatten()
        X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=42)  # noqa
        del y_full, X_full
        cls.X_train = X_train
        cls.y_train = y_train
        
    @staticmethod
    def to_string(obj):
        return str(obj). \
            replace(", '", ",\n'"). \
            replace('{', '{\n'). \
            replace('}', '\n}'). \
            replace(', ({', ',\n({')

    def test_pipeline(self):
        # this will be the same for all inherited classes, so just test base.
        with open(get_test_path() + '/test_files/sklearn_search/search_space_base__pipeline.txt', 'w') as file:
            file.write(str(BayesianSearchSpaceBase.pipeline(data=self.X_train)))

    def test_ModelBayesianSearchSpace_params(self):
        search_space = LogisticBayesianSearchSpace(
            C=None,
        )
        model_space = search_space.search_spaces()[0][0]
        model_params = [x for x in model_space.keys() if x.startswith('model__')]
        self.assertEqual(len(model_params), 0)

        model_mappings = [x for x in search_space.param_name_mappings().keys() if x.startswith('model__')]
        self.assertEqual(len(model_mappings), 0)
        del search_space, model_space, model_mappings

        search_space = LinearSVCBayesianSearchSpace(
            C=None,
        )
        model_space = search_space.search_spaces()[0][0]
        model_params = [x for x in model_space.keys() if x.startswith('model__')]
        self.assertEqual(len(model_params), 0)

        model_mappings = [x for x in search_space.param_name_mappings().keys() if x.startswith('model__')]
        self.assertEqual(len(model_mappings), 0)
        del search_space, model_space, model_mappings

        search_space = ExtraTreesBayesianSearchSpace(
            max_features=None,
            max_depth=None,
            min_samples_split=None,
            min_samples_leaf=None,
            max_samples=None,
            criterion=None,
        )
        model_space = search_space.search_spaces()[0][0]
        model_params = [x for x in model_space.keys() if x.startswith('model__')]
        self.assertEqual(len(model_params), 0)

        model_mappings = [x for x in search_space.param_name_mappings().keys() if x.startswith('model__')]
        self.assertEqual(len(model_mappings), 0)

        search_space = RandomForestBayesianSearchSpace(
            max_features=None,
            max_depth=None,
            min_samples_split=None,
            min_samples_leaf=None,
            max_samples=None,
            criterion=None,
        )
        model_space = search_space.search_spaces()[0][0]
        model_params = [x for x in model_space.keys() if x.startswith('model__')]
        self.assertEqual(len(model_params), 0)

        model_mappings = [x for x in search_space.param_name_mappings().keys() if x.startswith('model__')]
        self.assertEqual(len(model_mappings), 0)

        search_space = XGBoostBayesianSearchSpace(
            max_depth=None,
            learning_rate=None,
            n_estimators=None,
            min_child_weight=None,
            subsample=None,
            colsample_bytree=None,
            colsample_bylevel=None,
            reg_alpha=None,
            reg_lambda=None,
        )
        model_space = search_space.search_spaces()[0][0]
        model_params = [x for x in model_space.keys() if x.startswith('model__')]
        self.assertEqual(len(model_params), 0)

        model_mappings = [x for x in search_space.param_name_mappings().keys() if x.startswith('model__')]
        self.assertEqual(len(model_mappings), 0)

    def test_ModelBayesianSearchSpace(self):

        def test_search_space(search_object, modified_args):
            default_space = search_object()
            class_name = default_space.__class__.__name__

            with open(get_test_path() + f'/test_files/sklearn_search/{class_name}_search_spaces__default.txt', 'w') as file:
                file.write(TestSklearnSearch.to_string(default_space.search_spaces()))

            default_mappings = default_space.param_name_mappings()
            self.assertEqual(default_space.search_spaces()[0][0].keys(), default_mappings.keys())
            with open(get_test_path() + f'/test_files/sklearn_search/{class_name}_param_name_mappings.txt', 'w') as file:
                file.write(TestSklearnSearch.to_string(default_mappings))

            self.assertIsInstance(default_space.search_spaces(), list)
            self.assertEqual(len(default_space.search_spaces()), 2)
            self.assertIsInstance(default_space.search_spaces()[0], tuple)
            self.assertIsInstance(default_space.search_spaces()[1], tuple)
            self.assertIsInstance(default_space.search_spaces()[0][0], dict)
            self.assertIsInstance(default_space.search_spaces()[0][1], int)
            self.assertIsInstance(default_space.search_spaces()[1][0], dict)
            self.assertIsInstance(default_space.search_spaces()[1][1], int)

            from skopt.space import Categorical
            categorical = default_space.search_spaces()[0][0]['model']
            self.assertIsInstance(categorical, Categorical)
            del default_space, categorical

            modified_space = search_object(
                **modified_args,
                iterations=30,
                include_default_model=False,
                imputers=Categorical([SimpleImputer(strategy='most_frequent')]),
                scalers=Categorical([None]),
                encoders=Categorical([CustomOrdinalEncoder()]),
                random_state=42
            )

            with open(get_test_path() + f'/test_files/sklearn_search/{class_name}_search_spaces__modified.txt', 'w') as file:
                file.write(TestSklearnSearch.to_string(modified_space.search_spaces()))

            self.assertEqual(default_mappings, modified_space.param_name_mappings())

            self.assertIsInstance(modified_space.search_spaces(), list)
            self.assertEqual(len(modified_space.search_spaces()), 1)
            self.assertIsInstance(modified_space.search_spaces()[0], tuple)
            self.assertIsInstance(modified_space.search_spaces()[0][0], dict)
            self.assertIsInstance(modified_space.search_spaces()[0][1], int)
            self.assertEqual(modified_space.search_spaces()[0][1], 30)

            categorical = modified_space.search_spaces()[0][0]['model']
            self.assertIsInstance(categorical, Categorical)
            del modified_space, categorical

        # with self.subTest(i='LogisticBayesianSearchSpace'):
        args = dict(
            C=Real(1e-5, 1e+3, prior='uniform'),
            solver='sag',
            max_iter=999,
        )
        test_search_space(LogisticBayesianSearchSpace, modified_args=args)

        args = dict(
            C=Real(1e-5, 1e+3),
        )
        test_search_space(LinearSVCBayesianSearchSpace, modified_args=args)

        args = dict(
            max_features=Real(0.06, 0.92),
            max_depth=Integer(4, 101),
            min_samples_split=Integer(5, 55),
            min_samples_leaf=Integer(2, 55),
            max_samples=Real(0.8, 0.9),
            criterion=Categorical(['entropy']),
        )
        test_search_space(ExtraTreesBayesianSearchSpace, modified_args=args)

        args = dict(
            max_features=Real(0.06, 0.92),
            max_depth=Integer(5, 102),
            min_samples_split=Integer(3, 52),
            min_samples_leaf=Integer(5, 52),
            max_samples=Real(0.7, 0.99),
            criterion=Categorical(['gini']),
        )
        test_search_space(RandomForestBayesianSearchSpace, modified_args=args)

        args = dict(
            eval_metric='logloss',
            max_depth=Integer(2, 30),
            n_estimators=Integer(10, 10000),
            learning_rate=Real(0.01111, 0.3333),
            colsample_bytree=Real(0.01234, 1123),
            subsample=Real(0.1111, 0.999),
        )
        test_search_space(XGBoostBayesianSearchSpace, modified_args=args)

    def test_BayesianSearchSpace(self):
        search_space = BayesianSearchSpace(self.X_train, iterations=45, random_state=42)
        self.assertEqual(str(BayesianSearchSpaceBase.pipeline(data=self.X_train)),
                         str(search_space.pipeline()))

        with open(get_test_path() + '/test_files/sklearn_search/BayesianSearchSpace_search_spaces.txt', 'w') as file:
            file.write(TestSklearnSearch.to_string(search_space.search_spaces()))

        self.assertIsInstance(search_space.search_spaces(), list)

        for space in search_space.search_spaces():
            self.assertIsInstance(space, tuple)
            self.assertIsInstance(space[0], dict)
            self.assertIsInstance(space[1], int)

        with open(get_test_path() + '/test_files/sklearn_search/BayesianSearchSpace_param_name_mappings.txt', 'w') as file:
            file.write(TestSklearnSearch.to_string(search_space.param_name_mappings()))

    def test_MLExperimentResults_multi_model(self):

        search_space = BayesianSearchSpace(self.X_train,
                                           model_search_spaces=[
                                               XGBoostBayesianSearchSpace(iterations=4),
                                               LogisticBayesianSearchSpace(iterations=4),
                                           ])

        bayes_search = BayesSearchCV(
                estimator=search_space.pipeline(),  # noqa
                search_spaces=search_space.search_spaces(),  # noqa
                cv=RepeatedKFold(n_splits=3, n_repeats=1, random_state=42),  # 3 fold 1 repeat CV
                scoring='roc_auc',
                refit=False,  # required if passing in multiple scorers
                return_train_score=False,
                n_jobs=-1,
                verbose=0,
                random_state=42,
            )
        _ = bayes_search.fit(self.X_train, self.y_train)

        results = MLExperimentResults.from_sklearn_search_cv(
            searcher=bayes_search,
            higher_score_is_better=True,
            description='BayesSearchCV using ClassifierSearchSpace',
            parameter_name_mappings=search_space.param_name_mappings()
        )

        results.to_yaml_file(get_test_path() + '/test_files/sklearn_search/multi-model-search.yaml')

        with redirect_stdout_to_file(get_test_path() + '/test_files/sklearn_search/multi-model-search-dataframe.txt'):
            print_dataframe(results.to_dataframe())

        with redirect_stdout_to_file(get_test_path() + '/test_files/sklearn_search/multi-model-search-dataframe__xgboost.txt'):
            print_dataframe(results.to_dataframe(query="model == 'XGBClassifier()'"))

        with redirect_stdout_to_file(get_test_path() + '/test_files/sklearn_search/multi-model-search-dataframe__logistic.txt'):
            print_dataframe(results.to_dataframe(query="model == 'LogisticRegression()'"))

        with open(get_test_path() + '/test_files/sklearn_search/multi-model-search-dataframe.html', 'w') as file:
            file.write(clean_formatted_dataframe(results.to_formatted_dataframe().render()))

        with open(get_test_path() + '/test_files/sklearn_search/multi-model-search-dataframe__xgboost.html', 'w') as file:
            file.write(clean_formatted_dataframe(results.to_formatted_dataframe(query="model == 'XGBClassifier()'").render()))

        with open(get_test_path() + '/test_files/sklearn_search/multi-model-search-dataframe__logistic.html', 'w') as file:
            file.write(clean_formatted_dataframe(results.to_formatted_dataframe(query="model == 'LogisticRegression()'").render()))

        # pd.set_option('display.max_columns', 500)
        # pd.set_option('display.width', 10000)

        def label_column_formatter(label_column):
            return [str(value).replace('<br>', '; ').
                        replace('; model', '\nmodel').
                        replace('; ', '\n       ') for value in label_column]  # noqa

        labeled_df = results.to_labeled_dataframe()
        with redirect_stdout_to_file(get_test_path() + '/test_files/sklearn_search/multi-model-search-labeled_dataframe.txt'):
            print_dataframe(labeled_df)
        with open(get_test_path() + '/test_files/sklearn_search/multi-model-search-labeled_dataframe__label.txt', 'w') as file:
            for x in label_column_formatter(labeled_df['label']):
                file.write(x)

        labeled_df = results.to_labeled_dataframe(query="model == 'XGBClassifier()'")
        with redirect_stdout_to_file(get_test_path() + '/test_files/sklearn_search/multi-model-search-labeled_dataframe__xgboost.txt'):
            print_dataframe(labeled_df)
        with open(get_test_path() + '/test_files/sklearn_search/multi-model-search-labeled_dataframe__xgboost__label.txt', 'w') as file:
            for x in label_column_formatter(labeled_df['label']):
                file.write(x)

        labeled_df = results.to_labeled_dataframe(query="model == 'LogisticRegression()'")
        with redirect_stdout_to_file(get_test_path() + '/test_files/sklearn_search/multi-model-search-labeled_dataframe__logistic.txt'):
            print_dataframe(labeled_df)
        with open(get_test_path() + '/test_files/sklearn_search/multi-model-search-labeled_dataframe__logistic__label.txt', 'w') as file:
            for x in label_column_formatter(labeled_df['label']):
                file.write(x)

        def to_string(obj):
            return str(obj).\
                replace(", '", ",\n'").\
                replace('{', '{\n').\
                replace('}', '\n}').\
                replace(', ({', ',\n({')

        labels = results.trial_labels(order_from_best_to_worst=True)
        with open(get_test_path() + '/test_files/sklearn_search/trial_labels_sorted.txt', 'w') as file:
            file.write(to_string(labels))
        del labels

        labels = results.trial_labels(order_from_best_to_worst=False)
        with open(get_test_path() + '/test_files/sklearn_search/trial_labels_not_sorted.txt', 'w') as file:
            file.write(to_string(labels))
        del labels

        self.assertEqual(results.higher_score_is_better, True)
        self.assertEqual(results.cross_validation_type, "<class 'skopt.searchcv.BayesSearchCV'>")
        self.assertEqual(results.number_of_splits,
                         bayes_search.cv.n_repeats * bayes_search.n_splits_)
        self.assertEqual(results.score_names, ['roc_auc'])
        self.assertEqual(results.parameter_names_original,
                         list(search_space.param_name_mappings().keys()))
        self.assertEqual(results.parameter_names, list(search_space.param_name_mappings().values()))
        self.assertIsNotNone(results.parameter_names_mapping)
        self.assertTrue(isinstance(results.test_score_rankings, dict))
        self.assertEqual(list(results.test_score_rankings.keys()), results.score_names)
        self.assertTrue(all(np.array(results.test_score_rankings['roc_auc']) == bayes_search.cv_results_['rank_test_score']))
        self.assertTrue(np.isclose(results.best_score, bayes_search.best_score_))

        def assert_np_arrays_are_close(array1, array2):
            self.assertEqual(len(array1), len(array2))
            for index in range(len(array1)):
                is_close = validation.is_close(array1[index], array2[index])
                both_nan = np.isnan(array1[index]) and np.isnan(array2[index])
                self.assertTrue(is_close or both_nan)

        assert_np_arrays_are_close(np.array([1, 2, 3]), np.array([1, 2, 3]))
        assert_np_arrays_are_close(np.array([1, 2, np.nan]), np.array([1, 2, np.nan]))
        self.assertRaises(AssertionError,
                          lambda: assert_np_arrays_are_close(np.array([1, 2, 3]), np.array([1, 2, 3.001])))
        self.assertRaises(AssertionError,
                          lambda: assert_np_arrays_are_close(np.array([1, 2, 3]), np.array([1, 2, np.nan])))

        assert_np_arrays_are_close(np.array(results.test_score_averages[results.primary_score_name]),
                                   bayes_search.cv_results_['mean_test_score'])

        self.assertEqual(list(results.best_trial_indexes), list(results.to_dataframe().index))
        cv_dataframe = results.to_dataframe().sort_index()
        validation.assert_dataframes_match([results.to_dataframe(sort_by_score=False), cv_dataframe])
        assert_np_arrays_are_close(cv_dataframe[f'{results.score_names[0]} Mean'],
                                   bayes_search.cv_results_['mean_test_score'])

        labeled_dataframe = results.to_labeled_dataframe()
        self.assertTrue(all(labeled_dataframe['Trial Index'] == list(range(1, results.number_of_trials + 1))))
        self.assertIsNotNone(labeled_dataframe['label'])
        validation.assert_dataframes_match(dataframes=[results.to_dataframe(sort_by_score=False),
                                                           labeled_dataframe.drop(columns=['Trial Index',
                                                                                           'label'])])

        def remove_parentheses(obj):
            obj = str(obj)
            obj = obj.replace("OneHotEncoder(handle_unknown='ignore')", "OneHotEncoder()")
            return re.sub(r'\(.*', '', obj.replace('\n', ''))

        self.assertEqual([remove_parentheses(x) for x in bayes_search.cv_results_['param_model'].data],
                         [remove_parentheses(x) for x in cv_dataframe['model']])

        # check that all of the hyper-param values match in the dataframe vs bayes_search.cv_results_
        hyper_param_df = cv_dataframe.iloc[:, 4:]
        for key, value in search_space.param_name_mappings().items():
            if key != 'model':
                mask = bayes_search.cv_results_['param_' + key].mask
                cv_data = list(bayes_search.cv_results_['param_' + key].data)
                cv_data = [np.nan if m else d for m, d in zip(mask, cv_data)]

                if key.startswith('model__'):
                    assert_np_arrays_are_close(np.array(cv_data),
                                               np.array(hyper_param_df.loc[:, value].tolist()))
                if key.startswith('prep__'):
                    self.assertEqual([str(x).replace("OneHotEncoder(handle_unknown='ignore')",
                                                     "OneHotEncoder()")
                                      for x in cv_data],
                                     hyper_param_df.loc[:, value].tolist())

        self.assertTrue(isinstance(results.test_score_averages, dict))
        self.assertEqual(list(results.test_score_averages.keys()), results.score_names)

        assert_np_arrays_are_close(results.primary_score_averages,
                                   np.array(results.test_score_averages[results.primary_score_name]))
        assert_np_arrays_are_close(results.primary_score_averages,
                                   bayes_search.cv_results_['mean_test_score'])

        self.assertEqual(len(results.score_standard_errors(score_name=results.score_names[0])),
                         results.number_of_trials)

        self.assertTrue(isinstance(results.test_score_standard_deviations, dict))
        self.assertEqual(list(results.test_score_standard_deviations.keys()), results.score_names)

        assert_np_arrays_are_close(np.array(results.test_score_standard_deviations[results.primary_score_name]),
                                   bayes_search.cv_results_['std_test_score'])

        self.assertIsNone(results.train_score_averages)
        self.assertIsNone(results.train_score_standard_deviations)

        self.assertTrue(isinstance(results.trials, list))
        self.assertEqual(len(results.trials), results.number_of_trials)

        self.assertTrue(isinstance(results.timings, dict))
        self.assertEqual(list(results.timings.keys()), ['fit time averages',
                                                       'fit time standard deviations',
                                                       'score time averages',
                                                       'score time standard deviations'])

        assert_np_arrays_are_close(np.array(results.timings[list(results.timings.keys())[0]]),
                                   bayes_search.cv_results_['mean_fit_time'])
        assert_np_arrays_are_close(np.array(results.timings[list(results.timings.keys())[1]]),
                                   bayes_search.cv_results_['std_fit_time'])
        assert_np_arrays_are_close(np.array(results.timings[list(results.timings.keys())[2]]),
                                   bayes_search.cv_results_['mean_score_time'])
        assert_np_arrays_are_close(np.array(results.timings[list(results.timings.keys())[3]]),
                                   bayes_search.cv_results_['std_score_time'])

        self.assertEqual(results.primary_score_name, 'roc_auc')

        self.assertTrue(all(results.trial_rankings == bayes_search.cv_results_['rank_test_score']))  # noqa

        self.assertTrue(all([results.to_dataframe().loc[results.best_score_index, key] == value
                             for key, value in results.best_params.items()]))

        for original_name, new_name in results.parameter_names_mapping.items():
            if new_name in results.best_params:
                if original_name == 'model':
                    self.assertEqual(remove_parentheses(results.best_params[new_name]),
                                     remove_parentheses(bayes_search.best_params_[original_name]))
                elif original_name.startswith('model__'):
                    self.assertTrue(np.isclose(results.best_params[new_name],
                                               bayes_search.best_params_[original_name]))
                else:
                    self.assertEqual(results.best_params[new_name],
                                     str(bayes_search.best_params_[original_name]).replace("OneHotEncoder(handle_unknown='ignore')",
                                                                                           "OneHotEncoder()"))

    def test_MLExperimentResults_all_models(self):
        search_space = BayesianSearchSpace(
            data=self.X_train,
            iterations=1,
            include_default_model=False
        )
        bayes_search = BayesSearchCV(
            estimator=search_space.pipeline(),  # noqa
            search_spaces=search_space.search_spaces(),  # noqa
            cv=RepeatedKFold(n_splits=3, n_repeats=1, random_state=42),  # 3 fold 1 repeat CV
            scoring='roc_auc',
            refit=False,  # required if use_label_encoder passing in multiple scorers
            return_train_score=False,
            n_jobs=-1,
            verbose=0,
            random_state=42,
        )
        # we get convergence warnings with LinearSVC, ignore for now; I do not see this when running with
        # other datasets
        bayes_search.fit(self.X_train, self.y_train)  # noqa

        results = MLExperimentResults.from_sklearn_search_cv(bayes_search,
                                                             parameter_name_mappings=search_space.param_name_mappings())
        self.assertIsNotNone(results)

    def test_MLExperimentResults_many_models(self):
        # loads in a yaml created from running ClassifierSearchSpace in external jupyter notebook
        # the purpose is not to check the values against the sklearn cv_results_ object, but rather to
        # make sure that the results can be parsed by MLExperimentResults
        results = MLExperimentResults.from_yaml_file(get_test_path() + '/test_files/sklearn_search/multi-model-many.yaml')

        with open(get_test_path() + '/test_files/sklearn_search/multi-model-many-all.html', 'w') as file:
            file.write(clean_formatted_dataframe(results.to_formatted_dataframe(num_rows=1000,
                                                                                include_rank=True).render()))

        with open(get_test_path() + '/test_files/sklearn_search/multi-model-many-num_rows.html', 'w') as file:
            file.write(clean_formatted_dataframe(results.to_formatted_dataframe(num_rows=40,
                                                                                return_style=True).render()))

        with open(get_test_path() + '/test_files/sklearn_search/multi-model-many-num_rows.html', 'w') as file:
            file.write(clean_formatted_dataframe(results.to_formatted_dataframe(num_rows=40,
                                                                                return_style=True).render()))

        with open(get_test_path() + '/test_files/sklearn_search/multi-model-many-XGBClassifier.html', 'w') as file:
            file.write(clean_formatted_dataframe(results.to_formatted_dataframe(query='model == "XGBClassifier()"').render()))

        labeled_dataframe = results.to_labeled_dataframe()
        self.assertTrue(all(labeled_dataframe['Trial Index'] == list(range(1, results.number_of_trials + 1))))
        self.assertIsNotNone(labeled_dataframe['label'])
        validation.assert_dataframes_match(dataframes=[results.to_dataframe(sort_by_score=False),
                                                       labeled_dataframe.drop(columns=['Trial Index',
                                                                                       'label'])])

        def to_string(obj):
            return str(obj).\
                replace(", '", ",\n'").\
                replace('{', '{\n').\
                replace('}', '\n}').\
                replace(', ({', ',\n({')

        labels = results.trial_labels(order_from_best_to_worst=True)
        with open(get_test_path() + '/test_files/sklearn_search/multi-model-many-trial_labels_sorted.txt', 'w') as file:
            file.write(to_string(labels))
        del labels

        labels = results.trial_labels(order_from_best_to_worst=False)
        with open(get_test_path() + '/test_files/sklearn_search/multi-model-many-trial_labels_not_sorted.txt', 'w') as file:
            file.write(to_string(labels))
        del labels

        _ = results.plot_performance_across_trials(facet_by='model')
        _ = results.plot_performance_across_trials(query='model == "XGBClassifier()"')
        _ = results.plot_parameter_values_across_trials(query='model == "XGBClassifier()"')
        _ = results.plot_parallel_coordinates(query='model == "XGBClassifier()"')
        _ = results.plot_scatter_matrix(query='model == "XGBClassifier()"')
        _ = results.plot_score_vs_parameter(parameter='max_depth', query='model == "XGBClassifier()"')
        _ = results.plot_parameter_values_across_trials(query='model == "XGBClassifier()"')
        _ = results.plot_parameter_vs_parameter(parameter_x='max_depth', parameter_y='min_child_weight',
                                                query='model == "XGBClassifier()"')
        _ = results.plot_performance_non_numeric_params(query='model == "XGBClassifier()"')
        _ = results.plot_performance_numeric_params(query='model == "XGBClassifier()"')
