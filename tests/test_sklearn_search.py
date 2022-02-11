import re
import unittest

import numpy as np
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize, MinMaxScaler, StandardScaler, OneHotEncoder
from skopt import BayesSearchCV

from helpsk import validation
from helpsk.pandas import print_dataframe
from helpsk.sklearn_eval import MLExperimentResults
from helpsk.sklearn_pipeline import CustomOrdinalEncoder
from helpsk.sklearn_search import ClassifierSearchSpace, ClassifierSearchSpaceModels
from helpsk.utility import redirect_stdout_to_file
from tests.helpers import get_data_credit, get_test_path, clean_formatted_dataframe


# noinspection PyMethodMayBeStatic
class TestSklearnSearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        credit_data = get_data_credit()
        credit_data.loc[0:46, ['duration']] = np.nan
        credit_data.loc[25:75, ['checking_status']] = np.nan
        credit_data.loc[10:54, ['credit_amount']] = 0
        y_full = credit_data['target']
        X_full = credit_data.drop(columns='target')  # noqa
        y_full = label_binarize(y_full, classes=['good', 'bad']).flatten()
        X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=42)  # noqa
        del y_full, X_full
        cls.default_search_space = ClassifierSearchSpace(data=X_train)
        cls.X_train = X_train
        cls.y_train = y_train
        cls.search_space_used = ClassifierSearchSpace(
            data=X_train,
            models=[
                ClassifierSearchSpaceModels.XGBoost,
                ClassifierSearchSpaceModels.LogisticRegression
            ],
            iterations=[5, 5]
        )
        # pip install scikit-optimize

        cls.bayes_search = BayesSearchCV(
            estimator=cls.search_space_used.pipeline(),  # noqa
            search_spaces=cls.search_space_used.search_spaces(),  # noqa
            cv=RepeatedKFold(n_splits=3, n_repeats=1, random_state=42),  # 3 fold 1 repeat CV
            scoring='roc_auc',
            refit=False,  # required if passing in multiple scorers
            return_train_score=False,
            n_jobs=-1,
            verbose=0,
            random_state=42,
        )
        cls.bayes_search.fit(X_train, y_train)  # noqa

    def test_ClassifierSearchSpace(self):

        transformer_search_space = ClassifierSearchSpace._build_transformer_search_space()
        self.assertEqual(list(transformer_search_space.keys()),
                         ['prep__numeric__imputer__transformer',
                          'prep__numeric__scaler__transformer',
                          'prep__non_numeric__encoder__transformer'])

        categorical = transformer_search_space['prep__numeric__imputer__transformer']
        self.assertEqual(len(categorical.categories), 3)
        self.assertEqual(categorical.categories[0].strategy, 'mean')
        self.assertEqual(categorical.categories[1].strategy, 'median')
        self.assertEqual(categorical.categories[2].strategy, 'most_frequent')
        del categorical

        categorical = transformer_search_space['prep__numeric__scaler__transformer']
        self.assertEqual(len(categorical.categories), 3)
        self.assertIsNone(categorical.categories[0])
        self.assertIsInstance(categorical.categories[1], MinMaxScaler)
        self.assertIsInstance(categorical.categories[2], StandardScaler)
        del categorical

        categorical = transformer_search_space['prep__non_numeric__encoder__transformer']
        self.assertEqual(len(categorical.categories), 2)
        self.assertIsInstance(categorical.categories[0], OneHotEncoder)
        self.assertIsInstance(categorical.categories[1], CustomOrdinalEncoder)
        del categorical

        transformer_search_space = ClassifierSearchSpace._build_transformer_search_space(
            imputer_strategies=['most_frequent'],
            scaler_min_max=False,
            scaler_standard=False,
            encoder_one_hot=True,
            encoder_ordinal=False
        )
        self.assertEqual(list(transformer_search_space.keys()),
                         ['prep__numeric__imputer__transformer',
                          'prep__numeric__scaler__transformer',
                          'prep__non_numeric__encoder__transformer'])

        categorical = transformer_search_space['prep__numeric__imputer__transformer']
        self.assertEqual(len(categorical.categories), 1)
        self.assertEqual(categorical.categories[0].strategy, 'most_frequent')
        del categorical

        categorical = transformer_search_space['prep__numeric__scaler__transformer']
        self.assertEqual(len(categorical.categories), 1)
        self.assertIsNone(categorical.categories[0])
        del categorical

        categorical = transformer_search_space['prep__non_numeric__encoder__transformer']
        self.assertEqual(len(categorical.categories), 1)
        self.assertIsInstance(categorical.categories[0], OneHotEncoder)
        del categorical

        def to_string(obj):
            return str(obj).\
                replace(", '", ",\n'").\
                replace('{', '{\n').\
                replace('}', '\n}').\
                replace(', ({', ',\n({')

        # default space for logistic regression
        logistic_space = ClassifierSearchSpace._search_space_logistic()
        self.assertEqual(list(logistic_space.keys()),
                         ['model',
                          'model__C',
                          'prep__numeric__imputer__transformer',
                          'prep__numeric__scaler__transformer',
                          'prep__non_numeric__encoder__transformer'])

        with open(get_test_path() + '/test_files/sklearn_search/logistic_space_default.txt', 'w') as file:
            file.write(to_string(logistic_space))
        del logistic_space

        # search space for logistic regression with modified params
        logistic_space = ClassifierSearchSpace._search_space_logistic(
            solver='sag',
            max_iter=999,
            C=(1e-5, 1e+3),  # noqa
            C_prior='uniform',  # noqa
            imputer_strategies=['most_frequent'],  # noqa
            random_state=42
        )
        self.assertEqual(list(logistic_space.keys()),
                         ['model',
                          'model__C',
                          'prep__numeric__imputer__transformer',
                          'prep__numeric__scaler__transformer',
                          'prep__non_numeric__encoder__transformer'])
        with open(get_test_path() + '/test_files/sklearn_search/logistic_space_modified.txt', 'w') as file:
            file.write(to_string(logistic_space))
        del logistic_space

        # default space for xgboost
        xgboost_space = ClassifierSearchSpace._search_space_xgboost()
        self.assertEqual(list(xgboost_space.keys()),
                         ['model',
                          'model__max_depth',
                          'model__n_estimators',
                          'model__learning_rate',
                          'model__colsample_bytree',
                          'model__subsample',
                          'prep__numeric__imputer__transformer',
                          'prep__numeric__scaler__transformer',
                          'prep__non_numeric__encoder__transformer'])
        with open(get_test_path() + '/test_files/sklearn_search/xgboost_space_default.txt', 'w') as file:
            file.write(to_string(xgboost_space))
        del xgboost_space

        # search space for xgboost with modified params
        xgboost_space = ClassifierSearchSpace._search_space_xgboost(
            eval_metric='logloss',
            use_label_encoder=False,
            max_depth=(2, 30),
            n_estimators=(10, 10000),
            learning_rate=(0.01111, 0.3333),
            colsample_bytree=(0.01234, 1123),
            subsample=(0.1111, 0.999),
            imputer_strategies=['median'],
            random_state=42
        )
        self.assertEqual(list(xgboost_space.keys()),
                         ['model',
                          'model__max_depth',
                          'model__n_estimators',
                          'model__learning_rate',
                          'model__colsample_bytree',
                          'model__subsample',
                          'prep__numeric__imputer__transformer',
                          'prep__numeric__scaler__transformer',
                          'prep__non_numeric__encoder__transformer'])
        with open(get_test_path() + '/test_files/sklearn_search/xgboost_space_modified.txt', 'w') as file:
            file.write(to_string(xgboost_space))
        del xgboost_space

        pipeline = self.default_search_space.pipeline()
        with open(get_test_path() + '/test_files/sklearn_search/pipeline_default.txt', 'w') as file:
            file.write(to_string(pipeline))
        del pipeline

        search_spaces = self.default_search_space.search_spaces()
        with open(get_test_path() + '/test_files/sklearn_search/search_spaces_default.txt', 'w') as file:
            file.write(to_string(search_spaces))
        del search_spaces

        mappings = self.default_search_space.param_name_mappings()
        with open(get_test_path() + '/test_files/sklearn_search/param_name_mappings_default.txt', 'w') as file:
            file.write(to_string(mappings))

    def test_MLExperimentResults_multi_model(self):

        from xgboost import XGBClassifier
        XGBClassifier()

        results = MLExperimentResults.from_sklearn_search_cv(
            searcher=self.bayes_search,
            higher_score_is_better=True,
            description='BayesSearchCV using ClassifierSearchSpace',
            parameter_name_mappings=self.search_space_used.param_name_mappings()
        )

        results.to_yaml_file(get_test_path() + '/test_files/sklearn_search/multi-model-search.yaml')

        with redirect_stdout_to_file(get_test_path() + '/test_files/sklearn_search/multi-model-search-dataframe.txt'):
            print_dataframe(results.to_dataframe())

        with redirect_stdout_to_file(get_test_path() + '/test_files/sklearn_search/multi-model-search-dataframe__xgboost.txt'):
            print_dataframe(results.to_dataframe(query="model == 'XGBClassifier(...)'"))

        with redirect_stdout_to_file(get_test_path() + '/test_files/sklearn_search/multi-model-search-dataframe__logistic.txt'):
            print_dataframe(results.to_dataframe(query="model == 'LogisticRegression(...)'"))

        with open(get_test_path() + '/test_files/sklearn_search/multi-model-search-dataframe.html', 'w') as file:
            file.write(clean_formatted_dataframe(results.to_formatted_dataframe().render()))

        with open(get_test_path() + '/test_files/sklearn_search/multi-model-search-dataframe__xgboost.html', 'w') as file:
            file.write(clean_formatted_dataframe(results.to_formatted_dataframe(query="model == 'XGBClassifier(...)'").render()))

        with open(get_test_path() + '/test_files/sklearn_search/multi-model-search-dataframe__logistic.html', 'w') as file:
            file.write(clean_formatted_dataframe(results.to_formatted_dataframe(query="model == 'LogisticRegression(...)'").render()))

        # pd.set_option('display.max_columns', 500)
        # pd.set_option('display.width', 10000)

        def label_column_formatter(label_column):
            return [str(value).replace('<br>', '; ').replace('; model', '\nmodel').replace('; ', '\n       ') for value in label_column]

        labeled_df = results.to_labeled_dataframe()
        with redirect_stdout_to_file(get_test_path() + '/test_files/sklearn_search/multi-model-search-labeled_dataframe.txt'):
            print_dataframe(labeled_df)
        with open(get_test_path() + '/test_files/sklearn_search/multi-model-search-labeled_dataframe__label.txt', 'w') as file:
            for x in label_column_formatter(labeled_df['label']):
                file.write(x)

        labeled_df = results.to_labeled_dataframe(query="model == 'XGBClassifier(...)'")
        with redirect_stdout_to_file(get_test_path() + '/test_files/sklearn_search/multi-model-search-labeled_dataframe__xgboost.txt'):
            print_dataframe(labeled_df)
        with open(get_test_path() + '/test_files/sklearn_search/multi-model-search-labeled_dataframe__xgboost__label.txt', 'w') as file:
            for x in label_column_formatter(labeled_df['label']):
                file.write(x)

        labeled_df = results.to_labeled_dataframe(query="model == 'LogisticRegression(...)'")
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

    def test_MLExperimentResults_multi_model2(self):
        # test grid search object that has one score (classification)
        # not passing in parameter mappings
        bayes_search = self.bayes_search
        results = MLExperimentResults.from_sklearn_search_cv(
            searcher=self.bayes_search,
            higher_score_is_better=True,
            description='BayesSearchCV using ClassifierSearchSpace',
            parameter_name_mappings=self.search_space_used.param_name_mappings()
        )
        self.assertEqual(results.higher_score_is_better, True)
        self.assertEqual(results.cross_validation_type, "<class 'skopt.searchcv.BayesSearchCV'>")
        self.assertEqual(results.number_of_splits,
                         bayes_search.cv.n_repeats * bayes_search.n_splits_)
        self.assertEqual(results.score_names, ['roc_auc'])
        self.assertEqual(results.parameter_names_original,
                         list(self.search_space_used.param_name_mappings().keys()))
        self.assertEqual(results.parameter_names, list(self.search_space_used.param_name_mappings().values()))
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
            return re.sub(r'\(.*', '', str(obj).replace('\n', ''))

        self.assertEqual([remove_parentheses(x) for x in bayes_search.cv_results_['param_model'].data],
                         [remove_parentheses(x) for x in cv_dataframe['model']])

        # check that all of the hyper-param values match in the dataframe vs bayes_search.cv_results_
        hyper_param_df = cv_dataframe.iloc[:, 4:]
        for key, value in self.search_space_used.param_name_mappings().items():
            if key != 'model':
                mask = bayes_search.cv_results_['param_' + key].mask
                cv_data = list(bayes_search.cv_results_['param_' + key].data)
                cv_data = [np.nan if m else d for m, d in zip(mask, cv_data)]

                if key.startswith('model__'):
                    assert_np_arrays_are_close(np.array(cv_data),
                                               np.array(hyper_param_df.loc[:, value].tolist()))
                if key.startswith('prep__'):
                    self.assertEqual([str(x) for x in cv_data],
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
                                     str(bayes_search.best_params_[original_name]))

    def test_MLExperimentResults_all_models(self):
        search_space = ClassifierSearchSpace(
            data=self.X_train,
            iterations=[1] * len(ClassifierSearchSpaceModels.list())
        )
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
            file.write(clean_formatted_dataframe(results.to_formatted_dataframe(query='model == "XGBClassifier(...)"').render()))

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
        _ = results.plot_performance_across_trials(query='model == "XGBClassifier(...)"')
        _ = results.plot_parameter_values_across_trials(query='model == "XGBClassifier(...)"')
        _ = results.plot_parallel_coordinates(query='model == "XGBClassifier(...)"')
        _ = results.plot_scatter_matrix(query='model == "XGBClassifier(...)"')
        _ = results.plot_score_vs_parameter(query='model == "XGBClassifier(...)"')
        _ = results.plot_parameter_values_across_trials(query='model == "XGBClassifier(...)"')
        _ = results.plot_parameter_vs_parameter(query='model == "XGBClassifier(...)"')
        _ = results.plot_performance_non_numeric_params(query='model == "XGBClassifier(...)"')
        _ = results.plot_performance_numeric_params(query='model == "XGBClassifier(...)"')
