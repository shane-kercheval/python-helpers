# helpsk

Helper package for python.

- package source in `/src/helpsk`
- unit tests in `tests`

## Installing

`pip install helpsk`

## Pre-Checkin

### Unit Tests

The unit tests in this project are all found in the `tests` directory.

In the terminal, run the following in the root project directory:

```commandline
cd python-helpers
python -m unittest discover ./tests
```

### `pylint`

Run pylint to maintain clean code.

```commandline
cd python-helpers
pylint helpsk
```


## search_space


BayesSearchCV.search_spaces

```
[
    (
        {
            'model': Categorical(categories=(LogisticRegression(max_iter=1000, random_state=42),), prior=None),
            'model__C': Real(low=1e-06, high=100, prior='log-uniform', transform='identity'),
            'prep__numeric__imputer__transformer': Categorical(categories=(SimpleImputer(), SimpleImputer(strategy='median'), SimpleImputer(strategy='most_frequent')), prior=[0.5, 0.25, 0.25]),
            'prep__numeric__scaler__transformer': Categorical(categories=(StandardScaler(), MinMaxScaler()), prior=[0.65, 0.35]),
            'prep__numeric__pca__transformer': Categorical(categories=(None, PCA(n_components='mle')), prior=None),
            'prep__non_numeric__encoder__transformer': Categorical(categories=(OneHotEncoder(handle_unknown='ignore'), CustomOrdinalEncoder()), prior=[0.65, 0.35])
        },
        45
    ),
    (
        {
            'model': Categorical(categories=(LogisticRegression(max_iter=1000, random_state=42),), prior=None),
            'prep__numeric__imputer__transformer': Categorical(categories=(SimpleImputer(),), prior=None),
            'prep__numeric__scaler__transformer': Categorical(categories=(StandardScaler(),), prior=None),
            'prep__numeric__pca__transformer': Categorical(categories=(None,), prior=None),
            'prep__non_numeric__encoder__transformer': Categorical(categories=(OneHotEncoder(handle_unknown='ignore'),), prior=None)
        }, 
        1
    ),
]
```

Whereas GridSearchCV.param_grid expects something like

```
[
    {
        'preparation__non_numeric_pipeline__encoder_chooser__transformer': [
            OneHotEncoder(),
            CustomOrdinalEncoder(),
        ],
        'model__min_samples_split': [2],
        'model__max_features': [100, 'auto'],
        'model__n_estimators': [10, 50],
    },
    {
        'preparation__non_numeric_pipeline__encoder_chooser__transformer': [
            OneHotEncoder(),
        ],
        'model__min_samples_split': [2],
        'model__max_features': [100],
        'model__n_estimators': [10, 50],
    },
]
```


        param_space = {key:value for key, value in self._transformation_space.items()}
        param_space.update(self._model_space)

        if self._include_default_space:
            default

        return param_space


transformation_pipeline=transformation_pipeline,
transformation_space=transformation_space,
model_space=model_space,
include_default_space=include_default_space,
random_state=random_state,