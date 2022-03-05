# helpsk

Helper package for python.

- package source in `/src/helpsk`
- unit tests in `tests`

## Installing

`pip install helpsk`

## Pre-Checkin

### Unit Tests

The unit tests in this project are all found in the `tests` directory.

In the terminal, in the project directory, either run the Makefile command,

```commandline
make tests
```

or the python command for running all tests

```commandline
python -m unittest discover ./tests
```

### `pylint`

Run pylint to maintain clean code.

```commandline
cd python-helpers
pylint helpsk
```
