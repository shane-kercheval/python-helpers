# helpsk package

Helper package for python.

**NOTE: This package requires Python >=3.9**

- package source in `/helpsk`
- unit tests in `/tests`

---

## Installing

```commandline
pip install helpsk
```

---

## Contributing

### Coding Standards

- Coding standards should follow PEP 8 (Style Guide for Python Code)
    - https://peps.python.org/pep-0008/
    - Exceptions:
        - use max line length of `110` rather than the suggested `79`
- document all files, classes, functions
    - following existing documentation style


### Create Virtual Environment

The following command will install all dependencies as a virtual environment located in `./venv/`

```commandline
make environment
```

To activate the virtual environment in the terminal, use:

```commandline
source .venv/bin/activate
```

to deactivate, simply use:

```commandline
source .venv/bin/activate
```

To configure PyCharm (etc.) to use the virtual environment, point the environment to `.venv/bin/python3`

### Pre-Check-in

#### Unit Tests

The unit tests in this project are all found in the `/tests` directory.

In the terminal, in the project directory, either run the Makefile command,

```commandline
make tests
```

or the python command for running all tests

```commandline
python -m unittest discover ./tests
```
