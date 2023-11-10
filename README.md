# helpsk package

Helper package for python.

**NOTE: This package is tested on Python `3.10`, `3.11`**

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
        - use max line length of `99` rather than the suggested `79`
- document all files, classes, functions
    - following existing documentation style


### Docker

See `Makefile` for all commands.

To build the docker container:

```commandline
make docker_build
```

To run the terminal inside the docker container:

```commandline
make docker_zsh
```

To run the unit tests (including linting and doc-tests) from the commandline inside the docker container:

```commandline
make tests
```

To run the unit tests (including linting and doc-tests) from the commandline outside the docker container:

```commandline
make docker_tests
```

To build the python package and uploat do PyPI via twine from the commandline outside the docker container:

```commandline
make all
```

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
