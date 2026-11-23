# helpsk package

Helper package for python.

helpsk supports Python 3.12 through 3.15. CI runs the tests on each of those versions.

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


### Development

The project uses [uv](https://docs.astral.sh/uv/). See the `Makefile` for all commands.

To install the dependencies, run:

```commandline
make build-env
```

To run linting, the unit tests in `/tests`, and the doctests, run:

```commandline
make tests
```

To build the package and publish it to PyPI, set `UV_PUBLISH_TOKEN`, then run:

```commandline
make package
```
