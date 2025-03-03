.PHONY: tests

-include .env
export

build-env:
	uv sync
####
# project commands
####
# commands to run inside docker container
linting:
	uv run ruff check helpsk

unittests:
	rm -f tests/test_files/logging/log.log
	uv run python -m pytest tests

doctest:
	uv run python -m doctest helpsk/text.py

tests: linting unittests doctest	

package-build:
	rm -rf dist/*
	uv build --no-sources

package-publish:
	uv publish --token ${UV_PUBLISH_TOKEN}

package: package-build package-publish

