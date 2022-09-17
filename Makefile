.PHONY: tests

####
# docker commands
####
docker_build:
	cp ~/.pypirc ./.pypirc
	docker compose -f docker-compose.yml build
	rm -f .pypirc

docker_run: docker_build
	docker compose -f docker-compose.yml up

docker_down:
	docker compose down --remove-orphans

docker_rebuild:
	cp ~/.pypirc ./.pypirc
	docker compose -f docker-compose.yml build --no-cache
	rm -f .pypirc

docker_zsh:
	docker exec -it python-helpers-bash-1 /bin/zsh

docker_tests:
	docker compose run --no-deps --entrypoint "make tests" bash

docker_package:
	docker compose run --no-deps --entrypoint "make package" bash

all: docker_build docker_tests, docker_package

####
# conda commands
####
# conda activate python_helpers
env:
	conda env create -f environment.yml

export_env:
	conda env export > environment.yml	

remove_env:
	conda env remove -n $(conda_env_name)

####
# project commands
####
# commands to run inside docker container
linting:
	flake8 --max-line-length 99 --ignore=E127 helpsk

unittest:
	rm -f tests/test_files/logging/log.log
	python -m unittest discover tests

doctest:
	python -m doctest helpsk/text.py

tests: linting unittest doctest	

## Build package
package: tests clean
	rm -fr dist
	python -m build
	twine upload dist/*

## Delete all generated files
clean:
	rm -rf dist
	rm -rf helpsk.egg-info
