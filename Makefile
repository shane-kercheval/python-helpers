.PHONY: tests

####
# docker commands
####
docker_build:
	# build the docker container used to run tests and build package
	cp ~/.pypirc ./.pypirc
	docker compose -f docker-compose.yml build
	rm -f .pypirc

docker_run: docker_build
	# run the docker container
	docker compose -f docker-compose.yml up

docker_down:
	docker compose down --remove-orphans

docker_rebuild:
	# rebuild docker container
	cp ~/.pypirc ./.pypirc
	docker compose -f docker-compose.yml build --no-cache
	rm -f .pypirc

docker_zsh:
	# run container and open up zsh command-line
	docker exec -it python-helpers-bash-1 /bin/zsh

docker_tests:
	# run tests within docker container
	docker compose run --no-deps --entrypoint "make tests" bash

docker_package:
	# create package and upload via twine from within docker container
	docker compose run --no-deps --entrypoint "make package" bash

all: docker_build docker_tests, docker_package

####
# project commands
####
# commands to run inside docker container
linting:
	ruff check helpsk

unittests:
	rm -f tests/test_files/logging/log.log
	python -m pytest tests

doctest:
	python -m doctest helpsk/text.py

tests: linting unittests doctest	

## Build package
package: tests clean
	rm -fr dist
	python -m build
	twine upload dist/*

## Delete all generated files
clean:
	rm -rf dist
	rm -rf helpsk.egg-info
