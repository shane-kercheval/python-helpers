.PHONY: tests

docker_build:
	cp ~/.pypirc ./.pypirc
	docker compose -f docker-compose.yml build
	rm -f .pypirc


docker_run: docker_build
	docker compose -f docker-compose.yml up

docker_rebuild:
	cp ~/.pypirc ./.pypirc
	docker compose -f docker-compose.yml build --no-cache
	rm -f .pypirc

docker_zsh:
	docker exec -it python-helpers-bash-1 /bin/zsh

linting:
	flake8 --max-line-length 110 --ignore=E127 helpsk

tests: linting
	rm -f tests/test_files/logging/log.log
	python -m unittest discover tests

## Build package
build: tests clean
	rm -fr dist
	python -m build
	twine upload dist/*

## Delete all generated files
clean:
	rm -rf dist
	rm -rf helpsk.egg-info
