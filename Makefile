.PHONY: tests

docker_run:
	cp ~/.pypirc ./.pypirc
	docker compose -f docker-compose.yml up --build

docker_rebuild:
	docker compose -f docker-compose.yml build --no-cache

docker_zsh:
	docker exec -it python-helpers-bash-1 /bin/zsh

linting:
	flake8 --max-line-length 110 --ignore=E127 helpsk

tests: linting
	python -m unittest discover tests

## Build package
build: tests
	rm -fr dist
	python -m build
	twine upload dist/*

## Delete all generated files
clean:
	rm -rf dist
	rm -rf helpsk.egg-info
	rm -f .pypirc
