docker_build:
	docker compose -f docker-compose.yml up --build

docker_run:
	docker exec -it python-helpers-bash-1 /bin/zsh

## Run unit-tests
tests:
	python -m unittest discover tests

## Build package
build:
	rm -fr dist
	python -m build
	twine upload dist/*

## Delete all generated files
clean:
	rm -rf dist
