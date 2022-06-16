docker_build:
	cp ~/.pypirc ./.pypirc
	docker compose -f docker-compose.yml up --build

docker_run:
	docker exec -it python-helpers-bash-1 /bin/zsh

unittests:
	python -m unittest discover tests

linting:
	flake8 --max-line-length 110 --ignore=E127 helpsk

## Build package
build:
	rm -fr dist
	python -m build
	twine upload dist/*

## Delete all generated files
clean:
	rm -rf dist
	rm -rf helpsk.egg-info
	rm -f .pypirc
