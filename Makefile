docker_build:
	docker build -t helpsk .

docker_run:
	docker run -d helpsk

#################################################################################
# GLOBALS
#################################################################################
PYTHON_VERSION := 3.9
PYTHON_VERSION_SHORT := $(subst .,,$(PYTHON_VERSION))
PYTHON_INTERPRETER := python$(PYTHON_VERSION)
SNOWFLAKE_VERSION := 2.7.4

#################################################################################
# Project-specific Commands
#################################################################################

## Run unit-tests
tests: environment
	@echo "\n[MAKE tests] >>>  Running unit tests."
	. .venv/bin/activate && $(PYTHON_INTERPRETER) -m unittest discover tests

## Build 
build:
	rm -fr dist
	@echo "\n[MAKE build] >>>  Building package."
	. .venv/bin/activate && $(PYTHON_INTERPRETER) -m build
	@echo "\n[MAKE build] >>>  Uploading package."
	. .venv/bin/activate && twine upload dist/*

## Delete all generated files (e.g. virtual environment)
clean:
	rm -rf .venv
	rm -rf dist

#################################################################################
# Generic Commands
#################################################################################

## Set up python virtual environment and install python dependencies
environment:
ifneq ($(wildcard .venv/.*),)
	@echo "\n[MAKE environment] >>>  Found .venv, skipping virtual environment creation."
	@echo "\n[MAKE environment] >>>  Activating virtual environment."
	@echo "\n[MAKE environment] >>>  Installing packages from requirements.txt."
	. .venv/bin/activate && pip install -q -r requirements.txt
else
	@echo "\n[MAKE environment] >>>  Did not find .venv, creating virtual environment."
	python -m pip install --upgrade pip
	python -m pip install -q virtualenv
	@echo "\n[MAKE environment] >>>  Installing virtualenv."
	virtualenv .venv --python=$(PYTHON_INTERPRETER)
	@echo "\n[MAKE environment] >>>  NOTE: Creating environment at .venv."
	@echo "\n[MAKE environment] >>>  NOTE: To activate virtual environment, run: 'source .venv/bin/activate'."
	@echo "\n[MAKE environment] >>>  Activating virtual environment."
	@echo "\n[MAKE environment] >>>  Installing packages from requirements.txt."
	. .venv/bin/activate && $(PYTHON_INTERPRETER) -m pip install --upgrade pip
	. .venv/bin/activate && $(PYTHON_INTERPRETER) -m pip install --upgrade build
	. .venv/bin/activate && $(PYTHON_INTERPRETER) -m pip install --upgrade twine
	. .venv/bin/activate && pip install -r requirements.txt

	@echo $(call FORMAT_MESSAGE,"environment_python","Installing snowflake packages.")
	. .venv/bin/activate && pip install -r https://raw.githubusercontent.com/snowflakedb/snowflake-connector-python/v$(SNOWFLAKE_VERSION)/tested_requirements/requirements_$(PYTHON_VERSION_SHORT).reqs
	. .venv/bin/activate && pip install snowflake-connector-python==v$(SNOWFLAKE_VERSION)

endif

#################################################################################
# Self Documenting Commands
#################################################################################
.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
