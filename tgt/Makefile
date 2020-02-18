.PHONY: container

CONTAINER_NAME := iui20
PWD := $(shell pwd)
DOCKER_FILESYSTEM := -v "${PWD}:/code" -v "${PWD}/../models:/models"
DOCKER_NETWORK := -p 5000:5000
DOCKER_IMAGE := iui20
RUN_IN_CONTAINER := docker run -it ${DOCKER_FILESYSTEM} --rm ${DOCKER_IMAGE}
RUN_PYTHON_SCRIPT = ${RUN_IN_CONTAINER} poetry run python

container:
	docker build -t iui20 .
	
shell:
	${RUN_IN_CONTAINER} /bin/bash

## Run backend Python server.
run_backend:
	docker run -it ${DOCKER_FILESYSTEM} ${DOCKER_NETWORK} --rm ${DOCKER_IMAGE} poetry run python -m tornado.autoreload -m textrec.runserver

## Run frontend server (devel)
run_frontend:
	cd src/frontend && exec yarn start

replay_recs:
	${RUN_PYTHON_SCRIPT} scripts/replay_recs.py no_suggestions_texts.csv actions.csv

data/analyzed/trial_gc1.csv: src/textrec/logs_to_csv.py
	python -m textrec.logs_to_csv gc1

data/analyzed/trial_withmanual_gc1.csv: src/textrec/gruntwork.py data/analyzed/trial_gc1.csv data/gruntwork/corrections_batch*.csv
	python -m textrec.gruntwork gc1
