.PHONY: container

CONTAINER_NAME := iui20
PWD := $(shell pwd)
RUN_IN_CONTAINER := docker run -it --rm -v "${PWD}:/code" -v "${PWD}/../models:/models" iui20
RUN_PYTHON = ${RUN_PYTHON}

container:
	docker build -t iui20 .
	
shell:
	${RUN_IN_CONTAINER} /bin/bash

## Run backend Python server.
run_backend:
	${RUN_PYTHON} -m tornado.autoreload -m textrec.runserver

## Run frontend server (devel)
run_frontend:
	cd src/frontend && exec yarn start

replay_recs:
	${RUN_PYTHON} scripts/replay_recs.py no_suggestions_texts.csv actions.csv

data/analyzed/trial_gc1.csv: src/textrec/logs_to_csv.py
	python -m textrec.logs_to_csv gc1

data/analyzed/trial_withmanual_gc1.csv: src/textrec/gruntwork.py data/analyzed/trial_gc1.csv data/gruntwork/corrections_batch*.csv
	python -m textrec.gruntwork gc1
