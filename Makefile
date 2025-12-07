
.PHONY: init list run all

init:
	python -m pipeline init

list:
	python -m pipeline list-steps

run:
	python -m pipeline run

dry:
	python -m pipeline run --dry-run

all: init dry
