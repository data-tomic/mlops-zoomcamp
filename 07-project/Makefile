# Makefile for the project in 07-project

.PHONY: install test lint format

# Assumes you are running 'make' from within the 07-project directory
# Or you can run 'make -C 07-project install' from the root

install:
	pip install -r requirements.txt

test:
	python -m pytest tests/

lint:
	flake8 src/

format:
	black src/