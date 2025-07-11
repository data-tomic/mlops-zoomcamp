# 07-project/Makefile (Final Corrected Version)

VENV_PYTHON = venv/bin/python

# --- SERVICE MANAGEMENT ---
run-services:
	docker compose -f deployment/docker-compose.yml up --build -d
stop-services:
	docker compose -f deployment/docker-compose.yml down
restart-api:
	docker compose -f deployment/docker-compose.yml restart sepsis_api
logs-api:
	docker compose -f deployment/docker-compose.yml logs -f sepsis_api
logs-mlflow:
	docker compose -f deployment/docker-compose.yml logs -f mlflow-server

# --- LOCAL DEVELOPMENT & TESTING ---
install:
	python3.12 -m venv venv
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install -r requirements.txt

process-data:
	$(VENV_PYTHON) -m src.data_processing

train:
	$(VENV_PYTHON) -m src.train --num-trials 20

# THIS IS THE CORRECTED COMMAND
orchestrate:
	$(VENV_PYTHON) -m src.orchestrate

test:
	$(VENV_PYTHON) -m pytest tests/
lint:
	$(VENV_PYTHON) -m flake8 src/ tests/
format:
	$(VENV_PYTHON) -m black src/ tests/