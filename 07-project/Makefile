# 07-project/Makefile (Final Corrected Version)

# Define the local python interpreter from the virtual environment
VENV_PYTHON = venv/bin/python

# --- SERVICE MANAGEMENT ---
# (This section remains unchanged)
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
# This target creates the venv and installs dependencies
install:
	python3.12 -m venv venv
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install -r requirements.txt

# Run the data processing script using the venv python
process-data:
	$(VENV_PYTHON) src/data_processing.py

# Run the training script using the venv python
train:
	$(VENV_PYTHON) src/train.py --num-trials 20

# Run tests with pytest from the virtual environment
test:
	$(VENV_PYTHON) -m pytest tests/

# Lint code with flake8 from the virtual environment
lint:
	$(VENV_PYTHON) -m flake8 src/ tests/

# Format code with black from the virtual environment
format:
	$(VENV_PYTHON) -m black src/ tests/