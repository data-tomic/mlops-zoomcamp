Of course. Creating a high-quality `README.md` is the final and most important step to make your project understandable, reproducible, and professional.

Based on our entire journey, here is a comprehensive `README.md` file written in English. It covers the project's architecture, file structure, dataset, and provides a clear, step-by-step guide for anyone to set up and run your project.

**Recommendation:** Copy the entire content below and replace the content of your `07-project/README.md` file with it.

---

# End-to-End MLOps Pipeline for Neonatal Sepsis Prediction

![CI Badge](https://github.com/data-tomic/mlops-zoomcamp/actions/workflows/ci.yml/badge.svg)

This project demonstrates a complete, end-to-end MLOps lifecycle for a machine learning model that predicts mortality outcomes for neonatal sepsis patients based on clinical data. It serves as a practical example of MLOps best practices, including experiment tracking, model versioning, containerized deployment, and CI/CD for testing.

The primary goal is to build a reliable and reproducible system that automates the process from data processing to a live prediction API.

## Core Technologies

-   **Containerization:** Docker & Docker Compose
-   **Experiment Tracking & Model Registry:** MLflow (Client-Server setup)
-   **Prediction Service:** FastAPI
-   **Code Quality & Testing:** `pytest`, `flake8`, `black`
-   **CI/CD:** GitHub Actions
-   **Core Libraries:** Scikit-learn, Pandas, Hyperopt

## Project Architecture

This project uses a robust client-server architecture to decouple training from deployment:

1.  **MLflow Tracking Server:** A dedicated Docker container that runs the MLflow server. It is the central hub for all experiments, model artifacts, and metadata.
    -   **Backend Store:** Uses a SQLite database (`mlruns/mlflow.db`) to store metadata (parameters, metrics, etc.).
    -   **Artifact Store:** Uses the local filesystem (`mlruns/`) to store model files, which is made available to the container via a Docker volume.

2.  **Prediction API (`sepsis_api`):** A Docker container running a FastAPI application.
    -   On startup, it queries the MLflow Tracking Server over the internal Docker network to fetch the model version currently aliased as `"production"`.
    -   It exposes a `/predict` endpoint to serve predictions.

3.  **Local Training Scripts (`train.py`, `data_processing.py`):** These Python scripts are run locally. They communicate with the MLflow Tracking Server via HTTP to log experiments, artifacts, and register new model versions.

This setup ensures that the training environment and the production environment are completely separate but share a centralized model registry, which is a core MLOps principle.

## File Structure

The project is organized to separate concerns, making it clean and scalable.

```
07-project/
├── data/
│   └── clinical_data.csv      # The raw input dataset for the project.
├── deployment/
│   ├── Dockerfile             # Defines the Docker image for the FastAPI prediction service.
│   └── docker-compose.yml     # Defines and orchestrates the multi-container setup (API + MLflow Server).
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py            # The FastAPI application code.
│   ├── __init__.py
│   ├── data_processing.py     # Script to clean and prepare the raw data.
│   └── train.py               # Script to run HPO, train the model, and log to MLflow.
├── tests/
│   └── test_api.py            # Unit tests for the FastAPI application.
├── .gitignore                 # Specifies files and directories to be ignored by Git.
├── Makefile                   # Provides convenience commands for managing the project.
└── requirements.txt           # A list of all Python dependencies for the project.
```

## Dataset

The dataset used is `data/clinical_data.csv`, which contains clinical and laboratory data for neonatal patients.

-   **Target Variable:** The goal is to predict the `Outcome_int` column, where `1` likely represents mortality and `0` represents survival.
-   **Features:** The dataset includes a rich set of features such as:
    -   Demographics: `Age`, `Gender`, `Gestation`
    -   Vital Signs: `Temperature`, `Heart_rate`, `Breath_rate`
    -   Lab Results: `WBC`, `CRP`, `Lactate`, `Bilirubin`
    -   Immunological Markers: `CD64_NEU_DAY_1`, `HLA-DR_MON_MFI_DAY_1`
    -   Scoring Systems: `Total_SOFA`

## Setup and Installation

To run this project, you will need `git`, `Docker`, `Docker Compose`, and a local installation of `Python 3.12`.

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/data-tomic/mlops-zoomcamp.git
    cd mlops-zoomcamp/07-project
    ```

2.  **Set up the Local Environment**
    This project uses a `Makefile` to simplify setup. The `install` command will create a local Python virtual environment (`venv/`) and install all required dependencies from `requirements.txt`.

    ```bash
    # This command creates the venv and installs dependencies into it.
    make install
    ```
    *Note: This requires `python3.12` to be available on your system.*

3.  **Activate the Virtual Environment**
    Before running any local scripts, you must activate the environment.
    ```bash
    source venv/bin/activate
    ```

## End-to-End Workflow

This project is orchestrated by Prefect, allowing you to run the entire training pipeline with a single command.

### Step 1: Start Services
First, start the MLflow Tracking Server in the background.

```bash
make run-services
```
- This command starts the MLflow container.
- You can access the MLflow UI at **`http://127.0.0.1:5000`**.

### Step 2: Run the Orchestrated Pipeline
Instead of running individual scripts, you can now execute the entire workflow with one command. This will process the data and then train the model, logging everything to your running MLflow server.

```bash
make orchestrate
```

### Step 3: Promote the Model
This manual governance step remains the same.

1.  Go to the MLflow UI at **`http://127.0.0.1:5000`**.
2.  Navigate to the **"Models"** page and select the **`sepsis-outcome-classifier`** model.
3.  Find the latest version and assign it the **`production`** alias.

### Step 4: Deploy the Promoted Model
Restart the API service to force it to load the newly promoted model.

```bash
make restart-api
```

### Step 5: Verify and Test
1.  Check the API logs to confirm the model loaded successfully: `make logs-api`.
2.  Test the prediction endpoint via the interactive docs at **`http://127.0.0.1:8000/docs`**.

## Makefile Commands

This project uses a `Makefile` to provide a simple, unified interface for common tasks.

| Command             | Description                                                                 |
| ------------------- | --------------------------------------------------------------------------- |
| `make install`      | Creates a local Python virtual environment and installs all dependencies.     |
| `make run-services` | Builds and starts the MLflow and API containers in the background.            |
| `make stop-services`| Stops and removes all running project containers.                           |
| `make restart-api`  | Restarts only the API container, typically to load a newly promoted model.    |
| `make process-data` | Runs the local data processing script.                                      |
| `make train`        | Runs the local training and hyperparameter optimization script.             |
| `make test`         | Runs the local unit tests for the API using `pytest`.                       |
| `make lint`         | Checks the code for style issues using `flake8`.                            |
| `make format`       | Automatically formats the code using `black`.                               |
| `make logs-api`     | Follows the real-time logs for the `sepsis_api` container.                  |
| `make logs-mlflow`  | Follows the real-time logs for the `mlflow_tracking_server` container.        |