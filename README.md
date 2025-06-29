# MLOps Zoomcamp 2025 - My Projects and Coursework

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-brightgreen.svg)](https://www.python.org/)

## About This Repository

This repository contains my personal solutions, projects, and notes for the assignments from the **[MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp)**, hosted by **DataTalks.Club**.

MLOps Zoomcamp is a practical, hands-on course focused on the operational aspects of machine learning. The curriculum covers the entire ML model lifecycle, from experiment tracking and orchestration to deployment, monitoring, and applying software engineering best practices.

## Repository Structure

Each directory corresponds to a specific course module, containing the relevant code, artifacts, and homework solutions.

*   **[`01-intro/`](./01-intro/)**: An introduction to MLOps, the model lifecycle, and building a foundational model.
*   **[`02-experiment-tracking/`](./02-experiment-tracking/)**: Tracking experiments, logging parameters, metrics, and managing models using **MLflow**.
*   **[`03-orchestration/`](./03-orchestration/)**: Orchestrating machine learning pipelines with **Mage AI** and/or **Prefect**.
*   **[`04-deployment/`](./04-deployment/)**: Deploying models for both batch scoring and real-time online inference using **Docker**.
*   **[`05-monitoring/`](./05-monitoring/)**: Monitoring ML model health, analyzing performance, and detecting data drift with **Evidently AI** and **Grafana**.
*   **[`06-best-practices/`](./06-best-practices/)**: Applying software engineering best practices, including unit testing, integration testing, linting, and an introduction to CI/CD.
*   **[`<other-modules>/`](./)**: (Future modules will be linked here)

## Key Skills & Technologies Learned

Throughout this course, I have gained practical experience with the following tools and concepts:

*   **Experiment Tracking**:
    *   **MLflow**: Logging parameters, metrics, and artifacts; managing the model lifecycle through the Model Registry.

*   **Pipeline Orchestration**:
    *   **Mage AI / Prefect**: Building, scheduling, and monitoring complex data processing and model training pipelines.

*   **ML Model Deployment**:
    *   **Docker**: Containerizing ML applications to ensure consistent and reproducible environments.
    *   **Flask / FastAPI**: Building web services to serve models for real-time predictions.
    *   **Batch & Online Scoring**: Implementing and managing both major deployment patterns.

*   **Model & Data Monitoring**:
    *   **Evidently AI**: Generating detailed reports to detect data drift, concept drift, and model performance degradation.
    *   **Grafana**: Visualizing model metrics and system health on interactive dashboards.

*   **Software Engineering for MLOps**:
    *   **Pytest**: Writing and running unit and integration tests for ML pipelines.
    *   **S3-Compatible Storage (MinIO, LocalStack)**: Using object storage for artifacts, models, and data.
    *   **CI/CD**: Automating testing and deployment workflows with GitHub Actions.
    *   **Infrastructure as Code (IaC)**: Managing services declaratively using `docker-compose`.

## General Setup and Usage

Most projects in this repository follow a similar setup process:

1.  Navigate to the relevant module's directory:
    ```bash
    cd <module-directory>
    ```

2.  Create and activate a Python virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  Install the required dependencies (if a `requirements.txt` file is present):
    ```bash
    pip install -r requirements.txt
    ```

4.  Follow the instructions in the specific module's `README.md` to run the code or tests.

## Acknowledgements

A huge thank you to the **[DataTalks.Club](https://datatalks.club/)** team, **Alexey Grigorev**, and the entire community for creating and supporting this invaluable, free, and practical course.