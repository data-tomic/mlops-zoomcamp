# Project: Monitoring for ML Batch Services

## 1. Project Goal

The primary goal of this project is to implement a monitoring system for a batch machine learning service. The service in question predicts taxi trip durations. This project demonstrates the process of preparing data, calculating data quality and drift metrics, storing them in a time-series-friendly database, and visualizing them on a dashboard.

This serves as a practical exercise in MLOps, focusing on the critical post-deployment phase of monitoring.

## 2. Key Technologies Used

*   **Python 3.11**
*   **Docker & Docker Compose:** For containerizing and running infrastructure services.
*   **PostgreSQL:** As a database to store the calculated metrics over time.
*   **Grafana:** For visualizing metrics and creating monitoring dashboards.
*   **Evidently:** The core library for calculating ML-specific metrics like data drift and data quality.
*   **Prefect:** (As seen in the original script) For orchestrating the monitoring workflow.
*   **Scikit-learn:** For the baseline machine learning model.

## 3. Setup and Installation

Follow these steps to set up the environment and run the project.

### Prerequisites

*   Docker and Docker Compose must be installed and running.
*   Python 3.11 should be installed on your system.

### Steps

1.  **Clone the Repository**
    ```bash
    git clone <your-repository-url>
    cd <repository-folder>/05-monitoring
    ```

2.  **Create and Activate Virtual Environment**
    It is crucial to use a dedicated virtual environment to manage dependencies and avoid conflicts.
    ```bash
    # Create the virtual environment
    python3 -m venv venv

    # Activate it
    source ./venv/bin/activate
    ```

3.  **Install Dependencies**
    The project comes with a `requirements.txt` file that specifies the exact versions of the libraries used, which is critical for reproducibility.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Launch Infrastructure**
    The `docker-compose.yml` file defines the PostgreSQL database and Grafana services. Launch them in detached mode:
    ```bash
    docker-compose up -d
    ```
    This will start both containers in the background. You can check their status with `docker-compose ps`.

## 4. How to Run the Monitoring Job

The core logic is contained in the `evidently_metrics_calculation.py` script (or our simplified `get_q3_answer.py`). This script simulates a daily batch job that:
*   Loads the March 2024 taxi data.
*   Iterates through each day of the month.
*   Calculates metrics using Evidently for that day's data.
*   Stores the results in the PostgreSQL database.

To run the calculation, execute the script from your activated virtual environment:
```bash
python <your_script_name>.py
```

## 5. Homework Results

This project successfully answered the following key questions:

#### Q1: Dataset Shape
The March 2024 Green Taxi dataset was downloaded and analyzed.
*   **Shape:** (57457, 20)
*   **Number of Rows:** **57,457**

#### Q2: Custom Metric Implementation
To enhance data quality monitoring, a new metric was added to track the median of the `fare_amount` column.
*   **Chosen Metric:** **`ColumnQuantileMetric`** (with `quantile=0.5`)

#### Q3: Monitoring Analysis
After running the daily monitoring script for the entire month of March 2024, the metrics were analyzed.
*   **Maximum Daily Median for `fare_amount`:** **14.2**

#### Q4: Dashboard Configuration
For persistence and version control, Grafana dashboard configurations should be saved as JSON files.
*   **Correct Location:** **`project_folder/dashboards`**

## 6. Lessons Learned

A key challenge during this project was navigating version discrepancies in the `evidently` library. The API has evolved significantly, with notable differences in module structure (e.g., `evidently.report` vs. `evidently`), class names (`ValueDrift` vs. `ColumnDriftMetric`), and result dictionary schemas.

**Resolution:** The most reliable path to success was strictly adhering to the versions specified in the `requirements.txt` file (`evidently==0.6.7`). This reinforces the critical importance of pinned dependencies for reproducible ML systems.