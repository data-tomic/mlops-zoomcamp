# Module 2: Experiment Tracking with MLflow

## Objective

The goal of this homework was to use **MLflow** to track machine learning experiments. This involved logging parameters, metrics, and artifacts (like the trained model) for multiple runs to compare their performance and identify the best-performing model for the NYC taxi trip duration prediction task.

## Key Technologies
- **MLflow**: For tracking experiments, logging, and managing models.
- **Scikit-learn**: For model training (e.g., `RandomForestRegressor`).
- **Pandas**: For data manipulation.
- **PyArrow**: For working with Parquet files.

## Setup

1.  Create and activate a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

1.  **Prepare the data**: Make sure you have the necessary data files (e.g., `green_tripdata_2023-01.parquet`, etc.) in a `data/` subdirectory.

2.  **Run the main training script**: This script preprocesses the data, trains multiple models, and logs all relevant information to MLflow.
    ```bash
    python train.py
    ```

3.  **Launch the MLflow UI**: To inspect the results, compare runs, and see the logged artifacts, start the MLflow tracking server. The runs will be logged by default in a local `mlruns` directory.
    ```bash
    mlflow ui --backend-store-uri file:./mlruns
    ```
    Now, open your browser and navigate to `http://127.0.0.1:5000`.

4.  **Register the best model**: After identifying the best run in the UI, run the script to register it in the MLflow Model Registry.
    ```bash
    python register_model.py
    ```

## Key Learnings
- How to set up an MLflow tracking server.
- Logging parameters, metrics, and model artifacts for each experiment.
- Using the MLflow UI to compare different runs.
- Programmatically querying MLflow to find the best model and promote it to the Model Registry.