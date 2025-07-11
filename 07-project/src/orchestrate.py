# 07-project/src/orchestrate.py (Final Corrected Version for Synchronous Execution)

from prefect import flow, task
from src.data_processing import run_data_processing
from src.train import run_train_hpo


@task(retries=3, retry_delay_seconds=10, name="Process Sepsis Data")
def process_data_task():
    """
    Prefect task to run the data processing script.
    """
    print("--- Running Data Processing Task ---")
    run_data_processing(
        raw_data_path="data/clinical_data.csv",
        dest_path="data/processed",
        target_col="Outcome_int",
    )


@task(name="Train Sepsis Model")
def train_model_task():
    """
    Prefect task to run the model training script.
    """
    print("--- Running Model Training Task ---")
    run_train_hpo(
        data_path="data/processed",
        num_trials=20,
        model_name="sepsis-outcome-classifier",
    )


@flow(name="Sepsis Training Pipeline")
def sepsis_training_pipeline():
    """
    The main Prefect flow that orchestrates the tasks sequentially.

    """
    print("=== Starting Sepsis Training Pipeline ===")

    # --- THIS IS THE FIX ---
    # We call the tasks directly, without .submit().
    # The flow will now wait for each task to complete before moving on.
    process_data_task()
    train_model_task()
    # -----------------------

    print("=== Sepsis Training Pipeline Finished Successfully ===")


if __name__ == "__main__":
    # This allows the flow to be run directly from the command line.
    sepsis_training_pipeline()
