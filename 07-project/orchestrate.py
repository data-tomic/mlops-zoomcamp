cat <<EOF > src/orchestrate.py
# 07-project/src/orchestrate.py (Final Corrected Version)

from prefect import flow, task
# --- THIS IS THE FIX ---
# We import the correct function name 'run_data_processing' from our module.
from .data_processing import run_data_processing
# And the correct function name 'run_train_hpo' from our training module.
from .train import run_train_hpo
# ---------------------

@task(retries=3, retry_delay_seconds=10, name="Process Sepsis Data")
def process_data_task():
    """
    Prefect task to run the data processing script.
    It calls the core logic function with the project's default paths.
    """
    print("--- Running Data Processing Task ---")
    # And we call the correct function here.
    run_data_processing(
        raw_data_path='data/clinical_data.csv',
        dest_path='data/processed',
        target_col='Outcome_int'
    )

@task(name="Train Sepsis Model")
def train_model_task():
    """
    Prefect task to run the model training script.
    It calls the core logic function with the project's default parameters.
    """
    print("--- Running Model Training Task ---")
    run_train_hpo(
        data_path='data/processed',
        num_trials=20,
        model_name='sepsis-outcome-classifier'
    )

@flow(name="Sepsis Training Pipeline")
def sepsis_training_pipeline():
    """
    The main Prefect flow that orchestrates the data processing and
    model training tasks in the correct sequence.
    """
    print("=== Starting Sepsis Training Pipeline ===")
    
    # Submit the data processing task to run.
    process_data_result = process_data_task.submit()

    # Submit the training task, ensuring it only starts after the
    # data processing task has successfully completed.
    train_model_task.submit(wait_for=[process_data_result])

    print("=== Sepsis Training Pipeline Finished ===")


if __name__ == "__main__":
    # This allows the flow to be run directly from the command line.
    sepsis_training_pipeline()
EOF