from prefect import flow, task
from src.data_processing import process_data
from src.train import run_optimization


@task(retries=3, retry_delay_seconds=5)
def process_data_task(raw_data_dir, processed_data_dir):
    """Prefect-таск для запуска обработки данных."""
    print("--- Running Data Processing Task ---")

    process_data(raw_data_dir, processed_data_dir)
    print("--- Data Processing Task Finished ---")
    return processed_data_dir


@task
def train_model_task(processed_data_dir):
    """Prefect-таск для запуска обучения и HPO."""
    print("--- Running Model Training Task ---")
    run_optimization(
        data_path=processed_data_dir, num_trials=10
    )  # Запускаем 10 итераций HPO
    print("--- Model Training Task Finished ---")


@flow(name="TCGA-BRCA Training Pipeline")
def main_flow():
    """
    Основной воркфлоу, который объединяет обработку данных и обучение.
    """
    # Определяем пути
    raw_dir = "./data/raw"
    processed_dir = "./data/processed"

    # Запускаем таски
    processed_path = process_data_task(raw_dir, processed_dir)
    train_model_task(processed_path)


if __name__ == "__main__":
    main_flow()
