import os
import pickle
import click
import mlflow

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient # MlflowClient уже импортирован
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

HPO_EXPERIMENT_NAME = "random-forest-hyperopt"
EXPERIMENT_NAME = "random-forest-best-models" # Эксперимент для оценки топ-N моделей
MODEL_REGISTRY_NAME = "GreenTaxiBestRF" # Имя для регистрируемой модели
RF_PARAMS = ['max_depth', 'n_estimators', 'min_samples_split', 'min_samples_leaf', 'random_state']

# Убедись, что этот URI соответствует твоему запущенному серверу MLflow из Q4
# Если сервер на порту 5001, измени здесь:
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000" # Пример, если сервер на 5000S
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
# mlflow.sklearn.autolog() # Автологирование здесь, вероятно, не нужно, так как мы логируем метрики и модель вручную.
                           # Но если оно не мешает и логирует модель, можно оставить.
                           # Задание в Q5 просило НЕ использовать автолог. Для Q6 нет явного указания,
                           # но ручное логирование модели более контролируемо.

def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

# Изменим функцию, чтобы она возвращала test_rmse и run_id нового запуска
def train_and_log_model(data_path, params, experiment_id_for_new_run):
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    # X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl")) # Валидационный набор не используется для финальной оценки
    X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))

    # Запускаем новый run в рамках эксперимента EXPERIMENT_NAME
    with mlflow.start_run(experiment_id=experiment_id_for_new_run, nested=True) as run: # nested=True если вызывается из другого run
        run_id = run.info.run_id
        mlflow.set_tag("source_hpo_run_params", str(params)) # Сохраняем исходные HPO параметры как тег

        # Преобразуем параметры к int
        new_params = {}
        for param_name in RF_PARAMS:
            if param_name in params: # Убедимся, что параметр есть в params из HPO
                 new_params[param_name] = int(float(params[param_name])) # float() на случай если строка

        mlflow.log_params(new_params) # Логируем параметры, с которыми обучалась модель

        rf = RandomForestRegressor(**new_params)
        rf.fit(X_train, y_train)

        # Оцениваем модель на тестовом наборе
        test_rmse = mean_squared_error(y_test, rf.predict(X_test), squared=False)
        mlflow.log_metric("test_rmse", test_rmse)
        print(f"Run ID {run_id}: Trained with params {new_params}, Test RMSE: {test_rmse}")

        # Логируем модель
        mlflow.sklearn.log_model(rf, "model")
        
        return test_rmse, run_id


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
@click.option(
    "--top_n",
    default=5,
    type=int,
    help="Number of top models that need to be evaluated to decide which one to promote"
)
def run_register_model(data_path: str, top_n: int):
    print(f"Attempting to connect to MLflow server at: {MLFLOW_TRACKING_URI}")
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    try:
        hpo_experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
        if hpo_experiment is None:
            print(f"ERROR: HPO Experiment '{HPO_EXPERIMENT_NAME}' not found on server {MLFLOW_TRACKING_URI}.")
            return
        hpo_experiment_id = hpo_experiment.experiment_id
        print(f"Found HPO experiment '{HPO_EXPERIMENT_NAME}' with ID: {hpo_experiment_id}")
    except Exception as e:
        print(f"ERROR: Could not get HPO experiment '{HPO_EXPERIMENT_NAME}'. Exception: {e}")
        return

    try:
        best_models_experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        if best_models_experiment is None:
            print(f"Experiment '{EXPERIMENT_NAME}' does not exist. Creating a new experiment.")
            best_models_experiment_id = client.create_experiment(EXPERIMENT_NAME)
            print(f"Created new experiment '{EXPERIMENT_NAME}' with ID: {best_models_experiment_id}")
        else:
            best_models_experiment_id = best_models_experiment.experiment_id
            print(f"Found existing experiment '{EXPERIMENT_NAME}' with ID: {best_models_experiment_id}")
        mlflow.set_experiment(experiment_id=best_models_experiment_id)
    except Exception as e:
        print(f"ERROR: Could not get/create experiment '{EXPERIMENT_NAME}'. Exception: {e}")
        return

    print(f"Searching top {top_n} runs in HPO experiment ID: {hpo_experiment_id}")
    try:
        runs = client.search_runs(
            experiment_ids=[hpo_experiment_id],
            run_view_type=ViewType.ACTIVE_ONLY,
            max_results=top_n,
            order_by=["metrics.rmse ASC"]
        )
    except Exception as e:
        print(f"ERROR: Failed to search runs in HPO experiment. Exception: {e}")
        return
        
    if not runs:
        print(f"WARNING: No runs found in HPO experiment '{HPO_EXPERIMENT_NAME}' (ID: {hpo_experiment_id}). Cannot proceed.")
        return
    
    print(f"Found {len(runs)} runs from HPO experiment to process.")

    evaluated_runs = []
    for run in runs:
        print(f"Processing HPO run {run.info.run_id} with params: {run.data.params}")
        test_rmse, new_run_id = train_and_log_model(
            data_path=data_path,
            params=run.data.params,
            experiment_id_for_new_run=best_models_experiment_id
        )
        evaluated_runs.append({'run_id': new_run_id, 'test_rmse': test_rmse, 'source_hpo_params': run.data.params})
    
    if not evaluated_runs:
        print(f"ERROR: No models were evaluated from HPO runs, something went wrong in the loop.")
        return

    best_evaluated_run = min(evaluated_runs, key=lambda x: x['test_rmse'])
    best_run_id = best_evaluated_run['run_id']
    lowest_test_rmse = best_evaluated_run['test_rmse']
    
    print(f"\nBest model for registration is from run_id: {best_run_id} with Test RMSE: {lowest_test_rmse}")
    print(f"Source HPO params for best model: {best_evaluated_run['source_hpo_params']}")

    model_uri = f"runs:/{best_run_id}/model"
    print(f"Registering model from URI: {model_uri} with name: {MODEL_REGISTRY_NAME}")
    try:
        mlflow.register_model(
            model_uri=model_uri,
            name=MODEL_REGISTRY_NAME
        )
        print(f"Model '{MODEL_REGISTRY_NAME}' version 1 registered successfully.")
    except Exception as e:
        print(f"ERROR registering model: {e}")

if __name__ == '__main__':
    run_register_model()