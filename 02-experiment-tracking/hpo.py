import os
import pickle
import click
import mlflow
import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Убедись, что этот URI соответствует твоему запущенному серверу MLflow из Q4
# Если сервер на порту 5001, измени здесь:
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000" # Пример, если сервер на 5001
EXPERIMENT_NAME = "random-forest-hyperopt"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
@click.option(
    "--num_trials",
    default=15, # По умолчанию 15, как в твоем коде
    type=int,   # Добавил type=int для num_trials, чтобы click правильно парсил
    help="The number of parameter evaluations for the optimizer to explore"
)
def run_optimization(data_path: str, num_trials: int):

    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    def objective(params):
        # Так как search_space использует scope.int, params уже должны содержать int для этих ключей.
        # Но для явности и если бы scope.int не использовался, можно было бы сделать так:
        # params_int = params.copy()
        # params_int['max_depth'] = int(params['max_depth'])
        # params_int['n_estimators'] = int(params['n_estimators'])
        # params_int['min_samples_split'] = int(params['min_samples_split'])
        # params_int['min_samples_leaf'] = int(params['min_samples_leaf'])
        # rf = RandomForestRegressor(**params_int)

        with mlflow.start_run(): # <--- Начинаем MLflow run для каждой итерации
            mlflow.set_tag("optimizer", "hyperopt")
            # Логируем параметры, которые пришли от hyperopt
            # (они уже должны быть правильного типа благодаря scope.int)
            mlflow.log_params(params) 

            rf = RandomForestRegressor(**params) # Используем params напрямую
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_val)
            rmse = mean_squared_error(y_val, y_pred, squared=False)

            mlflow.log_metric("rmse", rmse) # <--- Логируем RMSE

        return {'loss': rmse, 'status': STATUS_OK}

    search_space = {
        'max_depth': scope.int(hp.quniform('max_depth', 1, 20, 1)),
        'n_estimators': scope.int(hp.quniform('n_estimators', 10, 50, 1)),
        'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1)),
        'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 4, 1)),
        'random_state': 42 # Фиксированный random_state для RandomForestRegressor
    }

    rstate = np.random.default_rng(42)  # for reproducible results for hyperopt's search
    fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=num_trials,
        trials=Trials(),
        rstate=rstate
    )


if __name__ == '__main__':
    run_optimization()