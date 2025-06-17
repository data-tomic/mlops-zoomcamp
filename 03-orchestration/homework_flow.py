#!/usr/bin/env python
# coding: utf-8

import pickle
import argparse
from pathlib import Path

import pandas as pd
import xgboost as xgb

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error

import mlflow
from prefect import task, flow


@task
def read_dataframe(year: int, month: int) -> pd.DataFrame:
    """Читает данные для YELLOW taxi и выполняет базовую предобработку."""
    
    # Используем датасет YELLOW
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
    df = pd.read_parquet(url)

    # Используем колонки tpep_* для YELLOW taxi
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    # Фильтруем аномальные поездки
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    # Приводим категориальные признаки к строковому типу
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    return df


@task
def create_X(df: pd.DataFrame, dv: DictVectorizer = None) -> tuple:
    """Создает матрицу признаков X и возвращает ее вместе с векторизатором."""
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    categorical = ['PU_DO']
    
    # ВАЖНО: В датасете YELLOW нет колонки 'trip_distance'. 
    # Используем 'passenger_count' или другую числовую колонку, либо убираем.
    # Для соответствия заданию, давайте уберем числовые признаки из этой функции.
    dicts = df[categorical].to_dict(orient='records')

    if dv is None:
        dv = DictVectorizer()
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)
        
    return X, dv


@task
def train_model_and_log_results(X_train, y_train, X_val, y_val, dv: DictVectorizer):
    """Обучает модель XGBoost, логирует все в MLflow."""
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("nyc-taxi-experiment-homework-yellow")

    with mlflow.start_run():
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        num_features = X_train.shape[1]
        print(f"Number of features: {num_features}")
        mlflow.log_param("num_features", num_features)

        best_params = {
            'learning_rate': 0.09585355369315604, 'max_depth': 30,
            'min_child_weight': 1.060597050922164, 'objective': 'reg:linear',
            'reg_alpha': 0.018060244040060163, 'reg_lambda': 0.011658731377413597,
            'seed': 42
        }
        mlflow.log_params(best_params)

        booster = xgb.train(
            params=best_params, dtrain=train, num_boost_round=100, # Увеличим количество раундов для лучшего результата
            evals=[(valid, 'validation')], early_stopping_rounds=10
        )

        y_pred = booster.predict(valid)
        rmse = root_mean_squared_error(y_val, y_pred)
        print(f"Validation RMSE: {rmse}")
        mlflow.log_metric("rmse", rmse)

        models_folder = Path('models')
        models_folder.mkdir(exist_ok=True)
        preprocessor_path = models_folder / "preprocessor.b"
        with open(preprocessor_path, "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact(preprocessor_path, artifact_path="preprocessor")

        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")


@flow
def main_run(train_year: int, train_month: int, val_month_offset: int = 1):
    """Главный пайплайн, который принимает год и месяц в качестве параметров."""
    val_year = train_year
    val_month = train_month + val_month_offset
    
    if val_month > 12:
        val_month = 1
        val_year += 1

    df_train = read_dataframe(year=train_year, month=train_month)
    df_val = read_dataframe(year=val_year, month=val_month)

    X_train, dv = create_X(df_train)
    X_val, _ = create_X(df_val, dv=dv)

    target = 'duration'
    y_train = df_train[target].values
    y_val = df_val[target].values

    print(f"Size of training matrix: {X_train.shape}")
    print(f"Size of validation matrix: {X_val.shape}")

    train_model_and_log_results(X_train, y_train, X_val, y_val, dv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model to predict taxi trip duration using a Prefect flow.')
    parser.add_argument('--year', type=int, default=2023, help='Year of the data to train on (e.g., 2023)')
    parser.add_argument('--month', type=int, default=1, help='Month of the data to train on (e.g., 1 for January)')
    args = parser.parse_args()

    main_run(train_year=args.year, train_month=args.month, val_month_offset=1)