#!/usr/bin/env python
# coding: utf-8

import pickle
import argparse
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression  # <-- Убедитесь, что XGBoost удален
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error

import mlflow
from prefect import task, flow


@task
def read_dataframe(year: int, month: int) -> pd.DataFrame:
    """Reads data for YELLOW taxi and performs basic preprocessing."""
    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet"  # noqa: E501
    df = pd.read_parquet(url)

    print(f"Loaded {len(df)} records from the file.")

    df["duration"] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    # Фильтруем аномальные поездки
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    # Приводим категориальные признаки к строковому типу
    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)

    return df


@task
def create_X(df: pd.DataFrame, dv: DictVectorizer = None) -> tuple:
    """Creates the feature matrix X using separate pickup and dropoff locations."""
    categorical = ["PULocationID", "DOLocationID"]
    # Для этого задания числовые признаки не используются
    dicts = df[categorical].to_dict(orient="records")

    if dv is None:
        dv = DictVectorizer()
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, dv


@task
def train_linear_regression_model(X_train, y_train, X_val, y_val, dv: DictVectorizer):
    """Trains a Linear Regression model and logs results to MLflow."""
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("nyc-taxi-homework-linear-regression")

    with mlflow.start_run():
        # Обучаем модель
        lr = LinearRegression()
        lr.fit(X_train, y_train)

        # Вывод intercept для ответа на вопрос домашнего задания
        intercept = lr.intercept_
        print(f"Intercept of the trained model: {intercept}")

        # Оцениваем модель
        y_pred = lr.predict(X_val)
        rmse = root_mean_squared_error(y_val, y_pred)

        # Логируем все в MLflow
        mlflow.log_param("model_class", "LinearRegression")
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("intercept", intercept)

        # Сохраняем и логируем препроцессор (DictVectorizer)
        models_folder = Path("models")
        models_folder.mkdir(exist_ok=True)
        preprocessor_path = models_folder / "preprocessor.b"
        with open(preprocessor_path, "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact(preprocessor_path, artifact_path="preprocessor")

        # Логируем саму scikit-learn модель
        mlflow.sklearn.log_model(lr, artifact_path="models_mlflow")


@flow
def main_run(train_year: int, train_month: int, val_month_offset: int = 1):
    """The main pipeline that now trains a Linear Regression model."""
    val_year = train_year
    val_month = train_month + val_month_offset

    if val_month > 12:
        val_month = 1
        val_year += 1

    df_train = read_dataframe(year=train_year, month=train_month)
    df_val = read_dataframe(year=val_year, month=val_month)

    print(f"Records for training (after filtering): {len(df_train)}")
    print(f"Records for validation (after filtering): {len(df_val)}")

    X_train, dv = create_X(df_train)
    X_val, _ = create_X(df_val, dv=dv)

    target = "duration"
    y_train = df_train[target].values
    y_val = df_val[target].values

    print(f"The shape of the training matrix is: {X_train.shape}")

    # Вызываем нашу задачу для обучения
    train_linear_regression_model(X_train, y_train, X_val, y_val, dv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model to predict taxi trip duration."
    )
    parser.add_argument(
        "--year", type=int, default=2023, help="Year of the data to train on"
    )
    parser.add_argument(
        "--month", type=int, default=1, help="Month of the data to train on"
    )
    args = parser.parse_args()

    main_run(train_year=args.year, train_month=args.month, val_month_offset=1)
