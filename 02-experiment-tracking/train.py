import os
import pickle
import click
import mlflow # <--- Добавлено
import mlflow.sklearn # <--- Добавлено

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):
    
    mlflow.sklearn.autolog() # <--- Включаем автологирование для sklearn

    with mlflow.start_run(): # <--- Начинаем MLflow run

        # Опционально: можно добавить теги к запуску
        mlflow.set_tag("developer", "Ded Moroz") # Замени your_name
        mlflow.set_tag("model_type", "RandomForestRegressor")

        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = mean_squared_error(y_val, y_pred, squared=False)
        print(f"RMSE: {rmse}") 

if __name__ == '__main__':
    run_train()
