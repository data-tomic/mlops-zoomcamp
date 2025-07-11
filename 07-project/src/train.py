# src/train.py (Final Corrected Version)

import os
import pickle
import json
import click
import mlflow
from hyperopt import STATUS_OK, Trials, fmin, tpe, hp
from hyperopt.pyll import scope
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

# --- MLflow Setup ---
# Set the tracking server URI. This script will communicate with the MLflow server over HTTP.
mlflow.set_tracking_uri("http://127.0.0.1:5000")
# Set the experiment name. MLflow will create it if it does not exist.
mlflow.set_experiment("Neonatal Sepsis Outcome")
# --------------------


def load_pickle(filepath):
    """Loads a pickle file from the specified path."""
    with open(filepath, "rb") as f:
        return pickle.load(f)


def run_train_hpo(data_path: str, num_trials: int, model_name: str):
    """
    Main function to load data, run hyperparameter optimization,
    and log the best model and its artifacts to MLflow.
    """
    # Load the processed training and validation data
    X_train = load_pickle(os.path.join(data_path, "X_train.pkl"))
    y_train = load_pickle(os.path.join(data_path, "y_train.pkl"))
    X_val = load_pickle(os.path.join(data_path, "X_val.pkl"))
    y_val = load_pickle(os.path.join(data_path, "y_val.pkl"))
    print("Training and validation data loaded successfully.")

    # --- Artifact Creation: Column Order ---
    # Get the exact column order from the training data.
    # This is crucial for prediction consistency.
    features = X_train.columns.tolist()
    # Save it to a local file, which we will log as an artifact.
    with open("column_order.json", "w") as f:
        json.dump(features, f)
    # -------------------------------------

    def objective(params):
        """The function that Hyperopt will minimize."""
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            pipeline = Pipeline(
                [("scaler", StandardScaler()), ("rf", RandomForestClassifier(**params))]
            )
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_val)
            f1 = f1_score(y_val, y_pred)
            mlflow.log_metric("f1_score", f1)
        # Hyperopt minimizes, so we return the negative of our metric (F1-score)
        return {"loss": -f1, "status": STATUS_OK}

    # Define the search space for Hyperopt
    search_space = {
        "n_estimators": scope.int(hp.quniform("n_estimators", 10, 100, 1)),
        "max_depth": scope.int(hp.quniform("max_depth", 3, 15, 1)),
        "min_samples_split": scope.int(hp.quniform("min_samples_split", 2, 8, 1)),
        "min_samples_leaf": scope.int(hp.quniform("min_samples_leaf", 1, 4, 1)),
        "random_state": 42,
    }

    # Start a parent run to encapsulate the HPO process
    with mlflow.start_run(run_name="HPO_Parent_Run"):
        mlflow.set_tag("model_type", "RandomForestClassifier")

        # Run the hyperparameter optimization
        fmin(
            fn=objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=num_trials,
            trials=Trials(),
        )

        # Find the best run from all the nested HPO trials
        best_run = mlflow.search_runs(
            order_by=["metrics.f1_score DESC"], max_results=1
        ).iloc[0]
        best_f1 = best_run["metrics.f1_score"]

        # Extract the best parameters
        best_params = {
            key.replace("params.", ""): value
            for key, value in best_run.items()
            if key.startswith("params.")  # noqa E501
        }
        # Ensure params are the correct type (int)
        for key in [
            "n_estimators",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
        ]:
            if key in best_params:
                best_params[key] = int(best_params[key])
        best_params["random_state"] = 42

        print(f"HPO finished. Best F1-Score: {best_f1:.4f}")
        mlflow.log_metric("best_f1_score", best_f1)
        mlflow.log_params(best_params)

        # Train the final model pipeline on the full training data with the best parameters
        final_pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("rf", RandomForestClassifier(**best_params)),
            ]
        )
        final_pipeline.fit(X_train, y_train)

        # --- Log Artifacts ---
        # Log the column order file to the root of the run's artifacts
        mlflow.log_artifact("column_order.json")

        # Log the model itself, using a standard artifact path
        mlflow.sklearn.log_model(
            sk_model=final_pipeline,
            artifact_path="model",  # Standard path for the model files
            registered_model_name=model_name,
        )
        print(f"Best pipeline and column order registered in MLflow as '{model_name}'")


@click.command()
@click.option(
    "--data-path", default="data/processed", help="Path to the processed data folder."
)
@click.option("--num-trials", default=15, help="Number of trials for HPO.")
@click.option(
    "--model-name",
    default="sepsis-outcome-classifier",
    help="Name for the registered model in MLflow.",
)  # noqa E501
def train_command(data_path: str, num_trials: int, model_name: str):
    """Command-line entry point to run HPO and training."""
    run_train_hpo(data_path, num_trials, model_name)


if __name__ == "__main__":
    train_command()
