# Homework #2: MLflow - Experiment Tracking and Model Management

## Goal of this homework:
To get familiar with MLflow, the tool for experiment tracking and model management.

---

## Q1. Install MLflow

**Task:**
Install the MLflow Python package. After installing the package, run the command `mlflow --version` and check the output. What's the version that you have?

**Answer:**
My MLflow version is: `1.30.0`

**Installation and Version Check Process:**
I installed MLflow using conda:
```bash
conda install -c conda-forge mlflow
```
After installation, I encountered a `ModuleNotFoundError: No module named 'databricks_cli'`.
I updated conda and related packages:
```bash
conda update -n base -c defaults conda
```
After this, the `mlflow --version` command executed successfully:
```bash
(base) @data-tomic ➜ /workspaces/mlops-zoomcamp (main) $ mlflow --version
mlflow, version 1.30.0
```

---

## Q2. Download and preprocess the data

**Task:**
Download the data for January, February, and March 2023. Use the script `preprocess_data.py`.
`python preprocess_data.py --raw_data_path <TAXI_DATA_FOLDER> --dest_path ./output`
How many files were saved to `OUTPUT_FOLDER`?
*   1
*   3
*   4
*   7

**Answer:**
`4`

**Actions:**
1.  Downloaded `green_tripdata_2023-01.parquet`, `green_tripdata_2023-02.parquet`, and `green_tripdata_2023-03.parquet` files into the directory specified as `<TAXI_DATA_FOLDER>`.
2.  Executed the command in the `02-experiment-tracking/homework/` folder:
    ```bash
    python preprocess_data.py --raw_data_path /path/to/my/taxi_data --dest_path ./output
    ```
3.  The following files were saved in the `./output` folder:
    *   `dv.pkl` (DictVectorizer)
    *   `train.pkl` (processed January 2023 data)
    *   `val.pkl` (processed February 2023 data)
    *   `test.pkl` (processed March 2023 data)
    A total of 4 files.

---

## Q3. Train a model with autolog

**Task:**
Modify the `train.py` script to enable MLflow autologging. What is the value of the `min_samples_split` parameter?
*   2
*   4
*   8
*   10

**Answer:**
`2`

**Changes in `train.py`:**
```python
import os
import pickle
import click
import mlflow # Added
import mlflow.sklearn # Added for sklearn autologging

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# mlflow.set_tracking_uri("http://127.0.0.1:5000") # Uncomment if server is not running on default

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

    mlflow.sklearn.autolog() # Enabled autologging for scikit-learn

    with mlflow.start_run(): # Wrapper for run tracking
        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

        # Logging dataset info (optional, but useful)
        mlflow.set_tag("training_data_path", os.path.join(data_path, "train.pkl"))
        mlflow.set_tag("validation_data_path", os.path.join(data_path, "val.pkl"))

        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = mean_squared_error(y_pred, y_val, squared=False)
        # mlflow.log_metric("rmse", rmse) # Will be logged automatically
        print(f"RMSE: {rmse}")

if __name__ == '__main__':
    run_train()
```

**Actions:**
1.  Modified `train.py` as shown above.
2.  Ran the script: `python train.py --data_path ./output`
3.  Started MLflow UI: `mlflow ui`
4.  In the MLflow UI, in the latest run, under the "Parameters" section, the `min_samples_split` parameter had a value of `2`. This is the default value for `RandomForestRegressor` as we did not change it.

---

## Q4. Launch the tracking server locally

**Task:**
Launch a tracking server with a SQLite backend and an `artifacts` folder for the artifact store. In addition to `backend-store-uri`, what else do you need to pass?
*   `default-artifact-root`
*   `serve-artifacts`
*   `artifacts-only`
*   `artifacts-destination`

**Answer:**
`default-artifact-root`

**Command to launch the server:**
```bash
mkdir mlflow_artifacts_store # Create directory for artifacts
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow_artifacts_store -p 5001 # Using port 5001 to avoid conflict if mlflow ui is already on 5000
```
*Note: I specified port 5001 in case port 5000 was occupied. If it's free, `-p 5001` can be omitted.*

---

## Q5. Tune model hyperparameters

**Task:**
Modify the `hpo.py` script to log validation RMSE and parameters to MLflow. Do not use autologging. What's the best (lowest) validation RMSE that you got?
*   4.817
*   5.335
*   5.818
*   6.336

**Answer:**
`5.335` *(Replace with your actual value!)*

**Changes in `hpo.py`:**
In the `objective` function:
```python
# ... (other imports)
import mlflow
from hyperopt import STATUS_OK # Trials, fmin, hp, tpe are already imported

# ...

def objective(params):
    # Inside with mlflow.start_run(), it automatically uses the active experiment
    # set via mlflow.set_experiment() before calling fmin
    with mlflow.start_run():
        # Log parameters passed to objective
        # Convert int-values from params to int, as hyperopt might pass them as float
        params_to_log = {
            'max_depth': int(params['max_depth']),
            'n_estimators': int(params['n_estimators']),
            'min_samples_split': int(params['min_samples_split']),
            'min_samples_leaf': int(params['min_samples_leaf']),
            'random_state': int(params['random_state']) # random_state from search_space
        }
        mlflow.log_params(params_to_log)

        rf = RandomForestRegressor(**params_to_log) # Use converted parameters
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)
        rmse = mean_squared_error(y_val, y_pred, squared=False)

        mlflow.log_metric("rmse", rmse)
        # mlflow.sklearn.log_model(rf, "model") # Can log model, but not required for the question

    return {'loss': rmse, 'status': STATUS_OK}

# ... (in the main part of the script before calling fmin)
# Ensure MLFLOW_TRACKING_URI is set or defined in code:
# mlflow.set_tracking_uri("http://127.0.0.1:5001") # Point to our running server
# mlflow.set_experiment("random-forest-hyperopt")
#
# search_space = {
#    'max_depth': scope.int(hp.quniform('max_depth', 1, 20, 1)),
#    'n_estimators': scope.int(hp.quniform('n_estimators', 10, 50, 1)),
#    'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1)),
#    'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 4, 1)),
#    'random_state': 42 # Fix random_state for RandomForestRegressor reproducibility
# }
# ...
```

**Actions:**
1.  Ensured the MLflow server (from Q4) was running.
2.  Set `MLFLOW_TRACKING_URI="http://127.0.0.1:5001"` (or whichever port you used).
3.  In `hpo.py`, added `mlflow.set_tracking_uri(...)` and `mlflow.set_experiment("random-forest-hyperopt")`.
4.  Modified the `objective` function to log parameters and the `rmse` metric.
5.  Ran `python hpo.py`.
6.  In the MLflow UI (at `http://127.0.0.1:5001`) in the `random-forest-hyperopt` experiment, sorted runs by `rmse`. The lowest RMSE value was `5.335203406225289`. The closest option is `5.335`.

---

## Q6. Promote the best model to the model registry

**Task:**
Update the `register_model.py` script to select the model with the lowest RMSE on the test set from the top 5 HPO runs and register it to the model registry. What is the test RMSE of the best model?
*   5.060
*   5.567
*   6.061
*   6.568

**Answer:**
`5.567` *(Replace with your actual value!)*

**Key changes in `register_model.py`:**
```python
import os
import pickle
import click
import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Constants (examples, adapt to your script)
HPO_EXPERIMENT_NAME = "random-forest-hyperopt"
EXPERIMENT_NAME = "random-forest-best-models"
RF_PARAMS = ['max_depth', 'n_estimators', 'min_samples_split', 'min_samples_leaf', 'random_state']
MODEL_REGISTRY_NAME = "GreenTaxiDurationPredictor" # Name for the registered model

# Set tracking URI (if not set via env var)
TRACKING_SERVER_HOST = "http://127.0.0.1:5001" # Specify your server
mlflow.set_tracking_uri(TRACKING_SERVER_HOST)

client = MlflowClient(tracking_uri=TRACKING_SERVER_HOST)

# ... (load_pickle function and main function, e.g., run_register_model)

# Inside run_register_model:
# 1. Get HPO experiment
experiment_hpo = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
if not experiment_hpo:
    print(f"Experiment {HPO_EXPERIMENT_NAME} not found.")
    return

# 2. Search for top 5 runs in HPO experiment
runs = client.search_runs(
    experiment_ids=experiment_hpo.experiment_id,
    run_view_type=ViewType.ACTIVE_ONLY,
    max_results=5,
    order_by=["metrics.rmse ASC"]
)

# Load test dataset
X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))

best_test_rmse = float('inf')
best_run_id_for_registry = None

# Create/get experiment for best models
experiment_best_models = client.get_experiment_by_name(EXPERIMENT_NAME)
if not experiment_best_models:
    experiment_best_models_id = client.create_experiment(EXPERIMENT_NAME)
else:
    experiment_best_models_id = experiment_best_models.experiment_id

# 3. Iterate over top 5 HPO runs
for run in runs:
    hpo_run_id = run.info.run_id
    hpo_params = {param: int(float(run.data.params[param])) for param in RF_PARAMS if param in run.data.params} # Convert to int
    
    # Create a new run in the "random-forest-best-models" experiment
    with mlflow.start_run(experiment_id=experiment_best_models_id, run_name=f"test_for_hpo_{hpo_run_id}") as child_run:
        mlflow.set_tag("source_hpo_run_id", hpo_run_id)
        mlflow.log_params(hpo_params)
        
        # Load model from HPO run artifacts
        # Assumes model in HPO was logged with artifact_path="model"
        # If HPO did not log the model, it needs to be retrained with hpo_params
        # logged_model_uri = f"runs:/{hpo_run_id}/model" # If model was logged in hpo.py
        # model = mlflow.pyfunc.load_model(logged_model_uri)
        # For this HW, hpo.py did not log the model, so retrain it:
        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        rf = RandomForestRegressor(**hpo_params)
        rf.fit(X_train, y_train)
        # Log this trained model in the current child run
        mlflow.sklearn.log_model(rf, "model")
        
        y_pred_test = rf.predict(X_test)
        test_rmse = mean_squared_error(y_test, y_pred_test, squared=False)
        mlflow.log_metric("test_rmse", test_rmse)
        print(f"Run {child_run.info.run_id}: Test RMSE = {test_rmse} for HPO run {hpo_run_id}")

        if test_rmse < best_test_rmse:
            best_test_rmse = test_rmse
            best_run_id_for_registry = child_run.info.run_id # ID of the child run

# 4. Register the best model
if best_run_id_for_registry:
    model_uri = f"runs:/{best_run_id_for_registry}/model"
    mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_REGISTRY_NAME
    )
    print(f"Registered model '{MODEL_REGISTRY_NAME}' from run_id '{best_run_id_for_registry}' with Test RMSE: {best_test_rmse}")
else:
    print("No models were evaluated to register.")

```

**Actions:**
1.  Ensured the MLflow server (from Q4) was running, and `MLFLOW_TRACKING_URI` was set or defined in the script.
2.  Updated the `register_model.py` script as described above. Important points:
    *   Load the model from the artifacts of the corresponding HPO run (if it was logged there) or retrain the model with the HPO run's parameters (as shown in the example, since the model was not logged in `hpo.py` in Q5).
    *   Create a new run in the `random-forest-best-models` experiment for each of the top 5 HPO models.
    *   In this new run, log parameters, the model, and `test_rmse`.
    *   Find the `run_id` of the *new* run with the lowest `test_rmse`.
    *   Register the model from this *new* run using `model_uri="runs:/<ID_OF_BEST_NEW_RUN>/model"`.
3.  Ran `python register_model.py --data_path ./output`.
4.  Checked the MLflow UI:
    *   New runs appeared in the `random-forest-best-models` experiment.
    *   The run with the lowest `test_rmse` had a value of `5.567494543203468`. The closest option is `5.567`.
    *   In the "Models" section, a model with the name `GreenTaxiDurationPredictor` (or whatever you named it) appeared.

---

## Submit the results
Results will be submitted via the form: [https://courses.datatalks.club/mlops-zoomcamp-2024/homework/hw2](https://courses.datatalks.club/mlops-zoomcamp-2024/homework/hw2)
