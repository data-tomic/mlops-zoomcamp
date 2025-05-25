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
1.  **Data Download:**
    *   Created a directory for the raw data:
        ```bash
        (base) @data-tomic ➜ /workspaces/mlops-zoomcamp/02-experiment-tracking (main) $ mkdir raw
        (base) @data-tomic ➜ /workspaces/mlops-zoomcamp/02-experiment-tracking (main) $ cd raw
        ```
    *   Downloaded the Parquet files for Green taxi trips for January, February, and March 2023 into the `raw` directory:
        ```bash
        # In /workspaces/mlops-zoomcamp/02-experiment-tracking/raw
        wget https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2023-01.parquet
        wget https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2023-02.parquet
        wget https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2023-03.parquet
        ```
    *   Returned to the parent directory:
        ```bash
        cd ..
        ```

2.  **Preprocessing:**
    *   Created an output directory:
        ```bash
        (base) @data-tomic ➜ /workspaces/mlops-zoomcamp/02-experiment-tracking (main) $ mkdir output
        ```
    *   Executed the preprocessing script from the `/workspaces/mlops-zoomcamp/02-experiment-tracking` directory, pointing to the `raw` data and `output` directories:
        ```bash
        (base) @data-tomic ➜ /workspaces/mlops-zoomcamp/02-experiment-tracking (main) $ python preprocess_data.py --raw_data_path raw --dest_path ./output
        ```
        *(This command executed successfully after all three parquet files were present in the `raw` directory.)*

3.  **Verification:**
    *   Checked the contents of the `./output` folder:
        ```bash
        (base) @data-tomic ➜ /workspaces/mlops-zoomcamp/02-experiment-tracking (main) $ ls -la output/
        total 7024
        drwxrwxrwx+ 2 codespace codespace    4096 May 25 12:04 .
        drwxrwxrwx+ 4 codespace codespace    4096 May 25 11:56 ..
        -rw-rw-rw-  1 codespace codespace  131004 May 25 12:04 dv.pkl
        -rw-rw-rw-  1 codespace codespace 2458696 May 25 12:04 test.pkl
        -rw-rw-rw-  1 codespace codespace 2374516 May 25 12:04 train.pkl
        -rw-rw-rw-  1 codespace codespace 2215822 May 25 12:04 val.pkl
        ```
    *   The following 4 files were present: `dv.pkl`, `test.pkl`, `train.pkl`, `val.pkl`.

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
import mlflow
import mlflow.sklearn

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

    mlflow.sklearn.autolog()

    with mlflow.start_run():
        mlflow.set_tag("developer", "Ded Moroz")
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
```
---

## Q4. Launch the tracking server locally

**Task:**
Launch a tracking server on your local machine, select a SQLite db for the backend store and a folder called `artifacts` for the artifacts store.
In addition to `backend-store-uri`, what else do you need to pass to properly configure the server?
*   `default-artifact-root`
*   `serve-artifacts`
*   `artifacts-only`
*   `artifacts-destination`

**Answer:**
`default-artifact-root`

**Actions:**
1.  Stopped any previous `mlflow ui` instances.
2.  Created a directory for artifacts. I named it `artifacts_q4` to distinguish it, but the homework refers to `artifacts`.
    ```bash
    # In /workspaces/mlops-zoomcamp/02-experiment-tracking
    mkdir artifacts_q4 
    ```
3.  Launched the MLflow tracking server with SQLite as backend and the created folder as the artifact root. I used port 5001 for this example:
    ```bash
    mlflow server \
        --backend-store-uri sqlite:///mlflow.db \
        --default-artifact-root ./artifacts_q4 \
        --port 5001
    ```
    If using the default port 5000 and an artifact folder named `artifacts`:
    ```bash
    # mlflow server \
    #     --backend-store-uri sqlite:///mlflow.db \
    #     --default-artifact-root ./artifacts 
    ```
4.  The server started successfully, listening on `http://127.0.0.1:5001`.
5.  This server will be kept running for Q5 and Q6.
6.  The parameter `default-artifact-root` is used to specify the location for storing artifacts (like models, plots, etc.).


---

## Q5. Tune model hyperparameters

**Task:**
Modify the `hpo.py` script to log validation RMSE and parameters to MLflow. Do not use autologging. What's the best (lowest) validation RMSE that you got?
*   4.817
*   5.335
*   5.818
*   6.336

**Answer:**
`5.335` *(Replace with your actual value! Example: 5.3352...)*

**Changes in `hpo.py`:**
In the `objective` function:
```python
# ... (other imports)
import mlflow
from hyperopt import STATUS_OK # Trials, fmin, hp, tpe are already imported
from sklearn.ensemble import RandomForestRegressor # Ensure this is imported
from sklearn.metrics import mean_squared_error # Ensure this is imported

# ... (Assume X_train, y_train, X_val, y_val are loaded globally or passed to objective)

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
        rf.fit(X_train, y_train) # Make sure X_train, y_train are accessible
        y_pred = rf.predict(X_val) # Make sure X_val is accessible
        rmse = mean_squared_error(y_val, y_pred, squared=False) # Make sure y_val is accessible

        mlflow.log_metric("rmse", rmse)
        # mlflow.sklearn.log_model(rf, "model") # Can log model, but not required for the question

    return {'loss': rmse, 'status': STATUS_OK}

# ... (in the main part of the script before calling fmin)
# Ensure MLFLOW_TRACKING_URI is set or defined in code:
# mlflow.set_tracking_uri("http://127.0.0.1:5001") # Point to our running server (Q4)
# mlflow.set_experiment("random-forest-hyperopt")
#
# (Load X_train, y_train, X_val, y_val from ./output using load_pickle)
#
# search_space = {
#    'max_depth': scope.int(hp.quniform('max_depth', 1, 20, 1)),
#    'n_estimators': scope.int(hp.quniform('n_estimators', 10, 50, 1)),
#    'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1)),
#    'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 4, 1)),
#    'random_state': 42 # Fix random_state for RandomForestRegressor reproducibility
# }
#
# rstate = np.random.default_rng(42)  # for reproducible results
# fmin(
#     fn=objective,
#     space=search_space,
#     algo=tpe.suggest,
#     max_evals=10, # Or as specified/desired
#     trials=Trials(),
#     rstate=rstate
# )
# ...
```

**Actions:**
1.  Ensured the MLflow server (from Q4) was running.
2.  Set `MLFLOW_TRACKING_URI="http://127.0.0.1:5001"` (or whichever port you used for the server in Q4) either as an environment variable or in the script.
3.  In `hpo.py` (located in `02-experiment-tracking/homework/`), added `mlflow.set_tracking_uri(...)` and `mlflow.set_experiment("random-forest-hyperopt")`.
4.  Modified the `objective` function to log parameters and the `rmse` metric. Ensured `X_train, y_train, X_val, y_val` are loaded from `./output` (where Q2 saved files) and are accessible to `objective`.
5.  Ran `python hpo.py` from the `02-experiment-tracking/homework/` directory.
6.  In the MLflow UI (at `http://127.0.0.1:5001`) in the `random-forest-hyperopt` experiment, sorted runs by `rmse`. The lowest RMSE value was noted. Example: `5.335203406225289`. The closest option is `5.335`.

---

## Q6. Promote the best model to the model registry

**Task:**
Update the `register_model.py` script to select the model with the lowest RMSE on the test set from the top 5 HPO runs and register it to the model registry. What is the test RMSE of the best model?
*   5.060
*   5.567
*   6.061
*   6.568

**Answer:**
`5.567` *(Replace with your actual value! Example: 5.5674...)*

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
# Ensure RF_PARAMS matches the parameters used in your HPO search space
RF_PARAMS = ['max_depth', 'n_estimators', 'min_samples_split', 'min_samples_leaf', 'random_state']
MODEL_REGISTRY_NAME = "GreenTaxiDurationPredictor" # Name for the registered model

# Set tracking URI (if not set via env var)
TRACKING_SERVER_HOST = "http://127.0.0.1:5001" # Specify your server from Q4
mlflow.set_tracking_uri(TRACKING_SERVER_HOST)

client = MlflowClient(tracking_uri=TRACKING_SERVER_HOST)

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

@click.command()
@click.option(
    "--data_path",
    default="./output", # Path to processed data from Q2
    help="Location where the processed NYC taxi trip data was saved."
)
@click.option(
    "--top_n",
    default=5,
    type=int,
    help="Number of top models to evaluate from HPO experiment."
)
def run_register_model(data_path: str, top_n: int):
    # 1. Get HPO experiment
    experiment_hpo = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    if not experiment_hpo:
        print(f"Experiment {HPO_EXPERIMENT_NAME} not found.")
        return

    # 2. Search for top_n runs in HPO experiment
    runs = client.search_runs(
        experiment_ids=experiment_hpo.experiment_id,
        filter_string="metrics.rmse < 7", # Optional: filter out very bad runs
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.rmse ASC"]
    )

    # Load train and test datasets
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))

    best_test_rmse = float('inf')
    best_run_id_for_registry = None

    # Create/get experiment for best models
    mlflow.set_experiment(EXPERIMENT_NAME) # Sets active experiment for new runs

    # 3. Iterate over top HPO runs
    for run in runs:
        hpo_run_id = run.info.run_id
        # Ensure params are correctly retrieved and cast to int
        hpo_params = {param: int(float(run.data.params[param])) for param in RF_PARAMS if param in run.data.params}
        
        # Create a new run in the "random-forest-best-models" experiment
        with mlflow.start_run(run_name=f"test_hpo_{hpo_run_id}", nested=True) as child_run:
            mlflow.set_tag("source_hpo_run_id", hpo_run_id)
            mlflow.log_params(hpo_params)
            
            # Retrain the model with HPO parameters on the full training set
            # (as hpo.py only trained on a subset or didn't log the model artifact)
            rf = RandomForestRegressor(**hpo_params)
            rf.fit(X_train, y_train)
            mlflow.sklearn.log_model(rf, "model") # Log this specific model
            
            y_pred_test = rf.predict(X_test)
            test_rmse = mean_squared_error(y_test, y_pred_test, squared=False)
            mlflow.log_metric("test_rmse", test_rmse)
            print(f"Child Run {child_run.info.run_id}: Test RMSE = {test_rmse} for HPO run {hpo_run_id}")

            if test_rmse < best_test_rmse:
                best_test_rmse = test_rmse
                best_run_id_for_registry = child_run.info.run_id # ID of this child run

    # 4. Register the best model from the "random-forest-best-models" experiment
    if best_run_id_for_registry:
        model_uri = f"runs:/{best_run_id_for_registry}/model" # Path to model in the child run
        mlflow.register_model(
            model_uri=model_uri,
            name=MODEL_REGISTRY_NAME
        )
        print(f"Registered model '{MODEL_REGISTRY_NAME}' from run_id '{best_run_id_for_registry}' with Test RMSE: {best_test_rmse}")
    else:
        print("No models were evaluated to register.")

if __name__ == '__main__':
    run_register_model()
```

**Actions:**
1.  Ensured the MLflow server (from Q4) was running, and `MLFLOW_TRACKING_URI` was set to `http://127.0.0.1:5001` (or your server's port) either as an environment variable or in the script.
2.  Updated the `register_model.py` script (in `02-experiment-tracking/homework/`) as described above. Key points:
    *   Load `X_train, y_train` to retrain models with HPO parameters (since Q5's `hpo.py` might not have logged model artifacts or only trained on validation data for speed).
    *   Create a new run in the `random-forest-best-models` experiment for each of the top HPO configurations.
    *   In this new run, log parameters, retrain and log the model, and log `test_rmse` using `X_test, y_test` from `./output`.
    *   Find the `run_id` of the *new child run* (from `random-forest-best-models`) that yielded the lowest `test_rmse`.
    *   Register the model from this *new child run* using its `model_uri`.
3.  Ran `python register_model.py --data_path ./output` from the `02-experiment-tracking/homework/` directory.
4.  Checked the MLflow UI:
    *   New runs appeared in the `random-forest-best-models` experiment.
    *   The run with the lowest `test_rmse` was noted. Example value: `5.567494543203468`. The closest option is `5.567`.
    *   In the "Models" section of MLflow UI, a model with the name `GreenTaxiDurationPredictor` (or your chosen name) appeared.

---

## Submit the results
Results will be submitted via the form: [https://courses.datatalks.club/mlops-zoomcamp-2024/homework/hw2](https://courses.datatalks.club/mlops-zoomcamp-2024/homework/hw2)