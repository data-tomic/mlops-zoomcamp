# MLOps Project: End-to-End Genomics Pipeline for Cancer Classification

This project is part of the [DataTalksClub MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp). The primary goal is to build a complete, end-to-end MLOps pipeline for a real-world bioinformatics problem: classifying tumor samples from gene expression data.

**Project Goal:** To automate the process of data ingestion, processing, model training, deployment, and monitoring for classifying tumor vs. normal tissue samples from the TCGA-BRCA (Breast Cancer) dataset.

## Project Status

**`In Progress`**

## Core Technologies

*   **Data Source**: [Genomic Data Commons (GDC) Portal](https://portal.gdc.cancer.gov/) - TCGA-BRCA Project
*   **Experiment Tracking**: [MLflow](https://mlflow.org/)
*   **Workflow Orchestration**: [Prefect](https://www.prefect.io/) `(Next Step)`
*   **Model Deployment**: [FastAPI](https://fastapi.tiangolo.com/), [Docker](https://www.docker.com/) `(Next Step)`
*   **Monitoring**: [Evidently AI](https://www.evidentlyai.com/) `(Next Step)`
*   **CI/CD**: [GitHub Actions](https://github.com/features/actions) `(Next Step)`
*   **Core Libraries**: Scikit-learn, Pandas

## Pipeline Architecture

The end-to-end pipeline is designed as follows:

`Data Acquisition (GDC API)` -> `Data Processing (Python/Pandas)` -> `Model Training & Experimenting (MLflow)` -> `Workflow Orchestration (Prefect)` -> `Model Deployment (FastAPI + Docker)` -> `Monitoring (Evidently AI)`

---

## Progress So Far (Completed Steps)

### 1. Project Setup
*   **Structured Repository:** Established a clean project structure within `07-project/` to separate concerns (`src`, `data`, `notebooks`, `tests`, etc.).
*   **Environment Management:** Created a `requirements.txt` to manage all Python dependencies and a `Makefile` for convenience commands.

### 2. Data Acquisition & Ingestion
*   **Data Sourcing:** Implemented a Jupyter Notebook (`notebooks/1-eda-and-data-download.ipynb`) to programmatically query the GDC API.
*   **Data Selection:** Pivoted from the controlled-access `TARGET-ALL` dataset to the publicly available **`TCGA-BRCA`** project. We are using `Gene Expression Quantification` data (`STAR-Counts` workflow).
*   **Data Download:** Successfully downloaded a balanced subset of 20 samples (10 Tumor, 10 Normal) using a `curl`-based script after extensive debugging of the official `gdc-client`. The download process is driven by a `manifest.txt` file, making it reproducible.

### 3. Data Processing
*   **Data Consolidation:** Created a robust Python script (`src/data_processing.py`) that reads the 20 raw, separate data files.
*   **Data Cleaning & Aggregation:** The script correctly handles the specific format of `STAR-Counts` files by:
    1.  Skipping the first 4 header lines.
    2.  Parsing the multi-column format.
    3.  Cleaning gene IDs to remove version numbers (e.g., `ENSG000...1.5` -> `ENSG000...1`).
    4.  Aggregating duplicate gene entries by summing their counts.
*   **Final Matrix Creation:** The script generates a final, analysis-ready matrix of `16377 genes x 20 samples` and saves it to `data/processed/gene_expression_matrix.csv`.

### 4. Model Training & Experiment Tracking
*   **MLflow Setup:** Successfully configured MLflow to track experiments locally in a `mlruns` directory.
*   **Baseline Model:** Implemented a training script (`src/train.py`) that trains a `LogisticRegression` model as a baseline.
*   **Iteration & Comparison:** Ran a second experiment with a `RandomForestClassifier` to demonstrate iterative model improvement.
*   **Comprehensive Logging:** For each run, we are logging:
    *   **Parameters:** Model type and its hyperparameters.
    *   **Metrics:** `accuracy` and `f1_score`.
    *   **Artifacts:** The trained Scikit-learn model itself.
*   **UI Visualization:** Successfully launched and used the `mlflow ui` to compare the two experimental runs.

---

## Next Steps (To-Do)

### 1. Model Improvement & Hyperparameter Tuning
*   [ ] **Data Preprocessing:** Implement data normalization/scaling (e.g., `TMM normalization` or `log-transformation`), which is crucial for RNA-Seq data.
*   [ ] **Feature Selection:** Add a step to select the most relevant genes (features) to reduce dimensionality and improve performance (e.g., using variance thresholding or statistical tests).
*   [ ] **Hyperparameter Tuning:** Use a library like `Hyperopt` integrated with MLflow to automatically find the best hyperparameters for our model.

### 2. Workflow Orchestration
*   [ ] **Create a Prefect Flow:** Convert the data processing and training scripts into a single, automated Prefect workflow (`@flow` and `@task` decorators). The flow will be: `process_data` -> `train_model`.

### 3. Model Deployment
*   [ ] **Develop a Prediction Service:** Create a `FastAPI` application that loads the best model from the MLflow Model Registry and exposes a prediction endpoint.
*   [ ] **Containerize the Service:** Write a `Dockerfile` to package the FastAPI application, ensuring all dependencies are included.

### 4. CI/CD Automation
*   [ ] **Set up GitHub Actions:** Create `.github/workflows/ci-cd.yaml`.
*   [ ] **Continuous Integration (CI):** Implement a workflow that runs on every push to automatically lint (`flake8`) and test (`pytest`) the code.
*   [ ] **Continuous Deployment (CD):** Implement a workflow that, on a Git tag (e.g., `v1.0`), triggers the Prefect flow to retrain the model and, if successful, builds and pushes the new Docker image to a container registry.

### 5. Monitoring
*   [ ] **Implement Evidently AI:** Create a script to generate monitoring reports.
*   [ ] **Data Drift:** Compare incoming data distributions against the training data.
*   [ ] **Model Performance:** Track model metrics on new data (if labels are available).
*   [ ] **Integrate into Pipeline:** Add a monitoring step to the Prefect flow or CI/CD pipeline.

---

Excellent point. The `curl` command is a critical, non-obvious part of our current data acquisition process. I will integrate it directly into the `README.md`.

Here is the updated section `How to Run the Project (Current State)`. I've replaced the generic instruction with the exact `curl` loop we developed.

---

## How to Run the Project (Current State)

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/data-tomic/mlops-zoomcamp.git
    cd mlops-zoomcamp
    ```

2.  **Set up the environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r 07-project/requirements.txt
    ```

3.  **Run the data pipeline:**
    This process is divided into two parts: generating the download manifest and then downloading the data.

    **A. Generate the Download Manifest:**
    *   Run the Jupyter Notebook to create the list of files to download. This will create `manifest.txt` and `metadata.csv` in the `07-project/data/` directory.
    *   You can execute the notebook manually or use a command-line tool like `nbconvert`.

    **B. Download the Raw Data:**
    *   Navigate to the target directory for raw data:
    ```bash
    mkdir -p 07-project/data/raw
    cd 07-project/data/raw
    ```
    *   Run the following `curl` loop in your terminal to download the files listed in `../manifest.txt`. This method is used as a robust alternative to the official `gdc-client`.
    ```bash
    while read -r file_id; do
      # Skip potential empty lines
      if [ -z "$file_id" ]; then
        continue
      fi
      
      echo ">>> Downloading file with ID: ${file_id}"
      
      # Use curl to download the data file
      # -O: Saves the file with its original name from the server (which is the file_id)
      # -L: Follows any HTTP redirects, which is crucial for the GDC API
      curl -O -L "https://api.gdc.cancer.gov/data/${file_id}"
      
      echo "--- Done: ${file_id}"
      echo "" # Newline for readability
      
    done < ../manifest.txt

    echo "===== DATA DOWNLOAD COMPLETE ====="
    ```
    *   Return to the project root directory:
    ```bash
    cd ../../../
    ```

    **C. Process the Raw Data:**
    *   Run the processing script to combine the downloaded files into a single matrix.
    ```bash
    python 07-project/src/data_processing.py
    ```

4.  **Run a training experiment:**
    ```bash
    python 07-project/src/train.py
    ```

5.  **View the results:**
    ```bash
    # Make sure you are in the root of the repository
    mlflow ui --backend-store-uri file:./mlruns
    ```
    Then open `http://127.0.0.1:5000` in your browser.

