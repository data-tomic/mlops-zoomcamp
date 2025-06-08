# MLOps Orchestration: From Script to Production Pipeline

This project demonstrates the process of converting a standalone machine learning model training script into a robust, automated, and observable MLOps pipeline using modern orchestration and experiment tracking tools.

The core task is to train a model that predicts the duration of taxi trips in New York City. We transform the initial Python script into a Prefect flow, track experiments with MLflow, and package the entire environment for reproducible execution in GitHub Codespaces.

### Tech Stack
*   **Workflow Orchestration:** [Prefect](https://www.prefect.io/)
*   **Experiment Tracking:** [MLflow](https://mlflow.org/)
*   **Containerization:** [Docker](https://www.docker.com/) & [Docker Compose](https://docs.docker.com/compose/)
*   **Development Environment:** [GitHub Codespaces](https://github.com/features/codespaces)

---

## 1. Orchestrator Selection: Why Prefect?

Choosing the right workflow orchestrator is a critical first step. For this project, **Prefect** was selected for its simplicity, flexibility, and "Python-native" approach, which makes it ideal for data scientists and ML engineers transitioning from scripts to pipelines.

### Comparison with Alternatives

| Tool | Key Concept | Pros | Cons | Suitability for This Project |
| :--- | :--- | :--- | :--- | :--- |
| **Prefect** | **Python-native.** Flows and tasks are decorated Python functions. | - **Minimal Refactoring:** Convert functions to tasks with a simple `@task` decorator.<br>- **High Flexibility:** Easily create dynamic, data-dependent workflows.<br>- **Excellent local development** and modern UI. | - Smaller community compared to Airflow. | **Excellent Choice.** The fastest and most intuitive way to add orchestration to an existing Python script without a steep learning curve. |
| **Airflow** | **Configuration as Code.** DAGs are Python scripts that *define* workflows. | - **Industry Standard:** Huge community and vast library of integrations (providers).<br>- **Proven Scalability** and battle-tested reliability. | - **High Boilerplate:** Requires understanding Operators, DAGs, and XComs.<br>- More complex local setup.<br>- Less flexible for dynamic workflows. | **Overkill.** The added complexity and boilerplate would slow down development for a single-script pipeline. The learning curve is significantly steeper. |
| **Mage** | **All-in-one platform.** Integrates notebooks, orchestration, and data transformation. | - **Interactive Development:** Build and run code blocks like in a notebook.<br>- Combines multiple tools into one. | - A newer tool with a smaller community.<br>- Can feel opinionated in its structure. | **Good Alternative.** A strong contender, also recommended by the course. However, Prefect's direct mapping of functions to tasks is a more seamless transition from our starting script. |
| **Dagster** | **Data-aware.** Models pipelines as graphs of "data assets" (e.g., tables, models). | - **Strong Data Lineage:** Excellent observability of how data assets are produced and consumed.<br>- Great local development and testing tools. | - Requires a shift in mindset from "tasks" to "assets". | **Good Alternative.** Highly valuable for complex projects with many interdependent data sources. For this project, the focus is on orchestrating tasks, making Prefect's model more direct. |

**Conclusion:** Prefect provides the best balance of power and simplicity for our goal: turning a single script into a production-ready pipeline with minimal friction.

---

## 2. Setup and Execution Guide

This project is configured to run out-of-the-box in GitHub Codespaces, which automates the entire setup process.

### Step 1: Launch the Environment
1.  Click the **"Code"** button on the GitHub repository page.
2.  Select the **"Codespaces"** tab.
3.  Click **"Create codespace on main"**.

GitHub will automatically build the development environment based on the configuration in the `.devcontainer` folder. This includes:
*   Starting a Docker-in-Docker service.
*   Running the `docker-compose.yaml` file to launch the MLflow tracking server.
*   Installing all Python dependencies from `requirements.txt`.

### Step 2: Understand the Project Structure

```
.
├── .devcontainer/
│   └── devcontainer.json   # Defines the Codespaces environment
├── 03-orchestration/
│   ├── homework_flow.py      # The main Prefect pipeline script
│   ├── mlflow.dockerfile     # Dockerfile for the MLflow service
│   ├── docker-compose.yaml   # Defines how to run the MLflow service
│   └── requirements.txt      # Python dependencies
└── README.md
```
*   **`devcontainer.json`**: Instructs Codespaces on how to configure the environment, including which Docker image to use and which commands to run after creation (`postCreateCommand`).
*   **`docker-compose.yaml`**: Launches the MLflow server as a service, making it available to our training script.
*   **`homework_flow.py`**: Contains the core logic: reading data, training a model, and logging results, all structured as a Prefect flow with tasks.

### Step 3: Run the Pipeline
Once the Codespace is ready, open a terminal and execute the following command:

```bash
cd 03-orchestration
python homework_flow.py --year 2023 --month 3
```
This command runs the main pipeline script with parameters:
*   `--year 2023`: Use data from the year 2023.
*   `--month 3`: Use data from the 3rd month (March) for the training set. The validation set will automatically use the next month (April).

### Step 4: Review the Results

**1. In the Terminal:**
The script will print the final metrics and data statistics to the console.

```
Number of non-zero elements in train matrix: 138784
Number of non-zero elements in validation matrix: 122674
Number of features: 6053
...
Validation RMSE: 5.470922595809529
```

**2. In the MLflow UI:**
1.  Go to the **"Ports"** tab in your VS Code window.
2.  You will see that port `5000` has been forwarded. Click the **"Open in Browser"** icon (a small globe) next to it.
3.  The MLflow UI will open in a new tab.
4.  Navigate to the `nyc-taxi-experiment-homework` experiment to see your run. You can inspect all logged parameters, the final RMSE metric, and the saved artifacts, including the `preprocessor.b` and the XGBoost model itself.

---

## 3. Final Results Summary

| Question | Metric | Value |
| :--- | :--- | :--- |
| Q1 | Tool Used | Prefect |
| Q2 | Tool Version | 3.4.5 |
| Q3 | Train Matrix Size (NNZ) | 138,784 |
| Q4 | Validation Matrix Size (NNZ)| 122,674 |
| Q5 | Validation RMSE | ~5.47 |
| Q6 | Number of Model Features | 6,053 |
