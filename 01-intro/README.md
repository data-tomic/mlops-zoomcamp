# Module 1: Introduction to MLOps

## Objective

This homework serves as an introduction to the machine learning lifecycle. The goal was to take a raw dataset of NYC taxi trips, perform basic feature engineering, train a simple linear regression model, and evaluate its performance using Root Mean Squared Error (RMSE).

## Key Technologies
- **Jupyter Notebook**: For interactive development and data exploration.
- **Pandas**: For loading and manipulating the data.
- **Scikit-learn**: For training the `LinearRegression` model and calculating metrics.
- **PyArrow**: For reading Parquet files efficiently.

## Setup

1.  Create and activate a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

1.  **Download the data**: This project requires the Yellow Taxi Trip data for January and February 2023. You can download them from the [TLC Trip Record Data website](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page). Place the Parquet files into a `data/` subdirectory.
    - `yellow_tripdata_2023-01.parquet`
    - `yellow_tripdata_2023-02.parquet`

2.  **Launch Jupyter Notebook**:
    ```bash
    jupyter notebook
    ```

3.  **Open and run the notebook**: Open the `homework.ipynb` file and execute the cells sequentially to see the data processing, model training, and evaluation steps.

## Homework Results

The notebook contains the step-by-step process to answer the homework questions:
- Calculating the standard deviation of trip durations.
- Filtering outliers and analyzing the remaining fraction of data.
- Preparing the feature matrix using `DictVectorizer` and checking its dimensionality.
- Training a model and calculating the RMSE on the training and validation datasets.