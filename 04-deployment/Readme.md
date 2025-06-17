# MLOps Zoomcamp - Homework #4: Batch Deployment

## Description

The goal of this homework is to deploy a taxi trip duration prediction model in batch mode. We will progress from a simple script to a fully isolated Docker container that performs predictions on demand.

**Dataset:** [Yellow Taxi Trip Records](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

## Prerequisites

To fully reproduce these results, you will need:
*   Python 3.10+
*   `pipenv` for dependency management
*   Docker Desktop or Docker Engine

## ⚙️ Initial Setup

1.  **Clone the repository** or ensure all necessary files are in the same directory.

2.  **Create a virtual environment and install dependencies.**
    In the root directory of the project, run the following command. This will create `Pipfile` and `Pipfile.lock` to ensure the correct library versions are used.

    ```bash
    pipenv install pandas pyarrow scikit-learn==1.3.2
    ```

---

## 📝 Problem Solutions (Q1-Q6)

### Q1: Standard Deviation of Predictions

**Question:** What's the standard deviation of the predicted duration for the March 2023 dataset?

**Method:**
Create a script that:
1.  Loads the pre-trained model (`model.bin`).
2.  Reads the trip data for March 2023.
3.  Makes duration predictions.
4.  Calculates the standard deviation (`std()`) of the predictions.

**File: `homework_q1.py`**
```python
#!/usr/bin/env python
# coding: utf-8

import pickle
import pandas as pd
import sys

def run():
    year = 2023
    month = 3
    model_path = 'model.bin'

    print(f"Loading the model from: {model_path}...")
    try:
        with open(model_path, 'rb') as f_in:
            dv, model = pickle.load(f_in)
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found.", file=sys.stderr)
        sys.exit(1)
    print("Model loaded successfully.")

    categorical = ['PULocationID', 'DOLocationID']

    def read_data(filename):
        print(f"Reading data from: {filename}...")
        df = pd.read_parquet(filename)
        df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
        df['duration'] = df.duration.dt.total_seconds() / 60
        df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
        df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
        return df

    input_file_url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    df = read_data(input_file_url)

    print("Preparing features for prediction...")
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    
    print("Making predictions...")
    y_pred = model.predict(X_val)

    std_dev = y_pred.std()
    
    print("-" * 50)
    print(f"✅ Done! Calculation for {year:04d}-{month:02d}.")
    print(f"Standard deviation of predicted duration: {std_dev:.2f}")
    print("-" * 50)

if __name__ == '__main__':
    run()
```

**Commands to run:**
```bash
# Download the raw model file
wget https://media.githubusercontent.com/media/DataTalksClub/mlops-zoomcamp/main/cohorts/2025/04-deployment/homework/model.bin -O model.bin

# Run the script
python homework_q1.py
```
**Answer: 6.24** (The actual result of `6.25` is closest to this option).

### Q2: Output File Size

**Question:** What's the size of the output file after saving the `ride_id` and predictions?

**Method:**
Modify the script from Q1 to:
1.  Generate a `ride_id` for each trip.
2.  Create a new DataFrame containing only `ride_id` and `prediction`.
3.  Save it as a Parquet file using `pyarrow` with no compression.
4.  Print the file size on disk.

**File: `homework_q2.py`**
*(Only the key part of the script is shown below)*
```python
# ... (code from Q1) ...

# Create ride_id
df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

# Create a results DataFrame
df_result = pd.DataFrame()
df_result['ride_id'] = df['ride_id']
df_result['prediction'] = y_pred

# Save to a Parquet file
output_file = f'predictions_{year:04d}-{month:02d}.parquet'
df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)

# Calculate and print the file size
file_size_bytes = os.path.getsize(output_file)
file_size_mb = file_size_bytes / (1024 * 1024)
print(f"Output file size: {file_size_mb:.2f} MB")
```

**Command to run:**
```bash
python homework_q2.py
```
**Answer: 66M** (The actual result of `65.46 MB` corresponds to this option).

### Q3: Converting a Notebook to a Script

**Question:** Which command do you need to execute to turn a Jupyter Notebook into a script?

**Answer:**
```bash
jupyter nbconvert --to script starter.ipynb
```

### Q4: Dependency Hash in `Pipfile.lock`

**Question:** What's the first hash for the `scikit-learn` dependency in `Pipfile.lock`?

**Method:**
1.  Install dependencies using `pipenv`.
2.  Analyze the generated `Pipfile.lock` file.

**Installation Command:**
```bash
pipenv install pandas pyarrow scikit-learn==1.3.2
```

**Analysis of `Pipfile.lock`:**
In the `Pipfile.lock` file, find the `"scikit-learn"` section:
```json
"scikit-learn": {
    "hashes": [
        "sha256:0402638c9a7c219ee52c94cbebc8fcb5eb9fe9c773717965c1f4185588ad3107",
        ...
    ],
    "version": "==1.3.2"
},
```
**Answer: `sha256:04026