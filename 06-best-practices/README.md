# Homework 6: Best Practices and Testing

This project focuses on improving the reliability of a taxi ride duration prediction model by applying software engineering best practices. The work involved significant code refactoring and the implementation of both unit and integration tests.

## Technology Stack

*   Python 3.12
*   `venv` for virtual environment management
*   `pandas` for data manipulation
*   `scikit-learn` for model interaction
*   `pyarrow` for working with the Parquet format
*   `pytest` for unit testing
*   **MinIO** as the S3-compatible storage for integration tests
*   `awscli` and `mc` for command-line interaction with S3

## Completed Steps and Results

### Q1: Code Refactoring

**Objective:** To refactor the code by eliminating global variables, moving the core logic into a `main(year, month)` function, and improving the overall structure.

**Actions Taken:**
1.  A `main(year, month)` function was created to accept the year and month as parameters.
2.  All core logic, except for the data reading function, was moved inside `main`.
3.  The `read_data` function was modified to accept the list of categorical features as a parameter.
4.  The main script invocation block was structured using the standard `if __name__ == '__main__':` construct.

**Result:** The code became more modular and reusable. The correct invocation statement is `if __name__ == '__main__':`.

### Q2: Pytest Installation

**Objective:** To set up the testing environment by installing `pytest` and creating the necessary file structure.

**Actions Taken:**
1.  A `tests/` directory was created to store all test files.
2.  The main test file, `tests/test_batch.py`, was created inside it.
3.  A second, essential file was created to allow Python to correctly handle imports from the test directory.

**Result:** The second required file is **`tests/__init__.py`**, which makes the `tests` directory a proper Python package.

### Q3: Writing the First Unit Test

**Objective:** To test the data preprocessing logic in isolation from file I/O operations.

**Actions Taken:**
1.  The `read_data` function was split into two: `prepare_data` (containing the pure DataFrame transformation logic) and `read_data` (responsible for reading the file and calling `prepare_data`).
2.  A unit test, `test_prepare_data`, was written in `tests/test_batch.py`.
3.  A test DataFrame with 4 rows was created to cover different scenarios (valid data, a trip duration that is too short, and one that is too long).
4.  The test asserts that only valid rows remain after the preprocessing logic is applied.

**Result:** After filtering, the resulting DataFrame should contain **2** rows. The `pytest` test suite passed successfully.

### Q4: S3 Integration Setup (MinIO)

**Objective:** To prepare the script for integration testing by making the input/output paths and S3 connection configurable. **MinIO** was used instead of Localstack.

**Actions Taken:**
1.  Environment variables were configured for the MinIO connection (`S3_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`).
2.  A bucket named `nyc-duration` was created using the MinIO Client (`mc`).
3.  The code was refactored to use an `endpoint_url` for the S3 connection, allowing it to target a custom S3 server.

**Result:** To connect to an S3-compatible storage service other than AWS S3, the **`endpoint_url`** parameter is used.

### Q5: Creating Test Data for the Integration Test

**Objective:** To create a script that generates a test Parquet file and uploads it to the MinIO S3 storage.

**Actions Taken:**
1.  An `integration_test.py` script was created.
2.  The script uses the same test DataFrame from Q3.
3.  The DataFrame is saved to the `nyc-duration` bucket at the path `in/2023-01.parquet`.
4.  During the process, issues with missing dependencies (`fsspec`, `s3fs`) and self-signed SSL certificates (`verify: False`) were resolved.

**Result:** The size of the created file in MinIO was approximately 3.6 KB. The closest answer option is **3620** bytes.

### Q6: Finalizing the Integration Test

**Objective:** To run the complete data processing pipeline (read from MinIO -> process -> write back to MinIO) and verify the final output.

**Actions Taken:**
1.  The `integration_test.py` script was extended to execute `homework_q1.py` with the correct parameters using `os.system`.
2.  The `homework_q1.py` script was configured to accept `year` and `month` from command-line arguments (`sys.argv`).
3.  After the main script ran, `integration_test.py` read the resulting output file (`out/2023-01.parquet`) from MinIO and calculated the sum of the predicted durations.

**Result:** The final sum of predicted durations for the test data was **~36.28**.

---

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/data-tomic/mlops-zoomcamp.git
    cd mlops-zoomcamp/06-best-practices
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    *It's recommended to create a `requirements.txt` file.*
    ```bash
    pip install pandas scikit-learn pyarrow pytest awscli fsspec s3fs
    # You may also need to install the MinIO Client (mc) separately.
    ```

4.  **Set up environment variables for MinIO:**
    ```bash
    export S3_ENDPOINT_URL="<your_minio_endpoint>"
    export AWS_ACCESS_KEY_ID="<your_access_key>"
    export AWS_SECRET_ACCESS_KEY="<your_secret_key>"
    ```

5.  **Create a bucket in MinIO:**
    ```bash
    mc alias set my-minio $S3_ENDPOINT_URL $AWS_ACCESS_KEY_ID $AWS_SECRET_ACCESS_KEY --api S3v4
    mc mb my-minio/nyc-duration
    ```

6.  **Run the unit tests:**
    ```bash
    pytest
    ```

7.  **Run the full integration test:**
    ```bash
    python3 integration_test.py
    ```