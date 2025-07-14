# tests/test_data_processing.py (Final Corrected Version)

import os
import pandas as pd
from src.data_processing import run_data_processing


def test_data_processing_smoke_test(tmpdir):
    """
    Smoke test: does the script run without errors and create output files?
    """
    # Create a fake input file with enough data for a stratified train/test split.
    raw_data_path = os.path.join(tmpdir, "raw_data.csv")

    # CORRECTED: Use 10 samples with at least 2 samples per class for stratification.
    dummy_data = {
        "ID": range(10),
        "Group": ["Ctrl", "Sepsis"] * 5,
        "Outcome_int": [
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
        ],  # Guarantees enough samples for each class
        "ICU_stay": [5, 10, 3, 8, 12, 4, 6, 9, 7, 11],
        "Gestation": [38, 39, 37, 40, 38, 39, 40, 37, 38, 39],
        "Temperature": [36.7, 38.1, 36.5, 37.0, 36.8, 36.9, 37.2, 36.6, 36.8, 37.1],
        "Heart_rate": [140, 150, 135, 142, 148, 133, 144, 155, 129, 141],
        "Total_SOFA": [6, 12, 5, 7, 9, 11, 8, 10, 4, 13],
    }
    dummy_df = pd.DataFrame(dummy_data)
    dummy_df.to_csv(raw_data_path, index=False, sep=";")

    # Run the processing function
    dest_path = os.path.join(tmpdir, "processed")
    run_data_processing(str(raw_data_path), str(dest_path), "Outcome_int")

    # Assert that the output files exist
    assert os.path.exists(os.path.join(dest_path, "X_train.pkl"))
    assert os.path.exists(os.path.join(dest_path, "y_train.pkl"))
    assert os.path.exists(os.path.join(dest_path, "X_val.pkl"))
    assert os.path.exists(os.path.join(dest_path, "y_val.pkl"))
