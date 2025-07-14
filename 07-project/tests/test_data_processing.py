# tests/test_data_processing.py
import os
import pandas as pd
from src.data_processing import run_data_processing


def test_data_processing_smoke_test(tmpdir):
    """
    Smoke test: does the script run without errors and create output files?
    """
    # Create a fake input file
    raw_data_path = os.path.join(tmpdir, "raw_data.csv")
    # Use a minimal dummy dataframe that matches the expected structure
    dummy_df = pd.DataFrame({
        'ID': [1], 'Group': ['Ctrl'], 'Outcome_int': [0], 'ICU_stay': [5],
        'Gestation': [38], 'Temperature': [36.7], 'Heart_rate': [140],
        # Add only a few key columns, not all are needed for a smoke test
        'Total_SOFA': [6]
    })
    dummy_df.to_csv(raw_data_path, index=False, sep=';')

    # Run the processing function
    dest_path = os.path.join(tmpdir, "processed")
    run_data_processing(str(raw_data_path), str(dest_path), 'Outcome_int')

    # Assert that the output files exist
    assert os.path.exists(os.path.join(dest_path, "X_train.pkl"))
    assert os.path.exists(os.path.join(dest_path, "y_train.pkl"))
