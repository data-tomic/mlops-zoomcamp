# 07-project/src/monitoring.py (Final Corrected Version for Evidently v0.4.x)

import json
import pandas as pd

# This import syntax is correct for evidently version 0.4.x
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


def generate_drift_report(
    reference_data_path: str, current_data_path: str, report_path: str
):
    """
    Generates a data drift report comparing reference and current data.
    """
    print("--- Generating Data Drift Report ---")

    # Load reference data
    try:
        reference_df = pd.read_pickle(reference_data_path)
        print(f"Reference data loaded from {reference_data_path}.")
    except FileNotFoundError:
        print(
            f"ERROR: Reference data not found at {reference_data_path}. Cannot generate report."
        )
        # Create an empty report to avoid a 404 error
        with open(report_path, "w") as f:
            f.write("<h1>Reference data not found. Cannot generate report.</h1>")
        return

    # Load current data from prediction logs
    current_data = []
    try:
        with open(current_data_path, "r") as f:
            for line in f:
                # Strip whitespace and check if the line is empty
                clean_line = line.strip()
                if clean_line:
                    current_data.append(json.loads(clean_line))

        if not current_data:
            print("No prediction data found to generate a report.")
            with open(report_path, "w") as f:
                f.write("<h1>No prediction data logged yet.</h1>")
            return

        current_df = pd.DataFrame(current_data)
        current_df = current_df[reference_df.columns]
        print(f"Current data loaded from {current_data_path} ({len(current_df)} rows).")

    except FileNotFoundError:
        print(f"Warning: Prediction log file not found at {current_data_path}.")
        with open(report_path, "w") as f:
            f.write("<h1>Prediction log file not found.</h1>")
        return
    except Exception as e:
        print(f"Error reading or processing prediction logs: {e}")
        return

    # Create and run the Evidently Data Drift Report
    print("Creating Evidently report...")
    data_drift_report = Report(metrics=[DataDriftPreset()])
    data_drift_report.run(reference_data=reference_df, current_data=current_df)

    # Save the report as an HTML file
    data_drift_report.save_html(report_path)
    print(f"Report saved successfully to {report_path}")
