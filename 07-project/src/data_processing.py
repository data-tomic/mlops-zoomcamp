# src/data_processing.py (English Version)

import os
import click
import pandas as pd
from sklearn.model_selection import train_test_split


def run_data_processing(raw_data_path: str, dest_path: str, target_col: str):
    """
    Main logic for processing clinical_data.csv.
    Reads the CSV, cleans it, splits it into train/validation sets, and saves them as pickle files.
    """
    print(f"1. Reading raw data from: {raw_data_path}")
    try:
        # Fixed: Added sep=';' to correctly parse the columns
        df = pd.read_csv(raw_data_path, sep=";")
        print("CSV file successfully read with ';' delimiter.")
    except FileNotFoundError:
        print(f"ERROR: File not found at {raw_data_path}. Please ensure it exists.")
        return

    # --- Data Cleaning ---
    df = df.drop(columns=["ID", "Group"], errors="ignore")
    print("Dropped 'ID' and 'Group' columns.")

    # Simple missing value strategy: fill with median
    # Fixed: Added numeric_only=True for robustness
    df.fillna(df.median(numeric_only=True), inplace=True)
    print("Filled missing values with median.")

    # Define features (X) and target (y)
    X = df.drop(columns=["Outcome_int", "ICU_stay"], errors="ignore")
    y = df[target_col]
    print(f"Target variable for the model: '{target_col}'")

    # --- Data Splitting ---
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(
        f"Data split into training ({X_train.shape[0]} rows) and validation ({X_val.shape[0]} rows) sets."  # noqa E501
    )

    # --- Save Processed Data ---
    os.makedirs(dest_path, exist_ok=True)
    X_train.to_pickle(os.path.join(dest_path, "X_train.pkl"))
    y_train.to_pickle(os.path.join(dest_path, "y_train.pkl"))
    X_val.to_pickle(os.path.join(dest_path, "X_val.pkl"))
    y_val.to_pickle(os.path.join(dest_path, "y_val.pkl"))

    print(f"Processed data successfully saved to: {dest_path}")


@click.command()
@click.option(
    "--raw-data-path",
    default="data/clinical_data.csv",
    help="Path to the raw CSV file.",
)  # noqa E501
@click.option(
    "--dest-path", default="data/processed", help="Folder to save processed data."
)
@click.option("--target-col", default="Outcome_int", help="Name of the target column.")
def process_data_command(raw_data_path: str, dest_path: str, target_col: str):
    """Command-line entry point to run data processing."""
    run_data_processing(raw_data_path, dest_path, target_col)


if __name__ == "__main__":
    process_data_command()
