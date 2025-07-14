# src/api/main.py (Final Version with All Fixes)

import json
import pandas as pd
import mlflow
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from src.monitoring import generate_drift_report

# --- CORRECTED IMPORT ---
from pydantic import BaseModel, Field, ConfigDict

# ------------------------
from contextlib import asynccontextmanager

# --- Global variables ---
MODEL_NAME = "sepsis-outcome-classifier"
MODEL_ALIAS = "production"
model = None
COLUMN_ORDER = None
# ------------------------

# --- CORRECTED LOGGING SETUP ---
# Create a dedicated logger for prediction data
prediction_logger = logging.getLogger("prediction_logger")
# Prevent it from propagating to the root Uvicorn logger
prediction_logger.propagate = False
# Set the level
prediction_logger.setLevel(logging.INFO)
# Add a handler that writes directly to a file
# This handler has NO formatter, so it writes the raw message.
file_handler = logging.FileHandler("predictions.log", mode="a")
prediction_logger.addHandler(file_handler)
# -----------------------------


# --- Lifespan Context Manager (replaces @app.on_event) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # This code runs ONCE when the API starts up
    global model, COLUMN_ORDER
    print(f"Loading model '{MODEL_NAME}' with alias '{MODEL_ALIAS}'...")
    try:
        model = mlflow.pyfunc.load_model(
            model_uri=f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
        )
        print("Model loaded successfully.")

        client = mlflow.tracking.MlflowClient()
        model_version_details = client.get_model_version_by_alias(
            name=MODEL_NAME, alias=MODEL_ALIAS
        )  # noqa E501
        run_id = model_version_details.run_id

        client.download_artifacts(run_id=run_id, path="column_order.json", dst_path=".")
        with open("column_order.json", "r") as f:
            COLUMN_ORDER = json.load(f)
        print("Column order artifact loaded successfully.")
    except Exception as e:
        print(f"FATAL: Could not load model or artifacts. Error: {e}")
        model = None
        COLUMN_ORDER = None

    yield
    # Code below yield would run on shutdown


# ------------------------------------


# --- Setup a simple logger to save prediction data ---
# This will create a 'predictions.log' file and append to it.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    filename="predictions.log",
    filemode="a",
)
prediction_logger = logging.getLogger("prediction_logger")
# ----------------------------------------------------


# --- Pydantic Model with modern syntax ---
class ClinicalRecord(BaseModel):
    # This replaces the old class Config
    model_config = ConfigDict(populate_by_name=True)

    # All fields are now defined using json_schema_extra for examples
    Age: int = Field(json_schema_extra={"example": 1})
    Gender: int = Field(json_schema_extra={"example": 0})
    Gestation: int = Field(json_schema_extra={"example": 38})
    Infect_locus: int = Field(json_schema_extra={"example": 0})
    Temperature: float = Field(json_schema_extra={"example": 36.7})
    Heart_rate: int = Field(json_schema_extra={"example": 140})
    S_AP: int = Field(json_schema_extra={"example": 70})
    D_AP: int = Field(json_schema_extra={"example": 45})
    Mean_AP: int = Field(json_schema_extra={"example": 55})
    Pressors: int = Field(json_schema_extra={"example": 0})
    Dopamine: int = Field(json_schema_extra={"example": 0})
    Norepi: int = Field(json_schema_extra={"example": 0})
    Epineph: int = Field(json_schema_extra={"example": 0})
    Breath_rate: int = Field(json_schema_extra={"example": 35})
    FIO2: float = Field(json_schema_extra={"example": 0.21})
    Oxy_index: float = Field(json_schema_extra={"example": 142.8})
    Ventilator: int = Field(json_schema_extra={"example": 1})
    Cons_dev: int = Field(json_schema_extra={"example": 0})
    Sedation: int = Field(json_schema_extra={"example": 0})
    Glasgow: int = Field(json_schema_extra={"example": 14})
    Lactate: float = Field(json_schema_extra={"example": 2.2})
    Bilirubin: float = Field(json_schema_extra={"example": 49.7})
    Creatinine: float = Field(json_schema_extra={"example": 72.9})
    Urea: float = Field(json_schema_extra={"example": 2.8})
    CRP: float = Field(json_schema_extra={"example": 0.4})
    PCT: float = Field(json_schema_extra={"example": 0.18})
    WBC: float = Field(json_schema_extra={"example": 21.49})
    NEU: float = Field(json_schema_extra={"example": 15.47})
    MON: float = Field(json_schema_extra={"example": 2.18})
    LYM: float = Field(json_schema_extra={"example": 3.43})
    CD64_NEU_DAY_1: int = Field(json_schema_extra={"example": 280})
    CD64_MON_DAY_1: int = Field(json_schema_extra={"example": 8845})
    HLA_DR_MON_MFI_DAY_1: int = Field(
        alias="HLA-DR_MON_MFI_DAY_1", json_schema_extra={"example": 37985}
    )  # noqa E501
    CD16_NEU_DAY_1: float = Field(
        alias="CD16_NEU_%_DAY_1", json_schema_extra={"example": 78.2}
    )
    CD16_NEU_MFI_DAY_1: int = Field(json_schema_extra={"example": 68626})
    Total_SOFA: int = Field(json_schema_extra={"example": 6})
    PLT: int = Field(json_schema_extra={"example": 250})
    PaO2: float = Field(json_schema_extra={"example": 95.0})
    Total_stay: int = Field(json_schema_extra={"example": 10})


# --- FastAPI App ---
app = FastAPI(
    title="Sepsis Outcome Prediction API",
    description="API to predict neonatal sepsis outcome based on clinical data.",
    lifespan=lifespan,
)


@app.post("/predict")
def predict_outcome(record: ClinicalRecord):
    if not model or not COLUMN_ORDER:
        raise HTTPException(
            status_code=503, detail="Model or its artifacts are not available."
        )

    data_dict = record.model_dump(by_alias=True)
    df = pd.DataFrame([data_dict])

    try:
        df_reordered = df[COLUMN_ORDER]
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing feature in request: {e}")

    prediction = model.predict(df_reordered)
    outcome = int(prediction[0])

    # --- THIS IS THE NEW PART ---
    # Log the incoming data (as a JSON string) for monitoring
    prediction_logger.info(record.model_dump_json(by_alias=True))
    # --------------------------

    return {
        "prediction": outcome,
        "interpretation": "Deceased" if outcome == 1 else "Survived",
    }


# --- THIS IS THE NEW ENDPOINT ---
@app.get("/monitor", response_class=HTMLResponse)
def get_monitoring_report():
    """
    Generates and returns the data drift monitoring report.
    """
    report_file = "evidently_drift_report.html"
    # The reference data is our training data, located relative to the project root
    # Inside the Docker container, we assume a specific path for it.
    # We will need to mount the processed data folder.
    reference_data_path = "data/processed/X_train.pkl"
    current_data_path = "predictions.log"

    generate_drift_report(
        reference_data_path=reference_data_path,
        current_data_path=current_data_path,
        report_path=report_file,
    )

    try:
        with open(report_file, "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>No report generated.</h1><p>This may be because no prediction data has been logged yet.</p>"  # noq E501
        )


# -------------------------------


@app.get("/")
def read_root():
    return {
        "status": "API is running",
        "model_name": MODEL_NAME,
        "model_alias": MODEL_ALIAS,
    }
