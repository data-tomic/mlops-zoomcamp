# 07-project/tests/test_api.py (Final Corrected Version)

from fastapi.testclient import TestClient


# We can now directly import from 'src' because of the __init__.py files,
# making the test environment-independent.
from src.api.main import app

# Create a TestClient instance for making requests
client = TestClient(app)


def test_read_root():
    """
    Test the root endpoint to ensure the API is running.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["status"] == "API is running"


def test_predict_endpoint_model_not_loaded():
    """
    Test the predict endpoint's behavior when the model is not loaded.
    In a test environment, the lifespan event that loads the model does not run by default,
    so the model should be None.
    """
    # Create a valid payload based on the Pydantic model
    valid_payload = {
        "Age": 1,
        "Gender": 0,
        "Gestation": 38,
        "Infect_locus": 0,
        "Temperature": 36.7,
        "Heart_rate": 140,
        "S_AP": 70,
        "D_AP": 45,
        "Mean_AP": 55,
        "Pressors": 0,
        "Dopamine": 0,
        "Norepi": 0,
        "Epineph": 0,
        "Breath_rate": 35,
        "FIO2": 0.21,
        "Oxy_index": 142.8,
        "Ventilator": 1,
        "Cons_dev": 0,
        "Sedation": 0,
        "Glasgow": 14,
        "Lactate": 2.2,
        "Bilirubin": 49.7,
        "Creatinine": 72.9,
        "Urea": 2.8,
        "CRP": 0.4,
        "PCT": 0.18,
        "WBC": 21.49,
        "NEU": 15.47,
        "MON": 2.18,
        "LYM": 3.43,
        "CD64_NEU_DAY_1": 280,
        "CD64_MON_DAY_1": 8845,
        "HLA-DR_MON_MFI_DAY_1": 37985,
        "CD16_NEU_%_DAY_1": 78.2,
        "CD16_NEU_MFI_DAY_1": 68626,
        "Total_SOFA": 6,
        "PLT": 250,
        "PaO2": 95.0,
        "Total_stay": 10,
    }

    # In a default test run, model is None, so this should return 503
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 503
    assert response.json() == {"detail": "Model or its artifacts are not available."}


def test_predict_invalid_payload():
    """
    Test the predict endpoint with an incomplete payload to check for validation errors.
    """
    # Payload missing several required fields
    invalid_payload = {"Age": 1, "Temperature": 36.7}
    response = client.post("/predict", json=invalid_payload)
    # FastAPI should return a 422 Unprocessable Entity for validation errors
    assert response.status_code == 422

