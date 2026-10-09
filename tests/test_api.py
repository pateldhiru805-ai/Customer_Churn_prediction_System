"""API endpoint integration tests using Flask test client (tests/test_api.py)."""

import pytest
import json
from app.app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def valid_payload():
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 4,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.0,
        "TotalCharges": 340.0,
    }


def test_health_endpoint(client):
    """Verify /api/health returns online status."""
    response = client.get("/api/health")
    assert response.status_code in [200, 207]
    data = response.get_json()
    assert data["api"] == "online"
    assert data["model"]["loaded"] is True


def test_predict_endpoint_success(client, valid_payload):
    """Verify /api/predict returns 200 with complete prediction payload."""
    response = client.post(
        "/api/predict",
        data=json.dumps(valid_payload),
        content_type="application/json",
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert "prediction" in data
    assert "churn_probability" in data
    assert "risk_level" in data
    assert "risk_factors" in data
    assert "retention_actions" in data


def test_predict_endpoint_missing_field(client, valid_payload):
    """Verify /api/predict returns 400 when a required field is missing."""
    del valid_payload["Contract"]
    response = client.post(
        "/api/predict",
        data=json.dumps(valid_payload),
        content_type="application/json",
    )
    assert response.status_code == 400
    data = response.get_json()
    assert data["success"] is False
    assert "Contract" in data["error"]


def test_predict_endpoint_invalid_category(client, valid_payload):
    """Verify /api/predict returns 400 when category is invalid."""
    valid_payload["InternetService"] = "5G Satellite"
    response = client.post(
        "/api/predict",
        data=json.dumps(valid_payload),
        content_type="application/json",
    )
    assert response.status_code == 400
    data = response.get_json()
    assert data["success"] is False
    assert "Invalid value for 'InternetService'" in data["error"]


def test_history_endpoint(client):
    """Verify /api/history returns list of predictions."""
    response = client.get("/api/history?limit=10")
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert isinstance(data["predictions"], list)


def test_export_endpoint(client):
    """Verify /api/export returns downloadable CSV response."""
    response = client.get("/api/export")
    assert response.status_code == 200
    assert response.mimetype == "text/csv"
    assert "attachment;filename=churn_prediction_history.csv" in response.headers.get("Content-Disposition", "")


def test_metrics_endpoint(client):
    """Verify /api/metrics returns model metadata."""
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert "metadata" in data


def test_sample_csv_endpoint(client):
    """Verify /api/sample-csv returns sample CSV file."""
    response = client.get("/api/sample-csv")
    assert response.status_code == 200
    assert response.mimetype == "text/csv"


def test_batch_predict_endpoint(client):
    """Verify /api/batch-predict accepts CSV and returns batch predictions."""
    import io
    csv_content = """customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges
CUST-TEST-1,Female,0,Yes,No,1,No,No phone service,DSL,No,Yes,No,No,No,No,Month-to-month,Yes,Electronic check,29.85,29.85
CUST-TEST-2,Male,0,No,No,60,Yes,Yes,DSL,Yes,Yes,Yes,Yes,No,No,Two year,No,Credit card (automatic),55.0,3300.0
"""
    data = {"file": (io.BytesIO(csv_content.encode("utf-8")), "customers.csv")}
    response = client.post(
        "/api/batch-predict?format=json",
        data=data,
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["success"] is True
    assert json_data["summary"]["total"] == 2
    assert "high_risk" in json_data["summary"]

