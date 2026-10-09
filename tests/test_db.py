"""Database operations unit tests (tests/test_db.py)."""

import pytest
from app.db import get_db_connection, init_db, save_prediction, get_recent_predictions, check_db_health


def test_init_db_and_health():
    """Verify database initialization creates schema and reports health."""
    init_db()
    health = check_db_health()
    assert health["status"] == "healthy"
    assert health["connected"] is True
    assert health["engine"] in ["mysql", "sqlite"]


def test_save_and_retrieve_prediction():
    """Verify saving a prediction persists data and can be retrieved."""
    input_sample = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 45.0,
        "TotalCharges": 225.0,
    }
    prediction_result = {
        "prediction": "Likely to churn",
        "churn_probability": 0.62,
        "risk_level": "Medium",
        "model_name": "Gradient Boosting",
        "model_version": "1.0.0",
    }

    saved, err = save_prediction(input_sample, prediction_result, customer_id="TEST-999")
    assert saved is True
    assert err is None

    recent = get_recent_predictions(limit=10)
    assert len(recent) > 0
    latest = recent[0]
    assert latest["contract"] == "Month-to-month"
    assert float(latest["churn_probability"]) == 0.62
    assert latest["prediction_label"] == "Likely to churn"
