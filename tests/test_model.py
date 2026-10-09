"""Model validation and inference unit tests (tests/test_model.py)."""

import pytest
import json
from src.config import MODEL_PATH, MODEL_METADATA_PATH, FEATURE_COLUMNS_PATH
from src.predict import load_model, predict_one


def test_model_artifacts_exist():
    """Verify serialized model and metadata artifacts exist."""
    assert MODEL_PATH.exists(), "Model binary missing!"
    assert MODEL_METADATA_PATH.exists(), "Metadata file missing!"
    assert FEATURE_COLUMNS_PATH.exists(), "Feature columns file missing!"


def test_load_model_returns_pipeline():
    """Verify load_model successfully returns pipeline and feature lists."""
    model, features, metadata = load_model()
    assert model is not None
    assert len(features) == 30
    assert "model_name" in metadata
    assert "optimal_threshold" in metadata


def test_predict_one_output_structure():
    """Verify prediction payload structure and probabilities."""
    sample = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
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
        "MonthlyCharges": 90.0,
        "TotalCharges": 180.0,
    }
    result = predict_one(sample)
    assert "prediction" in result
    assert "churn_probability" in result
    assert 0.0 <= result["churn_probability"] <= 1.0
    assert result["risk_level"] in ["Low", "Medium", "High"]
    assert isinstance(result["risk_factors"], list)
    assert isinstance(result["retention_actions"], list)


def test_loyal_customer_has_low_churn_risk():
    """Verify high-tenure multi-service customer exhibits low churn probability."""
    loyal_sample = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 65,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 50.0,
        "TotalCharges": 3250.0,
    }
    result = predict_one(loyal_sample)
    assert result["churn_probability"] < 0.25
    assert result["prediction"] == "Likely to stay"


def test_metadata_performance_target():
    """Verify test set ROC-AUC meets project PRD requirement (>= 0.83)."""
    with open(MODEL_METADATA_PATH, "r") as f:
        metadata = json.load(f)
    roc_auc = metadata["test_metrics"]["roc_auc"]
    assert roc_auc >= 0.83, f"ROC-AUC {roc_auc} does not meet target 0.83"
