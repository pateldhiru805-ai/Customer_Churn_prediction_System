"""Unit tests for feature engineering and input validation (src/features.py)."""

import pytest
import pandas as pd
from src.config import CANONICAL_FEATURE_COLUMNS
from src.features import build_features, validate_raw_input


@pytest.fixture
def sample_valid_customer():
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "One year",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 55.0,
        "TotalCharges": 660.0,
    }


def test_build_features_output_shape_and_columns(sample_valid_customer):
    """Verify build_features produces exactly 30 canonical columns in order."""
    features_df = build_features(sample_valid_customer)
    assert features_df.shape == (1, 30)
    assert list(features_df.columns) == CANONICAL_FEATURE_COLUMNS


def test_validate_raw_input_valid(sample_valid_customer):
    """Verify that a valid customer dictionary passes validation."""
    is_valid, error_msg = validate_raw_input(sample_valid_customer)
    assert is_valid is True
    assert error_msg is None


def test_validate_raw_input_missing_field(sample_valid_customer):
    """Verify validation fails when a mandatory field is missing."""
    del sample_valid_customer["tenure"]
    is_valid, error_msg = validate_raw_input(sample_valid_customer)
    assert is_valid is False
    assert "tenure" in error_msg


def test_validate_raw_input_invalid_category(sample_valid_customer):
    """Verify validation catches unauthorized category values."""
    sample_valid_customer["Contract"] = "Infinite Plan"
    is_valid, error_msg = validate_raw_input(sample_valid_customer)
    assert is_valid is False
    assert "Invalid value for 'Contract'" in error_msg


def test_validate_raw_input_negative_charge(sample_valid_customer):
    """Verify validation catches negative charges."""
    sample_valid_customer["MonthlyCharges"] = -25.0
    is_valid, error_msg = validate_raw_input(sample_valid_customer)
    assert is_valid is False
    assert "must be non-negative" in error_msg


def test_build_features_handles_dataframe():
    """Verify build_features handles DataFrames with multiple rows."""
    df_raw = pd.DataFrame([
        {
            "gender": "Male",
            "SeniorCitizen": 1,
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
            "MonthlyCharges": 95.0,
            "TotalCharges": 190.0,
        },
        {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "Yes",
            "tenure": 70,
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
            "MonthlyCharges": 60.0,
            "TotalCharges": 4200.0,
        },
    ])
    features = build_features(df_raw)
    assert features.shape == (2, 30)
    assert features.iloc[0]["SeniorCitizen"] == 1.0
    assert features.iloc[1]["SeniorCitizen"] == 0.0
    assert features.iloc[0]["InternetService_Fiber optic"] == 1.0
    assert features.iloc[1]["InternetService_Fiber optic"] == 0.0
