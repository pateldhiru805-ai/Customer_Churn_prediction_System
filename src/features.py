"""Feature engineering and input validation module.

Transforms raw input data dictionaries or DataFrames into the exact 30
one-hot encoded features required by the production machine learning pipeline.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

try:
    from src.config import CANONICAL_FEATURE_COLUMNS, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS
except ImportError:
    from config import CANONICAL_FEATURE_COLUMNS, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS


def validate_raw_input(data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate raw customer input fields against expected schema and ranges.

    Returns:
        (is_valid, error_message)
    """
    if not isinstance(data, dict):
        return False, "Input data must be a JSON object / dictionary."

    # Validate Numeric Fields
    for num_col in NUMERIC_COLUMNS:
        if num_col not in data:
            return False, f"Missing required numeric field: '{num_col}'."
        try:
            val = float(data[num_col])
            if np.isnan(val):
                return False, f"Numeric field '{num_col}' cannot be NaN."
            if val < 0:
                return False, f"Numeric field '{num_col}' must be non-negative (got {val})."
            if num_col == "tenure" and val > 120:
                return False, f"Field 'tenure' seems invalid (> 120 months)."
        except (ValueError, TypeError):
            return False, f"Numeric field '{num_col}' must be a valid number."

    # Validate Categorical Fields
    for cat_col, allowed_vals in CATEGORICAL_COLUMNS.items():
        if cat_col not in data:
            return False, f"Missing required categorical field: '{cat_col}'."
        val = data[cat_col]
        # Handle SeniorCitizen as int or string
        if cat_col == "SeniorCitizen":
            try:
                val = int(val)
                if val not in [0, 1]:
                    return False, f"Field 'SeniorCitizen' must be 0 or 1."
            except (ValueError, TypeError):
                return False, f"Field 'SeniorCitizen' must be 0 or 1."
        elif str(val) not in allowed_vals:
            return (
                False,
                f"Invalid value for '{cat_col}': '{val}'. Allowed values: {allowed_vals}.",
            )

    return True, None


def build_features(
    data: Union[Dict[str, Any], pd.DataFrame],
    feature_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Transform raw customer records into the model's 30 one-hot feature vector.

    Args:
        data: A dictionary of a single customer, or a pandas DataFrame of raw customers.
        feature_columns: List of feature columns to conform to (defaults to CANONICAL_FEATURE_COLUMNS).

    Returns:
        pd.DataFrame containing the exact 30 columns in exact canonical order.
    """
    if feature_columns is None:
        feature_columns = CANONICAL_FEATURE_COLUMNS

    if isinstance(data, dict):
        df_raw = pd.DataFrame([data])
    else:
        df_raw = data.copy()

    # Drop customerID and Churn if present
    cols_to_drop = [c for c in ["customerID", "Churn"] if c in df_raw.columns]
    if cols_to_drop:
        df_raw = df_raw.drop(columns=cols_to_drop)

    # Convert numeric fields
    for num_col in NUMERIC_COLUMNS:
        if num_col in df_raw.columns:
            df_raw[num_col] = pd.to_numeric(df_raw[num_col], errors="coerce").fillna(0.0)

    # Ensure SeniorCitizen is integer
    if "SeniorCitizen" in df_raw.columns:
        df_raw["SeniorCitizen"] = pd.to_numeric(df_raw["SeniorCitizen"], errors="coerce").fillna(0).astype(int)

    # Generate one-hot encoded representation
    # We create all 30 expected feature columns explicitly to guarantee reproducibility
    # regardless of whether the single row contains only a subset of categorical levels.
    n_rows = len(df_raw)
    features_df = pd.DataFrame(0.0, index=df_raw.index, columns=feature_columns)

    # 1. Direct numeric and binary fields
    features_df["SeniorCitizen"] = df_raw["SeniorCitizen"].astype(float)
    features_df["tenure"] = df_raw["tenure"].astype(float)
    features_df["MonthlyCharges"] = df_raw["MonthlyCharges"].astype(float)
    features_df["TotalCharges"] = df_raw["TotalCharges"].astype(float)

    # 2. Gender
    if "gender" in df_raw.columns:
        features_df["gender_Male"] = (df_raw["gender"] == "Male").astype(float)

    # 3. Partner & Dependents
    if "Partner" in df_raw.columns:
        features_df["Partner_Yes"] = (df_raw["Partner"] == "Yes").astype(float)
    if "Dependents" in df_raw.columns:
        features_df["Dependents_Yes"] = (df_raw["Dependents"] == "Yes").astype(float)

    # 4. Phone Service & Multiple Lines
    if "PhoneService" in df_raw.columns:
        features_df["PhoneService_Yes"] = (df_raw["PhoneService"] == "Yes").astype(float)
    if "MultipleLines" in df_raw.columns:
        features_df["MultipleLines_No phone service"] = (
            df_raw["MultipleLines"] == "No phone service"
        ).astype(float)
        features_df["MultipleLines_Yes"] = (df_raw["MultipleLines"] == "Yes").astype(float)

    # 5. Internet Service
    if "InternetService" in df_raw.columns:
        features_df["InternetService_Fiber optic"] = (
            df_raw["InternetService"] == "Fiber optic"
        ).astype(float)
        features_df["InternetService_No"] = (df_raw["InternetService"] == "No").astype(float)

    # 6. Internet Add-ons
    service_cols = [
        ("OnlineSecurity", "OnlineSecurity_No internet service", "OnlineSecurity_Yes"),
        ("OnlineBackup", "OnlineBackup_No internet service", "OnlineBackup_Yes"),
        ("DeviceProtection", "DeviceProtection_No internet service", "DeviceProtection_Yes"),
        ("TechSupport", "TechSupport_No internet service", "TechSupport_Yes"),
        ("StreamingTV", "StreamingTV_No internet service", "StreamingTV_Yes"),
        ("StreamingMovies", "StreamingMovies_No internet service", "StreamingMovies_Yes"),
    ]
    for raw_name, no_internet_col, yes_col in service_cols:
        if raw_name in df_raw.columns:
            features_df[no_internet_col] = (df_raw[raw_name] == "No internet service").astype(float)
            features_df[yes_col] = (df_raw[raw_name] == "Yes").astype(float)

    # 7. Contract
    if "Contract" in df_raw.columns:
        features_df["Contract_One year"] = (df_raw["Contract"] == "One year").astype(float)
        features_df["Contract_Two year"] = (df_raw["Contract"] == "Two year").astype(float)

    # 8. Paperless Billing
    if "PaperlessBilling" in df_raw.columns:
        features_df["PaperlessBilling_Yes"] = (df_raw["PaperlessBilling"] == "Yes").astype(float)

    # 9. Payment Method
    if "PaymentMethod" in df_raw.columns:
        features_df["PaymentMethod_Credit card (automatic)"] = (
            df_raw["PaymentMethod"] == "Credit card (automatic)"
        ).astype(float)
        features_df["PaymentMethod_Electronic check"] = (
            df_raw["PaymentMethod"] == "Electronic check"
        ).astype(float)
        features_df["PaymentMethod_Mailed check"] = (
            df_raw["PaymentMethod"] == "Mailed check"
        ).astype(float)

    # Return exactly the 30 columns ordered
    return features_df[feature_columns]
