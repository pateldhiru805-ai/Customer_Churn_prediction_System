"""Inference and prediction module for Customer Churn Prediction.

Loads the serialized production pipeline, validates input, executes inference,
derives risk bands, explainability factors, and prescriptive retention recommendations.
"""

from typing import Any, Dict, List, Optional, Tuple
import json
import joblib
import numpy as np
import pandas as pd

try:
    from src.config import (
        DEFAULT_CHURN_THRESHOLD,
        FEATURE_COLUMNS_PATH,
        MODEL_METADATA_PATH,
        MODEL_PATH,
        RISK_BANDS,
    )
    from src.features import build_features, validate_raw_input
except ImportError:
    from config import (
        DEFAULT_CHURN_THRESHOLD,
        FEATURE_COLUMNS_PATH,
        MODEL_METADATA_PATH,
        MODEL_PATH,
        RISK_BANDS,
    )
    from features import build_features, validate_raw_input

# Cached model instance and metadata
_CACHED_MODEL = None
_CACHED_FEATURES = None
_CACHED_METADATA = None


def load_model() -> Tuple[Any, List[str], Dict[str, Any]]:
    """Load model pipeline, feature columns, and metadata into memory (cached)."""
    global _CACHED_MODEL, _CACHED_FEATURES, _CACHED_METADATA

    if _CACHED_MODEL is not None:
        return _CACHED_MODEL, _CACHED_FEATURES, _CACHED_METADATA

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Run training first.")

    _CACHED_MODEL = joblib.load(MODEL_PATH)

    if FEATURE_COLUMNS_PATH.exists():
        with open(FEATURE_COLUMNS_PATH, "r") as f:
            _CACHED_FEATURES = json.load(f)
    else:
        _CACHED_FEATURES = None

    if MODEL_METADATA_PATH.exists():
        with open(MODEL_METADATA_PATH, "r") as f:
            _CACHED_METADATA = json.load(f)
    else:
        _CACHED_METADATA = {}

    return _CACHED_MODEL, _CACHED_FEATURES, _CACHED_METADATA


def determine_risk_level(probability: float) -> str:
    """Classify probability into risk level band: Low, Medium, High."""
    for level, (low, high) in RISK_BANDS.items():
        if low <= probability < high:
            return level
    return "High" if probability >= 0.65 else "Low"


def derive_risk_factors(data: Dict[str, Any], probability: float) -> List[Dict[str, str]]:
    """Explain key drivers for this customer profile based on domain features."""
    factors = []

    contract = data.get("Contract", "")
    if contract == "Month-to-month":
        factors.append({
            "type": "risk",
            "feature": "Contract",
            "impact": "High Risk (+)",
            "description": "Month-to-month contracts have historically highest churn rates (~42%).",
        })
    elif contract in ["One year", "Two year"]:
        factors.append({
            "type": "anchor",
            "feature": "Contract",
            "impact": "Retention Anchor (-)",
            "description": f"Long-term commitment ({contract}) strongly reduces churn propensity.",
        })

    tenure = float(data.get("tenure", 0))
    if tenure <= 6:
        factors.append({
            "type": "risk",
            "feature": "Tenure",
            "impact": "High Risk (+)",
            "description": f"Tenure of only {int(tenure)} months; new customers are most vulnerable to churn.",
        })
    elif tenure >= 24:
        factors.append({
            "type": "anchor",
            "feature": "Tenure",
            "impact": "Retention Anchor (-)",
            "description": f"Established tenure of {int(tenure)} months indicates high brand stickiness.",
        })

    internet = data.get("InternetService", "")
    if internet == "Fiber optic":
        factors.append({
            "type": "risk",
            "feature": "Internet Service",
            "impact": "Moderate Risk (+)",
            "description": "Fiber optic accounts exhibit higher churn due to competitive pricing alternatives.",
        })

    monthly = float(data.get("MonthlyCharges", 0))
    if monthly >= 75:
        factors.append({
            "type": "risk",
            "feature": "Monthly Charges",
            "impact": "Moderate Risk (+)",
            "description": f"Premium monthly spend (${monthly:.2f}/mo) increases price sensitivity.",
        })
    elif monthly < 35:
        factors.append({
            "type": "anchor",
            "feature": "Monthly Charges",
            "impact": "Retention Anchor (-)",
            "description": f"Affordable monthly plan (${monthly:.2f}/mo) reduces budget attrition.",
        })

    payment = data.get("PaymentMethod", "")
    if payment == "Electronic check":
        factors.append({
            "type": "risk",
            "feature": "Payment Method",
            "impact": "Moderate Risk (+)",
            "description": "Manual electronic check payments correlate with higher churn compared to autopay.",
        })
    elif "automatic" in payment:
        factors.append({
            "type": "anchor",
            "feature": "Payment Method",
            "impact": "Retention Anchor (-)",
            "description": "Automated payment method ensures seamless billing continuity.",
        })

    tech_support = data.get("TechSupport", "")
    if tech_support == "No" and internet != "No":
        factors.append({
            "type": "risk",
            "feature": "Tech Support",
            "impact": "Risk Factor (+)",
            "description": "Lack of technical support increases dissatisfaction during service issues.",
        })
    elif tech_support == "Yes":
        factors.append({
            "type": "anchor",
            "feature": "Tech Support",
            "impact": "Retention Anchor (-)",
            "description": "Active technical support significantly enhances customer satisfaction.",
        })

    return factors


def generate_retention_recommendations(data: Dict[str, Any], risk_level: str) -> List[str]:
    """Generate prescriptive actions to retain customers based on profile."""
    actions = []

    if data.get("Contract") == "Month-to-month":
        actions.append("Offer a 15% promotional discount on upgrading to an Annual Contract.")

    if data.get("PaymentMethod") == "Electronic check":
        actions.append("Provide a $10 one-time bill credit for enrolling in Auto-Pay (Bank or Credit Card).")

    if data.get("TechSupport") == "No" and data.get("InternetService") != "No":
        actions.append("Offer 3 months complimentary Premium Tech Support and Online Security bundle.")

    if float(data.get("tenure", 0)) < 6:
        actions.append("Schedule an automated customer onboarding check-in call and satisfaction survey.")

    if not actions:
        actions.append("Customer is on a balanced plan; maintain quarterly loyalty check-ins.")

    return actions


def predict_one(data: Dict[str, Any], threshold: Optional[float] = None) -> Dict[str, Any]:
    """Execute prediction pipeline on a single customer dictionary.

    Returns:
        Dictionary containing label, probability, risk level, threshold, and factors.
    """
    # 1. Validate
    is_valid, error_msg = validate_raw_input(data)
    if not is_valid:
        raise ValueError(error_msg)

    # 2. Load model
    model, feature_cols, metadata = load_model()

    # Determine threshold
    if threshold is None:
        threshold = float(metadata.get("optimal_threshold", DEFAULT_CHURN_THRESHOLD))

    # 3. Build features
    features_df = build_features(data, feature_columns=feature_cols)

    # 4. Predict probability
    prob_churn = float(model.predict_proba(features_df)[0, 1])
    is_churn = bool(prob_churn >= threshold)
    label = "Likely to churn" if is_churn else "Likely to stay"
    risk_level = determine_risk_level(prob_churn)

    # 5. Domain explainability & actions
    risk_factors = derive_risk_factors(data, prob_churn)
    recommendations = generate_retention_recommendations(data, risk_level)

    return {
        "prediction": label,
        "churn_probability": round(prob_churn, 4),
        "churn_percentage": round(prob_churn * 100, 1),
        "risk_level": risk_level,
        "threshold": round(threshold, 4),
        "model_name": metadata.get("model_name", "Best Model"),
        "model_version": metadata.get("model_version", "1.0.0"),
        "risk_factors": risk_factors,
        "retention_actions": recommendations,
    }
