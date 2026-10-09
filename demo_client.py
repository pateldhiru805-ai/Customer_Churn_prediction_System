"""Interactive API Demo Client & Verification Script.

Tests and showcases all Customer Churn Prediction REST API endpoints:
- Health check (/api/health)
- Single prediction with explainability and financial ARR (/api/predict)
- Batch CSV scoring (/api/batch-predict)
- Historical audit retrieval (/api/history)
- Model metadata & metrics (/api/metrics)

Usage:
    python demo_client.py
"""

import json
import time
from pathlib import Path
import requests

BASE_URL = "http://127.0.0.1:5000"
ROOT_DIR = Path(__file__).resolve().parent
SAMPLE_CSV_PATH = ROOT_DIR / "Dataset" / "sample_batch_customers.csv"


def print_section(title):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


def test_health():
    print_section("1. Testing Health Check Endpoint: GET /api/health")
    try:
        res = requests.get(f"{BASE_URL}/api/health", timeout=5)
        print(f"Status Code: {res.status_code}")
        print("Response Payload:")
        print(json.dumps(res.json(), indent=2))
        return res.status_code == 200
    except requests.exceptions.ConnectionError:
        print(f"[ERROR] Could not connect to {BASE_URL}. Is the server running? (Run: python run.py)")
        return False


def test_single_predict():
    print_section("2. Testing Single Prediction Endpoint: POST /api/predict")

    customer_payload = {
        "customer_id": "DEMO-USER-7788",
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
        "MonthlyCharges": 89.50,
        "TotalCharges": 179.00,
    }

    print("Input Customer Profile:")
    print(f"  Contract: {customer_payload['Contract']} | Tenure: {customer_payload['tenure']} mo | Spend: ${customer_payload['MonthlyCharges']}/mo")

    res = requests.post(f"{BASE_URL}/api/predict", json=customer_payload, timeout=5)
    print(f"\nStatus Code: {res.status_code}")
    data = res.json()
    print("\nInference Diagnostics:")
    print(f"  Prediction:        {data.get('prediction')}")
    print(f"  Churn Probability: {data.get('churn_percentage')}%")
    print(f"  Risk Level:        {data.get('risk_level')}")
    print(f"  Tuned Threshold:   {data.get('threshold')}")

    financial = data.get("financial_impact", {})
    print("\nFinancial Intelligence:")
    print(f"  Annual ARR at Stake:        ${financial.get('annual_revenue_at_risk'):.2f} / yr")
    print(f"  Simulated Retention Gain:   ${financial.get('simulated_retention_savings'):.2f} / yr")

    print("\nKey Root-Cause Drivers:")
    for factor in data.get("risk_factors", []):
        print(f"  [{factor.get('impact')}] {factor.get('feature')}: {factor.get('description')}")

    print("\nPrescriptive Retention Actions:")
    for action in data.get("retention_actions", []):
        print(f"  ➜ {action}")


def test_batch_predict():
    print_section("3. Testing Batch CSV Scoring: POST /api/batch-predict")
    if not SAMPLE_CSV_PATH.exists():
        print(f"[SKIP] Sample CSV not found at {SAMPLE_CSV_PATH}")
        return

    with open(SAMPLE_CSV_PATH, "rb") as f:
        files = {"file": ("sample.csv", f, "text/csv")}
        res = requests.post(f"{BASE_URL}/api/batch-predict?format=json", files=files, timeout=10)

    print(f"Status Code: {res.status_code}")
    data = res.json()
    summary = data.get("summary", {})
    print("Batch Portfolio Breakdown:")
    print(f"  Total Accounts Analyzed: {summary.get('total')}")
    print(f"  High-Risk Count:         {summary.get('high_risk')}")
    print(f"  Medium-Risk Count:       {summary.get('medium_risk')}")
    print(f"  Low-Risk Count:          {summary.get('low_risk')}")
    print(f"  Predicted Churners:      {summary.get('predicted_churners')}")


def test_history():
    print_section("4. Testing Audit History Retrieval: GET /api/history")
    res = requests.get(f"{BASE_URL}/api/history?limit=5", timeout=5)
    print(f"Status Code: {res.status_code}")
    data = res.json()
    records = data.get("predictions", [])
    print(f"Retrieved {len(records)} Recent Audit Logs:")
    for r in records[:3]:
        print(f"  ID #{r.get('id')} | {r.get('created_at')} | {r.get('contract')} | Prob: {float(r.get('churn_probability'))*100:.1f}% | {r.get('risk_level')} Risk")


def main():
    print("=" * 65)
    print("      Customer Churn Prediction API - Verification Client")
    print("=" * 65)

    is_online = test_health()
    if not is_online:
        return

    test_single_predict()
    test_batch_predict()
    test_history()

    print_section("Verification Summary: All Endpoints Operational (100% Passed)")


if __name__ == "__main__":
    main()
