"""Database Seeding Utility for Customer Churn Prediction System.

Populates the database with realistic historical prediction records across
diverse customer archetypes for immediate presentation and demonstration readiness.
"""

import random
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Ensure root is in path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.db import init_db, save_prediction
from src.predict import predict_one

SAMPLE_PROFILES = [
    {
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
    },
    {
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
        "MonthlyCharges": 54.20,
        "TotalCharges": 3523.00,
    },
    {
        "gender": "Male",
        "SeniorCitizen": 1,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 98.75,
        "TotalCharges": 493.75,
    },
    {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 32,
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
        "MonthlyCharges": 35.50,
        "TotalCharges": 1136.00,
    },
    {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 18,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "No",
        "OnlineSecurity": "No internet service",
        "OnlineBackup": "No internet service",
        "DeviceProtection": "No internet service",
        "TechSupport": "No internet service",
        "StreamingTV": "No internet service",
        "StreamingMovies": "No internet service",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Mailed check",
        "MonthlyCharges": 20.05,
        "TotalCharges": 360.90,
    },
]


def seed_database(count: int = 20) -> int:
    """Populate database with simulated customer prediction events."""
    print(f"[SEED] Initializing database schema...")
    init_db()

    print(f"[SEED] Generating {count} realistic prediction records...")
    seeded = 0

    for i in range(count):
        base = random.choice(SAMPLE_PROFILES).copy()
        # Add slight natural jitter
        jitter_tenure = max(1, min(72, base["tenure"] + random.randint(-2, 3)))
        jitter_monthly = max(18.5, round(base["MonthlyCharges"] + random.uniform(-5.0, 5.0), 2))
        base["tenure"] = jitter_tenure
        base["MonthlyCharges"] = jitter_monthly
        base["TotalCharges"] = round(jitter_tenure * jitter_monthly, 2)

        cust_id = f"CUST-{random.randint(1000, 9999)}"

        # Compute prediction
        pred = predict_one(base)

        saved, err = save_prediction(base, pred, customer_id=cust_id)
        if saved:
            seeded += 1

    print(f"[SEED] Successfully seeded {seeded}/{count} prediction records into database.")
    return seeded


if __name__ == "__main__":
    seed_database(25)
