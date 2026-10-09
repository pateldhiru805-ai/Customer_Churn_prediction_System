# API Documentation: Customer Churn Prediction System

The prediction service exposes RESTful JSON endpoints for real-time inference, health monitoring, audit logging, and metric export.

Base URL: `http://localhost:5000`

---

## 1. POST `/api/predict`
Calculates churn probability and classification label for a customer.

### Request Headers
```http
Content-Type: application/json
```

### Request Payload Example
```json
{
  "customer_id": "CUST-1042",
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
  "TotalCharges": 179.00
}
```

### Response Example (`200 OK`)
```json
{
  "success": true,
  "prediction": "Likely to churn",
  "churn_probability": 0.7248,
  "churn_percentage": 72.5,
  "risk_level": "High",
  "threshold": 0.3594,
  "saved": true,
  "save_error": null,
  "model_name": "Gradient Boosting",
  "model_version": "1.0.0",
  "risk_factors": [
    {
      "type": "risk",
      "feature": "Contract",
      "impact": "High Risk (+)",
      "description": "Month-to-month contracts have historically highest churn rates (~42%)."
    },
    {
      "type": "risk",
      "feature": "Tenure",
      "impact": "High Risk (+)",
      "description": "Tenure of only 2 months; new customers are most vulnerable to churn."
    }
  ],
  "retention_actions": [
    "Offer a 15% promotional discount on upgrading to an Annual Contract.",
    "Provide a $10 one-time bill credit for enrolling in Auto-Pay (Bank or Credit Card)."
  ]
}
```

### Error Response (`400 Bad Request`)
Returned when schema validation fails or a field has an invalid value.
```json
{
  "success": false,
  "error": "Missing required categorical field: 'Contract'."
}
```

---

## 2. GET `/api/history`
Returns recently stored prediction audit records.

### Query Parameters
- `limit` *(optional, integer, default=50)*: Number of records to return.

### Response Example (`200 OK`)
```json
{
  "success": true,
  "count": 1,
  "predictions": [
    {
      "id": 1,
      "created_at": "2026-10-09 01:37:20",
      "customer_id": "CUST-1042",
      "tenure": 2,
      "contract": "Month-to-month",
      "monthly_charges": 89.50,
      "total_charges": 179.00,
      "churn_probability": 0.7248,
      "prediction_label": "Likely to churn",
      "risk_level": "High",
      "model_name": "Gradient Boosting",
      "model_version": "1.0.0"
    }
  ]
}
```

---

## 3. GET `/api/health`
Checks health of API, cached ML model, and database.

### Response Example (`200 OK`)
```json
{
  "api": "online",
  "database": {
    "connected": true,
    "engine": "sqlite",
    "predictions_count": 12,
    "status": "healthy"
  },
  "model": {
    "loaded": true,
    "name": "Gradient Boosting"
  },
  "status": "healthy"
}
```

---

## 4. GET `/api/metrics`
Returns benchmark model comparison statistics and feature weights.

---

## 5. GET `/api/export`
Exports prediction audit history as a downloadable CSV stream (`text/csv`).
