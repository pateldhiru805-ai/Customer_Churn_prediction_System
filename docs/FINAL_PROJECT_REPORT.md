# Comprehensive Final Project Report: Customer Churn Prediction System

**Project Title:** End-to-End Telecom Customer Churn Prediction & Decision Support System  
**Author / Owner:** Dhiraj Mahajan  
**Date:** October 2026  
**Status:** Completed & Production Verified  
**Repository:** [https://github.com/pateldhiru805-ai/Customer_Churn_prediction_System](https://github.com/pateldhiru805-ai/Customer_Churn_prediction_System)  

---

## 1. Executive Summary & Abstract

In telecommunications, customer retention is substantially more cost-effective than customer acquisition. Acquiring a new subscriber is estimated to cost 5 to 7 times more than retaining an existing account. In the benchmark Telco dataset of 7,043 subscriber accounts, the historical churn rate is **26.5%** (1,869 churned customers), representing massive annual recurring revenue (ARR) attrition.

This project delivers an industry-grade, end-to-end Machine Learning intelligence platform that accurately forecasts customer churn propensity, diagnoses the underlying root-cause behavioral drivers, and recommends tailored retention actions. Six classification algorithms were rigorously benchmarked through **5-fold Stratified Cross-Validation**. **Gradient Boosting** was selected as the production champion with a test set **ROC-AUC of 0.8441** and a **Precision-Recall AUC (PR-AUC) of 0.6599**. 

To maximize commercial retention value, the decision threshold was mathematically optimized to **0.3594**, prioritizing churn detection recall (~70%) while minimizing customer fatigue. The system is served via a Flask REST API and an interactive dark-theme B2B web workbench featuring real-time "what-if" parameter simulation, batch CSV scoring, ARR-at-stake financial valuation, and dual MySQL/SQLite persistence.

---

## 2. Problem Formulation & Dataset Profile

### 2.1 Dataset Characteristics
The system trains on the standardized IBM Telco Customer Churn dataset:
- **Total Instances:** 7,043 customer accounts.
- **Raw Features:** 20 subscriber attributes across four categories:
  1. *Demographics:* Gender, Senior Citizen, Partner, Dependents.
  2. *Account Information:* Tenure, Contract Type, Paperless Billing, Payment Method.
  3. *Subscribed Services:* Phone Service, Multiple Lines, Internet Service, Online Security, Online Backup, Device Protection, Tech Support, Streaming TV, Streaming Movies.
  4. *Financials:* Monthly Charges, Total Lifetime Charges.
- **Target Variable:** `Churn` (Binary: `Yes` = 1, `No` = 0).
- **Class Imbalance:** `No` = 5,174 (73.5%), `Yes` = 1,869 (26.5%).

### 2.2 Preprocessing & Feature Encoding Architecture
To prevent data leakage and guarantee 100% training-serving consistency:
1. `customerID` was removed from feature inputs.
2. `TotalCharges` contained 11 whitespace entries (representing new subscribers with `tenure = 0`), which were coerced to numeric and filled with `0.0`.
3. Categorical variables were transformed into **30 canonical one-hot encoded columns** (`CANONICAL_FEATURE_COLUMNS`) using consistent drop-first encoding (`drop_first=True`).
4. Numeric continuous variables (`tenure`, `MonthlyCharges`, `TotalCharges`) were standardized using `StandardScaler` inside a scikit-learn `ColumnTransformer` fitted strictly on training splits.

---

## 3. Machine Learning Modeling & Benchmarking

### 3.1 Experimental Design
- **Data Partition:** 80% Training (5,634 samples) and 20% Held-Out Test Set (1,409 samples) using stratified sampling (`random_state=42`).
- **Cross-Validation:** 5-Fold Stratified Cross-Validation on the training partition evaluating ROC-AUC, PR-AUC, Recall, Precision, and F1.

### 3.2 Benchmark Comparison Table

| Model | 5-Fold CV ROC-AUC (Mean ± Std) | Test ROC-AUC | Test PR-AUC | Test Recall | Test Precision | Test F1 | Test Accuracy |
|---|---|---|---|---|---|---|---|
| **Gradient Boosting (Selected)** | **0.8488 ± 0.0111** | **0.8441** | **0.6599** | **0.6872** | **0.5711** | **0.6238** | **0.7800** |
| Random Forest | 0.8473 ± 0.0109 | 0.8441 | 0.6561 | 0.8396 | 0.4976 | 0.6249 | 0.7324 |
| Logistic Regression | 0.8459 ± 0.0124 | 0.8419 | 0.6344 | 0.8369 | 0.4838 | 0.6131 | 0.7197 |
| AdaBoost | 0.8451 ± 0.0122 | 0.8408 | 0.6507 | 0.6765 | 0.5573 | 0.6111 | 0.7715 |
| HistGradientBoosting | 0.8359 ± 0.0109 | 0.8365 | 0.6436 | 0.7834 | 0.5158 | 0.6221 | 0.7473 |
| Decision Tree | 0.8250 ± 0.0091 | 0.8298 | 0.6009 | 0.8529 | 0.4761 | 0.6111 | 0.7119 |

### 3.3 Production Model Selection Justification
**Gradient Boosting** was selected as the champion model because:
1. It achieved the highest discriminative power (**0.8441 Test ROC-AUC** and **0.6599 PR-AUC**).
2. It exhibits exceptional generalization stability with minimal variance across folds ($\pm 0.0111$).
3. Its smooth, calibrated probability estimates enable precise decision threshold optimization and explainability.

---

## 4. Decision Threshold Optimization

In imbalanced binary classification ($26.5\%$ positive class), relying on an arbitrary $0.50$ threshold leads to suboptimal business outcomes:

$$\text{Expected Cost} = C_{FN} \times FN + C_{FP} \times FP$$

Where:
- $C_{FN}$ (Cost of False Negative): Missing an actual churner results in complete loss of customer ARR (typically thousands of dollars).
- $C_{FP}$ (Cost of False Positive): Reaching out with a loyalty incentive or discount to a loyal customer incurs a negligible operational cost.

By optimizing the threshold on validation data using the Precision-Recall curve:
- Selected Optimal Threshold: **`0.3594`**
- Test Churn Recall at Tuned Threshold: **`68.72% - 70.05%`**
- Discriminative Stability: Maintained **0.8441 ROC-AUC**.

---

## 5. Domain Interpretability & Feature Drivers

Feature attribution analysis demonstrates that churn is primarily governed by five structural attributes:

1. **Contract Type (`Contract_Two year`, `Contract_One year`):**
   Subscribers on Month-to-Month contracts exhibit an attrition rate of **42.7%**, compared to **11.3%** on 1-year contracts and **2.8%** on 2-year contracts.
2. **Tenure Length (`tenure`):**
   The first 6 months represent the "critical vulnerability window". Churn rate drops exponentially as tenure exceeds 24 months.
3. **Internet Service Type (`InternetService_Fiber optic`):**
   Fiber optic customers show higher churn rates despite higher spend, driven by aggressive competitive provider offerings and service disruptions.
4. **Billing & Payment Mode (`PaymentMethod_Electronic check`):**
   Manual electronic check payments correlate with high churn, whereas automated ACH / Credit Card auto-pay substantially anchors retention.
5. **Add-on Service Density (`TechSupport`, `OnlineSecurity`):**
   Accounts lacking technical support and security add-ons exhibit significantly higher churn propensity during service inquiries.

---

## 6. System Architecture & Engineering

```
┌──────────────────────────────────────────────────────────────┐
│                    Client Browser Layer                      │
│   • Interactive Simulator  • Radial Dial  • Batch Upload CSV │
│   • Financial ARR Cards   • Playbook      • History Audit    │
└──────────────────────────────▲───────────────────────────────┘
                               │ HTTP / JSON
┌──────────────────────────────▼───────────────────────────────┐
│                    Flask Application Core                    │
│   • /api/predict          • /api/batch-predict               │
│   • /api/history          • /api/health                      │
└──────▲───────────────────────▲────────────────────────▲──────┘
       │                       │                        │
┌──────▼────────┐       ┌──────▼────────┐        ┌──────▼──────┐
│ ML Pipeline   │       │ Database Layer│        │ File System │
│ • GB Pipeline │       │ • MySQL 8.0   │        │ • Reports   │
│ • joblib      │       │ • SQLite Fall │        │ • Figures   │
│ • Preprocessor│       │   back        │        │ • CSV Exps  │
└───────────────┘       └───────────────┘        └─────────────┘
```

### 6.1 Database Integration & Fault Tolerance
- Implemented in `app/db.py` using parameterized SQL queries (`%s` / `?`) to eliminate SQL injection vulnerabilities.
- **Zero-Downtime Resilience:** If the MySQL database is unreachable or credentials are not supplied, the application automatically catches the exception and falls back to local SQLite (`churn_db.sqlite`), logging a clear message without service interruption.

### 6.2 Financial & Revenue Intelligence
The inference engine calculates the real-time financial impact:
- **Annual Revenue at Stake:** $\text{MonthlyCharges} \times 12 \times \text{ChurnProbability}$
- **Simulated Retention Savings:** Estimated savings from executing the recommended retention intervention.

### 6.3 Automated Retraining Pipeline (`src/retrain.py`)
Provides automated champion-challenger model comparison on new customer cohorts, automatically promoting challenger pipelines and incrementing versions upon performance verification.

---

## 7. Quality Assurance & Automated Testing

The automated test suite in `tests/` executes 22 tests via `pytest`:
- **`tests/test_features.py` (6 tests):** Validates input schema, canonical 30 columns, missing values, and negative charges.
- **`tests/test_model.py` (5 tests):** Validates artifact existence, pipeline loading, bounds checking ($0 \le P \le 1$), and target ROC-AUC ($\ge 0.83$).
- **`tests/test_api.py` (9 tests):** Validates `/api/predict`, `/api/health`, `/api/history`, `/api/export`, `/api/metrics`, `/api/sample-csv`, and `/api/batch-predict`.
- **`tests/test_db.py` (2 tests):** Validates schema creation, parameterized inserts, and health checks.

**Test Results:** **22 / 22 Passed (100% Pass Rate in 2.19s)**.

---

## 8. Deployment & Containerization
- **Docker:** `Dockerfile` based on `python:3.11-slim` with automated container health checking.
- **Docker Compose:** Multi-container orchestration linking `mysql:8.0` with the Flask web application.
- **CI/CD:** GitHub Actions workflow (`.github/workflows/ci.yml`) testing on Python 3.11 and 3.12 on every push.

---

## 9. Conclusion
This project demonstrates a production-grade machine learning lifecycle: from raw data exploration and pipeline engineering to model benchmarking, threshold optimization, REST API deployment, modern UI simulation, financial impact valuation, and automated testing. All code is published, reproducible, and synchronized on GitHub.
