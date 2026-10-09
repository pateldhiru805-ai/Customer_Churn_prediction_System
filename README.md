# Customer Churn Prediction System

An end-to-end Machine Learning intelligence platform and web application for predicting telecom customer churn, analyzing risk drivers, recommending retention actions, and logging predictions into MySQL / SQLite.

![Python](https://img.shields.io/badge/Python-3.13%2B-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.9.0-orange.svg)
![Flask](https://img.shields.io/badge/Flask-3.1.3-green.svg)
![Tests](https://img.shields.io/badge/Tests-20%2F20%20Passing-brightgreen.svg)
![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.8441-blueviolet.svg)

---

## 🌟 Key Features

1. **High-Performance ML Pipeline**:
   - 6 candidate algorithms evaluated: **Gradient Boosting (Selected)**, Random Forest, Logistic Regression, AdaBoost, HistGradientBoosting, Decision Tree.
   - **ROC-AUC: 0.8441** on held-out test data (exceeds 0.83 PRD goal).
   - **Tuned Decision Threshold (0.3594)**: Optimizes churn recall to **~70%** to capture vulnerable customers before they cancel.
2. **Interactive AI Workbench**:
   - Modern dark-mode responsive dashboard with animated radial risk dial.
   - Live **"What-If" parameter sliders** for tenure and charges.
   - **1-Click Customer Presets**: *High-Risk Newbie*, *Fiber Optic At-Risk*, *Loyal Enterprise*, *Budget Basic*.
   - **Factor Drivers & Explainability**: Highlights reasons pushing churn risk up or down.
   - **Prescriptive Retention Playbook**: Generates actionable retention strategies (e.g. promotional contract upgrade, autopay incentive).
3. **Resilient Database Layer**:
   - Production MySQL integration with automatic table creation (`sql/schema.sql`).
   - Seamless, zero-downtime **SQLite fallback** if MySQL server credentials are offline or not running.
   - Fully parameterized SQL queries to prevent SQL injection.
4. **Audit History & Reporting**:
   - Searchable, filterable `/history` web page with CSV export capability.
   - Publication-quality evaluation plots in `reports/figures/` (Confusion matrix, ROC curves, PR curves, Feature importance, Threshold curves).
5. **Production Quality**:
   - 100% automated test pass rate (20/20 tests passing via `pytest`).
   - Dedicated REST API (`/api/predict`, `/api/history`, `/api/health`, `/api/metrics`, `/api/export`).

---

## 📂 Project Structure

```
Customer_Churn_Prediction/
├── Dataset/
│   ├── cleaned_telco_churn.csv            # Cleaned ML-ready dataset (7,043 rows)
│   └── notebooks/
│       └── 01_data_analysis.ipynb         # Original EDA and preprocessing
├── Telco-Customer-Churn_dataset.csv       # Raw Kaggle Telco dataset
├── code/                                  # Initial exploration scripts
├── graphs/                                # Initial EDA visualization plots
├── notebooks/
│   ├── 02_model_training.ipynb            # Pipeline creation & 5-fold CV comparison
│   └── 03_model_evaluation.ipynb          # Metric trade-offs & interpretability
├── src/
│   ├── config.py                          # Constants, paths, canonical 30 columns
│   ├── features.py                        # build_features() & validate_raw_input()
│   ├── train.py                           # 6-model trainer, CV, threshold tuning
│   ├── evaluate.py                        # Curves, confusion matrices & plots
│   └── predict.py                         # Inference engine, explainability & actions
├── models/
│   ├── churn_model.joblib                 # Serialized production ML pipeline
│   ├── feature_columns.json               # Exact 30 canonical one-hot columns
│   └── model_metadata.json                # Hyperparameters, versions & metrics
├── reports/
│   ├── model_comparison.csv               # Complete metric benchmark table
│   └── figures/                           # High-res evaluation visualizations
├── app/
│   ├── app.py                             # Flask routes & REST endpoints
│   ├── db.py                              # MySQL connector with SQLite fallback
│   ├── templates/
│   │   ├── index.html                     # Interactive workbench simulator
│   │   └── history.html                   # Prediction audit history table
│   └── static/
│       ├── css/style.css                  # Dark-theme B2B AI stylesheet
│       └── js/main.js                     # Sliders, gauge animations & fetch calls
├── sql/
│   └── schema.sql                         # MySQL database schema DDL
├── tests/
│   ├── test_features.py                   # Feature & validation tests
│   ├── test_model.py                      # Model loading & inference tests
│   ├── test_api.py                        # REST endpoint integration tests
│   └── test_db.py                         # Persistence & retrieval tests
├── docs/
│   ├── PRD.md                             # Product Requirements Document
│   ├── API.md                             # REST API specification
│   └── TESTING.md                         # Test strategy and test results
├── conftest.py                            # Pytest root configuration
├── .env.example                           # Environment configuration template
├── requirements.txt                       # Project dependencies
└── README.md                              # Main documentation
```

---

## 🚀 Quickstart Guide

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone <repo-url>
cd Customer_Churn_Prediction

# Install required packages
pip install -r requirements.txt
```

### 2. Configure Environment (Optional MySQL)
Copy the environment template:
```bash
cp .env.example .env
```
Edit `.env` with your MySQL database credentials:
```ini
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_password
DB_NAME=churn_db
```
> **Note:** If MySQL is not configured or offline, the app automatically and gracefully falls back to local SQLite (`churn_db.sqlite`) so you can test and run the entire application out-of-the-box!

### 3. Train Models
Run the end-to-end training and evaluation pipeline:
```bash
python src/train.py
```
This trains all 6 algorithms, runs 5-fold cross-validation, tunes decision thresholds, saves `models/churn_model.joblib`, and outputs comparison tables and figures into `reports/`.

### 4. Run Automated Tests
Execute the 20 unit and integration tests:
```bash
python -m pytest -v
```

### 5. Launch the Web Application
Start the Flask development server:
```bash
python app/app.py
```
Open your browser and navigate to:
👉 **`http://localhost:5000`**

---

## 📊 Model Benchmark Results

| Model | CV ROC-AUC (5-Fold) | Test ROC-AUC | Test PR-AUC | Test Recall | Test Precision | Test F1 |
|---|---|---|---|---|---|---|
| **Gradient Boosting (Winner)** | **0.8488 ± 0.0111** | **0.8441** | **0.6599** | **0.6872** | **0.5711** | **0.6238** |
| Random Forest | 0.8473 ± 0.0109 | 0.8441 | 0.6561 | 0.8396 | 0.4976 | 0.6249 |
| Logistic Regression | 0.8459 ± 0.0124 | 0.8419 | 0.6344 | 0.8369 | 0.4838 | 0.6131 |
| AdaBoost | 0.8451 ± 0.0122 | 0.8408 | 0.6507 | 0.6765 | 0.5573 | 0.6111 |
| HistGradientBoosting | 0.8359 ± 0.0109 | 0.8365 | 0.6436 | 0.7834 | 0.5158 | 0.6221 |
| Decision Tree | 0.8250 ± 0.0091 | 0.8298 | 0.6009 | 0.8529 | 0.4761 | 0.6111 |

---

## 📡 API Endpoints Overview

- `POST /api/predict` - Real-time customer churn prediction & explainability payload.
- `GET /api/history` - Historical prediction records.
- `GET /api/health` - Health check (API, model status, database status).
- `GET /api/metrics` - Model benchmark performance and feature weights.
- `GET /api/export` - Download predictions audit history as CSV.

Detailed examples are in [`docs/API.md`](docs/API.md).
