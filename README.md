# Customer Churn Prediction System

An end-to-end Machine Learning intelligence platform and web application for predicting telecom customer churn, analyzing risk drivers, calculating financial revenue at risk, recommending retention actions, and logging predictions into MySQL / SQLite.

![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.14-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.9.0-orange.svg)
![Flask](https://img.shields.io/badge/Flask-3.1.3-green.svg)
![Tests](https://img.shields.io/badge/Tests-22%2F22%20Passing-brightgreen.svg)
![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.8441-blueviolet.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF.svg)

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
   - **Factor Drivers & Visual Progress Meters**: Dynamic bars highlight reasons pushing churn risk up or down.
   - **Prescriptive Retention Playbook**: Generates actionable retention strategies (e.g. promotional contract upgrade, autopay incentive).
   - **Executive PDF Dossier Export**: 1-click printable Customer Retention Assessment report.
3. **Financial & Executive Business Intelligence**:
   - **Annual ARR at Stake ($ / yr)**: Calculates risk-weighted annual revenue exposure.
   - **Retention Value Gain ($ / yr)**: Simulates estimated annual savings if the retention playbook is executed.
4. **Batch Portfolio CSV Scoring**:
   - Upload any customer CSV spreadsheet to score hundreds of customer accounts simultaneously.
   - Instant KPI risk breakdown and 1-click **Download Scored CSV**.
   - Built-in sample CSV generator (`/api/sample-csv`).
5. **Resilient Database Layer & Seeder**:
   - Production MySQL integration with automatic table creation (`sql/schema.sql`).
   - Seamless, zero-downtime **SQLite fallback** if MySQL server is offline.
   - Fully parameterized SQL queries to prevent SQL injection.
   - **Seed Script (`sql/seed_data.py`)**: Generates 25 realistic historical prediction records for immediate demonstration.
6. **Continuous Integration & Retraining**:
   - Automated Champion/Challenger Retraining Pipeline (`src/retrain.py`) with zero-downtime model promotion.
   - Complete GitHub Actions CI workflow (`.github/workflows/ci.yml`).
   - Production `Dockerfile` and `docker-compose.yml` (MySQL + Flask).
   - 100% automated test pass rate (**22/22 tests passing** via `pytest`).

---

## 📂 Project Structure

```
Customer_Churn_Prediction/
├── Dataset/
│   ├── cleaned_telco_churn.csv            # Cleaned ML-ready dataset (7,043 rows)
│   ├── sample_batch_customers.csv         # Sample batch dataset for testing
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
│   ├── predict.py                         # Inference engine, financial metrics & actions
│   └── retrain.py                         # Automated Champion-Challenger retraining
├── models/
│   ├── churn_model.joblib                 # Serialized production ML pipeline
│   ├── feature_columns.json               # Exact 30 canonical one-hot columns
│   └── model_metadata.json                # Hyperparameters, versions & metrics
├── reports/
│   ├── model_comparison.csv               # Complete metric benchmark table
│   └── figures/                           # High-res evaluation visualizations
│       ├── 01_confusion_matrix.png
│       ├── 02_roc_curves.png
│       ├── 03_precision_recall_curves.png
│       ├── 04_feature_importance.png
│       └── 05_threshold_tuning.png
├── app/
│   ├── app.py                             # Flask routes & REST endpoints
│   ├── db.py                              # MySQL connector with SQLite fallback
│   ├── templates/
│   │   ├── index.html                     # Interactive workbench simulator & batch UI
│   │   └── history.html                   # Prediction audit history table
│   └── static/
│       ├── css/style.css                  # Dark-theme B2B AI & print stylesheet
│       └── js/main.js                     # Sliders, gauge animations & batch fetch calls
├── sql/
│   ├── schema.sql                         # MySQL database schema DDL
│   └── seed_data.py                       # Database seeding utility (25 records)
├── tests/
│   ├── test_features.py                   # Feature & validation tests
│   ├── test_model.py                      # Model loading & inference tests
│   ├── test_api.py                        # REST endpoint integration tests
│   └── test_db.py                         # Persistence & retrieval tests
├── docs/
│   ├── PRD.md                             # Product Requirements Document
│   ├── API.md                             # REST API specification
│   └── TESTING.md                         # Test strategy and test results
├── .github/
│   └── workflows/
│       └── ci.yml                         # GitHub Actions CI workflow
├── Dockerfile                             # Container build file
├── docker-compose.yml                     # Docker Compose file (MySQL + Flask)
├── .dockerignore                          # Docker build ignore rules
├── conftest.py                            # Pytest root configuration
├── .env.example                           # Environment configuration template
├── requirements.txt                       # Project dependencies
├── run.py                                 # Master runner with browser auto-launch
├── start.bat                              # 1-Click Windows launcher
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
DB_TYPE=auto
```
> **Note:** If MySQL is not configured or offline, the app automatically and gracefully falls back to local SQLite (`churn_db.sqlite`) so you can run the entire application out-of-the-box!

### 3. Launch the Web Application

#### Option A: 1-Click Launcher (Windows)
Double-click [`start.bat`](start.bat) in the project folder.

#### Option B: From Command Line
```powershell
python run.py
```
> Starts server at `http://127.0.0.1:5000` and automatically opens your web browser!

#### Option C: With Docker
```bash
docker compose up --build
```

---

## 🧪 Run Automated Tests
Execute the 22 unit and integration tests:
```bash
python -m pytest -v
```

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

- `POST /api/predict` - Real-time customer churn prediction, financial ARR at stake, and explainability payload.
- `POST /api/batch-predict` - High-throughput batch CSV scoring and downloadable enriched CSV.
- `GET /api/sample-csv` - Download sample customer CSV for batch testing.
- `GET /api/history` - Historical prediction records.
- `GET /api/health` - Health check (API status, model status, database status).
- `GET /api/metrics` - Model benchmark performance and feature weights.
- `GET /api/export` - Download predictions audit history as CSV.

Detailed examples and curl requests are in [`docs/API.md`](docs/API.md).
