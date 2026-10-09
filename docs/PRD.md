# Product Requirements Document (PRD)

## Customer Churn Prediction System
**Project Name:** Customer_Churn_Prediction  
**Owner:** Dhiraj  
**Status:** Production Ready  
**Tech Stack:** Python, Pandas, NumPy, Scikit-learn, Flask, HTML5, CSS3, JavaScript (ES6+), MySQL / SQLite

---

## 1. Executive Summary & Problem Statement
Customer attrition (churn) directly damages the revenue and valuation of telecommunication service providers. In this dataset, the historical churn rate is approximately **26.5%**, representing high potential lost Annual Recurring Revenue (ARR).

This system provides an end-to-end Machine Learning intelligence platform that:
1. Predicts whether a customer will churn based on 20 raw behavioral, contractual, and demographic features.
2. Serves predictions through a REST API and an interactive B2B workbench interface.
3. Optimizes decision thresholds to prioritize churn detection recall ($\ge 70\%$) while maintaining high precision.
4. Explains specific customer churn drivers and offers automated prescriptive retention playbooks.
5. Persists all prediction events and model runs into a relational database for auditing and tracking.

---

## 2. Core Functional Requirements

| ID | Requirement | Priority | Status |
|---|---|---|---|
| **FR-1** | Train and compare at least five classifiers with a stratified 80/20 train/test split. | Must | Completed |
| **FR-2** | Benchmark models on Accuracy, Precision, Recall, F1, ROC-AUC, and PR-AUC with 5-fold cross-validation. | Must | Completed |
| **FR-3** | Select and serialize the best performing model pipeline, feature schema, and metadata. | Must | Completed |
| **FR-4** | Expose Flask REST endpoint `POST /api/predict` returning risk label and probability. | Must | Completed |
| **FR-5** | Strict input validation with descriptive HTTP 400 error responses on invalid data. | Must | Completed |
| **FR-6** | Modern interactive Web GUI with sliders, preset profiles, and animated risk gauge. | Must | Completed |
| **FR-7** | Persistent prediction logging to relational database with parameterized queries. | Must | Completed |
| **FR-8** | History audit log page (`/history`) with live filtering, search, and CSV export. | Should | Completed |
| **FR-9** | Health check endpoint (`GET /api/health`) reporting API, model, and database state. | Should | Completed |
| **FR-10** | Metrics section displaying comparison table and publication-quality evaluation curves. | Could | Completed |
| **FR-11** | CSV export endpoint (`GET /api/export`) for prediction history. | Could | Completed |

---

## 3. Machine Learning Architecture & Benchmark Results

### Model Comparison Table
Evaluated on held-out 20% test set (1,409 customers) with 5-fold stratified cross-validation on the training set:

| Model | CV ROC-AUC (Mean ± Std) | Test ROC-AUC | Test PR-AUC | Test Recall | Test Precision | Test F1 | Test Accuracy |
|---|---|---|---|---|---|---|---|
| **Gradient Boosting** | **0.8488 ± 0.0111** | **0.8441** | **0.6599** | **0.6872** | **0.5711** | **0.6238** | **0.7800** |
| Random Forest | 0.8473 ± 0.0109 | 0.8441 | 0.6561 | 0.8396 | 0.4976 | 0.6249 | 0.7324 |
| Logistic Regression | 0.8459 ± 0.0124 | 0.8419 | 0.6344 | 0.8369 | 0.4838 | 0.6131 | 0.7197 |
| AdaBoost | 0.8451 ± 0.0122 | 0.8408 | 0.6507 | 0.6765 | 0.5573 | 0.6111 | 0.7715 |
| HistGradientBoosting | 0.8359 ± 0.0109 | 0.8365 | 0.6436 | 0.7834 | 0.5158 | 0.6221 | 0.7473 |
| Decision Tree | 0.8250 ± 0.0091 | 0.8298 | 0.6009 | 0.8529 | 0.4761 | 0.6111 | 0.7119 |

### Threshold Optimization Strategy
In retention scenarios, the financial cost of a **False Negative** (a churner leaving undetected) is significantly higher than a **False Positive** (offering a loyalty discount to an existing customer). Therefore, rather than defaulting blindly to a 0.50 threshold, the decision threshold was optimized on validation data to **0.3594**, lifting churn recall to ~70% while maintaining an ROC-AUC of **0.8441**.

---

## 4. Key Churn Drivers
1. **Contract Type**: Month-to-month contracts have the strongest correlation with customer churn (~42% churn rate vs <3% on 2-year contracts).
2. **Tenure**: Early-tenure subscribers (<6 months) are the most volatile segment.
3. **Internet Service**: Fiber optic subscribers show higher attrition due to market competition and price sensitivity.
4. **Payment Method**: Customers paying by Electronic Check churn at higher rates than automated ACH/Credit Card subscribers.
5. **Technical Support**: Lack of online security and tech support services sharply elevates churn risk.
