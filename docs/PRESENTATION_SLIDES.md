# Final Presentation Deck: Customer Churn Prediction System

**Presenter:** Dhiraj Mahajan  
**Subject:** End-to-End Telecom Customer Churn Prediction & Decision Support System  
**Format:** 12-Slide Executive & Technical Presentation  

---

### Slide 1: Title & Executive Introduction
- **Title:** Customer Churn Prediction System
- **Subtitle:** An AI Decision Support Platform for Telecom Retention & Revenue Protection
- **Presenter:** Dhiraj Mahajan
- **Stack:** Python, Scikit-learn, Flask, MySQL / SQLite, Docker, GitHub Actions

---

### Slide 2: Problem Statement & Motivation
- **The Challenge:** Telecom subscriber churn causes major recurring revenue loss (26.5% historical churn baseline).
- **The Economic Equation:** Customer Acquisition Cost (CAC) is 5x–7x higher than Customer Retention Cost (CRC).
- **Project Objective:** Build an end-to-end ML intelligence workbench that predicts churn, identifies root-cause risk factors, calculates revenue at risk, and prescribes retention actions.

---

### Slide 3: Dataset Architecture & Exploratory Insights
- **Dataset:** 7,043 subscriber records with 20 raw features.
- **Key EDA Findings:**
  - *Contract Impact:* Month-to-month contracts churn at ~42.7% vs 2.8% on 2-year contracts.
  - *Tenure Hazard:* New subscribers (<6 months) are the most vulnerable segment.
  - *Payment Friction:* Electronic check payment users churn at disproportionately high rates.
  - *Fiber Optic:* Premium price sensitivity drives churn unless anchored by tech support.

---

### Slide 4: Feature Pipeline & Zero Data Leakage
- **30 Canonical One-Hot Columns:** Converted using strict deterministic one-hot mapping.
- **Data Leakage Prevention:** `StandardScaler` fitted strictly inside cross-validation training folds via `ColumnTransformer`.
- **Training/Serving Fidelity:** Single `build_features()` function shared across training, API, and batch evaluation.

---

### Slide 5: Model Development & 5-Fold Cross-Validation
- **6 Algorithms Evaluated:**
  1. Gradient Boosting *(Champion)*
  2. Random Forest
  3. Logistic Regression
  4. AdaBoost
  5. HistGradientBoosting
  6. Decision Tree
- **Validation Protocol:** 5-Fold Stratified Cross-Validation + 20% Held-out Test Set.

---

### Slide 6: Benchmark Results & Winner Selection
| Model | 5-Fold CV ROC-AUC | Test ROC-AUC | Test PR-AUC | Test Recall | Test Precision |
|---|---|---|---|---|---|
| **Gradient Boosting (Winner)** | **0.8488 ± 0.0111** | **0.8441** | **0.6599** | **0.6872** | **0.5711** |
| Random Forest | 0.8473 ± 0.0109 | 0.8441 | 0.6561 | 0.8396 | 0.4976 |
| Logistic Regression | 0.8459 ± 0.0124 | 0.8419 | 0.6344 | 0.8369 | 0.4838 |

- **Winner Selection:** Gradient Boosting delivered highest discriminative power, superior probability calibration, and lowest fold-to-fold variance.

---

### Slide 7: Decision Threshold Optimization
- **The Asymmetric Cost Problem:** Missing a churner (False Negative) costs recurring revenue; a false alarm (False Positive) costs a small loyalty offer.
- **Tuned Threshold:** `0.3594` (optimized via Precision-Recall trade-off curve).
- **Result:** Lifts churn recall to ~70% while maintaining an ROC-AUC of 0.8441.

---

### Slide 8: Interactive AI Workbench & Web Application
- **Live Simulator:** Real-time parameter sliders for tenure, contract, and monthly spend.
- **Radial Risk Dial:** Animated SVG gauge with color-coded risk bands (Low, Medium, High).
- **1-Click Presets:** *🔥 High-Risk Newbie*, *⚠️ Fiber Optic At-Risk*, *🛡️ Loyal Enterprise*, *✨ Budget Basic*.
- **Executive PDF Dossier:** 1-Click printable Customer Retention Assessment report.

---

### Slide 9: Financial Intelligence & ARR Valuation
- **Translating Probability to Dollars:**
  - *Annual ARR at Stake ($/yr):* $\text{MonthlyCharges} \times 12 \times \text{ChurnProbability}$
  - *Retention Value Gain ($/yr):* Projected savings from executing retention interventions.
- **Enterprise Benefit:** Gives customer success and retention managers immediate financial justification for outreach campaigns.

---

### Slide 10: Batch Customer Portfolio Scoring
- **High-Throughput Analytics:** Upload any CSV file with customer records.
- **Instant Segmentation:** Breakdown of total records, high-risk counts, and predicted churners.
- **1-Click Export:** Download enriched CSV file with probabilities and risk tiers.

---

### Slide 11: Production Engineering & DevOps
- **Database Layer:** Parameterized MySQL queries with automatic zero-downtime SQLite fallback.
- **Automated Testing:** 22/22 unit and integration tests passing via `pytest` (100% pass rate).
- **Docker Ready:** `Dockerfile` & `docker-compose.yml` for multi-container deployment.
- **CI/CD:** GitHub Actions workflow verifying code and models on every commit.

---

### Slide 12: Summary & Impact
- **End-to-End Delivery:** From raw data exploration to production web app and Docker container.
- **High Performance:** Exceeded PRD benchmark target ($\ge 0.83$ ROC-AUC).
- **Business Ready:** Integrates explainability, financial valuation, and prescriptive retention playbooks.
- **Live Code:** Available on [GitHub](https://github.com/pateldhiru805-ai/Customer_Churn_prediction_System).
