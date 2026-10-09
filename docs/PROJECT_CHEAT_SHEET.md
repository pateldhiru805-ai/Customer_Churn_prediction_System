# Project Defense Cheat Sheet: Top 5 Examiner Questions & Answers

**Project:** Customer Churn Prediction System  
**Owner:** Dhiraj Mahajan  
**Stack:** Python, Scikit-learn, Flask, MySQL, Docker  

Use this quick-reference guide during viva examinations, project defenses, or client demos.

---

## ❓ Question 1: "Why did you choose Gradient Boosting over Random Forest and Logistic Regression?"

**Answer:**
> "We evaluated 6 classification algorithms using 5-Fold Stratified Cross-Validation on training data and tested them on an untouched 20% held-out test set (1,409 customers).
> 
> While Random Forest and Logistic Regression were strong baselines, **Gradient Boosting** achieved:
> 1. The highest test **ROC-AUC (0.8441)** and **PR-AUC (0.6599)**.
> 2. The lowest fold-to-fold cross-validation variance ($\pm 0.0111$), demonstrating superior generalization on unseen customer cohorts.
> 3. Superior probability calibration across the entire distribution, which was essential for our decision threshold optimization and financial ARR calculations."

---

## ❓ Question 2: "How did you handle the 26.5% class imbalance in the dataset?"

**Answer:**
> "The dataset exhibits a ~3:1 imbalance (5,174 retain vs 1,869 churn). We tackled this through a 4-layer strategy:
> 1. **Stratified Splitting:** Both the 80/20 train/test partition and all 5 cross-validation folds enforced stratified sampling to maintain identical class ratios across splits.
> 2. **Algorithmic Weighting:** Models utilized class weighting (`class_weight='balanced'`) to penalize false negatives on the minority churn class.
> 3. **Metric Selection:** Rather than evaluating on Accuracy (which is misleading for imbalanced problems), we evaluated models using **ROC-AUC, Precision-Recall AUC (PR-AUC), and Churn Recall**.
> 4. **Threshold Tuning:** We tuned the decision threshold on validation data from the default 0.50 down to **0.3594**, directly lifting churn recall to ~70%."

---

## ❓ Question 3: "Why did you tune the threshold to 0.3594 instead of leaving it at 0.50?"

**Answer:**
> "In business retention, classification costs are asymmetric:
> - **Cost of False Negative ($C_{FN}$):** An actual churner leaves undetected, resulting in the total loss of the customer's Annual Recurring Revenue ($ARR = \$500 - \$1,400/yr$).
> - **Cost of False Positive ($C_{FP}$):** A loyal customer receives a proactive retention check-in or promotional discount, incurring negligible operational cost.
> 
> At a 0.50 threshold, default models miss over 40% of churners. By sweeping the Precision-Recall curve and selecting **0.3594**, we achieved a **~70% churn detection recall** while maintaining high precision and protecting gross margin."

---

## ❓ Question 4: "How do you guarantee zero data leakage between training and production inference?"

**Answer:**
> "We implemented three strict architectural controls:
> 1. **Pipeline Encapsulation:** Continuous feature scaling (`StandardScaler` on `tenure`, `MonthlyCharges`, `TotalCharges`) is packaged inside a scikit-learn `ColumnTransformer` that fits strictly on training folds and never sees test or production data in advance.
> 2. **Single Canonical Feature Function:** A single function `build_features()` in `src/features.py` converts raw customer dictionaries into the exact 30 canonical one-hot columns in the exact required order across training, unit tests, single API inference, and batch CSV scoring.
> 3. **Input Validation:** Incoming API payloads are checked against strict schema boundaries before feature generation."

---

## ❓ Question 5: "What is the tangible business ROI of this platform?"

**Answer:**
> "The platform translates raw probabilities into actionable dollar metrics:
> 1. **Annual ARR at Stake ($/yr):** Quantifies financial exposure as $\text{MonthlyCharges} \times 12 \times \text{ChurnProbability}$.
> 2. **Prescriptive Playbooks:** Rather than just predicting a label, the system prescribes targeted retention interventions (e.g., offering a 15% discount to migrate month-to-month subscribers to annual contracts, which historically reduces churn by over 75%).
> 3. **Simulated Retention Savings:** Projects net expected annual savings if the retention playbook is executed, giving customer success teams immediate business justification for outreach."

---

## 🚀 Quick Commands Reference

| Task | Command |
|---|---|
| **1-Click Launch (Windows)** | Double-click `start.bat` |
| **Terminal Launch** | `python run.py` (auto-opens `http://127.0.0.1:5000`) |
| **API Demo Client** | `python demo_client.py` |
| **Run All 22 Tests** | `python -m pytest -v` |
| **Retrain Pipeline** | `python src/retrain.py` |
| **Seed Database** | `python sql/seed_data.py` |
| **Docker Compose** | `docker compose up --build` |
