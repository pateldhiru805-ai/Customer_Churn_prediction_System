# Test Strategy & Quality Assurance Report

**Project:** Customer Churn Prediction System  
**Test Framework:** Pytest  
**Coverage:** Feature transformations, Model inference, REST API, Database persistence  

---

## 1. Test Architecture

The automated test suite is organized into 4 modular test files under `tests/`:

1. **`tests/test_features.py`**:
   - `test_build_features_output_shape_and_columns`: Confirms exact 30-column matrix in canonical order.
   - `test_validate_raw_input_valid`: Tests valid payload acceptance.
   - `test_validate_raw_input_missing_field`: Confirms HTTP 400 validation for missing parameters.
   - `test_validate_raw_input_invalid_category`: Rejects invalid categorical options.
   - `test_validate_raw_input_negative_charge`: Detects negative charge anomalies.
   - `test_build_features_handles_dataframe`: Tests batch DataFrame transformations.

2. **`tests/test_model.py`**:
   - `test_model_artifacts_exist`: Validates `.joblib`, `feature_columns.json`, and `model_metadata.json`.
   - `test_load_model_returns_pipeline`: Verifies in-memory caching and loading speed.
   - `test_predict_one_output_structure`: Verifies dictionary response structure, risk levels, and bounds.
   - `test_loyal_customer_has_low_churn_risk`: Validates model domain logic on high-tenure customer profile.
   - `test_metadata_performance_target`: Verifies test ROC-AUC meets project target ($\ge 0.83$).

3. **`tests/test_api.py`**:
   - `test_health_endpoint`: Tests `/api/health` 200 response and subcomponent statuses.
   - `test_predict_endpoint_success`: Validates JSON inference and schema adherence.
   - `test_predict_endpoint_missing_field`: Validates graceful error reporting on missing fields.
   - `test_predict_endpoint_invalid_category`: Validates error handling on invalid categories.
   - `test_history_endpoint`: Tests `/api/history` pagination and retrieval.
   - `test_export_endpoint`: Confirms CSV attachment headers and content-type.
   - `test_metrics_endpoint`: Confirms model performance metrics JSON.

4. **`tests/test_db.py`**:
   - `test_init_db_and_health`: Validates auto-creation of tables and health reporting.
   - `test_save_and_retrieve_prediction`: Tests parameterized SQL insert and chronological retrieval.

---

## 2. Test Execution Summary

Executed via:
```bash
python -m pytest -v
```

**Results:**
- Total Tests: **20**
- Passed: **20 (100%)**
- Execution Duration: **2.67s**
