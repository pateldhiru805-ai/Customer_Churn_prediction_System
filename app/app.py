"""Flask Application for Customer Churn Prediction System.

Serves the interactive web interface, REST API endpoints, model inferences,
database tracking, and performance metrics.
"""

import csv
import io
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

# Ensure project root is in sys.path and app directory does not shadow package imports
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

while str(APP_DIR) in sys.path:
    sys.path.remove(str(APP_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, Response, jsonify, render_template, request, send_from_directory

from app.db import check_db_health, get_recent_predictions, init_db, save_prediction
from src.config import (
    COMPARISON_CSV_PATH,
    FIGURES_DIR,
    MODEL_METADATA_PATH,
    REPORTS_DIR,
)
from src.features import build_features
from src.predict import load_model, predict_one

# Initialize Flask application
app = Flask(
    __name__,
    template_folder=str(APP_DIR / "templates"),
    static_folder=str(APP_DIR / "static"),
)

logger = logging.getLogger("churn_app")
logging.basicConfig(level=logging.INFO)

# Initialize database schema and pre-warm model on startup
with app.app_context():
    try:
        init_db()
        load_model()
        logger.info("Application initialized: Database verified and ML model cached in memory.")
    except Exception as init_err:
        logger.warning(f"Startup warning (will retry on request): {init_err}")


@app.route("/")
def index():
    """Render the main churn prediction simulator dashboard."""
    metadata = {}
    if MODEL_METADATA_PATH.exists():
        try:
            with open(MODEL_METADATA_PATH, "r") as f:
                metadata = json.load(f)
        except Exception:
            pass

    return render_template("index.html", metadata=metadata)


@app.route("/history")
def history_page():
    """Render the historical predictions table page."""
    predictions = get_recent_predictions(limit=100)
    return render_template("history.html", predictions=predictions)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Predict customer churn risk from raw JSON input.

    Accepts raw customer attributes, transforms features, executes model inference,
    logs the event to MySQL/SQLite, and returns a rich prediction payload.
    """
    if not request.is_json:
        return jsonify({
            "success": False,
            "error": "Request body must be valid JSON.",
        }), 400

    payload = request.get_json()
    if not payload:
        return jsonify({
            "success": False,
            "error": "Empty JSON payload received.",
        }), 400

    customer_id = payload.get("customer_id") or payload.get("customerID")

    try:
        # Run inference
        result = predict_one(payload)

        # Log prediction to database
        saved, save_err = save_prediction(payload, result, customer_id=customer_id)
        if not saved:
            logger.warning(f"Prediction made but database save encountered issue: {save_err}")

        response_data = {
            "success": True,
            "prediction": result["prediction"],
            "churn_probability": result["churn_probability"],
            "churn_percentage": result["churn_percentage"],
            "risk_level": result["risk_level"],
            "threshold": result["threshold"],
            "saved": saved,
            "save_error": save_err if not saved else None,
            "model_name": result["model_name"],
            "model_version": result["model_version"],
            "risk_factors": result["risk_factors"],
            "retention_actions": result["retention_actions"],
            "financial_impact": result.get("financial_impact", {}),
        }
        return jsonify(response_data), 200

    except ValueError as val_err:
        return jsonify({
            "success": False,
            "error": str(val_err),
        }), 400
    except Exception as exc:
        logger.error(f"Inference error: {exc}", exc_info=True)
        return jsonify({
            "success": False,
            "error": f"Internal prediction engine error: {str(exc)}",
        }), 500


@app.route("/api/history", methods=["GET"])
def api_history():
    """Return recent prediction records as JSON."""
    limit = request.args.get("limit", default=50, type=int)
    records = get_recent_predictions(limit=limit)
    return jsonify({
        "success": True,
        "count": len(records),
        "predictions": records,
    })


@app.route("/api/health", methods=["GET"])
def api_health():
    """Health check endpoint reporting API, ML model, and database status."""
    db_status = check_db_health()
    model_loaded = False
    model_name = "None"
    
    try:
        m, _, meta = load_model()
        model_loaded = m is not None
        model_name = meta.get("model_name", "Loaded")
    except Exception as e:
        model_loaded = False
        model_name = f"Error: {e}"

    is_overall_healthy = model_loaded and db_status.get("connected", False)

    return jsonify({
        "status": "healthy" if is_overall_healthy else "degraded",
        "api": "online",
        "model": {
            "loaded": model_loaded,
            "name": model_name,
        },
        "database": db_status,
    }), 200 if is_overall_healthy else 207


@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    """Retrieve model performance comparisons and metadata."""
    metadata = {}
    if MODEL_METADATA_PATH.exists():
        try:
            with open(MODEL_METADATA_PATH, "r") as f:
                metadata = json.load(f)
        except Exception:
            pass

    return jsonify({
        "success": True,
        "metadata": metadata,
    })


@app.route("/api/export", methods=["GET"])
def api_export_csv():
    """Export prediction history as a downloadable CSV file."""
    records = get_recent_predictions(limit=1000)
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        "ID",
        "Timestamp",
        "Customer ID",
        "Tenure",
        "Contract",
        "Monthly Charges",
        "Total Charges",
        "Churn Probability",
        "Prediction Label",
        "Risk Level",
        "Model Name",
    ])

    for r in records:
        writer.writerow([
            r.get("id"),
            r.get("created_at"),
            r.get("customer_id") or "N/A",
            r.get("tenure"),
            r.get("contract"),
            r.get("monthly_charges"),
            r.get("total_charges"),
            r.get("churn_probability"),
            r.get("prediction_label"),
            r.get("risk_level"),
            r.get("model_name"),
        ])

    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=churn_prediction_history.csv"},
    )


@app.route("/api/batch-predict", methods=["POST"])
def api_batch_predict():
    """Execute batch churn prediction on uploaded CSV file."""
    import pandas as pd
    import numpy as np

    if "file" not in request.files:
        return jsonify({"success": False, "error": "No CSV file uploaded under form key 'file'."}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"success": False, "error": "No selected file."}), 400

    try:
        df_input = pd.read_csv(file)
        model, feature_cols, metadata = load_model()
        threshold = float(metadata.get("optimal_threshold", 0.3594))

        X = build_features(df_input, feature_columns=feature_cols)
        probs = model.predict_proba(X)[:, 1]

        df_result = df_input.copy()
        df_result["churn_probability"] = np.round(probs, 4)
        df_result["churn_percentage"] = np.round(probs * 100, 1)
        df_result["prediction_label"] = np.where(probs >= threshold, "Likely to churn", "Likely to stay")
        df_result["risk_level"] = np.where(
            probs >= 0.65, "High", np.where(probs >= 0.35, "Medium", "Low")
        )

        if request.args.get("format") == "json":
            summary = {
                "total": int(len(df_result)),
                "high_risk": int((df_result["risk_level"] == "High").sum()),
                "medium_risk": int((df_result["risk_level"] == "Medium").sum()),
                "low_risk": int((df_result["risk_level"] == "Low").sum()),
                "predicted_churners": int((df_result["prediction_label"] == "Likely to churn").sum()),
                "records": df_result.head(50).to_dict(orient="records"),
            }
            return jsonify({"success": True, "summary": summary})

        output = io.StringIO()
        df_result.to_csv(output, index=False)
        output.seek(0)
        return Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment;filename=batch_churn_predictions.csv"},
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({"success": False, "error": f"Failed to process CSV: {str(e)}"}), 400


@app.route("/api/sample-csv", methods=["GET"])
def api_sample_csv():
    """Download sample customer CSV for testing batch predictions."""
    sample_path = Path(__file__).resolve().parent.parent / "Dataset" / "sample_batch_customers.csv"
    if sample_path.exists():
        return send_from_directory(
            sample_path.parent,
            sample_path.name,
            as_attachment=True,
            mimetype="text/csv",
        )
    return jsonify({"error": "Sample file not found"}), 404


@app.route("/reports/figures/<path:filename>")
def serve_figures(filename):
    """Serve evaluation figures for documentation and dashboard embedding."""
    return send_from_directory(FIGURES_DIR, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

