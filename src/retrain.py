"""Automated Champion-Challenger Retraining Pipeline.

Allows retraining the model on expanded or newly ingested customer data,
benchmarking challenger performance against the production champion,
and safely promoting the winning pipeline with zero downtime.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

# Ensure root is in path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    CLEANED_DATA_PATH,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.evaluate import calculate_metrics, find_optimal_threshold
from src.features import build_features
from src.train import build_candidate_pipelines, load_and_prepare_data

logger = logging.getLogger("churn_retrain")
logging.basicConfig(level=logging.INFO)


def evaluate_and_promote(
    new_data_path: Optional[Path] = None,
    min_improvement: float = -0.01,
) -> Dict[str, Any]:
    """Train challenger pipeline and compare against production champion.

    Args:
        new_data_path: Optional path to updated customer CSV dataset.
        min_improvement: Minimum ROC-AUC delta required to promote challenger.

    Returns:
        Summary report containing comparison and promotion decision.
    """
    logger.info("Initiating champion-challenger retraining evaluation...")

    # 1. Load current champion metadata
    champion_metadata = {}
    if MODEL_METADATA_PATH.exists():
        with open(MODEL_METADATA_PATH, "r") as f:
            champion_metadata = json.load(f)

    champ_auc = champion_metadata.get("test_metrics", {}).get("roc_auc", 0.84)
    champ_model_name = champion_metadata.get("model_name", "Gradient Boosting")
    logger.info(f"Current Production Champion: {champ_model_name} (ROC-AUC = {champ_auc:.4f})")

    # 2. Load dataset
    data_path = new_data_path or CLEANED_DATA_PATH
    df = pd.read_csv(data_path)
    y = (df["Churn"] == "Yes").astype(int)
    X = build_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # 3. Fit challenger using optimal pipeline
    pipelines = build_candidate_pipelines()
    challenger_pipeline = pipelines.get(champ_model_name, pipelines["Gradient Boosting"])
    challenger_pipeline.fit(X_train, y_train)

    # 4. Evaluate challenger
    y_test_proba = challenger_pipeline.predict_proba(X_test)[:, 1]
    optimal_th, _ = find_optimal_threshold(y_test.values, y_test_proba, target_recall=0.70)
    challenger_metrics = calculate_metrics(y_test.values, y_test_proba, threshold=optimal_th)

    chall_auc = challenger_metrics["roc_auc"]
    auc_diff = chall_auc - champ_auc
    logger.info(f"Challenger ROC-AUC: {chall_auc:.4f} (Delta: {auc_diff:+.4f})")

    should_promote = auc_diff >= min_improvement

    result = {
        "timestamp": datetime.now().isoformat(),
        "champion_roc_auc": round(champ_auc, 4),
        "challenger_roc_auc": round(chall_auc, 4),
        "delta": round(auc_diff, 4),
        "promoted": should_promote,
        "challenger_metrics": challenger_metrics,
    }

    if should_promote:
        logger.info("Challenger performance meets criteria. Promoting to production...")
        joblib.dump(challenger_pipeline, MODEL_PATH)

        current_ver = champion_metadata.get("model_version", "1.0.0")
        parts = current_ver.split(".")
        new_ver = f"{parts[0]}.{int(parts[1]) + 1}.0" if len(parts) == 3 else "1.1.0"

        champion_metadata["model_version"] = new_ver
        champion_metadata["trained_at"] = datetime.now().isoformat()
        champion_metadata["optimal_threshold"] = round(optimal_th, 4)
        champion_metadata["test_metrics"] = challenger_metrics

        with open(MODEL_METADATA_PATH, "w") as f:
            json.dump(champion_metadata, f, indent=2)

        result["new_version"] = new_ver
        logger.info(f"Promotion complete: Model updated to version {new_ver}.")
    else:
        logger.info("Challenger did not outperform champion. Retaining current champion.")

    return result


if __name__ == "__main__":
    summary = evaluate_and_promote()
    print(json.dumps(summary, indent=2))
