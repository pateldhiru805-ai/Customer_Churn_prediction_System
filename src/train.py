"""Machine Learning Training and Model Selection Pipeline for Customer Churn Prediction.

Trains multiple classification algorithms, evaluates cross-validation & test metrics,
optimizes decision thresholds, saves the best model artifacts, and generates visual reports.
"""

import json
import platform
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    from src.config import (
        CLEANED_DATA_PATH,
        COMPARISON_CSV_PATH,
        CV_FOLDS,
        DEFAULT_CHURN_THRESHOLD,
        FEATURE_COLUMNS_PATH,
        FIGURES_DIR,
        MODEL_METADATA_PATH,
        MODEL_PATH,
        MODELS_DIR,
        NUMERIC_COLUMNS,
        RANDOM_STATE,
        REPORTS_DIR,
        TEST_SIZE,
    )
    from src.evaluate import (
        calculate_metrics,
        find_optimal_threshold,
        plot_confusion_matrix,
        plot_feature_importance,
        plot_precision_recall_curves,
        plot_roc_curves,
        plot_threshold_tuning,
    )
    from src.features import build_features
except ImportError:
    from config import (
        CLEANED_DATA_PATH,
        COMPARISON_CSV_PATH,
        CV_FOLDS,
        DEFAULT_CHURN_THRESHOLD,
        FEATURE_COLUMNS_PATH,
        FIGURES_DIR,
        MODEL_METADATA_PATH,
        MODEL_PATH,
        MODELS_DIR,
        NUMERIC_COLUMNS,
        RANDOM_STATE,
        REPORTS_DIR,
        TEST_SIZE,
    )
    from evaluate import (
        calculate_metrics,
        find_optimal_threshold,
        plot_confusion_matrix,
        plot_feature_importance,
        plot_precision_recall_curves,
        plot_roc_curves,
        plot_threshold_tuning,
    )
    from features import build_features


def load_and_prepare_data() -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Load cleaned churn data and construct X (30 features) and y (0/1)."""
    print(f"Loading cleaned data from: {CLEANED_DATA_PATH}")
    df = pd.read_csv(CLEANED_DATA_PATH)
    
    # Binary target encoding: Yes -> 1, No -> 0
    y = (df["Churn"] == "Yes").astype(int)

    # Build exact canonical 30 one-hot encoded features
    X = build_features(df)
    feature_columns = list(X.columns)

    print(f"Loaded dataset: X shape = {X.shape}, y distribution = {y.value_counts().to_dict()}")
    return X, y, feature_columns


def build_candidate_pipelines() -> Dict[str, Pipeline]:
    """Build preprocessing and classifier pipelines for candidate models."""
    # Preprocessor: StandardScaler on numeric columns, passthrough for one-hot indicators
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLUMNS),
        ],
        remainder="passthrough",
    )

    models = {
        "Logistic Regression": Pipeline([
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    C=0.5,
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                    solver="lbfgs",
                ),
            ),
        ]),
        "Decision Tree": Pipeline([
            ("preprocessor", preprocessor),
            (
                "classifier",
                DecisionTreeClassifier(
                    max_depth=5,
                    min_samples_leaf=20,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=250,
                    max_depth=8,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]),
        "Gradient Boosting": Pipeline([
            ("preprocessor", preprocessor),
            (
                "classifier",
                GradientBoostingClassifier(
                    n_estimators=150,
                    learning_rate=0.08,
                    max_depth=3,
                    subsample=0.85,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]),
        "HistGradientBoosting": Pipeline([
            ("preprocessor", preprocessor),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    max_iter=150,
                    learning_rate=0.08,
                    max_leaf_nodes=31,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]),
        "AdaBoost": Pipeline([
            ("preprocessor", preprocessor),
            (
                "classifier",
                AdaBoostClassifier(
                    n_estimators=100,
                    learning_rate=0.5,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]),
    }

    return models


def run_cross_validation(
    models: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Dict[str, Dict[str, float]]:
    """Run 5-fold stratified cross-validation on the training set."""
    print("\n--- Running 5-Fold Stratified Cross-Validation on Training Data ---")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_summary = {}

    scoring = ["roc_auc", "average_precision", "recall", "precision", "f1", "accuracy"]

    for name, pipeline in models.items():
        print(f"Evaluating {name}...")
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )
        cv_summary[name] = {
            "cv_roc_auc_mean": float(np.mean(scores["test_roc_auc"])),
            "cv_roc_auc_std": float(np.std(scores["test_roc_auc"])),
            "cv_pr_auc_mean": float(np.mean(scores["test_average_precision"])),
            "cv_recall_mean": float(np.mean(scores["test_recall"])),
            "cv_precision_mean": float(np.mean(scores["test_precision"])),
            "cv_f1_mean": float(np.mean(scores["test_f1"])),
            "cv_accuracy_mean": float(np.mean(scores["test_accuracy"])),
        }
        print(
            f"  -> CV ROC-AUC: {cv_summary[name]['cv_roc_auc_mean']:.4f} (+/- {cv_summary[name]['cv_roc_auc_std']:.4f}) | "
            f"Recall: {cv_summary[name]['cv_recall_mean']:.4f} | F1: {cv_summary[name]['cv_f1_mean']:.4f}"
        )

    return cv_summary


def train_and_evaluate_all() -> None:
    """Execute end-to-end training, model selection, evaluation, and serialization."""
    # Ensure directories exist
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    X, y, feature_columns = load_and_prepare_data()

    # 2. Stratified train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

    # 3. Build candidate models
    models = build_candidate_pipelines()

    # 4. Cross-validation
    cv_results = run_cross_validation(models, X_train, y_train)

    # 5. Fit each model on full training set and evaluate on test set
    print("\n--- Evaluating Models on Held-out Test Set (20%) ---")
    test_results = []
    roc_curves_dict = {}
    pr_curves_dict = {}
    fitted_models = {}

    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        fitted_models[name] = pipeline

        # Predict probabilities
        y_test_proba = pipeline.predict_proba(X_test)[:, 1]
        metrics = calculate_metrics(y_test.values, y_test_proba, threshold=DEFAULT_CHURN_THRESHOLD)

        roc_curves_dict[name] = (y_test_proba, metrics["roc_auc"])
        pr_curves_dict[name] = (y_test_proba, metrics["pr_auc"])

        row = {
            "Model": name,
            "CV ROC-AUC (Mean)": round(cv_results[name]["cv_roc_auc_mean"], 4),
            "CV ROC-AUC (Std)": round(cv_results[name]["cv_roc_auc_std"], 4),
            "CV Recall (Mean)": round(cv_results[name]["cv_recall_mean"], 4),
            "CV F1 (Mean)": round(cv_results[name]["cv_f1_mean"], 4),
            "Test ROC-AUC": round(metrics["roc_auc"], 4),
            "Test PR-AUC": round(metrics["pr_auc"], 4),
            "Test Accuracy": round(metrics["accuracy"], 4),
            "Test Precision": round(metrics["precision"], 4),
            "Test Recall": round(metrics["recall"], 4),
            "Test F1": round(metrics["f1"], 4),
        }
        test_results.append(row)
        print(
            f"{name:25s} | Test ROC-AUC: {metrics['roc_auc']:.4f} | PR-AUC: {metrics['pr_auc']:.4f} | "
            f"Recall: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f}"
        )

    # Save comparison table
    df_comparison = pd.DataFrame(test_results).sort_values(by="Test ROC-AUC", ascending=False)
    df_comparison.to_csv(COMPARISON_CSV_PATH, index=False)
    print(f"\nSaved model comparison table to: {COMPARISON_CSV_PATH}")

    # 6. Select Best Model based on PRD Criteria
    # Rule: Highest Test & CV ROC-AUC with solid churn recall & generalization.
    best_model_name = df_comparison.iloc[0]["Model"]
    print(f"\n[WINNER] Best Model Selected: '{best_model_name}'")
    best_pipeline = fitted_models[best_model_name]

    # 7. Tune Decision Threshold on training CV/validation split
    # Split train into train_sub and val to tune threshold without test leakage
    X_tr_sub, X_val, y_tr_sub, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=RANDOM_STATE, stratify=y_train
    )
    val_pipeline = build_candidate_pipelines()[best_model_name]
    val_pipeline.fit(X_tr_sub, y_tr_sub)
    y_val_proba = val_pipeline.predict_proba(X_val)[:, 1]
    
    optimal_threshold, val_metrics = find_optimal_threshold(
        y_val.values, y_val_proba, target_recall=0.70
    )
    print(f"Optimal Decision Threshold Tuned: {optimal_threshold:.3f}")
    print(f"Validation Metrics at Tuned Threshold: {val_metrics}")

    # Evaluate best model on test set with tuned threshold
    y_best_test_proba = best_pipeline.predict_proba(X_test)[:, 1]
    best_test_metrics = calculate_metrics(
        y_test.values, y_best_test_proba, threshold=optimal_threshold
    )
    print(f"Winner Test Metrics at Tuned Threshold: {best_test_metrics}")

    # 8. Generate Visualizations
    print("\nGenerating evaluation figures...")
    plot_roc_curves(roc_curves_dict, y_test.values, FIGURES_DIR / "02_roc_curves.png")
    plot_precision_recall_curves(pr_curves_dict, y_test.values, FIGURES_DIR / "03_precision_recall_curves.png")

    y_best_pred = (y_best_test_proba >= optimal_threshold).astype(int)
    plot_confusion_matrix(
        y_test.values,
        y_best_pred,
        model_name=f"{best_model_name} (Threshold {optimal_threshold:.2f})",
        output_path=FIGURES_DIR / "01_confusion_matrix.png",
    )

    plot_threshold_tuning(
        y_test.values,
        y_best_test_proba,
        chosen_threshold=optimal_threshold,
        output_path=FIGURES_DIR / "05_threshold_tuning.png",
    )

    # Extract feature importance or coefficients
    classifier = best_pipeline.named_steps["classifier"]
    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
        plot_feature_importance(
            feature_columns,
            importances,
            model_name=best_model_name,
            output_path=FIGURES_DIR / "04_feature_importance.png",
        )
    elif hasattr(classifier, "coef_"):
        importances = classifier.coef_[0]
        plot_feature_importance(
            feature_columns,
            importances,
            model_name=best_model_name,
            output_path=FIGURES_DIR / "04_feature_importance.png",
        )

    # 9. Retrain best model on full training set or full dataset
    # According to best practice, fit on entire training set (or full dataset) with same pipeline
    print(f"Serializing best model pipeline to: {MODEL_PATH}")
    joblib.dump(best_pipeline, MODEL_PATH)

    # Save feature columns JSON
    with open(FEATURE_COLUMNS_PATH, "w") as f:
        json.dump(feature_columns, f, indent=2)
    print(f"Saved feature columns list to: {FEATURE_COLUMNS_PATH}")

    # Extract top feature weights for web UI factor impact display
    feature_impact_summary = []
    if hasattr(classifier, "feature_importances_"):
        raw_imp = classifier.feature_importances_
        for feat, imp in zip(feature_columns, raw_imp):
            feature_impact_summary.append({"feature": feat, "weight": float(imp)})
    elif hasattr(classifier, "coef_"):
        raw_imp = classifier.coef_[0]
        for feat, imp in zip(feature_columns, raw_imp):
            feature_impact_summary.append({"feature": feat, "weight": float(imp)})
    
    # Sort by absolute impact
    feature_impact_summary.sort(key=lambda x: abs(x["weight"]), reverse=True)

    # Save Model Metadata
    metadata = {
        "model_name": best_model_name,
        "model_version": "1.0.0",
        "trained_at": datetime.now().isoformat(),
        "optimal_threshold": round(optimal_threshold, 4),
        "default_threshold": DEFAULT_CHURN_THRESHOLD,
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "dataset_rows": len(X),
        "feature_count": len(feature_columns),
        "test_metrics": best_test_metrics,
        "all_model_comparison": df_comparison.to_dict(orient="records"),
        "top_features": feature_impact_summary[:10],
    }
    with open(MODEL_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved model metadata to: {MODEL_METADATA_PATH}")

    print("\n[SUCCESS] Training and evaluation completed successfully!")


if __name__ == "__main__":
    train_and_evaluate_all()
