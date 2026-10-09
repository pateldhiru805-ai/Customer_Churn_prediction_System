"""Model evaluation and plotting utilities for Customer Churn Prediction.

Computes classification metrics, curves, and visualization artifacts.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

try:
    from src.config import FIGURES_DIR
except ImportError:
    from config import FIGURES_DIR

# Set clean aesthetic styling for publication-quality figures
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "figure.titlesize": 14,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
})


def calculate_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Calculate key classification metrics for churn evaluation."""
    y_pred = (y_pred_proba >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_pred_proba)),
        "pr_auc": float(average_precision_score(y_true, y_pred_proba)),
        "threshold": float(threshold),
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    target_recall: float = 0.70,
) -> Tuple[float, Dict[str, float]]:
    """Determine optimal probability threshold balancing recall >= 0.70 and precision/F1."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    # Calculate F1 for all valid thresholds
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
    
    # Find candidates that meet recall >= target_recall
    valid_indices = np.where(recalls[:-1] >= target_recall)[0]
    
    if len(valid_indices) > 0:
        # Choose the threshold among valid candidates that maximizes F1
        best_idx = valid_indices[np.argmax(f1_scores[valid_indices])]
        best_threshold = float(thresholds[best_idx])
    else:
        # Fallback to maximizing F1 directly
        best_idx = int(np.argmax(f1_scores))
        best_threshold = float(thresholds[best_idx])

    metrics = calculate_metrics(y_true, y_pred_proba, threshold=best_threshold)
    return best_threshold, metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    output_path: Path,
) -> None:
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = confusion_matrix(y_true, y_pred, normalize="true")

    fig, ax = plt.subplots(figsize=(6, 5))
    labels = [["TN", "FP"], ["FN", "TP"]]
    annotations = np.empty_like(cm, dtype=object)
    for i in range(2):
        for j in range(2):
            annotations[i, j] = f"{cm[i, j]:,}\n({cm_norm[i, j]:.1%})\n{labels[i][j]}"

    sns.heatmap(
        cm,
        annot=annotations,
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=["Stay (0)", "Churn (1)"],
        yticklabels=["Stay (0)", "Churn (1)"],
        ax=ax,
        annot_kws={"size": 11, "weight": "bold"},
    )
    ax.set_title(f"Confusion Matrix: {model_name}", pad=12, fontweight="bold")
    ax.set_xlabel("Predicted Label", labelpad=8)
    ax.set_ylabel("True Label", labelpad=8)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_roc_curves(
    models_dict: Dict[str, Tuple[np.ndarray, float]],
    y_true: np.ndarray,
    output_path: Path,
) -> None:
    """Plot overlaid ROC curves for all models."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for name, (y_proba, roc_auc) in models_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.7, label="Random Baseline (AUC = 0.50)")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)")
    ax.set_title("Receiver Operating Characteristic (ROC) Comparison", pad=12, fontweight="bold")
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_precision_recall_curves(
    models_dict: Dict[str, Tuple[np.ndarray, float]],
    y_true: np.ndarray,
    output_path: Path,
) -> None:
    """Plot Precision-Recall curves for all evaluated models."""
    fig, ax = plt.subplots(figsize=(8, 6))
    baseline = (y_true == 1).mean()

    for name, (y_proba, pr_auc) in models_dict.items():
        prec, rec, _ = precision_recall_curve(y_true, y_proba)
        ax.plot(rec, prec, lw=2, label=f"{name} (PR-AUC = {pr_auc:.3f})")

    ax.plot([0, 1], [baseline, baseline], "k--", lw=1.5, alpha=0.7, label=f"Baseline ({baseline:.1%})")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("Recall (Churn Detection Rate)")
    ax.set_ylabel("Precision (Positive Predictive Value)")
    ax.set_title("Precision-Recall (PR) Curves", pad=12, fontweight="bold")
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    model_name: str,
    output_path: Path,
    top_n: int = 15,
) -> None:
    """Plot horizontal bar chart of top feature importances or coefficients."""
    df_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    df_imp["AbsImportance"] = df_imp["Importance"].abs()
    df_imp = df_imp.sort_values(by="AbsImportance", ascending=False).head(top_n)
    df_imp = df_imp.sort_values(by="AbsImportance", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = ["#EF4444" if val > 0 else "#3B82F6" for val in df_imp["Importance"]]
    
    ax.barh(df_imp["Feature"], df_imp["Importance"], color=colors, alpha=0.85, edgecolor="none")
    ax.set_title(f"Top {top_n} Features Driving Customer Churn ({model_name})", pad=12, fontweight="bold")
    ax.set_xlabel("Impact Magnitude (Positive = Increases Churn, Negative = Reduces Churn)")
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_threshold_tuning(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    chosen_threshold: float,
    output_path: Path,
) -> None:
    """Plot Precision, Recall, and F1 across decision thresholds."""
    thresholds = np.linspace(0.1, 0.9, 100)
    precisions = []
    recalls = []
    f1s = []

    for t in thresholds:
        yp = (y_pred_proba >= t).astype(int)
        precisions.append(precision_score(y_true, yp, zero_division=0))
        recalls.append(recall_score(y_true, yp, zero_division=0))
        f1s.append(f1_score(y_true, yp, zero_division=0))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, precisions, label="Precision", color="#3B82F6", lw=2)
    ax.plot(thresholds, recalls, label="Recall", color="#EF4444", lw=2)
    ax.plot(thresholds, f1s, label="F1-Score", color="#10B981", lw=2)
    
    ax.axvline(chosen_threshold, color="#8B5CF6", linestyle="--", lw=2, label=f"Chosen Threshold ({chosen_threshold:.2f})")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Metric Score")
    ax.set_title("Threshold Optimization Trade-off Curve", pad=12, fontweight="bold")
    ax.set_xlim([0.1, 0.9])
    ax.set_ylim([0.0, 1.02])
    ax.legend(loc="best", frameon=True)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
