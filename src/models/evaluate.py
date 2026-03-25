"""Model evaluation module.

Provides comprehensive evaluation of the passage prediction model:
AUC-ROC, AUC-PR, Brier score, log loss, calibration analysis, SHAP-based
feature attribution, and temporal validation utilities.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.config import MODELS_DIR
from src.features.build_features import FEATURE_COLUMNS
from src.models.passage_model import PassageModel

logger = logging.getLogger(__name__)


def evaluate_model(
    model: PassageModel,
    test_df: pd.DataFrame,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run full evaluation suite on a test set.

    Computes classification metrics, calibration statistics, and
    generates evaluation plots.

    Args:
        model: Trained PassageModel instance.
        test_df: Test set feature matrix with 'enacted' target column.
        output_dir: Directory to save plots. Defaults to models/ohio/.

    Returns:
        Dict of metric name -> value.
    """
    if not model.is_fitted:
        raise RuntimeError("Model has not been trained")

    y_true = test_df["enacted"].astype(int).values
    y_prob = model.predict_proba(test_df)
    stages = model.predict_stages(test_df)

    metrics: dict[str, Any] = {}

    # --- Core metrics ---
    metrics["n_test"] = len(y_true)
    metrics["n_positive"] = int(y_true.sum())
    metrics["n_negative"] = int(len(y_true) - y_true.sum())
    metrics["base_rate"] = float(y_true.mean())

    # AUC-ROC
    if len(np.unique(y_true)) > 1:
        metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob))
        metrics["auc_pr"] = float(average_precision_score(y_true, y_prob))

        # Stage-level AUC
        y_committee = (test_df["progress"] >= 2).astype(int).values
        if len(np.unique(y_committee)) > 1:
            metrics["stage1_auc_roc"] = float(
                roc_auc_score(y_committee, stages["p_committee"])
            )
    else:
        logger.warning("Only one class in test set — AUC metrics cannot be computed")
        metrics["auc_roc"] = None
        metrics["auc_pr"] = None

    # Brier score (lower is better, 0 = perfect)
    metrics["brier_score"] = float(brier_score_loss(y_true, y_prob))

    # Log loss (needs both classes represented)
    if len(np.unique(y_true)) > 1:
        metrics["log_loss"] = float(log_loss(y_true, y_prob))
    else:
        metrics["log_loss"] = float(brier_score_loss(y_true, y_prob))

    # --- Calibration analysis ---
    calibration = compute_calibration(y_true, y_prob)
    metrics["calibration"] = calibration

    # --- Summary logging ---
    logger.info("=== Evaluation Results ===")
    logger.info("Test set: %d bills (%d enacted, %.1f%% base rate)",
                metrics["n_test"], metrics["n_positive"],
                100 * metrics["base_rate"])
    if metrics["auc_roc"] is not None:
        logger.info("AUC-ROC: %.4f", metrics["auc_roc"])
        logger.info("AUC-PR:  %.4f", metrics["auc_pr"])
    logger.info("Brier:   %.4f", metrics["brier_score"])
    logger.info("LogLoss: %.4f", metrics["log_loss"])

    # --- Save plots ---
    save_dir = output_dir or MODELS_DIR / "ohio"
    save_dir.mkdir(parents=True, exist_ok=True)

    try:
        _save_roc_curve(y_true, y_prob, save_dir / "roc_curve.png")
        _save_pr_curve(y_true, y_prob, save_dir / "pr_curve.png")
        _save_calibration_plot(calibration, save_dir / "calibration_plot.png")
        logger.info("Evaluation plots saved to %s", save_dir)
    except Exception as e:
        logger.warning("Failed to save plots: %s", e)

    return metrics


def compute_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute calibration statistics in probability bins.

    For a well-calibrated model, the average predicted probability in
    each bin should match the observed frequency of positives.

    Args:
        y_true: Binary true labels.
        y_prob: Predicted probabilities.
        n_bins: Number of probability bins.

    Returns:
        Dict with bin edges, predicted means, observed frequencies,
        and bin counts.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges[1:-1])

    bins_data = []
    for i in range(n_bins):
        mask = bin_indices == i
        count = int(mask.sum())
        if count > 0:
            pred_mean = float(y_prob[mask].mean())
            obs_freq = float(y_true[mask].mean())
        else:
            pred_mean = (bin_edges[i] + bin_edges[i + 1]) / 2
            obs_freq = 0.0

        bins_data.append({
            "bin_lower": float(bin_edges[i]),
            "bin_upper": float(bin_edges[i + 1]),
            "predicted_mean": pred_mean,
            "observed_frequency": obs_freq,
            "count": count,
        })

    # Expected Calibration Error (ECE)
    total = len(y_true)
    ece = sum(
        (b["count"] / total) * abs(b["predicted_mean"] - b["observed_frequency"])
        for b in bins_data
        if b["count"] > 0
    )

    return {
        "bins": bins_data,
        "ece": float(ece),
        "n_bins": n_bins,
    }


def compute_shap_values(
    model: PassageModel,
    df: pd.DataFrame,
    max_samples: int = 500,
) -> pd.DataFrame:
    """Compute SHAP values for feature attribution.

    Uses TreeExplainer for XGBoost, KernelExplainer for logistic regression.

    Args:
        model: Trained PassageModel.
        df: Feature matrix.
        max_samples: Max number of samples for SHAP computation.

    Returns:
        DataFrame of SHAP values with same columns as feature matrix.
    """
    import shap

    X = df[model.feature_columns].values.astype(np.float32)
    if len(X) > max_samples:
        indices = np.random.RandomState(42).choice(len(X), max_samples, replace=False)
        X = X[indices]

    # Get the base estimator from the stage 1 model
    base_model = model.stage1_model
    if hasattr(base_model, "estimator"):
        base_model = base_model.estimator
    elif hasattr(base_model, "calibrated_classifiers_"):
        base_model = base_model.calibrated_classifiers_[0].estimator

    if model.model_type == "xgboost":
        explainer = shap.TreeExplainer(base_model)
    else:
        background = shap.kmeans(X, min(50, len(X)))
        explainer = shap.KernelExplainer(base_model.predict_proba, background)

    shap_values = explainer.shap_values(X)

    # For binary classification, shap_values may be a list of 2 arrays
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # positive class

    return pd.DataFrame(shap_values, columns=model.feature_columns)


def get_bill_explanation(
    model: PassageModel,
    bill_features: dict[str, Any],
    top_n: int = 5,
) -> dict[str, Any]:
    """Generate a per-bill explanation with SHAP attribution.

    Args:
        model: Trained PassageModel.
        bill_features: Feature dict for a single bill.
        top_n: Number of top factors to return.

    Returns:
        Dict with probability, top positive/negative factors, and narrative.
    """
    import shap

    df = pd.DataFrame([bill_features])
    X = df[model.feature_columns].values.astype(np.float32)

    # Get stage probabilities
    stages = model.predict_stages(df)
    p_enacted = float(stages["p_enacted"][0])
    p_committee = float(stages["p_committee"][0])
    p_enacted_given_committee = float(stages["p_enacted_given_committee"][0])

    # SHAP for stage 1
    base_model = model.stage1_model
    if hasattr(base_model, "estimator"):
        base_model = base_model.estimator
    elif hasattr(base_model, "calibrated_classifiers_"):
        base_model = base_model.calibrated_classifiers_[0].estimator

    if model.model_type == "xgboost":
        explainer = shap.TreeExplainer(base_model)
    else:
        explainer = shap.KernelExplainer(
            base_model.predict_proba, X
        )

    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_values = shap_values[0]  # Single bill

    # Create feature attribution pairs
    attributions = list(zip(model.feature_columns, shap_values))
    attributions.sort(key=lambda x: abs(x[1]), reverse=True)

    positive_factors = [
        {"feature": f, "impact": float(v), "value": float(bill_features.get(f, 0))}
        for f, v in attributions if v > 0
    ][:top_n]

    negative_factors = [
        {"feature": f, "impact": float(v), "value": float(bill_features.get(f, 0))}
        for f, v in attributions if v < 0
    ][:top_n]

    return {
        "p_enacted": p_enacted,
        "p_committee": p_committee,
        "p_enacted_given_committee": p_enacted_given_committee,
        "positive_factors": positive_factors,
        "negative_factors": negative_factors,
        "all_attributions": [
            {"feature": f, "shap_value": float(v)} for f, v in attributions
        ],
    }


def _save_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, path: Path) -> None:
    """Save ROC curve plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(np.unique(y_true)) < 2:
        return

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Bill Passage Prediction")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_pr_curve(y_true: np.ndarray, y_prob: np.ndarray, path: Path) -> None:
    """Save Precision-Recall curve plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(np.unique(y_true)) < 2:
        return

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    base_rate = y_true.mean()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, label=f"AP = {ap:.3f}", linewidth=2)
    ax.axhline(y=base_rate, color="r", linestyle="--", alpha=0.5, label=f"Base rate = {base_rate:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — Bill Passage Prediction")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_calibration_plot(calibration: dict, path: Path) -> None:
    """Save calibration (reliability) plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins = calibration["bins"]
    predicted = [b["predicted_mean"] for b in bins if b["count"] > 0]
    observed = [b["observed_frequency"] for b in bins if b["count"] > 0]
    counts = [b["count"] for b in bins if b["count"] > 0]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot(predicted, observed, "o-", linewidth=2, markersize=8)
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax1.set_xlabel("Predicted Probability")
    ax1.set_ylabel("Observed Frequency")
    ax1.set_title(f"Calibration Plot (ECE = {calibration['ece']:.4f})")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.bar(predicted, counts, width=0.08, alpha=0.7)
    ax2.set_xlabel("Predicted Probability")
    ax2.set_ylabel("Count")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
