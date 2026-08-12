"""
src/evaluation/metrics.py
──────────────────────────────────────────────────────────────────────────────
Shared predictive-performance metrics, so every model (1–5) reports identically.

These measure *predictive* quality (how good is the model?), as opposed to
*faithfulness* quality (how good is the explanation?) — the latter lives in
src/xai/ (CMI, deletion curves). Consolidated here before Models 2–5 exist so all
five report the same numbers the same way.

Design
    ECG200 is binary and mildly imbalanced (~65/35), so accuracy alone is weak; we
    always report balanced accuracy and F1 alongside it. To stay usable on the
    multi-class datasets planned later (e.g. Sleep-EDF, 5-class), the number of
    classes is read from the labels: F1 uses the binary positive-class score when
    there are 2 classes (matching how Model 1 was reported) and macro-averaging
    otherwise; ROC-AUC (optional, needs probabilities) is binary or one-vs-rest.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)


def classification_metrics(y_true, y_pred, y_prob=None) -> dict:
    """Accuracy, balanced accuracy, and F1 (+ ROC-AUC if probabilities given).

    Parameters
    ----------
    y_true, y_pred : array-like of int, shape (n_samples,)
        True and predicted class labels.
    y_prob : array-like, shape (n_samples, n_classes), optional
        Predicted class probabilities; if given, ROC-AUC is added.

    Returns
    -------
    dict with keys: "accuracy", "balanced_accuracy", "f1" (and "roc_auc" if
    y_prob is provided). F1 is binary (positive class) for 2-class problems,
    macro-averaged otherwise — read from the number of distinct labels.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_classes = len(np.unique(y_true))
    average = "binary" if n_classes == 2 else "macro"

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average=average)),
    }

    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        if n_classes == 2:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
        else:
            metrics["roc_auc"] = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            )
    return metrics


def format_metrics(metrics: dict, prefix: str = "") -> str:
    """One-line, aligned string for printing a metrics dict."""
    order = ["accuracy", "balanced_accuracy", "f1", "roc_auc"]
    parts = [f"{k}={metrics[k]:.3f}" for k in order if k in metrics]
    return (prefix + "  ".join(parts)).strip()


def overfitting_gap(reference_accuracy: float, test_accuracy: float) -> float:
    """Overfitting indicator = (train or val) accuracy − test accuracy.

    A tracked confound for the complexity study: it is expected to GROW with
    parameter count up the model ladder, so a CMI change could partly reflect
    overfitting rather than complexity per se. Reported alongside CMI and
    concentration for every model (larger gap = more overfitting). Partly inherent
    to ECG200's small (80-sample) training set — a further reason the second
    dataset matters.
    """
    return float(reference_accuracy - test_accuracy)
