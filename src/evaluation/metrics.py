"""
src/evaluation/metrics.py
──────────────────────────────────────────────────────────────────────────────
Model-performance evaluation utilities.

These metrics measure *predictive* quality (how good is the model?), as opposed
to *faithfulness* quality (how good is the explanation?). Faithfulness metrics
live in src/xai/ (CMI, deletion curves).

Planned functions
    classification_report_dict(y_true, y_pred, y_prob) → dict
        Accuracy, balanced accuracy, F1, ROC-AUC in a single flat dict.
        Consistent format makes it easy to log results to results/metrics/.

    plot_confusion_matrix(y_true, y_pred, ax) → None

    plot_roc_curve(y_true, y_prob, ax, label) → None

TODO — implement as needed during model training notebooks.
"""
