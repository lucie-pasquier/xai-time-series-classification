"""
src/xai/cmi.py
──────────────────────────────────────────────────────────────────────────────
Consistency-Magnitude-Index (CMI) faithfulness metric.

Reference
    Šimić, A., Veas, E., & Sabol, V. (2025). A comprehensive analysis of
    perturbation methods in explainable AI feature attribution validation for
    neural time series classifiers.

    Reference implementation:
    https://github.com/perturbationeffect/cmi-am-validation-for-dl-ts-classifiers

Definition
    CMI is the harmonic mean of two underlying metrics:

        DDS — Deletion-based Drift Score
            Measures how consistently model confidence changes as the most
            important features (ranked by attribution) are progressively
            deleted/masked. Higher DDS = attributions correctly identify
            features the model relies on.

        PES — Perturbation Effect Score
            Measures the magnitude of the confidence change relative to the
            maximum achievable change. Higher PES = attributions are not just
            directionally correct, they are proportionally informative.

    Sign handling: if either DDS or PES is negative (attributions are
    anti-correlated with model reliance), CMI uses a sign-preserving harmonic
    mean formulation described in Section 3.2 of the paper.

    Formula (positive case):
        CMI = 2 · DDS · PES / (DDS + PES)

    See paper / reference implementation for full sign-handling details.

Status
    TODO — implement after model definitions and XAI wrappers are in place.
    This module is a documented stub. Expected inputs/outputs are sketched
    below so other modules can be written against the intended API.

Planned API
    compute_cmi(
        model,          # callable: X → confidence scores
        X,              # ndarray (n_samples, n_timesteps) — normalised
        attributions,   # ndarray (n_samples, n_timesteps) — feature importances
        n_steps=10,     # number of deletion steps
        baseline=0.0,   # replacement value for deleted features
    ) -> dict[str, float]
        Returns {"CMI": float, "DDS": float, "PES": float}
"""

from __future__ import annotations

import numpy as np


def compute_cmi(
    model,
    X: np.ndarray,
    attributions: np.ndarray,
    n_steps: int = 10,
    baseline: float = 0.0,
) -> dict[str, float]:
    """Compute CMI, DDS, and PES for a set of attributions.

    Parameters
    ----------
    model : callable
        Accepts X of shape (n_samples, n_timesteps) and returns confidence
        scores of shape (n_samples,) in [0, 1].
    X : ndarray, shape (n_samples, n_timesteps)
        Per-sample z-score normalised inputs (use load_ecg200("test")).
    attributions : ndarray, shape (n_samples, n_timesteps)
        Feature importance scores from TimeSHAP / Integrated Gradients /
        attention weights. Higher absolute value = more important.
    n_steps : int
        Number of deletion steps (default 10 = delete 10 % at a time).
    baseline : float
        Value used to replace deleted features (default 0.0 = zero, which
        corresponds to the feature mean under per-sample z-scoring).

    Returns
    -------
    dict with keys "CMI", "DDS", "PES" — all floats in [-1, 1].

    Raises
    ------
    NotImplementedError
        Until this function is implemented.
    """
    raise NotImplementedError(
        "CMI is not yet implemented. "
        "See src/xai/cmi.py docstring and "
        "https://github.com/perturbationeffect/cmi-am-validation-for-dl-ts-classifiers"
    )
