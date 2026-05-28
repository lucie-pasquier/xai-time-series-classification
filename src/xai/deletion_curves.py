"""
src/xai/deletion_curves.py
──────────────────────────────────────────────────────────────────────────────
Deletion / insertion curve computation for XAI faithfulness evaluation.

Deletion curves (also called AOPC curves) measure how model confidence changes
as features are progressively masked in descending order of attribution importance.
A faithful explanation should produce a steeper confidence drop than a random
feature ordering.

Insertion curves measure the same in reverse: confidence as features are
progressively *revealed*.

These curves provide visual and quantitative evidence of faithfulness and are
used as inputs to DDS / CMI (see src/xai/cmi.py).

TODO — implement after model definitions and XAI wrappers are in place.
"""

from __future__ import annotations

import numpy as np


def deletion_curve(
    model,
    X: np.ndarray,
    attributions: np.ndarray,
    n_steps: int = 10,
    baseline: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the deletion curve for a batch of samples.

    Parameters
    ----------
    model : callable
        X (n_samples, n_timesteps) → confidence scores (n_samples,).
    X : ndarray, shape (n_samples, n_timesteps)
    attributions : ndarray, shape (n_samples, n_timesteps)
    n_steps : int
        Number of deletion steps.
    baseline : float
        Replacement value for deleted features.

    Returns
    -------
    fractions : ndarray, shape (n_steps + 1,)
        Fraction of features deleted at each step (0.0 → 1.0).
    mean_confidences : ndarray, shape (n_steps + 1,)
        Mean model confidence across the batch at each deletion step.

    Raises
    ------
    NotImplementedError
    """
    raise NotImplementedError("deletion_curve not yet implemented.")


def insertion_curve(
    model,
    X: np.ndarray,
    attributions: np.ndarray,
    n_steps: int = 10,
    baseline: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the insertion curve. See deletion_curve for parameter docs.

    Raises
    ------
    NotImplementedError
    """
    raise NotImplementedError("insertion_curve not yet implemented.")
