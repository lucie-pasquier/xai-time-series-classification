"""
src/xai/timeshap.py
──────────────────────────────────────────────────────────────────────────────
TimeSHAP wrapper for ECG200 models.

TimeSHAP extends SHAP to sequential / time-series models by decomposing
the Shapley value contribution across timesteps (and optionally events/cells).
For a univariate 1D signal it produces a per-timestep importance score.

Reference
    Bento, J., Saleiro, P., Cruz, A. F., Figueiredo, M. A. T., & Bizarro, P.
    (2021). TimeSHAP: Explaining Recurrent Models through Sequence Perturbations.
    KDD 2021.  https://github.com/feedzai/timeshap

TODO — implement wrapper after model definitions are finalised.
"""

from __future__ import annotations

import numpy as np


def explain_timeshap(
    model,
    X: np.ndarray,
    baseline: np.ndarray | None = None,
    n_samples: int = 3200,
    random_seed: int = 42,
) -> np.ndarray:
    """Compute per-timestep TimeSHAP attributions for a batch of inputs.

    Parameters
    ----------
    model : callable
        Accepts X (n_samples, n_timesteps) and returns class probabilities.
    X : ndarray, shape (n_samples, n_timesteps)
    baseline : ndarray or None
        Background reference (default: zeros, matching z-score mean).
    n_samples : int
        Number of Monte Carlo samples for Shapley estimation.
    random_seed : int

    Returns
    -------
    attributions : ndarray, shape (n_samples, n_timesteps)

    Raises
    ------
    NotImplementedError
    """
    raise NotImplementedError("TimeSHAP wrapper not yet implemented.")
