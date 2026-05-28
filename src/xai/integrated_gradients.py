"""
src/xai/integrated_gradients.py
──────────────────────────────────────────────────────────────────────────────
Integrated Gradients (IG) wrapper using Captum.

IG computes feature attributions by integrating the model's gradients along
a straight-line path from a baseline input to the actual input. For ECG
signals this yields a per-timestep importance score that sums to the
difference in model output between input and baseline (completeness axiom).

Reference
    Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic Attribution for
    Deep Networks. ICML 2017.

Implementation
    Uses captum.attr.IntegratedGradients.
    https://captum.ai/docs/integrated_gradients

TODO — implement after PyTorch model definitions are finalised.
"""

from __future__ import annotations

import numpy as np


def explain_integrated_gradients(
    model,
    X: np.ndarray,
    baseline: np.ndarray | None = None,
    n_steps: int = 50,
    internal_batch_size: int = 32,
) -> np.ndarray:
    """Compute Integrated Gradients attributions for a batch of ECG inputs.

    Parameters
    ----------
    model : torch.nn.Module
        A trained PyTorch classifier with a differentiable forward pass.
    X : ndarray, shape (n_samples, n_timesteps)
        Per-sample z-score normalised inputs.
    baseline : ndarray or None
        Baseline input (default: zero vector, i.e. the normalised mean).
    n_steps : int
        Number of Riemann steps for the integral approximation (default 50).
    internal_batch_size : int
        Batch size for Captum's internal gradient accumulation.

    Returns
    -------
    attributions : ndarray, shape (n_samples, n_timesteps)

    Raises
    ------
    NotImplementedError
    """
    raise NotImplementedError("Integrated Gradients wrapper not yet implemented.")
