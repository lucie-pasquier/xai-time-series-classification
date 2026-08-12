"""
src/xai/attention.py
──────────────────────────────────────────────────────────────────────────────
Attention weight extraction for the Transformer model (Model 5).

Attention weights from the self-attention layers can be used directly as
per-timestep feature importances. Unlike gradient-based methods, attention
weights are a by-product of the forward pass and require no additional
computation — but their faithfulness is not guaranteed (attention ≠ explanation).

This module extracts, aggregates, and post-processes attention weights into
the same (n_samples, n_timesteps) format used by TimeSHAP and IG, enabling
direct comparison across methods with the CMI metric.

Note
    Raw attention matrices are (n_heads, n_timesteps, n_timesteps). Common
    aggregation strategies (mean over heads, rollout, gradient-weighted) are
    each implemented as separate functions so the choice is explicit in the
    experiment notebooks.

TODO — implement after the Transformer architecture is finalised.
"""

from __future__ import annotations

import numpy as np


def extract_attention_weights(
    model,
    X: np.ndarray,
    aggregation: str = "mean_heads",
) -> np.ndarray:
    """Extract and aggregate attention weights for a batch of ECG inputs.

    Parameters
    ----------
    model : torch.nn.Module
        A trained Transformer model with attention hooks registered.
    X : ndarray, shape (n_samples, n_timesteps)
    aggregation : {"mean_heads", "max_heads", "cls_token"}
        How to collapse the head dimension.
        - "mean_heads"  — average over all heads (default, most common)
        - "max_heads"   — take the maximum over heads
        - "cls_token"   — row of the [CLS] token (if architecture uses one)

    Returns
    -------
    attributions : ndarray, shape (n_samples, n_timesteps)

    Raises
    ------
    NotImplementedError
    """
    raise NotImplementedError("Attention weight extraction not yet implemented.")
