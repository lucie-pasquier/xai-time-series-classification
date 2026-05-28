"""
src/models/transformer.py
──────────────────────────────────────────────────────────────────────────────
Model 5: Time-series Transformer for ECG200 binary classification.

Architecture sketch
    Input: (batch, 96)  →  patch / positional embedding  →  (batch, T, d_model)
    → N × TransformerEncoderLayer(d_model, nhead, dim_feedforward)
    → mean pooling over time  →  FC(2)

    Attention weights are extracted via forward hooks for the attention.py
    XAI wrapper (see src/xai/attention.py).

Design choices to finalise
    - Patch size vs. per-timestep tokenisation (96 timesteps is small enough
      for per-timestep tokens)
    - Positional encoding: sinusoidal vs learned
    - Number of layers and heads (keep small for CPU training on 100 samples)

TODO — implement in notebook 06_model_transformer.ipynb, then move the
       architecture class here.
"""

from __future__ import annotations


def build_transformer():
    """Build the Transformer model for ECG200.

    Returns
    -------
    torch.nn.Module

    Raises
    ------
    NotImplementedError
    """
    raise NotImplementedError("Transformer not yet implemented.")
