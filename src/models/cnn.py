"""
src/models/cnn.py
──────────────────────────────────────────────────────────────────────────────
1D CNN architectures: Shallow (Model 2), Medium (Model 3), Deep (Model 4).

All three share the same interface and differ only in depth / filter counts,
so they live in one module. A factory function build_cnn(depth) is the
intended entry point.

Architecture sketch
    Input: (batch, 1, 96)  ← channel-first for PyTorch Conv1d

    Shallow  (2 conv blocks):  [Conv→BN→ReLU→Pool] × 2 → GAP → FC(2)
    Medium   (4 conv blocks):  [Conv→BN→ReLU→Pool] × 4 → GAP → FC(2)
    Deep     (6 conv blocks):  [Conv→BN→ReLU→Pool] × 6 + residual → GAP → FC(2)

    GAP = Global Average Pooling (preserves temporal interpretability for IG)

TODO — implement after data pipeline is validated (notebook 03/04/05).
"""

from __future__ import annotations


def build_cnn(depth: str):
    """Factory function for 1D CNN models.

    Parameters
    ----------
    depth : {"shallow", "medium", "deep"}

    Returns
    -------
    torch.nn.Module

    Raises
    ------
    NotImplementedError
    """
    raise NotImplementedError(f"CNN({depth}) not yet implemented.")
