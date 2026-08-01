"""
src/xai/integrated_gradients.py
──────────────────────────────────────────────────────────────────────────────
Integrated Gradients (IG) over temporal regions — a gradient-based attribution.

Unlike KernelSHAP / FeatureAblation (black-box, perturbation-based), IG needs the
model's GRADIENTS, so it applies to the PyTorch Models 2–5 (not the sklearn Model
1). It attributes the prediction by integrating the model's gradients along a
straight-line path from a baseline input to the actual input.

    Reference: Sundararajan, Taly & Yan (2017), "Axiomatic Attribution for Deep
    Networks" (ICML). Engine: captum.attr.IntegratedGradients (BSD-3-Clause).

Three deliberate choices (per the design brief)
    1. Baseline = the ZERO perturbation-method background (a zeros input), the same
       reference as our Zero PM. Rationale: IG's attributions are defined relative
       to a baseline; using one of our PM baselines makes IG directly comparable to
       the perturbation-based methods. On ECG200 the signals are per-sample
       z-scored (mean ≈ 0), so the zero baseline also ≈ the sample-mean baseline —
       the two PM baselines nearly coincide here. `method` selects which PM
       background to use if a different baseline is ever wanted.
    2. Pool to the grid by SUMMING per-timestep attributions within each region.
       IG satisfies COMPLETENESS: the per-timestep attributions sum to
       f(input) − f(baseline). Summing within a region therefore preserves that
       additive decomposition — each region's value is its share of the total
       output change — so summing (not averaging) is the meaningful pooling. The
       region is the reported unit; per-timestep values are only an intermediate.
    3. BatchNorm / gradient hygiene: the model is put in EVAL mode so BatchNorm
       uses fixed running statistics (in train mode the interpolated IG batch would
       share batch stats and corrupt the gradients). Attribution is computed WITH
       gradients enabled (no torch.no_grad). Callers should sanity-check the output
       is not all-zero / degenerate.

Dataset-agnostic: input length, channel count, class count are read from the model
    / signal. Multi-channel (C, T): a region spans all channels (across-channel
    design), so per-region = sum of attributions over all (channel, timestep) in
    that time segment.
"""

from __future__ import annotations

import numpy as np
import torch
from captum.attr import IntegratedGradients

from src.xai.regions import RegionGrid
from src.xai.perturbation import PERTURBATION_METHODS


def _to_model_input(signal: np.ndarray) -> np.ndarray:
    """(T,) -> (1, 1, T);  (C, T) -> (1, C, T).  Adds batch (and channel if 1-D)."""
    if signal.ndim == 1:
        return signal[None, None, :]
    return signal[None, :]


def integrated_gradients(
    model: torch.nn.Module,
    signal: np.ndarray,
    grid: RegionGrid,
    method: str = "zero",
    target_class: int | None = None,
    n_steps: int = 50,
) -> np.ndarray:
    """Per-region Integrated Gradients attribution for one signal.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model mapping (n, C, T) -> logits (n, n_classes). (Gradient-based,
        so this needs the actual torch model, not the black-box predict_proba.)
    signal : ndarray, shape (T,) or (C, T)
        A single raw signal.
    grid : RegionGrid
        The region grid over time; per-timestep IG is summed within each region.
    method : str
        PM whose background is the IG baseline ("zero" default; "sample_mean",
        "laplace" also available).
    target_class : int or None
        Class attributed. If None, the class the model predicts on the signal.
    n_steps : int
        Number of Riemann steps for the path integral (Captum default 50).

    Returns
    -------
    ndarray, shape (n_regions,): per-region IG attribution (sum of per-timestep IG
    within each time region, across all channels). Signed: positive = pushes the
    prediction toward the target class.
    """
    if method not in PERTURBATION_METHODS:
        raise ValueError(
            f"unknown method {method!r}; choose from {tuple(PERTURBATION_METHODS)}"
        )
    signal = np.asarray(signal, dtype=float)
    model.eval()                                  # BatchNorm uses running stats
    param_dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device

    background = PERTURBATION_METHODS[method](signal)
    inputs = torch.as_tensor(_to_model_input(signal), dtype=param_dtype, device=device)
    baselines = torch.as_tensor(_to_model_input(background), dtype=param_dtype, device=device)

    # Attribute the softmax PROBABILITY of the target class (same 0–1 scale the
    # perturbation methods act on), so IG is comparable to them.
    def forward_func(x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(model(x), dim=1)

    if target_class is None:
        with torch.no_grad():
            target_class = int(forward_func(inputs)[0].argmax())

    ig = IntegratedGradients(forward_func)
    attr = ig.attribute(inputs, baselines=baselines, target=int(target_class), n_steps=n_steps)
    attr = attr.detach().cpu().numpy()[0]         # (C, T) (channel dim present)

    # Sum per-timestep (and across channels) within each time region — completeness.
    return np.array([
        attr[..., grid.bounds[r, 0]:grid.bounds[r, 1]].sum()
        for r in range(grid.n_regions)
    ])
