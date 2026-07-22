"""
src/xai/feature_ablation.py
──────────────────────────────────────────────────────────────────────────────
Layer 6 of the faithfulness harness: FeatureAblation as the first real attribution
method (AM), used here as a VALIDATION CONTROL for the whole pipeline.

FeatureAblation attributes importance to a region by deleting (ablating) that
region — replacing it with a baseline — and measuring how much the model's
predicted-class probability drops. That is almost exactly how CMI's deletion
curves test an explanation, so FeatureAblation should score highly by
construction; a low score means the wiring is broken, not that the method is poor.

Engine: Captum's FeatureAblation (paper-aligned)
    We use captum.attr.FeatureAblation — the same implementation the reference
    paper used — so Model 1 stays on the same code path as the PyTorch Models 2–5.
    Captum's FeatureAblation is perturbation-based (no gradients), so its
    `forward_func` can be ANY callable; we wrap our generic numpy `predict_proba`
    in a tiny torch forward function. No PyTorch model is required for Model 1
    (a scikit-learn pipeline).
    Captum: Kokhlikyan et al. (2020), https://captum.ai  (BSD-3-Clause).

Grid-native attribution (single source of truth)
    Captum groups features via a `feature_mask`; our layer-1 RegionGrid already
    provides exactly that as `grid.labels` (region id per timestep). The PM's
    background (layer 2) is Captum's `baselines`. So the attribution is produced
    per region directly — no per-timestep output pooled ad hoc.

Equivalence guarantee
    With a given PM as the baseline, Captum's grouped FeatureAblation computes,
    per region, (P_full - P_ablated) for the predicted class — identical to the
    hand-rolled `region_reliance` (src/xai/concentration.py). We assert this
    equivalence in the notebook, so Model 1 (Captum-wrapped sklearn) and Models
    2–5 (native Captum) are guaranteed consistent.

Dataset-agnostic by design
    Region grouping comes from the grid; the class is read from predict_proba's
    output width (tracks the predicted class; no binary assumption). Applies
    unchanged to a longer, multi-class signal such as Sleep-EDF.
"""

from __future__ import annotations

import numpy as np
import torch
from captum.attr import FeatureAblation

from src.xai.regions import RegionGrid
from src.xai.perturbation import PERTURBATION_METHODS


def feature_ablation(
    predict_proba,
    signal: np.ndarray,
    grid: RegionGrid,
    method: str = "zero",
    target_class: int | None = None,
    dtype: torch.dtype = torch.float64,
) -> np.ndarray:
    """Per-region FeatureAblation attribution for one signal, via Captum.

    Parameters
    ----------
    predict_proba : callable
        Maps a batch (n_samples, length) -> class probabilities
        (n_samples, n_classes). For Model 1 this wraps band-power feature
        extraction + the logistic-regression pipeline.
    signal : ndarray, shape (length,)
        A single raw signal.
    grid : RegionGrid
        The layer-1 region grid; `grid.labels` is used as Captum's feature_mask.
    method : str
        Perturbation method (layer 2) whose background is used as Captum's
        ablation baseline: "zero", "sample_mean", or "laplace". Matching the PM
        used for the CMI deletion curves keeps attribution and evaluation
        consistent.
    target_class : int or None
        Class attributed. If None, the class predicted on the ORIGINAL signal.

    Returns
    -------
    ndarray, shape (n_regions,): per-region relevance = (P_full - P_ablated) for
    the target class (higher = ablating this region hurts the prediction more).
    """
    if method not in PERTURBATION_METHODS:
        raise ValueError(
            f"unknown method {method!r}; choose from {tuple(PERTURBATION_METHODS)}"
        )
    signal = np.asarray(signal, dtype=float)

    p_full = predict_proba(signal[None, :])[0]
    if target_class is None:
        target_class = int(np.argmax(p_full))

    # Ablation baseline = the PM's background, computed from the original signal.
    background = PERTURBATION_METHODS[method](signal)

    # Wrap the numpy predict_proba as a torch forward function for Captum.
    # float64 by default so the result is bit-for-bit equal to the float64
    # harness (perturb_regions / region_reliance); a native-torch model later
    # can pass its own dtype (e.g. torch.float32).
    def forward_func(x: torch.Tensor) -> torch.Tensor:
        probs = predict_proba(x.detach().cpu().numpy())
        return torch.as_tensor(np.asarray(probs), dtype=dtype)

    ablator = FeatureAblation(forward_func)
    inputs = torch.as_tensor(signal[None, :], dtype=dtype)
    baselines = torch.as_tensor(background[None, :], dtype=dtype)
    feature_mask = torch.as_tensor(grid.labels[None, :], dtype=torch.long)

    attr = ablator.attribute(
        inputs,
        baselines=baselines,
        target=int(target_class),
        feature_mask=feature_mask,
    )
    attr = attr.detach().cpu().numpy()[0]        # (length,), equal within each region

    # Gather one value per region (grouped features share the value).
    return np.array([attr[grid.bounds[r, 0]] for r in range(grid.n_regions)])
