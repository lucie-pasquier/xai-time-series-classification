"""
src/xai/kernel_shap.py
──────────────────────────────────────────────────────────────────────────────
KernelSHAP over temporal regions — the Shapley-based through-line attribution.

The one attribution method run on all five models. Unlike FeatureAblation (which,
with a zero baseline, computes literally the oracle's single-region reliance — the
same quantity CMI's deletion curves reward, so it is a control, not an
instrument), KernelSHAP estimates Shapley values by sampling many random
COALITIONS of regions and solving a weighted least-squares. That is a genuinely
different computation from the MoRF/LeRF deletion ordering, so KernelSHAP is not
circular with the metric.

    Honest nuance: KernelSHAP still perturbs by replacing regions with a baseline
    (the same PM family the harness uses), so it is not *fully* orthogonal to
    perturbation — but it does not measure the deletion quantity itself.

Engine: Captum's KernelShap (paper-aligned, grid-native)
    Šimić et al. (2025) benchmarked KernelSHAP via captum.attr.KernelShap; we use
    the same. Captum groups features with a `feature_mask`, which is exactly our
    layer-1 RegionGrid (`grid.labels`), and replaces ablated groups with
    `baselines` (the layer-2 PM background). With return_input_shape=False it
    returns one Shapley coefficient PER REGION directly — no post-hoc pooling, the
    grid stays the single source of truth. This is the identical mechanism used in
    src/xai/feature_ablation.py; KernelSHAP is a one-symbol swap of the engine.

    Method: Lundberg & Lee (2017), "A Unified Approach to Interpreting Model
    Predictions" (KernelSHAP). Framed here as "KernelSHAP over temporal regions";
    TimeSHAP (Bento et al. 2021) — KernelSHAP adapted for sequences — was
    considered and set aside because it imposes its own event segmentation and
    cannot attribute onto our region grid (see DECISIONS_LOG.md).
    Captum: Kokhlikyan et al. (2020), https://captum.ai (BSD-3-Clause).

Stochasticity & reproducibility
    KernelSHAP SAMPLES coalitions, so results vary between runs. This function
    seeds both torch and numpy at the start of every call (default seed=0), so
    each per-sample attribution is reproducible. Using a fixed seed per call also
    means all samples share the same coalition design (common random numbers) —
    a legitimate variance-reduction choice, not a bias.

Dataset-agnostic by design
    Region grouping comes from the grid; the class is read from predict_proba's
    output width (tracks the predicted class; no binary assumption). Applies
    unchanged to a longer, multi-class signal such as Sleep-EDF.
"""

from __future__ import annotations

import numpy as np
import torch
from captum.attr import KernelShap

from src.xai.regions import RegionGrid
from src.xai.perturbation import PERTURBATION_METHODS

# Number of coalitions sampled per attribution. Captum's default (25) is coarse
# for a 10-region grid; 200 is a good stability/cost trade-off on CPU here.
DEFAULT_N_SAMPLES = 200
DEFAULT_SEED = 0


def kernel_shap(
    predict_proba,
    signal: np.ndarray,
    grid: RegionGrid,
    method: str = "zero",
    target_class: int | None = None,
    n_samples: int = DEFAULT_N_SAMPLES,
    seed: int | None = DEFAULT_SEED,
    dtype: torch.dtype = torch.float64,
) -> np.ndarray:
    """Per-region KernelSHAP attribution for one signal, via Captum.

    Parameters
    ----------
    predict_proba : callable
        Maps a batch (n_samples, length) -> class probabilities
        (n_samples, n_classes). For Model 1 this wraps band-power feature
        extraction + the logistic-regression pipeline.
    signal : ndarray, shape (length,)
        A single raw signal.
    grid : RegionGrid
        The layer-1 region grid; `grid.labels` is Captum's feature_mask.
    method : str
        Perturbation method (layer 2) whose background is the ablation baseline:
        "zero", "sample_mean", or "laplace". Matching the CMI deletion PM keeps
        attribution and evaluation consistent.
    target_class : int or None
        Class attributed. If None, the class predicted on the ORIGINAL signal.
    n_samples : int
        Number of coalitions KernelSHAP samples (higher = more stable, slower).
    seed : int or None
        Seeds torch and numpy for reproducibility (None = do not seed).
    dtype : torch.dtype
        Precision of the forward/input tensors (float64 for the Model-1 harness;
        native-torch Models 2-5 can pass float32). NB: Captum returns the Shapley
        coefficients in float32 regardless — immaterial for a stochastic estimate.

    Returns
    -------
    ndarray, shape (n_regions,): one Shapley value per region (higher = the region
    contributes more to the target class).
    """
    if method not in PERTURBATION_METHODS:
        raise ValueError(
            f"unknown method {method!r}; choose from {tuple(PERTURBATION_METHODS)}"
        )
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    signal = np.asarray(signal, dtype=float)
    p_full = predict_proba(signal[None, :])[0]
    if target_class is None:
        target_class = int(np.argmax(p_full))

    background = PERTURBATION_METHODS[method](signal)

    def forward_func(x: torch.Tensor) -> torch.Tensor:
        probs = predict_proba(x.detach().cpu().numpy())
        return torch.as_tensor(np.asarray(probs), dtype=dtype)

    explainer = KernelShap(forward_func)
    inputs = torch.as_tensor(signal[None, :], dtype=dtype)
    baselines = torch.as_tensor(background[None, :], dtype=dtype)
    feature_mask = torch.as_tensor(grid.labels[None, :], dtype=torch.long)

    attr = explainer.attribute(
        inputs,
        baselines=baselines,
        target=int(target_class),
        feature_mask=feature_mask,
        n_samples=n_samples,
        return_input_shape=False,      # -> (1, n_regions): one coeff per region
    )
    return attr.detach().cpu().numpy().ravel().astype(float)
