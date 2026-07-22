"""
src/xai/concentration.py
──────────────────────────────────────────────────────────────────────────────
Layer 5 of the faithfulness harness: the concentration measure.

A SUPPORTING analysis, not a headline metric. CMI responds not only to how
faithful an explanation is, but also to how CONCENTRATED a model's reliance is: a
model that depends on a few regions yields a steep degradation curve and high CMI,
while a model relying diffusely yields a shallow curve and lower CMI — regardless
of attribution quality. When comparing CMI across models of increasing complexity,
concentration lets us tell "faithfulness degraded" apart from "reliance became
more diffuse". It is reported ALONGSIDE CMI, never as a replacement.

Critical: concentration is a property of the MODEL, not of an attribution
    It is computed from the model's OWN per-region reliance — each region's true
    single-deletion impact on the prediction (the same ground-truth importance the
    layer-4 oracle used) — NOT from any XAI method's output. If it were derived
    from an attribution it would measure the attribution, not the model, and would
    be useless for disentangling the confound.

The measure: 1 − normalised Shannon entropy
    Given the per-region importances w (>= 0), form the distribution p = w / sum(w)
    over the n regions and take its Shannon entropy H = -sum p log p. Normalising by
    log(n) puts H in [0, 1] (1 = perfectly uniform). We report

        concentration = 1 - H / log(n)   in [0, 1]

        0  = maximally DIFFUSE   (reliance spread evenly across all regions)
        1  = maximally CONCENTRATED (one region carries all the reliance)

    Why entropy (vs Gini): standard and interpretable; the log(n) normalisation
    makes it comparable across grids/models with different region counts (needed
    for the cross-model CMI comparison); and it hits both extremes exactly.

Dataset-agnostic by design
    Region count comes from the grid; the number of classes is read from the
    predict_proba output width (the reliance tracks whichever class the model
    predicts on the original signal — no binary assumption). Applies unchanged to
    a longer, multi-class signal such as Sleep-EDF.
"""

from __future__ import annotations

import numpy as np

from src.xai.regions import RegionGrid
from src.xai.perturbation import perturb_region


def concentration_from_importances(importances) -> float:
    """Concentration (1 − normalised entropy) of a per-region importance vector.

    Core measure, model-free — this is what the synthetic-extremes check exercises.

    Parameters
    ----------
    importances : sequence of float, length n_regions
        Per-region importance. Magnitudes are used (abs), so signed reliance is
        fine; the sign of an individual region's effect does not change how
        *concentrated* the distribution is.

    Returns
    -------
    float in [0, 1]: 0 = maximally diffuse (uniform), 1 = maximally concentrated
    (all mass in one region). Returns np.nan if n_regions < 2 (undefined), and
    0.0 if every importance is zero (no reliance anywhere -> treated as diffuse).
    """
    w = np.abs(np.asarray(importances, dtype=float))
    n = w.size
    if n < 2:
        return float("nan")            # concentration undefined for a single region
    total = w.sum()
    if total == 0.0:
        return 0.0                     # no reliance on any region -> maximally diffuse

    p = w / total
    nz = p[p > 0]                      # 0·log0 := 0
    entropy = -np.sum(nz * np.log(nz))
    normalised_entropy = entropy / np.log(n)
    return float(1.0 - normalised_entropy)


def region_reliance(
    predict_proba,
    signal: np.ndarray,
    grid: RegionGrid,
    method: str = "zero",
    target_class: int | None = None,
) -> np.ndarray:
    """Model's true per-region reliance: single-deletion impact on the prediction.

    For each region, hide ONLY that region and measure how much the model's
    predicted-class probability drops. This is the model-level ground-truth
    importance (the same quantity the layer-4 oracle attribution used).

    Parameters
    ----------
    predict_proba : callable
        Maps a batch (n_samples, length) -> class probabilities (n_samples, n_classes).
    signal : ndarray, shape (length,)
        A single raw signal.
    grid : RegionGrid
        The layer-1 region grid.
    method : str
        Perturbation method (layer 2); "zero" by default, matching the oracle.
    target_class : int or None
        Class whose probability is tracked. If None, the class predicted on the
        ORIGINAL signal is used.

    Returns
    -------
    ndarray, shape (n_regions,): signed reliance per region (positive = hiding the
    region reduces the predicted-class probability).
    """
    signal = np.asarray(signal, dtype=float)
    p_full = predict_proba(signal[None, :])[0]
    if target_class is None:
        target_class = int(np.argmax(p_full))
    p0 = p_full[target_class]

    reliance = np.empty(grid.n_regions, dtype=float)
    for r in range(grid.n_regions):
        perturbed = perturb_region(signal, grid, r, method)
        reliance[r] = p0 - predict_proba(perturbed[None, :])[0, target_class]
    return reliance


def region_concentration(
    predict_proba,
    signal: np.ndarray,
    grid: RegionGrid,
    method: str = "zero",
    target_class: int | None = None,
) -> float:
    """Concentration of one signal's model-level per-region reliance.

    Convenience: region_reliance -> concentration_from_importances.
    Returns a float in [0, 1] (0 = diffuse, 1 = concentrated).
    """
    reliance = region_reliance(predict_proba, signal, grid, method, target_class)
    return concentration_from_importances(reliance)


def dataset_concentration(
    predict_proba,
    X: np.ndarray,
    grid: RegionGrid,
    method: str = "zero",
) -> dict:
    """Aggregate concentration across a set of signals.

    Parameters
    ----------
    predict_proba : callable  (see region_reliance)
    X : ndarray, shape (n_samples, length)
        Signals to evaluate.
    grid : RegionGrid
    method : str

    Returns
    -------
    dict with keys:
        "mean"       : float, mean per-sample concentration (the dataset figure)
        "std"        : float, standard deviation across samples
        "per_sample" : ndarray, concentration for each signal
    NaN per-sample values (undefined cases) are ignored in mean/std.
    """
    per_sample = np.array([
        region_concentration(predict_proba, s, grid, method) for s in X
    ])
    return {
        "mean": float(np.nanmean(per_sample)),
        "std": float(np.nanstd(per_sample)),
        "per_sample": per_sample,
    }
