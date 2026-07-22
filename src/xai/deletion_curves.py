"""
src/xai/deletion_curves.py
──────────────────────────────────────────────────────────────────────────────
Layer 3 of the faithfulness harness: deletion / perturbation curves.

The core faithfulness test. Given an attribution (a relevance score per region),
we hide regions in order of claimed importance, cumulatively, and record how the
model's predicted-class probability changes. Two orders are built:

    MoRF (most-relevant-first)  — hide the most-relevant region first, then the
                                  next, ... A faithful attribution should make
                                  the prediction fall FAST (steep curve).
    LeRF (least-relevant-first) — hide the least-relevant region first. A faithful
                                  attribution should keep the prediction high
                                  LONGER (shallow curve).

The gap between the two curves is what layer 4 (DDS/PES/CMI) turns into a
faithfulness score.

Output format — matched to the reference CMI/DDS input (IMPORTANT)
    Layer 4 will adapt the authors' metric functions
    (utils/res_utils.py: decaying_degradation_score(morf_values, lerf_values)).
    That function expects each curve as a 1-D sequence of the PREDICTED CLASS's
    probability on a 0–100 scale, with index 0 = the unperturbed prediction, and
    MoRF/LeRF of equal length. Their code multiplies softmax probabilities by 100
    and normalises DDS by a max of 100 (res_utils.py). We therefore return curves
    already scaled to [0, 100] so they slot into their metric code unchanged.

Reference / attribution (Apache License 2.0)
    The MoRF/LeRF construction here — per-region relevance ordering, LeRF = MoRF
    reversed, cumulative perturbation, tracking the originally-predicted class's
    probability, ×100 scaling, and the 50%-of-input stop
    (n_steps = ceil(n_regions * 0.5)) — follows the authors' official code:
        repo : https://github.com/perturbationeffect/cmi-am-validation-for-dl-ts-classifiers
        file : interpret_model_regions.py  (MoRF/LeRF perturbation loop)
        paper: Šimić, Veas & Sabol (2025);  license: Apache License 2.0
    Adaptation: their loop is entangled with tsai/fastai models, torch batching,
    and MongoDB storage; we re-express the same procedure as a small, pure
    function driven by (a) the layer-1 RegionGrid for regions and (b) a generic
    predict_proba callable, so it works with our sklearn/numpy pipeline and any
    dataset. See NOTICE and DECISIONS_LOG.md.

Dataset-agnostic by design
    Signal length and region count come from the grid; the number of classes is
    read from the predict_proba output width (no binary assumption — the curve
    tracks whichever class the model predicts on the original signal); the 50%
    stop is a ratio, not a fixed step count. Applies unchanged to a longer,
    multi-class signal such as Sleep-EDF.
"""

from __future__ import annotations

import math

import numpy as np

from src.xai.regions import RegionGrid
from src.xai.perturbation import perturb_regions

# Fraction of the input that is perturbed before stopping (Šimić et al. 2025).
MAX_PERTURBATION_RATIO = 0.5


def _n_perturbation_steps(n_regions: int, max_perturbation_ratio: float) -> int:
    """Number of perturbation steps = ceil(n_regions * ratio), capped at n_regions.

    Matches the reference: perturb up to `max_perturbation_ratio` of the regions
    (rounded up), never more than all of them.
    """
    steps = math.ceil(n_regions * max_perturbation_ratio)
    return int(min(steps, n_regions))


def relevance_orders(relevance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Region orderings from a per-region relevance vector.

    Returns
    -------
    morf_order : ndarray of region ids, most-relevant first
    lerf_order : ndarray of region ids, least-relevant first (MoRF reversed)

    LeRF is exactly MoRF reversed, matching the reference (`np.flip` of the MoRF
    order) so both curves perturb the same regions in opposite order.
    """
    relevance = np.asarray(relevance, dtype=float)
    morf_order = np.argsort(relevance)[::-1]     # descending relevance
    lerf_order = morf_order[::-1]                # least-relevant first
    return morf_order, lerf_order


def _build_curve(
    predict_proba,
    signal: np.ndarray,
    grid: RegionGrid,
    order: np.ndarray,
    method: str,
    n_steps: int,
    target_class: int,
    **method_kwargs,
) -> np.ndarray:
    """Cumulative perturbation curve for one region order.

    Returns a 1-D array of length n_steps+1: index 0 is the unperturbed
    predicted-class probability (×100), then the probability after cumulatively
    hiding order[0], order[:2], ... order[:n_steps].
    """
    curve = np.empty(n_steps + 1, dtype=float)
    curve[0] = predict_proba(signal[None, :])[0, target_class] * 100.0
    for step in range(n_steps):
        ids = order[: step + 1]
        perturbed = perturb_regions(signal, grid, ids, method, **method_kwargs)
        curve[step + 1] = predict_proba(perturbed[None, :])[0, target_class] * 100.0
    return curve


def perturbation_curves(
    predict_proba,
    signal: np.ndarray,
    grid: RegionGrid,
    relevance: np.ndarray,
    method: str = "zero",
    max_perturbation_ratio: float = MAX_PERTURBATION_RATIO,
    target_class: int | None = None,
    **method_kwargs,
) -> dict:
    """Build the MoRF and LeRF perturbation curves for one signal.

    Parameters
    ----------
    predict_proba : callable
        Maps a batch of raw signals, shape (n_samples, length), to class
        probabilities, shape (n_samples, n_classes). For Model 1 this wraps
        band-power feature extraction + the logistic-regression pipeline.
    signal : ndarray, shape (length,)
        A single raw signal (per-sample z-scored, as the model expects).
    grid : RegionGrid
        The layer-1 region grid; defines the regions to hide.
    relevance : ndarray, shape (grid.n_regions,)
        Per-region attribution (higher = more relevant). For now this is a dummy
        attribution; real XAI is wired in a later layer.
    method : str
        Perturbation method name (layer 2): "zero", "sample_mean", "laplace".
    max_perturbation_ratio : float
        Fraction of regions to hide before stopping (default 0.5, per the paper).
    target_class : int or None
        Class whose probability the curves track. If None, the class the model
        predicts on the ORIGINAL (unperturbed) signal is used — matching the
        reference, which follows the originally-predicted class.
    **method_kwargs
        Forwarded to the perturbation method.

    Returns
    -------
    dict with keys:
        "MoRF"            : ndarray (n_steps+1,), predicted-class prob ×100
        "LeRF"            : ndarray (n_steps+1,), predicted-class prob ×100
        "fraction_perturbed" : ndarray (n_steps+1,), fraction of the input length
                               hidden at each step (0 at index 0) — for plotting
        "target_class"    : int, the class the curves track
        "n_regions"       : int
        "n_steps"         : int
        "method"          : str
    The "MoRF"/"LeRF" arrays are exactly the format the reference DDS function
    consumes (0–100 scale, index 0 = unperturbed, equal length).
    """
    signal = np.asarray(signal, dtype=float)
    relevance = np.asarray(relevance, dtype=float)
    if relevance.shape != (grid.n_regions,):
        raise ValueError(
            f"relevance must have shape (n_regions,)=({grid.n_regions},); "
            f"got {relevance.shape}"
        )

    if target_class is None:
        target_class = int(np.argmax(predict_proba(signal[None, :])[0]))

    n_steps = _n_perturbation_steps(grid.n_regions, max_perturbation_ratio)
    morf_order, lerf_order = relevance_orders(relevance)

    morf = _build_curve(predict_proba, signal, grid, morf_order, method,
                        n_steps, target_class, **method_kwargs)
    lerf = _build_curve(predict_proba, signal, grid, lerf_order, method,
                        n_steps, target_class, **method_kwargs)

    # Fraction of the input length hidden at each step, following the MoRF order
    # (region sizes are near-equal but not identical, so measure real coverage).
    sizes = grid.sizes
    cumulative = np.concatenate([[0], np.cumsum(sizes[morf_order][:n_steps])])
    fraction_perturbed = cumulative / grid.length

    return {
        "MoRF": morf,
        "LeRF": lerf,
        "fraction_perturbed": fraction_perturbed,
        "target_class": target_class,
        "n_regions": grid.n_regions,
        "n_steps": n_steps,
        "method": method,
    }
