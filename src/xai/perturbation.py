"""
src/xai/perturbation.py
──────────────────────────────────────────────────────────────────────────────
Layer 2 of the faithfulness harness: region perturbation.

The "covering-up" mechanism. Given a signal and a region of that signal (defined
by the RegionGrid from layer 1), a perturbation method (PM) replaces the region's
values with something uninformative. Later layers hide the regions an explanation
claims are important and measure whether the model's prediction changes.

Why replace rather than delete
    A neural network needs a value at every timestep — you cannot hand it a
    signal with a "hole". So "hiding" a region means overwriting it with a value
    that carries as little information as possible, and seeing how much the
    prediction moves.

Why three methods
    Every replacement value has side effects: zeros inject a hard edge, a flat
    mean erases local shape, a smooth blur leaves a plausible-looking curve. Any
    single PM can therefore flatter or penalise an explanation for the wrong
    reason. Using three PMs (Šimić, Veas & Sabol 2025, Table 4) and checking that
    conclusions hold across them guards against being fooled by one method's
    artefacts.

The three PMs (Šimić et al. 2025, Table 4)
    - zero        : replace the region with 0.
    - sample_mean : replace the region with the mean of THIS signal (sample-level,
                    not a training-set mean — matches the paper's per-sample
                    definitions and keeps each perturbation self-contained).
    - laplace     : replace the region with the second-derivative (Laplacian)
                    transform of the signal — an edge/curvature operator, NOT a
                    smoother. See the Laplace note below.

Laplace — adapted from the authors' reference implementation (Apache 2.0)
    Table 4 of the paper only names "Laplace" and its behaviour was ambiguous
    from the text, so this PM is defined by the authors' OWN source code, which
    is authoritative:
        repo   : https://github.com/perturbationeffect/cmi-am-validation-for-dl-ts-classifiers
        file   : utils/subsequence_perturbation.py  (class Laplace)
        paper  : Šimić, Veas & Sabol (2025);  license: Apache License 2.0
    Their implementation is `scipy.ndimage.laplace(original)`, i.e. convolution
    with the discrete Laplacian second-derivative kernel [1, -2, 1] using scipy's
    default boundary mode 'reflect'. This is an EDGE / CURVATURE operator (it
    highlights where the signal bends sharply and sends smooth stretches toward
    ~0), NOT the low-pass smoother a "Laplace" name might suggest.
    We replicate their behaviour exactly by calling scipy.ndimage.laplace with
    mode='reflect', computed on the whole original sample, then copying the
    region's values in (matching how they cache `laplace(self.original)` and
    slice the subsequence). See NOTICE and DECISIONS_LOG.md.
    History: an earlier version of this module WRONGLY implemented Laplace as a
    double-exponential smoothing kernel; that was corrected to the authors'
    [1, -2, 1] operator after inspecting their source.

Dataset-agnostic by design
    Nothing here hard-codes ECG200 specifics. Signal length is read from the
    signal / grid (never a literal 96), and no class count is assumed (PMs are
    label-independent). The same code applies unchanged to a longer, multi-class
    signal such as Sleep-EDF.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import laplace as _ndimage_laplace

from src.xai.regions import RegionGrid

# Canonical method names (Šimić et al. 2025, Table 4).
PM_ZERO = "zero"
PM_SAMPLE_MEAN = "sample_mean"
PM_LAPLACE = "laplace"


# ── Perturbation methods: each maps a signal → a full-length "background" ──────
# A PM returns a replacement signal of the same length; perturb_region then
# overwrites only the target region's timesteps with the background there. This
# uniform "background" interface keeps the three PMs interchangeable and makes
# multi-region perturbation (a later layer) trivial.

def zero_background(signal: np.ndarray, **_kwargs) -> np.ndarray:
    """Background of all zeros (PM: zero)."""
    return np.zeros_like(signal)


def sample_mean_background(signal: np.ndarray, **_kwargs) -> np.ndarray:
    """Background filled with the sample-level mean of `signal` (PM: sample_mean).

    Sample-level: the mean of this single signal, matching the paper's per-sample
    definitions — not a training-set statistic.
    """
    return np.full_like(signal, float(np.mean(signal)))


def laplace_background(signal: np.ndarray, **_kwargs) -> np.ndarray:
    """Background = the Laplacian (2nd-derivative) transform of `signal` (PM: laplace).

    Adapted from the authors' reference implementation (Apache 2.0):
    https://github.com/perturbationeffect/cmi-am-validation-for-dl-ts-classifiers
    — utils/subsequence_perturbation.py, class Laplace, which computes
    `scipy.ndimage.laplace(self.original)` and copies the region's slice in.
    Adaptation: we expose it as a full-length "background" (their class caches the
    same array and slices it), and pass mode='reflect' explicitly (scipy's default,
    matching their call). The operation itself — convolution with the discrete
    Laplacian [1, -2, 1], reflect boundaries — is unchanged from their code.

    NOT a smoother: this highlights curvature (sharp bends) and sends smooth
    stretches toward ~0. See the module docstring and DECISIONS_LOG.md for the
    correction history.
    """
    return _ndimage_laplace(signal, mode="reflect")


PERTURBATION_METHODS = {
    PM_ZERO: zero_background,
    PM_SAMPLE_MEAN: sample_mean_background,
    PM_LAPLACE: laplace_background,
}


def perturb_regions(
    signal: np.ndarray,
    grid: RegionGrid,
    region_ids,
    method: str = PM_ZERO,
    **method_kwargs,
) -> np.ndarray:
    """Perturb one or more regions of a single signal with a chosen PM.

    The background is computed ONCE from the ORIGINAL (unperturbed) signal and
    then written into every region in `region_ids`. This matches the reference
    implementation, whose perturber caches the background of `self.original`
    (e.g. its mean or its Laplacian) and reuses it as further regions are hidden
    cumulatively — so hiding more regions never changes the replacement values of
    the ones already hidden. This is what the deletion-curve layer (layer 3)
    needs to build cumulative MoRF/LeRF curves correctly.

    Parameters
    ----------
    signal : ndarray, shape (length,)
        A single signal. `length` is read from the signal/grid — not hard-coded.
    grid : RegionGrid
        The region grid; `grid.length` must equal `len(signal)`.
    region_ids : iterable of int
        Indices of the regions to perturb, each in [0, grid.n_regions).
    method : str
        One of PERTURBATION_METHODS ("zero", "sample_mean", "laplace").
    **method_kwargs
        Forwarded to the PM; ignored by the ones that take no arguments.

    Returns
    -------
    ndarray, shape (length,)
        A copy of `signal` with every listed region replaced by the PM's
        background (computed from the original signal).
    """
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1:
        raise ValueError(
            f"perturb_regions operates on a single 1-D signal; got shape {signal.shape}"
        )
    if signal.shape[0] != grid.length:
        raise ValueError(
            f"signal length {signal.shape[0]} != grid.length {grid.length}"
        )
    if method not in PERTURBATION_METHODS:
        raise ValueError(
            f"unknown method {method!r}; choose from {tuple(PERTURBATION_METHODS)}"
        )

    background = PERTURBATION_METHODS[method](signal, **method_kwargs)

    perturbed = signal.copy()
    for region_id in region_ids:
        if not (0 <= region_id < grid.n_regions):
            raise ValueError(
                f"region_id {region_id} out of range [0, {grid.n_regions})"
            )
        start, stop = grid.bounds[region_id]
        perturbed[start:stop] = background[start:stop]
    return perturbed


def perturb_region(
    signal: np.ndarray,
    grid: RegionGrid,
    region_id: int,
    method: str = PM_ZERO,
    **method_kwargs,
) -> np.ndarray:
    """Perturb one region of a single signal with a chosen PM.

    Thin wrapper over `perturb_regions` for the single-region case. The region is
    defined by `grid` (layer 1) — this layer consumes the grid and never redefines
    any boundaries. Operates on a single 1-D signal.

    Parameters
    ----------
    signal : ndarray, shape (length,)
        A single signal. `length` is read from the signal/grid — not hard-coded.
    grid : RegionGrid
        The region grid; `grid.length` must equal `len(signal)`.
    region_id : int
        Index of the region to perturb, in [0, grid.n_regions).
    method : str
        One of PERTURBATION_METHODS ("zero", "sample_mean", "laplace").
    **method_kwargs
        Forwarded to the PM; ignored by the ones that take no arguments.

    Returns
    -------
    ndarray, shape (length,)
        A copy of `signal` with the target region replaced by the PM's background.
    """
    return perturb_regions(signal, grid, [region_id], method, **method_kwargs)
