"""
src/xai/regions.py
──────────────────────────────────────────────────────────────────────────────
Region grid — the single source of truth for how a time-series signal is
divided into contiguous regions for faithfulness evaluation.

Why this module exists
    The faithfulness harness has two sides that MUST agree on the same
    partition of the signal:
        - the XAI side produces relevance/importance on this grid, and
        - the CMI side perturbs (masks) this grid.
    If the two sides ever computed the grid differently, results would be
    silently corrupted. So the grid is defined here, once, and every other part
    of the harness imports it rather than re-deriving it.

    This module builds ONLY the grid. No averaging, perturbation, deletion
    curves, CMI, or XAI logic lives here (those are later layers). Note: the
    existing per-timestep stubs in cmi.py / deletion_curves.py predate this grid
    and will later be refactored to consume it.

Region size convention (data-justified for ECG200)
    Region size is expressed as a PERCENTAGE of the input length. Two sizes are
    used across the harness:
        - primary   : 10 %   (data-justified anchor; ≈ the signal's ~10-timestep
                              autocorrelation length — see notebook 01b)
        - secondary : 5  %   (fine grid; ≈5 timesteps — a data-matched fine
                              resolution that still stays above the correlation
                              length rather than fragmenting features)
    The percentage idea follows Šimić, Veas & Sabol (2025), but the specific
    sizes are chosen from ECG200's own autocorrelation scale rather than
    inherited: their signals were much longer, so their 2.5 % was a different
    physical scale. See DECISIONS_LOG.md for the full reasoning trail.
    These are exposed as constants below so the whole harness references one
    definition.

Non-integer handling (the key design decision)
    A percentage rarely divides the length into whole timesteps: 5 % of 96 is
    4.8 timesteps, not an integer. We resolve this by fixing the region COUNT,
    not the region length:

        n_regions = round(100 / region_size_pct)

    (10 regions for 10 %, 20 regions for 5 %). The timeline arange(length) is
    then split into n_regions contiguous, near-equal pieces, distributing the
    remainder so region sizes differ by AT MOST 1 — exactly np.array_split
    semantics: the first (length % n_regions) regions get one extra timestep.

    For length 96:
        10 %  → 10 regions:  6 of size 10 + 4 of size 9    (avg 9.6)
        5  %  → 20 regions: 16 of size 5  + 4 of size 4    (avg 4.8)

    Why fix the count rather than round the region length: rounding the region
    length to a whole number of timesteps drifts away from the intended
    percentage — most starkly for small percentages on a short signal (e.g. a
    2.5 % region on length 96 would round 2.4 → 2 or 3, i.e. 48 or 32 regions at
    2.08 % or 3.125 %, which is exactly why 2.5 % was rejected for ECG200).
    Fixing the count keeps every region as close to the target percentage as
    integer timesteps allow, and guarantees gapless, non-overlapping cover.

    The partition is fully deterministic: the same (length, region_size_pct)
    always yields the identical grid. This determinism is the whole point — the
    XAI side and the CMI side must get the exact same regions.

Representation
    A frozen RegionGrid dataclass carrying two views, both built from the single
    array_split result so they can never disagree:
        - bounds : (n_regions, 2) half-open [start, stop) intervals — canonical,
                   answers "which timesteps are in region k" and drives plotting.
        - labels : (length,) region id per timestep — enables mapping a
                   per-timestep vector to a per-region one by averaging, and back.
    All intervals are half-open [start, stop) to match Python/numpy slicing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ── Region sizes used across the harness (percent of input length) ────────────
# Source: Šimić, Veas & Sabol (2025). Referenced everywhere so the harness has
# one definition of "region size".
REGION_SIZE_PRIMARY_PCT: float = 10.0    # primary — data-justified anchor (≈ correlation length)
REGION_SIZE_SECONDARY_PCT: float = 5.0   # secondary — fine grid (data-matched fine resolution)
REGION_SIZES: dict[str, float] = {
    "primary": REGION_SIZE_PRIMARY_PCT,     # 10 %  anchor
    "secondary": REGION_SIZE_SECONDARY_PCT,  # 5 %  fine grid
}


@dataclass(frozen=True)
class RegionGrid:
    """A deterministic partition of a signal's timesteps into contiguous regions.

    Attributes
    ----------
    length : int
        Number of timesteps in the signal (e.g. 96 for ECG200).
    region_size_pct : float
        Region size as a percentage of `length` (e.g. 10.0 or 5.0).
    bounds : np.ndarray, shape (n_regions, 2), dtype int
        Half-open [start, stop) index intervals, one row per region, in order.
        Region k covers timesteps bounds[k, 0] .. bounds[k, 1] - 1. Canonical
        definition of the grid; use for slicing and for drawing boundaries.
    labels : np.ndarray, shape (length,), dtype int
        Region id (0 .. n_regions-1) for each timestep. Built from the same
        split as `bounds`. Enables per-timestep ↔ per-region mapping, e.g.
            per_region   = np.bincount(labels, weights=rel) / np.bincount(labels)
            per_timestep = per_region[labels]
        (Averaging itself is a later layer — this only provides the grid.)
    """

    length: int
    region_size_pct: float
    bounds: np.ndarray
    labels: np.ndarray

    @property
    def n_regions(self) -> int:
        """Number of regions in the grid."""
        return self.bounds.shape[0]

    @property
    def sizes(self) -> np.ndarray:
        """Per-region length (number of timesteps), shape (n_regions,)."""
        return self.bounds[:, 1] - self.bounds[:, 0]


def build_region_grid(length: int, region_size_pct: float) -> RegionGrid:
    """Build the deterministic region grid for a signal length and region size.

    See the module docstring for the non-integer handling rule. In brief:
    n_regions = round(100 / region_size_pct); the timeline is split into that
    many contiguous, near-equal regions with the remainder distributed so region
    sizes differ by at most 1.

    Parameters
    ----------
    length : int
        Signal length in timesteps; must be positive.
    region_size_pct : float
        Region size as a percentage of `length`, in (0, 100].

    Returns
    -------
    RegionGrid
        Frozen grid with `bounds` and `labels`, both derived from one split.

    Raises
    ------
    ValueError
        If `length` is not positive or `region_size_pct` is outside (0, 100].
    """
    if length <= 0:
        raise ValueError(f"length must be positive; got {length}")
    if not (0.0 < region_size_pct <= 100.0):
        raise ValueError(
            f"region_size_pct must be in (0, 100]; got {region_size_pct}"
        )

    # Fix the region count from the percentage (see module docstring), then
    # clamp to a valid range: at least 1 region, at most one region per timestep.
    n_regions = int(round(100.0 / region_size_pct))
    n_regions = max(1, min(n_regions, length))

    # Single source of the partition: split the timeline into n_regions
    # contiguous, near-equal pieces. np.array_split distributes the remainder by
    # giving the first (length % n_regions) pieces one extra timestep, so region
    # sizes differ by at most 1. bounds and labels are both derived from this one
    # `pieces` result, so they can never disagree.
    pieces = np.array_split(np.arange(length), n_regions)

    bounds = np.array([(piece[0], piece[-1] + 1) for piece in pieces], dtype=int)

    labels = np.empty(length, dtype=int)
    for region_id, piece in enumerate(pieces):
        labels[piece] = region_id

    return RegionGrid(
        length=length,
        region_size_pct=float(region_size_pct),
        bounds=bounds,
        labels=labels,
    )


def describe_grid(grid: RegionGrid) -> dict:
    """Summarise a grid for sanity-checking: region count, sizes, coverage.

    Parameters
    ----------
    grid : RegionGrid

    Returns
    -------
    dict
        Keys:
            "region_size_pct" : the requested percentage
            "n_regions"       : number of regions
            "min_size"        : smallest region length
            "max_size"        : largest region length
            "mean_size"       : average region length (= length / n_regions)
            "size_counts"     : {region_length: how_many_regions_have_it}
            "total_coverage"  : sum of region sizes (must equal grid.length)
            "covers_fully"    : bool, total_coverage == grid.length

    Notes
    -----
    Also asserts the partition covers exactly [0, length) with no gaps or
    overlaps, so a malformed grid fails loudly rather than silently.
    """
    sizes = grid.sizes
    unique, counts = np.unique(sizes, return_counts=True)
    size_counts = {int(s): int(c) for s, c in zip(unique, counts)}
    total = int(sizes.sum())

    # Guard: contiguous, gapless, non-overlapping cover of [0, length).
    assert grid.bounds[0, 0] == 0, "grid does not start at timestep 0"
    assert grid.bounds[-1, 1] == grid.length, "grid does not end at length"
    assert np.array_equal(
        grid.bounds[1:, 0], grid.bounds[:-1, 1]
    ), "regions are not contiguous (gap or overlap detected)"

    return {
        "region_size_pct": grid.region_size_pct,
        "n_regions": grid.n_regions,
        "min_size": int(sizes.min()),
        "max_size": int(sizes.max()),
        "mean_size": grid.length / grid.n_regions,
        "size_counts": size_counts,
        "total_coverage": total,
        "covers_fully": total == grid.length,
    }
