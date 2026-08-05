"""
src/config/sleep_edf.py
──────────────────────────────────────────────────────────────────────────────
Sleep-EDF dataset configuration.

These values parameterise the SHARED harness/models for Sleep-EDF (single-channel
EEG Fpz-Cz sleep staging). They are the Sleep-EDF analogue of ECG200's constants;
nothing here changes ECG200. The region sizes come from Sleep-EDF's own region-size
analysis (notebooks/sleep_edf/03_region_size_analysis.ipynb) — NOT reused from
ECG200, whose natural scale is different. See DECISIONS_LOG.md ("PHASE 2 —
SLEEP-EDF", Region-size entry) for the full reasoning trail.

Region-size decision (summary)
    Stage-4 autocorrelation measured Sleep-EDF's 1/e correlation length at
    15 samples (150 ms). Rather than take that raw grain as the region size
    (which would give 200 regions/epoch — 0.5% of a 3000-sample epoch), the grid
    is deliberately set at the EEG FEATURE scale (sleep spindles / K-complexes /
    slow waves ≈ 50–150 samples), using the 15-sample correlation length only as a
    floor. This is a documented, physiologically-motivated adaptation chosen BEFORE
    any faithfulness results were computed — not a tuned parameter.

How the harness consumes these
    src/xai/regions.py::build_region_grid(length, region_size_pct) takes the region
    size as an explicit PERCENTAGE and fixes the region count as
    round(100 / region_size_pct). Sleep-EDF therefore passes its own *_PCT values
    explicitly; regions.py's ECG200 defaults (10% / 5%) are NOT touched. So the
    canonical field the harness needs is the percentage; the sample-count and
    region-count fields below are the same decision expressed in absolute units for
    readability and sanity-checking.
"""

from __future__ import annotations

# ── Signal / task shape (now known from the Stage 1–3 loader) ─────────────────
INPUT_LENGTH: int = 3000     # timesteps per epoch = 30 s × 100 Hz
IN_CHANNELS: int = 1         # single EEG channel (Fpz-Cz)
N_CLASSES: int = 5           # AASM sleep stages: W, N1, N2, N3, REM

SAMPLING_RATE: int = 100     # Hz (for converting samples ↔ milliseconds)

# ── Region grid (Sleep-EDF-native; see module docstring + DECISIONS_LOG) ──────
# Primary and fine keep ECG200's 2:1 primary:fine ratio, but at the EEG feature
# scale rather than the raw 15-sample correlation length.
#
#   PRIMARY : 60 samples = 600 ms = 2% of 3000  → 50 regions  (≈4 corr. lengths;
#             ≈ one spindle / K-complex / slow-wave event)
#   FINE    : 30 samples = 300 ms = 1% of 3000  → 100 regions (half the primary;
#             still ≥2 corr. lengths, so never fragments a coherent grain)
REGION_SIZE_PRIMARY: int = 60    # samples (primary/coarse grid)
REGION_SIZE_FINE: int = 30       # samples (fine grid; half of primary — 2:1 ratio)

# The percentage form is what build_region_grid() actually consumes (it derives
# n_regions = round(100 / pct)). Passed explicitly per-call; regions.py unchanged.
REGION_SIZE_PRIMARY_PCT: float = 2.0    # 60 / 3000  → round(100/2)  = 50 regions
REGION_SIZE_FINE_PCT: float = 1.0       # 30 / 3000  → round(100/1)  = 100 regions

# Resulting region counts (exact, no fractional regions — see asserts below).
N_REGIONS_PRIMARY: int = 50
N_REGIONS_FINE: int = 100

# Parity with regions.py's naming: a {name: pct} map the harness can iterate over.
REGION_SIZES_PCT: dict[str, float] = {
    "primary": REGION_SIZE_PRIMARY_PCT,   # 2%  → 50 regions of 60 samples
    "fine": REGION_SIZE_FINE_PCT,         # 1%  → 100 regions of 30 samples
}

# ── Arithmetic sanity check: both sizes divide 3000 EXACTLY (no remainder) ─────
# 3000 ÷ 60 = 50 and 3000 ÷ 30 = 100 — every region is a whole, equal number of
# samples; no fractional regions and no remainder to distribute.
assert INPUT_LENGTH % REGION_SIZE_PRIMARY == 0, "primary region size must divide 3000 exactly"
assert INPUT_LENGTH % REGION_SIZE_FINE == 0, "fine region size must divide 3000 exactly"
assert INPUT_LENGTH // REGION_SIZE_PRIMARY == N_REGIONS_PRIMARY == 50
assert INPUT_LENGTH // REGION_SIZE_FINE == N_REGIONS_FINE == 100
# Percentage form agrees with the sample-count form (round(100/pct) == n_regions).
assert round(100.0 / REGION_SIZE_PRIMARY_PCT) == N_REGIONS_PRIMARY
assert round(100.0 / REGION_SIZE_FINE_PCT) == N_REGIONS_FINE
# The 2:1 primary:fine ratio (matching ECG200's convention) holds.
assert REGION_SIZE_PRIMARY == 2 * REGION_SIZE_FINE
