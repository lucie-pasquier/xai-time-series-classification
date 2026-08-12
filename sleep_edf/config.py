"""
src/config/sleep_edf.py
──────────────────────────────────────────────────────────────────────────────
Sleep-EDF dataset configuration.

These values parameterise the shared harness/models for Sleep-EDF (single-channel
EEG Fpz-Cz sleep staging). The region size is derived from the temporal scale of the
waveform events that define sleep stages under the AASM standard — see DECISIONS_LOG.md
("PHASE 2 — SLEEP-EDF", Region-size entry) for the full reasoning trail.

Region-size decision (summary)
    Sleep stages are defined by characteristic waveform events — sleep spindles
    (~0.5–1.5 s), K-complexes (~0.5 s) and slow waves (~1 s), i.e. ≈50–150 samples at
    100 Hz. A 60-sample (600 ms) region corresponds to approximately one such event,
    so the perturbation grid is aligned with the physiologically meaningful units of
    sleep staging. A signal-scale (autocorrelation) analysis is confirmatory only: the
    measured ~15-sample correlation length sits well below 60 samples, so a 60-sample
    region comfortably contains the signal's correlated structure.

How the harness consumes this
    src/xai/regions.py::build_region_grid(length, region_size_pct) takes the region
    size as an explicit PERCENTAGE and fixes the region count as
    round(100 / region_size_pct). Sleep-EDF passes REGION_SIZE_PRIMARY_PCT explicitly
    per call. So the canonical field the harness needs is the percentage; the
    sample-count and region-count fields below are the same decision expressed in
    absolute units for readability and sanity-checking.
"""

from __future__ import annotations

# ── Signal / task shape (now known from the Stage 1–3 loader) ─────────────────
INPUT_LENGTH: int = 3000     # timesteps per epoch = 30 s × 100 Hz
IN_CHANNELS: int = 1         # single EEG channel (Fpz-Cz)
N_CLASSES: int = 5           # AASM sleep stages: W, N1, N2, N3, REM

SAMPLING_RATE: int = 100     # Hz (for converting samples ↔ milliseconds)

# ── Region grid (see module docstring + DECISIONS_LOG) ────────────────────────
# Region size = the temporal scale of the waveform events that define sleep stages
# (spindles/K-complexes/slow waves ≈ 50–150 samples at 100 Hz):
#
#   PRIMARY : 60 samples = 600 ms = 2% of 3000  → 50 regions
#             (≈ one spindle / K-complex / slow-wave event)
REGION_SIZE_PRIMARY: int = 60    # samples (perturbation region size)

# The percentage form is what build_region_grid() actually consumes (it derives
# n_regions = round(100 / pct)). Passed explicitly per-call.
REGION_SIZE_PRIMARY_PCT: float = 2.0    # 60 / 3000  → round(100/2)  = 50 regions

# Resulting region count (exact, no fractional regions — see asserts below).
N_REGIONS_PRIMARY: int = 50

# A {name: pct} map the harness can iterate over.
REGION_SIZES_PCT: dict[str, float] = {
    "primary": REGION_SIZE_PRIMARY_PCT,   # 2%  → 50 regions of 60 samples
}

# ── Architecture parameters for the shared harness/models (see DECISIONS_LOG) ──
# These override the harness defaults (kernel 7 / patch 1), which were derived from
# ECG200 and have no standing here. Both are already constructor arguments on
# build_cnn()/build_transformer(); Sleep-EDF passes these values explicitly. Nothing
# in harness/ hardcodes them — the defaults there stay 7 / 1.
#
#   CNN first-layer kernel : 15 samples = 150 ms at 100 Hz. Matches Sleep-EDF's own
#       ~15-sample autocorrelation length, so each first-layer filter spans roughly
#       one coherence unit of the EEG. Derived from the signal, not inherited.
CNN_KERNEL_SIZE: int = 15
#
#   Transformer patch size : 60 samples = 600 ms = one token per 60-sample region.
#       Gives 3000 / 60 = 50 tokens — exactly one token per RegionGrid region, so
#       attention weights map onto the CMI region grid with no aggregation, and
#       full O(N^2) attention stays tractable (50 tokens, not 3000). 600 ms is also
#       the AASM stage-defining event scale used to set the region size.
TRANSFORMER_PATCH_SIZE: int = 60
TRANSFORMER_N_TOKENS: int = 50    # 3000 / 60 — one token per primary region

# ── Arithmetic sanity check: the region size divides 3000 EXACTLY (no remainder) ─
# 3000 ÷ 60 = 50 — every region is a whole, equal number of samples; no fractional
# regions and no remainder to distribute.
assert INPUT_LENGTH % REGION_SIZE_PRIMARY == 0, "primary region size must divide 3000 exactly"
assert INPUT_LENGTH // REGION_SIZE_PRIMARY == N_REGIONS_PRIMARY == 50
# Percentage form agrees with the sample-count form (round(100/pct) == n_regions).
assert round(100.0 / REGION_SIZE_PRIMARY_PCT) == N_REGIONS_PRIMARY
# Architecture params: odd kernel (padding kernel_size//2 preserves length); patch
# size divides the epoch exactly (no padding/truncation) and equals the region size.
assert CNN_KERNEL_SIZE % 2 == 1, "CNN kernel must be odd so padding=kernel_size//2 preserves length"
assert INPUT_LENGTH % TRANSFORMER_PATCH_SIZE == 0, "patch size must divide 3000 exactly (no padding)"
assert INPUT_LENGTH // TRANSFORMER_PATCH_SIZE == TRANSFORMER_N_TOKENS == 50
assert TRANSFORMER_PATCH_SIZE == REGION_SIZE_PRIMARY, "one token per primary region"
