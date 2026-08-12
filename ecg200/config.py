"""
ecg200/config.py
──────────────────────────────────────────────────────────────────────────────
ECG200 region-size constants (percent of input length).

These were previously baked into the shared regions.py; they now live here so the
harness stays dataset-agnostic (region size is passed to build_region_grid()).
Source: Šimić, Veas & Sabol (2025). Mirrors how sleep_edf/config.py owns its 2%.
"""

from __future__ import annotations

REGION_SIZE_PRIMARY_PCT: float = 10.0    # primary — data-justified anchor (≈ correlation length)
REGION_SIZE_SECONDARY_PCT: float = 5.0   # secondary — fine grid (data-matched fine resolution)
REGION_SIZES: dict[str, float] = {
    "primary": REGION_SIZE_PRIMARY_PCT,     # 10 %  anchor
    "secondary": REGION_SIZE_SECONDARY_PCT,  # 5 %  fine grid
}
