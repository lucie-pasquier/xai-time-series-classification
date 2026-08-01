"""
src/config/sleep_edf.py
──────────────────────────────────────────────────────────────────────────────
Sleep-EDF dataset configuration — PLACEHOLDER (values to be filled).

These values parameterise the shared harness/models for Sleep-EDF. They are left
commented out because they depend on analyses not yet run:

  - REGION_SIZE_PRIMARY_PCT / REGION_SIZE_SECONDARY_PCT
        from Sleep-EDF's own region-size analysis (the 01b-equivalent:
        autocorrelation / feature-scale characterisation of the EEG epochs).
        Do NOT reuse ECG200's 10% / 5% — Sleep-EDF has a different natural scale.
  - N_CLASSES
        Sleep-EDF sleep-stage classification (expected 5: W, N1, N2, N3, REM;
        confirm against the chosen labelling scheme).
  - IN_CHANNELS
        number of signal channels used (e.g. single EEG channel vs. multi-channel
        EEG/EOG); confirm during data-format recon.
  - INPUT_LENGTH
        timesteps per epoch (depends on sampling rate × epoch length, e.g. 30 s).

# REGION_SIZE_PRIMARY_PCT: float = ...
# REGION_SIZE_SECONDARY_PCT: float = ...
# N_CLASSES: int = ...
# IN_CHANNELS: int = ...
# INPUT_LENGTH: int = ...
"""

from __future__ import annotations

# Intentionally empty — see the module docstring. Fill after the Sleep-EDF
# data-characterisation and region-size analyses.
