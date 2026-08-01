"""
src/data/sleep_edf.py
──────────────────────────────────────────────────────────────────────────────
Sleep-EDF loader — STUB (not yet implemented).

The Sleep-EDF replication's data loader will live here, mirroring the role of
src/data/preprocessing.py for ECG200 (load raw -> normalise -> split -> cache).
It is intentionally a stub: the on-disk format (raw .edf PSG recordings +
hypnogram annotations, 30-s epoching, channel selection, 5-class sleep stages,
and the train/val/test split) has not been characterised yet.

Raw data lives at data/sleep_edf/raw/ (git-ignored, ~8 GB). Processed arrays will
be written to data/sleep_edf/processed/ once this is implemented.

NOTE: this module is NOT exported from src/data/__init__.py yet — it is kept inert
so nothing imports it until the loader is real.
"""

from __future__ import annotations


def load_sleep_edf(split: str, normalised: bool = True):
    """Load Sleep-EDF arrays for a given split — NOT YET IMPLEMENTED.

    Intended signature/contract (to match load_ecg200 so the shared harness and
    models plug in unchanged):
        split : {"train", "val", "test"}
        returns (X, y) with X shape (n_samples, n_channels, n_timesteps) or
        (n_samples, n_timesteps), y int labels in {0, ..., n_classes-1}.
    """
    raise NotImplementedError(
        "Sleep-EDF loader not yet implemented — pending data access/format recon."
    )
