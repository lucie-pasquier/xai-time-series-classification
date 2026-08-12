"""
src/data/preprocessing.py
──────────────────────────────────────────────────────────────────────────────
Data loading, normalisation, and split management for ECG200.

ECG200 overview (UCR archive)
    Binary ECG classification: normal vs abnormal heartbeat.
    100 training samples, 100 test samples.
    1 univariate channel, 96 timesteps.
    Raw labels: {-1, 1}  →  remapped to {0, 1}.

Per-sample z-score normalisation — design rationale
    Each trace is normalised against its *own* mean and std (μᵢ, σᵢ), not any
    population statistic computed over the training set. This is the UCR archive
    standard for removing inter-recording baseline drift (e.g. electrode offset
    between sessions). Because normalisation statistics are computed per sample,
    no statistics are ever *fit* on a collection of samples — there is no
    train→test leakage concern and no separate fit/transform step is needed.
    The same _zscore_per_sample() function is applied identically to train,
    validation, and test samples without any ordering dependency.

Public API
    build_processed_data(force=False)   one-time preprocessing; writes .npy files
    load_ecg200(split, normalised=True) primary entry point for all downstream code
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
from aeon.datasets import load_from_ts_file

# ── Project paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent          # ecg200/
RAW_DIR       = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_FILE = RAW_DIR / "ECG200_TRAIN.ts"
TEST_FILE  = RAW_DIR / "ECG200_TEST.ts"

VAL_FRACTION = 0.20
RANDOM_SEED  = 42


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load_ts(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a UCR .ts file via aeon and return arrays in canonical form.

    Parameters
    ----------
    path : Path
        Absolute path to a .ts file.

    Returns
    -------
    X : ndarray, shape (n_samples, n_timesteps)   float64, raw signal values
    y : ndarray, shape (n_samples,)               int, labels in {0, 1}
    """
    # aeon returns X as (n_samples, n_channels, n_timepoints) for return_type="numpy3d"
    X_raw, y_raw = load_from_ts_file(str(path), return_type="numpy3d")

    # ECG200 is univariate — squeeze the channel dimension
    X = X_raw[:, 0, :]  # (n_samples, n_timesteps)

    # Raw labels are stored as strings, values "-1.0" and "1.0".
    # Remap: -1 → 0,  1 → 1  via  (label + 1) / 2
    y = ((y_raw.astype(float) + 1) / 2).astype(int)

    return X, y


def _zscore_per_sample(X: np.ndarray) -> np.ndarray:
    """Per-sample z-score normalisation.

    Each row is normalised against its own mean and std. See module docstring
    for the rationale (no train/test leakage, no fit/transform split needed).

    A flat trace (std == 0) is returned as all-zeros without raising — ECG200
    contains no such traces, but the guard keeps the function safe if reused
    on other datasets.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_timesteps)

    Returns
    -------
    ndarray, shape (n_samples, n_timesteps), mean ≈ 0, std ≈ 1 per row
    """
    mu       = X.mean(axis=1, keepdims=True)
    std      = X.std(axis=1, keepdims=True)
    std_safe = np.where(std == 0.0, 1.0, std)  # avoid division by zero
    return (X - mu) / std_safe


def _make_val_split(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified 80/20 train/val split with a fixed random seed.

    Returns
    -------
    X_train, y_train, X_val, y_val
    """
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=VAL_FRACTION, random_state=RANDOM_SEED
    )
    train_idx, val_idx = next(sss.split(X, y))
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


# ── Public API ────────────────────────────────────────────────────────────────

def build_processed_data(force: bool = False) -> None:
    """Load raw .ts files, normalise, split, and persist as .npy arrays.

    Idempotent: skips processing unless *force* is True or files are missing.

    Files written to data/processed/
        X_train.npy  y_train.npy
        X_val.npy    y_val.npy
        X_test.npy   y_test.npy

    Parameters
    ----------
    force : bool
        If True, re-process even when processed files already exist.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    sentinel = PROCESSED_DIR / "X_train.npy"
    if sentinel.exists() and not force:
        return  # already processed; fast path

    # 1. Load raw data
    X_tr_full, y_tr_full = _load_ts(TRAIN_FILE)
    X_test,    y_test    = _load_ts(TEST_FILE)

    # 2. Per-sample z-score normalisation
    #    See module docstring: no population statistics are fit here, so there
    #    is no leakage between splits. Each sample is normalised against itself.
    X_tr_full = _zscore_per_sample(X_tr_full)
    X_test    = _zscore_per_sample(X_test)

    # 3. Stratified train / val split from the training portion
    X_train, y_train, X_val, y_val = _make_val_split(X_tr_full, y_tr_full)

    # 4. Persist
    arrays = {
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,
        "X_test":  X_test,  "y_test":  y_test,
    }
    for name, arr in arrays.items():
        np.save(PROCESSED_DIR / f"{name}.npy", arr)

    print(f"[preprocessing] Saved processed arrays to {PROCESSED_DIR}/")
    for name, arr in arrays.items():
        print(f"  {name:<10}  shape={arr.shape}  dtype={arr.dtype}")


def load_ecg200(
    split: str,
    normalised: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Load ECG200 arrays for a given split.

    This is the primary entry point for all downstream code (model training,
    XAI, evaluation). On the first call it runs build_processed_data()
    automatically; subsequent calls load directly from .npy files.

    Parameters
    ----------
    split : {"train", "val", "test"}
    normalised : bool, default True
        True  — per-sample z-score normalised; standard path for model training
                and XAI evaluation. Loads from data/processed/*.npy.
        False — raw signal values with labels remapped to {0, 1} but no
                normalisation applied. Useful for visualising what z-scoring
                does (e.g. in 01_data_exploration.ipynb).

    Returns
    -------
    X : ndarray, shape (n_samples, n_timesteps)
    y : ndarray of int, shape (n_samples,)  — labels in {0, 1}
    """
    if split not in {"train", "val", "test"}:
        raise ValueError(
            f"split must be one of 'train', 'val', 'test'; got {split!r}"
        )

    # ── Normalised path (standard) ────────────────────────────────────────────
    if normalised:
        build_processed_data()  # no-op if files exist
        X = np.load(PROCESSED_DIR / f"X_{split}.npy")
        y = np.load(PROCESSED_DIR / f"y_{split}.npy")
        return X, y

    # ── Unnormalised path (exploration / visualisation only) ──────────────────
    # Loads from raw .ts files and applies the same split logic as
    # build_processed_data() so indices are consistent between the two paths.
    if split == "test":
        return _load_ts(TEST_FILE)

    X_full, y_full = _load_ts(TRAIN_FILE)
    X_train, y_train, X_val, y_val = _make_val_split(X_full, y_full)
    return (X_train, y_train) if split == "train" else (X_val, y_val)
