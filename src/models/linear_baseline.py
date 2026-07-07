"""
src/models/linear_baseline.py
──────────────────────────────────────────────────────────────────────────────
Model 1 (linear baseline): band-power feature extraction + logistic regression.

Pipeline
    1. Band-pass-filter each ECG signal (scipy.signal.butter + filtfilt) into a
       small set of frequency bands.
    2. Extract ONE feature per band — its band power (mean squared amplitude of
       the band-filtered signal).
    3. Fit a logistic regression classifier (sklearn) on the band-power matrix.

Why one feature per band
    This model is the interpretable ground-truth anchor for the thesis. With a
    single band-power feature per band, the fitted logistic-regression weights
    form a clean 1:1 map — one coefficient per frequency band — so "which bands
    does the model rely on?" has an unambiguous answer. That answer is what
    TimeSHAP will later be tested against.

Frequency conventions — IMPORTANT
    All band edges are expressed in NORMALISED frequency: a fraction of the
    Nyquist rate, where 1.0 == Nyquist. scipy.signal.butter interprets Wn this
    way when no sampling rate is supplied, so nothing in this module depends on
    knowing the physical sampling rate of ECG200.

    ECG200 (UCR archive) does NOT document a sampling rate. Any statement in Hz
    is an INTERPRETATION ONLY and lives in notebook markdown, never in code. For
    reference, if one heartbeat (~0.8 s) spans 96 samples then fs ≈ 120 Hz and
    Nyquist ≈ 60 Hz — but this figure is an assumption to be verified and is
    deliberately absent from every computation below.

Per-sample z-scoring interaction
    Inputs are per-sample z-scored upstream (see src/data/preprocessing.py), so
    each trace has ~unit total power. The band powers therefore roughly sum to a
    constant across bands and are mildly anti-correlated by construction: they
    describe how a fixed energy budget is DISTRIBUTED across frequencies, not
    absolute energy. This must be accounted for when reading the coefficients.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── Band definitions (NORMALISED frequency: fraction of Nyquist, 1.0 == Nyquist)
# Each entry: (name, low, high). low=None → low-pass; high=None → high-pass.
# Physiological motivation (interpretation only; see module docstring on fs):
#   B1  baseline drift / P- and T-wave energy      (lowest frequencies)
#   B2  P/T-wave detail, QRS onset
#   B3  QRS core energy                            (dominant beat structure)
#   B4  sharp QRS edges
#   B5  high-frequency content / noise             (highest frequencies)
DEFAULT_BANDS: list[tuple[str, float | None, float | None]] = [
    ("B1_low",       None, 0.08),
    ("B2_low_mid",   0.08, 0.25),
    ("B3_mid",       0.25, 0.42),
    ("B4_high",      0.42, 0.67),
    ("B5_very_high", 0.67, None),
]

RANDOM_SEED = 42


def bandpass_filter(
    x: np.ndarray,
    low: float | None,
    high: float | None,
    order: int = 4,
    axis: int = -1,
) -> np.ndarray:
    """Zero-phase Butterworth filter in NORMALISED frequency.

    Dispatches to low-pass, band-pass, or high-pass depending on which edges are
    given. Edges are fractions of the Nyquist rate (1.0 == Nyquist); no physical
    sampling rate is used anywhere.

    Parameters
    ----------
    x : ndarray
        Signal(s); filtering is applied along `axis`.
    low, high : float or None
        Normalised band edges in (0, 1). low=None → low-pass at `high`;
        high=None → high-pass at `low`; both given → band-pass [low, high].
    order : int, default 4
        Butterworth order.
    axis : int, default -1
        Axis along which to filter (use axis=1 for an (n_samples, n_timesteps)
        matrix).

    Returns
    -------
    ndarray, same shape as `x` — the band-filtered signal (zero-phase, via
    filtfilt so no temporal shift is introduced).
    """
    if low is None and high is None:
        raise ValueError("At least one of `low`/`high` must be given.")

    if low is None:
        b, a = butter(order, high, btype="lowpass")
    elif high is None:
        b, a = butter(order, low, btype="highpass")
    else:
        b, a = butter(order, [low, high], btype="bandpass")

    return filtfilt(b, a, x, axis=axis)


def band_power(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Mean squared amplitude (average power) along `axis`."""
    return np.mean(np.square(x), axis=axis)


def extract_band_power_features(
    X: np.ndarray,
    bands: list[tuple[str, float | None, float | None]] = DEFAULT_BANDS,
    order: int = 4,
) -> tuple[np.ndarray, list[str]]:
    """Turn raw ECG traces into a band-power feature matrix.

    For each band, the whole batch is band-filtered and the per-trace band power
    (mean squared amplitude) is taken as the single feature for that band.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_timesteps)
        Per-sample z-scored ECG traces (as produced by load_ecg200).
    bands : list of (name, low, high)
        Normalised-frequency band definitions; defaults to DEFAULT_BANDS.
    order : int, default 4
        Butterworth order passed to bandpass_filter.

    Returns
    -------
    features : ndarray, shape (n_samples, n_bands)  — one band-power per band
    names    : list[str]                            — band names, column order
    """
    if X.ndim != 2:
        raise ValueError(f"Expected X of shape (n_samples, n_timesteps); got {X.shape}")

    columns = []
    names = []
    for name, low, high in bands:
        filtered = bandpass_filter(X, low, high, order=order, axis=1)
        columns.append(band_power(filtered, axis=1))
        names.append(name)

    features = np.column_stack(columns)  # (n_samples, n_bands)
    return features, names


def build_linear_baseline() -> Pipeline:
    """Model 1 estimator: standardise band-power features, then logistic reg.

    StandardScaler is fit on the training features only (inside the pipeline) so
    the learned coefficients are directly comparable in magnitude across bands.
    class_weight="balanced" compensates for the ~65/35 class imbalance in ECG200
    so the minority (abnormal) class is not ignored.

    Returns
    -------
    sklearn.pipeline.Pipeline  — steps: 'scaler' → 'clf'.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )
