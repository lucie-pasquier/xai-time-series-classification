"""
sleep_edf/bandpower.py
──────────────────────────────────────────────────────────────────────────────
Band-power features for the Sleep-EDF interpretable baseline (Model 1b).

The point of this rung is LEGIBILITY: each 30-s epoch is reduced to ONE named number
per standard sleep-EEG band, so a logistic-regression coefficient (or an attribution)
maps cleanly onto a physiological band ("how much does the model rely on delta?").
This is deliberately a tiny, named feature set — NOT a large opaque vector — because
the whole purpose is to have a model whose reasoning is readable against known sleep
physiology (esp. N3 ↔ delta), so the XAI/CMI machinery can be validated against
ground truth before it is trusted on the opaque CNNs.

Method: Welch PSD per epoch, integrated over each band (trapezoid). Optionally
log1p-compressed (log-power is the conventional EEG feature and conditions the linear
model better). Single source of truth — the notebook and any later XAI import from here.

Bands (standard sleep-staging set):
    delta 0.5–4 Hz · theta 4–8 · alpha 8–12 · sigma/spindle 12–16 · beta 16–30
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch

SAMPLING_RATE = 100          # Hz (Fpz-Cz)

# (name, low_hz, high_hz) — order fixes the feature-column order.
BANDS = [
    ("delta", 0.5, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 12.0),
    ("sigma", 12.0, 16.0),   # sleep-spindle band
    ("beta", 16.0, 30.0),
]
BAND_NAMES = [b[0] for b in BANDS]
N_BANDS = len(BANDS)


def band_power_features(X, fs: int = SAMPLING_RATE, nperseg: int = 512, log: bool = True):
    """Reduce raw epochs to per-band power features.

    Parameters
    ----------
    X : ndarray, shape (n_epochs, n_samples) or (n_samples,)
        Raw (per-epoch z-normalised) EEG epochs.
    fs : int
        Sampling rate (Hz).
    nperseg : int
        Welch segment length (clamped to the signal length). 512 ≈ 5.1 s windows →
        ~0.2 Hz resolution with averaging over the 30-s epoch — fine enough to
        resolve the low delta band that defines N3.
    log : bool
        If True, return log1p(power) (log-power; the conventional EEG feature).

    Returns
    -------
    ndarray, shape (n_epochs, 5) float32 — one column per band in BAND_NAMES order.
    (A 1-D input returns shape (5,).)
    """
    X = np.asarray(X, dtype=float)
    single = X.ndim == 1
    if single:
        X = X[None, :]
    npg = min(nperseg, X.shape[-1])
    freqs, psd = welch(X, fs=fs, nperseg=npg, axis=-1)          # psd: (n_epochs, n_freqs)
    feats = np.empty((X.shape[0], N_BANDS), dtype=np.float64)
    for j, (_name, lo, hi) in enumerate(BANDS):
        mask = (freqs >= lo) & (freqs < hi)
        feats[:, j] = np.trapz(psd[:, mask], freqs[mask], axis=-1)   # integrate PSD over the band
    if log:
        feats = np.log1p(feats)
    feats = feats.astype(np.float32)
    return feats[0] if single else feats
