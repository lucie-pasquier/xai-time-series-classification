"""
src/data/sleep_edf.py
──────────────────────────────────────────────────────────────────────────────
Sleep-EDF loader — single-channel EEG Fpz-Cz sleep staging (AASM 5-class).

Hardened from the Stage 1–2 validation notebooks
(notebooks/sleep_edf/03_loader_stage1.ipynb, 02_dataset_construction.ipynb).
This module only ASSEMBLES the already-validated pieces — it does not re-derive
or change the logic. It mirrors the ECG200 pattern in src/data/preprocessing.py:
`build_processed_data()` builds + caches once; `load_sleep_edf(split)` loads the
cache (building it on first use).

Pipeline (per Sleep-Cassette recording), all validated in Stages 1–2:
    1. Pair PSG↔hypnogram by shared 6-char filename prefix (not sorted index).
    2. Read the single channel EEG Fpz-Cz at 100 Hz (MNE), and the hypnogram
       annotation spans.
    3. Build per-epoch labels from the spans, mapping R&K → AASM 5-class
       (stages 3+4 → N3); "Movement time"/"?"/Unknown → dropped.
    4. Tail-trim guard: if the hypnogram labels more 30-s epochs than the signal
       has, trim the extra label tail (hypnogram-past-signal off-by-one).
    5. Cut the signal into non-overlapping 30-s / 3000-sample epochs, aligned to
       the labels at the same (epoch) granularity so they cannot desync.
    6. Wake-trim: keep only 30 min (== 60 epochs; the ×2 is 2 epochs/min) of Wake
       at each end of the sleep period (first→last non-Wake epoch).
    7. Per-epoch z-normalisation (each epoch scaled by its own mean/std).

Robustness: the per-recording loop catches and REPORTS (never silently drops) any
missing-channel / missing-hypnogram / all-Wake / read-error recording.

Split: leakage-free SUBJECT-level split (seed 42, ~80/20 by subject); a subject's
two nights are always kept on the same side.

NO-LEAKAGE NOTE (guard against a future mistake):
    Normalisation here is PER-EPOCH — each 30-s epoch is scaled by ITS OWN mean and
    std. There are therefore NO dataset-level statistics (no global/train mean or
    std) that could leak across the train/test split. If a future change replaces
    this with a global or train-fit normalisation, those statistics MUST be fit on
    the TRAIN split only and applied to test — otherwise the subject-level split's
    leakage-freedom is silently broken. This is a comment, not a change.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import mne

# ── Paths (mirrors ECG200's data/{raw,processed} layout, under sleep_edf/) ────
PROJECT_ROOT = Path(__file__).resolve().parent          # sleep_edf/
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "sleep-cassette"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# ── Fixed pipeline constants (validated in Stages 1–2; do not "improve") ──────
CHANNEL = "EEG Fpz-Cz"
SAMPLING_RATE = 100
EPOCH_SEC = 30
EPOCH_SAMPLES = SAMPLING_RATE * EPOCH_SEC          # 3000
N_CLASSES = 5

# R&K → AASM 5-class: W,N1,N2,N3(=3+4),REM. Anything else (Movement/"?") → DROP.
AASM_LABELS = {
    "Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2,
    "Sleep stage 3": 3, "Sleep stage 4": 3, "Sleep stage R": 4,
}
DROP = -1

WAKE_EDGE_EPOCHS = 30 * 2      # keep 30 min of Wake each side; ×2 = 2 epochs/min
SEED = 42
TEST_FRACTION = 0.20

# Training subsample (validated in notebooks/sleep_edf/02_dataset_construction.ipynb,
# Step D; this is the single source of truth). Default for TRAINING loads so every
# model on the complexity ladder trains on the IDENTICAL subsample.
SUBSAMPLE_SEED = 42
SUBSAMPLE_TARGET = 20_000

mne.set_log_level("CRITICAL")


# ── Per-recording pipeline (raises on any quirk; caller catches & reports) ────

def _epoch_and_label(psg_path: Path):
    """One recording → (X (n,3000) float32 pre-z-norm, y (n,) in {0..4}), post-drop.

    Raises ValueError for: no hypnogram, missing channel, or all-Wake.
    """
    prefix = psg_path.name[:6]
    hyps = list(RAW_DIR.glob(f"{prefix}*-Hypnogram.edf"))
    if not hyps:
        raise ValueError(f"no hypnogram for prefix {prefix}")

    raw = mne.io.read_raw_edf(psg_path, preload=True)
    if CHANNEL not in raw.ch_names:
        raise ValueError(f"missing '{CHANNEL}' (channels: {raw.ch_names})")
    signal = raw.copy().pick([CHANNEL]).get_data()[0].astype(np.float32)   # (n_samples,)
    n_sig = len(signal) // EPOCH_SAMPLES

    ann = mne.read_annotations(hyps[0])
    labels = []
    for onset, dur, stage in zip(ann.onset, ann.duration, ann.description):
        labels += [AASM_LABELS.get(stage, DROP)] * int(round(dur / EPOCH_SEC))
    labels = np.array(labels)

    # Tail-trim guard: hypnogram may cover more epochs than the signal (or vice versa).
    n_keep = min(n_sig, len(labels))
    X = signal[:n_keep * EPOCH_SAMPLES].reshape(n_keep, EPOCH_SAMPLES)
    labels = labels[:n_keep]

    # Drop Movement/Unknown at the SAME (epoch) granularity via one mask.
    keep = labels != DROP
    X = X[keep]
    y = labels[keep].astype(np.int64)
    if (y != 0).sum() == 0:
        raise ValueError("all-Wake (no sleep epochs)")
    return X, y


def _wake_trim(X: np.ndarray, y: np.ndarray):
    """Keep WAKE_EDGE_EPOCHS of Wake each side of the sleep period (first→last non-Wake)."""
    nw = np.where(y != 0)[0]
    if len(nw) == 0:                                   # guard: all-Wake
        return X[:0], y[:0]
    start = max(0, nw[0] - WAKE_EDGE_EPOCHS)
    end = min(len(y) - 1, nw[-1] + WAKE_EDGE_EPOCHS)
    return X[start:end + 1], y[start:end + 1]


def _znorm_per_epoch(X: np.ndarray) -> np.ndarray:
    """Per-epoch z-normalisation (each epoch by its own mean/std; guard std==0)."""
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd = np.where(sd == 0.0, 1.0, sd)
    return ((X - mu) / sd).astype(np.float32)


def _subject_split(recording_ids):
    """Group recordings by subject (filename chars [3:5]); seeded ~80/20 subject split.

    Returns (train_subjects, test_subjects, by_subject dict).
    """
    by_subject = {}
    for rid in recording_ids:
        by_subject.setdefault(rid[3:5], []).append(rid)
    subjects = sorted(by_subject)
    perm = np.random.RandomState(SEED).permutation(subjects)
    n_test = max(1, round(TEST_FRACTION * len(subjects)))
    return set(perm[n_test:]), set(perm[:n_test]), by_subject


def _cache_paths(split):
    return (PROCESSED_DIR / f"sleep_edf_{split}_X.npy",
            PROCESSED_DIR / f"sleep_edf_{split}_y.npy")


# ── Public API ────────────────────────────────────────────────────────────────

def build_processed_data(force_rebuild: bool = False, verbose: bool = True) -> dict:
    """Build the full Sleep-Cassette dataset and cache train/test arrays to disk.

    Loads all recordings, epochs, maps labels, wake-trims, and does the
    subject-level split (seed 42), then saves
    data/sleep_edf/processed/sleep_edf_{train,test}_{X,y}.npy.

    Idempotent: if the cache already exists and `force_rebuild` is False, returns
    a summary without rebuilding. Returns a summary dict (issues + counts).
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if not force_rebuild and all(p.exists() for s in ("train", "test") for p in _cache_paths(s)):
        return {"status": "cache_exists"}

    psg_paths = sorted(RAW_DIR.glob("*PSG.edf"))
    clean, issues = {}, []
    for p in psg_paths:
        rid = p.name.replace("-PSG.edf", "")
        try:
            X, y = _epoch_and_label(p)
            X, y = _wake_trim(X, y)
            clean[rid] = (_znorm_per_epoch(X), y)
        except Exception as e:                        # catch & REPORT, never silent
            issues.append((rid, str(e)[:80]))

    if verbose:
        print(f"[sleep_edf] recordings: {len(psg_paths)} found, {len(clean)} clean, {len(issues)} issue(s)")
        for rid, reason in issues:
            print(f"[sleep_edf]   ISSUE {rid}: {reason}")

    train_subj, test_subj, by_subject = _subject_split(clean.keys())
    train_recs = [r for s in sorted(train_subj) for r in by_subject[s]]
    test_recs = [r for s in sorted(test_subj) for r in by_subject[s]]

    summary = {"status": "built", "issues": issues, "n_clean": len(clean),
               "subjects": {"train": len(train_subj), "test": len(test_subj)},
               "recordings": {"train": len(train_recs), "test": len(test_recs)}, "epochs": {}}

    for split, recs in (("train", train_recs), ("test", test_recs)):
        X = np.concatenate([clean[r][0] for r in recs]).astype(np.float32)
        y = np.concatenate([clean[r][1] for r in recs]).astype(np.int64)
        Xp, yp = _cache_paths(split)
        np.save(Xp, X)
        np.save(yp, y)
        summary["epochs"][split] = int(len(y))
        summary[f"class_counts_{split}"] = {int(c): int((y == c).sum()) for c in range(N_CLASSES)}
        if verbose:
            print(f"[sleep_edf] saved {split}: X{X.shape} y{y.shape} -> {Xp.name}, {yp.name}")
    return summary


def subsample_indices(y, target: int = SUBSAMPLE_TARGET, seed: int = SUBSAMPLE_SEED):
    """Deterministic, class-STRATIFIED subsample of a label array -> sorted indices.

    For each class we keep the SAME fraction (~target/len(y)) of that class's epochs,
    chosen at random. One shared fraction preserves the natural class balance EXACTLY
    (no rebalancing); drawing at random across the pooled epochs (which span all
    subjects) makes every subject contribute a proportional, randomly-placed slice.
    Deterministic given (y, seed) — the identical subsample regenerates every call, so
    every model on the complexity ladder trains on the same epochs.

    Validated in notebooks/sleep_edf/02_dataset_construction.ipynb (Step D): on the full
    155,334-epoch training set with target=20000, seed=42 this yields exactly 20,000
    epochs, natural balance preserved, N3 = 1,329, all 62 subjects kept. This is the
    single source of truth — the notebook imports it from here.
    """
    rng = np.random.RandomState(seed)
    frac = target / len(y)
    picks = []
    for c in range(N_CLASSES):
        pos = np.where(y == c)[0]              # all epochs of class c
        k = int(round(frac * len(pos)))        # same fraction of every class
        picks.append(rng.choice(pos, size=k, replace=False))
    return np.sort(np.concatenate(picks))


def load_sleep_edf(split: str = "train", subsample: int | None = SUBSAMPLE_TARGET,
                   seed: int = SUBSAMPLE_SEED, force_rebuild: bool = False,
                   verbose: bool = True):
    """Load cached Sleep-EDF arrays for a split, building the cache on first use.

    Parameters
    ----------
    split : {"train", "test"}
    subsample : int or None, default 20000
        TRAINING loads only. If a positive int, return a deterministic class-stratified
        subsample of that many epochs (see `subsample_indices`) — this is the DEFAULT so
        that every model on the complexity ladder trains on the IDENTICAL 20,000-epoch
        set without anyone having to remember to ask. Pass ``subsample=None`` (or 0) to
        opt out and get the FULL training set (155,334) — a deliberate, visible act.
        IGNORED for the test split, which is ALWAYS returned whole (never subsampled).
    seed : int, default 42
        Seed for the training subsample (fixed, so the subsample is reproducible).
    force_rebuild : bool
        Regenerate the cache from the raw EDFs even if it exists.
    verbose : bool, default True
        Print a one-line record of what a training load returned (subsample vs full), so
        any training run visibly logs the data it trained on.

    Returns
    -------
    X : ndarray (n_epochs, 3000) float32   — per-epoch z-normalised EEG Fpz-Cz
    y : ndarray (n_epochs,) int in {0,1,2,3,4}  — AASM stages W,N1,N2,N3,REM

    Notes
    -----
    Subsampling is pure index selection on the already-cached full arrays (fast, in
    memory) — it never rebuilds the cache or re-reads any EDF.
    """
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test'; got {split!r}")
    Xp, yp = _cache_paths(split)
    if force_rebuild or not (Xp.exists() and yp.exists()):
        build_processed_data(force_rebuild=force_rebuild)
    X, y = np.load(Xp), np.load(yp)

    # Test is ALWAYS whole — never subsampled, regardless of `subsample`.
    if split == "test":
        return X, y

    # Training: subsample by default; explicit opt-out (None / 0) returns the full set.
    if subsample:
        idx = subsample_indices(y, target=subsample, seed=seed)
        X, y = X[idx], y[idx]
        if verbose:
            print(f"[sleep_edf] training load: {len(y):,}-epoch stratified subsample "
                  f"(seed {seed})")
    elif verbose:
        print(f"[sleep_edf] training load: full training set ({len(y):,} epochs)")
    return X, y
