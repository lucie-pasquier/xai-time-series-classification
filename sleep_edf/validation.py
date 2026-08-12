"""
sleep_edf/validation.py
──────────────────────────────────────────────────────────────────────────────
Fixed, subject-level early-stopping validation split for the Sleep-EDF complexity
ladder — the single source of truth every model (2 → 5) uses so early stopping
fires against the SAME target on every rung.

Why subject-level (not a random epoch split)
    Sleep-EDF has subject structure: epochs from one subject share electrode
    placement, skull geometry and noise floor. A random class-stratified split of
    EPOCHS would place the same subject on both sides, letting the model partially
    memorise subject identity, so the early-stopping signal would be optimistic and
    stopping would fire late. That effect grows with model capacity, so it would
    vary along the very axis (complexity) the thesis measures. Holding out whole
    SUBJECTS makes the early-stopping signal a real generalisation-to-unseen-subject
    signal — consistent with the leakage-free subject-level train/test split the
    loader already uses. (A random-epoch split is harmless only without subject
    structure, e.g. ECG200; that assumption does not transfer here.)

What this splits
    The held-out subjects are chosen from the full 62 training subjects and applied
    to whatever training set is loaded — by default the fixed 20,000-epoch subsample
    every ladder model trains on. The TEST set is untouched: it remains the separate,
    leakage-free subject-level holdout (16 different subjects).

Fixed & reusable
    The held-out subject list is deterministic AND frozen to
    data/processed/sleep_edf_val_subjects.json, so later models LOAD it rather than
    regenerate it — the same reason the 20K subsample is fixed. A per-model val split
    would move the early-stopping target between rungs and inject variance along the
    measured axis.

Split seed = 8 (NOT the project-wide seed 42)
    Everything else in Sleep-EDF uses seed 42 (subject-level train/test split, 20K
    subsample). The VALIDATION split alone uses seed 8. Reason: with seed 42, three of
    the six held-out subjects had ZERO N3 epochs, so the early-stopping signal would be
    effectively blind to N3 (the hardest, rarest stage). A rejection criterion — no
    empty-N3 validation subject and an adequate absolute N3 count — was fixed BEFORE any
    candidate seed was examined (selection for measurability of a known-hard class, not
    selection on results). Seed 8 is the ~10% (6-subject) split meeting it with the
    healthiest minority counts (N3 = 164, N1 = 231, no empty-N3 subject). See
    DECISIONS_LOG.md for the audit trail of candidates compared.

Nothing here depends on ecg200/ or on harness/.
"""

from __future__ import annotations

import json

import numpy as np

from sleep_edf.loader import (load_sleep_edf, PROCESSED_DIR,
                              SUBSAMPLE_TARGET, SUBSAMPLE_SEED, N_CLASSES)

# ── Split policy (fixed across the whole ladder) ──────────────────────────────
VAL_HELDOUT_FRACTION = 0.10     # ~10% of the 62 training subjects -> 6 held-out subjects
VAL_SPLIT_SEED = 8              # NB: 8, not the project-wide 42 (seed 42 left 3/6 val
                                # subjects with zero N3). See module docstring / DECISIONS_LOG.

_FROZEN = PROCESSED_DIR / "sleep_edf_val_subjects.json"   # single source of truth once written

# The committed canonical split. Loading asserts against this so a corrupted/edited/
# regenerated file fails loudly instead of silently changing every model's early-stopping
# target. Changing the ladder's val split means changing this constant deliberately.
EXPECTED_VAL_SUBJECTS = [2, 16, 37, 48, 60, 70]


def _all_train_subjects() -> list[int]:
    """Sorted list of the (62) subject ids present in the FULL training split."""
    _, _, subj = load_sleep_edf("train", subsample=None, verbose=False, return_subjects=True)
    return sorted(int(s) for s in np.unique(subj))


def compute_val_subjects(frac: float = VAL_HELDOUT_FRACTION,
                         seed: int = VAL_SPLIT_SEED) -> list[int]:
    """Deterministically choose the held-out validation subjects (pure; writes nothing).

    ~`frac` of the training subjects, drawn by a seeded permutation. Whole subjects,
    so a subject's two nights are always on the same side.
    """
    subjects = _all_train_subjects()
    n_val = max(1, round(frac * len(subjects)))
    perm = np.random.RandomState(seed).permutation(subjects)
    return sorted(int(s) for s in perm[:n_val])


def freeze_val_subjects(frac: float = VAL_HELDOUT_FRACTION,
                        seed: int = VAL_SPLIT_SEED, force: bool = False) -> dict:
    """Materialise the frozen val-subject record (idempotent). Returns the record dict.

    Stores the chosen subjects plus provenance (fraction, seed, per-class train/val
    counts on the fixed 20K subsample) so the split is auditable and reproducible.
    """
    if _FROZEN.exists() and not force:
        return json.load(open(_FROZEN))
    val_subjects = compute_val_subjects(frac, seed)
    # Provenance: per-class counts on the fixed 20K subsample this split is used with.
    _, y, subj = load_sleep_edf("train", subsample=SUBSAMPLE_TARGET, seed=SUBSAMPLE_SEED,
                                verbose=False, return_subjects=True)
    val_mask = np.isin(subj, val_subjects)
    record = {
        "val_subjects": val_subjects,
        "n_val_subjects": len(val_subjects),
        "heldout_fraction": frac,
        "split_seed": seed,
        "applies_to": {"subsample": SUBSAMPLE_TARGET, "subsample_seed": SUBSAMPLE_SEED},
        "counts_on_20k": {
            "train": {"n": int((~val_mask).sum()),
                      "per_class": np.bincount(y[~val_mask], minlength=N_CLASSES).tolist()},
            "val":   {"n": int(val_mask.sum()),
                      "per_class": np.bincount(y[val_mask], minlength=N_CLASSES).tolist()},
        },
    }
    _FROZEN.parent.mkdir(parents=True, exist_ok=True)
    json.dump(record, open(_FROZEN, "w"), indent=2)
    return record


def load_val_subjects() -> list[int]:
    """Held-out validation subject ids from the frozen canonical file.

    NEVER regenerates and NEVER falls back to a default: the frozen file is the
    ladder-wide source of truth, and silently recreating it would change every model's
    early-stopping target with nothing visibly breaking. Raises if the file is missing or
    if its contents do not match EXPECTED_VAL_SUBJECTS.
    """
    if not _FROZEN.exists():
        raise FileNotFoundError(
            f"Canonical validation split not found at {_FROZEN}. This file is committed and "
            "is the single source of truth for the ladder-wide early-stopping split; it is "
            "never regenerated at load time (a silently regenerated split would invisibly "
            "change what every model early-stops against). Restore it from version control. "
            "To (re)create it deliberately, call freeze_val_subjects(force=True)."
        )
    subjects = json.load(open(_FROZEN))["val_subjects"]
    if subjects != EXPECTED_VAL_SUBJECTS:
        raise ValueError(
            f"Frozen val subjects {subjects} != expected {EXPECTED_VAL_SUBJECTS}. The canonical "
            "ladder-wide split has changed — refusing to proceed. If this change is intended, "
            "update EXPECTED_VAL_SUBJECTS and the DECISIONS_LOG entry together."
        )
    return subjects


def train_val_split(subsample: int | None = SUBSAMPLE_TARGET, seed: int = SUBSAMPLE_SEED):
    """Load the (subsampled) training set and split it by the fixed held-out subjects.

    ORDER: subsample FIRST (the fixed 20K), THEN hold out the val subjects — so the split
    partitions the identical 20,000-epoch subsample every model trains on (17,742 train /
    2,258 val), never the full 155,334 set.

    Integrity is asserted on every call (fail loudly): all six frozen subjects are present,
    train and val are subject-disjoint (=> both nights of a held-out subject sit on the val
    side, never split), and the held-out subjects do not appear in the test set.

    Returns
    -------
    X_tr, y_tr, X_val, y_val : ndarrays
        Subject-disjoint training / validation partition of the loaded 20K subsample.
    """
    X, y, subj = load_sleep_edf("train", subsample=subsample, seed=seed,
                                verbose=False, return_subjects=True)
    val_subjects = load_val_subjects()
    val_mask = np.isin(subj, val_subjects)

    # (1) all six frozen val subjects are actually present in the loaded set.
    present = sorted(int(s) for s in np.unique(subj[val_mask]))
    if present != val_subjects:
        raise ValueError(f"expected val subjects {val_subjects} but the load contains {present}")
    # (2) train/val subject-disjoint — no held-out subject's epoch (either night) in train.
    train_subjects = set(int(s) for s in np.unique(subj[~val_mask]))
    leaked = train_subjects & set(val_subjects)
    if leaked:
        raise ValueError(f"train/val not subject-disjoint; shared subjects: {sorted(leaked)}")
    # (3) val subjects disjoint from the test set (leakage-free w.r.t. the held-out test).
    _, _, test_subj = load_sleep_edf("test", verbose=False, return_subjects=True)
    test_overlap = set(val_subjects) & set(int(s) for s in np.unique(test_subj))
    if test_overlap:
        raise ValueError(f"val subjects overlap the test set: {sorted(test_overlap)}")

    return X[~val_mask], y[~val_mask], X[val_mask], y[val_mask]
