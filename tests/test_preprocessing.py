"""
tests/test_preprocessing.py
Tests for src/data/preprocessing.py.

Run with:  pytest tests/test_preprocessing.py -v
"""

import numpy as np
import pytest

from src.data.preprocessing import (
    _zscore_per_sample,
    _make_val_split,
    load_ecg200,
    build_processed_data,
    VAL_FRACTION,
    RANDOM_SEED,
)


# ── Unit tests for _zscore_per_sample ─────────────────────────────────────────

class TestZscorePerSample:
    def test_zero_mean(self):
        """Each row should have mean ≈ 0 after normalisation."""
        rng = np.random.default_rng(0)
        X = rng.normal(loc=5.0, scale=2.0, size=(20, 96))
        X_norm = _zscore_per_sample(X)
        np.testing.assert_allclose(X_norm.mean(axis=1), 0.0, atol=1e-10)

    def test_unit_std(self):
        """Each row should have std ≈ 1 after normalisation."""
        rng = np.random.default_rng(1)
        X = rng.normal(loc=0.0, scale=3.0, size=(20, 96))
        X_norm = _zscore_per_sample(X)
        np.testing.assert_allclose(X_norm.std(axis=1), 1.0, atol=1e-10)

    def test_shape_preserved(self):
        """Output shape must match input shape."""
        X = np.ones((5, 96))
        assert _zscore_per_sample(X).shape == (5, 96)

    def test_flat_trace_no_error(self):
        """A flat trace (std == 0) should return zeros without raising."""
        X = np.full((3, 96), fill_value=7.0)
        X_norm = _zscore_per_sample(X)
        np.testing.assert_array_equal(X_norm, 0.0)

    def test_per_sample_independence(self):
        """Normalising sample i should not depend on the values of sample j."""
        X = np.array([[0.0, 1.0, 2.0],   # mean=1, std=~0.816
                      [10.0, 20.0, 30.0]])  # mean=20, std=~8.165
        X_norm = _zscore_per_sample(X)
        # Both rows should have mean 0, std 1 independently
        np.testing.assert_allclose(X_norm.mean(axis=1), [0.0, 0.0], atol=1e-10)
        np.testing.assert_allclose(X_norm.std(axis=1), [1.0, 1.0], atol=1e-10)


# ── Unit tests for _make_val_split ────────────────────────────────────────────

class TestMakeValSplit:
    @pytest.fixture
    def dummy_data(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(100, 96))
        y = np.array([0] * 50 + [1] * 50)
        return X, y

    def test_correct_sizes(self, dummy_data):
        X, y = dummy_data
        X_tr, y_tr, X_val, y_val = _make_val_split(X, y)
        assert X_tr.shape[0] + X_val.shape[0] == 100
        assert abs(X_val.shape[0] / 100 - VAL_FRACTION) < 0.05

    def test_no_overlap(self, dummy_data):
        """Train and val splits must be disjoint — checked via label arrays."""
        X, y = dummy_data
        X_tr, y_tr, X_val, y_val = _make_val_split(X, y)
        # If any row appears in both splits, the sum of shape[0] wouldn't add up.
        # For a stronger check: use a unique marker per row.
        markers = np.arange(100)
        X_m = markers[:, None] * np.ones((1, 96))
        X_m_tr, _, X_m_val, _ = _make_val_split(X_m, y)
        tr_ids  = set(X_m_tr[:, 0].astype(int))
        val_ids = set(X_m_val[:, 0].astype(int))
        assert tr_ids.isdisjoint(val_ids)

    def test_deterministic(self, dummy_data):
        """Same input → same split (fixed RANDOM_SEED)."""
        X, y = dummy_data
        result_1 = _make_val_split(X, y)
        result_2 = _make_val_split(X, y)
        for a, b in zip(result_1, result_2):
            np.testing.assert_array_equal(a, b)

    def test_stratified(self, dummy_data):
        """Class ratio in val should be approximately preserved."""
        X, y = dummy_data
        _, _, _, y_val = _make_val_split(X, y)
        expected_ratio = y.mean()
        actual_ratio   = y_val.mean()
        assert abs(actual_ratio - expected_ratio) < 0.15  # within 15 pp


# ── Integration tests for load_ecg200 ────────────────────────────────────────

class TestLoadEcg200:
    def test_shapes_2d(self):
        """X must be 2D, y must be 1D, and they must have the same leading dim."""
        for split in ("train", "val", "test"):
            X, y = load_ecg200(split)
            assert X.ndim == 2, f"{split}: X should be 2D"
            assert y.ndim == 1, f"{split}: y should be 1D"
            assert X.shape[0] == y.shape[0], f"{split}: sample count mismatch"

    def test_timesteps(self):
        """ECG200 has 96 timesteps."""
        X, _ = load_ecg200("train")
        assert X.shape[1] == 96

    def test_labels_binary(self):
        """Labels must be in {0, 1} only."""
        for split in ("train", "val", "test"):
            _, y = load_ecg200(split)
            assert set(y.tolist()).issubset({0, 1}), \
                f"{split}: unexpected label values {set(y.tolist())}"

    def test_normalised_stats(self):
        """Normalised signals should have per-sample mean ≈ 0, std ≈ 1."""
        X, _ = load_ecg200("train", normalised=True)
        np.testing.assert_allclose(X.mean(axis=1), 0.0, atol=1e-6)
        np.testing.assert_allclose(X.std(axis=1),  1.0, atol=1e-6)

    def test_train_val_partition(self):
        """train + val sample counts should equal 100 (ECG200 training set)."""
        X_tr, _ = load_ecg200("train")
        X_val, _ = load_ecg200("val")
        assert X_tr.shape[0] + X_val.shape[0] == 100

    def test_invalid_split_raises(self):
        with pytest.raises(ValueError, match="split must be"):
            load_ecg200("bogus")

    def test_unnormalised_raw_path(self):
        """normalised=False should return same shapes but different values."""
        X_norm, y_norm = load_ecg200("train", normalised=True)
        X_raw,  y_raw  = load_ecg200("train", normalised=False)
        assert X_norm.shape == X_raw.shape
        np.testing.assert_array_equal(y_norm, y_raw)
        # Raw and normalised should NOT be identical
        assert not np.allclose(X_norm, X_raw), \
            "normalised and raw arrays should differ"
