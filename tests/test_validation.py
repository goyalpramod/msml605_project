"""Unit tests for src/validation.py."""

import numpy as np
import pytest

from src.validation import (
    validate_no_leakage,
    validate_pair_file,
    validate_scores,
    validate_threshold,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_pair_file(path, n=4, h=62, w=47, bad_labels=None):
    """Write a minimal valid .npz pair file to path."""
    rng = np.random.default_rng(0)
    img1 = rng.random((n, h, w)).astype(np.float32)
    img2 = rng.random((n, h, w)).astype(np.float32)
    label = np.array([1, 0, 1, 0][:n], dtype=np.int64)
    if bad_labels is not None:
        label = bad_labels
    np.savez(path, img1=img1, img2=img2, label=label)


# ---------------------------------------------------------------------------
# validate_pair_file
# ---------------------------------------------------------------------------


class TestValidatePairFile:
    def test_valid_file_passes(self, tmp_path):
        p = str(tmp_path / "pairs.npz")
        _write_pair_file(p)
        validate_pair_file(p)  # should not raise

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ValueError, match="not found"):
            validate_pair_file(str(tmp_path / "missing.npz"))

    def test_missing_key_raises(self, tmp_path):
        p = str(tmp_path / "pairs.npz")
        np.savez(p, img1=np.zeros((2, 4, 4)), img2=np.zeros((2, 4, 4)))  # no label
        with pytest.raises(ValueError, match="missing required keys"):
            validate_pair_file(p)

    def test_img_shape_mismatch_raises(self, tmp_path):
        p = str(tmp_path / "pairs.npz")
        np.savez(
            p,
            img1=np.zeros((2, 4, 4)),
            img2=np.zeros((2, 4, 5)),  # different width
            label=np.array([1, 0]),
        )
        with pytest.raises(ValueError, match="matching shapes"):
            validate_pair_file(p)

    def test_label_length_mismatch_raises(self, tmp_path):
        p = str(tmp_path / "pairs.npz")
        np.savez(
            p,
            img1=np.zeros((2, 4, 4)),
            img2=np.zeros((2, 4, 4)),
            label=np.array([1, 0, 1]),  # length 3, not 2
        )
        with pytest.raises(ValueError, match="label length"):
            validate_pair_file(p)

    def test_non_binary_labels_raise(self, tmp_path):
        p = str(tmp_path / "pairs.npz")
        _write_pair_file(p, bad_labels=np.array([0, 1, 2, 0]))
        with pytest.raises(ValueError, match="only 0 and 1"):
            validate_pair_file(p)


# ---------------------------------------------------------------------------
# validate_scores
# ---------------------------------------------------------------------------


class TestValidateScores:
    def test_valid_inputs_pass(self):
        validate_scores(np.array([0.8, 0.3]), np.array([1, 0]))  # should not raise

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            validate_scores(np.array([0.8, 0.3, 0.5]), np.array([1, 0]))

    def test_nan_in_scores_raises(self):
        with pytest.raises(ValueError, match="NaN or Inf"):
            validate_scores(np.array([0.8, np.nan]), np.array([1, 0]))

    def test_inf_in_scores_raises(self):
        with pytest.raises(ValueError, match="NaN or Inf"):
            validate_scores(np.array([0.8, np.inf]), np.array([1, 0]))

    def test_non_binary_labels_raise(self):
        with pytest.raises(ValueError, match="only 0 and 1"):
            validate_scores(np.array([0.8, 0.3]), np.array([1, 2]))

    def test_2d_scores_raise(self):
        with pytest.raises(ValueError, match="1D"):
            validate_scores(np.array([[0.8, 0.3]]), np.array([1, 0]))


# ---------------------------------------------------------------------------
# validate_threshold
# ---------------------------------------------------------------------------


class TestValidateThreshold:
    def test_valid_threshold_passes(self):
        validate_threshold(0.5, (0.0, 1.0))  # should not raise

    def test_below_range_raises(self):
        with pytest.raises(ValueError, match="outside the observed score range"):
            validate_threshold(-0.1, (0.0, 1.0))

    def test_above_range_raises(self):
        with pytest.raises(ValueError, match="outside the observed score range"):
            validate_threshold(1.1, (0.0, 1.0))

    def test_non_numeric_raises(self):
        with pytest.raises(ValueError, match="must be a number"):
            validate_threshold("high", (0.0, 1.0))

    def test_nan_raises(self):
        with pytest.raises(ValueError, match="finite"):
            validate_threshold(float("nan"), (0.0, 1.0))


# ---------------------------------------------------------------------------
# validate_no_leakage
# ---------------------------------------------------------------------------


class TestValidateNoLeakage:
    def test_disjoint_files_pass(self, tmp_path):
        val_path = str(tmp_path / "val.npz")
        test_path = str(tmp_path / "test.npz")
        rng = np.random.default_rng(0)
        # Distinct random images — extremely unlikely to collide.
        np.savez(
            val_path,
            img1=rng.random((2, 4, 4)),
            img2=rng.random((2, 4, 4)),
            label=np.array([1, 0]),
        )
        np.savez(
            test_path,
            img1=rng.random((2, 4, 4)),
            img2=rng.random((2, 4, 4)),
            label=np.array([0, 1]),
        )
        validate_no_leakage(val_path, test_path)  # should not raise

    def test_overlapping_files_raise(self, tmp_path):
        val_path = str(tmp_path / "val.npz")
        test_path = str(tmp_path / "test.npz")
        rng = np.random.default_rng(0)
        shared_img1 = rng.random((1, 4, 4)).astype(np.float32)
        shared_img2 = rng.random((1, 4, 4)).astype(np.float32)
        np.savez(val_path, img1=shared_img1, img2=shared_img2, label=np.array([1]))
        np.savez(test_path, img1=shared_img1, img2=shared_img2, label=np.array([1]))
        with pytest.raises(ValueError, match="leakage"):
            validate_no_leakage(val_path, test_path)
