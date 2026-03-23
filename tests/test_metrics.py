"""Unit tests for src/metrics.py.

Uses synthetic toy inputs with known correct answers so every assertion
can be verified by hand.
"""

import numpy as np
import pytest

from src.metrics import (
    balanced_accuracy,
    compute_confusion_matrix,
    compute_roc_points,
    equal_error_rate,
    f1_score_at_threshold,
    select_threshold,
)

# Shared threshold grid used across tests.
THRESHOLDS = np.linspace(0.0, 1.0, 101)


# ---------------------------------------------------------------------------
# compute_confusion_matrix
# ---------------------------------------------------------------------------


class TestComputeConfusionMatrix:
    def test_all_correct(self):
        # Scores well above threshold for positives, well below for negatives.
        scores = np.array([0.9, 0.8, 0.1, 0.2])
        labels = np.array([1, 1, 0, 0])
        cm = compute_confusion_matrix(scores, labels, threshold=0.5)
        assert cm == {"TP": 2, "FP": 0, "TN": 2, "FN": 0}

    def test_all_wrong(self):
        scores = np.array([0.1, 0.2, 0.9, 0.8])
        labels = np.array([1, 1, 0, 0])
        cm = compute_confusion_matrix(scores, labels, threshold=0.5)
        assert cm == {"TP": 0, "FP": 2, "TN": 0, "FN": 2}

    def test_all_predicted_positive(self):
        # Threshold of 0.0 → everything is predicted positive.
        scores = np.array([0.3, 0.7])
        labels = np.array([1, 0])
        cm = compute_confusion_matrix(scores, labels, threshold=0.0)
        assert cm == {"TP": 1, "FP": 1, "TN": 0, "FN": 0}

    def test_all_predicted_negative(self):
        # Threshold above all scores → everything predicted negative.
        scores = np.array([0.3, 0.7])
        labels = np.array([1, 0])
        cm = compute_confusion_matrix(scores, labels, threshold=1.1)
        assert cm == {"TP": 0, "FP": 0, "TN": 1, "FN": 1}

    def test_counts_are_integers(self):
        scores = np.array([0.6, 0.4])
        labels = np.array([1, 0])
        cm = compute_confusion_matrix(scores, labels, threshold=0.5)
        for key in ("TP", "FP", "TN", "FN"):
            assert isinstance(cm[key], int)


# ---------------------------------------------------------------------------
# compute_roc_points
# ---------------------------------------------------------------------------


class TestComputeRocPoints:
    def test_output_keys(self):
        scores = np.array([0.8, 0.3])
        labels = np.array([1, 0])
        result = compute_roc_points(scores, labels, THRESHOLDS)
        assert set(result.keys()) == {"tpr", "fpr", "fnr"}

    def test_output_length_matches_thresholds(self):
        scores = np.array([0.8, 0.3])
        labels = np.array([1, 0])
        result = compute_roc_points(scores, labels, THRESHOLDS)
        for key in ("tpr", "fpr", "fnr"):
            assert len(result[key]) == len(THRESHOLDS)

    def test_tpr_one_at_low_threshold(self):
        # At threshold=0.0, every pair is predicted positive → TPR=1.
        scores = np.array([0.8, 0.3])
        labels = np.array([1, 0])
        result = compute_roc_points(scores, labels, np.array([0.0]))
        assert result["tpr"][0] == pytest.approx(1.0)

    def test_tpr_zero_at_high_threshold(self):
        scores = np.array([0.8, 0.3])
        labels = np.array([1, 0])
        result = compute_roc_points(scores, labels, np.array([1.1]))
        assert result["tpr"][0] == pytest.approx(0.0)

    def test_tpr_plus_fnr_equals_one(self):
        scores = np.random.default_rng(0).random(20)
        labels = (scores > 0.5).astype(int)
        result = compute_roc_points(scores, labels, THRESHOLDS)
        np.testing.assert_allclose(result["tpr"] + result["fnr"], 1.0, atol=1e-9)


# ---------------------------------------------------------------------------
# balanced_accuracy
# ---------------------------------------------------------------------------


class TestBalancedAccuracy:
    def test_perfect_predictions(self):
        scores = np.array([0.9, 0.8, 0.1, 0.2])
        labels = np.array([1, 1, 0, 0])
        assert balanced_accuracy(scores, labels, threshold=0.5) == pytest.approx(1.0)

    def test_all_wrong(self):
        scores = np.array([0.1, 0.2, 0.9, 0.8])
        labels = np.array([1, 1, 0, 0])
        assert balanced_accuracy(scores, labels, threshold=0.5) == pytest.approx(0.0)

    def test_random_baseline(self):
        # Threshold=0 → all predicted positive → TPR=1, TNR=0 → balanced_acc=0.5
        scores = np.array([0.8, 0.3])
        labels = np.array([1, 0])
        assert balanced_accuracy(scores, labels, threshold=0.0) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# f1_score_at_threshold
# ---------------------------------------------------------------------------


class TestF1ScoreAtThreshold:
    def test_perfect_f1(self):
        scores = np.array([0.9, 0.8, 0.1, 0.2])
        labels = np.array([1, 1, 0, 0])
        assert f1_score_at_threshold(scores, labels, threshold=0.5) == pytest.approx(
            1.0
        )

    def test_zero_f1_no_true_positives(self):
        # All predictions are negative → F1 = 0.
        scores = np.array([0.1, 0.2])
        labels = np.array([1, 1])
        assert f1_score_at_threshold(scores, labels, threshold=0.5) == pytest.approx(
            0.0
        )

    def test_known_f1(self):
        # TP=1, FP=1, FN=1 → F1 = 2*1/(2+1+1) = 0.5
        scores = np.array([0.8, 0.6, 0.4])
        labels = np.array([1, 0, 1])
        result = f1_score_at_threshold(scores, labels, threshold=0.5)
        assert result == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# equal_error_rate
# ---------------------------------------------------------------------------


class TestEqualErrorRate:
    def test_eer_near_zero_for_perfect_separator(self):
        # Perfect separation → EER should be close to 0.
        scores = np.array([0.9, 0.85, 0.15, 0.1])
        labels = np.array([1, 1, 0, 0])
        eer_val, eer_thresh = equal_error_rate(scores, labels, THRESHOLDS)
        assert eer_val < 0.1

    def test_eer_near_half_for_random(self):
        # Completely random scores → EER ≈ 0.5.
        rng = np.random.default_rng(42)
        scores = rng.random(200)
        labels = rng.integers(0, 2, size=200)
        eer_val, _ = equal_error_rate(scores, labels, THRESHOLDS)
        assert 0.3 < eer_val < 0.7

    def test_eer_threshold_in_range(self):
        scores = np.array([0.9, 0.85, 0.15, 0.1])
        labels = np.array([1, 1, 0, 0])
        _, eer_thresh = equal_error_rate(scores, labels, THRESHOLDS)
        assert THRESHOLDS[0] <= eer_thresh <= THRESHOLDS[-1]


# ---------------------------------------------------------------------------
# select_threshold
# ---------------------------------------------------------------------------


class TestSelectThreshold:
    def test_invalid_rule_raises(self):
        scores = np.array([0.5])
        labels = np.array([1])
        with pytest.raises(ValueError, match="rule must be"):
            select_threshold(scores, labels, THRESHOLDS, rule="unknown")

    def test_balanced_acc_returns_float(self):
        scores = np.array([0.9, 0.8, 0.1, 0.2])
        labels = np.array([1, 1, 0, 0])
        t = select_threshold(scores, labels, THRESHOLDS, rule="balanced_acc")
        assert isinstance(t, float)

    def test_f1_returns_float(self):
        scores = np.array([0.9, 0.8, 0.1, 0.2])
        labels = np.array([1, 1, 0, 0])
        t = select_threshold(scores, labels, THRESHOLDS, rule="f1")
        assert isinstance(t, float)

    def test_eer_returns_float(self):
        scores = np.array([0.9, 0.8, 0.1, 0.2])
        labels = np.array([1, 1, 0, 0])
        t = select_threshold(scores, labels, THRESHOLDS, rule="eer")
        assert isinstance(t, float)

    def test_balanced_acc_selects_good_threshold_for_perfect_data(self):
        # With perfect separation, the selected threshold should sit between
        # the two score clusters (somewhere in (0.2, 0.8)).
        scores = np.array([0.9, 0.85, 0.15, 0.1])
        labels = np.array([1, 1, 0, 0])
        t = select_threshold(scores, labels, THRESHOLDS, rule="balanced_acc")
        assert 0.1 < t < 0.9
