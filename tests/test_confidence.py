"""Unit tests for src/confidence.py.

Tests the sigmoid confidence formula:
    compute_confidence(score, threshold, steepness=10) = sigmoid(10 * (score - threshold))
"""

import numpy as np

from src.confidence import compute_confidence


class TestComputeConfidence:
    # ------------------------------------------------------------------
    # Output range
    # ------------------------------------------------------------------

    def test_output_is_float(self):
        result = compute_confidence(0.8, 0.5)
        assert isinstance(result, float)

    def test_output_strictly_above_zero(self):
        for score in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            assert compute_confidence(score, 0.5) > 0.0, (
                f"Expected > 0, got {compute_confidence(score, 0.5)} for score={score}"
            )

    def test_output_strictly_below_one(self):
        for score in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            assert compute_confidence(score, 0.5) < 1.0, (
                f"Expected < 1, got {compute_confidence(score, 0.5)} for score={score}"
            )

    def test_output_in_open_unit_interval(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            score = float(rng.uniform(-1, 1))
            threshold = float(rng.uniform(-1, 1))
            c = compute_confidence(score, threshold)
            assert 0.0 < c < 1.0, (
                f"Out of range: {c} for score={score}, threshold={threshold}"
            )

    # ------------------------------------------------------------------
    # Boundary conditions
    # ------------------------------------------------------------------

    def test_score_at_threshold_is_half(self):
        for t in [0.0, 0.3, 0.5, 0.7, -0.2]:
            c = compute_confidence(t, t)
            assert abs(c - 0.5) < 1e-9, (
                f"Expected 0.5 at boundary, got {c} for threshold={t}"
            )

    def test_score_far_above_threshold_approaches_one(self):
        c = compute_confidence(
            1.0, -1.0
        )  # score - threshold = 2.0, steepness=10 → x=20
        assert c > 0.999, f"Expected near 1.0, got {c}"

    def test_score_far_below_threshold_approaches_zero(self):
        c = compute_confidence(
            -1.0, 1.0
        )  # score - threshold = -2.0, steepness=10 → x=-20
        assert c < 0.001, f"Expected near 0.0, got {c}"

    def test_above_threshold_gives_confidence_above_half(self):
        c = compute_confidence(0.8, 0.5)
        assert c > 0.5, f"Expected > 0.5 for score above threshold, got {c}"

    def test_below_threshold_gives_confidence_below_half(self):
        c = compute_confidence(0.2, 0.5)
        assert c < 0.5, f"Expected < 0.5 for score below threshold, got {c}"

    # ------------------------------------------------------------------
    # Monotonicity: higher score → higher confidence (cosine direction)
    # ------------------------------------------------------------------

    def test_monotonically_increasing_in_score(self):
        threshold = 0.5
        scores = np.linspace(-1.0, 1.0, 50)
        confidences = [compute_confidence(float(s), threshold) for s in scores]
        for i in range(len(confidences) - 1):
            assert confidences[i] <= confidences[i + 1], (
                f"Monotonicity violated at index {i}: "
                f"confidence({scores[i]:.3f}) = {confidences[i]:.4f} > "
                f"confidence({scores[i + 1]:.3f}) = {confidences[i + 1]:.4f}"
            )

    def test_strictly_increasing_away_from_boundary(self):
        threshold = 0.0
        scores = [-0.5, -0.3, -0.1, 0.1, 0.3, 0.5]
        confidences = [compute_confidence(s, threshold) for s in scores]
        for i in range(len(confidences) - 1):
            assert confidences[i] < confidences[i + 1], (
                f"Not strictly increasing: confidence({scores[i]}) = {confidences[i]:.4f}, "
                f"confidence({scores[i + 1]}) = {confidences[i + 1]:.4f}"
            )

    # ------------------------------------------------------------------
    # Steepness parameter
    # ------------------------------------------------------------------

    def test_steepness_controls_saturation(self):
        score, threshold = 0.6, 0.5
        c_low = compute_confidence(score, threshold, steepness=1.0)
        c_high = compute_confidence(score, threshold, steepness=20.0)
        assert c_high > c_low, (
            "Higher steepness should give higher confidence for score above threshold"
        )

    def test_steepness_one_matches_standard_sigmoid(self):
        score, threshold = 0.7, 0.3
        expected = 1.0 / (1.0 + np.exp(-(score - threshold)))
        result = compute_confidence(score, threshold, steepness=1.0)
        assert abs(result - expected) < 1e-9

    # ------------------------------------------------------------------
    # Determinism
    # ------------------------------------------------------------------

    def test_deterministic(self):
        args = (0.75, 0.5)
        assert compute_confidence(*args) == compute_confidence(*args)
