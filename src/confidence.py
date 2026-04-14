"""Calibrated confidence computation for face verification.

Formula: sigmoid of scaled score-threshold distance.

Given a cosine similarity score s and decision threshold t:

    confidence = sigmoid(k * (s - t))
               = 1 / (1 + exp(-k * (s - t)))

where k is a steepness constant (default: 10.0).

Properties:
- Output range: (0, 1) open interval — never exactly 0 or 1.
- At s == t: confidence == 0.5 — maximum uncertainty, score is on the boundary.
- s >> t (clearly same person): confidence approaches 1.0.
- s << t (clearly different person): confidence approaches 0.0.
- Monotonically increasing in s: higher cosine similarity → higher confidence.
- Reproducible and deterministic: no learned parameters, no randomness.

Steepness k=10 interpretation:
  s = threshold + 0.20  →  confidence ≈ 0.88
  s = threshold + 0.10  →  confidence ≈ 0.73
  s = threshold         →  confidence  = 0.50
  s = threshold - 0.10  →  confidence ≈ 0.27
  s = threshold - 0.20  →  confidence ≈ 0.12

Confidence interpretation:
  confidence reflects distance from the decision boundary (threshold),
  not the posterior probability of identity. A pair with confidence 0.9
  means the score is well above the threshold, not that there is a 90%
  chance they are the same person.
"""

import numpy as np

__all__ = ["compute_confidence"]

_DEFAULT_STEEPNESS: float = 10.0


def compute_confidence(
    score: float,
    threshold: float,
    steepness: float = _DEFAULT_STEEPNESS,
) -> float:
    """Compute calibrated confidence via sigmoid of score-threshold distance.

    Args:
        score: cosine similarity in [-1, 1]. Higher means more similar faces.
        threshold: decision boundary. score >= threshold → "same person".
        steepness: k in sigmoid(k * (score - threshold)). Default 10.0.
                   Larger values make confidence saturate faster away from
                   the boundary; smaller values produce a softer curve.

    Returns:
        float in (0, 1).
        Near 1.0: high confidence same person (score well above threshold).
        Near 0.0: high confidence different person (score well below threshold).
        0.5: score is exactly at threshold (maximum uncertainty).
    """
    x = float(steepness) * (float(score) - float(threshold))
    # np.exp handles float overflow safely for large |x|.
    return float(1.0 / (1.0 + np.exp(-x)))
