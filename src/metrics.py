"""Metric computation functions for face verification evaluation.

All functions are pure numpy — no I/O, no side effects.

Score direction convention:
  cosine similarity  → higher score = more likely same person
  euclidean distance → lower score  = more likely same person

For euclidean, pass (1 - normalised_score) or negate scores before calling,
OR use select_threshold which handles direction via the `score_direction` arg.
"""

import numpy as np

# Document the prediction convention for each similarity type.
SCORE_DIRECTION = {
    "cosine": "higher_is_same",
    "euclidean": "lower_is_same",
}


def compute_roc_points(
    scores: np.ndarray,
    labels: np.ndarray,
    thresholds: np.ndarray,
) -> dict:
    """Compute TPR, FPR, and FNR at each threshold.

    For cosine similarity (higher = same), a pair is predicted positive
    when score >= threshold.

    Args:
        scores: 1D array of similarity scores, shape (N,).
        labels: 1D binary array, shape (N,). 1 = same person, 0 = different.
        thresholds: 1D array of threshold values to sweep over.

    Returns:
        Dict with keys "tpr", "fpr", "fnr" — each a 1D array matching thresholds.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    thresholds = np.asarray(thresholds, dtype=np.float64)

    n_pos = np.sum(labels == 1)
    n_neg = np.sum(labels == 0)

    tpr = np.zeros(len(thresholds))
    fpr = np.zeros(len(thresholds))
    fnr = np.zeros(len(thresholds))

    for i, t in enumerate(thresholds):
        predicted_pos = scores >= t
        tp = np.sum(predicted_pos & (labels == 1))
        fp = np.sum(predicted_pos & (labels == 0))
        fn = np.sum(~predicted_pos & (labels == 1))

        tpr[i] = tp / n_pos if n_pos > 0 else 0.0
        fpr[i] = fp / n_neg if n_neg > 0 else 0.0
        fnr[i] = fn / n_pos if n_pos > 0 else 0.0

    return {"tpr": tpr, "fpr": fpr, "fnr": fnr}


def compute_confusion_matrix(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict:
    """Compute confusion matrix at a single threshold.

    Args:
        scores: 1D similarity scores, shape (N,).
        labels: 1D binary labels, shape (N,).
        threshold: Decision boundary. Predict same if score >= threshold.

    Returns:
        Dict with integer keys: TP, FP, TN, FN.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    predicted_pos = scores >= threshold
    tp = int(np.sum(predicted_pos & (labels == 1)))
    fp = int(np.sum(predicted_pos & (labels == 0)))
    tn = int(np.sum(~predicted_pos & (labels == 0)))
    fn = int(np.sum(~predicted_pos & (labels == 1)))

    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn}


def balanced_accuracy(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> float:
    """Balanced accuracy = (TPR + TNR) / 2.

    Robust to class imbalance, unlike plain accuracy.
    """
    cm = compute_confusion_matrix(scores, labels, threshold)
    tp, fp, tn, fn = cm["TP"], cm["FP"], cm["TN"], cm["FN"]

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return (tpr + tnr) / 2.0


def f1_score_at_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> float:
    """F1 score at a given threshold.

    F1 = 2 * TP / (2 * TP + FP + FN).
    Returns 0.0 when the denominator is zero.
    """
    cm = compute_confusion_matrix(scores, labels, threshold)
    tp, fp, fn = cm["TP"], cm["FP"], cm["FN"]

    denom = 2 * tp + fp + fn
    return (2 * tp) / denom if denom > 0 else 0.0


def equal_error_rate(
    scores: np.ndarray,
    labels: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[float, float]:
    """Find the Equal Error Rate (EER) and the threshold at which it occurs.

    EER is the point where FAR (FPR) == FRR (FNR). We find the threshold
    that minimises |FPR - FNR| across the sweep.

    Returns:
        (eer_value, eer_threshold) — both floats.
    """
    roc = compute_roc_points(scores, labels, thresholds)
    fpr = roc["fpr"]
    fnr = roc["fnr"]

    idx = int(np.argmin(np.abs(fpr - fnr)))
    eer_value = float((fpr[idx] + fnr[idx]) / 2.0)
    eer_threshold = float(thresholds[idx])
    return eer_value, eer_threshold


def select_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    thresholds: np.ndarray,
    rule: str,
) -> float:
    """Select an operating threshold using one of three rules.

    Args:
        scores: 1D similarity scores.
        labels: 1D binary labels.
        thresholds: Candidate thresholds to evaluate (from eval_config.json).
        rule: One of "balanced_acc", "f1", "eer".

    Returns:
        The selected threshold as a float.
    """
    if rule not in {"balanced_acc", "f1", "eer"}:
        raise ValueError(f"rule must be 'balanced_acc', 'f1', or 'eer'. Got: {rule!r}")

    if rule == "eer":
        _, eer_threshold = equal_error_rate(scores, labels, thresholds)
        return eer_threshold

    # For balanced_acc and f1, compute the metric at every threshold and pick the max.
    metric_fn = balanced_accuracy if rule == "balanced_acc" else f1_score_at_threshold
    values = np.array([metric_fn(scores, labels, t) for t in thresholds])
    best_idx = int(np.argmax(values))
    return float(thresholds[best_idx])
