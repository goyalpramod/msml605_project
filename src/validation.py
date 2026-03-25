"""Pipeline validation checks for the face verification evaluation loop.

All functions raise ValueError with a clear message on failure (fail-fast).
Call these at the top of evaluate.py before any computation.
"""

import numpy as np


def validate_pair_file(path: str) -> None:
    """Check that a .npz pair file has the correct schema.

    Validates:
    - File exists and is loadable.
    - Required keys: img1, img2, label.
    - img1 and img2 have matching 3D shapes (N, H, W).
    - label is 1D with length N and values in {0, 1}.
    """
    import os

    if not os.path.exists(path):
        raise ValueError(f"Pair file not found: {path}")

    try:
        data = np.load(path)
    except Exception as e:
        raise ValueError(f"Could not load pair file {path}: {e}") from e

    required_keys = {"img1", "img2", "label"}
    missing = required_keys - set(data.files)
    if missing:
        raise ValueError(
            f"Pair file {path} is missing required keys: {sorted(missing)}. "
            f"Found: {sorted(data.files)}"
        )

    img1, img2, label = data["img1"], data["img2"], data["label"]

    if img1.ndim != 3:
        raise ValueError(f"img1 must be 3D (N, H, W). Got ndim={img1.ndim}.")
    if img1.shape != img2.shape:
        raise ValueError(
            f"img1 and img2 must have matching shapes. Got {img1.shape} vs {img2.shape}."
        )
    if label.ndim != 1:
        raise ValueError(f"label must be 1D. Got shape={label.shape}.")
    if img1.shape[0] != label.shape[0]:
        raise ValueError(
            f"Number of pairs in images ({img1.shape[0]}) does not match "
            f"label length ({label.shape[0]})."
        )

    unique_labels = set(np.unique(label).tolist())
    if not unique_labels.issubset({0, 1}):
        raise ValueError(
            f"label must contain only 0 and 1. Found values: {unique_labels}."
        )


def validate_scores(scores: np.ndarray, labels: np.ndarray) -> None:
    """Check that scores and labels are compatible for metric computation.

    Validates:
    - Both are 1D arrays.
    - Same length.
    - No NaN or Inf in scores.
    - labels contains only 0 and 1.
    """
    scores = np.asarray(scores)
    labels = np.asarray(labels)

    if scores.ndim != 1:
        raise ValueError(f"scores must be 1D. Got shape={scores.shape}.")
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D. Got shape={labels.shape}.")
    if len(scores) != len(labels):
        raise ValueError(
            f"scores and labels must have the same length. "
            f"Got {len(scores)} and {len(labels)}."
        )
    if not np.isfinite(scores).all():
        n_bad = int(np.sum(~np.isfinite(scores)))
        raise ValueError(f"scores contains {n_bad} NaN or Inf value(s).")

    unique_labels = set(np.unique(labels).tolist())
    if not unique_labels.issubset({0, 1}):
        raise ValueError(
            f"labels must contain only 0 and 1. Found values: {unique_labels}."
        )


def validate_threshold(threshold: float, score_range: tuple[float, float]) -> None:
    """Check that a threshold is a finite number within the observed score range.

    Args:
        threshold:   The decision threshold to validate.
        score_range: (min_score, max_score) observed in the score array.
    """
    if not isinstance(threshold, (int, float)):
        raise ValueError(
            f"threshold must be a number. Got type {type(threshold).__name__}."
        )
    if not np.isfinite(threshold):
        raise ValueError(f"threshold must be finite. Got {threshold}.")

    lo, hi = score_range
    if threshold < lo or threshold > hi:
        raise ValueError(
            f"threshold {threshold:.4f} is outside the observed score range "
            f"[{lo:.4f}, {hi:.4f}]."
        )


def validate_no_leakage(val_path: str, test_path: str) -> None:
    """Check that val and test pair files share no identical pairs.

    Pairs are compared as (img1_flat, img2_flat) byte fingerprints so this
    works without needing identity metadata.

    Raises ValueError if any pair appears in both files.
    """
    val_data = np.load(val_path)
    test_data = np.load(test_path)

    # Use a hash of each flattened pair as a compact fingerprint.
    def pair_hashes(data):
        img1 = data["img1"].reshape(len(data["img1"]), -1)
        img2 = data["img2"].reshape(len(data["img2"]), -1)
        hashes = set()
        for a, b in zip(img1, img2):
            hashes.add((a.tobytes(), b.tobytes()))
        return hashes

    val_hashes = pair_hashes(val_data)
    test_hashes = pair_hashes(test_data)

    overlap = val_hashes & test_hashes
    if overlap:
        raise ValueError(
            f"Data leakage detected: {len(overlap)} pair(s) appear in both "
            f"{val_path} and {test_path}."
        )
