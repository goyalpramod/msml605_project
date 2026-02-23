import os

import numpy as np

__all__ = [
    "cosine_similarity",
    "euclidean_distance",
    "cosine_similarity_loop",
    "euclidean_distance_loop",
    "load_pair_vectors",
]


def _validate_pair_batches(
    a: np.ndarray, b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    # Force a shared float dtype up front so loop and NumPy paths are comparable.
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)

    if a_arr.ndim != 2 or b_arr.ndim != 2:
        raise ValueError("Inputs must be 2D arrays of shape (N, D).")
    if a_arr.shape != b_arr.shape:
        raise ValueError(
            f"Input shapes must match. Got a={a_arr.shape}, b={b_arr.shape}."
        )

    return a_arr, b_arr


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_arr, b_arr = _validate_pair_batches(a, b)
    numerator = np.einsum("ij,ij->i", a_arr, b_arr, optimize=True)
    a_sq = np.einsum("ij,ij->i", a_arr, a_arr, optimize=True)
    b_sq = np.einsum("ij,ij->i", b_arr, b_arr, optimize=True)
    denominator = np.sqrt(a_sq * b_sq)
    # If either row is all zeros, cosine is undefined; return 0.0 instead of NaN.
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator > 0,
    )


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_arr, b_arr = _validate_pair_batches(a, b)
    diff = a_arr - b_arr
    squared = np.einsum("ij,ij->i", diff, diff, optimize=True)
    return np.sqrt(squared)


def cosine_similarity_loop(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_arr, b_arr = _validate_pair_batches(a, b)
    n_rows, n_cols = a_arr.shape
    scores = np.zeros(n_rows, dtype=np.float64)

    # Intentionally plain Python math for a baseline against vectorized NumPy.
    for i in range(n_rows):
        dot = 0.0
        a_norm_sq = 0.0
        b_norm_sq = 0.0

        row_a = a_arr[i]
        row_b = b_arr[i]

        for j in range(n_cols):
            a_val = float(row_a[j])
            b_val = float(row_b[j])
            dot += a_val * b_val
            a_norm_sq += a_val * a_val
            b_norm_sq += b_val * b_val

        denom = float(np.sqrt(a_norm_sq * b_norm_sq))
        scores[i] = dot / denom if denom > 0 else 0.0

    return scores


def euclidean_distance_loop(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_arr, b_arr = _validate_pair_batches(a, b)
    n_rows, n_cols = a_arr.shape
    dists = np.zeros(n_rows, dtype=np.float64)

    for i in range(n_rows):
        row_a = a_arr[i]
        row_b = b_arr[i]
        squared_sum = 0.0

        for j in range(n_cols):
            diff = float(row_a[j]) - float(row_b[j])
            squared_sum += diff * diff

        dists[i] = float(np.sqrt(squared_sum))

    return dists


def load_pair_vectors(
    split: str, out_dir: str = "./outputs"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if split not in {"train", "test"}:
        raise ValueError(f"split must be 'train' or 'test'. Got: {split}")

    path = os.path.join(out_dir, f"pairs_{split}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Pair file not found at {path}. "
            "Run `python scripts/generate_pairs.py --seed 42` first."
        )

    data = np.load(path)
    required_keys = {"img1", "img2", "label"}
    if not required_keys.issubset(data.files):
        raise ValueError(
            f"Invalid pair file {path}. Expected keys {sorted(required_keys)}, "
            f"found {sorted(data.files)}."
        )

    img1 = data["img1"]
    img2 = data["img2"]
    label = data["label"]

    if img1.shape != img2.shape:
        raise ValueError(
            f"img1 and img2 must have matching shape. Got {img1.shape} vs {img2.shape}."
        )
    if img1.ndim != 3:
        raise ValueError(f"img1/img2 must be 3D arrays (N,H,W). Got ndim={img1.ndim}.")
    if label.ndim != 1:
        raise ValueError(f"label must be 1D shape (N,). Got shape={label.shape}.")
    if img1.shape[0] != label.shape[0]:
        raise ValueError(
            f"First dimension of images must match label length. "
            f"Got N={img1.shape[0]} and label={label.shape[0]}."
        )

    # Similarity functions expect (N, D), so flatten image tensors once here.
    n_rows = img1.shape[0]
    a = img1.reshape(n_rows, -1).astype(np.float64, copy=False)
    b = img2.reshape(n_rows, -1).astype(np.float64, copy=False)
    return a, b, label
