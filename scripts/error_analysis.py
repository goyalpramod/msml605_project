"""Error analysis script for the face verification pipeline.

Identifies two error slices and saves representative examples + summary JSON.

Slice 1 — Boundary false positives:
    Different-identity pairs with score within +/-0.05 of the decision threshold.

Slice 2 — High-variation false negatives:
    Same-identity pairs in the bottom 20th percentile of similarity scores.

Usage:
    python scripts/error_analysis.py \
        --scores outputs/scores_test.npz \
        --pairs outputs/pairs_test.npz \
        --threshold 0.55 \
        --output-dir outputs/error_analysis/
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MAX_EXAMPLES = 6  # representative image pairs to save per slice
BOUNDARY_MARGIN = 0.05


def _load_scores(scores_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load scores and labels from an npz file."""
    data = np.load(scores_path)
    return data["scores"], data["labels"]


def _load_pairs(pairs_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load image pair arrays from an npz file."""
    data = np.load(pairs_path)
    return data["img1"], data["img2"]


def _save_pair_image(
    img1: np.ndarray, img2: np.ndarray, score: float, label: int, save_path: str
) -> None:
    """Save a side-by-side image pair as a PNG."""
    fig, axes = plt.subplots(1, 2, figsize=(5, 3))
    for ax, img, title in zip(axes, [img1, img2], ["Image 1", "Image 2"]):
        ax.imshow(img, cmap="gray")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.suptitle(
        f"Score: {score:.4f} | Label: {'same' if label == 1 else 'diff'}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)


def analyze_slice1_boundary_fp(
    scores: np.ndarray,
    labels: np.ndarray,
    img1: np.ndarray,
    img2: np.ndarray,
    threshold: float,
    output_dir: str,
) -> dict:
    """Slice 1: boundary false positives — different-identity pairs near threshold.

    These are pairs the model incorrectly accepts (score >= threshold) despite
    being different people, and they fall within +/-margin of the threshold.
    """
    mask = (
        (labels == 0)  # different-identity pairs
        & (scores >= threshold - BOUNDARY_MARGIN)
        & (scores <= threshold + BOUNDARY_MARGIN)
    )
    indices = np.where(mask)[0]

    slice_dir = os.path.join(output_dir, "slice1")
    os.makedirs(slice_dir, exist_ok=True)

    # Save representative examples (sorted by closeness to threshold)
    sorted_by_closeness = indices[np.argsort(np.abs(scores[indices] - threshold))]
    examples_saved = 0
    for idx in sorted_by_closeness[:MAX_EXAMPLES]:
        _save_pair_image(
            img1[idx],
            img2[idx],
            float(scores[idx]),
            int(labels[idx]),
            os.path.join(slice_dir, f"example_{examples_saved}.png"),
        )
        examples_saved += 1

    return {
        "slice_id": "slice1_boundary_fp",
        "definition": (
            f"Different-identity pairs with score within +/-{BOUNDARY_MARGIN} "
            f"of threshold {threshold:.4f}"
        ),
        "count": int(len(indices)),
        "fraction": float(len(indices) / len(labels)) if len(labels) > 0 else 0.0,
        "examples_saved": examples_saved,
        "hypothesis": (
            "Visually similar faces (similar age, lighting, pose) fool "
            "pixel-level similarity metrics near the decision boundary."
        ),
    }


def analyze_slice2_highvar_fn(
    scores: np.ndarray,
    labels: np.ndarray,
    img1: np.ndarray,
    img2: np.ndarray,
    threshold: float,
    output_dir: str,
) -> dict:
    """Slice 2: high-variation false negatives — same-identity pairs with low scores.

    These are same-person pairs the model incorrectly rejects because their
    similarity score is very low (bottom 20th percentile of same-person scores).
    """
    same_mask = labels == 1
    same_scores = scores[same_mask]

    if len(same_scores) == 0:
        return {
            "slice_id": "slice2_highvar_fn",
            "definition": "Same-identity pairs in the bottom 20th percentile of scores",
            "count": 0,
            "fraction": 0.0,
            "examples_saved": 0,
            "hypothesis": "N/A — no same-identity pairs found.",
        }

    percentile_20 = np.percentile(same_scores, 20)
    # Same-identity pairs with scores below the 20th percentile
    mask = same_mask & (scores <= percentile_20)
    indices = np.where(mask)[0]

    slice_dir = os.path.join(output_dir, "slice2")
    os.makedirs(slice_dir, exist_ok=True)

    # Compute per-pair pixel variance as a variation proxy
    pair_variances = []
    for idx in indices:
        var1 = float(np.var(img1[idx]))
        var2 = float(np.var(img2[idx]))
        pair_variances.append((var1 + var2) / 2)

    # Save representative examples (sorted by lowest score first)
    sorted_by_score = indices[np.argsort(scores[indices])]
    examples_saved = 0
    for idx in sorted_by_score[:MAX_EXAMPLES]:
        _save_pair_image(
            img1[idx],
            img2[idx],
            float(scores[idx]),
            int(labels[idx]),
            os.path.join(slice_dir, f"example_{examples_saved}.png"),
        )
        examples_saved += 1

    return {
        "slice_id": "slice2_highvar_fn",
        "definition": (
            f"Same-identity pairs in the bottom 20th percentile of scores "
            f"(score <= {percentile_20:.4f})"
        ),
        "count": int(len(indices)),
        "fraction": float(len(indices) / len(labels)) if len(labels) > 0 else 0.0,
        "examples_saved": examples_saved,
        "mean_pair_pixel_variance": float(np.mean(pair_variances))
        if pair_variances
        else 0.0,
        "hypothesis": (
            "Extreme pose, lighting, or expression changes destroy pixel-level "
            "representation, causing same-person pairs to have very low similarity."
        ),
    }


def run_error_analysis(
    scores_path: str,
    pairs_path: str,
    threshold: float,
    output_dir: str,
) -> None:
    """Run all error slices and write slice_summary.json."""
    scores, labels = _load_scores(scores_path)
    img1, img2 = _load_pairs(pairs_path)

    os.makedirs(output_dir, exist_ok=True)

    slice1 = analyze_slice1_boundary_fp(
        scores, labels, img1, img2, threshold, output_dir
    )
    slice2 = analyze_slice2_highvar_fn(
        scores, labels, img1, img2, threshold, output_dir
    )

    summary = {"threshold": threshold, "slices": [slice1, slice2]}
    summary_path = os.path.join(output_dir, "slice_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Error analysis complete. Summary: {summary_path}")
    for s in [slice1, slice2]:
        print(
            f"  {s['slice_id']}: {s['count']} pairs "
            f"({s['fraction']:.1%} of total), {s['examples_saved']} examples saved"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Error analysis for face verification")
    parser.add_argument(
        "--scores",
        type=str,
        required=True,
        help="Path to scores npz (keys: scores, labels)",
    )
    parser.add_argument(
        "--pairs",
        type=str,
        required=True,
        help="Path to pairs npz (keys: img1, img2)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
        help="Decision threshold",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/error_analysis/",
        help="Directory to save error analysis outputs",
    )
    args = parser.parse_args()

    run_error_analysis(
        scores_path=args.scores,
        pairs_path=args.pairs,
        threshold=args.threshold,
        output_dir=args.output_dir,
    )
