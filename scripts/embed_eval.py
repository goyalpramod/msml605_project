"""Embedding-based evaluation script for face verification.

Applies the same threshold-selection discipline as scripts/evaluate.py (Milestone 2),
but replaces raw pixel similarity with FaceNet embedding cosine similarity.

Workflow for Milestone 3 threshold re-selection:

  # Step 1: Sweep thresholds on val split → identifies best threshold (run_06)
  python scripts/embed_eval.py \\
      --pairs outputs/pairs_val.npz --mode sweep --run-id run_06

  # Step 2: Report final metrics on test split at the selected threshold (run_07)
  python scripts/embed_eval.py \\
      --pairs outputs/pairs_test.npz --mode final --threshold <VALUE> --run-id run_07

Both runs are logged to outputs/runs_log.json.

Modes:
  sweep   Threshold sweep over [-1, 1] → saves ROC plot, logs AUC + suggested threshold.
  select  Apply balanced_acc selection rule on given split → saves confusion matrix.
  final   Evaluate at a fixed threshold on held-out split → logs final metrics.

Note: This script loads the FaceNet model and runs inference on all pairs in the
provided .npz file. On CPU, expect ~1–2 s per pair on first run (model load)
then ~100–400 ms per pair thereafter. Val (~457 pairs) takes a few minutes.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, ".")

from src.embedder import get_embedding, load_model
from src.metrics import (
    balanced_accuracy,
    compute_confusion_matrix,
    compute_roc_points,
    equal_error_rate,
    f1_score_at_threshold,
    select_threshold,
)
from src.plotting import plot_confusion_matrix, plot_roc, plot_score_distribution
from src.similarity import cosine_similarity
from src.tracker import log_run

# Threshold sweep range for embedding cosine similarity.
# Embeddings are L2-normalized so cosine is in [-1, 1]; face pairs cluster in [0.2, 1.0].
_THRESHOLD_LO = -1.0
_THRESHOLD_HI = 1.0
_THRESHOLD_STEPS = 200


def load_pairs(pairs_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load raw image pairs from a .npz file.

    Returns:
        img1: np.ndarray of shape (N, H, W) — first images.
        img2: np.ndarray of shape (N, H, W) — second images.
        labels: np.ndarray of shape (N,) — 1 = same person, 0 = different.
    """
    if not os.path.exists(pairs_path):
        raise FileNotFoundError(f"Pair file not found: {pairs_path}")
    data = np.load(pairs_path)
    required = {"img1", "img2", "label"}
    if not required.issubset(data.files):
        raise ValueError(f"Expected keys {sorted(required)}, got {sorted(data.files)}")
    return data["img1"], data["img2"], data["label"]


def compute_embedding_scores(
    img1: np.ndarray,
    img2: np.ndarray,
    model,
    verbose: bool = True,
) -> np.ndarray:
    """Extract FaceNet embeddings for all pairs and compute cosine similarity.

    Args:
        img1: (N, H, W) array of first face images.
        img2: (N, H, W) array of second face images.
        model: InceptionResnetV1 from load_model().
        verbose: Print progress every 50 pairs.

    Returns:
        np.ndarray of shape (N,) — cosine similarity of embedding pairs.
    """
    n = img1.shape[0]
    scores = np.zeros(n, dtype=np.float64)

    for i in range(n):
        emb1 = get_embedding(img1[i], model)
        emb2 = get_embedding(img2[i], model)
        scores[i] = float(cosine_similarity(emb1[np.newaxis], emb2[np.newaxis])[0])

        if verbose and (i + 1) % 50 == 0:
            print(f"  Embedded {i + 1}/{n} pairs...")

    return scores


def run_sweep(args, scores, labels, thresholds, out_dir):
    roc = compute_roc_points(scores, labels, thresholds)
    selected = select_threshold(scores, labels, thresholds, rule="balanced_acc")

    roc_path = os.path.join(out_dir, f"roc_{args.run_id}.png")
    plot_roc(roc["fpr"], roc["tpr"], thresholds, selected, roc_path)
    print(f"ROC plot saved to {roc_path}")

    fpr_sorted = roc["fpr"][::-1]
    tpr_sorted = roc["tpr"][::-1]
    auc = float(np.trapezoid(tpr_sorted, fpr_sorted))

    metrics = {
        "auc": round(auc, 4),
        "suggested_threshold": round(float(selected), 4),
        "threshold_rule_used": "balanced_acc",
    }
    print(f"Suggested threshold (balanced_acc): {selected:.6f}")
    print(f"AUC: {auc:.4f}")
    return metrics, selected


def run_select(args, scores, labels, thresholds, out_dir):
    rule = args.rule or "balanced_acc"
    selected = select_threshold(scores, labels, thresholds, rule=rule)
    print(f"Selected threshold ({rule}): {selected:.6f}")

    cm = compute_confusion_matrix(scores, labels, selected)
    cm_path = os.path.join(out_dir, f"cm_{args.run_id}.png")
    plot_confusion_matrix(cm, cm_path)
    print(f"Confusion matrix saved to {cm_path}")

    dist_path = os.path.join(out_dir, f"score_dist_{args.run_id}.png")
    plot_score_distribution(scores, labels, selected, dist_path)

    ba = balanced_accuracy(scores, labels, selected)
    f1 = f1_score_at_threshold(scores, labels, selected)
    eer_val, _ = equal_error_rate(scores, labels, thresholds)

    metrics = {
        "balanced_acc": round(ba, 4),
        "f1": round(f1, 4),
        "eer": round(eer_val, 4),
        "rule_used": rule,
        **{k: int(v) for k, v in cm.items()},
    }
    return metrics, selected


def run_final(args, scores, labels, thresholds, out_dir):
    threshold = args.threshold
    if threshold is None:
        raise ValueError("--mode final requires --threshold")

    cm = compute_confusion_matrix(scores, labels, threshold)
    cm_path = os.path.join(out_dir, f"cm_{args.run_id}.png")
    plot_confusion_matrix(cm, cm_path)

    dist_path = os.path.join(out_dir, f"score_dist_{args.run_id}.png")
    plot_score_distribution(scores, labels, threshold, dist_path)

    ba = balanced_accuracy(scores, labels, threshold)
    f1 = f1_score_at_threshold(scores, labels, threshold)
    eer_val, _ = equal_error_rate(scores, labels, thresholds)

    metrics = {
        "balanced_acc": round(ba, 4),
        "f1": round(f1, 4),
        "eer": round(eer_val, 4),
        **{k: int(v) for k, v in cm.items()},
    }
    print(f"Final metrics at threshold {threshold:.6f}: {metrics}")
    return metrics, threshold


def main():
    parser = argparse.ArgumentParser(
        description="Embedding-based face verification evaluation."
    )
    parser.add_argument("--pairs", required=True, help="Path to .npz pair file.")
    parser.add_argument("--mode", required=True, choices=["sweep", "select", "final"])
    parser.add_argument("--run-id", required=True, dest="run_id")
    parser.add_argument(
        "--rule",
        default=None,
        choices=["balanced_acc", "f1", "eer"],
        help="Threshold selection rule (used in --mode select).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Locked threshold (used in --mode final).",
    )
    parser.add_argument("--out-dir", default="outputs", dest="out_dir")
    parser.add_argument("--note", default="", help="Short description of this run.")
    args = parser.parse_args()

    if args.mode == "final" and args.threshold is None:
        parser.error("--mode final requires --threshold")

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading pairs from {args.pairs} ...")
    img1, img2, labels = load_pairs(args.pairs)
    n = img1.shape[0]
    print(f"Loaded {n} pairs.")

    print("Loading FaceNet model (downloads weights on first run) ...")
    model = load_model()
    print("Model loaded. Extracting embeddings ...")

    scores = compute_embedding_scores(img1, img2, model, verbose=True)
    print(f"Scores computed. Range: [{scores.min():.4f}, {scores.max():.4f}]")

    # Save scores for potential downstream use.
    split_name = os.path.splitext(os.path.basename(args.pairs))[0].replace("pairs_", "")
    scores_path = os.path.join(args.out_dir, f"scores_emb_{split_name}.npz")
    np.savez(scores_path, scores=scores, labels=labels)
    print(f"Embedding scores saved to {scores_path}")

    thresholds = np.linspace(_THRESHOLD_LO, _THRESHOLD_HI, _THRESHOLD_STEPS)

    if args.mode == "sweep":
        metrics, threshold = run_sweep(args, scores, labels, thresholds, args.out_dir)
    elif args.mode == "select":
        metrics, threshold = run_select(args, scores, labels, thresholds, args.out_dir)
    else:
        metrics, threshold = run_final(args, scores, labels, thresholds, args.out_dir)

    run_config = {
        "pairs": args.pairs,
        "similarity": "cosine",
        "model": "facenet_vggface2",
        "embedding_dim": 512,
        "mode": args.mode,
        "score_direction": "higher_is_same",
    }
    log_run(
        run_id=args.run_id,
        config=run_config,
        metrics=metrics,
        threshold=float(threshold),
        note=args.note,
        log_path=os.path.join(args.out_dir, "runs_log.json"),
    )
    print(
        f"Run {args.run_id!r} logged to {os.path.join(args.out_dir, 'runs_log.json')}"
    )


if __name__ == "__main__":
    main()
