"""Main evaluation script for the face verification pipeline.

Usage examples:
  python scripts/evaluate.py --pairs outputs/pairs_val.npz --mode sweep --similarity cosine --run-id run_01
  python scripts/evaluate.py --pairs outputs/pairs_val.npz --mode select --rule balanced_acc --run-id run_02
  python scripts/evaluate.py --pairs outputs/pairs_test.npz --mode final --threshold 0.55 --run-id run_03

Modes:
  sweep   Threshold sweep over the full range → saves ROC plot, logs metrics at every threshold.
  select  Apply a threshold-selection rule on the given split → saves confusion matrix plot.
  final   Evaluate at a locked threshold on the held-out split → logs final metrics.

Score direction (from configs/eval_config.json):
  cosine    → higher score = more likely same person  (scores used as-is)
  euclidean → lower score  = more likely same person  (scores negated before metric computation)
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")

from src.metrics import (
    balanced_accuracy,
    compute_confusion_matrix,
    compute_roc_points,
    equal_error_rate,
    f1_score_at_threshold,
    select_threshold,
)
from src.plotting import plot_confusion_matrix, plot_roc, plot_score_distribution
from src.similarity import cosine_similarity, euclidean_distance
from src.tracker import log_run
from src.validation import validate_pair_file, validate_scores, validate_threshold

CONFIG_PATH = "configs/eval_config.json"

SIMILARITY_FNS = {
    "cosine": cosine_similarity,
    "euclidean": euclidean_distance,
}


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


def load_pairs(pairs_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a pair file by path and return (img1_flat, img2_flat, labels)."""
    data = np.load(pairs_path)
    img1 = data["img1"]
    img2 = data["img2"]
    label = data["label"]
    n = img1.shape[0]
    a = img1.reshape(n, -1).astype(np.float64, copy=False)
    b = img2.reshape(n, -1).astype(np.float64, copy=False)
    return a, b, label


def orient_scores(scores: np.ndarray, similarity: str, config: dict) -> np.ndarray:
    """Negate euclidean scores so that higher always means 'more likely same person'."""
    direction = config["score_direction"][similarity]
    if direction == "lower_is_same":
        return -scores
    return scores


def save_scores(scores: np.ndarray, labels: np.ndarray, out_path: str) -> None:
    """Save scores and labels to .npz with pair_indices for error_analysis.py."""
    pair_indices = np.arange(len(scores))
    np.savez(out_path, scores=scores, labels=labels, pair_indices=pair_indices)


def run_sweep(args, config, scores, labels, thresholds, out_dir):
    """Threshold sweep: compute ROC, save plot, return metrics dict."""
    roc = compute_roc_points(scores, labels, thresholds)

    selected = select_threshold(
        scores, labels, thresholds, rule=config["threshold_rule"]
    )
    roc_path = os.path.join(out_dir, f"roc_{args.run_id}.png")
    plot_roc(roc["fpr"], roc["tpr"], thresholds, selected, roc_path)
    print(f"ROC plot saved to {roc_path}")

    # AUC via trapezoid rule (fpr is decreasing as threshold increases, so flip).
    fpr_sorted = roc["fpr"][::-1]
    tpr_sorted = roc["tpr"][::-1]
    auc = float(np.trapezoid(tpr_sorted, fpr_sorted))

    metrics = {
        "auc": round(auc, 4),
        "suggested_threshold": round(float(selected), 4),
        "threshold_rule_used": config["threshold_rule"],
    }
    return metrics, selected


def run_select(args, config, scores, labels, thresholds, out_dir):
    """Select threshold by rule, compute confusion matrix, save plot."""
    rule = args.rule or config["threshold_rule"]
    selected = select_threshold(scores, labels, thresholds, rule=rule)
    print(f"Selected threshold ({rule}): {selected:.4f}")

    cm = compute_confusion_matrix(scores, labels, selected)
    cm_path = os.path.join(out_dir, f"cm_{args.run_id}.png")
    plot_confusion_matrix(cm, cm_path)
    print(f"Confusion matrix saved to {cm_path}")

    dist_path = os.path.join(out_dir, f"score_dist_{args.run_id}.png")
    plot_score_distribution(scores, labels, selected, dist_path)
    print(f"Score distribution saved to {dist_path}")

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


def run_final(args, config, scores, labels, thresholds, out_dir):
    """Evaluate at a locked threshold and log final metrics."""
    threshold = args.threshold
    validate_threshold(threshold, (float(scores.min()), float(scores.max())))

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
    print(f"Final metrics at threshold {threshold:.4f}: {metrics}")
    return metrics, threshold


def main():
    parser = argparse.ArgumentParser(description="Face verification evaluation script.")
    parser.add_argument("--pairs", required=True, help="Path to .npz pair file.")
    parser.add_argument("--mode", required=True, choices=["sweep", "select", "final"])
    parser.add_argument(
        "--similarity", default="cosine", choices=["cosine", "euclidean"]
    )
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

    config = load_config()
    os.makedirs(args.out_dir, exist_ok=True)

    # --- Validate inputs ---
    validate_pair_file(args.pairs)

    a, b, labels = load_pairs(args.pairs)
    sim_fn = SIMILARITY_FNS[args.similarity]
    raw_scores = sim_fn(a, b)
    scores = orient_scores(raw_scores, args.similarity, config)

    validate_scores(scores, labels)

    # Save scores for downstream use (error_analysis.py).
    split_name = os.path.splitext(os.path.basename(args.pairs))[0].replace("pairs_", "")
    scores_path = os.path.join(args.out_dir, f"scores_{split_name}.npz")
    save_scores(scores, labels, scores_path)
    print(f"Scores saved to {scores_path}")

    # Build threshold grid from config.
    lo, hi = config["threshold_range"]
    steps = config["threshold_steps"]
    thresholds = np.linspace(lo, hi, steps)

    # --- Run selected mode ---
    if args.mode == "sweep":
        metrics, threshold = run_sweep(
            args, config, scores, labels, thresholds, args.out_dir
        )
    elif args.mode == "select":
        metrics, threshold = run_select(
            args, config, scores, labels, thresholds, args.out_dir
        )
    else:
        metrics, threshold = run_final(
            args, config, scores, labels, thresholds, args.out_dir
        )

    # --- Log run ---
    run_config = {
        "pairs": args.pairs,
        "similarity": args.similarity,
        "mode": args.mode,
        "score_direction": config["score_direction"][args.similarity],
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
