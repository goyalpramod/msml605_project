"""Visualization functions for the face verification evaluation pipeline.

All functions are pure side-effect-free except writing the figure at save_path.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")  # non-interactive backend for CI/headless use
import matplotlib.pyplot as plt


def plot_roc(
    fpr: np.ndarray,
    tpr: np.ndarray,
    thresholds: np.ndarray,
    selected_threshold: float,
    save_path: str,
) -> None:
    """Plot ROC curve with the selected operating point highlighted.

    Args:
        fpr: False positive rates at each threshold.
        tpr: True positive rates at each threshold.
        thresholds: Threshold values corresponding to fpr/tpr.
        selected_threshold: The chosen threshold to mark on the curve.
        save_path: File path to save the figure (e.g. .png).
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, linewidth=2, label="ROC curve")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random baseline")

    # Mark the selected operating point
    idx = np.argmin(np.abs(thresholds - selected_threshold))
    ax.scatter(
        fpr[idx],
        tpr[idx],
        color="red",
        s=100,
        zorder=5,
        label=f"Threshold = {selected_threshold:.3f}",
    )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm_dict: dict, save_path: str) -> None:
    """Plot a 2x2 confusion matrix heatmap.

    Args:
        cm_dict: Dictionary with keys TP, FP, TN, FN (integer counts).
        save_path: File path to save the figure.
    """
    tp = cm_dict["TP"]
    fp = cm_dict["FP"]
    tn = cm_dict["TN"]
    fn = cm_dict["FN"]

    matrix = np.array([[tn, fp], [fn, tp]])
    labels = np.array([[f"TN\n{tn}", f"FP\n{fp}"], [f"FN\n{fn}", f"TP\n{tp}"]])

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(matrix, cmap="Blues", aspect="equal")

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                labels[i, j],
                ha="center",
                va="center",
                fontsize=14,
                color="white" if matrix[i, j] > matrix.max() / 2 else "black",
            )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted Negative", "Predicted Positive"])
    ax.set_yticklabels(["Actual Negative", "Actual Positive"])
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_score_distribution(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    save_path: str,
) -> None:
    """Plot overlapping histograms of similarity scores by class.

    Args:
        scores: Similarity scores for each pair.
        labels: Binary labels (1=same, 0=different).
        threshold: Decision threshold to draw as a vertical line.
        save_path: File path to save the figure.
    """
    same_scores = scores[labels == 1]
    diff_scores = scores[labels == 0]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(same_scores, bins=50, alpha=0.6, label="Same person", color="steelblue")
    ax.hist(diff_scores, bins=50, alpha=0.6, label="Different person", color="salmon")
    ax.axvline(
        threshold,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Threshold = {threshold:.3f}",
    )

    ax.set_xlabel("Similarity Score")
    ax.set_ylabel("Count")
    ax.set_title("Score Distribution by Class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
