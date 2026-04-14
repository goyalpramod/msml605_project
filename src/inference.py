"""Single inference entry point for face verification.

Wires together:
  src/embedder.py   — FaceNet preprocessing + embedding extraction
  src/similarity.py — cosine similarity (Milestone 1)
  src/confidence.py — sigmoid confidence computation

Interface contract (agreed with CLI / load-test author):

  verify_pair(img1, img2, threshold) -> {
    "score":      float,   # cosine similarity of L2-normalized embeddings
    "threshold":  float,   # threshold used for decision
    "decision":   bool,    # True = same person (score >= threshold)
    "confidence": float,   # sigmoid confidence in (0, 1); see src/confidence.py
    "latency_ms": float,   # wall-clock time for full inference in milliseconds
    "breakdown": {
      "preprocess_ms": float,  # time for preprocessing both images
      "embed_ms":      float,  # time for model forward pass on both images
      "score_ms":      float,  # time for similarity + confidence computation
    }
  }

The model is lazy-loaded on first call and cached as a module-level singleton
so that repeated calls (e.g. load testing) do not reload weights each time.
"""

import time

import numpy as np
import torch

from src.confidence import compute_confidence
from src.embedder import load_model, preprocess_image
from src.similarity import cosine_similarity

__all__ = ["verify_pair"]

_model = None


def _get_model():
    """Return the singleton FaceNet model, loading it on first call."""
    global _model
    if _model is None:
        _model = load_model()
    return _model


def verify_pair(img1: np.ndarray, img2: np.ndarray, threshold: float) -> dict:
    """Verify whether two face images belong to the same person.

    Args:
        img1: numpy array (H, W) grayscale or (H, W, 3) RGB — first face image.
        img2: numpy array (H, W) grayscale or (H, W, 3) RGB — second face image.
        threshold: cosine similarity decision boundary. Read from
                   configs/inference_config.json by the CLI; passed in directly here.

    Returns:
        Dict with keys: score, threshold, decision, confidence, latency_ms, breakdown.
        See module docstring for full schema.
    """
    model = _get_model()
    wall_start = time.perf_counter()

    # --- Stage 1: Preprocess both images ---
    t0 = time.perf_counter()
    tensor1 = preprocess_image(img1)
    tensor2 = preprocess_image(img2)
    t1 = time.perf_counter()
    preprocess_ms = (t1 - t0) * 1000.0

    # --- Stage 2: Embed both images (model forward pass + L2 normalize) ---
    t0 = time.perf_counter()
    with torch.no_grad():
        raw1 = model(tensor1).squeeze(0).cpu().numpy()  # (512,)
        raw2 = model(tensor2).squeeze(0).cpu().numpy()  # (512,)

    norm1 = float(np.linalg.norm(raw1))
    norm2 = float(np.linalg.norm(raw2))
    emb1 = raw1 / norm1 if norm1 > 0 else raw1
    emb2 = raw2 / norm2 if norm2 > 0 else raw2
    t1 = time.perf_counter()
    embed_ms = (t1 - t0) * 1000.0

    # --- Stage 3: Similarity + decision + confidence ---
    t0 = time.perf_counter()
    score = float(cosine_similarity(emb1[np.newaxis], emb2[np.newaxis])[0])
    decision = bool(score >= threshold)
    confidence = compute_confidence(score, threshold)
    t1 = time.perf_counter()
    score_ms = (t1 - t0) * 1000.0

    wall_end = time.perf_counter()
    latency_ms = (wall_end - wall_start) * 1000.0

    return {
        "score": score,
        "threshold": float(threshold),
        "decision": decision,
        "confidence": confidence,
        "latency_ms": latency_ms,
        "breakdown": {
            "preprocess_ms": preprocess_ms,
            "embed_ms": embed_ms,
            "score_ms": score_ms,
        },
    }
