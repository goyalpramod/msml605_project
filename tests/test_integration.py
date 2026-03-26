"""Integration tests for the end-to-end evaluation pipeline.

Tests the full path: validate pair file → evaluate.py (sweep/select/final)
→ runs_log.json entry → plot files saved.  Uses synthetic fixtures so no
real data download is needed.  Must complete in < 5 seconds.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = str(Path(__file__).resolve().parent.parent)
REQUIRED_RUN_KEYS = {
    "run_id",
    "timestamp",
    "commit",
    "config",
    "metrics",
    "threshold",
    "note",
}


def _make_synthetic_pairs(path, n=30, seed=99):
    """Create a synthetic .npz pair file with valid schema."""
    rng = np.random.RandomState(seed)
    h, w = 62, 47
    img1 = rng.rand(n, h, w).astype(np.float32)
    img2 = rng.rand(n, h, w).astype(np.float32)
    label = np.array([1] * (n // 2) + [0] * (n // 2), dtype=np.int64)
    np.savez_compressed(path, img1=img1, img2=img2, label=label)


def _run_evaluate(args, out_dir):
    """Run scripts/evaluate.py via subprocess and return the CompletedProcess."""
    cmd = [sys.executable, "scripts/evaluate.py", "--out-dir", str(out_dir)] + args
    return subprocess.run(
        cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=30
    )


class TestEndToEndPipeline:
    """Full sweep → select → final pipeline on synthetic data."""

    def test_sweep_select_final(self, tmp_path):
        out_dir = tmp_path / "outputs"
        out_dir.mkdir()

        val_path = str(tmp_path / "pairs_val.npz")
        test_path = str(tmp_path / "pairs_test.npz")
        _make_synthetic_pairs(val_path, seed=99)
        _make_synthetic_pairs(test_path, seed=42)

        # --- sweep ---
        result = _run_evaluate(
            [
                "--pairs",
                val_path,
                "--mode",
                "sweep",
                "--similarity",
                "cosine",
                "--run-id",
                "int_run_01",
            ],
            out_dir,
        )
        assert result.returncode == 0, f"sweep failed:\n{result.stderr}"
        assert (out_dir / "roc_int_run_01.png").exists()

        scores_path = out_dir / "scores_val.npz"
        assert scores_path.exists()
        scores_data = np.load(str(scores_path))
        assert set(scores_data.files) == {"scores", "labels", "pair_indices"}

        log_path = str(out_dir / "runs_log.json")
        with open(log_path) as f:
            runs = json.load(f)
        assert len(runs) == 1
        assert REQUIRED_RUN_KEYS.issubset(runs[0].keys())

        # --- select ---
        result = _run_evaluate(
            [
                "--pairs",
                val_path,
                "--mode",
                "select",
                "--rule",
                "balanced_acc",
                "--run-id",
                "int_run_02",
            ],
            out_dir,
        )
        assert result.returncode == 0, f"select failed:\n{result.stderr}"
        assert (out_dir / "cm_int_run_02.png").exists()
        assert (out_dir / "score_dist_int_run_02.png").exists()

        with open(log_path) as f:
            runs = json.load(f)
        assert len(runs) == 2
        threshold = runs[1]["threshold"]
        assert isinstance(threshold, float)

        # --- final ---
        result = _run_evaluate(
            [
                "--pairs",
                test_path,
                "--mode",
                "final",
                "--threshold",
                str(threshold),
                "--run-id",
                "int_run_03",
            ],
            out_dir,
        )
        assert result.returncode == 0, f"final failed:\n{result.stderr}"
        assert (out_dir / "cm_int_run_03.png").exists()
        assert (out_dir / "score_dist_int_run_03.png").exists()

        with open(log_path) as f:
            runs = json.load(f)
        assert len(runs) == 3
        for run in runs:
            assert REQUIRED_RUN_KEYS.issubset(run.keys())

    def test_bad_input_rejected(self, tmp_path):
        """evaluate.py should fail on a pair file missing required keys."""
        out_dir = tmp_path / "outputs"
        out_dir.mkdir()

        bad_path = str(tmp_path / "bad_pairs.npz")
        np.savez(bad_path, img1=np.zeros((5, 62, 47)), img2=np.zeros((5, 62, 47)))

        result = _run_evaluate(
            [
                "--pairs",
                bad_path,
                "--mode",
                "sweep",
                "--similarity",
                "cosine",
                "--run-id",
                "bad_run",
            ],
            out_dir,
        )
        assert result.returncode != 0
