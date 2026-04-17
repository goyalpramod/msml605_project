"""Smoke and integration tests for the M3 inference CLI path.

Unit tests mock the FaceNet model (same _FixedMockModel pattern as
tests/test_embedder.py) so verify_pair runs without downloading weights.

The CLI smoke test runs scripts/verify.py as a subprocess and hits the
real FaceNet model; it relies on facenet-pytorch's cached weights and
completes in a few seconds on warm runs.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import src.inference as inference_module
from src.inference import verify_pair

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VERIFY_SCRIPT = PROJECT_ROOT / "scripts" / "verify.py"


class _FixedMockModel:
    """Returns a fixed non-unit 512-dim tensor regardless of input."""

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        vec = torch.arange(1, 513, dtype=torch.float32).unsqueeze(0)
        return vec

    def eval(self):
        return self


@pytest.fixture
def mock_inference_model(monkeypatch):
    """Inject a fixed mock into the inference module's singleton slot."""
    monkeypatch.setattr(inference_module, "_model", _FixedMockModel())
    yield


@pytest.fixture
def rgb_image_pair():
    rng = np.random.default_rng(42)
    img1 = rng.integers(0, 256, size=(160, 160, 3), dtype=np.uint8)
    img2 = rng.integers(0, 256, size=(160, 160, 3), dtype=np.uint8)
    return img1, img2


class TestVerifyPairIntegration:
    def test_returns_required_keys(self, rgb_image_pair, mock_inference_model):
        img1, img2 = rgb_image_pair
        result = verify_pair(img1, img2, threshold=0.5)

        required = {"score", "threshold", "decision", "confidence", "latency_ms", "breakdown"}
        assert required.issubset(result.keys())

        assert isinstance(result["score"], float)
        assert isinstance(result["threshold"], float)
        assert isinstance(result["decision"], bool)
        assert isinstance(result["confidence"], float)
        assert isinstance(result["latency_ms"], float)
        assert isinstance(result["breakdown"], dict)

    def test_breakdown_has_stage_timings(self, rgb_image_pair, mock_inference_model):
        img1, img2 = rgb_image_pair
        result = verify_pair(img1, img2, threshold=0.5)

        breakdown = result["breakdown"]
        assert {"preprocess_ms", "embed_ms", "score_ms"} == set(breakdown.keys())
        for v in breakdown.values():
            assert isinstance(v, float)
            assert v >= 0.0

    def test_confidence_in_open_unit_interval(self, rgb_image_pair, mock_inference_model):
        img1, img2 = rgb_image_pair
        result = verify_pair(img1, img2, threshold=0.5)
        assert 0.0 < result["confidence"] < 1.0

    def test_score_in_valid_cosine_range(self, rgb_image_pair, mock_inference_model):
        img1, img2 = rgb_image_pair
        result = verify_pair(img1, img2, threshold=0.5)
        assert -1.0 <= result["score"] <= 1.0

    def test_decision_matches_threshold(self, rgb_image_pair, mock_inference_model):
        img1, img2 = rgb_image_pair
        # Fixed mock → score is deterministic (both images embed to same vector → score = 1.0)
        below = verify_pair(img1, img2, threshold=-0.5)
        above = verify_pair(img1, img2, threshold=1.5)
        assert below["decision"] is True
        assert above["decision"] is False


class TestVerifyCli:
    def test_cli_smoke_exits_zero(self, tmp_path):
        """Real FaceNet model run via subprocess. Uses cached weights after first run."""
        rng = np.random.default_rng(123)
        arr1 = rng.integers(0, 256, size=(160, 160, 3), dtype=np.uint8)
        arr2 = rng.integers(0, 256, size=(160, 160, 3), dtype=np.uint8)

        p1 = tmp_path / "a.png"
        p2 = tmp_path / "b.png"
        Image.fromarray(arr1).save(p1)
        Image.fromarray(arr2).save(p2)

        result = subprocess.run(
            [sys.executable, str(VERIFY_SCRIPT), "--img1", str(p1), "--img2", str(p2)],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=60,
        )

        assert result.returncode == 0, (
            f"CLI exited with {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        for label in ("Score:", "Threshold:", "Decision:", "Confidence:", "Latency:"):
            assert label in result.stdout, f"Missing '{label}' in CLI output:\n{result.stdout}"

    def test_cli_bad_input_exits_nonzero(self, tmp_path):
        """Nonexistent image paths → non-zero exit + error on stderr. No model load."""
        result = subprocess.run(
            [
                sys.executable,
                str(VERIFY_SCRIPT),
                "--img1",
                str(tmp_path / "nope1.png"),
                "--img2",
                str(tmp_path / "nope2.png"),
            ],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=30,
        )
        assert result.returncode != 0
        assert "Error" in result.stderr or "not found" in result.stderr.lower()

    def test_cli_missing_args_exits_nonzero(self):
        """Neither --batch nor --img1/--img2 → error."""
        result = subprocess.run(
            [sys.executable, str(VERIFY_SCRIPT)],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=10,
        )
        assert result.returncode != 0
