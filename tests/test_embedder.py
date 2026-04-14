"""Unit tests for src/embedder.py.

All tests use synthetic numpy arrays — no real face images, no model weight downloads.
The FaceNet model is mocked so tests run instantly without network access.
"""

import numpy as np
import pytest
import torch

from src.embedder import get_embedding, preprocess_image


# ---------------------------------------------------------------------------
# Mock model — returns a fixed non-unit tensor so we can test L2 normalization.
# Always returns the same output regardless of input (sufficient for unit tests).
# ---------------------------------------------------------------------------


class _FixedMockModel:
    """Returns a fixed non-unit 512-dim vector regardless of input."""

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # Batch size 1, dim 512; not unit-length so normalization is exercised.
        vec = torch.arange(1, 513, dtype=torch.float32).unsqueeze(0)  # (1, 512)
        return vec

    def eval(self):
        return self


@pytest.fixture
def mock_model():
    return _FixedMockModel()


@pytest.fixture
def gray_image():
    """160x160 grayscale uint8 image with predictable pixel values."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(160, 160), dtype=np.uint8)


@pytest.fixture
def rgb_image():
    """160x160 RGB uint8 image."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(160, 160, 3), dtype=np.uint8)


@pytest.fixture
def small_gray_image():
    """64x64 grayscale float image in [0, 1] range — matches actual LFW npz format."""
    rng = np.random.default_rng(7)
    return rng.random(size=(64, 64)).astype(np.float32)  # values in [0, 1]


@pytest.fixture
def small_gray_image_255():
    """64x64 grayscale float image in [0, 255] range (tests the other float branch)."""
    rng = np.random.default_rng(7)
    return rng.random(size=(64, 64)).astype(np.float32) * 255.0


# ---------------------------------------------------------------------------
# preprocess_image tests
# ---------------------------------------------------------------------------


class TestPreprocessImage:
    def test_output_shape_from_grayscale(self, gray_image):
        tensor = preprocess_image(gray_image)
        assert tensor.shape == (1, 3, 160, 160), (
            f"Expected (1, 3, 160, 160), got {tensor.shape}"
        )

    def test_output_shape_from_rgb(self, rgb_image):
        tensor = preprocess_image(rgb_image)
        assert tensor.shape == (1, 3, 160, 160)

    def test_output_shape_from_float_image_0_1(self, small_gray_image):
        tensor = preprocess_image(small_gray_image)
        assert tensor.shape == (1, 3, 160, 160)

    def test_output_shape_from_float_image_0_255(self, small_gray_image_255):
        tensor = preprocess_image(small_gray_image_255)
        assert tensor.shape == (1, 3, 160, 160)

    def test_value_range(self, gray_image):
        tensor = preprocess_image(gray_image)
        assert float(tensor.min()) >= -1.0, f"Min below -1: {tensor.min()}"
        assert float(tensor.max()) <= 1.0, f"Max above +1: {tensor.max()}"

    def test_dtype_is_float(self, gray_image):
        tensor = preprocess_image(gray_image)
        assert tensor.dtype == torch.float32

    def test_deterministic_same_input(self, gray_image):
        t1 = preprocess_image(gray_image)
        t2 = preprocess_image(gray_image)
        assert torch.allclose(t1, t2), "preprocess_image is not deterministic"

    def test_different_inputs_differ(self, gray_image, rgb_image):
        t1 = preprocess_image(gray_image)
        t2 = preprocess_image(rgb_image)
        assert not torch.allclose(t1, t2), (
            "Expected different tensors for different inputs"
        )


# ---------------------------------------------------------------------------
# get_embedding tests
# ---------------------------------------------------------------------------


class TestGetEmbedding:
    def test_output_shape(self, gray_image, mock_model):
        emb = get_embedding(gray_image, mock_model)
        assert emb.shape == (512,), f"Expected shape (512,), got {emb.shape}"

    def test_output_is_l2_normalized(self, gray_image, mock_model):
        emb = get_embedding(gray_image, mock_model)
        norm = float(np.linalg.norm(emb))
        assert abs(norm - 1.0) < 1e-5, f"Embedding is not unit-norm: ||emb|| = {norm}"

    def test_output_dtype_is_float64(self, gray_image, mock_model):
        emb = get_embedding(gray_image, mock_model)
        assert emb.dtype == np.float64 or emb.dtype == np.float32, (
            f"Unexpected dtype: {emb.dtype}"
        )

    def test_deterministic(self, gray_image, mock_model):
        emb1 = get_embedding(gray_image, mock_model)
        emb2 = get_embedding(gray_image, mock_model)
        np.testing.assert_array_equal(
            emb1, emb2, err_msg="get_embedding is not deterministic"
        )

    def test_accepts_rgb_input(self, rgb_image, mock_model):
        emb = get_embedding(rgb_image, mock_model)
        assert emb.shape == (512,)

    def test_accepts_float_0_1_image(self, small_gray_image, mock_model):
        emb = get_embedding(small_gray_image, mock_model)
        assert emb.shape == (512,)
        norm = float(np.linalg.norm(emb))
        assert abs(norm - 1.0) < 1e-5

    def test_accepts_float_0_255_image(self, small_gray_image_255, mock_model):
        emb = get_embedding(small_gray_image_255, mock_model)
        assert emb.shape == (512,)
        norm = float(np.linalg.norm(emb))
        assert abs(norm - 1.0) < 1e-5
