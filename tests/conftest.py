import numpy as np
import pytest


@pytest.fixture()
def synthetic_pairs_npz(tmp_path):
    """Create a temporary .npz file with valid pair schema (img1, img2, label).

    Returns (npz_path, n_pairs) where n_pairs=30.
    """
    rng = np.random.RandomState(99)
    n = 30
    h, w = 62, 47
    img1 = rng.rand(n, h, w).astype(np.float32)
    img2 = rng.rand(n, h, w).astype(np.float32)
    label = np.array([1] * (n // 2) + [0] * (n // 2), dtype=np.int64)

    path = tmp_path / "pairs_synthetic.npz"
    np.savez_compressed(path, img1=img1, img2=img2, label=label)
    return str(path), n


@pytest.fixture()
def tmp_output_dir(tmp_path):
    """Provide a clean temporary directory for test outputs."""
    out = tmp_path / "outputs"
    out.mkdir()
    return str(out)
