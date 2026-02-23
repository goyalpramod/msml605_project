# Milestone 1 — Face Verification (LFW)
**Due: Thu Feb 26, 2026 11:59pm**

---

## Context
Build a reproducible project skeleton, ingest the LFW (Labeled Faces in the Wild) dataset, generate deterministic verification pairs, and implement vectorized similarity scoring (cosine + Euclidean). Final deliverable is a tagged Git release (`v0.1`) that reproduces all outputs from a clean clone.

---

## Pramod's Tasks

### 1. Repo Structure & .gitignore
Set up the agreed folder skeleton:
```
src/
scripts/
configs/
tests/
notebooks/
data/         # gitignored
outputs/      # gitignored
```
- Add a proper `.gitignore` (ignore `data/`, `outputs/`, `__pycache__`, `.env`, `*.pyc`, `*.egg-info`)
- Commit the skeleton with an empty `.gitkeep` in each folder

### 2. LFW Ingestion Script (`scripts/ingest_lfw.py`)
- Download LFW using `sklearn.datasets.fetch_lfw_pairs` (or `fetch_lfw_people`) with a fixed seed
- Produce a **dataset manifest** as `outputs/manifest.json` containing at minimum:
  ```json
  {
    "seed": 42,
    "split_policy": "10fold",
    "train_count": ...,
    "test_count": ...,
    "total_identities": ...,
    "image_shape": [62, 47]
  }
  ```
- Script must be fully deterministic (same manifest on every run)
- Usage: `python scripts/ingest_lfw.py --seed 42`

### 3. Benchmark Script (`scripts/benchmark.py`)
- Compare Python loop vs NumPy vectorized for both cosine and Euclidean (imports from `src/similarity.py`)
- Print timing results and assert correctness (outputs match within tolerance `1e-6`)
- Output example:
  ```
  Cosine  | loop: 1.23s | numpy: 0.01s | match: True
  Euclid  | loop: 1.10s | numpy: 0.01s | match: True
  ```

### 4. Git Tag & Release (`v0.1`)
- Ensure all Milestone 1 outputs are reproducible from a clean clone
- Tag the final commit: `git tag v0.1 && git push origin v0.1`
- Verify by cloning fresh and running the How-to-run commands

---

## Arun's Tasks

### 1. Pair Generation Script (`scripts/generate_pairs.py`)
- Use the ingested LFW data to produce deterministic **positive pairs** (same identity) and **negative pairs** (different identities)
- Save splits to `outputs/pairs_train.npz` and `outputs/pairs_test.npz`
- Each file should contain: `img1`, `img2`, `label` (1=same, 0=different)
- Must be seeded and deterministic: `python scripts/generate_pairs.py --seed 42`

### 2. Vectorized Similarity Module (`src/similarity.py`)
Implement the following using NumPy (no loops in the vectorized versions):
```python
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray: ...
def euclidean_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray: ...
```
- Both accept batches of shape `(N, D)` and return `(N,)`
- Also include Python loop equivalents (used by the benchmark script)

### 3. README.md
Must include:
- Brief project overview (what it does, dataset used)
- **How to run** section with copy-pastable commands:
  ```bash
  uv pip install -r requirements.txt
  python scripts/ingest_lfw.py --seed 42
  python scripts/generate_pairs.py --seed 42
  python scripts/benchmark.py
  ```

---

## Shared

### `requirements.txt`
Both contribute as you add dependencies. Pin versions. Minimum: `scikit-learn`, `numpy`, `scipy`.

---

## Definition of Done (Milestone 1)
- [ ] Repo has clean structure with `.gitignore`
- [ ] `ingest_lfw.py` runs deterministically and produces `manifest.json`
- [ ] `generate_pairs.py` produces saved pair splits (`pairs_train.npz`, `pairs_test.npz`)
- [ ] `src/similarity.py` has vectorized cosine + Euclidean
- [ ] `benchmark.py` shows timing + correctness check
- [ ] README with How-to-run section
- [ ] Tagged `v0.1` that reproduces everything from clean clone
