# Milestone 3 — Embedding-Based Inference & Deployment
**Due: TBD 11:59pm**

---

## Context
Upgrade the face verifier from the pixel-level baseline to an embedding-based representation (FaceNet / InceptionResnetV1), expose it through a clean CLI, package it in Docker, and measure runtime behavior under concurrent usage. The deterministic pair files and threshold-selection discipline from Milestones 1 and 2 are carried forward unchanged.

Key terms for this milestone:
- **Embedding**: fixed-length face vector produced by FaceNet; similarity is now computed from these, not raw pixels
- **Calibrated confidence**: a documented value derived from score + threshold (formula TBD — must be explained in README and report)
- **Load test**: script that runs inference under multiple workers and reports throughput + p95 latency (CPU-only, feasible local workload)
- **Smoke test**: tiny end-to-end check that the main CLI path completes without error

---

## Interface Contract (agree before coding)

Pramod defines this function; Arun calls it in the CLI and load test:

```python
# src/inference.py
def verify_pair(img1: np.ndarray, img2: np.ndarray, threshold: float) -> dict:
    """
    Returns:
      {
        "score":      float,   # cosine similarity of embeddings
        "threshold":  float,   # threshold used for decision
        "decision":   bool,    # True = same person
        "confidence": float,   # calibrated confidence (TBD formula, documented)
        "latency_ms": float,   # wall-clock time for full inference
        "breakdown":  {        # stage-level latency (needed for Milestone 4)
          "preprocess_ms": float,
          "embed_ms":      float,
          "score_ms":      float,
        }
      }
    """
```

Config contract:
```json
// configs/inference_config.json  (Arun creates, Pramod populates threshold)
{
  "embedding_model": "facenet",        // Pramod fills
  "embedding_dim": 512,                // Pramod fills
  "threshold": null,                   // Pramod fills after re-selection
  "score_direction": "higher_is_same", // cosine from embeddings
  "confidence_formula": "TBD",         // Pramod fills + documents
  "load_test": {
    "num_pairs": 100,
    "num_workers": 4,
    "seed": 42
  }
}
```

---

## Pramod's Tasks

### 1. `src/embedder.py` — FaceNet Embedding Module
Implement deterministic preprocessing + embedding extraction:
```python
def preprocess_image(img: np.ndarray) -> torch.Tensor:
    """Resize to 160×160, normalize to [-1, 1], deterministic."""

def get_embedding(img: np.ndarray, model) -> np.ndarray:
    """Return L2-normalized 512-dim FaceNet embedding."""

def load_model() -> InceptionResnetV1:
    """Load pretrained FaceNet (facenet-pytorch, pretrained='vggface2')."""
```
- Model weights must be downloaded deterministically (fixed source, pinned version)
- Document embedding dimensionality and all preprocessing assumptions in docstrings
- Add FaceNet model choice rationale note in `configs/inference_config.json`

### 2. `src/confidence.py` — Confidence Computation Module
```python
def compute_confidence(score: float, threshold: float) -> float:
    """
    Compute calibrated confidence. Formula TBD — must be:
    - Reproducible and deterministic
    - Output range clearly stated (e.g. [0, 1])
    - Documented in README and report
    """
```
- Options: sigmoid of normalized distance, linear clamp, isotonic regression — pick one and document it

### 3. `src/inference.py` — Single Inference Entry Point
Wire together embedder + similarity + threshold + confidence into the interface contract function above:
```python
def verify_pair(img1: np.ndarray, img2: np.ndarray, threshold: float) -> dict: ...
```
- Stage-level timing must be measured separately (preprocess / embed / score) — needed for Milestone 4 profiling
- Import from `src/embedder.py`, `src/similarity.py` (M1), `src/confidence.py`

### 4. Threshold Re-selection for Embedding-Based System
Once embeddings are working, re-run the M2 threshold-selection discipline:
- Use `scripts/evaluate.py` (M2) with embedding scores on the val split
- Apply the same rule as M2 (balanced accuracy)
- Record new threshold in `configs/inference_config.json`
- Log as `run_06` in `outputs/runs_log.json` (sweep on embedding scores, val split)
- Log as `run_07` (final report on test split at new embedding threshold)
- **Do not tune threshold on the test split**

### 5. Unit Tests (`tests/test_embedder.py`, `tests/test_confidence.py`)
- `test_embedder.py`:
  - Test `preprocess_image` output shape and value range
  - Test `get_embedding` returns normalized vector of correct dim (use a random 160×160 toy image — no real data download in tests)
  - Test determinism: same input → same embedding
- `test_confidence.py`:
  - Test confidence output range
  - Test boundary conditions: score exactly at threshold, far above, far below
  - Test monotonicity (higher score → higher confidence for cosine)

### 6. Report Sections (Pramod writes)
In `reports/milestone3_report.pdf`:
- **Section 1**: Embedding model choice — why FaceNet over other options (ArcFace/InsightFace, Dlib, MobileFaceNet); discuss tradeoffs of each, justify final choice
- **Section 2**: Confidence formula — what it means, output range, why this formula was chosen
- **Section 3**: Threshold re-selection on embedding scores — ROC plot, new threshold value vs old pixel-based threshold, metrics comparison

---

## Arun's Tasks

### 1. `scripts/verify.py` — CLI Inference Script
```
# Single pair
python scripts/verify.py --img1 path/to/img1.jpg --img2 path/to/img2.jpg

# Batch mode (optional but useful)
python scripts/verify.py --batch path/to/pairs.csv
```
Output format for each pair:
```
Pair:        img1.jpg vs img2.jpg
Score:       0.823
Threshold:   0.712
Decision:    SAME
Confidence:  0.91
Latency:     47.3 ms
```
- Reads threshold from `configs/inference_config.json`
- Calls `verify_pair()` from `src/inference.py`
- Supports `--config` flag to override config path
- Exits with code 0 on success; prints clear error messages on bad input

### 2. `Dockerfile` — Reproducible Container Build
```dockerfile
# Target: grader can do the following from a clean clone:
#   docker build -t face-verifier .
#   docker run --rm -v $(pwd)/data:/app/data face-verifier \
#       python scripts/verify.py --img1 data/img1.jpg --img2 data/img2.jpg
```
- Base image: `python:3.10-slim` (CPU-only, no GPU)
- Pin all dependencies; install from `requirements.txt`
- Model weights should be downloaded at build time (not at runtime)
- CLI must be runnable inside the container without manual fixes

### 3. `scripts/load_test.py` — Concurrency / Load Test Script
```
python scripts/load_test.py --num-pairs 100 --workers 4 --seed 42 --output outputs/load_test_results.json
```
- Use a deterministic set of pairs from existing `outputs/pairs_test.npz` (or a small synthetic set)
- Run with configurable number of workers (Python `concurrent.futures.ProcessPoolExecutor`)
- Record per-request latency
- Report and save to JSON:
  ```json
  {
    "total_requests": 100,
    "workers": 4,
    "wall_clock_s": 12.4,
    "throughput_rps": 8.1,
    "latency_p50_ms": 45.2,
    "latency_p95_ms": 89.7,
    "latency_mean_ms": 48.3,
    "failures": 0
  }
  ```
- Workload must be feasible locally on CPU (100 pairs, 4 workers is a reasonable default)

### 4. `configs/inference_config.json` — Inference & Load Test Config
Create the file with the schema from the interface contract above. Leave `threshold` and `confidence_formula` as `null` initially — Pramod fills them after re-selection.

### 5. `tests/test_inference_smoke.py` — Smoke & Integration Test
```python
# Smoke: does the main CLI path complete without error?
# Integration: does verify_pair return the right keys?
```
- Use a tiny synthetic fixture (two random 160×160 numpy arrays) — no real data download
- Call `verify_pair(img1, img2, threshold=0.5)` and assert all required keys exist with correct types
- Smoke test the CLI: `subprocess.run(["python", "scripts/verify.py", "--img1", ..., "--img2", ...])` and assert exit code 0
- Must complete in < 10 seconds

### 6. README.md Update (Milestone 3 section)
Add a **Milestone 3** section. Must include:
- What M3 adds (embedding upgrade, CLI, Docker, load test)
- Embedding model choice (FaceNet) and a 1–2 sentence rationale pointer to the report
- Confidence formula explanation and output range
- **How to run** (all copy-pastable):
  ```bash
  # Build Docker image
  docker build -t face-verifier .

  # Run CLI inference (Docker)
  docker run --rm -v $(pwd)/data:/app/data face-verifier \
      python scripts/verify.py --img1 data/img1.jpg --img2 data/img2.jpg

  # Run CLI inference (local)
  python scripts/verify.py --img1 data/img1.jpg --img2 data/img2.jpg

  # Run load test
  python scripts/load_test.py --num-pairs 100 --workers 4 --seed 42

  # Run tests
  python -m pytest tests/ -v
  ```
- Artifact locations: `outputs/load_test_results.json`, `outputs/runs_log.json`, `configs/inference_config.json`

### 7. Report Sections (Arun writes)
In `reports/milestone3_report.pdf`:
- **Section 4**: Load test results — methodology, throughput, p50/p95 latency table, failure count, discussion of bottlenecks
- **Section 5**: Docker packaging notes — what was hard, any tradeoffs made (e.g. model weight download at build vs runtime)

---

## Shared

### `requirements.txt` additions
Both add dependencies as needed. Expected new additions:
- `facenet-pytorch` (embeddings — Pramod)
- `torch`, `torchvision` (FaceNet dependency — Pramod)
- `Pillow` (image loading — shared)

### Git Tag `v0.3`
- Run a clean-clone test following README exactly before tagging
- Tag only after Docker build, sample CLI inference, load test, and all tests pass
- `git tag v0.3 && git push origin v0.3`

---

## Definition of Done (Milestone 3)

**Pramod**
- [ ] `src/embedder.py` with deterministic preprocessing + FaceNet embedding extraction
- [ ] `src/confidence.py` with documented formula, output range, and interpretation
- [ ] `src/inference.py` implementing the `verify_pair()` interface contract with stage-level timing
- [ ] Threshold re-selected on embedding scores (runs `run_06`, `run_07` in `runs_log.json`)
- [ ] `configs/inference_config.json` fully populated (model, dim, threshold, confidence formula)
- [ ] Unit tests in `tests/test_embedder.py` and `tests/test_confidence.py`
- [ ] Report sections 1–3 complete

**Arun**
- [ ] `scripts/verify.py` CLI with required output fields (score, threshold, decision, confidence, latency)
- [ ] `Dockerfile` buildable from clean clone, CLI runnable inside container
- [ ] `scripts/load_test.py` with configurable workers, per-request latency, JSON summary output
- [ ] `configs/inference_config.json` scaffolded (Pramod fills threshold/formula)
- [ ] `tests/test_inference_smoke.py` with smoke + integration tests (< 10 seconds, no downloads)
- [ ] README updated with M3 section + full How-to-run
- [ ] Report sections 4–5 complete

**Shared outputs (grader checks)**
- [ ] `docker build -t face-verifier . && docker run ...` works from clean clone
- [ ] CLI prints: score, threshold, decision, confidence, latency for a pair
- [ ] `outputs/load_test_results.json` with throughput + p95 latency
- [ ] `outputs/runs_log.json` has ≥ 7 entries (5 from M2 + run_06 + run_07)
- [ ] `reports/milestone3_report.pdf` complete
- [ ] Tagged `v0.3` pointing to the reproducible commit
