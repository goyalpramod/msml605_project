# msml605_project

## Project Overview
Milestone 1 implements a reproducible LFW face verification pipeline:
- deterministic dataset ingestion
- deterministic pair generation
- similarity scoring APIs (NumPy vectorized and vanilla Python loop baselines)

The dataset used is Labeled Faces in the Wild (LFW), using the official dev train/test pair splits.

## Repository Layout
```text
scripts/
  ingest_lfw.py
  generate_pairs.py
  benchmark.py
  error_analysis.py            # M2: error slice analysis
src/
  similarity.py
  plotting.py                  # M2: ROC, confusion matrix, score distribution
tests/
  conftest.py                  # shared pytest fixtures
  test_generate_pairs.py       # M2: pair generation tests
outputs/                       # generated, gitignored
  manifest.json
  pairs_train.npz
  pairs_test.npz
  pairs_val.npz                # M2: validation split
  pairs_val_capped.npz         # M2: identity-capped val pairs
  pairs_test_capped.npz        # M2: identity-capped test pairs
  pairs_meta.json              # M2: pair generation metadata
data/                          # generated, gitignored
```

`data/` and `outputs/` are generated artifacts and should remain gitignored.

## Implemented Components

### `scripts/ingest_lfw.py`
- Fetches LFW dev split metadata/cache (`train` and `test` subsets).
- Writes deterministic dataset manifest to `outputs/manifest.json`.
- Current manifest policy: `split_policy = "dev_train_test"`.

### `scripts/generate_pairs.py`
- Depends on ingestion output manifest (`--manifest ./outputs/manifest.json`).
- Does not auto-download (`download_if_missing=False`); ingestion must run first.
- Writes:
  - `outputs/pairs_train.npz`
  - `outputs/pairs_test.npz`
- Each `.npz` contains keys: `img1`, `img2`, `label`.

### `src/similarity.py`
Exports:
- `cosine_similarity`
- `euclidean_distance`
- `cosine_similarity_loop`
- `euclidean_distance_loop`
- `load_pair_vectors`

`load_pair_vectors("train" | "test")` validates generated pair files, checks schema, and returns flattened vectors `(N, D)` plus labels `(N,)`.

### `scripts/benchmark.py`
- Imports loop and vectorized similarity functions from `src/similarity.py`.
- Benchmarks cosine and Euclidean loop vs NumPy implementations.
- Checks numerical agreement with `np.allclose(..., atol=1e-6)`.
- Prints timing summary lines.

## Setup and Run

### Option A: Use `uv` (recommended)

#### 1) Install `uv` (choose one method)

##### Method 1: Official installer (macOS/Linux)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

##### Method 2: Official installer (Windows PowerShell)
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

##### Method 3: Install via `pipx`
```bash
pipx install uv
```

##### Method 4: Install via `pip` (fallback)
```bash
pip install uv
```

Verify install:
```bash
uv --version
```

#### 2) Create a virtual environment
```bash
uv venv .venv
```

Activation is optional when using `uv pip` and `uv run`:

Windows (PowerShell):
```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:
```bash
source .venv/bin/activate
```

#### 3) Install dependencies and run
```bash
uv pip install -r requirements.txt

uv run python scripts/ingest_lfw.py --seed 42
uv run python scripts/generate_pairs.py --seed 42
uv run python scripts/benchmark.py
```

### Option B: Classic `venv` + `pip`

#### 1) Create and activate a virtual environment
```bash
python -m venv .venv
```

Activate it:

Windows (PowerShell):
```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:
```bash
source .venv/bin/activate
```

#### 2) Install dependencies and run
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt

python scripts/ingest_lfw.py --seed 42
python scripts/generate_pairs.py --seed 42
python scripts/benchmark.py
```

## Data Artifacts and Contracts

### `outputs/manifest.json`
Primary fields used by the pipeline:
- `seed`
- `split_policy`
- `train_count`
- `test_count`
- `total_identities`
- `image_shape`

### Pair file schema
For both `pairs_train.npz` and `pairs_test.npz`:
- `img1`: `(N, 62, 47)`
- `img2`: `(N, 62, 47)`
- `label`: `(N,)`, where `1 = same person`, `0 = different person`

## Milestone 2 (In Progress)

Milestone 2 adds an evaluation loop with tracked runs, data-centric improvements, and error analysis.

### What's been built (Arun)
- **Identity-capped pair generation**: `generate_pairs.py` now supports `--val-fraction` for identity-level train/val splitting (no leakage) and `--cap-per-identity N` to limit over-represented identities with 1:1 ratio rebalancing.
- **Visualization module**: `src/plotting.py` — ROC curve, confusion matrix, and score distribution plots.
- **Error analysis script**: `scripts/error_analysis.py` — two error slices (boundary false positives, high-variation false negatives) with representative examples.
- **Unit tests**: `tests/test_generate_pairs.py` — 13 tests covering pair parsing, identity splitting, capping, and output schema.

### What still needs Pramod's work
- `src/metrics.py`, `src/tracker.py`, `src/validation.py`, `scripts/evaluate.py`
- 5 tracked evaluation runs and threshold selection
- Integration test (`tests/test_integration.py`)
- Report sections and `v0.2` tag

### New outputs
```text
outputs/
  pairs_val.npz             # validation split (identity-level)
  pairs_val_capped.npz      # identity-capped validation pairs (1:1 ratio)
  pairs_test_capped.npz     # identity-capped test pairs (1:1 ratio)
  pairs_meta.json            # metadata (seed, val_fraction, cap_per_identity)
  error_analysis/            # generated by error_analysis.py
    slice_summary.json
    slice1/                  # boundary false positive examples
    slice2/                  # high-variation false negative examples
```

### M2 verification commands
```bash
# Install dependencies
uv pip install -r requirements.txt

# Generate validation split + identity-capped pairs
uv run python scripts/generate_pairs.py --seed 42 --val-fraction 0.15
uv run python scripts/generate_pairs.py --seed 42 --val-fraction 0.15 --cap-per-identity 10

# Run tests
uv run python -m pytest tests/ -v

# Run ruff checks
uv run ruff check . && uv run ruff format --check .
```

## Release
- Milestone tag created: `v0.1`
- Tag pushed to remote: `git push origin v0.1`
- Reproducibility verified from a clean clone using the run commands above.

## Reproducibility Notes
- Default workflow uses fixed seed `42`.
- Required run order:
  1. ingestion
  2. pair generation
  3. benchmark
