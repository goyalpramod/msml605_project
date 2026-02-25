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
src/
  similarity.py
outputs/                       # generated, gitignored
  manifest.json
  pairs_train.npz
  pairs_test.npz
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
