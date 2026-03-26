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
  evaluate.py                  # M2: evaluation pipeline (sweep/select/final)
  error_analysis.py            # M2: error slice analysis
src/
  similarity.py
  metrics.py                   # M2: ROC, confusion matrix, balanced acc, F1, EER
  tracker.py                   # M2: JSON run logging with git hash + timestamp
  validation.py                # M2: fail-fast input validation
  plotting.py                  # M2: ROC, confusion matrix, score distribution plots
configs/
  eval_config.json             # M2: evaluation configuration
tests/
  conftest.py                  # shared pytest fixtures
  test_generate_pairs.py       # M2: pair generation tests
  test_metrics.py              # M2: metric computation tests
  test_tracker.py              # M2: run logging tests
  test_validation.py           # M2: validation tests
  test_integration.py          # M2: end-to-end pipeline tests
reports/
  milestone2_report.pdf        # M2: evaluation report
outputs/                       # generated, gitignored
  manifest.json
  pairs_train.npz
  pairs_test.npz
  pairs_val.npz                # M2: validation split
  pairs_val_capped.npz         # M2: identity-capped val pairs
  pairs_test_capped.npz        # M2: identity-capped test pairs
  pairs_meta.json              # M2: pair generation metadata
  runs_log.json                # M2: tracked evaluation runs
  roc_run_*.png                # M2: ROC curve plots
  cm_run_*.png                 # M2: confusion matrix plots
  score_dist_run_*.png         # M2: score distribution plots
  scores_*.npz                 # M2: computed similarity scores
  error_analysis/              # M2: error slice analysis outputs
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

## Milestone 2

Milestone 2 builds a disciplined evaluation loop around the Milestone 1 backbone. It adds:
- **5 tracked evaluation runs** logged to `outputs/runs_log.json` with metrics, thresholds, and git commit hashes
- **Threshold selection** via balanced accuracy (maximizes (TPR + TNR) / 2) on the validation split
- **Data-centric improvement**: identity-capped pair generation that limits over-represented identities and enforces 1:1 positive/negative ratio
- **Error analysis** with two defined error slices and representative image pair examples

### Threshold selection rationale

We selected **balanced accuracy** as the threshold rule. It weights true positive rate and true negative rate equally, making it robust to class imbalance and appropriate for a verification task where both false accepts and false rejects matter.

### Data-centric improvement: identity capping

- **Before (baseline)**: Validation split has 457 pairs with unbalanced positive/negative ratio (159 pos, 298 neg); some identities contribute disproportionately many pairs
- **After (identity-capped)**: Capped to max 10 pairs per identity, then rebalanced to exactly 1:1 ratio (159 pos, 159 neg = 318 total pairs)
- **Trade-off**: Reduced dataset size in exchange for more balanced identity representation and class ratio

### Tracked runs summary

| Run ID | Mode | Split | Threshold | Key Metric | Note |
|--------|------|-------|-----------|------------|------|
| run_01 | sweep | val (baseline) | 0.9497 | AUC = 0.6109 | Baseline threshold sweep |
| run_02 | select | val (baseline) | 0.9497 | Balanced acc = 0.5899 | Threshold locked via balanced accuracy |
| run_03 | final | test (baseline) | 0.9497 | Balanced acc = 0.6140 | Baseline final on held-out test |
| run_04 | sweep | val (capped) | 0.9598 | AUC = 0.6069 | Identity-capped threshold sweep |
| run_05 | final | test (capped) | 0.9598 | Balanced acc = 0.5880 | Identity-capped final on test |

### How to reproduce Milestone 2 results

#### Option A: Using `uv` (recommended)

```bash
# 0. Setup (skip if already done for M1)
uv venv .venv
uv pip install -r requirements.txt

# 1. Ingest LFW data
uv run python scripts/ingest_lfw.py --seed 42

# 2. Generate all pairs (baseline + val split + identity-capped) in one call
uv run python scripts/generate_pairs.py --seed 42 --val-fraction 0.15 --cap-per-identity 10

# 3. Baseline evaluation — threshold sweep on val split (run_01)
uv run python scripts/evaluate.py \
  --pairs outputs/pairs_val.npz --mode sweep --similarity cosine \
  --run-id run_01 --note "Baseline threshold sweep on val split"

# 4. Baseline evaluation — lock threshold via balanced accuracy (run_02)
uv run python scripts/evaluate.py \
  --pairs outputs/pairs_val.npz --mode select --rule balanced_acc \
  --run-id run_02 --note "Threshold selection via balanced accuracy"

# 5. Extract the selected threshold from run_02
THRESH_BASELINE=$(uv run python -c "import json; runs=json.load(open('outputs/runs_log.json')); print([r for r in runs if r['run_id']=='run_02'][0]['threshold'])")

# 6. Baseline evaluation — final report on held-out test (run_03)
uv run python scripts/evaluate.py \
  --pairs outputs/pairs_test.npz --mode final --threshold $THRESH_BASELINE \
  --run-id run_03 --note "Baseline final on test split"

# 7. Identity-capped evaluation — threshold sweep on capped val (run_04)
uv run python scripts/evaluate.py \
  --pairs outputs/pairs_val_capped.npz --mode sweep --similarity cosine \
  --run-id run_04 --note "Identity-capped threshold sweep on val"

# 8. Extract threshold from run_04
THRESH_CAPPED=$(uv run python -c "import json; runs=json.load(open('outputs/runs_log.json')); print([r for r in runs if r['run_id']=='run_04'][0]['threshold'])")

# 9. Identity-capped evaluation — final report on capped test (run_05)
uv run python scripts/evaluate.py \
  --pairs outputs/pairs_test_capped.npz --mode final --threshold $THRESH_CAPPED \
  --run-id run_05 --note "Identity-capped final on test split"

# 10. Error analysis on baseline test results
uv run python scripts/error_analysis.py \
  --scores outputs/scores_test.npz --pairs outputs/pairs_test.npz \
  --threshold $THRESH_BASELINE --output-dir outputs/error_analysis/

# 11. Run tests
uv run python -m pytest tests/ -v

# 12. Lint check
uv run ruff check . && uv run ruff format --check .
```

#### Option B: Classic `venv` + `pip`

```bash
# 0. Setup (skip if already done for M1)
python -m venv .venv
source .venv/bin/activate   # or .\.venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt

# 1. Ingest LFW data
python scripts/ingest_lfw.py --seed 42

# 2. Generate all pairs (baseline + val split + identity-capped) in one call
python scripts/generate_pairs.py --seed 42 --val-fraction 0.15 --cap-per-identity 10

# 3. Baseline evaluation — threshold sweep on val split (run_01)
python scripts/evaluate.py \
  --pairs outputs/pairs_val.npz --mode sweep --similarity cosine \
  --run-id run_01 --note "Baseline threshold sweep on val split"

# 4. Baseline evaluation — lock threshold via balanced accuracy (run_02)
python scripts/evaluate.py \
  --pairs outputs/pairs_val.npz --mode select --rule balanced_acc \
  --run-id run_02 --note "Threshold selection via balanced accuracy"

# 5. Extract the selected threshold from run_02
THRESH_BASELINE=$(python -c "import json; runs=json.load(open('outputs/runs_log.json')); print([r for r in runs if r['run_id']=='run_02'][0]['threshold'])")

# 6. Baseline evaluation — final report on held-out test (run_03)
python scripts/evaluate.py \
  --pairs outputs/pairs_test.npz --mode final --threshold $THRESH_BASELINE \
  --run-id run_03 --note "Baseline final on test split"

# 7. Identity-capped evaluation — threshold sweep on capped val (run_04)
python scripts/evaluate.py \
  --pairs outputs/pairs_val_capped.npz --mode sweep --similarity cosine \
  --run-id run_04 --note "Identity-capped threshold sweep on val"

# 8. Extract threshold from run_04
THRESH_CAPPED=$(python -c "import json; runs=json.load(open('outputs/runs_log.json')); print([r for r in runs if r['run_id']=='run_04'][0]['threshold'])")

# 9. Identity-capped evaluation — final report on capped test (run_05)
python scripts/evaluate.py \
  --pairs outputs/pairs_test_capped.npz --mode final --threshold $THRESH_CAPPED \
  --run-id run_05 --note "Identity-capped final on test split"

# 10. Error analysis on baseline test results
python scripts/error_analysis.py \
  --scores outputs/scores_test.npz --pairs outputs/pairs_test.npz \
  --threshold $THRESH_BASELINE --output-dir outputs/error_analysis/

# 11. Run tests
python -m pytest tests/ -v
```

### Output artifacts

After running the full pipeline, the following files are generated in `outputs/`:

| File | Description |
|------|-------------|
| `runs_log.json` | 5 tracked evaluation runs with metrics, thresholds, and git hashes |
| `roc_run_01.png` | ROC curve for baseline val sweep |
| `cm_run_02.png` | Confusion matrix at selected baseline threshold |
| `score_dist_run_02.png` | Score distribution for baseline val |
| `cm_run_03.png` | Confusion matrix for baseline test (final) |
| `score_dist_run_03.png` | Score distribution for baseline test |
| `roc_run_04.png` | ROC curve for identity-capped val sweep |
| `cm_run_05.png` | Confusion matrix for identity-capped test (final) |
| `score_dist_run_05.png` | Score distribution for identity-capped test |
| `scores_val.npz` | Computed similarity scores for val split |
| `scores_test.npz` | Computed similarity scores for test split |
| `scores_val_capped.npz` | Computed similarity scores for capped val |
| `scores_test_capped.npz` | Computed similarity scores for capped test |
| `error_analysis/slice_summary.json` | Error slice definitions, counts, and hypotheses |
| `error_analysis/slice1/` | Boundary false positive example image pairs |
| `error_analysis/slice2/` | High-variation false negative example image pairs |

### Report

Full evaluation report: `reports/milestone2_report.pdf`

## Releases
- `v0.1` — Milestone 1: reproducible LFW pipeline (ingestion, pair generation, similarity benchmarks)
- `v0.2` — Milestone 2: evaluation loop, tracked runs, data-centric improvement, error analysis

## Reproducibility Notes
- Default workflow uses fixed seed `42`.
- Required run order:
  1. ingestion
  2. pair generation
  3. benchmark (M1) / evaluation (M2)
