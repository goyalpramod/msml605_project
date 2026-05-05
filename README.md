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
- Always writes:
  - `outputs/pairs_train.npz` (full train set, or reduced if `--val-fraction` is used)
  - `outputs/pairs_test.npz` (full test set, always unconditional)
  - `outputs/pairs_meta.json`
- With `--val-fraction`: also writes `outputs/pairs_val.npz`
- With `--cap-per-identity`: also writes `outputs/pairs_test_capped.npz` (and `pairs_val_capped.npz` if val split exists)
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

# 2. Generate pair files (always: pairs_train, pairs_test, pairs_meta;
#    with --val-fraction: pairs_val; with --cap-per-identity: pairs_val_capped, pairs_test_capped)
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

# 2. Generate pair files (always: pairs_train, pairs_test, pairs_meta;
#    with --val-fraction: pairs_val; with --cap-per-identity: pairs_val_capped, pairs_test_capped)
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
| `pairs_train.npz` | Train verification pairs (always generated) |
| `pairs_test.npz` | Test verification pairs (always generated) |
| `pairs_val.npz` | Validation split pairs (with `--val-fraction`) |
| `pairs_val_capped.npz` | Identity-capped validation pairs (with `--cap-per-identity`) |
| `pairs_test_capped.npz` | Identity-capped test pairs (with `--cap-per-identity`) |
| `pairs_meta.json` | Pair generation metadata (seed, val_fraction, cap) |
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

## Milestone 3

Milestone 3 upgrades the verifier from pixel-level similarity to **FaceNet (InceptionResnetV1, pretrained on VGGFace2)** embeddings, exposes it through a CLI, packages it in Docker, and measures runtime behavior under concurrency.

### What M3 adds
- **Embedding upgrade**: `src/embedder.py` extracts 512-dim L2-normalized FaceNet embeddings; similarity is cosine between these embeddings instead of raw pixels.
- **Calibrated confidence**: `src/confidence.py` produces a sigmoid confidence in `(0, 1)` from the score/threshold gap.
- **Single-pair inference entry point**: `src/inference.py` wires embed → score → confidence and measures stage-level latency.
- **CLI**: `scripts/verify.py` for single pairs or CSV batches.
- **Docker**: reproducible CPU-only build with FaceNet weights baked in at build time.
- **Load test**: `scripts/load_test.py` runs concurrent inference and reports throughput + p50/p95 latency.

### Embedding model choice
`InceptionResnetV1` pretrained on VGGFace2 via `facenet-pytorch`. Chosen for strong open-source support, L2-normalized 512-dim embeddings, and proven LFW benchmark performance.

### Confidence formula
```
confidence = sigmoid(10 * (score - threshold)) ∈ (0, 1)
```
- `0.5` at the threshold (maximum uncertainty),
- saturates monotonically toward `1.0` above and `0.0` below,
- deterministic, no learned parameters.

### Threshold re-selection

Re-ran the M2 threshold-selection discipline on FaceNet embedding scores — same balanced-accuracy rule, same val/test roles.

| Run ID | Mode | Split | Threshold | Key Metric | Note |
|--------|------|-------|-----------|------------|------|
| run_06 | sweep | val (embedding) | 0.3970 | AUC = 0.9969 | FaceNet embedding threshold sweep on val split |
| run_07 | final | test (embedding) | 0.3970 | Balanced acc = 0.98, F1 = 0.98, EER = 0.024 | FaceNet embedding final evaluation on test |

The new embedding-based threshold `0.397` far outperforms the M2 pixel-based threshold (`0.95`, balanced acc 0.61).

### How to reproduce Milestone 3 results

Assumes the Milestone 2 pipeline has been run first (so `outputs/pairs_val.npz` and `outputs/pairs_test.npz` exist). Steps 1–2 re-run FaceNet threshold selection and regenerate `outputs/roc_run_06.png` + `outputs/cm_run_07.png`. Steps 3–5 exercise the CLI, Docker, and load-test artifacts.

**Runtime note:** Embedding all val pairs (~457) takes ~2–4 minutes on CPU; test pairs (~1000) take ~5–10 minutes. The FaceNet model is downloaded once on first use (~90 MB) and cached.

#### Option A: Using `uv` (recommended)

```bash
# 0. Setup (skip if already done for M2)
uv venv .venv
uv pip install -r requirements.txt

# 1. Embedding threshold sweep on val split (run_06 — produces outputs/roc_run_06.png)
uv run python scripts/embed_eval.py \
  --pairs outputs/pairs_val.npz --mode sweep \
  --run-id run_06 --note "FaceNet embedding threshold sweep on val split"

# 2. Extract the selected threshold from run_06
THRESH_EMB=$(uv run python -c "import json; runs=json.load(open('outputs/runs_log.json')); print([r for r in runs if r['run_id']=='run_06'][0]['threshold'])")

# 3. Embedding final evaluation on test split (run_07 — produces outputs/cm_run_07.png)
uv run python scripts/embed_eval.py \
  --pairs outputs/pairs_test.npz --mode final --threshold $THRESH_EMB \
  --run-id run_07 --note "FaceNet embedding final evaluation on test split"

# 4. CLI inference on a single pair (local)
uv run python scripts/verify.py \
  --img1 data/lfw_home/lfw_funneled/Aaron_Peirsol/Aaron_Peirsol_0001.jpg \
  --img2 data/lfw_home/lfw_funneled/Aaron_Peirsol/Aaron_Peirsol_0002.jpg

# 5. Concurrent load test (100 pairs, 4 workers, seed=42)
uv run python scripts/load_test.py \
  --num-pairs 100 --workers 4 --seed 42 \
  --pairs outputs/pairs_test.npz \
  --output outputs/load_test_results.json

# 6. Smoke + integration tests
uv run python -m pytest tests/test_inference_smoke.py -v

# 7. Full test suite
uv run python -m pytest tests/ -v
```

#### Option B: Classic `venv` + `pip`

```bash
# 0. Setup (skip if already done for M2)
python -m venv .venv
source .venv/bin/activate   # or .\.venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt

# 1. Embedding threshold sweep on val split (run_06)
python scripts/embed_eval.py \
  --pairs outputs/pairs_val.npz --mode sweep \
  --run-id run_06 --note "FaceNet embedding threshold sweep on val split"

# 2. Extract the selected threshold from run_06
THRESH_EMB=$(python -c "import json; runs=json.load(open('outputs/runs_log.json')); print([r for r in runs if r['run_id']=='run_06'][0]['threshold'])")

# 3. Embedding final evaluation on test split (run_07)
python scripts/embed_eval.py \
  --pairs outputs/pairs_test.npz --mode final --threshold $THRESH_EMB \
  --run-id run_07 --note "FaceNet embedding final evaluation on test split"

# 4. CLI inference on a single pair (local)
python scripts/verify.py \
  --img1 data/lfw_home/lfw_funneled/Aaron_Peirsol/Aaron_Peirsol_0001.jpg \
  --img2 data/lfw_home/lfw_funneled/Aaron_Peirsol/Aaron_Peirsol_0002.jpg

# 5. Concurrent load test
python scripts/load_test.py \
  --num-pairs 100 --workers 4 --seed 42 \
  --pairs outputs/pairs_test.npz \
  --output outputs/load_test_results.json

# 6. Smoke + integration tests
python -m pytest tests/test_inference_smoke.py -v

# 7. Full test suite
python -m pytest tests/ -v
```

#### Docker (grader-facing path)

```bash
# Build reproducible image (FaceNet weights baked in at build time)
docker build -t face-verifier .

# Run CLI inference inside the container (mount local data directory)
docker run --rm -v $(pwd)/data:/app/data face-verifier \
  python scripts/verify.py \
    --img1 data/lfw_home/lfw_funneled/Aaron_Peirsol/Aaron_Peirsol_0001.jpg \
    --img2 data/lfw_home/lfw_funneled/Aaron_Peirsol/Aaron_Peirsol_0002.jpg
```

Docker Desktop with the WSL2 backend is the target environment. Run Docker commands from a WSL shell so `$(pwd)` expands to a Linux-style path. If running from Git Bash on Windows, prefix with `MSYS_NO_PATHCONV=1` to stop MSYS from mangling the mount path:

```bash
MSYS_NO_PATHCONV=1 docker run --rm -v "$(pwd)/data:/app/data" face-verifier \
  python scripts/verify.py \
    --img1 data/lfw_home/lfw_funneled/Aaron_Peirsol/Aaron_Peirsol_0001.jpg \
    --img2 data/lfw_home/lfw_funneled/Aaron_Peirsol/Aaron_Peirsol_0002.jpg
```

### Batch CSV format for `scripts/verify.py --batch`

CSV must have a header row `img1,img2`. Example:
```
img1,img2
data/lfw_home/lfw_funneled/Aaron_Peirsol/Aaron_Peirsol_0001.jpg,data/lfw_home/lfw_funneled/Aaron_Peirsol/Aaron_Peirsol_0002.jpg
data/lfw_home/lfw_funneled/Aaron_Peirsol/Aaron_Peirsol_0001.jpg,data/lfw_home/lfw_funneled/Adam_Sandler/Adam_Sandler_0001.jpg
```

### M3 output artifacts

| File | Description |
|------|-------------|
| `configs/inference_config.json` | FaceNet threshold, confidence formula, load-test defaults |
| `outputs/runs_log.json` | Now contains 7 runs (adds `run_06` embedding sweep, `run_07` embedding final) |
| `outputs/roc_run_06.png` | ROC curve for FaceNet embedding sweep on val split |
| `outputs/cm_run_07.png` | Confusion matrix for FaceNet embedding final on test split |
| `outputs/score_dist_run_07.png` | Score distribution at the selected FaceNet threshold (test split) |
| `outputs/scores_emb_val.npz` | Cached embedding cosine scores for val pairs |
| `outputs/scores_emb_test.npz` | Cached embedding cosine scores for test pairs |
| `outputs/load_test_results.json` | Throughput + p50/p95 latency from concurrent workload |

## Milestone 4 / Final Release

Milestone 4 freezes the Milestone 3 embedding-based system as the final release. No new modeling work — the focus is on responsible-ML documentation, CPU profiling, reproducibility, and a clean-clone release path.

### Final pipeline summary
Input image pair → FaceNet preprocessing (resize 160×160, normalize to `[-1, 1]`) → InceptionResnetV1 forward pass (VGGFace2 weights) → 512-dim L2-normalized embeddings → cosine similarity → threshold `0.3969849246231156` → decision (SAME/DIFFERENT) plus calibrated sigmoid confidence in `(0, 1)`.

### Final test metrics

These are the canonical released numbers, frozen in [`outputs/final_system_summary.json`](outputs/final_system_summary.json) and traceable to `run_07` in [`outputs/runs_log.json`](outputs/runs_log.json).

|       Metric      |  Value |
| :---------------- | :----: |
| Balanced accuracy | 0.980  |
| F1                | 0.980  |
| Equal error rate  | 0.024  |
| TP                | 489    |
| FP                | 9      |
| TN                | 491    |
| FN                | 11     |

### Release caveats

- **Confidence is a margin, not a probability.** The reported confidence is `sigmoid(10 * (score - threshold))`, a smooth function of distance from the operating threshold. A confidence of `0.90` does not mean a 90% probability that the two faces match.
- **Benchmark scope only.** This system is intended for course demonstration and reproducible LFW-based evaluation. It is not validated for surveillance, law-enforcement, or any high-stakes identity verification deployment.
- **No subgroup fairness claims.** The project does not include reliable demographic metadata for the evaluation data, so this release does not support quantitative subgroup fairness claims. Fairness-related risks exist and are discussed qualitatively in the System Card.

### Final artifacts

|                Artifact                | Path |
| :------------------------------------- | :--- |
| Final config (single source of truth)  | `configs/inference_config.json` |
| Final system summary                   | `outputs/final_system_summary.json` |
| Runs log (run_06 sweep, run_07 final)  | `outputs/runs_log.json` |
| CPU profiling summary (JSON)           | `outputs/profiling/cpu_profile_summary.json` |
| CPU profiling summary (sidecar table)  | `outputs/profiling/cpu_profile_summary.md` |
| Profiling report (PDF)                 | `reports/milestone4_profiling_report.pdf` |
| Profiling report (markdown source)     | `reports/milestone4_profiling_report.md` |
| System Card (PDF)                      | `reports/milestone4_system_card.pdf` |
| System Card (markdown source)          | `reports/milestone4_system_card.md` |
| Reproducibility checklist              | `reports/milestone4_reproducibility_checklist.md` |
| Release-alignment validator            | `scripts/validate_release_alignment.py` |
| Final tag                              | `v1.0-final` |

### CPU baseline (from `outputs/profiling/cpu_profile_summary.json`)
On a 32-logical-core AMD64 / Windows 11 / `torch 2.11.0+cpu` machine, the final verifier processes a single pair in **81.3 ms end-to-end** (preprocess 1.16 ms / embed 79.96 ms / score 0.21 ms) at **~12.3 pairs/s**. Stacked batching scales throughput to **~81 pairs/s at batch size 16**. Embedding accounts for >93% of latency at every tested batch size. See [`reports/milestone4_profiling_report.md`](reports/milestone4_profiling_report.md) for the full per-stage and batch-size table.

### How to reproduce Milestone 4 results

#### Option A — uv (recommended)

```bash
# 0. Setup (skip if already done for M2/M3)
uv venv .venv
uv pip install -r requirements.txt

# 1. Local CLI inference on a sample pair
uv run python scripts/verify.py \
    --img1 data/lfw_home/lfw_funneled/Aaron_Peirsol/Aaron_Peirsol_0001.jpg \
    --img2 data/lfw_home/lfw_funneled/Aaron_Peirsol/Aaron_Peirsol_0002.jpg

# 2. Dockerized CLI inference
docker build -t face-verifier .
docker run --rm -v "$(pwd)/data:/app/data" face-verifier \
    python scripts/verify.py \
        --img1 data/lfw_home/lfw_funneled/Aaron_Peirsol/Aaron_Peirsol_0001.jpg \
        --img2 data/lfw_home/lfw_funneled/Aaron_Peirsol/Aaron_Peirsol_0002.jpg

# 3. CPU profiling (per-stage + batch-size sensitivity)
uv run python scripts/profile_inference.py \
    --device cpu \
    --output outputs/profiling/cpu_profile_summary.json

# 4. Tests
uv run python -m pytest tests/ -v

# 5. Release-alignment check
uv run python scripts/validate_release_alignment.py
```

#### Option B — venv + pip

```bash
# 0. Setup (skip if already done for M2/M3)
python -m venv .venv
source .venv/bin/activate   # or .\.venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt

# 1. Local CLI inference on a sample pair
python scripts/verify.py \
    --img1 data/lfw_home/lfw_funneled/Aaron_Peirsol/Aaron_Peirsol_0001.jpg \
    --img2 data/lfw_home/lfw_funneled/Aaron_Peirsol/Aaron_Peirsol_0002.jpg

# 2. Dockerized CLI inference (same as above)
docker build -t face-verifier .
docker run --rm -v "$(pwd)/data:/app/data" face-verifier \
    python scripts/verify.py \
        --img1 data/lfw_home/lfw_funneled/Aaron_Peirsol/Aaron_Peirsol_0001.jpg \
        --img2 data/lfw_home/lfw_funneled/Aaron_Peirsol/Aaron_Peirsol_0002.jpg

# 3. CPU profiling
python scripts/profile_inference.py \
    --device cpu \
    --output outputs/profiling/cpu_profile_summary.json

# 4. Tests
python -m pytest tests/ -v

# 5. Release-alignment check
python scripts/validate_release_alignment.py
```

### Reproducibility checklist
The grader-facing step-by-step guide is at [`reports/milestone4_reproducibility_checklist.md`](reports/milestone4_reproducibility_checklist.md).

### Final tag
```bash
git fetch --tags
git checkout v1.0-final
```

## Releases
- `v0.1` — Milestone 1: reproducible LFW pipeline (ingestion, pair generation, similarity benchmarks)
- `v0.2` — Milestone 2: evaluation loop, tracked runs, data-centric improvement, error analysis
- `v0.3` — Milestone 3: FaceNet embeddings, CLI, Docker, concurrent load test
- `v1.0-final` — Milestone 4: final audit, CPU profiling with batch-size sensitivity, System Card, reproducibility checklist

## Reproducibility Notes
- Default workflow uses fixed seed `42`.
- Required run order:
  1. ingestion
  2. pair generation
  3. benchmark (M1) / evaluation (M2)
