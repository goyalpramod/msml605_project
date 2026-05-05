# Milestone 4: Reproducibility Checklist

This checklist reproduces every Milestone 4 artifact from a clean clone. Both `uv` (recommended) and classic `venv + pip` paths are shown; pick one.

**Final tag:** `v1.0-final`
**Final config:** `configs/inference_config.json` (threshold `0.3969849246231156`, FaceNet, embedding dim 512)
**Final system summary:** `outputs/final_system_summary.json` (single source of truth for release metadata, run IDs, and test metrics)
**Companion documents:** [`reports/milestone4_system_card.pdf`](milestone4_system_card.pdf) (responsible-ML audit), [`reports/milestone4_profiling_report.pdf`](milestone4_profiling_report.pdf) (CPU profile)

---

## 0. Prerequisites

- Python 3.13 (or compatible 3.x as pinned by the project)
- Git
- (Optional) Docker Desktop with WSL2 backend, if exercising the Docker path

---

## 1. Clone and enter the repo

```bash
git clone <repo-url> msml605_project
cd msml605_project
git checkout v1.0-final     # grader path: use the final tag
```

> Pre-tag testing (used during the clean-clone gate before the tag is pushed): stay on `main` and skip the `git checkout` line. The contents of `main` immediately prior to tagging are identical to what `v1.0-final` will point at.

---

## 2. Environment setup

### Option A — uv (recommended)

```bash
uv venv .venv
uv pip install -r requirements.txt
```

### Option B — venv + pip

```bash
python -m venv .venv
# Activate:
#   Windows PowerShell: .\.venv\Scripts\Activate.ps1
#   macOS/Linux:         source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Regenerate prerequisite data (only needed if `outputs/pairs_*.npz` are missing)

```bash
# uv
uv run python scripts/ingest_lfw.py --seed 42
uv run python scripts/generate_pairs.py --seed 42 --val-fraction 0.15 --cap-per-identity 10

# venv + pip
python scripts/ingest_lfw.py --seed 42
python scripts/generate_pairs.py --seed 42 --val-fraction 0.15 --cap-per-identity 10
```

`outputs/pairs_test.npz` is the input source for the profiling step.

---

## 4. Run a single-pair CLI inference (sanity check)

```bash
# uv
uv run python scripts/verify.py \
    --img1 data/lfw_home/lfw_funneled/Aaron_Peirsol/Aaron_Peirsol_0001.jpg \
    --img2 data/lfw_home/lfw_funneled/Aaron_Peirsol/Aaron_Peirsol_0002.jpg

# venv + pip
python scripts/verify.py \
    --img1 data/lfw_home/lfw_funneled/Aaron_Peirsol/Aaron_Peirsol_0001.jpg \
    --img2 data/lfw_home/lfw_funneled/Aaron_Peirsol/Aaron_Peirsol_0002.jpg
```

Expected output: a `Pair / Score / Threshold / Decision / Confidence / Latency` block, decision `SAME` for the same-identity pair above.

---

## 5. Build and run the Dockerized CLI

```bash
docker build -t face-verifier .

docker run --rm -v "$(pwd)/data:/app/data" face-verifier \
    python scripts/verify.py \
        --img1 data/lfw_home/lfw_funneled/Aaron_Peirsol/Aaron_Peirsol_0001.jpg \
        --img2 data/lfw_home/lfw_funneled/Aaron_Peirsol/Aaron_Peirsol_0002.jpg
```

On Git Bash for Windows, prefix with `MSYS_NO_PATHCONV=1` to prevent path mangling.

---

## 6. Reproduce the CPU profiling summary

```bash
# uv
uv run python scripts/profile_inference.py \
    --device cpu \
    --output outputs/profiling/cpu_profile_summary.json

# venv + pip
python scripts/profile_inference.py \
    --device cpu \
    --output outputs/profiling/cpu_profile_summary.json
```

Produces:
- `outputs/profiling/cpu_profile_summary.json` — machine-readable per-stage and batch-size results
- `outputs/profiling/cpu_profile_summary.md` — sidecar markdown table for paste-into-report

Default profile settings: batch sizes `1,2,4,8,16`, 64 pairs per batch, 3 warmup runs, 10 timed repeats.

---

## 7. Run tests

```bash
# uv
uv run python -m pytest tests/ -v

# venv + pip
python -m pytest tests/ -v
```

The suite includes `tests/test_final_release_alignment.py`, which exercises the release-alignment validator against a synthetic consistent contract, a synthetic inconsistent contract, and the actual repo state. All three must pass.

---

## 8. Run the release-alignment check

```bash
# uv
uv run python scripts/validate_release_alignment.py

# venv + pip
python scripts/validate_release_alignment.py
```

Confirms that `configs/inference_config.json`, `outputs/final_system_summary.json`, and `outputs/runs_log.json` all describe the same final system. Specifically, the validator checks:

- `embedding_model`, `embedding_dim`, `threshold`, `score_direction`, and `confidence_formula` agree between the config and the summary.
- Every `final_run_ids` entry in the summary exists in the runs log.
- `run_06` is a `val` sweep using the same threshold and selection rule as the summary.
- `run_07` is a `test` final run using the same threshold and same metrics (balanced accuracy, F1, EER, TP/FP/TN/FN) as the summary.
- Every artifact path declared by the summary's `artifact_paths` exists on disk.
- The three required final report files are present: profiling report PDF, reproducibility checklist, and System Card PDF.

---

## 9. Lint check (optional)

```bash
# uv
uv run ruff check . && uv run ruff format --check .

# venv + pip
ruff check . && ruff format --check .
```

---

## Artifact locations

|                Artifact                | Path |
| :------------------------------------- | :--- |
| Final config (single source of truth)  | `configs/inference_config.json` |
| Final system summary                   | `outputs/final_system_summary.json` |
| Runs log                               | `outputs/runs_log.json` |
| CPU profiling summary (JSON)           | `outputs/profiling/cpu_profile_summary.json` |
| CPU profiling summary (sidecar table)  | `outputs/profiling/cpu_profile_summary.md` |
| Profiling report (PDF)                 | `reports/milestone4_profiling_report.pdf` |
| Profiling report (markdown source)     | `reports/milestone4_profiling_report.md` |
| System Card (PDF)                      | `reports/milestone4_system_card.pdf` |
| System Card (markdown source)          | `reports/milestone4_system_card.md` |
| Reproducibility checklist              | `reports/milestone4_reproducibility_checklist.md` |
| Release-alignment validator            | `scripts/validate_release_alignment.py` |
| Release-alignment tests                | `tests/test_final_release_alignment.py` |
| Final tag                              | `v1.0-final` |
