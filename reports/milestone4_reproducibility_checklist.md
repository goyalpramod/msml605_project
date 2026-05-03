# Milestone 4 — Reproducibility Checklist

This checklist reproduces every Milestone 4 artifact from a clean clone. Both `uv` (recommended) and classic `venv + pip` paths are shown — pick one.

**Final tag:** `v1.0-final`
**Final config:** `configs/inference_config.json` (threshold 0.397, FaceNet, embedding dim 512)

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
git checkout v1.0-final     # use the final tag
```

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

---

## 8. Run the release-alignment check

```bash
# uv
uv run python scripts/validate_release_alignment.py

# venv + pip
python scripts/validate_release_alignment.py
```

Confirms that `configs/inference_config.json`, `outputs/final_system_summary.json`, and `outputs/runs_log.json` all describe the same final system.

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

| Artifact | Path |
|---|---|
| Final config (single source of truth) | `configs/inference_config.json` |
| Final system summary | `outputs/final_system_summary.json` |
| Runs log | `outputs/runs_log.json` |
| CPU profiling summary (JSON) | `outputs/profiling/cpu_profile_summary.json` |
| CPU profiling summary (sidecar table) | `outputs/profiling/cpu_profile_summary.md` |
| Profiling report (PDF) | `reports/milestone4_profiling_report.pdf` |
| Profiling report (markdown source) | `reports/milestone4_profiling_report.md` |
| System Card (PDF) | `reports/milestone4_system_card.pdf` |
| Reproducibility checklist | `reports/milestone4_reproducibility_checklist.md` |
| Final tag | `v1.0-final` |
