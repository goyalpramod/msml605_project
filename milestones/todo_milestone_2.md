# Milestone 2 — Face Verification Evaluation Loop (LFW)
**Due: Thu Feb 26, 2026 11:59pm**

---

## Context
Build a disciplined evaluation loop around the Milestone 1 backbone. The system must evaluate the verifier repeatedly across tracked runs, choose a threshold using a stated rule on a non-test split (we will compare all three rules: max balanced accuracy, max F1, min EER), make at least one data-centric improvement, perform error analysis, and add validation checks + tests. Final deliverable is a tagged Git release (`v0.2`).

---

## Shared Upfront Decisions (agree before coding)

- **Score direction**: cosine similarity → higher = more likely same person; Euclidean distance → lower = more likely same person (document this clearly in configs + report)
- **Split roles**: validation split → threshold selection; test split → final reporting only
- **Threshold rule**: we will evaluate all three on the validation split and select the one that best balances FAR/FRR for this dataset, with written rationale:
  - Maximize **balanced accuracy**
  - Maximize **F1 score**
  - Minimize **Equal Error Rate (EER)**
- **Run tracking format**: lightweight JSON log at `outputs/runs_log.json` (one entry per run)
- **Output interface** (Pramod defines, Arun consumes for error_analysis):
  - `outputs/scores_val.npz` and `outputs/scores_test.npz` → keys: `scores`, `labels`, `pair_indices`
  - `outputs/runs_log.json` → keys per run: `run_id`, `timestamp`, `commit`, `config`, `metrics`, `threshold`, `note`

---

## Pramod's Tasks

### 1. `src/metrics.py` — Metric Computation Module
Implement reusable metric functions (no I/O, pure numpy):
```python
def compute_roc_points(scores, labels, thresholds) -> dict  # tpr, fpr, fnr at each threshold
def compute_confusion_matrix(scores, labels, threshold) -> dict  # TP, FP, TN, FN
def balanced_accuracy(scores, labels, threshold) -> float
def f1_score_at_threshold(scores, labels, threshold) -> float
def equal_error_rate(scores, labels) -> tuple[float, float]  # eer_value, eer_threshold
def select_threshold(scores, labels, rule: str) -> float  # rule in {"balanced_acc", "f1", "eer"}
```
- All functions accept numpy arrays; return plain dicts or scalars (JSON-serializable)
- Include a `SCORE_DIRECTION` constant per metric type (higher=same or lower=same)

### 2. `src/tracker.py` — Run Logging Module
```python
def log_run(run_id, config, metrics, threshold, note, log_path="outputs/runs_log.json") -> None
def load_runs(log_path) -> list[dict]
def print_run_summary(log_path) -> None  # tabular print for README
```
- Auto-captures: `timestamp`, git commit hash (via `subprocess` or `git rev-parse HEAD`)
- Appends to existing log (does not overwrite previous runs)
- Each run entry must have: `run_id`, `timestamp`, `commit`, `config`, `threshold`, `metrics`, `note`

### 3. `src/validation.py` — Pipeline Validation Checks
```python
def validate_pair_file(path) -> None   # schema check, binary labels, required keys
def validate_scores(scores, pairs) -> None  # length match, no NaN/inf
def validate_threshold(threshold, score_range) -> None  # numeric, in valid range
def validate_no_leakage(val_path, test_path) -> None  # no shared pair indices
```
- All functions raise `ValueError` with a clear message on failure (fail-fast)
- Called at the top of `evaluate.py` before any computation

### 4. `scripts/evaluate.py` — Main Evaluation Script
Entry point for all evaluation. Produces all 5+ tracked runs:
```
python scripts/evaluate.py --pairs outputs/pairs_val.npz --mode sweep --similarity cosine --run-id run_01
python scripts/evaluate.py --pairs outputs/pairs_val.npz --mode select --rule balanced_acc --run-id run_02
python scripts/evaluate.py --pairs outputs/pairs_test.npz --mode final --threshold 0.XX --run-id run_03
```
- `--mode sweep`: threshold sweep on given split → logs metrics at every threshold, saves ROC plot to `outputs/roc_<run_id>.png`
- `--mode select`: apply threshold-selection rule → logs selected threshold + confusion matrix to `outputs/cm_<run_id>.png`
- `--mode final`: evaluate at locked threshold on held-out split → logs final metrics
- `--similarity`: `cosine` or `euclidean` (handles score direction internally)
- Calls `validate_pair_file`, `validate_scores`, `validate_threshold` before running
- Calls `log_run(...)` at end of every execution

**The 5 required tracked runs:**
| Run ID  | Mode   | Data version   | Split | Purpose |
|---------|--------|----------------|-------|---------|
| run_01  | sweep  | baseline pairs | val   | Baseline threshold sweep → ROC plot |
| run_02  | select | baseline pairs | val   | Lock threshold (all 3 rules, pick 1) → confusion matrix |
| run_03  | final  | baseline pairs | test  | Baseline final report on held-out test |
| run_04  | sweep  | improved pairs | val   | Post-data-change threshold sweep |
| run_05  | final  | improved pairs | test  | Post-data-change final report |

### 5. `configs/eval_config.json` — Evaluation Configuration
```json
{
  "seed": 42,
  "threshold_range": [0.0, 1.0],
  "threshold_steps": 200,
  "threshold_rule": "balanced_acc",
  "score_direction": {"cosine": "higher_is_same", "euclidean": "lower_is_same"},
  "split_roles": {"val": "threshold_selection", "test": "final_reporting"},
  "pair_policy": "baseline_v1"
}
```

### 6. Unit Tests (`tests/test_metrics.py`, `tests/test_validation.py`, `tests/test_tracker.py`)
- `test_metrics.py`: test each metric function with known toy inputs (e.g., all-correct predictions, all-wrong, balanced)
- `test_validation.py`: test that validation functions raise on bad inputs (wrong schema, mismatched lengths, out-of-range threshold, overlapping splits)
- `test_tracker.py`: test that `log_run` appends correctly and `load_runs` returns the right structure

### 7. Report Sections (Pramod writes)
In `reports/milestone2_report.pdf` (coordinate with Arun on format/layout):
- **Section 1**: Brief system overview, baseline config, split roles
- **Section 2**: Threshold-selection procedure — ROC plot + all three rules evaluated on val split, rationale for final chosen rule
- **Section 3**: Selected threshold + confusion matrix at that threshold (baseline and post-change)

### 8. Git Tag `v0.2` + Clean-Clone Test
- Run a clean-clone test following README exactly before tagging
- Tag only after all outputs are reproducible: `git tag v0.2 && git push origin v0.2`
- Confirm: `outputs/runs_log.json` has ≥ 5 entries; report is at `reports/milestone2_report.pdf`

---

## Arun's Tasks

### 1. Data-Centric Improvement — `scripts/generate_pairs.py` (REQUIRED)
**Identity capping + rebalance** (required — must be implemented):
- Add `--cap-per-identity N` flag (default: `None` = no cap, suggested cap: 10 pairs per identity)
- When capping: deterministically select pairs per identity using seeded sampling
- Ensure positive:negative ratio is exactly 1:1 after capping
- Save capped version as `outputs/pairs_val_capped.npz` and `outputs/pairs_test_capped.npz`
- Log the change in the pair file as metadata (e.g., `cap_per_identity` in the `.npz` array header or a companion `outputs/pairs_meta.json`)

**Quality filtering** (optional — implement if time allows, add `--quality-filter` flag):
- Proxy: compute pixel-level variance for each image as a sharpness estimate
- Filter out pairs where either image variance falls below a deterministic threshold (e.g., bottom 5th percentile)
- Filtering must be deterministic and documented

**Proper val split** (optional — add `--val-fraction` flag, e.g., `--val-fraction 0.15`):
- Assign identities (not pairs) to train/val/test splits to prevent identity leakage across splits
- Save `outputs/pairs_val.npz` (currently M1 only has train/test)
- This would be the val split Pramod uses for threshold selection in `evaluate.py`

> **Note**: Start with the required identity-capping change. It is the clearest, most grader-legible data-centric improvement. Optional items are bonuses.

### 2. `src/plotting.py` — Visualization Module
Standalone plotting functions (no I/O beyond saving to a path):
```python
def plot_roc(fpr, tpr, thresholds, selected_threshold, save_path) -> None
def plot_confusion_matrix(cm_dict, save_path) -> None  # cm_dict: {TP, FP, TN, FN}
def plot_score_distribution(scores, labels, threshold, save_path) -> None  # optional but useful for report
```
- `evaluate.py` (Pramod) imports and calls these to save plots to `outputs/`
- `error_analysis.py` (Arun) also imports `plot_score_distribution` for slice visualizations
- All functions must be side-effect-free except writing the file at `save_path`

### 3. `scripts/error_analysis.py` — Error Slice Analysis Script
```
python scripts/error_analysis.py --scores outputs/scores_test.npz --pairs outputs/pairs_test.npz --threshold 0.XX --output-dir outputs/error_analysis/
```
Implements **two required error slices** (plus one optional):

**Slice 1 — Boundary false positives** (pairs near the decision threshold, wrong=same-person accept):
- Definition: different-identity pairs with score within ±0.05 of selected threshold
- Output: count of such pairs, 4–6 representative image pair examples saved to `outputs/error_analysis/slice1/`
- Hypothesis: visually similar faces (similar age, lighting, pose) fool pixel-level similarity metrics

**Slice 2 — High-variation false negatives** (same-identity pairs with low similarity scores):
- Definition: same-identity pairs in the bottom 20th percentile of cosine similarity scores
- Output: count, representative examples, per-pair pixel variance as variation proxy
- Hypothesis: extreme pose/lighting changes destroy pixel-level representation

**Slice 3 — Rare identity pairs** (optional, pairs where either identity has ≤ 2 images in the dataset):
- Definition: filter pairs by identity image count from `outputs/manifest.json`
- Hypothesis: rare identities produce harder pairs because the dataset has less diversity

Each slice output:
- A count + percentage of total pairs
- Representative examples (image pairs saved as side-by-side PNGs)
- A summary JSON: `outputs/error_analysis/slice_summary.json`

### 4. Unit Test (`tests/test_generate_pairs.py`) — Test Your Own Code
- Test that the baseline pair generation is still deterministic (same seed → same pairs)
- Test that `--cap-per-identity N` produces at most N pairs per identity
- Test that the positive:negative ratio is exactly 1:1 after capping
- Test that capped output has the same `.npz` schema as baseline (`img1`, `img2`, `label`)
- If quality filtering is implemented: test that low-variance images are removed deterministically

### 5. Integration Test (`tests/test_integration.py`) — End-to-End Pipeline Check
- Create a tiny **synthetic fixture** (30 pairs, random scores + binary labels) in `tests/fixtures/`
- Run the full path: validate pair file → run evaluate.py → check `runs_log.json` has a new entry with all required keys → check that a plot file was saved
- Must complete in < 5 seconds; must NOT download any real data
- Confirm output structure is correct (not just that the script didn't crash)

### 6. README.md Update (Milestone 2 section)
Add a **Milestone 2** section to the existing README. Must include:
- Brief summary of what M2 adds (evaluation loop, tracked runs, data-centric improvement)
- The selected threshold rule and its rationale (1–2 sentences)
- Summary of data-centric change (before vs. after in 2–3 bullet points)
- Updated **How to run** section:
  ```bash
  # Generate improved pairs (identity-capped)
  python scripts/generate_pairs.py --seed 42 --cap-per-identity 10

  # Run baseline evaluation (produces run_01 through run_03)
  python scripts/evaluate.py --pairs outputs/pairs_val.npz --mode sweep --similarity cosine --run-id run_01
  python scripts/evaluate.py --pairs outputs/pairs_val.npz --mode select --rule balanced_acc --run-id run_02
  python scripts/evaluate.py --pairs outputs/pairs_test.npz --mode final --threshold <from run_02> --run-id run_03

  # Run post-change evaluation (produces run_04 and run_05)
  python scripts/evaluate.py --pairs outputs/pairs_val_capped.npz --mode sweep --similarity cosine --run-id run_04
  python scripts/evaluate.py --pairs outputs/pairs_test_capped.npz --mode final --threshold <from run_04> --run-id run_05

  # Error analysis
  python scripts/error_analysis.py --scores outputs/scores_test.npz --pairs outputs/pairs_test.npz --threshold <selected>

  # Run tests
  python -m pytest tests/ -v
  ```
- Location of the report: `reports/milestone2_report.pdf`
- Location of run evidence: `outputs/runs_log.json`

### 7. Report Sections (Arun writes)
In `reports/milestone2_report.pdf` (coordinate with Pramod on format/layout):
- **Section 4**: Error analysis — define both slices clearly, include example image pairs, counts, hypotheses, and suggested future improvement per slice
- **Section 5**: Data-centric before-vs-after comparison — key metrics table (baseline vs. identity-capped), honest discussion of both gains and tradeoffs
- **Closing** (1–2 sentences): most important lesson from the iteration loop

---

## Shared / Interface Contract

Both must agree on this before coding begins — Pramod defines, Arun consumes:

| Artifact | Format | Owner | Consumer |
|---|---|---|---|
| `outputs/pairs_val.npz` | keys: `img1`, `img2`, `label` | Arun | Pramod (evaluate.py) |
| `outputs/pairs_val_capped.npz` | same schema + optional `identity_ids` key | Arun | Pramod (evaluate.py) |
| `outputs/scores_test.npz` | keys: `scores`, `labels`, `pair_indices` | Pramod (evaluate.py output) | Arun (error_analysis.py) |
| `outputs/runs_log.json` | list of run dicts (see tracker.py spec) | Pramod (tracker.py) | Arun (README summary table) |
| `outputs/error_analysis/slice_summary.json` | `{slice_id, definition, count, fraction, hypothesis}` | Arun | report |

### `requirements.txt` additions
Both add dependencies as needed. New expected additions:
- `matplotlib` (plots — Arun, `src/plotting.py`)
- `pytest` (tests — both)

---

## Definition of Done (Milestone 2)

**Pramod**
- [ ] `src/metrics.py` with ROC, confusion matrix, balanced accuracy, F1, EER, threshold selector
- [ ] `src/tracker.py` with JSON run logging (auto git hash, timestamp)
- [ ] `src/validation.py` with fail-fast checks (schema, labels, lengths, leakage)
- [ ] `scripts/evaluate.py` with `--mode sweep/select/final` and run logging
- [ ] `configs/eval_config.json` with threshold range, split roles, score direction
- [ ] Unit tests in `tests/test_metrics.py`, `tests/test_validation.py`, `tests/test_tracker.py`
- [ ] Report sections 1–3 complete
- [ ] Clean-clone test passes; `v0.2` tagged

**Arun**
- [ ] `scripts/generate_pairs.py` updated with `--cap-per-identity` (required data-centric change)
- [ ] `src/plotting.py` with ROC, confusion matrix, and score distribution plot functions
- [ ] `scripts/error_analysis.py` with 2 defined slices + representative examples
- [ ] `tests/test_generate_pairs.py` with determinism, cap, and balance ratio checks
- [ ] `tests/test_integration.py` using synthetic fixtures (< 5 seconds, no downloads)
- [ ] README updated with M2 summary + How-to-run for all new scripts + artifact locations
- [ ] Report sections 4–5 + closing complete

**Shared outputs (grader checks)**
- [ ] `outputs/runs_log.json` with ≥ 5 tracked runs
- [ ] ROC-style plot saved (e.g., `outputs/roc_run_01.png`)
- [ ] Confusion matrix plot committed or in report
- [ ] `reports/milestone2_report.pdf` (~2 pages, all sections complete)
- [ ] Tagged `v0.2` pointing to the reproducible commit
