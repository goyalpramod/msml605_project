# Milestone 4 — Final Audit, Profiling, and Release
**Due: TBD 11:59pm**

---

## Context
Milestone 4 is the final audit, profiling, and release milestone for the face verification project. The goal is not to add another major model change. The goal is to freeze the final embedding-based system from Milestone 3, document it responsibly, profile it on CPU, make the release reproducible from a clean clone, and ensure the README, System Card, profiling report, CLI behavior, and final tag all refer to the same final system version.

This milestone must preserve the Milestone 3 inference path and present it as a coherent final release:
- responsible ML documentation via a professional **System Card**
- **CPU-only hardware-aware profiling** with stage-wise latency breakdown
- **batch-size sensitivity** analysis
- a concise **reproducibility checklist**
- a final README that points clearly to all final artifacts
- a **clean-clone verification path**
- a final Git tag such as `v1.0-final`

---

## Shared Upfront Decisions (already agreed)

- **Ownership model:** preserve prior ownership as much as possible
  - Pramod owns final system semantics, metrics alignment, and system-level audit content
  - Arun owns runtime/release packaging, profiling execution, reproducibility path, and final tag
- **Writing split:** both contribute, but by artifact rather than by co-editing the same file whenever possible
- **Hardware scope:** CPU-only submission
- **Machine context:** Pramod works on macOS, Arun works on Windows
- **Release owner:** Arun handles the final clean-clone release gate and tag push
- **Planning preference:** cleaner ownership is preferred over maximum parallelism

---

## Shared Interface Contract (must not drift)

These artifacts are the contract between both of you for Milestone 4:

| Artifact | Format | Owner | Consumer | Purpose |
|---|---|---|---|---|
| `configs/inference_config.json` | JSON | Pramod | Arun | Source of truth for final threshold, model name, embedding dim, confidence formula |
| `outputs/runs_log.json` | JSON list | Pramod | Arun | Source of truth for run IDs, final threshold-selection rule, final metrics |
| `outputs/final_system_summary.json` | JSON | Pramod | Arun | Final threshold, split/rule, key metrics, artifact references for README/checklist |
| `outputs/profiling/cpu_profile_summary.json` | JSON | Arun | Pramod | CPU hardware, methodology, batch sizes, per-stage latency, end-to-end latency, throughput |
| `reports/milestone4_system_card.pdf` | PDF | Pramod | Arun / grader | Final responsible ML audit for the released system |
| `reports/milestone4_profiling_report.pdf` | PDF | Arun | Pramod / grader | CPU profiling report with batch-size sensitivity |
| `reports/milestone4_reproducibility_checklist.md` | Markdown | Arun | grader | Exact commands to reproduce core artifacts and run Dockerized CLI |
| `README.md` | Markdown | Arun | grader | Final entry point to the release and artifact locations |

### Alignment rules
- `configs/inference_config.json` is the single source of truth for the operating threshold used by the final CLI.
- The threshold, metric values, and reported system version in the README, System Card, profiling report, and checklist must match one another.
- No one should change the operating threshold after the System Card or profiling report is finalized unless both artifacts are updated together.
- The final tag must be created only after the clean-clone path passes and all documents are aligned.

---

## Pramod's Tasks

### 1. Freeze the final system version
- Confirm which exact embedding-based system version from Milestone 3 is the final release candidate.
- Verify that `scripts/verify.py`, `src/inference.py`, `Dockerfile`, and `configs/inference_config.json` all describe the same final system.
- Confirm that the final operating threshold in `configs/inference_config.json` is the embedding-based threshold, not the old pixel baseline threshold.
- Record the finalized release assumptions in `outputs/final_system_summary.json`.

Required fields in `outputs/final_system_summary.json`:
```json
{
  "release_candidate": "milestone4_final",
  "embedding_model": "facenet",
  "embedding_dim": 512,
  "threshold": 0.3969849246231156,
  "threshold_selection_split": "val",
  "threshold_selection_rule": "balanced_acc",
  "score_direction": "higher_is_same",
  "confidence_formula": "...",
  "final_run_ids": ["run_06", "run_07"],
  "test_metrics": {
    "balanced_accuracy": 0.0,
    "f1": 0.0,
    "eer": 0.0,
    "tp": 0,
    "fp": 0,
    "tn": 0,
    "fn": 0
  }
}
```

### 2. Confirm the final operating threshold and key metrics
- Re-read `outputs/runs_log.json` and extract the final threshold-selection evidence for the embedding-based system.
- Verify and document:
  - which split chose the threshold
  - which rule chose the threshold
  - which run IDs are the final embedding sweep and final embedding test runs
  - the final key metrics at the operating threshold
- If needed, regenerate the final metric summary so the System Card and README cite the same numbers.
- Ensure the metrics used in Milestone 4 are taken from the final embedding-based system only.

### 3. Add final alignment evidence
- Add a lightweight release-alignment helper so Arun can validate the release without manually re-reading every file.
- Recommended implementation options:
  - `scripts/validate_release_alignment.py`
  - or `tests/test_final_release_alignment.py`
- The alignment check should verify at minimum:
  - threshold in `configs/inference_config.json` matches `outputs/final_system_summary.json`
  - run IDs referenced in `outputs/final_system_summary.json` exist in `outputs/runs_log.json`
  - final artifact paths referenced in the summary exist
  - the expected final report files exist before tagging

### 4. Write the final System Card
Create `reports/milestone4_system_card.pdf`.

Required sections:
- **System overview**
  - what the verifier does
  - accepted input format
  - high-level embedding-based pipeline
- **Intended use**
  - what this project is meant to support
  - clearly stated out-of-scope uses
- **Data summary**
  - LFW-based development/evaluation context
  - major data limitations relevant to interpretation
- **Operating threshold and metrics**
  - selected threshold
  - selection rule and split
  - final test metrics at that threshold
- **Failure modes and limitations**
  - visually similar different-identity pairs
  - pose/lighting/occlusion variation
  - confidence near the decision boundary
- **Fairness-related risks and misuse concerns**
  - responsible risk discussion without unsupported subgroup claims
  - clearly state lack of reliable demographic metadata if that remains true
- **Operational constraints**
  - expected input assumptions
  - CPU-only context
  - pointer to profiling report for latency details
- **Reproducibility pointer**
  - final tag
  - README path
  - profiling report path
  - checklist path

### 5. System Card quality rules
- Keep the document focused and professional, roughly within the guide’s 1 to 6 page expectation.
- Do not write unsupported fairness claims.
- Do not let the System Card refer to any old threshold, old model, or pre-embedding metric.
- Leave any exact CPU latency numbers until Arun has produced the final CPU profiling summary.

### 6. Final support for Arun
- Hand Arun these finalized facts before he freezes the README/checklist/tag:
  - exact threshold
  - exact threshold-selection rule and split
  - exact final metric values
  - final System Card path
  - any caveats the README should mention about confidence interpretation or out-of-scope uses

### Pramod handoff status for Arun
- Completed: `outputs/final_system_summary.json`
- Completed: `scripts/validate_release_alignment.py`
- Completed: `reports/milestone4_system_card.pdf`
- Final threshold: `0.3969849246231156`
- Threshold-selection split: `val`
- Threshold-selection rule: `balanced_acc`
- Final embedding run IDs: `run_06` (val sweep), `run_07` (test final)
- Final test metrics: balanced accuracy `0.980`, F1 `0.980`, EER `0.024`, TP `489`, FP `9`, TN `491`, FN `11`
- Final System Card path: `reports/milestone4_system_card.pdf`
- README caveat: confidence is a sigmoid transform of distance from the threshold, not a calibrated probability of identity match
- README caveat: intended for benchmark/course use, not high-stakes or surveillance deployment
- README caveat: do not make subgroup fairness claims because this project does not include reliable demographic metadata

---

## Arun's Tasks

### 1. Build the Milestone 4 profiling entrypoint
- Create or refine a dedicated profiling script, recommended path:
  - `scripts/profile_inference.py`
- It must measure the final embedding-based system on **CPU** and save a machine-readable summary to:
  - `outputs/profiling/cpu_profile_summary.json`
- The profiling script must report at minimum:
  - preprocessing latency
  - embedding latency
  - scoring latency
  - end-to-end latency
  - throughput where useful
  - batch-size sensitivity across a practical set of batch sizes

Recommended JSON shape:
```json
{
  "device": "cpu",
  "os": "Windows 11",
  "hardware": {
    "cpu": "TBD"
  },
  "methodology": {
    "num_repeats": 0,
    "warmup_runs": 0,
    "input_source": "outputs/pairs_test.npz",
    "batch_sizes": [1, 2, 4, 8, 16]
  },
  "batch_results": [
    {
      "batch_size": 1,
      "preprocess_ms_mean": 0.0,
      "embed_ms_mean": 0.0,
      "score_ms_mean": 0.0,
      "end_to_end_ms_mean": 0.0,
      "throughput_pairs_per_s": 0.0
    }
  ]
}
```

### 2. Run the required CPU baseline profile
- Run the profiling entrypoint on Arun’s Windows machine using CPU only.
- Capture the actual hardware and OS details in the profiling summary and report.
- Use a consistent timing methodology:
  - warmup runs
  - repeated measurements
  - same input source/config across all tested batch sizes
- The CPU profile is the required baseline submission artifact.

### 3. Analyze batch-size sensitivity
- Use at least a small practical range of batch sizes, for example:
  - `1, 2, 4, 8, 16`
- Report how latency and throughput change as batch size increases.
- Briefly interpret the tradeoff instead of only dumping a table.
- If memory or runtime on Windows makes a batch size impractical, document that honestly and keep the tested range feasible.

### 4. Prepare the final profiling report
Create `reports/milestone4_profiling_report.pdf`.

Required sections:
- **Measurement environment**
  - OS
  - CPU details
  - Python / dependency assumptions if relevant
- **Methodology**
  - how timing was measured
  - warmup/repeat policy
  - input source and final system version being profiled
- **Per-stage latency**
  - preprocessing vs embedding vs scoring
  - end-to-end latency
- **CPU baseline**
  - clearly labeled required baseline result
- **Batch-size sensitivity**
  - latency/throughput table or figure
  - short interpretation
- **Interpretation**
  - which stage dominates latency
  - what that implies for the final system

### 5. Prepare the reproducibility checklist
Create `reports/milestone4_reproducibility_checklist.md`.

It must include:
- exact environment setup commands
- exact Docker build command
- exact CLI command for a sample pair
- exact command to reproduce the profiling summary
- exact command to run tests and the release-alignment check
- locations of:
  - System Card
  - profiling report
  - final config
  - final tag

### 6. Finalize the README for the final release
- Add a **Milestone 4 / Final Release** section.
- Make the README the clear entry point for the grader.
- It must include:
  - short final project overview
  - final pipeline summary
  - artifact locations
  - Docker command
  - local CLI command
  - profiling command
  - testing command
  - reproducibility checklist path
  - final tag name/path once ready

### 7. Run the clean-clone final check
- Arun owns the final release gate.
- From a fresh clone, follow the README and the reproducibility checklist exactly.
- Confirm:
  - environment setup works
  - Dockerized CLI works
  - profiling command works
  - the key final artifacts exist where the README says they do
  - the release-alignment check passes
- Only after all of that passes should Arun create and push the final tag:
```bash
git tag v1.0-final
git push origin v1.0-final
```

---

## Shared / Dependency Notes

### Dependencies that are necessary
- Arun cannot finalize the README/checklist/tag until Pramod gives him:
  - `outputs/final_system_summary.json`
  - final System Card path
  - final threshold/metric wording
- Pramod should not finalize the System Card’s operational-constraints wording until Arun gives him:
  - `outputs/profiling/cpu_profile_summary.json`
  - final CPU baseline wording

### Dependencies that should be avoided
- Do not have both people editing `README.md` at the same time.
- Do not have both people editing the same final PDF artifact.
- Do not change `configs/inference_config.json` after Arun begins the clean-clone release check unless both of you explicitly re-sync every artifact.

### Low-blocking execution order
1. Pramod freezes the final system version and writes `outputs/final_system_summary.json`.
2. Arun builds/runs the CPU profiling path and produces `outputs/profiling/cpu_profile_summary.json`.
3. Pramod finalizes the System Card using the profiling summary for operational constraints.
4. Arun finalizes the profiling report, reproducibility checklist, and README.
5. Arun runs the clean-clone verification path and pushes `v1.0-final`.

---

## Suggested final commands for the grader path

These should appear in the final README/checklist after Arun finalizes them:

```bash
# Environment
uv venv .venv
uv pip install -r requirements.txt

# Local CLI inference
uv run python scripts/verify.py --img1 data/img1.jpg --img2 data/img2.jpg

# Docker build + CLI inference
docker build -t face-verifier .
docker run --rm -v $(pwd)/data:/app/data face-verifier \
    python scripts/verify.py --img1 data/img1.jpg --img2 data/img2.jpg

# CPU profiling
uv run python scripts/profile_inference.py --device cpu --output outputs/profiling/cpu_profile_summary.json

# Tests + release alignment
uv run python -m pytest tests/ -v
uv run python scripts/validate_release_alignment.py
```

---

## Definition of Done (Milestone 4)

**Pramod**
- [x] Final embedding-based system version frozen and recorded in `outputs/final_system_summary.json`
- [x] Final threshold, rule, split, and key metrics confirmed from `outputs/runs_log.json`
- [x] Release-alignment helper added (`scripts/validate_release_alignment.py` or equivalent)
- [x] `reports/milestone4_system_card.pdf` completed
- [x] System Card includes intended use, limitations, failure modes, fairness-risk discussion, operating threshold, metrics, and reproducibility pointer

**Arun**
- [ ] `scripts/profile_inference.py` (or equivalent profiling entrypoint) implemented and working
- [ ] `outputs/profiling/cpu_profile_summary.json` produced on CPU with stage-wise latency + batch-size sensitivity
- [ ] `reports/milestone4_profiling_report.pdf` completed
- [ ] `reports/milestone4_reproducibility_checklist.md` completed
- [ ] `README.md` updated to reflect the final release and all artifact locations
- [ ] Clean-clone final check completed
- [ ] Final tag `v1.0-final` created and pushed

**Shared grader-facing outcomes**
- [ ] README, System Card, profiling report, config, and final tag all refer to the same final system version
- [ ] CPU baseline is clearly reported
- [ ] Batch-size sensitivity is clearly reported
- [ ] Dockerized CLI is runnable from a fresh clone
- [ ] Reproducibility checklist contains exact commands and artifact paths
- [ ] Final release is easy for a grader to inspect and reproduce
