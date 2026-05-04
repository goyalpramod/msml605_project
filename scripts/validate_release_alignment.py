"""Validate that final-release artifacts all describe the same system."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

SUMMARY_PATH = Path("outputs/final_system_summary.json")
CONFIG_PATH = Path("configs/inference_config.json")
RUNS_LOG_PATH = Path("outputs/runs_log.json")


def _load_json(path: Path) -> object:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _metric_mismatches(summary_metrics: dict, run_metrics: dict) -> list[str]:
    checks = [
        ("balanced_accuracy", "balanced_acc"),
        ("f1", "f1"),
        ("eer", "eer"),
        ("tp", "TP"),
        ("fp", "FP"),
        ("tn", "TN"),
        ("fn", "FN"),
    ]
    errors = []
    for summary_key, run_key in checks:
        if not _values_match(
            summary_metrics.get(summary_key), run_metrics.get(run_key)
        ):
            errors.append(
                "test metric mismatch for "
                f"{summary_key}: summary={summary_metrics.get(summary_key)!r}, "
                f"runs_log={run_metrics.get(run_key)!r}"
            )
    return errors


def _values_match(left: object, right: object) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
    return left == right


def validate_release_alignment(repo_root: Path | str = ".") -> list[str]:
    root = Path(repo_root)
    errors: list[str] = []

    required_json = {
        "config": root / CONFIG_PATH,
        "runs_log": root / RUNS_LOG_PATH,
        "summary": root / SUMMARY_PATH,
    }
    for label, path in required_json.items():
        if not path.is_file():
            errors.append(f"missing required {label} file: {path.relative_to(root)}")
    if errors:
        return errors

    config = _load_json(root / CONFIG_PATH)
    runs = _load_json(root / RUNS_LOG_PATH)
    summary = _load_json(root / SUMMARY_PATH)
    if (
        not isinstance(config, dict)
        or not isinstance(summary, dict)
        or not isinstance(runs, list)
    ):
        return ["config, runs log, or summary has an unexpected JSON shape"]

    for key in [
        "embedding_model",
        "embedding_dim",
        "threshold",
        "score_direction",
        "confidence_formula",
    ]:
        if not _values_match(summary.get(key), config.get(key)):
            errors.append(
                f"{key} mismatch between config and summary: "
                f"config={config.get(key)!r}, summary={summary.get(key)!r}"
            )

    run_map = {
        run.get("run_id"): run
        for run in runs
        if isinstance(run, dict) and run.get("run_id")
    }
    for run_id in summary.get("final_run_ids", []):
        if run_id not in run_map:
            errors.append(f"missing run_id '{run_id}' referenced by summary")

    run_06 = run_map.get("run_06")
    if run_06 is not None:
        if run_06.get("config", {}).get("pairs") != "outputs/pairs_val.npz":
            errors.append("run_06 should use outputs/pairs_val.npz")
        if run_06.get("config", {}).get("mode") != "sweep":
            errors.append("run_06 should be a val sweep run")
        if run_06.get("metrics", {}).get("threshold_rule_used") != summary.get(
            "threshold_selection_rule"
        ):
            errors.append(
                "threshold-selection rule mismatch between run_06 and summary"
            )
        if not _values_match(run_06.get("threshold"), summary.get("threshold")):
            errors.append("threshold mismatch between run_06 and summary")

    run_07 = run_map.get("run_07")
    if run_07 is not None:
        if run_07.get("config", {}).get("pairs") != "outputs/pairs_test.npz":
            errors.append("run_07 should use outputs/pairs_test.npz")
        if run_07.get("config", {}).get("mode") != "final":
            errors.append("run_07 should be a final test run")
        if not _values_match(run_07.get("threshold"), summary.get("threshold")):
            errors.append("threshold mismatch between run_07 and summary")
        errors.extend(
            _metric_mismatches(
                summary.get("test_metrics", {}),
                run_07.get("metrics", {}),
            )
        )

    artifact_paths = summary.get("artifact_paths", {})
    if not isinstance(artifact_paths, dict):
        errors.append("summary artifact_paths must be a JSON object")
    else:
        for label, rel_path in artifact_paths.items():
            artifact = root / rel_path
            if not artifact.exists():
                errors.append(f"missing artifact for {label}: {rel_path}")

    for rel_path in [
        "reports/milestone4_profiling_report.pdf",
        "reports/milestone4_reproducibility_checklist.md",
        "reports/milestone4_system_card.pdf",
    ]:
        if not (root / rel_path).exists():
            errors.append(f"missing expected final report file: {rel_path}")

    return errors


def main() -> int:
    errors = validate_release_alignment(Path.cwd())
    if errors:
        print("Release alignment check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print("Release alignment check passed.")
    print(f"- Config: {CONFIG_PATH}")
    print(f"- Summary: {SUMMARY_PATH}")
    print(f"- Runs log: {RUNS_LOG_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
