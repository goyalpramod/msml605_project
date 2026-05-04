import json
from pathlib import Path

from scripts.validate_release_alignment import validate_release_alignment


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_validate_release_alignment_accepts_consistent_release_contract(tmp_path):
    _write_json(
        tmp_path / "configs" / "inference_config.json",
        {
            "embedding_model": "facenet",
            "embedding_dim": 512,
            "threshold": 0.3969849246231156,
            "score_direction": "higher_is_same",
            "confidence_formula": "sigmoid(10 * (score - threshold))",
        },
    )
    _write_json(
        tmp_path / "outputs" / "runs_log.json",
        [
            {
                "run_id": "run_06",
                "config": {"pairs": "outputs/pairs_val.npz", "mode": "sweep"},
                "metrics": {"threshold_rule_used": "balanced_acc"},
                "threshold": 0.3969849246231156,
            },
            {
                "run_id": "run_07",
                "config": {"pairs": "outputs/pairs_test.npz", "mode": "final"},
                "metrics": {
                    "balanced_acc": 0.98,
                    "f1": 0.98,
                    "eer": 0.024,
                    "TP": 489,
                    "FP": 9,
                    "TN": 491,
                    "FN": 11,
                },
                "threshold": 0.3969849246231156,
            },
        ],
    )
    _write_json(
        tmp_path / "outputs" / "final_system_summary.json",
        {
            "release_candidate": "milestone4_final",
            "embedding_model": "facenet",
            "embedding_dim": 512,
            "threshold": 0.3969849246231156,
            "threshold_selection_split": "val",
            "threshold_selection_rule": "balanced_acc",
            "score_direction": "higher_is_same",
            "confidence_formula": "sigmoid(10 * (score - threshold))",
            "final_run_ids": ["run_06", "run_07"],
            "test_metrics": {
                "balanced_accuracy": 0.98,
                "f1": 0.98,
                "eer": 0.024,
                "tp": 489,
                "fp": 9,
                "tn": 491,
                "fn": 11,
            },
            "artifact_paths": {
                "system_card_pdf": "reports/milestone4_system_card.pdf",
                "profiling_report_pdf": "reports/milestone4_profiling_report.pdf",
                "reproducibility_checklist": "reports/milestone4_reproducibility_checklist.md",
            },
        },
    )

    for rel_path in [
        "reports/milestone4_system_card.pdf",
        "reports/milestone4_profiling_report.pdf",
        "reports/milestone4_reproducibility_checklist.md",
    ]:
        artifact = tmp_path / rel_path
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("placeholder", encoding="utf-8")

    assert validate_release_alignment(tmp_path) == []


def test_validate_release_alignment_reports_mismatches_and_missing_artifacts(tmp_path):
    _write_json(
        tmp_path / "configs" / "inference_config.json",
        {
            "embedding_model": "facenet",
            "embedding_dim": 512,
            "threshold": 0.5,
            "score_direction": "higher_is_same",
            "confidence_formula": "sigmoid(10 * (score - threshold))",
        },
    )
    _write_json(
        tmp_path / "outputs" / "runs_log.json",
        [
            {
                "run_id": "run_06",
                "config": {"pairs": "outputs/pairs_val.npz", "mode": "sweep"},
                "metrics": {"threshold_rule_used": "f1"},
                "threshold": 0.4,
            }
        ],
    )
    _write_json(
        tmp_path / "outputs" / "final_system_summary.json",
        {
            "release_candidate": "milestone4_final",
            "embedding_model": "facenet",
            "embedding_dim": 512,
            "threshold": 0.3969849246231156,
            "threshold_selection_split": "val",
            "threshold_selection_rule": "balanced_acc",
            "score_direction": "higher_is_same",
            "confidence_formula": "sigmoid(10 * (score - threshold))",
            "final_run_ids": ["run_06", "run_07"],
            "test_metrics": {
                "balanced_accuracy": 0.98,
                "f1": 0.98,
                "eer": 0.024,
                "tp": 489,
                "fp": 9,
                "tn": 491,
                "fn": 11,
            },
            "artifact_paths": {
                "system_card_pdf": "reports/milestone4_system_card.pdf",
            },
        },
    )

    errors = validate_release_alignment(tmp_path)

    assert any("threshold mismatch" in error for error in errors)
    assert any("missing run_id 'run_07'" in error for error in errors)
    assert any("threshold-selection rule mismatch" in error for error in errors)
    assert any("missing artifact" in error for error in errors)


def test_repository_release_alignment_passes():
    repo_root = Path(__file__).resolve().parent.parent
    assert validate_release_alignment(repo_root) == []
