"""Unit tests for src/tracker.py."""

import json
import os

import pytest

from src.tracker import load_runs, log_run, print_run_summary


SAMPLE_CONFIG = {
    "similarity": "cosine",
    "pairs": "outputs/pairs_val.npz",
    "mode": "sweep",
}
SAMPLE_METRICS = {"balanced_acc": 0.85, "f1": 0.82}


class TestLogRun:
    def test_creates_file_if_missing(self, tmp_path):
        log_path = str(tmp_path / "runs_log.json")
        log_run("run_01", SAMPLE_CONFIG, SAMPLE_METRICS, 0.55, "test run", log_path)
        assert os.path.exists(log_path)

    def test_entry_has_required_keys(self, tmp_path):
        log_path = str(tmp_path / "runs_log.json")
        log_run("run_01", SAMPLE_CONFIG, SAMPLE_METRICS, 0.55, "test run", log_path)
        runs = json.load(open(log_path))
        entry = runs[0]
        for key in (
            "run_id",
            "timestamp",
            "commit",
            "config",
            "metrics",
            "threshold",
            "note",
        ):
            assert key in entry

    def test_appends_not_overwrites(self, tmp_path):
        log_path = str(tmp_path / "runs_log.json")
        log_run("run_01", SAMPLE_CONFIG, SAMPLE_METRICS, 0.55, "first", log_path)
        log_run("run_02", SAMPLE_CONFIG, SAMPLE_METRICS, 0.60, "second", log_path)
        runs = json.load(open(log_path))
        assert len(runs) == 2
        assert runs[0]["run_id"] == "run_01"
        assert runs[1]["run_id"] == "run_02"

    def test_stores_correct_values(self, tmp_path):
        log_path = str(tmp_path / "runs_log.json")
        log_run("run_01", SAMPLE_CONFIG, SAMPLE_METRICS, 0.55, "my note", log_path)
        entry = json.load(open(log_path))[0]
        assert entry["run_id"] == "run_01"
        assert entry["threshold"] == pytest.approx(0.55)
        assert entry["note"] == "my note"
        assert entry["config"] == SAMPLE_CONFIG
        assert entry["metrics"] == SAMPLE_METRICS

    def test_creates_parent_directory(self, tmp_path):
        log_path = str(tmp_path / "subdir" / "runs_log.json")
        log_run("run_01", SAMPLE_CONFIG, SAMPLE_METRICS, 0.55, "test", log_path)
        assert os.path.exists(log_path)

    def test_same_run_id_replaces_not_duplicates(self, tmp_path):
        log_path = str(tmp_path / "runs_log.json")
        log_run("run_01", SAMPLE_CONFIG, SAMPLE_METRICS, 0.55, "first", log_path)
        log_run("run_01", SAMPLE_CONFIG, SAMPLE_METRICS, 0.60, "replaced", log_path)
        runs = json.load(open(log_path))
        assert len(runs) == 1
        assert runs[0]["threshold"] == pytest.approx(0.60)
        assert runs[0]["note"] == "replaced"

        
class TestLoadRuns:
    def test_returns_empty_list_if_no_file(self, tmp_path):
        log_path = str(tmp_path / "missing.json")
        assert load_runs(log_path) == []

    def test_returns_list_of_dicts(self, tmp_path):
        log_path = str(tmp_path / "runs_log.json")
        log_run("run_01", SAMPLE_CONFIG, SAMPLE_METRICS, 0.55, "test", log_path)
        runs = load_runs(log_path)
        assert isinstance(runs, list)
        assert isinstance(runs[0], dict)

    def test_preserves_order(self, tmp_path):
        log_path = str(tmp_path / "runs_log.json")
        for i in range(3):
            log_run(
                f"run_0{i}", SAMPLE_CONFIG, SAMPLE_METRICS, 0.5 + i * 0.05, "", log_path
            )
        runs = load_runs(log_path)
        assert [r["run_id"] for r in runs] == ["run_00", "run_01", "run_02"]


class TestPrintRunSummary:
    def test_prints_no_runs_message(self, tmp_path, capsys):
        log_path = str(tmp_path / "missing.json")
        print_run_summary(log_path)
        assert "No runs found" in capsys.readouterr().out

    def test_prints_run_ids(self, tmp_path, capsys):
        log_path = str(tmp_path / "runs_log.json")
        log_run("run_01", SAMPLE_CONFIG, SAMPLE_METRICS, 0.55, "sweep", log_path)
        log_run("run_02", SAMPLE_CONFIG, SAMPLE_METRICS, 0.60, "final", log_path)
        print_run_summary(log_path)
        out = capsys.readouterr().out
        assert "run_01" in out
        assert "run_02" in out
