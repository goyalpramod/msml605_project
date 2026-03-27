"""Run logging for the face verification evaluation pipeline.

Appends one JSON entry per evaluation run to outputs/runs_log.json.
Each entry records enough context to reproduce or compare any run.
"""

import json
import os
import subprocess
from datetime import datetime, timezone


def _get_git_commit() -> str:
    """Return the current HEAD commit hash, or 'unknown' if git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def log_run(
    run_id: str,
    config: dict,
    metrics: dict,
    threshold: float,
    note: str,
    log_path: str = "outputs/runs_log.json",
) -> None:
    """Append a run entry to the JSON log file.

    Creates the file if it does not exist. Never overwrites existing entries.

    Args:
        run_id:    Unique identifier for this run (e.g. "run_01").
        config:    Dict describing the run setup (similarity, pairs file, mode, rule).
        metrics:   Dict of computed metrics (accuracy, f1, eer, etc.).
        threshold: The decision threshold used or selected in this run.
        note:      Short human-readable description of the run's purpose.
        log_path:  Path to the JSON log file.
    """
    entry = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "commit": _get_git_commit(),
        "config": config,
        "metrics": metrics,
        "threshold": threshold,
        "note": note,
    }

    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            runs = json.load(f)
    else:
        runs = []

    # Replace existing entry with same run_id (idempotent re-runs)
    runs = [r for r in runs if r.get("run_id") != run_id]
    runs.append(entry)

    with open(log_path, "w") as f:
        json.dump(runs, f, indent=2)


def load_runs(log_path: str = "outputs/runs_log.json") -> list[dict]:
    """Load all run entries from the log file.

    Returns:
        List of run dicts, in the order they were logged.
        Returns an empty list if the file does not exist.
    """
    if not os.path.exists(log_path):
        return []
    with open(log_path, "r") as f:
        return json.load(f)


def print_run_summary(log_path: str = "outputs/runs_log.json") -> None:
    """Print a tabular summary of all logged runs to stdout."""
    runs = load_runs(log_path)
    if not runs:
        print(f"No runs found in {log_path}")
        return

    header = f"{'run_id':<10} {'timestamp':<28} {'threshold':>10}  {'note'}"
    print(header)
    print("-" * len(header))
    for run in runs:
        ts = run.get("timestamp", "")[:19]  # trim microseconds
        print(
            f"{run['run_id']:<10} {ts:<28} {run['threshold']:>10.4f}  {run.get('note', '')}"
        )
