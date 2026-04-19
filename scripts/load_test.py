"""Concurrent load test for face verification inference.

Usage:
  python scripts/load_test.py --num-pairs 100 --workers 4 --seed 42 \
      --pairs outputs/pairs_test.npz \
      --output outputs/load_test_results.json

Samples a deterministic subset of pairs from outputs/pairs_test.npz and runs
`verify_pair()` concurrently across a ProcessPoolExecutor. Each worker
pre-loads the FaceNet model in its initializer so per-request latency
reflects inference time, not weight loading.

Output JSON schema:
  {
    "total_requests": int,
    "workers": int,
    "wall_clock_s": float,
    "throughput_rps": float,
    "latency_p50_ms": float,
    "latency_p95_ms": float,
    "latency_mean_ms": float,
    "failures": int
  }
"""

import argparse
import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference import verify_pair

DEFAULT_CONFIG_PATH = "configs/inference_config.json"
DEFAULT_PAIRS_PATH = "outputs/pairs_test.npz"
DEFAULT_OUTPUT_PATH = "outputs/load_test_results.json"


def _warmup_worker() -> None:
    """Pre-load FaceNet model into each worker's process-local singleton."""
    from src.inference import _get_model

    _get_model()


def _run_one(args: tuple) -> dict:
    """Worker function: run a single verify_pair call."""
    img1, img2, threshold = args
    return verify_pair(img1, img2, threshold)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return json.load(f)


def sample_pairs(
    pairs_path: str, num_pairs: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic subset of (img1, img2) arrays, shape (k, H, W)."""
    if not os.path.isfile(pairs_path):
        raise FileNotFoundError(f"Pairs file not found: {pairs_path}")
    data = np.load(pairs_path)
    img1_all = data["img1"]
    img2_all = data["img2"]
    n = img1_all.shape[0]
    if num_pairs > n:
        raise ValueError(f"Requested {num_pairs} pairs but only {n} available.")

    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=num_pairs, replace=False)
    return img1_all[idx], img2_all[idx]


def run_load_test(
    img1: np.ndarray,
    img2: np.ndarray,
    threshold: float,
    workers: int,
) -> dict:
    n = img1.shape[0]
    task_args = [(img1[i], img2[i], threshold) for i in range(n)]

    latencies_ms: list[float] = []
    failures = 0

    t_start = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers, initializer=_warmup_worker
    ) as executor:
        futures = [executor.submit(_run_one, a) for a in task_args]
        for fut in concurrent.futures.as_completed(futures):
            try:
                result = fut.result()
                latencies_ms.append(float(result["latency_ms"]))
            except Exception as e:
                failures += 1
                print(f"  [warn] request failed: {e}", file=sys.stderr)
    t_end = time.perf_counter()

    wall_clock_s = t_end - t_start
    latencies = np.asarray(latencies_ms, dtype=np.float64)
    if latencies.size > 0:
        p50 = float(np.percentile(latencies, 50))
        p95 = float(np.percentile(latencies, 95))
        mean = float(latencies.mean())
    else:
        p50 = p95 = mean = 0.0

    throughput = n / wall_clock_s if wall_clock_s > 0 else 0.0

    return {
        "total_requests": n,
        "workers": workers,
        "wall_clock_s": round(wall_clock_s, 3),
        "throughput_rps": round(throughput, 2),
        "latency_p50_ms": round(p50, 2),
        "latency_p95_ms": round(p95, 2),
        "latency_mean_ms": round(mean, 2),
        "failures": failures,
    }


def print_summary(summary: dict) -> None:
    print()
    print("Load test results")
    print("-" * 40)
    print(f"  total requests : {summary['total_requests']}")
    print(f"  workers        : {summary['workers']}")
    print(f"  wall clock     : {summary['wall_clock_s']:.2f} s")
    print(f"  throughput     : {summary['throughput_rps']:.2f} req/s")
    print(f"  latency p50    : {summary['latency_p50_ms']:.2f} ms")
    print(f"  latency p95    : {summary['latency_p95_ms']:.2f} ms")
    print(f"  latency mean   : {summary['latency_mean_ms']:.2f} ms")
    print(f"  failures       : {summary['failures']}")
    print()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Concurrent load test for face verification inference."
    )
    parser.add_argument("--num-pairs", dest="num_pairs", type=int, default=100)
    parser.add_argument("--workers", dest="workers", type=int, default=4)
    parser.add_argument("--seed", dest="seed", type=int, default=42)
    parser.add_argument(
        "--pairs",
        dest="pairs",
        default=DEFAULT_PAIRS_PATH,
        help=f"Pairs .npz file (default: {DEFAULT_PAIRS_PATH}).",
    )
    parser.add_argument(
        "--output",
        dest="output",
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--config",
        dest="config",
        default=DEFAULT_CONFIG_PATH,
        help=f"Inference config path (default: {DEFAULT_CONFIG_PATH}).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        config = load_config(args.config)
        threshold = float(config["threshold"])
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        return 1

    try:
        img1, img2 = sample_pairs(args.pairs, args.num_pairs, args.seed)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error sampling pairs: {e}", file=sys.stderr)
        return 1

    print(
        f"Running load test: {args.num_pairs} pairs, {args.workers} workers, seed={args.seed}"
    )
    summary = run_load_test(img1, img2, threshold, args.workers)
    print_summary(summary)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved results to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
