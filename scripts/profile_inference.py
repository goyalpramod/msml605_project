"""CPU profiling entrypoint for the final embedding-based face verifier.

Measures stage-wise latency (preprocess / embed / score) and end-to-end
latency for the FaceNet pipeline using true stacked batching across a set
of batch sizes. Writes a machine-readable JSON summary plus a sidecar
markdown table for the profiling report.

Usage:
    python scripts/profile_inference.py
    python scripts/profile_inference.py --batch-sizes 1,2,4,8,16 --repeats 10
    python scripts/profile_inference.py --num-pairs 64 --warmup 3

The script reuses the production preprocessing, model, similarity, and
confidence functions; the only difference from `src/inference.py` is that
forward passes here run on a stacked batch of B images at once instead
of one image at a time. Production `verify_pair()` semantics are unchanged.
"""

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.confidence import compute_confidence
from src.embedder import load_model, preprocess_image
from src.similarity import cosine_similarity

DEFAULT_BATCH_SIZES = [1, 2, 4, 8, 16]
DEFAULT_OUTPUT = "outputs/profiling/cpu_profile_summary.json"
DEFAULT_PAIRS = "outputs/pairs_test.npz"
DEFAULT_CONFIG = "configs/inference_config.json"


def _capture_hardware() -> dict:
    info = {
        "cpu": platform.processor() or "unknown",
        "machine": platform.machine(),
        "physical_cores": "unknown",
        "logical_cores": os.cpu_count(),
    }
    try:
        import psutil  # type: ignore

        info["physical_cores"] = psutil.cpu_count(logical=False)
        info["logical_cores"] = psutil.cpu_count(logical=True)
    except ImportError:
        pass
    return info


def _capture_software() -> dict:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
    }


def _load_pairs(pairs_path: str, num_pairs: int) -> tuple[np.ndarray, np.ndarray]:
    if not os.path.isfile(pairs_path):
        raise FileNotFoundError(f"Pair file not found: {pairs_path}")
    data = np.load(pairs_path)
    img1 = data["img1"]
    img2 = data["img2"]
    n = min(num_pairs, img1.shape[0])
    return img1[:n], img2[:n]


def _stack_preprocess(images: np.ndarray) -> torch.Tensor:
    """Preprocess a batch of images and stack into shape (B, 3, 160, 160)."""
    tensors = [preprocess_image(images[i]) for i in range(images.shape[0])]
    return torch.cat(tensors, dim=0)


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return x / norms


def _profile_batch_size(
    model,
    img1_pool: np.ndarray,
    img2_pool: np.ndarray,
    batch_size: int,
    threshold: float,
    warmup: int,
    repeats: int,
) -> dict:
    pool_size = img1_pool.shape[0]
    if pool_size < batch_size:
        raise ValueError(
            f"Pair pool ({pool_size}) smaller than batch_size ({batch_size}); "
            f"increase --num-pairs."
        )

    # Warmup: full pipeline forward passes, timings discarded.
    for w in range(warmup):
        offset = (w * batch_size) % (pool_size - batch_size + 1)
        a = img1_pool[offset : offset + batch_size]
        b = img2_pool[offset : offset + batch_size]
        ta = _stack_preprocess(a)
        tb = _stack_preprocess(b)
        with torch.no_grad():
            _ = model(ta)
            _ = model(tb)

    pre_times = []
    emb_times = []
    score_times = []
    e2e_times = []

    for r in range(repeats):
        offset = (r * batch_size) % (pool_size - batch_size + 1)
        a = img1_pool[offset : offset + batch_size]
        b = img2_pool[offset : offset + batch_size]

        wall_start = time.perf_counter()

        t0 = time.perf_counter()
        ta = _stack_preprocess(a)
        tb = _stack_preprocess(b)
        t1 = time.perf_counter()
        pre_ms = (t1 - t0) * 1000.0

        t0 = time.perf_counter()
        with torch.no_grad():
            raw_a = model(ta).cpu().numpy()
            raw_b = model(tb).cpu().numpy()
        emb_a = _l2_normalize(raw_a)
        emb_b = _l2_normalize(raw_b)
        t1 = time.perf_counter()
        emb_ms = (t1 - t0) * 1000.0

        t0 = time.perf_counter()
        scores = cosine_similarity(emb_a, emb_b)
        decisions = scores >= threshold
        confidences = [compute_confidence(float(s), threshold) for s in scores]
        t1 = time.perf_counter()
        score_ms = (t1 - t0) * 1000.0

        wall_end = time.perf_counter()
        e2e_ms = (wall_end - wall_start) * 1000.0

        pre_times.append(pre_ms)
        emb_times.append(emb_ms)
        score_times.append(score_ms)
        e2e_times.append(e2e_ms)

        # Use values so they aren't optimized away.
        _ = bool(decisions[0]), confidences[0]

    def _mean_std(xs):
        return (
            float(statistics.mean(xs)),
            float(statistics.stdev(xs)) if len(xs) > 1 else 0.0,
        )

    pre_mean, pre_std = _mean_std(pre_times)
    emb_mean, emb_std = _mean_std(emb_times)
    score_mean, score_std = _mean_std(score_times)
    e2e_mean, e2e_std = _mean_std(e2e_times)

    throughput = (batch_size / (e2e_mean / 1000.0)) if e2e_mean > 0 else 0.0

    return {
        "batch_size": batch_size,
        "preprocess_ms_mean": round(pre_mean, 4),
        "preprocess_ms_std": round(pre_std, 4),
        "embed_ms_mean": round(emb_mean, 4),
        "embed_ms_std": round(emb_std, 4),
        "score_ms_mean": round(score_mean, 4),
        "score_ms_std": round(score_std, 4),
        "end_to_end_ms_mean": round(e2e_mean, 4),
        "end_to_end_ms_std": round(e2e_std, 4),
        "throughput_pairs_per_s": round(throughput, 4),
    }


def _print_table(results: list) -> None:
    header = (
        f"{'batch':>5} | {'preprocess_ms':>14} | {'embed_ms':>10} | "
        f"{'score_ms':>10} | {'e2e_ms':>10} | {'pairs/s':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['batch_size']:>5} | "
            f"{r['preprocess_ms_mean']:>14.3f} | "
            f"{r['embed_ms_mean']:>10.3f} | "
            f"{r['score_ms_mean']:>10.3f} | "
            f"{r['end_to_end_ms_mean']:>10.3f} | "
            f"{r['throughput_pairs_per_s']:>10.3f}"
        )


def _write_markdown_sidecar(summary: dict, md_path: str) -> None:
    lines = []
    hw = summary["hardware"]
    sw = summary["software"]
    lines.append("# CPU Profile Summary (auto-generated)")
    lines.append("")
    lines.append(f"- Device: `{summary['device']}`")
    lines.append(f"- OS: `{summary['os']}`")
    lines.append(
        f"- CPU: `{hw.get('cpu')}` (physical={hw.get('physical_cores')}, "
        f"logical={hw.get('logical_cores')})"
    )
    lines.append(
        f"- Software: torch `{sw['torch']}`, numpy `{sw['numpy']}`, "
        f"python `{sw['python']}`"
    )
    method = summary["methodology"]
    lines.append(
        f"- Methodology: warmup={method['warmup_runs']}, "
        f"repeats={method['num_repeats']}, "
        f"timer=`{method['timer']}`, "
        f"torch_num_threads={method['torch_num_threads']}"
    )
    lines.append(f"- Input source: `{method['input_source']}`")
    lines.append(
        f"- Final system: threshold={summary['config']['threshold']:.4f}, "
        f"model={summary['config']['embedding_model']}, "
        f"dim={summary['config']['embedding_dim']}"
    )
    lines.append("")
    lines.append("## Per-batch results (mean)")
    lines.append("")
    lines.append(
        "| batch_size | preprocess_ms | embed_ms | score_ms | "
        "end_to_end_ms | throughput_pairs_per_s |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for r in summary["batch_results"]:
        lines.append(
            f"| {r['batch_size']} | "
            f"{r['preprocess_ms_mean']:.3f} ± {r['preprocess_ms_std']:.3f} | "
            f"{r['embed_ms_mean']:.3f} ± {r['embed_ms_std']:.3f} | "
            f"{r['score_ms_mean']:.3f} ± {r['score_ms_std']:.3f} | "
            f"{r['end_to_end_ms_mean']:.3f} ± {r['end_to_end_ms_std']:.3f} | "
            f"{r['throughput_pairs_per_s']:.3f} |"
        )
    lines.append("")
    Path(md_path).write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "CPU profiling for the final embedding-based face verifier. "
            "Measures per-stage and end-to-end latency across batch sizes."
        )
    )
    parser.add_argument("--device", default="cpu", choices=["cpu"])
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--batch-sizes",
        default=",".join(str(b) for b in DEFAULT_BATCH_SIZES),
        help="Comma-separated batch sizes (default: 1,2,4,8,16).",
    )
    parser.add_argument("--num-pairs", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--pairs", default=DEFAULT_PAIRS)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    batch_sizes = [int(b.strip()) for b in args.batch_sizes.split(",") if b.strip()]
    if not batch_sizes:
        print("Error: --batch-sizes must contain at least one value.", file=sys.stderr)
        return 1
    if args.num_pairs < max(batch_sizes):
        print(
            f"Error: --num-pairs ({args.num_pairs}) must be >= max batch size "
            f"({max(batch_sizes)}).",
            file=sys.stderr,
        )
        return 1

    with open(args.config) as f:
        config = json.load(f)
    threshold = float(config["threshold"])

    img1_pool, img2_pool = _load_pairs(args.pairs, args.num_pairs)

    torch.set_num_threads(os.cpu_count() or 1)
    print(f"Loading FaceNet model on CPU (torch_num_threads={torch.get_num_threads()})")
    model = load_model()

    print(
        f"Profiling batch sizes={batch_sizes}, "
        f"warmup={args.warmup}, repeats={args.repeats}, "
        f"num_pairs={args.num_pairs}"
    )
    print()

    batch_results = []
    for B in batch_sizes:
        print(f"  -> batch_size={B} ...", flush=True)
        result = _profile_batch_size(
            model=model,
            img1_pool=img1_pool,
            img2_pool=img2_pool,
            batch_size=B,
            threshold=threshold,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        batch_results.append(result)

    summary = {
        "device": args.device,
        "os": platform.platform(),
        "hardware": _capture_hardware(),
        "software": _capture_software(),
        "config": {
            "threshold": threshold,
            "embedding_model": config.get("embedding_model"),
            "embedding_dim": config.get("embedding_dim"),
            "score_direction": config.get("score_direction"),
            "confidence_formula": config.get("confidence_formula"),
        },
        "methodology": {
            "num_repeats": args.repeats,
            "warmup_runs": args.warmup,
            "input_source": args.pairs,
            "num_pairs_per_batch": args.num_pairs,
            "batch_sizes": batch_sizes,
            "timer": "time.perf_counter",
            "torch_num_threads": torch.get_num_threads(),
        },
        "batch_results": batch_results,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md_path = out_path.with_suffix(".md")
    _write_markdown_sidecar(summary, str(md_path))

    print()
    _print_table(batch_results)
    print()
    print(f"Wrote {out_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
