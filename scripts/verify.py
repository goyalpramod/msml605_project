"""CLI inference script for face verification.

Usage examples:
  # Single pair
  python scripts/verify.py --img1 path/to/img1.jpg --img2 path/to/img2.jpg

  # Batch mode — CSV with header "img1,img2"
  python scripts/verify.py --batch path/to/pairs.csv

  # Override config path (defaults to configs/inference_config.json)
  python scripts/verify.py --img1 a.jpg --img2 b.jpg --config path/to/config.json

Reads the decision threshold from the config file and calls
`verify_pair()` from src/inference.py. Prints a per-pair result block:

  Pair:        <img1_basename> vs <img2_basename>
  Score:       0.823
  Threshold:   0.397
  Decision:    SAME
  Confidence:  0.91
  Latency:     47.3 ms

Exit codes:
  0 — success
  1 — bad input (missing file, malformed CSV, invalid image)
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference import verify_pair

DEFAULT_CONFIG_PATH = "configs/inference_config.json"


def load_config(config_path: str) -> dict:
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path) as f:
        return json.load(f)


def load_image(path: str) -> np.ndarray:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found: {path}")
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def print_result(img1_path: str, img2_path: str, result: dict) -> None:
    decision = "SAME" if result["decision"] else "DIFFERENT"
    print(
        f"Pair:        {os.path.basename(img1_path)} vs {os.path.basename(img2_path)}"
    )
    print(f"Score:       {result['score']:.3f}")
    print(f"Threshold:   {result['threshold']:.3f}")
    print(f"Decision:    {decision}")
    print(f"Confidence:  {result['confidence']:.2f}")
    print(f"Latency:     {result['latency_ms']:.1f} ms")
    print()


def run_single(img1_path: str, img2_path: str, threshold: float) -> None:
    img1 = load_image(img1_path)
    img2 = load_image(img2_path)
    result = verify_pair(img1, img2, threshold)
    print_result(img1_path, img2_path, result)


def run_batch(csv_path: str, threshold: float) -> None:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Batch CSV not found: {csv_path}")

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if (
            reader.fieldnames is None
            or "img1" not in reader.fieldnames
            or "img2" not in reader.fieldnames
        ):
            raise ValueError(
                f"Batch CSV must have header 'img1,img2'. Got: {reader.fieldnames}"
            )
        rows = list(reader)

    for i, row in enumerate(rows):
        img1_path = row["img1"]
        img2_path = row["img2"]
        img1 = load_image(img1_path)
        img2 = load_image(img2_path)
        result = verify_pair(img1, img2, threshold)
        print_result(img1_path, img2_path, result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Face verification CLI using FaceNet embeddings."
    )
    parser.add_argument("--img1", dest="img1", help="Path to first image.")
    parser.add_argument("--img2", dest="img2", help="Path to second image.")
    parser.add_argument(
        "--batch",
        dest="batch",
        help="Path to CSV with header 'img1,img2' for batch inference.",
    )
    parser.add_argument(
        "--config",
        dest="config",
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to inference config JSON (default: {DEFAULT_CONFIG_PATH}).",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.batch and (args.img1 or args.img2):
        print(
            "Error: --batch is mutually exclusive with --img1/--img2.", file=sys.stderr
        )
        return 1
    if not args.batch and not (args.img1 and args.img2):
        print(
            "Error: provide either --batch CSV or both --img1 and --img2.",
            file=sys.stderr,
        )
        return 1

    try:
        config = load_config(args.config)
        threshold = float(config["threshold"])
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        return 1

    try:
        if args.batch:
            run_batch(args.batch, threshold)
        else:
            run_single(args.img1, args.img2, threshold)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
