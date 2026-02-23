import argparse
import json
import os

import numpy as np
from sklearn.datasets import fetch_lfw_pairs


def _save_pairs(split_name: str, pairs, out_dir: str) -> None:
    img1 = pairs.pairs[:, 0, :, :].astype(np.float32, copy=False)
    img2 = pairs.pairs[:, 1, :, :].astype(np.float32, copy=False)
    label = pairs.target.astype(np.int64, copy=False)

    out_path = os.path.join(out_dir, f"pairs_{split_name}.npz")
    np.savez_compressed(out_path, img1=img1, img2=img2, label=label)

    negatives = int((label == 0).sum())
    positives = int((label == 1).sum())
    print(
        f"{split_name}: saved {out_path} | total={label.shape[0]} "
        f"neg={negatives} pos={positives}"
    )


def _load_manifest(manifest_path: str) -> dict:
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"Manifest not found at {manifest_path}. "
            "Run `python scripts/ingest_lfw.py --seed 42` first."
        )
    with open(manifest_path, "r") as f:
        return json.load(f)


def generate_pairs(
    seed: int | None, data_home: str, out_dir: str, manifest_path: str
) -> None:
    manifest = _load_manifest(manifest_path)
    split_policy = manifest.get("split_policy")
    if split_policy != "dev_train_test":
        raise ValueError(
            f"Unsupported split_policy in manifest: {split_policy}. "
            "Expected 'dev_train_test'."
        )

    effective_seed = manifest.get("seed", 42) if seed is None else seed
    np.random.seed(effective_seed)
    os.makedirs(out_dir, exist_ok=True)

    lfw_cache_dir = os.path.join(data_home, "lfw_home")
    if not os.path.isdir(lfw_cache_dir):
        raise FileNotFoundError(
            f"LFW cache directory not found at {lfw_cache_dir}. "
            "Run `python scripts/ingest_lfw.py --seed 42` first."
        )

    try:
        train_pairs = fetch_lfw_pairs(
            data_home=data_home,
            subset="train",
            download_if_missing=False,
        )
        test_pairs = fetch_lfw_pairs(
            data_home=data_home,
            subset="test",
            download_if_missing=False,
        )
    except OSError as exc:
        raise FileNotFoundError(
            "LFW pair data is missing from local cache. "
            "Run `python scripts/ingest_lfw.py --seed 42` first."
        ) from exc

    expected_shape = tuple(manifest.get("image_shape", []))
    if expected_shape and train_pairs.pairs.shape[2:] != expected_shape:
        raise ValueError(
            f"Manifest image_shape {expected_shape} does not match loaded data "
            f"{train_pairs.pairs.shape[2:]}."
        )

    expected_train = manifest.get("train_count")
    expected_test = manifest.get("test_count")
    if expected_train is not None and train_pairs.pairs.shape[0] != expected_train:
        raise ValueError(
            f"Manifest train_count {expected_train} does not match loaded data "
            f"{train_pairs.pairs.shape[0]}."
        )
    if expected_test is not None and test_pairs.pairs.shape[0] != expected_test:
        raise ValueError(
            f"Manifest test_count {expected_test} does not match loaded data "
            f"{test_pairs.pairs.shape[0]}."
        )

    _save_pairs("train", train_pairs, out_dir)
    _save_pairs("test", test_pairs, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate deterministic LFW train/test verification pairs"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed override. Defaults to seed from manifest.",
    )
    parser.add_argument(
        "--data-home",
        type=str,
        default="./data",
        help="Directory for LFW data cache",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./outputs",
        help="Directory to save pair npz files",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="./outputs/manifest.json",
        help="Path to manifest produced by ingest_lfw.py",
    )
    args = parser.parse_args()

    generate_pairs(
        seed=args.seed,
        data_home=args.data_home,
        out_dir=args.out_dir,
        manifest_path=args.manifest,
    )
