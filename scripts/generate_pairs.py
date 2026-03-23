import argparse
import json
import os
from collections import defaultdict

import numpy as np
from sklearn.datasets import fetch_lfw_pairs


def _parse_pair_spec(spec_path: str) -> list[dict]:
    """Parse an LFW pair specification file and return identity info per pair.

    Returns a list of dicts with keys:
      - identities: tuple of identity name(s) involved
      - label: 1 (same person) or 0 (different person)
      - line_index: 0-based index matching the order fetch_lfw_pairs returns
    """
    pairs = []
    with open(spec_path) as f:
        f.readline()  # skip header line (pair count)
        for i, line in enumerate(f):
            parts = line.strip().split("\t")
            if len(parts) == 3:
                # Same-person pair: name  idx1  idx2
                pairs.append(
                    {
                        "identities": (parts[0],),
                        "label": 1,
                        "line_index": i,
                    }
                )
            elif len(parts) == 4:
                # Different-person pair: name1  idx1  name2  idx2
                pairs.append(
                    {
                        "identities": (parts[0], parts[2]),
                        "label": 0,
                        "line_index": i,
                    }
                )
            else:
                raise ValueError(
                    f"Unexpected format at line {i + 2} in {spec_path}: {line.strip()}"
                )
    return pairs


def _save_pairs(split_name: str, img1, img2, label, out_dir: str) -> None:
    img1 = np.asarray(img1, dtype=np.float32)
    img2 = np.asarray(img2, dtype=np.float32)
    label = np.asarray(label, dtype=np.int64)

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
    with open(manifest_path) as f:
        return json.load(f)


def _get_all_identities(pair_specs: list[dict]) -> set[str]:
    """Extract all unique identity names from parsed pair specs."""
    identities = set()
    for spec in pair_specs:
        identities.update(spec["identities"])
    return identities


def _split_identities_for_val(
    pair_specs: list[dict], val_fraction: float, rng: np.random.RandomState
) -> tuple[list[int], list[int]]:
    """Split pair indices into train and val based on identity-level assignment.

    Identities are assigned to val or train. A pair goes to val if ANY of its
    identities are in the val set. This prevents identity leakage.
    """
    all_identities = sorted(_get_all_identities(pair_specs))
    rng.shuffle(all_identities)

    n_val = max(1, int(len(all_identities) * val_fraction))
    val_identities = set(all_identities[:n_val])

    train_indices = []
    val_indices = []
    for spec in pair_specs:
        if any(ident in val_identities for ident in spec["identities"]):
            val_indices.append(spec["line_index"])
        else:
            train_indices.append(spec["line_index"])

    return train_indices, val_indices


def _cap_pairs_per_identity(
    pair_specs: list[dict],
    indices: list[int],
    cap: int,
    rng: np.random.RandomState,
) -> list[int]:
    """Cap the number of pairs per identity, then rebalance to 1:1 ratio.

    Each pair is attributed to all identities it involves. Iteratively removes
    excess pairs until no identity exceeds the cap. After capping, the
    positive:negative ratio is rebalanced to exactly 1:1.
    """
    index_set = set(indices)
    subset_specs = [s for s in pair_specs if s["line_index"] in index_set]
    spec_by_idx = {s["line_index"]: s for s in subset_specs}

    # Start with all pairs as candidates, then iteratively prune
    candidates = set(s["line_index"] for s in subset_specs)

    changed = True
    while changed:
        changed = False
        # Rebuild identity->pairs mapping from current candidates
        identity_pairs: dict[str, list[int]] = defaultdict(list)
        for s in subset_specs:
            if s["line_index"] in candidates:
                for ident in s["identities"]:
                    identity_pairs[ident].append(s["line_index"])

        for ident in sorted(identity_pairs.keys()):
            pair_idxs = [p for p in identity_pairs[ident] if p in candidates]
            if len(pair_idxs) > cap:
                to_keep = set(rng.choice(pair_idxs, size=cap, replace=False).tolist())
                candidates -= set(pair_idxs) - to_keep
                changed = True

    # Rebalance to 1:1 positive:negative ratio
    pos_indices = sorted([i for i in candidates if spec_by_idx[i]["label"] == 1])
    neg_indices = sorted([i for i in candidates if spec_by_idx[i]["label"] == 0])

    n_balanced = min(len(pos_indices), len(neg_indices))
    if n_balanced < len(pos_indices):
        pos_indices = sorted(rng.choice(pos_indices, size=n_balanced, replace=False))
    if n_balanced < len(neg_indices):
        neg_indices = sorted(rng.choice(neg_indices, size=n_balanced, replace=False))

    return sorted(pos_indices + neg_indices)


def generate_pairs(
    seed: int | None,
    data_home: str,
    out_dir: str,
    manifest_path: str,
    val_fraction: float | None = None,
    cap_per_identity: int | None = None,
) -> None:
    manifest = _load_manifest(manifest_path)
    split_policy = manifest.get("split_policy")
    if split_policy != "dev_train_test":
        raise ValueError(
            f"Unsupported split_policy in manifest: {split_policy}. "
            "Expected 'dev_train_test'."
        )

    effective_seed = manifest.get("seed", 42) if seed is None else seed
    rng = np.random.RandomState(effective_seed)
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

    # Parse pair specification files to get identity information
    train_spec_path = os.path.join(lfw_cache_dir, "pairsDevTrain.txt")
    test_spec_path = os.path.join(lfw_cache_dir, "pairsDevTest.txt")
    train_specs = _parse_pair_spec(train_spec_path)
    test_specs = _parse_pair_spec(test_spec_path)

    # Extract image arrays
    train_img1 = train_pairs.pairs[:, 0, :, :].astype(np.float32, copy=False)
    train_img2 = train_pairs.pairs[:, 1, :, :].astype(np.float32, copy=False)
    train_label = train_pairs.target.astype(np.int64, copy=False)

    test_img1 = test_pairs.pairs[:, 0, :, :].astype(np.float32, copy=False)
    test_img2 = test_pairs.pairs[:, 1, :, :].astype(np.float32, copy=False)
    test_label = test_pairs.target.astype(np.int64, copy=False)

    # --- Baseline train/test (always saved) ---
    if val_fraction is not None:
        # Split train into train + val at the identity level
        train_indices, val_indices = _split_identities_for_val(
            train_specs, val_fraction, rng
        )
        _save_pairs(
            "train",
            train_img1[train_indices],
            train_img2[train_indices],
            train_label[train_indices],
            out_dir,
        )
        _save_pairs(
            "val",
            train_img1[val_indices],
            train_img2[val_indices],
            train_label[val_indices],
            out_dir,
        )
    else:
        _save_pairs("train", train_img1, train_img2, train_label, out_dir)

    _save_pairs("test", test_img1, test_img2, test_label, out_dir)

    # --- Capped versions (if requested) ---
    if cap_per_identity is not None:
        if val_fraction is not None:
            val_capped_indices = _cap_pairs_per_identity(
                train_specs, val_indices, cap_per_identity, rng
            )
            _save_pairs(
                "val_capped",
                train_img1[val_capped_indices],
                train_img2[val_capped_indices],
                train_label[val_capped_indices],
                out_dir,
            )

        # Cap test pairs
        all_test_indices = list(range(len(test_specs)))
        test_capped_indices = _cap_pairs_per_identity(
            test_specs, all_test_indices, cap_per_identity, rng
        )
        _save_pairs(
            "test_capped",
            test_img1[test_capped_indices],
            test_img2[test_capped_indices],
            test_label[test_capped_indices],
            out_dir,
        )

    # Save metadata
    meta = {
        "seed": effective_seed,
        "val_fraction": val_fraction,
        "cap_per_identity": cap_per_identity,
    }
    meta_path = os.path.join(out_dir, "pairs_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"metadata: saved {meta_path}")


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
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=None,
        help="Fraction of identities to hold out for validation (e.g. 0.15)",
    )
    parser.add_argument(
        "--cap-per-identity",
        type=int,
        default=None,
        help="Max pairs per identity for capped outputs (e.g. 10)",
    )
    args = parser.parse_args()

    generate_pairs(
        seed=args.seed,
        data_home=args.data_home,
        out_dir=args.out_dir,
        manifest_path=args.manifest,
        val_fraction=args.val_fraction,
        cap_per_identity=args.cap_per_identity,
    )
