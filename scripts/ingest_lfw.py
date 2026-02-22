import argparse
import json
import os

import numpy as np
from sklearn.datasets import fetch_lfw_pairs


def ingest_lfw_dataset(seed: int):
    np.random.seed(seed)

    lfw_pairs = fetch_lfw_pairs(data_home="./data")

    n_samples = lfw_pairs.data.shape[0]
    test_count = n_samples // 10
    train_count = n_samples - test_count
    total_identities = len(np.unique(lfw_pairs.target))

    # fetch_lfw_pairs stores pairs as flattened rows; derive image shape from pair dimensions
    # each row is 2 images concatenated: (N, 2*H*W), and pair_shape gives (2, H, W)
    image_shape = list(lfw_pairs.pairs.shape[2:])

    manifest = {
        "seed": seed,
        "split_policy": "10fold",
        "train_count": train_count,
        "test_count": test_count,
        "total_identities": total_identities,
        "image_shape": image_shape,
    }

    with open("outputs/manifest.json", "w") as f:
        json.dump(manifest, f, indent=4, sort_keys=True)

    return lfw_pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest LFW dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    ingest_lfw_dataset(args.seed)


# to use the data from this script kindly do the following
# from scripts.ingest_lfw import ingest_lfw_dataset
# lfw_pairs = ingest_lfw_dataset(seed=42)
# print(lfw_pairs.data.shape)