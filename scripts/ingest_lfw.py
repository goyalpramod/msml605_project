import argparse
import json
import os

import numpy as np
from sklearn.datasets import fetch_lfw_pairs


def ingest_lfw_dataset(seed: int):
    if seed is None:
        seed = 42

    np.random.seed(seed)

    train_pairs = fetch_lfw_pairs(data_home="./data", subset="train")
    test_pairs = fetch_lfw_pairs(data_home="./data", subset="test")

    train_count = train_pairs.pairs.shape[0]
    test_count = test_pairs.pairs.shape[0]
    lfw_root = os.path.join("data", "lfw_home", "lfw_funneled")
    total_identities = sum(
        os.path.isdir(os.path.join(lfw_root, name)) for name in os.listdir(lfw_root)
    )

    # fetch_lfw_pairs stores pairs as flattened rows; derive image shape from pair dimensions
    # each row is 2 images concatenated: (N, 2*H*W), and pair_shape gives (2, H, W)
    image_shape = list(train_pairs.pairs.shape[2:])

    manifest = {
        "seed": seed,
        "split_policy": "dev_train_test",
        "train_count": train_count,
        "test_count": test_count,
        "total_identities": total_identities,
        "image_shape": image_shape,
    }

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/manifest.json", "w") as f:
        json.dump(manifest, f, indent=4, sort_keys=True)

    # return tuple of (train_pairs, test_pairs) in case we need either/both
    return train_pairs, test_pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest LFW dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    ingest_lfw_dataset(args.seed)


# to use the data from this script kindly do the following
# from scripts.ingest_lfw import ingest_lfw_dataset
# train_pairs, test_pairs = ingest_lfw_dataset(seed=42)
# print(train_pairs.data.shape)
# print(test_pairs.data.shape)
