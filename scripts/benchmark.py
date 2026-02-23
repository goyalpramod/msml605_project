import sys
import time

import numpy as np

sys.path.insert(0, ".")  # allow `from src.*` imports when run from project root

from src.similarity import (
    cosine_similarity,
    cosine_similarity_loop,
    euclidean_distance,
    euclidean_distance_loop,
    load_pair_vectors,
)

TOLERANCE = 1e-6
REPEATS = 50


def _time_fn(fn, a, b, repeats=1):
    start = time.perf_counter()
    for _ in range(repeats):
        result = fn(a, b)
    elapsed = time.perf_counter() - start
    return result, elapsed


def main():
    a, b, _ = load_pair_vectors("test")
    print(f"Loaded {a.shape[0]} pairs  (D={a.shape[1]})  x{REPEATS} repeats\n")

    metrics = [
        ("Cosine", cosine_similarity_loop, cosine_similarity),
        ("Euclid", euclidean_distance_loop, euclidean_distance),
    ]

    for name, loop_fn, vec_fn in metrics:
        loop_result, loop_time = _time_fn(loop_fn, a, b, REPEATS)
        vec_result, vec_time = _time_fn(vec_fn, a, b, REPEATS)

        match = bool(np.allclose(loop_result, vec_result, atol=TOLERANCE))

        print(
            f"{name:<7} | loop: {loop_time:.4f}s | numpy: {vec_time:.4f}s | match: {match}"
        )

        if not match:
            max_diff = float(np.max(np.abs(loop_result - vec_result)))
            print(f"max diff = {max_diff:.2e} (tolerance {TOLERANCE:.0e})")
            sys.exit(1)


if __name__ == "__main__":
    main()
