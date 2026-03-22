"""Tests for scripts/generate_pairs.py — synthetic data only, no downloads."""

import sys
import os

import numpy as np
import pytest

# Allow importing from scripts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.generate_pairs import (
    _cap_pairs_per_identity,
    _parse_pair_spec,
    _save_pairs,
    _split_identities_for_val,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def pair_spec_file(tmp_path):
    """Write a synthetic pair spec file in LFW format.

    Layout (20 pairs total):
      - 10 same-person pairs across 5 identities (Alice x4, Bob x3, Carol x2, Dave x1)
      - 10 different-person pairs
    """
    lines = [
        "10",  # header: number of positive pairs (LFW convention)
        # Same-person pairs (label=1): identity  img_a  img_b
        "Alice\t1\t2",
        "Alice\t1\t3",
        "Alice\t2\t3",
        "Alice\t3\t4",
        "Bob\t1\t2",
        "Bob\t1\t3",
        "Bob\t2\t3",
        "Carol\t1\t2",
        "Carol\t1\t3",
        "Dave\t1\t2",
        # Different-person pairs (label=0): id1  img  id2  img
        "Alice\t1\tBob\t1",
        "Alice\t2\tCarol\t1",
        "Alice\t3\tDave\t1",
        "Bob\t1\tCarol\t1",
        "Bob\t2\tDave\t1",
        "Carol\t1\tDave\t1",
        "Alice\t1\tEve\t1",
        "Bob\t1\tEve\t1",
        "Carol\t1\tEve\t1",
        "Dave\t1\tEve\t1",
    ]
    path = tmp_path / "pairs_test.txt"
    path.write_text("\n".join(lines) + "\n")
    return str(path)


# ---------------------------------------------------------------------------
# _parse_pair_spec
# ---------------------------------------------------------------------------


class TestParsePairSpec:
    def test_total_count(self, pair_spec_file):
        specs = _parse_pair_spec(pair_spec_file)
        assert len(specs) == 20

    def test_same_person_format(self, pair_spec_file):
        specs = _parse_pair_spec(pair_spec_file)
        same = [s for s in specs if s["label"] == 1]
        assert len(same) == 10
        # Same-person pairs have exactly one identity
        for s in same:
            assert len(s["identities"]) == 1

    def test_different_person_format(self, pair_spec_file):
        specs = _parse_pair_spec(pair_spec_file)
        diff = [s for s in specs if s["label"] == 0]
        assert len(diff) == 10
        # Different-person pairs have two distinct identities
        for s in diff:
            assert len(s["identities"]) == 2
            assert s["identities"][0] != s["identities"][1]

    def test_line_indices_sequential(self, pair_spec_file):
        specs = _parse_pair_spec(pair_spec_file)
        indices = [s["line_index"] for s in specs]
        assert indices == list(range(20))


# ---------------------------------------------------------------------------
# _split_identities_for_val
# ---------------------------------------------------------------------------


class TestSplitIdentities:
    def test_no_overlap(self, pair_spec_file):
        """Train and val index sets must be disjoint."""
        specs = _parse_pair_spec(pair_spec_file)
        rng = np.random.RandomState(42)
        train_idx, val_idx = _split_identities_for_val(specs, 0.3, rng)
        assert set(train_idx).isdisjoint(set(val_idx))

    def test_complete_coverage(self, pair_spec_file):
        """All pair indices must appear in either train or val."""
        specs = _parse_pair_spec(pair_spec_file)
        rng = np.random.RandomState(42)
        train_idx, val_idx = _split_identities_for_val(specs, 0.3, rng)
        assert sorted(train_idx + val_idx) == list(range(20))

    def test_determinism(self, pair_spec_file):
        """Same seed produces identical splits."""
        specs = _parse_pair_spec(pair_spec_file)
        rng1 = np.random.RandomState(42)
        train1, val1 = _split_identities_for_val(specs, 0.3, rng1)
        rng2 = np.random.RandomState(42)
        train2, val2 = _split_identities_for_val(specs, 0.3, rng2)
        assert train1 == train2
        assert val1 == val2

    def test_no_identity_leakage(self, pair_spec_file):
        """Val identities must not appear in any train pair.

        Different-person pairs cross identity sets, so train identities may
        appear in val (via cross-set pairs). But the reverse must not happen:
        identities assigned to val must be absent from all train pairs.
        """
        specs = _parse_pair_spec(pair_spec_file)
        rng = np.random.RandomState(42)
        train_idx, val_idx = _split_identities_for_val(specs, 0.3, rng)

        train_set = set(train_idx)

        # Identify which identities were assigned to val
        all_identities = sorted({i for s in specs for i in s["identities"]})
        rng2 = np.random.RandomState(42)
        rng2.shuffle(all_identities)
        n_val = max(1, int(len(all_identities) * 0.3))
        val_identities = set(all_identities[:n_val])

        # No train pair should involve a val identity
        for s in specs:
            if s["line_index"] in train_set:
                for ident in s["identities"]:
                    assert ident not in val_identities, (
                        f"Val identity {ident} found in train pair {s['line_index']}"
                    )


# ---------------------------------------------------------------------------
# _cap_pairs_per_identity
# ---------------------------------------------------------------------------


class TestCapPairsPerIdentity:
    def test_respects_cap(self, pair_spec_file):
        """No identity should have more than `cap` pairs in the output."""
        specs = _parse_pair_spec(pair_spec_file)
        all_indices = list(range(len(specs)))
        rng = np.random.RandomState(42)
        capped = _cap_pairs_per_identity(specs, all_indices, cap=2, rng=rng)

        # Count pairs per identity in the capped set
        from collections import Counter

        identity_counts = Counter()
        capped_set = set(capped)
        for s in specs:
            if s["line_index"] in capped_set:
                for ident in s["identities"]:
                    identity_counts[ident] += 1

        for ident, count in identity_counts.items():
            assert count <= 2, f"Identity {ident} has {count} pairs (cap=2)"

    def test_balanced_ratio(self, pair_spec_file):
        """Capped output must have exactly 1:1 positive:negative ratio."""
        specs = _parse_pair_spec(pair_spec_file)
        all_indices = list(range(len(specs)))
        rng = np.random.RandomState(42)
        capped = _cap_pairs_per_identity(specs, all_indices, cap=3, rng=rng)

        spec_by_idx = {s["line_index"]: s for s in specs}
        pos = sum(1 for i in capped if spec_by_idx[i]["label"] == 1)
        neg = sum(1 for i in capped if spec_by_idx[i]["label"] == 0)
        assert pos == neg, f"Not balanced: {pos} pos vs {neg} neg"

    def test_determinism(self, pair_spec_file):
        """Same seed produces identical capped output."""
        specs = _parse_pair_spec(pair_spec_file)
        all_indices = list(range(len(specs)))

        rng1 = np.random.RandomState(42)
        capped1 = _cap_pairs_per_identity(specs, all_indices, cap=2, rng=rng1)
        rng2 = np.random.RandomState(42)
        capped2 = _cap_pairs_per_identity(specs, all_indices, cap=2, rng=rng2)
        assert capped1 == capped2

    def test_high_cap_keeps_all_balanced(self, pair_spec_file):
        """A cap larger than any identity's pair count keeps everything balanced."""
        specs = _parse_pair_spec(pair_spec_file)
        all_indices = list(range(len(specs)))
        rng = np.random.RandomState(42)
        capped = _cap_pairs_per_identity(specs, all_indices, cap=100, rng=rng)

        spec_by_idx = {s["line_index"]: s for s in specs}
        pos = sum(1 for i in capped if spec_by_idx[i]["label"] == 1)
        neg = sum(1 for i in capped if spec_by_idx[i]["label"] == 0)
        assert pos == neg


# ---------------------------------------------------------------------------
# _save_pairs — output schema
# ---------------------------------------------------------------------------


class TestSavePairs:
    def test_output_schema(self, tmp_output_dir):
        """Saved .npz must contain img1, img2, label with correct shapes."""
        rng = np.random.RandomState(0)
        n = 10
        img1 = rng.rand(n, 62, 47).astype(np.float32)
        img2 = rng.rand(n, 62, 47).astype(np.float32)
        label = np.array([1] * 5 + [0] * 5, dtype=np.int64)

        _save_pairs("test_out", img1, img2, label, tmp_output_dir)

        path = os.path.join(tmp_output_dir, "pairs_test_out.npz")
        assert os.path.exists(path)

        data = np.load(path)
        assert set(data.files) == {"img1", "img2", "label"}
        assert data["img1"].shape == (n, 62, 47)
        assert data["img2"].shape == (n, 62, 47)
        assert data["label"].shape == (n,)
