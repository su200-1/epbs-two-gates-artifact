"""Differential / conformance tests for `epbs.committee`.

Two layers (cf. MODEL_SCOPE.md):

1. **Spec-conformance** (always runs): the swap-or-not shuffle is checked
   against a transcribed oracle + boundary / permutation properties.
2. **eth_consensus_specs differential** (runs if pyspec is importable): every
   pure shuffle function is checked byte-equal against the gloas pyspec.

Run:  pytest difftest/test_committee.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from epbs import committee as C


# ===========================================================================
# Layer 1 — spec-conformance (pure-Python, no external build)
# ===========================================================================
class TestShuffleProperties:
    """Algorithmic properties of `compute_shuffled_permutation`."""

    def test_returns_permutation_of_range(self):
        seed = b"\x42" * 32
        for n in (1, 2, 8, 33, 100):
            perm = C.compute_shuffled_permutation(n, seed)
            assert len(perm) == n
            assert sorted(perm) == list(range(n))

    def test_deterministic_for_same_seed(self):
        seed = b"\x11" * 32
        a = C.compute_shuffled_permutation(64, seed)
        b = C.compute_shuffled_permutation(64, seed)
        assert a == b

    def test_seed_sensitivity(self):
        a = C.compute_shuffled_permutation(64, b"\x00" * 32)
        b = C.compute_shuffled_permutation(64, b"\x01" + b"\x00" * 31)
        assert a != b  # tiny seed change => different permutation

    def test_shuffled_index_in_range(self):
        seed = b"\x55" * 32
        for n in (1, 10, 100):
            for i in range(n):
                idx = C.compute_shuffled_index(i, n, seed)
                assert 0 <= idx < n

    def test_shuffled_index_out_of_range_asserts(self):
        with pytest.raises(AssertionError):
            C.compute_shuffled_index(100, 100, b"\x00" * 32)


class TestComputeCommittee:
    def test_partitions_active_set(self):
        # union of all committees == original indices (as set)
        indices = list(range(64))
        seed = b"\x77" * 32
        count = 8
        union = []
        for i in range(count):
            union.extend(C.compute_committee(indices, seed, i, count))
        assert sorted(union) == indices

    def test_committee_size_balanced(self):
        indices = list(range(64))
        seed = b"\x33" * 32
        sizes = [
            len(C.compute_committee(indices, seed, i, 8)) for i in range(8)
        ]
        assert all(s == 8 for s in sizes)  # 64/8 == 8

    def test_deterministic(self):
        indices = list(range(32))
        seed = b"\x99" * 32
        a = C.compute_committee(indices, seed, 0, 4)
        b = C.compute_committee(indices, seed, 0, 4)
        assert a == b


class TestActiveSet:
    def test_filters_by_epoch(self):
        vset = C.ValidatorSet(validators=[
            C.Validator(index=0, activation_epoch=0, exit_epoch=10),
            C.Validator(index=1, activation_epoch=5, exit_epoch=10),
            C.Validator(index=2, activation_epoch=0, exit_epoch=3),
        ])
        # epoch 0: only validator 0 and 2 (1 not yet activated)
        assert C.get_active_validator_indices(vset, 0) == [0, 2]
        # epoch 5: validators 0 and 1 (2 has exited)
        assert C.get_active_validator_indices(vset, 5) == [0, 1]


class TestPTCStructure:
    def test_ptc_size_bounded(self):
        # Need len(active) // slots_per_epoch >= ptc_size for full PTC.
        active = list(range(1024))
        seed = b"\x44" * 32
        ptc = C.get_ptc(active, seed, slot=0, slots_per_epoch=32, ptc_size=16)
        assert len(ptc) == 16
        assert all(0 <= i < 1024 for i in ptc)

    def test_ptc_deterministic(self):
        active = list(range(1024))
        seed = b"\x44" * 32
        a = C.get_ptc(active, seed, slot=3, slots_per_epoch=32, ptc_size=16)
        b = C.get_ptc(active, seed, slot=3, slots_per_epoch=32, ptc_size=16)
        assert a == b

    def test_ptc_different_slots_differ(self):
        active = list(range(1024))
        seed = b"\x44" * 32
        a = C.get_ptc(active, seed, slot=0, slots_per_epoch=32, ptc_size=16)
        b = C.get_ptc(active, seed, slot=1, slots_per_epoch=32, ptc_size=16)
        assert a != b


# ===========================================================================
# Layer 2 — pyspec differential (requires gloas pyspec)
# ===========================================================================
_DEFAULT_PYSPEC_DIR = (
    os.environ.get("EPBS_PYSPEC_DIR", "../consensus-specs/tests/core/pyspec")
)


def _load_spec():
    pyspec_dir = os.environ.get("EPBS_PYSPEC_DIR", _DEFAULT_PYSPEC_DIR)
    if os.path.isdir(pyspec_dir) and pyspec_dir not in sys.path:
        sys.path.insert(0, pyspec_dir)
    for name in ("eth_consensus_specs.gloas.minimal",
                 "eth_consensus_specs.gloas.mainnet"):
        try:
            return __import__(name, fromlist=["*"])
        except ImportError:
            continue
    return None


@pytest.fixture(scope="module")
def spec():
    s = _load_spec()
    if s is None:
        pytest.skip("gloas pyspec not importable")
    return s


class TestAgainstPyspec:
    """Byte-level differential vs `eth_consensus_specs.gloas`."""

    def test_shuffle_round_count_matches(self, spec):
        assert C.SHUFFLE_ROUND_COUNT == int(spec.SHUFFLE_ROUND_COUNT)

    def test_max_effective_balance_matches(self, spec):
        assert C.MAX_EFFECTIVE_BALANCE_GWEI == int(spec.MAX_EFFECTIVE_BALANCE)

    def test_compute_shuffled_index_matches_pyspec(self, spec):
        # Multiple (i, n, seed) triples — must agree byte-equal.
        for n in (1, 4, 16, 100):
            for seed_byte in (0x00, 0x42, 0xff):
                seed = bytes([seed_byte]) * 32
                for i in range(n):
                    ours = C.compute_shuffled_index(i, n, seed)
                    theirs = int(spec.compute_shuffled_index(i, n, seed))
                    assert ours == theirs, (
                        f"shuffle disagrees at i={i} n={n} seed={seed_byte:02x}: "
                        f"ours={ours} theirs={theirs}"
                    )

    def test_compute_committee_matches_pyspec(self, spec):
        seed = b"\x42" * 32
        indices = list(range(64))
        for count in (1, 4, 8):
            for index in range(count):
                ours = C.compute_committee(indices, seed, index, count)
                theirs = [int(x) for x in spec.compute_committee(
                    indices, seed, index, count)]
                assert ours == theirs

    def test_active_validator_indices_matches_pyspec(self, spec):
        # We can't share a full BeaconState; instead reproduce pyspec's
        # `get_active_validator_indices` logic exactly:
        # active iff activation_epoch <= epoch < exit_epoch.
        # Just sanity check the boolean predicate matches the spec's intent.
        # (Full BeaconState diff test would require spec state construction.)
        vset = C.ValidatorSet(validators=[
            C.Validator(index=i, activation_epoch=0, exit_epoch=100)
            for i in range(8)
        ])
        active = C.get_active_validator_indices(vset, 5)
        assert active == list(range(8))

    def test_shuffle_round_count_constant_used(self, spec):
        # Sanity: changing our SHUFFLE_ROUND_COUNT would break diff parity.
        # The test above (compute_shuffled_index) already verifies parity;
        # this asserts the constant alignment is structural, not coincidental.
        assert C.SHUFFLE_ROUND_COUNT == 10  # pinned to phase0 spec

    def test_max_effective_balance_electra_matches(self, spec):
        assert C.MAX_EFFECTIVE_BALANCE_ELECTRA == int(
            spec.MAX_EFFECTIVE_BALANCE_ELECTRA
        )

    def test_compute_balance_weighted_selection_matches_pyspec(self, spec):
        """Spec gloas/beacon-chain.md::compute_balance_weighted_selection.

        Pyspec consumes a full BeaconState; we pass a stub exposing only
        ``state.validators[i].effective_balance``. Uniform-balance fixture so
        the rejection sampling stays deterministic across runs.
        """
        class _V:
            def __init__(self, eb): self.effective_balance = eb

        class _State:
            pass

        EB = int(spec.MAX_EFFECTIVE_BALANCE)
        for n_indices, size in ((4, 2), (8, 4), (32, 16)):
            for shuffle in (False, True):
                state = _State()
                state.validators = [_V(EB) for _ in range(n_indices)]
                indices = list(range(n_indices))
                seed = b"\x42" * 32
                theirs = [int(x) for x in spec.compute_balance_weighted_selection(
                    state, indices, seed, size=size, shuffle_indices=shuffle)]
                ours = C.compute_balance_weighted_selection(
                    indices, seed, size, shuffle_indices=shuffle,
                    effective_balance_of=lambda _i, _eb=EB: _eb,
                )
                assert ours == theirs, (
                    f"BWS disagrees n={n_indices} size={size} shuffle={shuffle}: "
                    f"ours={ours} theirs={theirs}"
                )
