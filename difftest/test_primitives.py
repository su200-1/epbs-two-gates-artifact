"""Differential / conformance tests for `epbs.primitives`.

Two layers (cf. MODEL_SCOPE.md — testing is per-function, scoped):

1. **Spec-conformance** (runs now, no build): the gloas spec algorithm is
   transcribed independently as an oracle and the port is checked against it,
   plus boundary and property checks.
2. **eth2spec differential** (skipped unless `make pyspec` has been run in the
   consensus-specs repo): cross-checks names and constants against the official
   executable spec; catches drift of `config` constants from the spec.

Run:  pytest difftest/
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from epbs import primitives as p
from epbs.primitives import PayloadStatus


# ===========================================================================
# Layer 1 — spec-conformance (no external build needed)
# ===========================================================================
def _make_store(verified: bool, n_timely: int, n_total: int, root: int = 1):
    store = p.Store()
    store.payload_timeliness_vote[root] = (
        [True] * n_timely + [False] * (n_total - n_timely))
    store.payload_data_availability_vote[root] = (
        [True] * n_timely + [False] * (n_total - n_timely))
    if verified:
        store.payloads[root] = p.ExecutionPayloadEnvelope(1, root, 100)
    return store


class _SpecBoolLike:
    """Boolean-like object that compares equal without being identical."""

    def __init__(self, value: bool):
        self.value = value

    def __eq__(self, other):
        return self.value == other


class TestPayloadTimeliness:
    def test_not_verified_returns_not_timely(self):
        # spec: if not is_payload_verified -> return `not timely`
        store = _make_store(verified=False, n_timely=512, n_total=512)
        assert p.payload_timeliness(store, 1, timely=True) is False
        assert p.payload_timeliness(store, 1, timely=False) is True

    def test_verified_above_threshold(self):
        thr = config.PAYLOAD_TIMELY_THRESHOLD
        store = _make_store(verified=True, n_timely=thr + 1, n_total=512)
        assert p.payload_timeliness(store, 1, timely=True) is True

    def test_verified_at_threshold_is_false(self):
        # spec uses strict `>` — exactly threshold votes is NOT timely
        thr = config.PAYLOAD_TIMELY_THRESHOLD
        store = _make_store(verified=True, n_timely=thr, n_total=512)
        assert p.payload_timeliness(store, 1, timely=True) is False

    def test_unknown_root_asserts(self):
        with pytest.raises(AssertionError):
            p.payload_timeliness(p.Store(), 999, timely=True)

    def test_votes_use_equality_not_identity(self):
        # consensus-specs #5331 changed `vote is timely` to `vote == timely`.
        thr = config.PAYLOAD_TIMELY_THRESHOLD
        store = _make_store(verified=True, n_timely=0, n_total=0)
        store.payload_timeliness_vote[1] = (
            [_SpecBoolLike(True)] * (thr + 1)
            + [_SpecBoolLike(False)] * (config.PTC_SIZE - thr - 1)
        )
        assert p.payload_timeliness(store, 1, timely=True) is True

    def test_absent_seats_count_for_neither_side(self):
        """Silence is not a vote.

        gloas initialises the tally as ``[None] * PTC_SIZE``; because the
        predicate counts ``vote == timely`` and ``None`` equals neither
        ``True`` nor ``False``, an absent seat must contribute to neither
        direction. This is what makes an orphan via ``should_build_on_full``
        require 257 *signed* not-present votes, while merely *denying* the
        ``timely=True`` affirmation needs only 256 seats to stay silent.

        Added after a mutation audit (exp101) found the suite did not detect a
        port that counted ``None`` as a vote.
        """
        thr = config.PAYLOAD_TIMELY_THRESHOLD
        store = _make_store(verified=True, n_timely=0, n_total=0)

        # All seats silent: neither direction holds.
        store.payload_timeliness_vote[1] = [None] * config.PTC_SIZE
        assert p.payload_timeliness(store, 1, timely=True) is False
        assert p.payload_timeliness(store, 1, timely=False) is False

        # 256 not-present + 256 silent: one short of the orphan threshold,
        # so `timely=False` must still be false (fail-open).
        store.payload_timeliness_vote[1] = (
            [False] * thr + [None] * (config.PTC_SIZE - thr)
        )
        assert p.payload_timeliness(store, 1, timely=False) is False
        # ...and the affirmation is already denied at that point.
        assert p.payload_timeliness(store, 1, timely=True) is False

        # 257 not-present votes: the orphan threshold is met.
        store.payload_timeliness_vote[1] = (
            [False] * (thr + 1) + [None] * (config.PTC_SIZE - thr - 1)
        )
        assert p.payload_timeliness(store, 1, timely=False) is True


class TestPayloadDataAvailability:
    def test_votes_use_equality_not_identity(self):
        # Mirrors gloas `payload_data_availability` after #5331.
        thr = config.DATA_AVAILABILITY_TIMELY_THRESHOLD
        store = _make_store(verified=True, n_timely=0, n_total=0)
        store.payload_data_availability_vote[1] = (
            [_SpecBoolLike(True)] * (thr + 1)
            + [_SpecBoolLike(False)] * (config.PTC_SIZE - thr - 1)
        )
        assert p.payload_data_availability(store, 1, available=True) is True


class TestParentPayloadStatus:
    def _chain(self, parent_block_hash: int, child_parent_hash: int):
        store = p.Store()
        parent = p.BeaconBlock(
            root=1, parent_root=0, slot=0,
            bid=p.ExecutionPayloadBid(0, 0, parent_block_hash, 1, 0, 1000, 50))
        child = p.BeaconBlock(
            root=2, parent_root=1, slot=1,
            bid=p.ExecutionPayloadBid(child_parent_hash, 1, 200, 2, 1, 1000, 50))
        store.blocks[1] = parent
        store.blocks[2] = child
        return store, child

    def test_full_when_hashes_match(self):
        store, child = self._chain(parent_block_hash=111, child_parent_hash=111)
        assert p.get_parent_payload_status(store, child) == PayloadStatus.FULL
        assert p.is_parent_node_full(store, child) is True

    def test_empty_when_hashes_differ(self):
        store, child = self._chain(parent_block_hash=111, child_parent_hash=999)
        assert p.get_parent_payload_status(store, child) == PayloadStatus.EMPTY
        assert p.is_parent_node_full(store, child) is False


class TestTiming:
    def test_slot_component_duration(self):
        # oracle: BPS fraction of one slot
        for bps in (0, 2_500, 5_000, 10_000):
            expected = config.MILLIS_PER_SLOT * bps // 10_000
            assert p.slot_component_duration_ms(bps) == expected

    def test_payload_due_before_ptc_due(self):
        # builder must reveal before the PTC attests
        assert p.get_payload_due_ms() <= p.get_payload_attestation_due_ms()
        assert p.get_payload_due_ms() <= config.MILLIS_PER_SLOT


class TestClassifyPayload:
    def test_withheld_is_empty(self):
        assert p.classify_payload(revealed=False, reveal_ms=None) == PayloadStatus.EMPTY

    def test_revealed_early_is_full(self):
        assert p.classify_payload(True, p.get_payload_due_ms() - 1) == PayloadStatus.FULL

    def test_revealed_late_is_pending(self):
        assert p.classify_payload(True, p.get_payload_due_ms() + 1) == PayloadStatus.PENDING


# ===========================================================================
# Layer 2 — pyspec differential (requires the gloas pyspec to be built)
#
# Build:  in consensus-specs, `python -m pysetup.generate_specs --all-forks`
#         -> tests/core/pyspec/eth_consensus_specs/gloas/{mainnet,minimal}.py
# Point EPBS_PYSPEC_DIR at that pyspec dir, or rely on the default below.
# ===========================================================================
import os

_DEFAULT_PYSPEC_DIR = (
    os.environ.get("EPBS_PYSPEC_DIR", "../consensus-specs/tests/core/pyspec")
)
SPEC_MODULE_CANDIDATES = [
    "eth_consensus_specs.gloas.mainnet",
    "eth_consensus_specs.gloas.minimal",
]


# `uint64` was renamed `Uint64` upstream (#5469) after the pinned `015d7270`
# snapshot; accept either spelling so the suite runs against both.
def _u64(spec, value):
    ctor = getattr(spec, "Uint64", None) or spec.uint64
    return ctor(value)


def _load_spec():
    pyspec_dir = os.environ.get("EPBS_PYSPEC_DIR", _DEFAULT_PYSPEC_DIR)
    if os.path.isdir(pyspec_dir) and pyspec_dir not in sys.path:
        sys.path.insert(0, pyspec_dir)
    for name in SPEC_MODULE_CANDIDATES:
        try:
            return __import__(name, fromlist=["*"])
        except ImportError:
            continue
    return None


@pytest.fixture(scope="module")
def spec():
    s = _load_spec()
    if s is None:
        pytest.skip("gloas pyspec not importable — build it with "
                    "`python -m pysetup.generate_specs --all-forks` in "
                    "consensus-specs, or set EPBS_PYSPEC_DIR")
    return s


class TestAgainstPyspec:
    """Differential checks vs the official executable spec (gloas).

    Catches drift between this project's `config` constants / ported logic and
    the pinned consensus-specs commit. (Built `eth_consensus_specs.gloas`.)
    """

    def test_spec_exposes_ported_functions(self, spec):
        for fn in ("payload_timeliness", "get_parent_payload_status",
                   "is_parent_node_full", "is_payload_verified"):
            assert hasattr(spec, fn), f"spec missing {fn}"

    def test_payload_due_bps_matches_spec(self, spec):
        assert config.PAYLOAD_DUE_BPS == int(spec.config.PAYLOAD_DUE_BPS)

    def test_payload_attestation_due_bps_matches_spec(self, spec):
        assert config.PAYLOAD_ATTESTATION_DUE_BPS == int(
            spec.config.PAYLOAD_ATTESTATION_DUE_BPS)

    def test_gloas_attestation_due_bps_matches_spec(self, spec):
        assert config.ATTESTATION_DUE_BPS_GLOAS == int(
            spec.config.ATTESTATION_DUE_BPS_GLOAS)

    def test_slot_duration_matches_spec(self, spec):
        assert config.MILLIS_PER_SLOT == int(spec.config.SLOT_DURATION_MS)

    def test_ptc_size_matches_spec(self, spec):
        assert config.PTC_SIZE == int(spec.PTC_SIZE)

    def test_payload_timely_threshold_matches_spec(self, spec):
        assert config.PAYLOAD_TIMELY_THRESHOLD == int(spec.PAYLOAD_TIMELY_THRESHOLD)

    def test_payload_due_ms_matches_spec(self, spec):
        # real differential test: same pure-arithmetic function, both sides
        assert p.get_payload_due_ms() == int(spec.get_payload_due_ms())

    def test_payload_attestation_due_ms_matches_spec(self, spec):
        assert p.get_payload_attestation_due_ms() == int(
            spec.get_payload_attestation_due_ms())

    def test_attestation_due_ms_matches_spec(self, spec):
        assert p.get_attestation_due_ms() == int(spec.get_attestation_due_ms())


# ===========================================================================
# Layer 2b — behavioural differential on the timeliness predicates themselves
# ===========================================================================
class TestTimelinessPredicatesAgainstPyspec:
    """Run `payload_timeliness` / `payload_data_availability` against pyspec.

    `TestAgainstPyspec` above pins the *constants* these predicates read.
    That is not enough: the paper's cost claims are exact boundary claims
    about the predicates themselves at the real committee size, so the port
    and the executable spec are fed vote vectors built from shared parameters
    and their outputs compared cell by cell — including the absent-seat
    (`None`) semantics that Path~B rests on.
    """

    @staticmethod
    def _R(spec, i: int):
        return spec.Root(i.to_bytes(32, "little"))

    @staticmethod
    def _votes(n_false: int, n_true: int, size: int) -> list:
        """not-present seats, then affirmative seats, then absent (`None`) ones."""
        assert n_false + n_true <= size
        return ([False] * n_false + [True] * n_true
                + [None] * (size - n_false - n_true))

    @classmethod
    def _stores(cls, spec, n_false: int, n_true: int, verified: bool = True):
        """Build a sim store and a pyspec store carrying the same vote vector."""
        size = int(spec.PTC_SIZE)
        votes = cls._votes(n_false, n_true, size)
        root, pyroot = 2, cls._R(spec, 2)

        sim = p.Store()
        sim.payload_timeliness_vote[root] = list(votes)
        sim.payload_data_availability_vote[root] = list(votes)
        if verified:
            sim.payloads[root] = p.ExecutionPayloadEnvelope(1, root, 100)

        py = spec.Store(
            time=_u64(spec, 0), genesis_time=_u64(spec, 0),
            justified_checkpoint=spec.Checkpoint(),
            finalized_checkpoint=spec.Checkpoint(),
            unrealized_justified_checkpoint=spec.Checkpoint(),
            unrealized_finalized_checkpoint=spec.Checkpoint(),
            proposer_boost_root=spec.Root(), equivocating_indices=set(),
            blocks={}, block_states={}, block_timeliness={},
            checkpoint_states={}, latest_messages={},
            unrealized_justifications={},
            payloads={pyroot: spec.ExecutionPayloadEnvelope()} if verified else {},
            payload_timeliness_vote={pyroot: list(votes)},
            payload_data_availability_vote={pyroot: list(votes)},
        )
        return sim, py, root, pyroot

    # (n_false, n_true) — boundary cells, Path-B silence cells, and extremes
    GRID = [
        (255, 257), (256, 256), (257, 255), (258, 254),   # Path-A boundary
        (0, 257), (0, 256), (0, 255),                     # Path-B silence
        (256, 0), (257, 0), (0, 0), (512, 0), (0, 512),   # degenerate/extreme
        (100, 300), (128, 128),
    ]

    @pytest.mark.parametrize("n_false,n_true", GRID)
    @pytest.mark.parametrize("timely", [True, False])
    def test_payload_timeliness_matches_pyspec(self, spec, n_false, n_true, timely):
        sim, py, root, pyroot = self._stores(spec, n_false, n_true)
        assert (p.payload_timeliness(sim, root, timely=timely)
                == bool(spec.payload_timeliness(py, pyroot, timely=timely)))

    @pytest.mark.parametrize("n_false,n_true", GRID)
    @pytest.mark.parametrize("available", [True, False])
    def test_payload_data_availability_matches_pyspec(
            self, spec, n_false, n_true, available):
        sim, py, root, pyroot = self._stores(spec, n_false, n_true)
        assert (p.payload_data_availability(sim, root, available=available)
                == bool(spec.payload_data_availability(py, pyroot, available=available)))

    @pytest.mark.parametrize("timely", [True, False])
    def test_unverified_payload_short_circuits_like_pyspec(self, spec, timely):
        # Property (iii): with no local payload the tally is not consulted.
        sim, py, root, pyroot = self._stores(spec, 512, 0, verified=False)
        assert (p.payload_timeliness(sim, root, timely=timely)
                == bool(spec.payload_timeliness(py, pyroot, timely=timely))
                == (not timely))

    def test_path_a_pivotal_count_matches_pyspec(self, spec):
        """Full sweep: the least number of *signed* not-present votes that makes
        `timely=False` hold is 257 in both implementations."""
        size = int(spec.PTC_SIZE)
        flips = []
        for n in range(size + 1):
            sim, py, root, pyroot = self._stores(spec, n, size - n)
            sim_out = p.payload_timeliness(sim, root, timely=False)
            py_out = bool(spec.payload_timeliness(py, pyroot, timely=False))
            assert sim_out == py_out, f"divergence at {n} not-present seats"
            if sim_out:
                flips.append(n)
        assert min(flips) == int(spec.PAYLOAD_TIMELY_THRESHOLD) + 1 == 257

    def test_path_b_pivotal_count_matches_pyspec(self, spec):
        """Full sweep: with no not-present vote cast, the least number of seats
        that must stay *silent* to deny the `timely=True` affirmation is 256."""
        size = int(spec.PTC_SIZE)
        denied = []
        for silent in range(size + 1):
            sim, py, root, pyroot = self._stores(spec, 0, size - silent)
            sim_out = p.payload_timeliness(sim, root, timely=True)
            py_out = bool(spec.payload_timeliness(py, pyroot, timely=True))
            assert sim_out == py_out, f"divergence at {silent} silent seats"
            if not sim_out:
                denied.append(silent)
        assert min(denied) == size - int(spec.PAYLOAD_TIMELY_THRESHOLD) == 256

    def test_boundary_split_fires_neither_side_in_pyspec(self, spec):
        """Non-complementarity: a 256/256 split satisfies neither query."""
        sim, py, root, pyroot = self._stores(spec, 256, 256)
        for fn, name in ((p.payload_timeliness, "payload_timeliness"),
                         (p.payload_data_availability, "payload_data_availability")):
            kw = "timely" if name == "payload_timeliness" else "available"
            for q in (True, False):
                assert fn(sim, root, **{kw: q}) is False
                assert bool(getattr(spec, name)(py, pyroot, **{kw: q})) is False
