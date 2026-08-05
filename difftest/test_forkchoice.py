"""Differential / conformance tests for `epbs.forkchoice`.

This is the largest diff-test module — exercises each ported function plus a
small end-to-end trace under the honest-only assumption.

Run:  pytest difftest/test_forkchoice.py
"""
from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from epbs import attestation as A
from epbs import forkchoice as F
from epbs import primitives as p
from epbs.primitives import PayloadStatus


# ===========================================================================
# Helpers
# ===========================================================================
def _bid(parent_block_hash: int, block_hash: int, slot: int) -> p.ExecutionPayloadBid:
    return p.ExecutionPayloadBid(
        parent_block_hash=parent_block_hash,
        parent_block_root=0,
        block_hash=block_hash,
        builder_index=0,
        slot=slot,
        value=1000,
        execution_payment=50,
    )


def _block(root: int, parent_root: int, slot: int, *, parent_full: bool,
           parent_block_hash: int | None = None, proposer_index: int = 0) -> p.BeaconBlock:
    """Build a BeaconBlock whose bid commits to FULL or EMPTY parent.

    For the FULL case we expect the parent's bid.block_hash to be passed in
    via ``parent_block_hash``; for EMPTY any non-matching value works.
    """
    if parent_full:
        bid = _bid(parent_block_hash or root * 100, block_hash=root * 1000, slot=slot)
    else:
        bid = _bid(parent_block_hash=999_999, block_hash=root * 1000, slot=slot)
    blk = p.BeaconBlock(root=root, parent_root=parent_root, slot=slot, bid=bid)
    setattr(blk, "proposer_index", proposer_index)  # for is_parent_strong / boost
    return blk


def _make_store(
    n_validators: int = 8, total_active_balance: int | None = None,
) -> F.Store:
    eff = {i: 32 * 10**9 for i in range(n_validators)}
    tab = total_active_balance or sum(eff.values())
    anchor = p.BeaconBlock(
        root=1, parent_root=0, slot=0,
        bid=_bid(parent_block_hash=0, block_hash=100, slot=0))
    setattr(anchor, "proposer_index", 0)
    store = F.get_forkchoice_store(
        anchor, effective_balances=eff, total_active_balance=tab)
    return store


# ===========================================================================
# Layer 1 — spec-conformance unit tests
# ===========================================================================
class TestStoreConstruction:
    def test_get_forkchoice_store_initialises_fields(self):
        store = _make_store()
        assert store.blocks[1].slot == 0
        assert store.justified_root == 1
        assert store.finalized_root == 1
        assert store.block_timeliness[1] == [True, True]
        assert len(store.payload_timeliness_vote[1]) == config.PTC_SIZE


class TestGetAncestor:
    def test_same_block_returns_pending(self):
        store = _make_store()
        node = F.get_ancestor(store, root=1, slot=0)
        assert node == F.ForkChoiceNode(1, PayloadStatus.PENDING)

    def test_walks_parent_chain(self):
        store = _make_store()
        # build chain 1 -> 2 -> 3
        b2 = _block(root=2, parent_root=1, slot=1, parent_full=True,
                    parent_block_hash=100)
        b3 = _block(root=3, parent_root=2, slot=2, parent_full=True,
                    parent_block_hash=2000)
        store.blocks[2] = b2
        store.blocks[3] = b3
        node = F.get_ancestor(store, root=3, slot=1)
        assert node.root == 2
        assert node.payload_status == PayloadStatus.FULL  # b3 claims FULL b2


class TestSupportPredicate:
    """Node-centric support predicate (gloas #5249 fork-choice redesign).

    A message supports a node iff
    ``is_ancestor(store, get_supported_node(store, msg), node)`` — the exact
    dual of the removed vote-centric ``is_supporting_vote``. These are the
    former ``is_supporting_vote`` cases, re-expressed against the redesigned
    abstraction; they also pin that equivalence.
    """

    @staticmethod
    def _supports(store, node, msg):
        return F.is_ancestor(store, F.get_supported_node(store, msg), node)

    def test_pending_same_root_always_supported(self):
        store = _make_store()
        node = F.ForkChoiceNode(1, PayloadStatus.PENDING)
        msg = A.LatestMessage(slot=5, root=1, payload_present=False)
        assert self._supports(store, node, msg) is True

    def test_full_same_root_requires_payload_present(self):
        store = _make_store()
        # build a 2nd block to host later messages
        b2 = _block(root=2, parent_root=1, slot=1, parent_full=False)
        store.blocks[2] = b2
        node = F.ForkChoiceNode(2, PayloadStatus.FULL)
        # message at later slot voting FULL — supports; get_supported_node
        # resolves the payload to FULL.
        msg_full = A.LatestMessage(slot=2, root=2, payload_present=True)
        assert F.get_supported_node(store, msg_full) == F.ForkChoiceNode(
            2, PayloadStatus.FULL)
        assert self._supports(store, node, msg_full) is True
        # message voting EMPTY — does not support FULL node
        msg_empty = A.LatestMessage(slot=2, root=2, payload_present=False)
        assert self._supports(store, node, msg_empty) is False

    def test_descendant_pending_supported(self):
        store = _make_store()
        # anchor (slot 0) <- b2 (slot 1) <- b3 (slot 2)
        store.blocks[2] = _block(2, 1, 1, parent_full=True, parent_block_hash=100)
        store.blocks[3] = _block(3, 2, 2, parent_full=True, parent_block_hash=2000)
        # vote on b3 should support the PENDING b2 ancestor
        node = F.ForkChoiceNode(2, PayloadStatus.PENDING)
        msg = A.LatestMessage(slot=3, root=3, payload_present=True)
        assert self._supports(store, node, msg) is True

    def test_same_slot_message_leaves_payload_pending(self):
        store = _make_store()
        store.blocks[2] = _block(2, 1, 1, parent_full=False)
        node = F.ForkChoiceNode(2, PayloadStatus.FULL)
        msg = A.LatestMessage(slot=1, root=2, payload_present=True)  # same slot
        # a same-slot message supports only the PENDING node, not FULL/EMPTY
        assert F.get_supported_node(store, msg) == F.ForkChoiceNode(
            2, PayloadStatus.PENDING)
        assert self._supports(store, node, msg) is False


class TestGetNodeChildren:
    def test_pending_node_splits_into_empty_and_full_if_verified(self):
        store = _make_store()
        # add an envelope so FULL becomes available
        store.payloads[1] = p.ExecutionPayloadEnvelope(0, 1, 100)
        node = F.ForkChoiceNode(1, PayloadStatus.PENDING)
        children = F.get_node_children(store, store.blocks, node)
        statuses = {c.payload_status for c in children}
        assert statuses == {PayloadStatus.EMPTY, PayloadStatus.FULL}

    def test_pending_without_payload_only_empty(self):
        store = _make_store()
        node = F.ForkChoiceNode(1, PayloadStatus.PENDING)
        children = F.get_node_children(store, store.blocks, node)
        assert children == [F.ForkChoiceNode(1, PayloadStatus.EMPTY)]

    def test_status_node_descends_to_matching_children(self):
        store = _make_store()
        # b2 claims FULL parent, b3 claims EMPTY parent
        store.blocks[2] = _block(2, 1, 1, parent_full=True, parent_block_hash=100)
        store.blocks[3] = _block(3, 1, 1, parent_full=False)
        full_node = F.ForkChoiceNode(1, PayloadStatus.FULL)
        empty_node = F.ForkChoiceNode(1, PayloadStatus.EMPTY)
        full_children = F.get_node_children(store, store.blocks, full_node)
        empty_children = F.get_node_children(store, store.blocks, empty_node)
        assert {c.root for c in full_children} == {2}
        assert {c.root for c in empty_children} == {3}


class TestGetAttestationScore:
    def test_sums_supporting_effective_balances(self):
        store = _make_store(n_validators=4)
        # Add b2 at slot 1 so we have a non-anchor block to vote on
        store.blocks[2] = _block(2, 1, 1, parent_full=True, parent_block_hash=100)
        # All 4 validators vote PENDING for b2 (LMD vote)
        for i in range(4):
            store.latest_messages[i] = A.LatestMessage(
                slot=2, root=2, payload_present=False)
        node = F.ForkChoiceNode(2, PayloadStatus.PENDING)
        assert F.get_attestation_score(store, node) == 4 * 32 * 10**9

    def test_excludes_equivocators(self):
        store = _make_store(n_validators=4)
        store.blocks[2] = _block(2, 1, 1, parent_full=True, parent_block_hash=100)
        for i in range(4):
            store.latest_messages[i] = A.LatestMessage(2, 2, False)
        store.equivocating_indices = {1, 3}
        node = F.ForkChoiceNode(2, PayloadStatus.PENDING)
        assert F.get_attestation_score(store, node) == 2 * 32 * 10**9

    def test_incremental_index_matches_naive_across_chain(self):
        """Equivalence: incremental `_score_index` agrees with the naive
        per-validator loop for every (root, status) reachable after a
        non-trivial chain of `on_attestation` calls.
        """
        from epbs.primitives import PayloadStatus as PS
        store = _make_store(n_validators=8)
        # Build a 4-block linear chain on top of the anchor.
        for slot in range(1, 5):
            store.blocks[slot + 1] = _block(
                slot + 1, slot, slot, parent_full=True,
                parent_block_hash=(slot * 100) if slot > 1 else 100,
            )
            store.payloads[slot + 1] = p.ExecutionPayloadEnvelope(
                0, slot + 1, (slot + 1) * 100
            )
        # Drive votes through the public API so the index is populated.
        for vi in range(8):
            msg = A.LatestMessage(slot=4, root=5, payload_present=True)
            store.latest_messages[vi] = msg
            F._update_score_index(store, vi, msg)
        # The two implementations must agree across every visited node.
        for root in store.blocks:
            for status in (PS.PENDING, PS.FULL, PS.EMPTY):
                node = F.ForkChoiceNode(root, status)
                fast = F.get_attestation_score(store, node)
                naive = F._get_attestation_score_naive(store, node)
                assert fast == naive, (root, status, fast, naive)


class TestGetWeightAndHead:
    def test_get_head_returns_anchor_when_no_attestations(self):
        store = _make_store()
        head = F.get_head(store)
        # anchor has no children -> head is PENDING anchor
        # ... but get_node_children of PENDING anchor yields EMPTY anchor
        # which has no children -> head = EMPTY anchor
        assert head.root == 1

    def test_payload_status_splits_weight(self):
        """Core ePBS claim: a block with revealed payload + FULL votes
        accumulates weight on its FULL node, not its EMPTY node.

        Spec: FULL/EMPTY weight is 0 in the slot immediately after the block
        (tiebreaker territory); we advance to current_slot=3 for the slot-1
        block so attestation weight applies.
        """
        store = _make_store(n_validators=4)
        store.time_ms = 3 * config.MILLIS_PER_SLOT  # current slot = 3
        store.blocks[2] = _block(2, 1, 1, parent_full=True, parent_block_hash=100)
        store.payloads[2] = p.ExecutionPayloadEnvelope(0, 2, 2000)
        for i in range(4):
            store.latest_messages[i] = A.LatestMessage(2, 2, payload_present=True)
        full_w = F.get_weight(store, F.ForkChoiceNode(2, PayloadStatus.FULL))
        empty_w = F.get_weight(store, F.ForkChoiceNode(2, PayloadStatus.EMPTY))
        assert full_w == 4 * 32 * 10**9
        assert empty_w == 0

    def test_withheld_payload_pushes_weight_to_empty(self):
        """The H-5 attack lever: builder withholds => no FULL votes => EMPTY
        accumulates the committee weight."""
        store = _make_store(n_validators=4)
        store.time_ms = 3 * config.MILLIS_PER_SLOT
        store.blocks[2] = _block(2, 1, 1, parent_full=False)  # parent EMPTY
        for i in range(4):
            store.latest_messages[i] = A.LatestMessage(2, 2, payload_present=False)
        full_w = F.get_weight(store, F.ForkChoiceNode(2, PayloadStatus.FULL))
        empty_w = F.get_weight(store, F.ForkChoiceNode(2, PayloadStatus.EMPTY))
        assert empty_w == 4 * 32 * 10**9
        assert full_w == 0


class TestRecordBlockTimeliness:
    def test_timely_block_marked_both_indices(self):
        store = _make_store()
        # block at slot 1
        store.blocks[2] = _block(2, 1, 1, parent_full=True, parent_block_hash=100)
        store.time_ms = 1 * config.MILLIS_PER_SLOT + 100  # 100 ms into slot 1
        F.record_block_timeliness(store, 2)
        assert store.block_timeliness[2] == [True, True]

    def test_late_block_marked_not_timely(self):
        store = _make_store()
        store.blocks[2] = _block(2, 1, 1, parent_full=True, parent_block_hash=100)
        # very late inside the slot — past both deadlines
        store.time_ms = 1 * config.MILLIS_PER_SLOT + config.MILLIS_PER_SLOT - 1
        F.record_block_timeliness(store, 2)
        assert store.block_timeliness[2] == [False, False]

    def test_between_attestation_and_ptc_deadlines_marks_only_ptc_timely(self):
        store = _make_store()
        store.blocks[2] = _block(2, 1, 1, parent_full=True, parent_block_hash=100)
        store.time_ms = 1 * config.MILLIS_PER_SLOT + 4_000
        F.record_block_timeliness(store, 2)
        assert store.block_timeliness[2] == [False, True]


class TestShouldBuildOnFull:
    def _store_with_previous_slot_full(self):
        store = _make_store()
        store.blocks[2] = _block(2, 1, 1, parent_full=True, parent_block_hash=100)
        store.payloads[2] = p.ExecutionPayloadEnvelope(0, 2, 2000)
        store.payload_timeliness_vote[2] = [True] * config.PTC_SIZE
        store.payload_data_availability_vote[2] = [True] * config.PTC_SIZE
        store.time_ms = 2 * config.MILLIS_PER_SLOT
        return store

    def test_extends_previous_slot_full_when_ptc_is_positive(self):
        store = self._store_with_previous_slot_full()
        assert F.should_build_on_full(
            store, F.ForkChoiceNode(2, PayloadStatus.FULL),
        )

    def test_reorgs_late_previous_slot_payload(self):
        store = self._store_with_previous_slot_full()
        store.payload_timeliness_vote[2] = [False] * config.PTC_SIZE
        assert not F.should_build_on_full(
            store, F.ForkChoiceNode(2, PayloadStatus.FULL),
        )


class TestOnBlock:
    def test_adds_block_and_records_timeliness(self):
        store = _make_store()
        store.time_ms = 1 * config.MILLIS_PER_SLOT  # current slot = 1
        b2 = _block(2, 1, 1, parent_full=False)
        F.on_block(store, b2)
        assert 2 in store.blocks
        assert 2 in store.payload_timeliness_vote
        assert len(store.payload_timeliness_vote[2]) == config.PTC_SIZE

    def test_rejects_full_parent_without_verified_payload(self):
        store = _make_store()
        store.time_ms = 1 * config.MILLIS_PER_SLOT
        # b2 claims FULL anchor but anchor payload not in store.payloads
        b2 = _block(2, 1, 1, parent_full=True, parent_block_hash=100)
        with pytest.raises(AssertionError):
            F.on_block(store, b2)

    def test_accepts_full_parent_with_verified_payload(self):
        store = _make_store()
        store.payloads[1] = p.ExecutionPayloadEnvelope(0, 1, 100)
        store.time_ms = 1 * config.MILLIS_PER_SLOT
        b2 = _block(2, 1, 1, parent_full=True, parent_block_hash=100)
        F.on_block(store, b2)
        assert 2 in store.blocks


class TestOnExecutionPayloadEnvelope:
    def test_stores_envelope(self):
        store = _make_store()
        env = p.ExecutionPayloadEnvelope(0, 1, 100)
        F.on_execution_payload_envelope(store, env)
        assert store.payloads[1] == env


class TestOnPayloadAttestationMessage:
    def test_writes_ptc_vote(self):
        store = _make_store()
        # add b2 at slot 1
        store.blocks[2] = _block(2, 1, 1, parent_full=False)
        store.payload_timeliness_vote[2] = [None] * config.PTC_SIZE
        store.payload_data_availability_vote[2] = [None] * config.PTC_SIZE
        ptc = [42, 99]  # 42 occupies PTC index 0
        msg = A.PayloadAttestationMessage(
            validator_index=42,
            data=A.PayloadAttestationData(2, 1, True, True),
        )
        F.on_payload_attestation_message(store, msg, ptc)
        assert store.payload_timeliness_vote[2][0] is True
        assert store.payload_data_availability_vote[2][0] is True

    def test_ignores_message_for_wrong_slot(self):
        store = _make_store()
        store.blocks[2] = _block(2, 1, 1, parent_full=False)
        store.payload_timeliness_vote[2] = [None] * config.PTC_SIZE
        store.payload_data_availability_vote[2] = [None] * config.PTC_SIZE
        ptc = [42]
        msg = A.PayloadAttestationMessage(
            validator_index=42,
            data=A.PayloadAttestationData(2, slot=99, payload_present=True,
                                          blob_data_available=True),
        )
        # Spec returns early; we mirror that.
        F.on_payload_attestation_message(store, msg, ptc)
        assert store.payload_timeliness_vote[2][0] is None


# ===========================================================================
# Layer 2 — pyspec differential (constants + structural agreement)
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


# --- upstream-rename shims -------------------------------------------------
# The artifact pins consensus-specs `015d7270`. Later commits renamed the SSZ
# scalar types (`uint*` -> `Uint*`, #5469; `boolean` -> `Boolean`, #5466) and
# gave `should_build_on_full` an explicit `slot` parameter (#5497) in place of
# its internal `get_current_slot(store)` call. None of these change behaviour,
# so the tests accept either spelling rather than pinning the suite to one
# snapshot's surface.
def _u64(spec, value):
    ctor = getattr(spec, "Uint64", None) or spec.uint64
    return ctor(value)


def _should_build_on_full(spec, store, head, slot):
    fn = spec.should_build_on_full
    if len(inspect.signature(fn).parameters) >= 3:
        return fn(store, head, spec.Slot(slot))
    return fn(store, head)


def _load_spec_mainnet():
    """Load the mainnet preset specifically (PTC_SIZE=512).

    `_load_spec` prefers the minimal preset, which is fine for the
    preset-independent constants below but not for the seat-count boundary:
    minimal sets PTC_SIZE=16, and the port's `config` is pinned to mainnet.
    """
    pyspec_dir = os.environ.get("EPBS_PYSPEC_DIR", _DEFAULT_PYSPEC_DIR)
    if os.path.isdir(pyspec_dir) and pyspec_dir not in sys.path:
        sys.path.insert(0, pyspec_dir)
    try:
        return __import__("eth_consensus_specs.gloas.mainnet", fromlist=["*"])
    except ImportError:
        return None


@pytest.fixture(scope="module")
def spec():
    s = _load_spec()
    if s is None:
        pytest.skip("gloas pyspec not importable")
    return s


class TestAgainstPyspec:
    def test_atestation_timeliness_index_matches(self, spec):
        assert F.ATTESTATION_TIMELINESS_INDEX == int(spec.ATTESTATION_TIMELINESS_INDEX)

    def test_ptc_timeliness_index_matches(self, spec):
        assert F.PTC_TIMELINESS_INDEX == int(spec.PTC_TIMELINESS_INDEX)

    def test_num_block_timeliness_deadlines_matches(self, spec):
        assert F.NUM_BLOCK_TIMELINESS_DEADLINES == int(
            spec.NUM_BLOCK_TIMELINESS_DEADLINES)

    def test_payload_status_values_match(self, spec):
        assert int(PayloadStatus.EMPTY) == int(spec.PAYLOAD_STATUS_EMPTY)
        assert int(PayloadStatus.FULL) == int(spec.PAYLOAD_STATUS_FULL)
        assert int(PayloadStatus.PENDING) == int(spec.PAYLOAD_STATUS_PENDING)

    def test_reorg_thresholds_match(self, spec):
        assert config.REORG_HEAD_WEIGHT_THRESHOLD == int(
            spec.config.REORG_HEAD_WEIGHT_THRESHOLD)
        assert config.REORG_PARENT_WEIGHT_THRESHOLD == int(
            spec.config.REORG_PARENT_WEIGHT_THRESHOLD)
        assert config.PROPOSER_SCORE_BOOST == int(spec.config.PROPOSER_SCORE_BOOST)

    def test_spec_exposes_ported_handlers(self, spec):
        # Surface tracks the ec1c01f gloas fork-choice (#5249 node-centric
        # redesign): get_supported_node + is_ancestor + the previous-slot
        # payload predicate replace the removed vote-centric is_supporting_vote.
        for fn in (
            "on_block", "on_attestation", "on_execution_payload_envelope",
            "on_payload_attestation_message", "get_head", "get_weight",
            "get_supported_node", "is_ancestor",
            "is_previous_slot_payload_decision", "get_node_children",
            "get_attestation_score", "get_payload_status_tiebreaker",
            "should_apply_proposer_boost", "update_proposer_boost_root",
            "record_block_timeliness", "get_ancestor",
        ):
            assert hasattr(spec, fn), f"spec missing {fn}"


class TestNodeHelpersAgainstPyspec:
    """Behavioral differential test for the #5249/#5317 node-centric helpers.

    Builds matching sim and executable-pyspec stores from shared parameters and
    asserts `get_supported_node` + `is_previous_slot_payload_decision` agree
    across a grid. This is fork-choice *behavioral* conformance coverage that
    was previously absent — the prior TestAgainstPyspec only checked constants
    and function existence.
    """

    @staticmethod
    def _R(spec, i: int):
        return spec.Root(i.to_bytes(32, "little"))

    @classmethod
    def _mk_pyspec_store(cls, spec, blocks_slots: dict, current_slot: int):
        slot_ms = int(spec.config.SLOT_DURATION_MS)
        blocks = {cls._R(spec, r): spec.BeaconBlock(slot=spec.Slot(sl))
                  for r, sl in blocks_slots.items()}
        return spec.Store(
            time=_u64(spec, current_slot * slot_ms // 1000),
            genesis_time=_u64(spec, 0),
            justified_checkpoint=spec.Checkpoint(),
            finalized_checkpoint=spec.Checkpoint(),
            unrealized_justified_checkpoint=spec.Checkpoint(),
            unrealized_finalized_checkpoint=spec.Checkpoint(),
            proposer_boost_root=spec.Root(), equivocating_indices=set(),
            blocks=blocks, block_states={}, block_timeliness={},
            checkpoint_states={}, latest_messages={},
            unrealized_justifications={}, payloads={},
            payload_timeliness_vote={}, payload_data_availability_vote={},
        )

    @pytest.mark.parametrize("block_slot,msg_slot,payload_present", [
        (1, 2, True), (1, 2, False), (1, 1, True), (1, 1, False),
        (3, 5, True), (3, 5, False), (4, 4, False),
    ])
    def test_get_supported_node_matches_pyspec(
            self, spec, block_slot, msg_slot, payload_present):
        sim_store = _make_store()
        sim_store.blocks[2] = _block(2, 1, block_slot, parent_full=False)
        sim_node = F.get_supported_node(
            sim_store,
            A.LatestMessage(slot=msg_slot, root=2, payload_present=payload_present),
        )
        pst = self._mk_pyspec_store(spec, {2: block_slot}, current_slot=msg_slot)
        py_node = spec.get_supported_node(
            pst,
            spec.LatestMessage(slot=spec.Slot(msg_slot), root=self._R(spec, 2),
                               payload_present=payload_present),
        )
        assert int(sim_node.payload_status) == int(py_node.payload_status)
        assert sim_node.root == 2 and py_node.root == self._R(spec, 2)

    @pytest.mark.parametrize("block_slot,current_slot,status", [
        (1, 2, PayloadStatus.PENDING), (1, 2, PayloadStatus.EMPTY),
        (1, 2, PayloadStatus.FULL), (1, 3, PayloadStatus.FULL),
        (2, 3, PayloadStatus.EMPTY), (5, 6, PayloadStatus.FULL),
        (5, 6, PayloadStatus.PENDING),
    ])
    def test_is_previous_slot_payload_decision_matches_pyspec(
            self, spec, block_slot, current_slot, status):
        sim_store = _make_store()
        sim_store.blocks[2] = _block(2, 1, block_slot, parent_full=False)
        sim_store.time_ms = current_slot * config.MILLIS_PER_SLOT
        sim_res = F.is_previous_slot_payload_decision(
            sim_store, F.ForkChoiceNode(2, status))
        pst = self._mk_pyspec_store(spec, {2: block_slot}, current_slot=current_slot)
        py_res = spec.is_previous_slot_payload_decision(
            pst, spec.ForkChoiceNode(root=self._R(spec, 2),
                                     payload_status=int(status)))
        assert bool(sim_res) == bool(py_res)


# ===========================================================================
# Layer 3 — end-to-end honest trace (3-slot chain, all-honest, no withhold)
# ===========================================================================
class TestHonestTraceEndToEnd:
    """Build a 3-slot honest chain and check structural invariants.

    Full byte-equality with pyspec's Store would require constructing real
    BeaconBlock SSZ containers + state_transition; instead we check the
    end-state predicates that any spec-faithful implementation must satisfy.
    """

    def test_three_slot_honest_chain_head_advances(self):
        store = _make_store(n_validators=8)
        # tick to slot 1
        F.on_tick(store, config.MILLIS_PER_SLOT)
        # propose b2 at slot 1, with FULL anchor parent — reveal anchor payload
        store.payloads[1] = p.ExecutionPayloadEnvelope(0, 1, 100)
        b2 = _block(2, 1, 1, parent_full=True, parent_block_hash=100)
        F.on_block(store, b2)
        # reveal b2 payload
        F.on_execution_payload_envelope(
            store, p.ExecutionPayloadEnvelope(0, 2, 2000))
        # advance to slot 2
        F.on_tick(store, 2 * config.MILLIS_PER_SLOT)
        # 8 validators vote FULL on b2 (committee size 8, all in)
        att = A.Attestation(
            data=A.AttestationData(
                slot=2, index=1, beacon_block_root=2,
                source=A.Checkpoint(0, 1),
                target=A.Checkpoint(A.compute_epoch_at_slot(2), 2),
            ),
            aggregation_bits=tuple([True] * 8),
            committee_indices=tuple(range(8)),
        )
        # need to advance to slot 3 to be 1 slot past att.slot
        F.on_tick(store, 3 * config.MILLIS_PER_SLOT)
        F.on_attestation(store, att)
        # head should now be on b2 FULL
        head = F.get_head(store)
        assert head.root == 2
        assert head.payload_status == PayloadStatus.FULL

    def test_three_slot_withheld_head_goes_empty(self):
        """Same as above but b2's payload is never revealed -> head EMPTY."""
        store = _make_store(n_validators=8)
        F.on_tick(store, config.MILLIS_PER_SLOT)
        # b2 with EMPTY parent (anchor payload not revealed)
        b2 = _block(2, 1, 1, parent_full=False)
        F.on_block(store, b2)
        # b2 payload also withheld (not in store.payloads)
        F.on_tick(store, 2 * config.MILLIS_PER_SLOT)
        att = A.Attestation(
            data=A.AttestationData(
                slot=2, index=0, beacon_block_root=2,  # EMPTY vote
                source=A.Checkpoint(0, 1),
                target=A.Checkpoint(A.compute_epoch_at_slot(2), 2),
            ),
            aggregation_bits=tuple([True] * 8),
            committee_indices=tuple(range(8)),
        )
        F.on_tick(store, 3 * config.MILLIS_PER_SLOT)
        F.on_attestation(store, att)
        head = F.get_head(store)
        assert head.root == 2
        assert head.payload_status == PayloadStatus.EMPTY


# ===========================================================================
# Behavioural differential on the two consumers of the committee's verdict
# ===========================================================================
class TestPayloadGateConsumersAgainstPyspec:
    """Run `should_build_on_full` / `should_extend_payload` / the payload-status
    tiebreaker against the executable pyspec.

    These are the two functions that turn a PTC tally into an orphan, by two
    different routes: the next proposer's build rule and fork choice's
    payload tiebreaker. `TestAgainstPyspec` only checks that the spec *exposes*
    them, so the port is additionally executed side by side with the spec over
    a grid built from shared parameters. The grid spans the 257-seat build-rule
    boundary, the 256-seat withheld-affirmation boundary, and — because the
    port orders the EMPTY and slot checks the other way round (a difference the
    paper calls cosmetic) — both slot relations crossed with both head states.
    """

    TARGET, PROPOSER, ANCHOR = 2, 3, 1
    TARGET_HASH, OTHER_HASH = 777, 999

    @pytest.fixture(scope="class")
    def spec(self):
        """Shadow the module fixture with the mainnet preset.

        257 and 256 are `PTC_SIZE`-relative claims and the port's `config` is
        pinned to mainnet, so this class must not run against the minimal
        preset's 16-seat committee.
        """
        s = _load_spec_mainnet()
        if s is None:
            pytest.skip("gloas mainnet pyspec not importable")
        assert int(s.PTC_SIZE) == config.PTC_SIZE == 512
        return s

    @staticmethod
    def _R(spec, i: int):
        return spec.Root(i.to_bytes(32, "little"))

    @staticmethod
    def _H(spec, i: int):
        return spec.Hash32(i.to_bytes(32, "little"))

    @staticmethod
    def _votes(n_false: int, n_true: int, size: int) -> list:
        assert n_false + n_true <= size
        return ([False] * n_false + [True] * n_true
                + [None] * (size - n_false - n_true))

    @classmethod
    def _sim_store(cls, spec, *, target_slot, current_slot, n_false, n_true,
                   verified=True, boost=None, boost_parent=None,
                   boost_parent_full=True):
        """Sim-side store: target block at ``target_slot`` plus an optional
        boosted block whose bid commits to a FULL or EMPTY parent."""
        size = int(spec.PTC_SIZE)
        votes = cls._votes(n_false, n_true, size)
        store = F.Store(time_ms=current_slot * config.MILLIS_PER_SLOT)
        store.blocks[cls.ANCHOR] = _block(cls.ANCHOR, 0, 0, parent_full=False)
        store.blocks[cls.TARGET] = p.BeaconBlock(
            root=cls.TARGET, parent_root=cls.ANCHOR, slot=target_slot,
            bid=_bid(parent_block_hash=0, block_hash=cls.TARGET_HASH,
                     slot=target_slot))
        if verified:
            store.payloads[cls.TARGET] = p.ExecutionPayloadEnvelope(
                1, cls.TARGET, cls.TARGET_HASH)
        store.payload_timeliness_vote[cls.TARGET] = list(votes)
        store.payload_data_availability_vote[cls.TARGET] = list(votes)
        if boost is not None:
            store.blocks[cls.PROPOSER] = p.BeaconBlock(
                root=cls.PROPOSER, parent_root=boost_parent, slot=current_slot,
                bid=_bid(
                    parent_block_hash=(cls.TARGET_HASH if boost_parent_full
                                       else cls.OTHER_HASH),
                    block_hash=cls.PROPOSER * 1000, slot=current_slot))
            store.proposer_boost_root = boost
        return store

    @classmethod
    def _py_store(cls, spec, *, target_slot, current_slot, n_false, n_true,
                  verified=True, boost=None, boost_parent=None,
                  boost_parent_full=True):
        """pyspec-side store carrying the same parameters."""
        size = int(spec.PTC_SIZE)
        votes = cls._votes(n_false, n_true, size)
        slot_ms = int(spec.config.SLOT_DURATION_MS)

        def block(slot, parent_root, parent_block_hash, block_hash):
            return spec.BeaconBlock(
                slot=spec.Slot(slot), parent_root=cls._R(spec, parent_root),
                body=spec.BeaconBlockBody(
                    signed_execution_payload_bid=spec.SignedExecutionPayloadBid(
                        message=spec.ExecutionPayloadBid(
                            parent_block_hash=cls._H(spec, parent_block_hash),
                            block_hash=cls._H(spec, block_hash)))))

        blocks = {
            cls._R(spec, cls.ANCHOR): block(0, 0, 0, 0),
            cls._R(spec, cls.TARGET): block(
                target_slot, cls.ANCHOR, 0, cls.TARGET_HASH),
        }
        if boost is not None:
            blocks[cls._R(spec, cls.PROPOSER)] = block(
                current_slot, boost_parent,
                cls.TARGET_HASH if boost_parent_full else cls.OTHER_HASH,
                cls.PROPOSER * 1000)
        pytarget = cls._R(spec, cls.TARGET)
        return spec.Store(
            time=_u64(spec, current_slot * slot_ms // 1000),
            genesis_time=_u64(spec, 0),
            justified_checkpoint=spec.Checkpoint(),
            finalized_checkpoint=spec.Checkpoint(),
            unrealized_justified_checkpoint=spec.Checkpoint(),
            unrealized_finalized_checkpoint=spec.Checkpoint(),
            proposer_boost_root=(cls._R(spec, boost) if boost is not None
                                 else spec.Root()),
            equivocating_indices=set(), blocks=blocks, block_states={},
            block_timeliness={}, checkpoint_states={}, latest_messages={},
            unrealized_justifications={},
            payloads=({pytarget: spec.ExecutionPayloadEnvelope()}
                      if verified else {}),
            payload_timeliness_vote={pytarget: list(votes)},
            payload_data_availability_vote={pytarget: list(votes)},
        )

    # (n_false, n_true): the build-rule boundary, silence cells, extremes
    VOTE_GRID = [(0, 512), (0, 256), (0, 255), (255, 257),
                 (256, 256), (257, 255), (258, 254), (512, 0)]

    @pytest.mark.parametrize("n_false,n_true", VOTE_GRID)
    @pytest.mark.parametrize("target_slot,current_slot", [(1, 2), (1, 3)])
    @pytest.mark.parametrize("status", [PayloadStatus.FULL, PayloadStatus.EMPTY])
    def test_should_build_on_full_matches_pyspec(
            self, spec, n_false, n_true, target_slot, current_slot, status):
        kw = dict(target_slot=target_slot, current_slot=current_slot,
                  n_false=n_false, n_true=n_true)
        sim = F.should_build_on_full(
            self._sim_store(spec, **kw), F.ForkChoiceNode(self.TARGET, status))
        py = _should_build_on_full(
            spec, self._py_store(spec, **kw),
            spec.ForkChoiceNode(root=self._R(spec, self.TARGET),
                                payload_status=int(status)),
            current_slot)
        assert bool(sim) == bool(py)

    # boost shape: (label, boost_root, boost_parent, boost_parent_full)
    BOOST_GRID = [
        ("no-boost", None, None, True),
        ("boost-elsewhere", 3, 1, True),
        ("boost-on-target-full", 3, 2, True),
        ("boost-on-target-empty", 3, 2, False),
    ]

    @pytest.mark.parametrize("n_false,n_true", VOTE_GRID)
    @pytest.mark.parametrize("label,boost,boost_parent,boost_full", BOOST_GRID)
    def test_should_extend_payload_matches_pyspec(
            self, spec, n_false, n_true, label, boost, boost_parent, boost_full):
        kw = dict(target_slot=1, current_slot=2, n_false=n_false, n_true=n_true,
                  boost=boost, boost_parent=boost_parent,
                  boost_parent_full=boost_full)
        sim = F.should_extend_payload(self._sim_store(spec, **kw), self.TARGET)
        py = spec.should_extend_payload(self._py_store(spec, **kw),
                                        self._R(spec, self.TARGET))
        assert bool(sim) == bool(py), label

    @pytest.mark.parametrize("verified", [True, False])
    def test_should_extend_payload_unverified_matches_pyspec(self, spec, verified):
        kw = dict(target_slot=1, current_slot=2, n_false=0, n_true=512,
                  verified=verified, boost=3, boost_parent=2,
                  boost_parent_full=False)
        assert bool(F.should_extend_payload(self._sim_store(spec, **kw), self.TARGET)) \
            == bool(spec.should_extend_payload(self._py_store(spec, **kw),
                                               self._R(spec, self.TARGET)))

    @pytest.mark.parametrize("n_false,n_true", VOTE_GRID)
    @pytest.mark.parametrize("target_slot,current_slot", [(1, 2), (1, 3)])
    @pytest.mark.parametrize("status", [PayloadStatus.FULL, PayloadStatus.EMPTY])
    def test_payload_status_tiebreaker_matches_pyspec(
            self, spec, n_false, n_true, target_slot, current_slot, status):
        kw = dict(target_slot=target_slot, current_slot=current_slot,
                  n_false=n_false, n_true=n_true, boost=3, boost_parent=2,
                  boost_parent_full=False)
        sim = F.get_payload_status_tiebreaker(
            self._sim_store(spec, **kw), F.ForkChoiceNode(self.TARGET, status))
        py = spec.get_payload_status_tiebreaker(
            self._py_store(spec, **kw),
            spec.ForkChoiceNode(root=self._R(spec, self.TARGET),
                                payload_status=int(status)))
        assert int(sim) == int(py)

    def test_two_path_seat_counts_match_pyspec(self, spec):
        """The paper's headline boundary, checked against the executable spec.

        Path A: 257 *signed* not-present votes flip the next proposer's build
        rule with no boosted proposer in play; 256 do not. Path B: 256 seats
        *withholding* their affirmation leave the build rule fail-open — an
        honest proposer still builds FULL — but defeat the tiebreaker once a
        boosted proposer builds EMPTY on the target; 255 silent seats do not.
        """
        full = F.ForkChoiceNode(self.TARGET, PayloadStatus.FULL)
        pyfull = spec.ForkChoiceNode(root=self._R(spec, self.TARGET),
                                     payload_status=int(PayloadStatus.FULL))

        def build_rule(n_false):
            kw = dict(target_slot=1, current_slot=2,
                      n_false=n_false, n_true=512 - n_false)
            sim = bool(F.should_build_on_full(self._sim_store(spec, **kw), full))
            py = bool(_should_build_on_full(spec, self._py_store(spec, **kw), pyfull, 2))
            assert sim == py
            return sim

        def tiebreaker(silent):
            kw = dict(target_slot=1, current_slot=2, n_false=0,
                      n_true=512 - silent, boost=3, boost_parent=2,
                      boost_parent_full=False)
            sim = bool(F.should_extend_payload(self._sim_store(spec, **kw), self.TARGET))
            py = bool(spec.should_extend_payload(self._py_store(spec, **kw),
                                                 self._R(spec, self.TARGET)))
            assert sim == py
            return sim

        # Path A — the fail-open build rule flips at exactly 257.
        assert build_rule(256) is True
        assert build_rule(257) is False
        # Path B — silence never reaches the build rule ...
        for silent in (255, 256, 257):
            kw = dict(target_slot=1, current_slot=2, n_false=0,
                      n_true=512 - silent)
            assert bool(F.should_build_on_full(self._sim_store(spec, **kw), full)) is True
            assert bool(
                _should_build_on_full(spec, self._py_store(spec, **kw), pyfull, 2)) is True
        # ... but 256 withheld affirmations do defeat the tiebreaker, given the
        # boosted proposer that Path B additionally requires.
        assert tiebreaker(255) is True
        assert tiebreaker(256) is False
