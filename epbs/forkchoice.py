"""LMD-GHOST + ePBS payload-status fork choice — ported from `gloas/fork-choice.md`.

This is the core of Tier 2a. Casper FFG is **not** ported: the
``justified_checkpoint`` is static (set at store creation) and the
unrealized-justification machinery is omitted. Every other helper that
fork-choice transitively needs is ported faithfully.

Layout:
    Store + ForkChoiceNode dataclasses
    LMD-GHOST: get_supported_node / is_ancestor / get_node_children / get_attestation_score
               / get_weight / get_head / get_payload_status_tiebreaker
    Proposer boost: should_apply_proposer_boost / update_proposer_boost_root
                    / is_head_weak / is_parent_strong
    Block timing: record_block_timeliness
    Ancestor: get_ancestor (returns ForkChoiceNode in gloas)
    Handlers: on_block / on_execution_payload_envelope /
              on_payload_attestation_message / on_attestation / on_tick

Spec source: ``gloas/fork-choice.md`` (every public function below carries the
section reference inline). Helpers that are unchanged from phase0 are pulled
in from ``epbs.attestation`` / ``epbs.primitives``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import config
from epbs import attestation as A
from epbs import primitives as p
from epbs.primitives import (
    BeaconBlock, ExecutionPayloadEnvelope, PayloadStatus,
    get_parent_payload_status, is_parent_node_full,
)


# --------------------------------------------------------------------------
# Constants — pinned to gloas/fork-choice.md
# --------------------------------------------------------------------------
ATTESTATION_TIMELINESS_INDEX: int = 0
PTC_TIMELINESS_INDEX: int = 1
NUM_BLOCK_TIMELINESS_DEADLINES: int = 2


# --------------------------------------------------------------------------
# ForkChoiceNode  (gloas/fork-choice.md `class ForkChoiceNode`)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ForkChoiceNode:
    """The unit on which LMD-GHOST operates — (root, payload_status)."""

    root: int
    payload_status: PayloadStatus


# --------------------------------------------------------------------------
# Store  (gloas/fork-choice.md `class Store`, slimmed)
# --------------------------------------------------------------------------
# Phase0/Altair FFG constants  (pinned to consensus-specs; diff-tested)
JUSTIFICATION_BITS_LENGTH: int = 4
TIMELY_TARGET_FLAG_INDEX: int = 1
GENESIS_EPOCH: int = 0


@dataclass
class Store:
    """Tier 2a/b fork-choice store.

    Tier 2a-only fields (LMD-GHOST + payload-status): ``blocks``,
    ``payloads``, ``block_timeliness``, ``payload_timeliness_vote``,
    ``payload_data_availability_vote``, ``latest_messages``,
    ``effective_balances``, ``total_active_balance``.

    Tier 2b additions (Casper FFG; see epbs.ffg): ``current_justified_checkpoint``,
    ``previous_justified_checkpoint``, ``finalized_checkpoint``,
    ``justification_bits``. The legacy ``justified_root`` / ``finalized_root``
    fields are kept as convenience aliases (mirror ``*_checkpoint.root``).

    Fields **still omitted** vs spec: ``unrealized_*_checkpoint``,
    ``unrealized_justifications`` (Tier 2b derives realized values directly
    from latest_messages at epoch boundaries; no per-block pull-up cache),
    ``checkpoint_states`` (no BeaconState model), ``block_states`` (idem).
    """

    # time (ms since genesis-equivalent)
    time_ms: int = 0
    genesis_time_ms: int = 0
    # legacy aliases — kept in sync with *_checkpoint.root by FFG helpers
    justified_root: int = 0
    finalized_root: int = 0
    # FFG checkpoints (Tier 2b)
    current_justified_checkpoint: Optional["A.Checkpoint"] = None
    previous_justified_checkpoint: Optional["A.Checkpoint"] = None
    finalized_checkpoint: Optional["A.Checkpoint"] = None
    # 4-bit history of recent epoch justifications (gloas inherits altair)
    # bit 0 = current epoch, bit 1 = previous, ...
    justification_bits: list[bool] = field(
        default_factory=lambda: [False] * JUSTIFICATION_BITS_LENGTH
    )
    proposer_boost_root: int = 0
    equivocating_indices: set[int] = field(default_factory=set)
    # block / payload storage
    blocks: dict[int, BeaconBlock] = field(default_factory=dict)
    payloads: dict[int, ExecutionPayloadEnvelope] = field(default_factory=dict)
    # [Modified in Gloas] timeliness per block: [attestation_timely, ptc_timely]
    block_timeliness: dict[int, list[bool]] = field(default_factory=dict)
    # [New in Gloas] PTC votes per block
    payload_timeliness_vote: dict[int, list[Optional[bool]]] = field(default_factory=dict)
    payload_data_availability_vote: dict[int, list[Optional[bool]]] = field(default_factory=dict)
    # LMD-GHOST latest message per validator
    latest_messages: dict[int, A.LatestMessage] = field(default_factory=dict)
    # validator effective balance lookup (slot-invariant in Tier 2a)
    effective_balances: dict[int, int] = field(default_factory=dict)
    # total active balance — committee fraction reference
    total_active_balance: int = 0
    # epoch at which FFG was last processed (Tier 2b)
    last_ffg_processed_epoch: int = -1
    # perf cache: (root, slot) -> ForkChoiceNode. Stable because both
    # store.blocks[root] and get_parent_payload_status are computed from
    # immutable bid fields (block_hash equality), independent of payloads/
    # latest_messages. Never invalidated; only grown.
    _ancestor_cache: dict = field(default_factory=dict)
    # perf cache: inverted attestation index. score_index[(root, status_int)]
    # is the accumulated effective balance of validators whose latest message
    # supports that node (see get_supported_node / is_ancestor). Maintained incrementally by
    # _update_score_index on every update_latest_messages call. Equivalent to
    # the old per-call loop in get_attestation_score, but O(1) lookup.
    _score_index: dict = field(default_factory=dict)
    _validator_supports: dict = field(default_factory=dict)
    # FFG target balance: epoch_target_balance[(epoch, target_root)] is the
    # accumulated effective balance of validators who attested target_root in
    # `epoch`. This replaces the latest_messages-derived computation that
    # silently lost fork-A votes when a validator's latest message rotated to
    # fork B. First attestation per (vi, epoch) wins; later overrides don't
    # double-count (and don't switch the locked-in target — matches the
    # altair invariant that participation flags are monotonic OR-only).
    epoch_target_balance: dict = field(default_factory=dict)
    _ffg_target_seen: set = field(default_factory=set)  # (vi, epoch) lock-in
    # Phase H1 — gloas builder_pending_payments ledger.
    # builder_pending_payments[absolute_slot] = PendingPayment (un-settled bid).
    # proposer_credits[proposer_index] = cumulative gwei RECEIVED from builders
    # builder_debits[builder_index]    = cumulative gwei PAID to proposers
    # Spec direction: builder → proposer (gloas/builder.md:135 — `bid.value` is
    # what the builder will pay the proposer if the bid is accepted).
    builder_pending_payments: dict = field(default_factory=dict)
    proposer_credits: dict = field(default_factory=dict)
    builder_debits: dict = field(default_factory=dict)
    # Paper 2 observability only: resolved payment telemetry. These fields do
    # not participate in settlement decisions; they let experiment scripts
    # report successful vs expired commitments without inferring outcomes from
    # attacker actions. Payments still pending at episode end remain censored.
    settled_payment_slots: list[int] = field(default_factory=list)
    expired_payment_slots: list[int] = field(default_factory=list)
    settled_payment_amount_gwei: int = 0
    expired_payment_amount_gwei: int = 0
    # Builder balance (gwei) per builder index. Spec analogue is
    # `state.builders[builder_index].balance` (gloas/beacon-chain.md:589-599
    # `can_builder_cover_bid`). When a key is missing the env uses
    # `builder.DEFAULT_BUILDER_STAKE_GWEI`. Free balance reserves
    # `MIN_DEPOSIT_AMOUNT` plus outstanding pending payments. Settlements
    # decrease this balance immediately in the compact ledger model.
    builder_stakes: dict = field(default_factory=dict)
    # Phase H2 — proposer attestation-inclusion rewards.
    # proposer_rewards[proposer_index] = cumulative gwei.
    # _proposer_reward_credited[(vi, epoch)] = first-time-only gate.
    proposer_rewards: dict = field(default_factory=dict)
    _proposer_reward_credited: set = field(default_factory=set)
    # Phase H+ — per-attester attestation reward/penalty net ledger (altair
    # get_flag_index_deltas). attester_rewards[vi] = cumulative net gwei
    # (reward when the committee member attests, penalty when it abstains).
    attester_rewards: dict = field(default_factory=dict)
    _attester_credited: set = field(default_factory=set)


def get_forkchoice_store(
    anchor_block: BeaconBlock, *, effective_balances: dict[int, int],
    total_active_balance: int, genesis_time_ms: int = 0,
) -> Store:
    """Port of gloas/fork-choice.md `get_forkchoice_store` (slimmed).

    Initialises FFG state: anchor block doubles as the GENESIS_EPOCH
    justified + finalized checkpoint (spec convention: justified_checkpoint
    starts at (anchor_epoch, anchor_root)).
    """
    anchor_cp = A.Checkpoint(
        epoch=A.compute_epoch_at_slot(anchor_block.slot), root=anchor_block.root)
    store = Store(
        time_ms=genesis_time_ms,
        genesis_time_ms=genesis_time_ms,
        justified_root=anchor_block.root,
        finalized_root=anchor_block.root,
        current_justified_checkpoint=anchor_cp,
        previous_justified_checkpoint=anchor_cp,
        finalized_checkpoint=anchor_cp,
        proposer_boost_root=0,
        effective_balances=dict(effective_balances),
        total_active_balance=total_active_balance,
    )
    store.blocks[anchor_block.root] = anchor_block
    # anchor block is considered timely on both deadlines
    store.block_timeliness[anchor_block.root] = [True, True]
    # PTC vote arrays — anchor block has no payload to vote on, but spec
    # initialises them so payload_timeliness assertions don't trip.
    store.payload_timeliness_vote[anchor_block.root] = [None] * config.PTC_SIZE
    store.payload_data_availability_vote[anchor_block.root] = [None] * config.PTC_SIZE
    return store


# --------------------------------------------------------------------------
# Current-slot helper  (phase0 fork-choice utility, gloas inherits)
# --------------------------------------------------------------------------
def get_current_slot(store: Store) -> int:
    """`get_current_slot` — derived from store.time_ms."""
    elapsed_ms = store.time_ms - store.genesis_time_ms
    return elapsed_ms // config.MILLIS_PER_SLOT


# --------------------------------------------------------------------------
# Ancestor / checkpoint helpers  (gloas/fork-choice.md)
# --------------------------------------------------------------------------
def get_ancestor(store: Store, root: int, slot: int) -> ForkChoiceNode:
    """Port of gloas/fork-choice.md `get_ancestor`.

    Returns the (root, payload_status) ancestor of ``root`` at ``slot``. If
    ``root`` is already at ``slot`` (or we ask for a future slot), the
    ancestor is PENDING. Otherwise we walk the parent chain; payload_status
    is determined by `get_parent_payload_status` of the *child* block.

    Memoised on (root, slot) — the chain above ``root`` is immutable once
    added (bids' block_hash fields are frozen at on_block time), and
    `get_parent_payload_status` only inspects those bid fields, so the
    result never changes. The cache is grown indefinitely; for episode-scale
    simulations the working set stays bounded.
    """
    key = (root, slot)
    cached = store._ancestor_cache.get(key)
    if cached is not None:
        return cached

    block = store.blocks[root]
    if block.slot <= slot:
        node = ForkChoiceNode(root=root, payload_status=PayloadStatus.PENDING)
        store._ancestor_cache[key] = node
        return node

    parent = store.blocks[block.parent_root]
    while parent.slot > slot:
        block = parent
        parent = store.blocks[block.parent_root]

    node = ForkChoiceNode(
        root=block.parent_root,
        payload_status=get_parent_payload_status(store, block),
    )
    store._ancestor_cache[key] = node
    return node


def get_checkpoint_block(store: Store, root: int, epoch: int) -> int:
    """Port of gloas/fork-choice.md `get_checkpoint_block`."""
    epoch_first_slot = epoch * config.SLOTS_PER_EPOCH
    return get_ancestor(store, root, epoch_first_slot).root


# --------------------------------------------------------------------------
# Node-centric vote support  (gloas/fork-choice.md `get_supported_node` +
# `is_ancestor`; #5249 fork-choice abstraction redesign + #5317)
#
# Replaces the pre-#5249 vote-centric `is_supporting_vote(store, node, msg)`.
# The spec now factors support into `get_supported_node(msg) -> node` (the one
# node a message supports) and `is_ancestor(store, descendant, ancestor)`. A
# message supports `node` iff `is_ancestor(store, get_supported_node(msg), node)`
# — the exact dual of the old predicate.
# --------------------------------------------------------------------------
def get_supported_node(store: Store, message: A.LatestMessage) -> ForkChoiceNode:
    """Port of gloas/fork-choice.md `get_supported_node`.

    The single node a latest message supports. A message from a slot later
    than its target block resolves that block's payload (FULL/EMPTY); a
    same-slot message leaves it PENDING.
    """
    block = store.blocks[message.root]
    if block.slot < message.slot:
        payload_status = (
            PayloadStatus.FULL if message.payload_present else PayloadStatus.EMPTY
        )
    else:
        payload_status = PayloadStatus.PENDING
    return ForkChoiceNode(root=message.root, payload_status=payload_status)


def is_ancestor(
    store: Store, node: ForkChoiceNode, ancestor: ForkChoiceNode
) -> bool:
    """Port of gloas/fork-choice.md `is_ancestor`.

    True iff ``ancestor`` lies on ``node``'s chain at ``ancestor``'s slot with
    compatible payload status (a PENDING ``ancestor`` matches any status).
    Inlines the spec's node-in/node-out `get_ancestor` walk, which preserves
    a node's own payload_status once it is at/below the target slot.
    """
    anc_slot = store.blocks[ancestor.root].slot
    cur = node
    while store.blocks[cur.root].slot > anc_slot:
        block = store.blocks[cur.root]
        cur = ForkChoiceNode(
            root=block.parent_root,
            payload_status=get_parent_payload_status(store, block),
        )
    if cur.root != ancestor.root:
        return False
    return (
        cur.payload_status == ancestor.payload_status
        or ancestor.payload_status == PayloadStatus.PENDING
    )


def is_previous_slot_payload_decision(store: Store, node: ForkChoiceNode) -> bool:
    """Port of gloas/fork-choice.md `is_previous_slot_payload_decision` (#5317).

    A previous-slot block whose payload has not yet been resolved as empty or
    full by attestations — so the FULL/EMPTY choice is made by the payload
    tiebreaker, not by weight (`get_weight` returns 0 for such a node).
    """
    is_previous_slot = store.blocks[node.root].slot + 1 == get_current_slot(store)
    is_payload_decision = node.payload_status in (
        PayloadStatus.EMPTY, PayloadStatus.FULL,
    )
    return is_previous_slot and is_payload_decision


# --------------------------------------------------------------------------
# Node children  (gloas/fork-choice.md `get_node_children`)
# --------------------------------------------------------------------------
def get_node_children(
    store: Store, blocks: dict[int, BeaconBlock], node: ForkChoiceNode
) -> list[ForkChoiceNode]:
    """Port of gloas/fork-choice.md `get_node_children`.

    From a PENDING node, descend into EMPTY (always available) and FULL (only
    if the payload has been locally verified). From an EMPTY/FULL node,
    descend to child blocks whose `get_parent_payload_status` matches.
    """
    if node.payload_status == PayloadStatus.PENDING:
        children = [ForkChoiceNode(node.root, PayloadStatus.EMPTY)]
        if p.is_payload_verified(store, node.root):
            children.append(ForkChoiceNode(node.root, PayloadStatus.FULL))
        return children
    return [
        ForkChoiceNode(root, PayloadStatus.PENDING)
        for root, blk in blocks.items()
        if blk.parent_root == node.root
        and node.payload_status == get_parent_payload_status(store, blk)
    ]


# --------------------------------------------------------------------------
# Attestation score  (gloas/fork-choice.md `get_attestation_score`)
# --------------------------------------------------------------------------
def _compute_message_supports(
    store: Store, msg: A.LatestMessage
) -> frozenset:
    """Enumerate every (root, payload_status_int) node the message supports.

    The inverted index of the node-centric predicate: equivalent to
    `{node : is_ancestor(store, get_supported_node(store, msg), node)}` over
    every node reachable from msg.root up the ancestor chain. Cost is
    O(chain_length) and uses the cached get_ancestor walk.
    """
    if msg.root not in store.blocks:
        return frozenset()
    block_r = store.blocks[msg.root]
    pending_i = int(PayloadStatus.PENDING)
    full_i = int(PayloadStatus.FULL)
    empty_i = int(PayloadStatus.EMPTY)

    supports: set = {(msg.root, pending_i)}
    if msg.slot > block_r.slot:
        supports.add(
            (msg.root, full_i if msg.payload_present else empty_i)
        )

    cur = block_r
    while cur.parent_root in store.blocks:
        parent = store.blocks[cur.parent_root]
        if parent.slot >= cur.slot:
            break
        parent_status_i = int(get_parent_payload_status(store, cur))
        supports.add((parent.root, pending_i))
        supports.add((parent.root, parent_status_i))
        cur = parent
    return frozenset(supports)


def _update_score_index(store: Store, vi: int, new_msg) -> None:
    """Maintain the inverted attestation index for validator ``vi``.

    Called by `update_latest_messages` (in attestation.py) immediately after
    `store.latest_messages[vi]` is rewritten. Decrement the previous supports
    set, compute new supports, increment.

    Defensive: bail silently if `store` lacks the perf-cache fields. The
    `_StubStore` used by difftest/test_attestation.py only provides the spec-
    visible Store surface; the index is a Tier-2 optimisation those tests
    don't exercise.
    """
    if not hasattr(store, "effective_balances"):
        return
    eff_bal = store.effective_balances.get(vi, 0)
    if eff_bal == 0:
        return

    old_supports = store._validator_supports.get(vi)
    if old_supports:
        for key in old_supports:
            store._score_index[key] = store._score_index.get(key, 0) - eff_bal

    if vi in store.equivocating_indices or new_msg is None:
        store._validator_supports.pop(vi, None)
        return

    new_supports = _compute_message_supports(store, new_msg)
    for key in new_supports:
        store._score_index[key] = store._score_index.get(key, 0) + eff_bal
    store._validator_supports[vi] = new_supports


def record_ffg_target_vote(
    store: Store, vi: int, target_epoch: int, target_root: int,
) -> None:
    """Lock in the (vi, epoch) -> (target_root, effective_balance) tally.

    First attestation per (vi, epoch) wins, mirroring altair's monotonic
    OR-only participation flags. Subsequent re-attestations by the same
    validator in the same epoch (whether on the same fork or a different
    fork) do NOT change the FFG accounting — the validator is committed.
    """
    if not hasattr(store, "effective_balances"):
        return
    key = (vi, target_epoch)
    if key in store._ffg_target_seen:
        return
    eff_bal = store.effective_balances.get(vi, 0)
    if eff_bal == 0:
        return
    store._ffg_target_seen.add(key)
    store.epoch_target_balance[(target_epoch, target_root)] = (
        store.epoch_target_balance.get((target_epoch, target_root), 0) + eff_bal
    )


def rebuild_score_index(store: Store) -> None:
    """Rebuild `_score_index` from scratch off the current `latest_messages`.

    Public helper for callers that wrote `latest_messages` directly (e.g.,
    diff tests, fixtures) without going through `update_latest_messages`.
    Also called lazily by `get_attestation_score` when it detects an empty
    index alongside non-empty latest_messages.
    """
    store._score_index.clear()
    store._validator_supports.clear()
    for vi, msg in store.latest_messages.items():
        if vi in store.equivocating_indices:
            continue
        _update_score_index(store, vi, msg)


def get_attestation_score(store: Store, node: ForkChoiceNode) -> int:
    """Port of gloas/fork-choice.md `get_attestation_score`.

    Sum of effective balances of unslashed-active validators whose latest
    message supports this `(root, payload_status)` node.

    Performance: backed by the incrementally-maintained `_score_index`
    (updated on every `on_attestation` / `update_latest_messages`). Old
    O(N) loop preserved as `_get_attestation_score_naive` for diff testing.
    """
    if not store._score_index and store.latest_messages:
        rebuild_score_index(store)
    return store._score_index.get(
        (node.root, int(node.payload_status)), 0
    )


def _get_attestation_score_naive(store: Store, node: ForkChoiceNode) -> int:
    """Reference (slow) implementation — kept for diff testing the index."""
    total = 0
    for i, eff_bal in store.effective_balances.items():
        if i in store.equivocating_indices:
            continue
        msg = store.latest_messages.get(i)
        if msg is None:
            continue
        if is_ancestor(store, get_supported_node(store, msg), node):
            total += eff_bal
    return total


# --------------------------------------------------------------------------
# Reorg thresholds  (gloas/fork-choice.md `is_head_weak` / `is_parent_strong`)
# --------------------------------------------------------------------------
def _calculate_committee_fraction(store: Store, committee_percent: int) -> int:
    """Port of phase0 `calculate_committee_fraction`.

    Returns ``(total_active_balance / SLOTS_PER_EPOCH) * (committee_percent / 100)``.
    """
    committee_weight = store.total_active_balance // config.SLOTS_PER_EPOCH
    return (committee_weight * committee_percent) // 100


def is_head_weak(store: Store, head_root: int) -> bool:
    """Port of gloas/fork-choice.md `is_head_weak` (simplified).

    The gloas version adds equivocating-validator weight to head weight, which
    relies on ``get_committee_count_per_slot`` (BeaconState path). Tier 2a
    omits that addition — equivocating validators are simply excluded from
    `get_attestation_score`, so head weight is a lower bound on the spec
    value. This makes our `is_head_weak` *conservative* (more likely to
    flag weak), which is the safer error direction for reorg-style attacks.
    """
    threshold = _calculate_committee_fraction(
        store, config.REORG_HEAD_WEIGHT_THRESHOLD)
    head_node = ForkChoiceNode(root=head_root, payload_status=PayloadStatus.PENDING)
    head_weight = get_attestation_score(store, head_node)
    return head_weight < threshold


def is_parent_strong(store: Store, root: int) -> bool:
    """Port of gloas/fork-choice.md `is_parent_strong`."""
    threshold = _calculate_committee_fraction(
        store, config.REORG_PARENT_WEIGHT_THRESHOLD)
    block = store.blocks[root]
    parent_status = get_parent_payload_status(store, block)
    parent_node = ForkChoiceNode(root=block.parent_root, payload_status=parent_status)
    parent_weight = get_attestation_score(store, parent_node)
    return parent_weight > threshold


# --------------------------------------------------------------------------
# Proposer boost  (gloas/fork-choice.md)
# --------------------------------------------------------------------------
def get_proposer_score(store: Store) -> int:
    """Port of `get_proposer_score` (boosting weight)."""
    return _calculate_committee_fraction(store, config.PROPOSER_SCORE_BOOST)


def should_apply_proposer_boost(store: Store) -> bool:
    """Port of gloas/fork-choice.md `should_apply_proposer_boost`."""
    if store.proposer_boost_root == 0:
        return False
    block = store.blocks[store.proposer_boost_root]
    parent = store.blocks[block.parent_root]
    slot = block.slot

    # Apply boost if parent is not from the previous slot
    if parent.slot + 1 < slot:
        return True
    # Apply boost if parent is not weak
    if not is_head_weak(store, block.parent_root):
        return True

    # parent is weak + from previous slot — apply boost iff no early
    # equivocation by the same proposer for the previous slot
    equivocations = [
        root for root, blk in store.blocks.items()
        if (store.block_timeliness.get(root, [False, False])[PTC_TIMELINESS_INDEX]
            and blk.proposer_index == parent.proposer_index
            and blk.slot + 1 == slot
            and root != block.parent_root)
    ]
    return len(equivocations) == 0


# --------------------------------------------------------------------------
# Weight  (gloas/fork-choice.md `get_weight`)
# --------------------------------------------------------------------------
def get_weight(store: Store, node: ForkChoiceNode) -> int:
    """Port of gloas/fork-choice.md `get_weight`.

    A previous-slot, not-yet-resolved payload node has zero weight so the
    FULL/EMPTY choice falls to `get_payload_status_tiebreaker` rather than to
    attestation weight (#5249/#5317). Proposer boost is applied when ``node``
    is an ancestor of the (PENDING) proposer-boost node.
    """
    if is_previous_slot_payload_decision(store, node):
        return 0
    attestation_score = get_attestation_score(store, node)
    if not should_apply_proposer_boost(store):
        return attestation_score
    proposer_score = 0
    proposer_boost_node = ForkChoiceNode(
        root=store.proposer_boost_root, payload_status=PayloadStatus.PENDING
    )
    if is_ancestor(store, proposer_boost_node, node):
        proposer_score = get_proposer_score(store)
    return attestation_score + proposer_score


# --------------------------------------------------------------------------
# Tiebreaker + filtered tree  (gloas/fork-choice.md)
# --------------------------------------------------------------------------
def should_extend_payload(store: Store, root: int) -> bool:
    """Port of gloas/fork-choice.md `should_extend_payload`."""
    if not p.is_payload_verified(store, root):
        return False
    proposer_root = store.proposer_boost_root
    timely = p.payload_timeliness(store, root, timely=True)
    available = p.payload_data_availability(store, root, available=True)
    return (
        (timely and available)
        or proposer_root == 0
        or store.blocks[proposer_root].parent_root != root
        or is_parent_node_full(store, store.blocks[proposer_root])
    )


def should_build_on_full(store: Store, head: ForkChoiceNode) -> bool:
    """Port of gloas/fork-choice.md `should_build_on_full`."""
    assert head.payload_status != PayloadStatus.PENDING
    if head.payload_status == PayloadStatus.EMPTY:
        return False
    if store.blocks[head.root].slot + 1 != get_current_slot(store):
        return True
    if p.payload_data_availability(store, head.root, available=False):
        return False
    if p.payload_timeliness(store, head.root, timely=False):
        return False
    return True


def get_payload_status_tiebreaker(store: Store, node: ForkChoiceNode) -> int:
    """Port of gloas/fork-choice.md `get_payload_status_tiebreaker`."""
    if is_previous_slot_payload_decision(store, node):
        # Decide a previous-slot payload by PTC view: EMPTY ranks 1, FULL ranks
        # 2 iff we should extend it, else 0.
        if node.payload_status == PayloadStatus.EMPTY:
            return 1
        return 2 if should_extend_payload(store, node.root) else 0
    return int(node.payload_status)


def _get_filtered_block_tree(store: Store) -> dict[int, BeaconBlock]:
    """Phase0 `get_filtered_block_tree` slim port.

    Returns blocks that descend from the **current** justified_root and don't
    conflict with the finalized_root (i.e., their finalized-epoch ancestor
    must match ``store.finalized_root``). Tier 2b: filter respects FFG.
    """
    if not store.blocks:
        return {}
    viable: dict[int, BeaconBlock] = {}
    finalized_epoch_first_slot = (
        store.finalized_checkpoint.epoch * config.SLOTS_PER_EPOCH
        if store.finalized_checkpoint is not None else 0
    )

    for root, blk in store.blocks.items():
        # 1) must descend from justified_root
        cur = blk
        descends_from_justified = False
        while True:
            if cur.root == store.justified_root:
                descends_from_justified = True
                break
            if cur.parent_root not in store.blocks:
                break
            cur = store.blocks[cur.parent_root]
        if not descends_from_justified:
            continue

        # 2) must not conflict with finalized_root: if a finalized checkpoint
        #    exists and this block's chain passes through finalized_epoch's
        #    first slot, the ancestor at that slot must be store.finalized_root.
        if (store.finalized_checkpoint is not None
                and store.finalized_checkpoint.epoch > 0):
            if blk.slot >= finalized_epoch_first_slot:
                anc_root = get_ancestor(store, root, finalized_epoch_first_slot).root
                if anc_root != store.finalized_root:
                    continue
        viable[root] = blk
    # The justified anchor itself is always viable
    if store.justified_root in store.blocks:
        viable[store.justified_root] = store.blocks[store.justified_root]
    return viable


# --------------------------------------------------------------------------
# get_head  (gloas/fork-choice.md)
# --------------------------------------------------------------------------
def get_head(store: Store) -> ForkChoiceNode:
    """Port of gloas/fork-choice.md `get_head` — LMD-GHOST main loop."""
    blocks = _get_filtered_block_tree(store)
    head = ForkChoiceNode(
        root=store.justified_root, payload_status=PayloadStatus.PENDING
    )
    while True:
        children = get_node_children(store, blocks, head)
        if not children:
            return head
        head = max(
            children,
            key=lambda child: (
                get_weight(store, child),
                child.root,
                get_payload_status_tiebreaker(store, child),
            ),
        )


# --------------------------------------------------------------------------
# Block timeliness  (gloas/fork-choice.md `record_block_timeliness`)
# --------------------------------------------------------------------------
def record_block_timeliness(store: Store, root: int) -> None:
    """Port of gloas/fork-choice.md `record_block_timeliness`."""
    block = store.blocks[root]
    elapsed_ms = store.time_ms - store.genesis_time_ms
    time_into_slot_ms = elapsed_ms % config.MILLIS_PER_SLOT
    is_current_slot = get_current_slot(store) == block.slot
    att_thr = _attestation_due_ms()
    ptc_thr = p.get_payload_attestation_due_ms()
    store.block_timeliness[root] = [
        is_current_slot and time_into_slot_ms < att_thr,
        is_current_slot and time_into_slot_ms < ptc_thr,
    ]


def _attestation_due_ms() -> int:
    """Port of gloas `get_attestation_due_ms` (25% of a slot)."""
    return p.get_attestation_due_ms()


def get_dependent_root(store: Store, root: int) -> int:
    """Port of gloas/fork-choice.md `get_dependent_root`."""
    epoch = get_current_slot(store) // config.SLOTS_PER_EPOCH
    if epoch <= config.MIN_SEED_LOOKAHEAD:
        return 0
    dependent_slot = (
        (epoch - config.MIN_SEED_LOOKAHEAD) * config.SLOTS_PER_EPOCH - 1
    )
    return get_ancestor(store, root, dependent_slot).root


def update_proposer_boost_root(store: Store, head: int, root: int) -> None:
    """Port of gloas/fork-choice.md `update_proposer_boost_root`."""
    is_first_block = store.proposer_boost_root == 0
    is_timely = store.block_timeliness.get(root, [False, False])[ATTESTATION_TIMELINESS_INDEX]
    is_same_dependent_root = (
        get_dependent_root(store, root) == get_dependent_root(store, head)
    )
    if is_timely and is_first_block and is_same_dependent_root:
        store.proposer_boost_root = root


# --------------------------------------------------------------------------
# Handlers  (gloas/fork-choice.md)
# --------------------------------------------------------------------------
def on_block(store: Store, block: BeaconBlock) -> None:
    """Port of gloas/fork-choice.md `on_block` (slimmed; no state_transition).

    Checks the parent-payload-verified invariant from gloas (block claiming
    FULL parent => parent payload must be locally verified). Adds block to
    store, initialises PTC vote arrays, records timeliness, updates
    proposer-boost root.

    **Skipped vs spec**: blob-data availability check (Tier 2a treats blobs
    as always-available); `state_transition` (no state model);
    `update_checkpoints` / `compute_pulled_up_tip` (no FFG).
    """
    head = get_head(store).root
    assert block.parent_root in store.blocks, "parent block unknown"
    if is_parent_node_full(store, block):
        assert p.is_payload_verified(store, block.parent_root), (
            "block claims FULL parent but parent payload not locally verified"
        )
    current_slot = get_current_slot(store)
    assert current_slot >= block.slot, "block in the future"
    # finalized-slot ancestry skipped: no FFG

    store.blocks[block.root] = block
    store.payload_timeliness_vote[block.root] = [None] * config.PTC_SIZE
    store.payload_data_availability_vote[block.root] = [None] * config.PTC_SIZE
    record_block_timeliness(store, block.root)
    update_proposer_boost_root(store, head, block.root)


def on_execution_payload_envelope(
    store: Store, envelope: ExecutionPayloadEnvelope
) -> None:
    """Port of gloas/fork-choice.md `on_execution_payload_envelope`."""
    assert envelope.beacon_block_root in store.blocks, "block unknown"
    # Tier 2a: blob data always available; no `verify_execution_payload_envelope`.
    store.payloads[envelope.beacon_block_root] = envelope


def on_payload_attestation_message(
    store: Store, msg: A.PayloadAttestationMessage, ptc: list[int],
    *, is_from_block: bool = False,
) -> None:
    """Port of gloas/fork-choice.md `on_payload_attestation_message`.

    ``ptc`` is the PTC committee for the message's slot (computed by
    ``epbs.committee.get_ptc``). Spec routes through BeaconState; Tier 2a
    passes it explicitly.
    """
    data = msg.data
    assert data.beacon_block_root in store.blocks, "block unknown"
    # PTC slot must match block.slot — spec returns early if not.
    block = store.blocks[data.beacon_block_root]
    if data.slot != block.slot:
        return

    ptc_indices = [i for i, vi in enumerate(ptc) if vi == msg.validator_index]
    assert len(ptc_indices) > 0, "attester not in PTC"
    # BLS signature check stubbed (see MODEL_SCOPE.md)

    timely = store.payload_timeliness_vote[data.beacon_block_root]
    da = store.payload_data_availability_vote[data.beacon_block_root]
    for ptc_index in ptc_indices:
        timely[ptc_index] = data.payload_present
        da[ptc_index] = data.blob_data_available


def on_attestation(
    store: Store, att: A.Attestation, *, is_from_block: bool = False,
    enforce_ffg_target: bool = False,
) -> None:
    """Port of phase0 `on_attestation` + gloas `update_latest_messages`.

    Tier 2a skips `store_target_checkpoint_state` (no FFG state caching).
    Tier 2b: passing ``enforce_ffg_target=True`` enables the spec's
    "LMD vote consistent with FFG vote target" assertion.
    """
    current_slot = get_current_slot(store)
    A.validate_on_attestation(
        store, att, is_from_block=is_from_block, current_slot=current_slot,
        enforce_ffg_target=enforce_ffg_target,
    )
    A.update_latest_messages(store, A.get_attesting_indices(att), att)


def on_tick(store: Store, time_ms: int) -> None:
    """Port of phase0 `on_tick` — clock advance only.

    Tier 2a skips proposer-boost reset (spec resets at slot boundary).
    """
    assert time_ms >= store.time_ms, "time may not move backwards"
    old_slot = get_current_slot(store)
    store.time_ms = time_ms
    new_slot = get_current_slot(store)
    # On a new slot boundary, reset proposer-boost (spec behaviour).
    if new_slot > old_slot:
        store.proposer_boost_root = 0
