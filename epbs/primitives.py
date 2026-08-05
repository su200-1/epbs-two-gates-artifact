"""ePBS consensus primitives — ported from `consensus-specs/specs/gloas`.

Lightweight rewrite (no BLS / no SSZ). Each function is a faithful port of the
named spec function so it can be differential-tested against `eth2spec`
(see ``difftest/``). Spec source: gloas `fork-choice.md` and `beacon-chain.md`.

Casper FFG (justify/finalize) is deliberately NOT ported — see MODEL_SCOPE.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import config


# --------------------------------------------------------------------------
# Payload status  (gloas fork-choice.md: `PayloadStatus`)
# --------------------------------------------------------------------------
class PayloadStatus(IntEnum):
    """Status of an execution payload within fork choice."""

    EMPTY = 0
    FULL = 1
    PENDING = 2


# --------------------------------------------------------------------------
# Containers  (gloas beacon-chain.md, slimmed: only incentive-relevant fields)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ExecutionPayloadBid:
    """`ExecutionPayloadBid` — the builder's commitment, in the beacon block.

    Spec: gloas/beacon-chain.md `class ExecutionPayloadBid`.

    Field semantics (post Phase I-1 — pre-I-1 docstring had ``value`` /
    ``execution_payment`` reversed and is now corrected):

    * ``value`` (gwei) — the amount the **builder pays the proposer** if the
      bid is accepted (gloas/builder.md:135). This is the conditional payment
      that goes through ``builder_pending_payments`` and settles on a FULL
      child or at epoch boundary when regular-attestation quorum is met.
    * ``execution_payment`` (gwei) — the *trusted-EL-payment* indicator.
      Per gloas/builder.md:138, gossiped bids MUST set this to 0; a non-zero
      value indicates an out-of-protocol EL-side payment and the bid MUST NOT
      be broadcast on the ``execution_payload_bid`` topic. The sim hardcodes
      this to 0 in env_tier2._run_one_slot.
    """

    parent_block_hash: int
    parent_block_root: int
    block_hash: int
    builder_index: int
    slot: int
    value: int             # Gwei — builder→proposer conditional payment
    execution_payment: int # Gwei — trusted-EL-payment indicator; MUST be 0 on gossip


@dataclass(frozen=True)
class ExecutionPayloadEnvelope:
    """`ExecutionPayloadEnvelope` — what the builder reveals after committing.

    Spec: beacon-chain.md `class ExecutionPayloadEnvelope`. The payload itself
    is opaque here; its economic value is carried by ``epbs.economics``.
    """

    builder_index: int
    beacon_block_root: int
    block_hash: int


@dataclass
class BeaconBlock:
    """Minimal beacon block — only the fields the free-option chain needs."""

    root: int
    parent_root: int
    slot: int
    bid: ExecutionPayloadBid


@dataclass
class Store:
    """Minimal fork-choice store slice.

    Only the maps the free-option causal chain touches. ``payloads`` holds
    locally-available revealed envelopes; ``payload_timeliness_vote`` and
    ``payload_data_availability_vote`` hold the PTC's per-block votes
    (True / False / None).
    """

    blocks: dict[int, BeaconBlock] = field(default_factory=dict)
    payloads: dict[int, ExecutionPayloadEnvelope] = field(default_factory=dict)
    payload_timeliness_vote: dict[int, list[bool | None]] = field(default_factory=dict)
    payload_data_availability_vote: dict[int, list[bool | None]] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Fork-choice payload-status slice  (gloas fork-choice.md)
# --------------------------------------------------------------------------
def is_payload_verified(store: Store, root: int) -> bool:
    """Whether the payload for ``root`` is locally available.

    Spec: fork-choice.md `is_payload_verified`. Here: the envelope has been
    revealed and received into the store.
    """
    return root in store.payloads


def payload_timeliness(store: Store, root: int, timely: bool) -> bool:
    """Port of gloas fork-choice.md `payload_timeliness`.

    Returns whether the payload for ``root`` is considered ``timely`` (or, when
    ``timely`` is False, not-timely), from local availability + PTC votes.
    """
    assert root in store.payload_timeliness_vote

    # If the payload is not locally available it is not timely regardless of
    # the PTC vote.
    if not is_payload_verified(store, root):
        return not timely

    votes = store.payload_timeliness_vote[root]
    return sum(vote == timely for vote in votes) > config.PAYLOAD_TIMELY_THRESHOLD


def payload_data_availability(store: Store, root: int, available: bool) -> bool:
    """Port of gloas fork-choice.md `payload_data_availability`."""
    assert root in store.payload_data_availability_vote

    if not is_payload_verified(store, root):
        return not available

    votes = store.payload_data_availability_vote[root]
    return (
        sum(vote == available for vote in votes)
        > config.DATA_AVAILABILITY_TIMELY_THRESHOLD
    )


def get_parent_payload_status(store: Store, block: BeaconBlock) -> PayloadStatus:
    """Port of gloas fork-choice.md `get_parent_payload_status`.

    FULL iff the block's bid commits to a parent_block_hash equal to the parent
    block's own bid block_hash; EMPTY otherwise.
    """
    parent = store.blocks[block.parent_root]
    parent_block_hash = block.bid.parent_block_hash
    message_block_hash = parent.bid.block_hash
    return (
        PayloadStatus.FULL
        if parent_block_hash == message_block_hash
        else PayloadStatus.EMPTY
    )


def is_parent_node_full(store: Store, block: BeaconBlock) -> bool:
    """Port of gloas fork-choice.md `is_parent_node_full`."""
    return get_parent_payload_status(store, block) == PayloadStatus.FULL


# --------------------------------------------------------------------------
# Intra-slot timing  (gloas fork-choice.md timing helpers)
# --------------------------------------------------------------------------
def slot_component_duration_ms(bps: int) -> int:
    """Port of `get_slot_component_duration_ms`: BPS fraction of one slot."""
    return config.MILLIS_PER_SLOT * bps // 10_000


def get_payload_due_ms() -> int:
    """Builder payload-reveal deadline within a slot (ms). Spec: gloas."""
    return slot_component_duration_ms(config.PAYLOAD_DUE_BPS)


def get_attestation_due_ms() -> int:
    """Regular beacon-attestation deadline within a slot (ms). Spec: gloas."""
    return slot_component_duration_ms(config.ATTESTATION_DUE_BPS_GLOAS)


def get_payload_attestation_due_ms() -> int:
    """PTC attestation deadline within a slot (ms). Spec: gloas."""
    return slot_component_duration_ms(config.PAYLOAD_ATTESTATION_DUE_BPS)


# --------------------------------------------------------------------------
# Derived helper — the core free-option mechanic
# --------------------------------------------------------------------------
def classify_payload(revealed: bool, reveal_ms: int | None) -> PayloadStatus:
    """Classify a block's payload status from the builder's reveal action.

    Spec-derived (not a single spec function): a payload revealed at or before
    the reveal deadline is FULL; a withheld payload is EMPTY; a payload revealed
    after the deadline is PENDING (too late to count as timely).
    """
    if not revealed or reveal_ms is None:
        return PayloadStatus.EMPTY
    if reveal_ms <= get_payload_due_ms():
        return PayloadStatus.FULL
    return PayloadStatus.PENDING
