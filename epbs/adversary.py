"""Adversary roles for Tier 2a — **not in spec**.

Three hardcoded adversaries:

* ``HonestAdversary``     — reveals payload promptly, votes per local view
* ``ExAnteReorgAdversary`` — single Byzantine builder + a fraction of Byzantine
  attesters; reorgs slot N by withholding b_N's payload and pivoting votes
  onto the EMPTY ancestor (Schwarz-Schilter et al. ex-ante reorg, ePBS-rewritten
  via payload-status lever).
* ``PayloadLeverAdversary`` — H-5 hypothesis: a Byzantine builder withholds
  payload to drive committee weight to the EMPTY node. Combined with a
  Byzantine attester fraction, can reorg with strictly less attester power
  than a pure attestation-only attack (the lever).

Each adversary exposes a uniform 3-method API consumed by ``env_tier2.py``.
Adversaries do NOT bypass spec functions — every action terminates as a
``Message`` on ``MessageBus`` which is then routed to ``epbs.forkchoice``
handlers.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Protocol

import config
from epbs import attestation as A
from epbs import forkchoice as F
from epbs import primitives as p
from epbs.network import Message, MessageBus, MessageKind
from epbs.primitives import PayloadStatus


# --------------------------------------------------------------------------
# Action vocabulary
# --------------------------------------------------------------------------
class BuilderAction:
    REVEAL_NOW = "reveal_now"
    REVEAL_LATE = "reveal_late"        # reveal after PAYLOAD_DUE_BPS
    WITHHOLD = "withhold"


class AttesterAction:
    HONEST = "honest"                   # vote per local view
    VOTE_EMPTY = "vote_empty"           # always vote EMPTY (favours reorg target)
    VOTE_FULL = "vote_full"             # always vote FULL (require local payload)
    WITHHOLD_VOTE = "withhold_vote"     # don't broadcast at all


@dataclass
class AdvCtx:
    """Per-call context passed to adversary methods.

    Bundles the simulation-side knobs an adversary needs to make decisions.
    ``store`` lets the adversary read fork-choice state; ``bus`` lets it
    schedule message deliveries.
    """

    store: F.Store
    bus: MessageBus
    current_slot: int
    current_ms: int


# --------------------------------------------------------------------------
# Adversary protocol
# --------------------------------------------------------------------------
class Adversary(Protocol):
    """Three-method API. Each method may emit zero or more messages onto
    ``ctx.bus`` and returns the symbolic action (for logging / metrics).
    """

    def builder_action(
        self, builder_index: int, block_root: int, ctx: AdvCtx
    ) -> str: ...

    def attester_action(
        self, validator_index: int, slot: int, view_root: int, ctx: AdvCtx
    ) -> str: ...

    def proposer_action(
        self, proposer_index: int, slot: int, ctx: AdvCtx
    ) -> tuple[int, PayloadStatus] | None: ...  # (parent_root, parent_status) or None to skip


def honest_parent_decision(store: F.Store) -> tuple[int, PayloadStatus]:
    """Choose the parent node the current gloas proposer should extend."""
    head = F.get_head(store)
    status = head.payload_status
    if status != PayloadStatus.PENDING:
        status = (
            PayloadStatus.FULL
            if F.should_build_on_full(store, head)
            else PayloadStatus.EMPTY
        )
    return (head.root, status)


# --------------------------------------------------------------------------
# Honest baseline
# --------------------------------------------------------------------------
class HonestAdversary:
    """Always spec-faithful. The retro-degenerate case."""

    def builder_action(self, builder_index, block_root, ctx) -> str:
        # Reveal immediately at current_ms (delivery before PAYLOAD_DUE).
        env = p.ExecutionPayloadEnvelope(
            builder_index=builder_index, beacon_block_root=block_root,
            block_hash=block_root * 1000,
        )
        ctx.bus.schedule(Message(
            sender=builder_index, kind=MessageKind.PAYLOAD_ENVELOPE,
            payload=env, deliver_at_ms=ctx.current_ms,
        ))
        return BuilderAction.REVEAL_NOW

    def attester_action(self, validator_index, slot, view_root, ctx) -> str:
        # Honest attesters vote FULL iff payload locally verified, EMPTY else
        # (spec convention: only `index==1` can be cast if `is_payload_verified`).
        return AttesterAction.HONEST

    def proposer_action(self, proposer_index, slot, ctx):
        # Build on canonical head per fork choice
        return honest_parent_decision(ctx.store)


# --------------------------------------------------------------------------
# Ex-ante reorg adversary  (BunnyFinder-style, ePBS-rewritten)
# --------------------------------------------------------------------------
class ExAnteReorgAdversary:
    """Single Byzantine builder + Byzantine attesters reorg slot N+1.

    Strategy (one-slot horizon):
      - At slot N (the *target* victim slot), the Byzantine builder withholds
        the payload. This drives honest committee N onto the EMPTY node.
      - At slot N+1, the Byzantine proposer (if assigned) builds on the
        EMPTY ancestor of N — bypassing the victim's FULL branch.
      - Byzantine attesters at N+1 also vote EMPTY-ancestor.

    Reorg succeeds iff combined Byzantine attester weight + the payload-
    status lever > honest weight on the FULL branch.
    """

    def __init__(
        self, *, byzantine_builders: set[int], byzantine_validators: set[int],
        target_slot: int,
    ):
        self.byz_builders = byzantine_builders
        self.byz_validators = byzantine_validators
        self.target_slot = target_slot

    def builder_action(self, builder_index, block_root, ctx) -> str:
        if (builder_index in self.byz_builders
                and ctx.current_slot == self.target_slot):
            # withhold by *not* scheduling any envelope message
            return BuilderAction.WITHHOLD
        # honest behaviour
        env = p.ExecutionPayloadEnvelope(builder_index, block_root, block_root * 1000)
        ctx.bus.schedule(Message(
            sender=builder_index, kind=MessageKind.PAYLOAD_ENVELOPE,
            payload=env, deliver_at_ms=ctx.current_ms,
        ))
        return BuilderAction.REVEAL_NOW

    def attester_action(self, validator_index, slot, view_root, ctx) -> str:
        if (validator_index in self.byz_validators
                and slot == self.target_slot + 1):
            # vote EMPTY-ancestor: index=0
            return AttesterAction.VOTE_EMPTY
        return AttesterAction.HONEST

    def proposer_action(self, proposer_index, slot, ctx):
        if (proposer_index in self.byz_validators
                and slot == self.target_slot + 1):
            # build on the EMPTY ancestor of the previous slot
            head = F.get_head(ctx.store)
            if head.payload_status == PayloadStatus.FULL:
                # pivot: pick the EMPTY twin of head's parent if it exists
                parent_root = ctx.store.blocks[head.root].parent_root
                return (parent_root, PayloadStatus.EMPTY)
            return (head.root, PayloadStatus.EMPTY)
        # honest fallback
        return honest_parent_decision(ctx.store)


# --------------------------------------------------------------------------
# Payload-lever adversary (H-5 hypothesis)
# --------------------------------------------------------------------------
class PayloadLeverAdversary:
    """A Byzantine builder always withholds; Byzantine attesters vote EMPTY.

    Differs from `ExAnteReorgAdversary` in being **slot-agnostic**: the
    builder withholds on every slot it controls. Used by ``exp7`` to scan
    the (σ, byzantine_attester_fraction) plane and find where the payload
    lever produces reorgs that pure-attestation attacks could not.
    """

    def __init__(self, *, byzantine_builders: set[int],
                 byzantine_validators: set[int]):
        self.byz_builders = byzantine_builders
        self.byz_validators = byzantine_validators

    def builder_action(self, builder_index, block_root, ctx) -> str:
        if builder_index in self.byz_builders:
            return BuilderAction.WITHHOLD
        env = p.ExecutionPayloadEnvelope(builder_index, block_root, block_root * 1000)
        ctx.bus.schedule(Message(
            sender=builder_index, kind=MessageKind.PAYLOAD_ENVELOPE,
            payload=env, deliver_at_ms=ctx.current_ms,
        ))
        return BuilderAction.REVEAL_NOW

    def attester_action(self, validator_index, slot, view_root, ctx) -> str:
        if validator_index in self.byz_validators:
            return AttesterAction.VOTE_EMPTY
        return AttesterAction.HONEST

    def proposer_action(self, proposer_index, slot, ctx):
        head = F.get_head(ctx.store)
        if proposer_index in self.byz_validators:
            # Always prefer the EMPTY branch (gives payload-withhold attack
            # a coherent extension chain).
            if head.payload_status == PayloadStatus.FULL:
                return (head.root, PayloadStatus.EMPTY)
            return (head.root, head.payload_status)
        return honest_parent_decision(ctx.store)


# --------------------------------------------------------------------------
# Staircase adversary  (Tier 2b — Casper FFG + LMD-GHOST combo attack)
# --------------------------------------------------------------------------
class StaircaseAdversary:
    """BunnyFinder-style staircase attack — reproduced under ePBS + Casper FFG.

    Strategy (one-epoch horizon):
      - Byzantine attesters withhold their attestations from the wire
        throughout the *target* epoch.
      - At the moment the next epoch's first slot's proposer-boost root is
        determined (or, more crudely, at the boundary), they release the
        withheld attestations on a fork that prevents the canonical
        epoch-target from reaching the 2/3 justification threshold.
      - Net effect: target epoch's first-slot block does *not* get justified,
        leaving the previous justified checkpoint unchanged for an extra
        epoch — the "stair" not climbed.

    In the ePBS variant, the Byzantine builder can compound this by
    withholding payload at the target slot, splitting honest attestation
    weight between FULL and EMPTY nodes — making the staircase climbable
    with strictly less attester power. ``payload_lever=True`` enables this.

    This is a simplified Tier 2b model — not a faithful reproduction of
    BunnyFinder's full multi-epoch trace, but exposes the same causal
    mechanism (attestation timing × FFG threshold).
    """

    def __init__(
        self, *, byzantine_validators: set[int], byzantine_builders: set[int],
        target_epoch: int, payload_lever: bool = False,
    ):
        self.byz_validators = byzantine_validators
        self.byz_builders = byzantine_builders
        self.target_epoch = target_epoch
        self.payload_lever = payload_lever

    def _is_target_epoch(self, slot: int) -> bool:
        import config
        return (slot // config.SLOTS_PER_EPOCH) == self.target_epoch

    def builder_action(self, builder_index, block_root, ctx) -> str:
        # Optional payload-status lever
        if (self.payload_lever and builder_index in self.byz_builders
                and self._is_target_epoch(ctx.current_slot)):
            return BuilderAction.WITHHOLD
        # Honest reveal
        env = p.ExecutionPayloadEnvelope(builder_index, block_root, block_root * 1000)
        ctx.bus.schedule(Message(
            sender=builder_index, kind=MessageKind.PAYLOAD_ENVELOPE,
            payload=env, deliver_at_ms=ctx.current_ms,
        ))
        return BuilderAction.REVEAL_NOW

    def attester_action(self, validator_index, slot, view_root, ctx) -> str:
        # Byzantine attesters withhold (don't broadcast) at target epoch
        if (validator_index in self.byz_validators
                and self._is_target_epoch(slot)):
            return AttesterAction.WITHHOLD_VOTE
        return AttesterAction.HONEST

    def proposer_action(self, proposer_index, slot, ctx):
        return honest_parent_decision(ctx.store)


# --------------------------------------------------------------------------
# IndependentAdversaries — non-colluding composition (Phase G)
# --------------------------------------------------------------------------
class IndependentAdversaries:
    """Two adversaries that do not share state.

    Use this to separate "two non-cooperating attackers" from "one entity
    controls both sides" (the implicit assumption of every single-adversary
    class above). The builder-side adversary handles ``builder_action``;
    the validator-side adversary handles ``attester_action`` and
    ``proposer_action``. Neither sees the other's decisions.

    Examples:
      - **colluding** (current default): pass a single
        ``PayloadLeverAdversary(byz_b, byz_v)`` to the env — that one entity
        coordinates withhold + EMPTY-vote on the same slot.
      - **independent**::

            IndependentAdversaries(
                validator_adv=StaircaseAdversary(
                    byz_v, set(), target_epoch=..., payload_lever=False),
                builder_adv=PayloadLeverAdversary(byz_b, set()),
            )

        Each runs its own attack with no coordination — the validator side
        does NOT condition its EMPTY votes on whether a builder withheld.

    The construct is deliberately minimal — it only re-routes calls; it
    does NOT enforce non-overlap of the two adversaries' Byzantine index
    sets. Caller is responsible for picking semantically distinct sets
    (typically: validator side passes empty ``byzantine_builders``, builder
    side passes empty ``byzantine_validators``).
    """

    def __init__(self, validator_adv, builder_adv):
        self.validator_adv = validator_adv
        self.builder_adv = builder_adv

    def builder_action(self, builder_index, block_root, ctx):
        return self.builder_adv.builder_action(builder_index, block_root, ctx)

    def attester_action(self, validator_index, slot, view_root, ctx):
        return self.validator_adv.attester_action(
            validator_index, slot, view_root, ctx)

    def proposer_action(self, proposer_index, slot, ctx):
        return self.validator_adv.proposer_action(proposer_index, slot, ctx)
