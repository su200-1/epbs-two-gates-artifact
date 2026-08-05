"""Attestation data + processing — ported from `phase0` and `gloas` pyspec.

Tier 2a scope: only the parts the LMD-GHOST + payload-status fork choice
reads. **Casper FFG (justify/finalize)** is out of scope — `Checkpoint` is
modelled as an opaque (epoch, root) tuple referencing a known block; we do not
process target-state caching, justification weights, or finality. This is the
same trade-off MODEL_SCOPE.md documents for Tier 1.

Simplifications vs spec:
- BLS signatures: not modelled. `is_valid_indexed_payload_attestation` checks
  only the structural validity (non-empty, sorted indices); signature check is
  stubbed `True`. Differential-test bypasses the BLS path.
- `get_checkpoint_block` (FFG-target consistency): skipped — Tier 2a treats
  every block as living in epoch=`block.slot // SLOTS_PER_EPOCH` and the
  target checkpoint root == ancestor at the epoch-first-slot. This is a
  spec-faithful subset.

Spec sources:
    phase0/beacon-chain.md::Attestation, AttestationData, Checkpoint
    gloas/fork-choice.md::validate_on_attestation, update_latest_messages,
        LatestMessage
    gloas/beacon-chain.md::PayloadAttestation, PayloadAttestationData,
        PayloadAttestationMessage, get_indexed_payload_attestation,
        is_valid_indexed_payload_attestation
"""
from __future__ import annotations

from dataclasses import dataclass, field

import config


# --------------------------------------------------------------------------
# Containers (slimmed — only fields fork-choice / `update_latest_messages` read)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Checkpoint:
    """Spec: `class Checkpoint(epoch, root)`. Tier 2a: opaque pointer."""

    epoch: int
    root: int


@dataclass(frozen=True)
class AttestationData:
    """Slim port of phase0 `AttestationData`.

    Fields ported:
        slot                — attestation slot
        index               — committee index; **gloas: 0 = empty/PENDING vote,
                              1 = FULL vote** (this is the ePBS overload of
                              `data.index ∈ {0,1}`)
        beacon_block_root   — LMD-GHOST head vote
        source / target     — FFG checkpoints (carried but not finality-processed)
    """

    slot: int
    index: int
    beacon_block_root: int
    source: Checkpoint
    target: Checkpoint


@dataclass(frozen=True)
class Attestation:
    """Spec: `class Attestation`. Tier 2a: aggregation bits over a single
    beacon committee (no committee-bits since we don't aggregate cross-committee).
    """

    data: AttestationData
    aggregation_bits: tuple[bool, ...]  # one bit per validator in the committee
    committee_indices: tuple[int, ...]   # the validator indices of this committee


@dataclass(frozen=True)
class LatestMessage:
    """Port of gloas/fork-choice.md `LatestMessage` (slot, root, payload_present)."""

    slot: int
    root: int
    payload_present: bool


# --- Payload attestation containers (gloas-only) -------------------------
@dataclass(frozen=True)
class PayloadAttestationData:
    """Spec: `class PayloadAttestationData`."""

    beacon_block_root: int
    slot: int
    payload_present: bool
    blob_data_available: bool


@dataclass(frozen=True)
class PayloadAttestation:
    """Spec: `class PayloadAttestation` — aggregated over the PTC."""

    aggregation_bits: tuple[bool, ...]  # one bit per PTC member
    data: PayloadAttestationData
    signature: bytes = b""              # stubbed (no BLS)


@dataclass(frozen=True)
class PayloadAttestationMessage:
    """Spec: `class PayloadAttestationMessage` — single-validator PTC vote."""

    validator_index: int
    data: PayloadAttestationData
    signature: bytes = b""              # stubbed


@dataclass(frozen=True)
class IndexedPayloadAttestation:
    """Spec: `class IndexedPayloadAttestation`."""

    attesting_indices: tuple[int, ...]
    data: PayloadAttestationData
    signature: bytes = b""


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def compute_epoch_at_slot(slot: int) -> int:
    """Port of phase0 `compute_epoch_at_slot`."""
    return slot // config.SLOTS_PER_EPOCH


def get_attesting_indices(att: Attestation) -> list[int]:
    """Return validator indices whose aggregation bit is set.

    Spec analogue: `get_attesting_indices` — Tier 2a slimmed to one committee
    (we don't aggregate cross-committee).
    """
    return [
        att.committee_indices[i]
        for i, bit in enumerate(att.aggregation_bits)
        if bit
    ]


# --------------------------------------------------------------------------
# Beacon-chain `validate_on_attestation`  (gloas/fork-choice.md)
# --------------------------------------------------------------------------
def validate_on_attestation(
    store, attestation: Attestation, *, is_from_block: bool, current_slot: int,
    enforce_ffg_target: bool = False,
) -> None:
    """Port of gloas/fork-choice.md `validate_on_attestation` (slimmed).

    Skipped vs spec by default: `validate_target_epoch_against_current_time`,
    and FFG-target consistency (the latter is enabled by
    ``enforce_ffg_target=True`` once Tier 2b's FFG processor is wired).

    Imports `store` lazily to avoid a circular import (forkchoice.Store).
    """
    target = attestation.data.target

    # epoch / slot agreement
    assert target.epoch == compute_epoch_at_slot(attestation.data.slot), (
        f"target epoch {target.epoch} != epoch_at_slot({attestation.data.slot})"
    )

    # block + target must be known
    assert target.root in store.blocks, "target root unknown"
    assert attestation.data.beacon_block_root in store.blocks, "head root unknown"

    block_slot = store.blocks[attestation.data.beacon_block_root].slot
    assert block_slot <= attestation.data.slot, "attestation for future block"

    # [New in Gloas:EIP7732] — `index` is restricted to {0, 1} and carries
    # FULL/EMPTY semantics, not committee number.
    assert attestation.data.index in (0, 1), (
        f"gloas: data.index must be 0 (EMPTY) or 1 (FULL), got "
        f"{attestation.data.index}"
    )
    if block_slot == attestation.data.slot:
        # same-slot attestations may only vote PENDING (index=0); the block's
        # payload state is not yet known.
        assert attestation.data.index == 0, (
            "same-slot attestation must vote EMPTY/PENDING (index=0)"
        )

    # If attesting FULL, payload must be locally available
    if attestation.data.index == 1:
        from epbs import primitives as p  # local import: same-package helper
        assert p.is_payload_verified(store, attestation.data.beacon_block_root), (
            "FULL attestation but payload not locally verified"
        )

    # Attestations only affect later slots
    assert current_slot >= attestation.data.slot + 1, (
        "attestation must be at least one slot old"
    )

    # [Spec] LMD vote must be consistent with FFG vote target
    if enforce_ffg_target:
        # Lazy import — forkchoice imports attestation, so deferred to avoid cycle.
        from epbs.forkchoice import get_checkpoint_block
        expected = get_checkpoint_block(
            store, attestation.data.beacon_block_root, target.epoch)
        assert target.root == expected, (
            f"FFG target {target.root} inconsistent with ancestor-at-epoch "
            f"({expected}) of head {attestation.data.beacon_block_root}"
        )


# --------------------------------------------------------------------------
# `update_latest_messages`  (gloas/fork-choice.md)
# --------------------------------------------------------------------------
def update_latest_messages(
    store, attesting_indices: list[int], attestation: Attestation
) -> None:
    """Port of gloas/fork-choice.md `update_latest_messages`.

    Key gloas change: `LatestMessage` now stores ``(slot, root, payload_present)``;
    `payload_present` is derived from `attestation.data.index == 1`.
    """
    slot = attestation.data.slot
    root = attestation.data.beacon_block_root
    payload_present = attestation.data.index == 1
    non_equiv = [
        i for i in attesting_indices if i not in store.equivocating_indices
    ]
    # Lazy import to break the forkchoice <-> attestation cycle.
    from epbs.forkchoice import _update_score_index, record_ffg_target_vote
    target = attestation.data.target
    for i in non_equiv:
        prev = store.latest_messages.get(i)
        if prev is None or slot > prev.slot:
            new_msg = LatestMessage(
                slot=slot, root=root, payload_present=payload_present
            )
            store.latest_messages[i] = new_msg
            _update_score_index(store, i, new_msg)
        # Always offer the FFG target vote — record_ffg_target_vote enforces
        # first-write-wins per (vi, epoch). LMD `latest_messages` update can
        # be gated on slot monotonicity; FFG must accept the first vote even
        # if it arrives after a later LMD update on a different branch.
        record_ffg_target_vote(store, i, target.epoch, target.root)


# --------------------------------------------------------------------------
# Payload attestation processing  (gloas/beacon-chain.md)
# --------------------------------------------------------------------------
def get_indexed_payload_attestation(
    ptc: list[int], att: PayloadAttestation
) -> IndexedPayloadAttestation:
    """Port of gloas `get_indexed_payload_attestation`.

    Tier 2a: ``ptc`` is passed in directly (computed by
    ``epbs.committee.get_ptc``) instead of derived from a BeaconState.
    """
    bits = att.aggregation_bits
    attesting = sorted(idx for i, idx in enumerate(ptc) if bits[i])
    return IndexedPayloadAttestation(
        attesting_indices=tuple(attesting),
        data=att.data,
        signature=att.signature,
    )


def is_valid_indexed_payload_attestation(att: IndexedPayloadAttestation) -> bool:
    """Port of gloas `is_valid_indexed_payload_attestation` (BLS stubbed).

    Verifies structural validity only (non-empty + sorted). BLS aggregate
    signature check is stubbed True (see MODEL_SCOPE.md: BLS is out of scope).
    """
    indices = list(att.attesting_indices)
    if not indices:
        return False
    if indices != sorted(indices):
        return False
    return True  # BLS stubbed
