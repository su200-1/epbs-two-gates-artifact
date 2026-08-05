"""Gloas `builder_pending_payments` ledger.

Ports the gloas conditional-payment mechanism from
`https://github.com/ethereum/consensus-specs/blob/master/specs/gloas/beacon-chain.md`:

* `process_execution_payload_bid` (lines 1427-1474) — enqueue at slot
* `process_attestation` gloas mod (lines 1737-1745) — accumulate weight from
  same-slot attestations that newly set flags
* `process_builder_pending_payments` (lines 1055-1067) — epoch-boundary settle
* `get_builder_payment_quorum_threshold` (lines 799-806) — 60% of one slot's
  active balance

The settled amount goes **builder → proposer**, indexed by ``bid.value``.
Payment weight comes from same-slot regular beacon attestations that newly
set participation flags. Payload reveal and PTC votes are separate signals:
withholding alone does not automatically avoid payment.

Storage layout
--------------
Spec keeps a `Vector[BuilderPendingPayment, 2 * SLOTS_PER_EPOCH]` indexed by
`SLOTS_PER_EPOCH + (slot % SLOTS_PER_EPOCH)` for the current epoch and
`[0, SLOTS_PER_EPOCH)` for the previous (being settled). At every epoch
boundary the vector shifts left by `SLOTS_PER_EPOCH`.

Sim uses a flat `dict[absolute_slot, PendingPayment]` which is cleaner for
Python; at the epoch boundary we settle all entries with
`slot ∈ [(epoch-1) * SLOTS_PER_EPOCH, epoch * SLOTS_PER_EPOCH)` (the same
"previous epoch" cohort). Behaviourally identical.

Sim still collapses BeaconState builder records and pending withdrawals into a
compact ledger. Builder balances are parameterised, and successful settlement
decreases that balance immediately so future cover-bid checks preserve the
spec's economic constraint.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import config
from epbs.builder import DEFAULT_BUILDER_STAKE_GWEI


# ---------------------------------------------------------------------------
# Spec constants  (gloas/beacon-chain.md lines 152-153)
# ---------------------------------------------------------------------------
BUILDER_PAYMENT_THRESHOLD_NUMERATOR: int = 6
BUILDER_PAYMENT_THRESHOLD_DENOMINATOR: int = 10
MIN_DEPOSIT_AMOUNT_GWEI: int = 1_000_000_000
# Self-build flag (no builder fee, executed by proposer directly).
# Not yet exercised by Tier 2 but reserved for future use.
BUILDER_INDEX_SELF_BUILD: int = (1 << 64) - 1


@dataclass
class PendingPayment:
    """Sim analogue of gloas ``BuilderPendingPayment``.

    Spec packs (weight, BuilderPendingWithdrawal(fee_recipient, amount,
    builder_index)). Sim flattens to a single dataclass since fee_recipient
    is collapsed to builder_index.
    """

    slot: int               # absolute slot the bid was for
    builder_index: int      # gloas builder pool index
    proposer_index: int     # the validator who included the bid (for analytics)
    amount: int             # gwei, the bid's `value`
    weight: int = 0         # accumulated effective_balance of same-slot voters
    attesting_validator_indices: set[int] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Spec ports
# ---------------------------------------------------------------------------
def get_quorum_threshold(total_active_balance: int) -> int:
    """Port of `get_builder_payment_quorum_threshold` (lines 799-806).

    ``(total_active_balance // SLOTS_PER_EPOCH) * 6 // 10`` — 60% of one
    slot's expected committee weight under uniform 32-ETH balances.
    """
    per_slot = total_active_balance // config.SLOTS_PER_EPOCH
    return per_slot * BUILDER_PAYMENT_THRESHOLD_NUMERATOR // BUILDER_PAYMENT_THRESHOLD_DENOMINATOR


def cover_bid_headroom(store, builder_index: int) -> int:
    """Phase L-α — remaining cover-bid headroom for ``builder_index``.

    Spec analogue: `can_builder_cover_bid` (gloas/beacon-chain.md:595-603).
    The sim omits a separate pending-withdrawals queue, but preserves the
    minimum deposit reserve and outstanding pending-payment reservation.
    """
    stake = store.builder_stakes.get(builder_index, DEFAULT_BUILDER_STAKE_GWEI)
    pending = sum(
        p.amount for p in store.builder_pending_payments.values()
        if p.builder_index == builder_index
    )
    return max(0, stake - MIN_DEPOSIT_AMOUNT_GWEI - pending)


def compute_effective_bid_value(
    store, builder_index: int, intended_amount: int,
) -> int:
    """Return the accepted amount, or zero when the spec cover check rejects.

    The returned amount is the **single number** that governs both
    auction ranking and settlement. Binding the two is the whole point
    of the effective-amount unification: a high-rank bid cannot settle a
    different lower amount.

    Auction ranking and settlement use the same accepted amount. Unlike the
    former conservative clipping approximation, a bid above free balance is
    rejected in full, matching `can_builder_cover_bid`.
    """
    if intended_amount <= 0:
        return 0
    if intended_amount > cover_bid_headroom(store, builder_index):
        return 0
    return intended_amount


def enqueue_pending_payment(
    store, slot: int, builder_index: int, proposer_index: int, amount: int,
) -> None:
    """Port of `process_execution_payload_bid` lines 1458-1470 (enqueue path).

    Skipped vs spec: signature verification, builder activeness/funding
    checks, `bid.parent_block_hash == state.latest_block_hash` consistency,
    blob limit. These are all unrelated to the conditional-payment mechanism
    being studied.
    """
    if amount <= 0:
        return
    # Self-builds carry no fee; spec requires amount == 0 for them.
    if builder_index == BUILDER_INDEX_SELF_BUILD:
        return
    store.builder_pending_payments[slot] = PendingPayment(
        slot=slot,
        builder_index=builder_index,
        proposer_index=proposer_index,
        amount=amount,
        weight=0,
    )


def accumulate_same_slot_attestation_weight(
    store, *, voted_block_slot: int, voted_block_root: int,
    attesting_validators: list[int],
) -> None:
    """Port the same-slot regular-attestation builder-payment increment."""
    block = store.blocks.get(voted_block_root)
    if block is None or block.slot != voted_block_slot:
        return
    payment = store.builder_pending_payments.get(voted_block_slot)
    if payment is None or payment.amount == 0:
        return
    for validator_index in attesting_validators:
        if validator_index in payment.attesting_validator_indices:
            continue
        payment.attesting_validator_indices.add(validator_index)
        payment.weight += store.effective_balances.get(validator_index, 0)


def _record_settlement(store, payment: PendingPayment) -> None:
    """Record a successful settlement once and reduce builder free balance."""
    if payment.slot in store.settled_payment_slots:
        return
    store.proposer_credits[payment.proposer_index] = (
        store.proposer_credits.get(payment.proposer_index, 0) + payment.amount
    )
    store.builder_debits[payment.builder_index] = (
        store.builder_debits.get(payment.builder_index, 0) + payment.amount
    )
    balance = store.builder_stakes.get(
        payment.builder_index, DEFAULT_BUILDER_STAKE_GWEI,
    )
    store.builder_stakes[payment.builder_index] = max(0, balance - payment.amount)
    store.settled_payment_slots.append(payment.slot)
    store.settled_payment_amount_gwei += payment.amount


def settle_parent_payment_immediately(
    store, *, slot: int, builder_index: int, proposer_index: int, amount: int,
) -> bool:
    """Settle a FULL parent's payment during child-block processing.

    This mirrors `process_parent_execution_payload`: a child claiming a FULL
    parent settles that parent's builder payment without waiting for its
    epoch-boundary quorum path.
    """
    if amount <= 0 or builder_index == BUILDER_INDEX_SELF_BUILD:
        return False
    payment = store.builder_pending_payments.pop(slot, None)
    if payment is None:
        payment = PendingPayment(
            slot=slot,
            builder_index=builder_index,
            proposer_index=proposer_index,
            amount=amount,
        )
    if slot in store.expired_payment_slots:
        store.expired_payment_slots.remove(slot)
        store.expired_payment_amount_gwei -= payment.amount
    before = len(store.settled_payment_slots)
    _record_settlement(store, payment)
    return len(store.settled_payment_slots) > before


def process_settlement_at_epoch_boundary(
    store, current_epoch: int,
) -> int:
    """Port of `process_builder_pending_payments` (lines 1055-1067).

    At end of epoch E, settle all payments belonging to epoch E-1 whose
    weight reached quorum. **Payment direction is builder → proposer**
    (per gloas/builder.md:135 — "Set `bid.value` to be the value (in gwei)
    that the builder will pay the proposer if the bid is accepted").
    Spec analogue: `state.builders[builder_index].balance -= amount` and
    the `BuilderPendingWithdrawal` enters the withdrawal queue with the
    proposer's fee_recipient. Sim records both sides separately:

      * ``store.proposer_credits[proposer_index]`` — gwei *received* by
        that proposer
      * ``store.builder_debits[builder_index]`` — gwei *paid out* by that
        builder

    Returns the count of payments settled. Unsettled payments from epoch
    E-1 are dropped (spec equivalent: vector shift overwrites them).
    """
    if current_epoch <= 0:
        return 0
    quorum = get_quorum_threshold(store.total_active_balance)
    prev_epoch_first_slot = (current_epoch - 1) * config.SLOTS_PER_EPOCH
    prev_epoch_first_excluded = current_epoch * config.SLOTS_PER_EPOCH

    settled_count = 0
    expired_slots = [
        slot for slot in store.builder_pending_payments
        if prev_epoch_first_slot <= slot < prev_epoch_first_excluded
    ]
    for slot in expired_slots:
        payment = store.builder_pending_payments.pop(slot)
        if payment.weight >= quorum:
            _record_settlement(store, payment)
            settled_count += 1
        else:
            store.expired_payment_slots.append(slot)
            store.expired_payment_amount_gwei += payment.amount
            # Expired payments receive no credit/debit. This is the
            # free-option path: the Byzantine builder avoided paying
            # bid.value.
    return settled_count
