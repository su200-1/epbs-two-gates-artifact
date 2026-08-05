"""Casper FFG — ported from `phase0/beacon-chain.md` + `altair/beacon-chain.md`.

This is the Tier 2b addition. Tier 2a left ``justified_checkpoint`` static;
Tier 2b runs ``process_justification_and_finalization`` at every epoch
boundary, advancing the justified/finalized checkpoints based on accumulated
attestation participation.

**Key simplification vs spec**: the spec routes target-balance computation
through ``BeaconState`` + ``get_unslashed_participating_indices``. Tier 2b
derives participation directly from ``store.latest_messages``: a validator
participated in epoch E with target T iff their latest attestation has
``data.slot // SLOTS_PER_EPOCH == E and data.target.root == T``. This is
spec-equivalent under the Tier 2 assumption that no per-flag participation
flags exist (no rewards/penalties model).

**Unrealized checkpoints**: spec maintains ``unrealized_justified_checkpoint``
via ``compute_pulled_up_tip`` (per-block speculation). Tier 2b skips this and
realises checkpoints at epoch boundaries only. This is the minimum needed for
staircase / justification-withholding attacks; if a future attack candidate
relies on per-block unrealized justification it must be added here.

Spec sources:
    altair/beacon-chain.md::weigh_justification_and_finalization,
        process_justification_and_finalization
    phase0/fork-choice.md::update_checkpoints, get_checkpoint_block,
        filter_block_tree
"""
from __future__ import annotations

import config
from epbs import attestation as A
from epbs import forkchoice as F
from epbs.forkchoice import (
    GENESIS_EPOCH, JUSTIFICATION_BITS_LENGTH, Store,
)


# --------------------------------------------------------------------------
# Target balance derivation (replaces BeaconState participation-flag path)
# --------------------------------------------------------------------------
def compute_target_attestation_balance(store: Store, target_epoch: int) -> int:
    """Sum effective balance of validators whose latest attestation had
    ``data.slot // SLOTS_PER_EPOCH == target_epoch`` and a target that resolves
    to the epoch-first-slot ancestor of `data.beacon_block_root`.

    Tier 2b derives this directly from ``latest_messages`` rather than
    altair's ``state.current_epoch_participation`` (no participation-flag
    bookkeeping).
    """
    total = 0
    for vi, msg in store.latest_messages.items():
        if vi in store.equivocating_indices:
            continue
        # The attestation's slot determines which epoch it counts for
        msg_epoch = msg.slot // config.SLOTS_PER_EPOCH
        if msg_epoch != target_epoch:
            continue
        # The vote's target == ancestor of msg.root at epoch_first_slot
        epoch_first_slot = target_epoch * config.SLOTS_PER_EPOCH
        if msg.root not in store.blocks:
            continue
        try:
            ancestor_root = F.get_ancestor(store, msg.root, epoch_first_slot).root
        except (KeyError, RecursionError):
            continue
        # The target the vote *would* have used is this ancestor — and only
        # votes whose target == justified ancestor of current store state
        # count. In Tier 2b we count any vote whose ancestor at epoch_first_slot
        # exists; the FFG processor compares against the canonical target.
        eff_bal = store.effective_balances.get(vi, 0)
        total += eff_bal
    return total


def get_target_root_for_epoch(store: Store, epoch: int) -> int:
    """Compute the canonical target root for ``epoch`` from the current head.

    Phase0 spec: ``get_block_root(state, epoch)`` returns the block root at
    the first slot of ``epoch`` from the post-state of the current block.
    Tier 2b: from the current canonical head, walk to the first slot of
    ``epoch``.
    """
    head = F.get_head(store)
    epoch_first_slot = epoch * config.SLOTS_PER_EPOCH
    return F.get_ancestor(store, head.root, epoch_first_slot).root


def compute_target_balance_for_root(
    store: Store, target_epoch: int, target_root: int
) -> int:
    """Sum effective balance of validators whose first attestation in
    ``target_epoch`` voted target=target_root.

    Backed by the explicit `store.epoch_target_balance` table maintained by
    `record_ffg_target_vote` (per-`(vi, epoch)` lock-in). This replaces the
    old `latest_messages`-derived walk, which silently lost fork-A votes
    when a validator's latest LMD message rotated to fork B — making forked-
    chain FFG amplification invisible. The new table preserves *all*
    target tallies, one per (epoch, target_root) bucket.
    """
    return store.epoch_target_balance.get((target_epoch, target_root), 0)


# --------------------------------------------------------------------------
# weigh_justification_and_finalization  (altair/beacon-chain.md)
# --------------------------------------------------------------------------
def weigh_justification_and_finalization(
    store: Store,
    total_active_balance: int,
    previous_epoch_target_balance: int,
    current_epoch_target_balance: int,
    previous_epoch: int,
    current_epoch: int,
) -> None:
    """Port of altair `weigh_justification_and_finalization`.

    Mutates ``store.{justification_bits, current/previous_justified_checkpoint,
    finalized_checkpoint}`` in place.

    Threshold: ``target_balance * 3 >= total_active_balance * 2`` (i.e. ≥2/3).
    Finalisation rules are the four `bits[...]` patterns in altair.
    """
    old_previous_justified = store.previous_justified_checkpoint
    old_current_justified = store.current_justified_checkpoint

    # Roll the justification bits forward (bit 0 = current epoch)
    store.previous_justified_checkpoint = store.current_justified_checkpoint
    bits = store.justification_bits
    # shift bits[1:4] = bits[0:3], then set bits[0] = 0
    for i in range(JUSTIFICATION_BITS_LENGTH - 1, 0, -1):
        bits[i] = bits[i - 1]
    bits[0] = False

    # Previous-epoch justification
    if previous_epoch_target_balance * 3 >= total_active_balance * 2:
        prev_root = get_target_root_for_epoch(store, previous_epoch)
        store.current_justified_checkpoint = A.Checkpoint(
            epoch=previous_epoch, root=prev_root)
        bits[1] = True

    # Current-epoch justification
    if current_epoch_target_balance * 3 >= total_active_balance * 2:
        cur_root = get_target_root_for_epoch(store, current_epoch)
        store.current_justified_checkpoint = A.Checkpoint(
            epoch=current_epoch, root=cur_root)
        bits[0] = True

    # Finalisation — the four altair patterns
    # 2/3/4 justified, 4 as source
    if (all(bits[1:4]) and old_previous_justified is not None
            and old_previous_justified.epoch + 3 == current_epoch):
        store.finalized_checkpoint = old_previous_justified
    # 2/3 justified, 3 as source
    if (all(bits[1:3]) and old_previous_justified is not None
            and old_previous_justified.epoch + 2 == current_epoch):
        store.finalized_checkpoint = old_previous_justified
    # 1/2/3 justified, 3 as source
    if (all(bits[0:3]) and old_current_justified is not None
            and old_current_justified.epoch + 2 == current_epoch):
        store.finalized_checkpoint = old_current_justified
    # 1/2 justified, 2 as source
    if (all(bits[0:2]) and old_current_justified is not None
            and old_current_justified.epoch + 1 == current_epoch):
        store.finalized_checkpoint = old_current_justified

    # Keep legacy aliases in sync (Tier 2a callers read justified_root)
    if store.current_justified_checkpoint is not None:
        store.justified_root = store.current_justified_checkpoint.root
    if store.finalized_checkpoint is not None:
        store.finalized_root = store.finalized_checkpoint.root


def process_justification_and_finalization(store: Store) -> None:
    """Port of altair `process_justification_and_finalization`.

    Called at the end of every epoch by the env's slot loop. Skipped for the
    first two epochs (spec convention — avoids corner cases with the genesis
    checkpoint stub).
    """
    current_slot = F.get_current_slot(store)
    current_epoch = current_slot // config.SLOTS_PER_EPOCH

    if current_epoch <= GENESIS_EPOCH + 1:
        store.last_ffg_processed_epoch = current_epoch
        return

    previous_epoch = current_epoch - 1
    prev_target_root = get_target_root_for_epoch(store, previous_epoch)
    cur_target_root = get_target_root_for_epoch(store, current_epoch)

    previous_target_balance = compute_target_balance_for_root(
        store, previous_epoch, prev_target_root)
    current_target_balance = compute_target_balance_for_root(
        store, current_epoch, cur_target_root)

    weigh_justification_and_finalization(
        store,
        total_active_balance=store.total_active_balance,
        previous_epoch_target_balance=previous_target_balance,
        current_epoch_target_balance=current_target_balance,
        previous_epoch=previous_epoch,
        current_epoch=current_epoch,
    )
    store.last_ffg_processed_epoch = current_epoch


# --------------------------------------------------------------------------
# Public checkpoint helpers used by `validate_on_attestation`
# --------------------------------------------------------------------------
def update_checkpoints(
    store: Store,
    justified_checkpoint: A.Checkpoint,
    finalized_checkpoint: A.Checkpoint,
) -> None:
    """Port of phase0 `update_checkpoints` (monotonic advance).

    Intended for use by on_block when the new block carries already-realized
    justifications (Tier 2b doesn't use this path; kept for spec parity).
    """
    if (store.current_justified_checkpoint is None
            or justified_checkpoint.epoch > store.current_justified_checkpoint.epoch):
        store.current_justified_checkpoint = justified_checkpoint
        store.justified_root = justified_checkpoint.root
    if (store.finalized_checkpoint is None
            or finalized_checkpoint.epoch > store.finalized_checkpoint.epoch):
        store.finalized_checkpoint = finalized_checkpoint
        store.finalized_root = finalized_checkpoint.root
