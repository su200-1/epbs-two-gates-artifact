"""Builder pool — independent of the validator set.

Gloas separates the builder role (submits ExecutionPayloadBid, reveals payload)
from the validator role (proposer, attester, PTC member). In Tier 2a/2b the
two were fused (`builder_index = proposer_index`), which collapsed any
"Byzantine builder, honest proposer" scenario into "Byzantine proposer".
This module restores the separation.

Tier 2c/2-fix scope:
- Builders form their own index space ``[0, num_builders)`` distinct from
  validator indices.
- Each slot every builder submits a bid; the proposer picks the highest-value
  bid (ties broken by lowest builder_index for determinism).
- Defaults all builders to ``value=DEFAULT_BID_VALUE``; byzantine builders bid
  ``DEFAULT_BID_VALUE + 1`` so they win when present (worst-case-for-network
  modelling). Tune via ``Builder.bid_value`` if a softer adversary is wanted.

The compact simulator models pending payments and parameterised builder
balances in ``epbs.builder_payments``. It does not model builder slashing or
the full BeaconState withdrawal queue.
"""
from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_BID_VALUE: int = 1000
# Opt-in: byzantine bid premium. Default 0 so byz win-rate equals byz pool
# share (slot-rotation tiebreak), matching the semantic "byz_builder_frac =
# fraction of blocks built by byzantine". Override via
# ``make_builder_set(..., byzantine_bid_premium=N)`` to model an attacker who
# outbids honest builders to capture more slots than their pool share.
BYZANTINE_BID_PREMIUM: int = 0

# Phase L-α: builders' staked balance cap (gwei). Spec's
# `can_builder_cover_bid` (gloas/beacon-chain.md:589-599) rejects any bid
# whose `value` exceeds the builder's free balance after reserving
# MIN_DEPOSIT_AMOUNT and pending settlements. Sim does not model real
# BeaconState balances, so we
# parameterise stake as a single constant (32 ETH ~ one validator deposit
# equivalent). The quantity is a sim parameter rather than BeaconState input.
DEFAULT_BUILDER_STAKE_GWEI: int = 32_000_000_000  # 32 ETH


@dataclass
class Builder:
    """Minimal builder record."""

    index: int
    is_byzantine: bool = False
    bid_value: int = DEFAULT_BID_VALUE


@dataclass
class BuilderSet:
    """Pool of builders. Indexing is independent of the validator set."""

    builders: list[Builder] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.builders)

    def byzantine_indices(self) -> set[int]:
        return {b.index for b in self.builders if b.is_byzantine}


def make_builder_set(
    n: int, *,
    byzantine_indices: set[int] | None = None,
    byzantine_bid_premium: int = BYZANTINE_BID_PREMIUM,
) -> BuilderSet:
    """Build a uniform builder pool of size ``n``.

    With default ``byzantine_bid_premium=0`` all bids tie at
    ``DEFAULT_BID_VALUE`` and the auction rotates by slot, so the byzantine
    win-rate equals ``len(byzantine_indices) / n`` (matches the fork-choice-
    weight intuition of "Byzantine builder share"). Override to model an
    over-paying attacker.
    """
    byz = byzantine_indices or set()
    return BuilderSet(builders=[
        Builder(
            index=i,
            is_byzantine=(i in byz),
            bid_value=DEFAULT_BID_VALUE + (byzantine_bid_premium if i in byz else 0),
        )
        for i in range(n)
    ])


def run_auction(
    builders: BuilderSet, slot: int,
    effective_amounts: dict[int, int] | None = None,
) -> tuple[Builder, int] | None:
    """Deterministic max-effective-amount auction.

    Returns ``(winner, effective_amount)`` for bids that passed the cover-bid
    check. The committed amount settles either through FULL-parent processing
    or the regular-attestation epoch-boundary quorum. Returns ``None`` if
    every external bid was rejected or zero.

    Phase L-α: Auction ranking and settlement amount are bound to a
    single number (``effective_amount``). This closes the "high rank, low
    payment" sim-only escape that would exist if auction used
    ``Builder.bid_value`` while settlement used a different ``intended``
    value.

    Tiebreak: lowest ``Builder.index``; if multiple builders share the max
    effective amount and the same index sorting, slot is used to rotate
    the winner across the tied set so honest baselines are balanced.

    Backward compatibility: when ``effective_amounts is None`` we fall
    back to ``Builder.bid_value`` (the legacy single-source-of-truth) and
    pre-Phase-L callers see the same selection. Empty-pool still raises
    ValueError (distinct from "pool present but every member effective=0").
    """
    if not builders.builders:
        raise ValueError("empty builder pool")
    if effective_amounts is None:
        effective_amounts = {b.index: b.bid_value for b in builders.builders}
    eligible = [
        b for b in builders.builders
        if effective_amounts.get(b.index, 0) > 0
    ]
    if not eligible:
        return None
    max_amount = max(effective_amounts[b.index] for b in eligible)
    top = [b for b in eligible if effective_amounts[b.index] == max_amount]
    if len(top) == 1:
        return (top[0], max_amount)
    # All-tied case: rotate by slot so honest baselines are balanced.
    return (top[slot % len(top)], max_amount)
