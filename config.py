"""Global configuration for the ePBS Tier 1 simulation.

Timing constants mirror consensus-specs/specs/gloas. The basis-point (BPS)
values are fractions of one slot; exact values must be taken from the pinned
consensus-specs commit at differential-test time (see difftest/).
"""
from __future__ import annotations

from dataclasses import dataclass

# --- slot timing (gloas) ---------------------------------------------------
SECONDS_PER_SLOT: int = 12
MILLIS_PER_SLOT: int = SECONDS_PER_SLOT * 1000

# Basis-point fractions of a slot. Values pinned to consensus-specs gloas
# (eth_consensus_specs.gloas.mainnet config). 10_000 BPS == one full slot.
PAYLOAD_DUE_BPS: int = 7_500              # builder payload-reveal deadline
PAYLOAD_ATTESTATION_DUE_BPS: int = 7_500  # PTC attestation deadline
ATTESTATION_DUE_BPS_GLOAS: int = 2_500    # regular beacon attestation deadline

# PTC
PTC_SIZE: int = 512
# > PAYLOAD_TIMELY_THRESHOLD votes needed for a payload to count as timely.
PAYLOAD_TIMELY_THRESHOLD: int = PTC_SIZE // 2
DATA_AVAILABILITY_TIMELY_THRESHOLD: int = PTC_SIZE // 2

# --- epoch / committee structure (gloas) ----------------------------------
# Mainnet defaults; difftest cross-checks against pyspec minimal preset
# (which uses SLOTS_PER_EPOCH=8, PTC_SIZE=16).
SLOTS_PER_EPOCH: int = 32
MAX_COMMITTEES_PER_SLOT: int = 64
MIN_SEED_LOOKAHEAD: int = 1

# --- fork-choice thresholds (gloas) ---------------------------------------
# `REORG_*_WEIGHT_THRESHOLD` and `PROPOSER_SCORE_BOOST` are used by
# `calculate_committee_fraction(state, pct)` as percentage multipliers of one
# slot's committee weight. Values pinned to pyspec; diff-tested in Tier 2a.
REORG_HEAD_WEIGHT_THRESHOLD: int = 20      # %; weak head iff < 20% of slot committee
REORG_PARENT_WEIGHT_THRESHOLD: int = 160   # %; strong parent iff > 160% of slot committee
PROPOSER_SCORE_BOOST: int = 40             # %; proposer-boost weight


@dataclass(frozen=True)
class SimConfig:
    """Per-experiment knobs for the Tier 1 free-option environment."""

    # external price process
    sigma: float = 0.01            # per-slot volatility of the CEX price
    drift: float = 0.0             # per-slot drift

    # block economics (units: ETH unless noted)
    mu: float = 0.05               # pure on-chain atomic MEV (option-free value)
    dex_liquidity: float = 100.0   # DEX liquidity L
    initial_cex_dex_gap: float = 0.0   # delta: initial CEX-DEX price gap

    # reveal window discretisation (time units within a slot)
    reveal_window_units: int = 8

    # payment model — does a withheld payload still settle the builder's bid?
    #   True  = spec-style `builder_pending_payments`: a withheld payload's bid
    #           never reaches the attestation-weight quorum, so it does NOT
    #           settle (builder pays nothing, proposer receives nothing).
    #   False = sunk-cost model (the Free Option paper's assumption): the bid
    #           is paid unconditionally; withholding still costs the builder
    #           `execution_payment` and the proposer keeps it.
    # See discussion note 2026-05-19_免费期权为何未进ePBS规范.
    conditional_payment: bool = True

    seed: int | None = None


DEFAULT = SimConfig()
