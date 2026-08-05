"""Proposer attestation-inclusion reward — Phase H2 port.

Minimal altair port of the proposer reward earned per included attestation.
Spec sources:
* `https://github.com/ethereum/consensus-specs/blob/master/specs/altair/beacon-chain.md`
  — `PROPOSER_WEIGHT`, `WEIGHT_DENOMINATOR`, `PARTICIPATION_FLAG_WEIGHTS`
* `https://github.com/ethereum/consensus-specs/blob/master/specs/gloas/beacon-chain.md`
  — `process_attestation` (lines 1729-1752), proposer_reward formula

Formula (gloas-modified altair)::

    proposer_reward_numerator += sum(
        get_base_reward(state, vi) * flag_weight
        for vi in attesting_validators
        if a participation flag is newly set
    )
    proposer_reward = numerator // ((WEIGHT_DENOMINATOR - PROPOSER_WEIGHT)
                                    * WEIGHT_DENOMINATOR // PROPOSER_WEIGHT)
                    = numerator // 448

Sim-divergent simplifications (load-bearing for exp11; documented)
------------------------------------------------------------------
1. **Only TIMELY_TARGET flag is rewarded.** Phase E already tracks `(vi, epoch)`
   target votes (`store._ffg_target_seen`). Adding TIMELY_SOURCE and
   TIMELY_HEAD would require two more bit-arrays of (vi, epoch) bookkeeping.
   The collusion experiment compares ``U_collude`` vs ``U_no_collude`` —
   under TARGET-only rewards, FULL and EMPTY committee attestations both
   earn the same proposer reward (target is the canonical ancestor at
   epoch_first_slot, independent of payload status). So this simplification
   means proposer reward CANNOT differentiate honest vs collude blocks via
   the HEAD flag — under this Phase H, collusion only costs the coalition
   the missed builder ``execution_payment`` (the H1 ledger).

   For exp11 to find rational coalition regimes via HEAD-flag differential,
   Phase H+ would add a `_head_flag_seen` set and reward FULL voters more.
   Documented in MODEL_SCOPE.

2. **Constant `base_reward = 14_000 gwei`** instead of altair's exact
   ``effective_balance * BASE_REWARD_FACTOR // integer_squareroot(total_active_balance)``.
   Mainnet base_reward at typical TAB sits in [13_000, 17_000] gwei per epoch,
   so the constant is within ±15% of any real-network value. Absolute reward
   magnitude is NOT load-bearing for the rationality comparison
   ``U_collude > U_no_collude`` — only relative ratios matter, and those
   cancel the constant.

3. **No "newly set" cross-attestation gating.** Spec rewards only the first
   attestation that flips a validator's flag bit. Sim's
   ``store._proposer_reward_credited`` set enforces the same first-time-only
   semantics per (vi, epoch).
"""
from __future__ import annotations

import config


# ---------------------------------------------------------------------------
# Spec constants (altair/beacon-chain.md)
# ---------------------------------------------------------------------------
PROPOSER_WEIGHT: int = 8                       # altair line 88
WEIGHT_DENOMINATOR: int = 64                   # altair line 89
# PARTICIPATION_FLAG_WEIGHTS = [14, 26, 14]    # TIMELY_SOURCE/TARGET/HEAD
TIMELY_SOURCE_WEIGHT: int = 14
TIMELY_TARGET_WEIGHT: int = 26
TIMELY_HEAD_WEIGHT: int = 14

# Denominator: spec formula `(WEIGHT_DENOMINATOR - PROPOSER_WEIGHT) * WEIGHT_DENOMINATOR // PROPOSER_WEIGHT`
# Numerically: (64 - 8) * 64 // 8 = 56 * 8 = 448
PROPOSER_REWARD_DENOMINATOR: int = (
    (WEIGHT_DENOMINATOR - PROPOSER_WEIGHT) * WEIGHT_DENOMINATOR // PROPOSER_WEIGHT
)
assert PROPOSER_REWARD_DENOMINATOR == 448, (
    f"spec exact value is 448; got {PROPOSER_REWARD_DENOMINATOR}"
)

# Sim simplification — constant base_reward per validator per epoch.
# Mainnet range ~13K-17K gwei. See module docstring.
SIM_BASE_REWARD_GWEI: int = 14_000

# Per-attester reward weight = source+target+head; penalty weight = source+target
# only (TIMELY_HEAD has no penalty), per altair get_flag_index_deltas.
ATTESTER_REWARD_WEIGHT: int = (
    TIMELY_SOURCE_WEIGHT + TIMELY_TARGET_WEIGHT + TIMELY_HEAD_WEIGHT  # 54
)
ATTESTER_PENALTY_WEIGHT: int = TIMELY_SOURCE_WEIGHT + TIMELY_TARGET_WEIGHT  # 40


# ---------------------------------------------------------------------------
# Sim API
# ---------------------------------------------------------------------------
def credit_proposer_reward(
    store, proposer_index: int, attesting_validators: list[int],
    attestation_epoch: int,
) -> int:
    """Add to `store.proposer_rewards[proposer_index]` for an included
    attestation. Returns the gwei credited this call (0 if all attesters
    already had their TIMELY_TARGET flag set this epoch).

    Spec analogue: per-attester loop inside gloas `process_attestation`
    lines 1729-1733. Only counts attesters whose flag is newly set —
    enforced via `store._proposer_reward_credited` set.
    """
    # Defensive: Store.__init__ defaults provide these, but stub stores in
    # difftest/test_attestation may lack them.
    if not hasattr(store, "_proposer_reward_credited"):
        store._proposer_reward_credited = set()
    if not hasattr(store, "proposer_rewards"):
        store.proposer_rewards = {}

    numerator = 0
    for vi in attesting_validators:
        key = (vi, attestation_epoch)
        if key in store._proposer_reward_credited:
            continue
        store._proposer_reward_credited.add(key)
        numerator += SIM_BASE_REWARD_GWEI * TIMELY_TARGET_WEIGHT

    reward = numerator // PROPOSER_REWARD_DENOMINATOR
    if reward > 0:
        store.proposer_rewards[proposer_index] = (
            store.proposer_rewards.get(proposer_index, 0) + reward
        )
    return reward


def attester_reward_gwei(base_reward: int = SIM_BASE_REWARD_GWEI) -> int:
    """Full timely-attestation reward (source+target+head) at unit participation.

    Spec: altair ``get_flag_index_deltas`` rewards each set flag by
    ``base_reward * weight * participating_increments // (active_increments *
    WEIGHT_DENOMINATOR)``. The sim approximates the participation ratio as 1 —
    a documented, conservative simplification (it over-states the reward a byz
    committee forgoes by abstaining, so the modelled attack cost is an upper
    bound).
    """
    return base_reward * ATTESTER_REWARD_WEIGHT // WEIGHT_DENOMINATOR


def attester_penalty_gwei(base_reward: int = SIM_BASE_REWARD_GWEI) -> int:
    """Penalty for a missed attestation: source+target flags only (no head
    penalty), per altair ``get_flag_index_deltas``. Not participation-scaled, so
    this is the exact spec value at the sim's constant base reward.
    """
    return base_reward * ATTESTER_PENALTY_WEIGHT // WEIGHT_DENOMINATOR


def credit_attester(store, validator_index: int, *, attested: bool,
                    epoch: int) -> int:
    """Net per-validator attestation reward/penalty into
    ``store.attester_rewards`` — reward if the committee member cast a timely
    attestation, penalty if it abstained (e.g. ``WITHHOLD_VOTE``). Applied once
    per ``(validator_index, epoch)`` (a validator attests once per epoch).
    Returns the signed gwei delta.
    """
    if not hasattr(store, "attester_rewards"):
        store.attester_rewards = {}
    if not hasattr(store, "_attester_credited"):
        store._attester_credited = set()
    key = (validator_index, epoch)
    if key in store._attester_credited:
        return 0
    store._attester_credited.add(key)
    delta = attester_reward_gwei() if attested else -attester_penalty_gwei()
    store.attester_rewards[validator_index] = (
        store.attester_rewards.get(validator_index, 0) + delta
    )
    return delta


# ---------------------------------------------------------------------------
# Reference helpers — exact altair formula (for diff testing)
# ---------------------------------------------------------------------------
def altair_proposer_reward_for_attestation(
    n_attesters: int, base_reward_per_attester: int,
    flag_weight: int = TIMELY_TARGET_WEIGHT,
) -> int:
    """Closed-form altair proposer reward when one attestation contributes
    `n_attesters` newly-set flags of weight `flag_weight`.

    Used by diff tests to confirm the sim's accumulator matches altair's
    arithmetic.
    """
    numerator = n_attesters * base_reward_per_attester * flag_weight
    return numerator // PROPOSER_REWARD_DENOMINATOR
