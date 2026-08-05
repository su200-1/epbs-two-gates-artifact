"""Probabilistic adversary statistics — Tier 2 / Phase G.

Models the two-layer probability structure the consensus literature uses:

  global priors            ->   per-slot realised outcomes
  (β_v, β_b)                    (proposer byz?, committee byz count/weight,
                                 PTC byz count/weight, builder byz?, joint
                                 conditions, ...)

The Byzantine validator and builder *sets* are fixed once per experiment;
randomness comes from the protocol's role-assignment shuffle (and from the
auction tiebreak rotation), so per-slot Byzantine counts inherit a
hypergeometric marginal distribution. This module:

1. Collects per-slot Byzantine counts/weights into `SlotStat` records (see
   `env_tier2._run_one_slot`).
2. Aggregates `SlotStat`s into per-episode and across-episode summaries with
   the tail probabilities that actually drive threshold attacks
   (P(committee byz_weight_frac >= 1/3 at any slot in chain), P(joint
   byz_builder AND byz_proposer), ...).
3. Provides closed-form hypergeometric helpers (`hypergeom_pmf`, `hypergeom_sf`)
   built on `math.comb` so the module imports cleanly in environments without
   scipy.

PTC sampling caveat: spec uses `compute_balance_weighted_selection` with
replacement over the slot's beacon-committee union. The hypergeometric
helpers are exact for `compute_committee` (sampling without replacement
from the active set) but only an approximation for PTC when
``S = |slot committee union|`` is large enough that duplicates are rare and
balances are uniform. Use the Monte-Carlo path (`SlotStat.ptc_byz_count`
aggregated over episodes) for small-N / non-uniform-balance regimes — the
diff test `test_stats.py::test_committee_byz_count_matches_hypergeom`
asserts the analytic-vs-empirical agreement only for the committee case.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable


# ---------------------------------------------------------------------------
# Per-slot record (populated by env_tier2)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SlotStat:
    """Probabilistic state of one slot.

    Counts and weights both recorded so threshold-attack questions can be
    answered in either domain. Under uniform 32-ETH effective balance the
    two are equivalent up to a constant; under future non-uniform balance
    (Electra compounding validators), weight is the load-bearing quantity.
    """
    slot: int
    proposer_index: int
    proposer_byz: bool

    committee_size: int
    committee_byz_count: int
    committee_byz_weight: int          # sum of effective_balance, gwei
    committee_total_weight: int

    ptc_size: int                      # may include duplicates per spec
    ptc_byz_count: int                 # duplicates count multiple times
    ptc_byz_weight: int
    ptc_total_weight: int

    builder_index: int
    builder_byz: bool

    @property
    def committee_byz_weight_fraction(self) -> float:
        if self.committee_total_weight == 0:
            return 0.0
        return self.committee_byz_weight / self.committee_total_weight

    @property
    def ptc_byz_weight_fraction(self) -> float:
        if self.ptc_total_weight == 0:
            return 0.0
        return self.ptc_byz_weight / self.ptc_total_weight

    @property
    def joint_byz_builder_and_proposer(self) -> bool:
        return self.builder_byz and self.proposer_byz

    def joint_byz_builder_and_super_third_committee(
        self, threshold: float = 1 / 3
    ) -> bool:
        return self.builder_byz and (
            self.committee_byz_weight_fraction >= threshold
        )


# ---------------------------------------------------------------------------
# Per-episode rollup
# ---------------------------------------------------------------------------
@dataclass
class EpisodeStats:
    """Per-chain summary of attack-relevant indicators.

    Computed once per episode; `aggregate_episodes` averages these across
    multiple episodes to produce empirical probabilities and expectations.
    """
    num_slots: int
    num_byz_proposer_slots: int
    num_byz_builder_slots: int
    num_committee_super_third_slots: int        # byz weight >= 1/3
    num_committee_super_two_thirds_slots: int   # byz weight >= 2/3 (FFG safety)
    num_ptc_super_half_slots: int               # byz weight >= 1/2 (PAYLOAD_TIMELY_THRESHOLD flip)
    num_joint_builder_proposer_slots: int
    num_joint_builder_super_third_committee_slots: int

    sum_committee_byz_weight: int
    sum_committee_total_weight: int
    sum_ptc_byz_weight: int
    sum_ptc_total_weight: int

    # chain-level indicators (any-slot OR)
    any_byz_proposer: bool
    any_committee_super_third: bool
    any_committee_super_two_thirds: bool
    any_ptc_super_half: bool
    any_joint_builder_proposer: bool
    any_joint_builder_super_third_committee: bool


def summarise_episode(slot_stats: Iterable[SlotStat]) -> EpisodeStats:
    """Roll `SlotStat`s into an `EpisodeStats`."""
    ss = list(slot_stats)
    n = len(ss)
    if n == 0:
        return EpisodeStats(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            False, False, False, False, False, False)
    byz_prop = sum(s.proposer_byz for s in ss)
    byz_build = sum(s.builder_byz for s in ss)
    super_third = sum(s.committee_byz_weight_fraction >= 1 / 3 for s in ss)
    super_two_thirds = sum(s.committee_byz_weight_fraction >= 2 / 3 for s in ss)
    ptc_super_half = sum(s.ptc_byz_weight_fraction >= 0.5 for s in ss)
    joint_bp = sum(s.joint_byz_builder_and_proposer for s in ss)
    joint_bc = sum(
        s.joint_byz_builder_and_super_third_committee() for s in ss
    )
    return EpisodeStats(
        num_slots=n,
        num_byz_proposer_slots=byz_prop,
        num_byz_builder_slots=byz_build,
        num_committee_super_third_slots=super_third,
        num_committee_super_two_thirds_slots=super_two_thirds,
        num_ptc_super_half_slots=ptc_super_half,
        num_joint_builder_proposer_slots=joint_bp,
        num_joint_builder_super_third_committee_slots=joint_bc,
        sum_committee_byz_weight=sum(s.committee_byz_weight for s in ss),
        sum_committee_total_weight=sum(s.committee_total_weight for s in ss),
        sum_ptc_byz_weight=sum(s.ptc_byz_weight for s in ss),
        sum_ptc_total_weight=sum(s.ptc_total_weight for s in ss),
        any_byz_proposer=byz_prop > 0,
        any_committee_super_third=super_third > 0,
        any_committee_super_two_thirds=super_two_thirds > 0,
        any_ptc_super_half=ptc_super_half > 0,
        any_joint_builder_proposer=joint_bp > 0,
        any_joint_builder_super_third_committee=joint_bc > 0,
    )


# ---------------------------------------------------------------------------
# Cross-episode aggregation
# ---------------------------------------------------------------------------
@dataclass
class AggregateStats:
    """Monte-Carlo empirical estimates of attack-relevant probabilities."""
    n_episodes: int

    # P(any slot in chain has condition X). The tail metrics that determine
    # threshold-attack feasibility across a multi-slot horizon.
    p_any_byz_proposer: float
    p_any_committee_super_third: float
    p_any_committee_super_two_thirds: float
    p_any_ptc_super_half: float
    p_any_joint_builder_proposer: float
    p_any_joint_builder_super_third_committee: float

    # E[# slots in chain with condition X]
    e_byz_proposer_slots: float
    e_byz_builder_slots: float
    e_committee_super_third_slots: float
    e_joint_builder_proposer_slots: float

    # E[byz weight fraction per slot] — first moment for sanity checks.
    e_committee_byz_weight_fraction: float
    e_ptc_byz_weight_fraction: float


def aggregate_episodes(episodes: Iterable[EpisodeStats]) -> AggregateStats:
    es = list(episodes)
    n = len(es)
    if n == 0:
        raise ValueError("aggregate_episodes: need ≥1 episode")
    total_slots = sum(e.num_slots for e in es)
    total_cw = sum(e.sum_committee_total_weight for e in es)
    total_pw = sum(e.sum_ptc_total_weight for e in es)
    return AggregateStats(
        n_episodes=n,
        p_any_byz_proposer=sum(e.any_byz_proposer for e in es) / n,
        p_any_committee_super_third=sum(
            e.any_committee_super_third for e in es) / n,
        p_any_committee_super_two_thirds=sum(
            e.any_committee_super_two_thirds for e in es) / n,
        p_any_ptc_super_half=sum(e.any_ptc_super_half for e in es) / n,
        p_any_joint_builder_proposer=sum(
            e.any_joint_builder_proposer for e in es) / n,
        p_any_joint_builder_super_third_committee=sum(
            e.any_joint_builder_super_third_committee for e in es) / n,
        e_byz_proposer_slots=sum(e.num_byz_proposer_slots for e in es) / n,
        e_byz_builder_slots=sum(e.num_byz_builder_slots for e in es) / n,
        e_committee_super_third_slots=sum(
            e.num_committee_super_third_slots for e in es) / n,
        e_joint_builder_proposer_slots=sum(
            e.num_joint_builder_proposer_slots for e in es) / n,
        e_committee_byz_weight_fraction=(
            sum(e.sum_committee_byz_weight for e in es) / total_cw
            if total_cw > 0 else 0.0),
        e_ptc_byz_weight_fraction=(
            sum(e.sum_ptc_byz_weight for e in es) / total_pw
            if total_pw > 0 else 0.0),
    )


# ---------------------------------------------------------------------------
# Closed-form references — hypergeometric distribution
# ---------------------------------------------------------------------------
def hypergeom_pmf(N: int, B: int, n: int, k: int) -> float:
    """P(X = k) where X ~ Hypergeometric(N, B, n).

    N total, B Byzantine, n drawn without replacement, k Byzantine in draw.
    Exact via `math.comb`; no scipy dependency.
    """
    if k < 0 or k > n or k > B or (n - k) > (N - B):
        return 0.0
    return math.comb(B, k) * math.comb(N - B, n - k) / math.comb(N, n)


def hypergeom_sf(N: int, B: int, n: int, k_min: int) -> float:
    """Survival: P(X >= k_min) where X ~ Hypergeometric(N, B, n)."""
    if k_min <= 0:
        return 1.0
    upper = min(n, B)
    return sum(hypergeom_pmf(N, B, n, k) for k in range(k_min, upper + 1))


def p_committee_super_threshold(
    n_validators: int, n_byzantine: int, committee_size: int,
    threshold: float = 1 / 3,
) -> float:
    """P(byzantine fraction in a single drawn committee >= threshold).

    Exact reference for `compute_committee` (sampling without replacement).
    For PTC this is only approximate (balance-weighted with replacement);
    use Monte Carlo for PTC truth.
    """
    k_min = math.ceil(threshold * committee_size)
    return hypergeom_sf(n_validators, n_byzantine, committee_size, k_min)


def p_any_committee_super_threshold_across_chain(
    n_validators: int, n_byzantine: int, committee_size: int,
    n_slots: int, threshold: float = 1 / 3,
) -> float:
    """P(at least one of `n_slots` committees has byz fraction >= threshold).

    Independence approximation: P(none) = (1 - p)^n_slots. Strictly,
    committees within an epoch share the same shuffle so they're negatively
    correlated (drawing a Byzantine into committee A reduces the chance for
    committee B). The independence bound is therefore conservative
    (overestimates the tail).
    """
    p_single = p_committee_super_threshold(
        n_validators, n_byzantine, committee_size, threshold)
    return 1.0 - (1.0 - p_single) ** n_slots


def p_at_least_one_byz_proposer(beta_v: float, n_slots: int) -> float:
    """P(at least one of `n_slots` proposers is Byzantine), given each
    proposer is independently drawn with probability `beta_v`. Tight when
    proposer rotation is balance-weighted with replacement (which is the
    spec for ``compute_proposer_indices``).
    """
    return 1.0 - (1.0 - beta_v) ** n_slots
