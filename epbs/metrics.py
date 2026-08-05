"""Metric I-V incentive / liveness flaw classifier (the SA component).

Metric I-IV are the BunnyFinder incentive metrics (validator/agent net payoff
vs fair share). Metric V is added by this work for ePBS: liveness / market-
efficiency harm with no payoff redistribution. See the research-plan note
section 3.3.

A strategic run is compared against an honest baseline run; the baseline's mean
payoff IS the fair share.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Metric(Enum):
    """Incentive / liveness flaw categories."""

    I = "I: 策略方与诚实方均获利，诚实方低于公平份额"
    II = "II: 双方均受损"
    III = "III: 策略方获利 + 诚实方受损"
    IV = "IV: 策略方受损 + 诚实方低于公平份额"
    V = "V: 活跃度 / 市场效率受损（空块），无收益再分配"


@dataclass
class AggregateResult:
    """Mean outcomes over many episodes of one run (strategic or honest)."""

    mean_builder_payoff: float
    mean_proposer_payoff: float
    empty_rate: float
    n_episodes: int


@dataclass
class FlawReport:
    """Result of classifying a strategic run against an honest baseline."""

    metrics: list[Metric]
    strategic: AggregateResult
    baseline: AggregateResult
    fair_share_builder: float
    byzantine_advantage: float  # R_strategic - fair share

    @property
    def is_flaw(self) -> bool:
        return len(self.metrics) > 0


def classify(strategic: AggregateResult,
             baseline: AggregateResult,
             eps: float = 1e-9) -> FlawReport:
    """Classify a strategic run vs the honest baseline into Metric I-V.

    The honest baseline's mean builder payoff is the fair share. A run may
    trigger several metrics at once (e.g. V together with III).
    """
    fs_builder = baseline.mean_builder_payoff
    r_strategic = strategic.mean_builder_payoff
    # In the free-option setting the "honest party" is the honest builder, whose
    # realised payoff equals the baseline (= fair share) by construction.
    r_honest = baseline.mean_builder_payoff

    metrics: list[Metric] = []

    # Metric V — liveness: strategic run produces materially more empty blocks.
    if strategic.empty_rate > baseline.empty_rate + eps:
        metrics.append(Metric.V)

    # Metric I-IV — payoff redistribution.
    honest_below_fs = r_honest < fs_builder - eps
    if r_strategic > eps and r_honest > eps and honest_below_fs:
        metrics.append(Metric.I)
    if r_strategic < -eps and r_honest < -eps:
        metrics.append(Metric.II)
    if r_strategic > eps and r_honest < -eps:
        metrics.append(Metric.III)
    if r_strategic < -eps and r_honest > eps and honest_below_fs:
        metrics.append(Metric.IV)

    return FlawReport(
        metrics=metrics,
        strategic=strategic,
        baseline=baseline,
        fair_share_builder=fs_builder,
        byzantine_advantage=r_strategic - fs_builder,
    )
