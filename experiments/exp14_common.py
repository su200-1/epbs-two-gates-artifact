"""Shared Paper 2 evaluation helpers for Experiment 14.

The benchmark enumerates *stationary joint-action templates*: a fixed 5-head
tuple is requested every slot and env preconditions naturally turn inapplicable
primitives into no-ops. This is a complete scan of the 108 stationary tuples,
not a claim of completeness over state-conditional strategies.
"""
from __future__ import annotations

import itertools
import json
import math
import statistics
import subprocess
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

import config
from epbs import builder_payments as bp
from epbs.env_tier2 import Tier2Config
from epbs.rl.coalition_env import (
    CoalitionEnvL, HEAD_SIZES_L, _marked_pending_flows,
)
from epbs.rl.template_metadata import (
    SPEC_FEASIBLE,
    action_names,
    classify_template,
    is_spec_feasible,
    validate_template,
)

SCHEMA_VERSION = "exp14.v1"
N_VALIDATORS = 64
N_BUILDERS = 8
N_SLOTS = 2 * config.SLOTS_PER_EPOCH + 1
COMMITTEE_SIZE = 8
SPEC_SNAPSHOT_PATH = Path(__file__).resolve().parent.parent / "SPEC_SNAPSHOT.json"

HONEST_TEMPLATE = (0, 0, 0, 0, 0)
MAIN_VALIDATOR_COUNTS = (6, 12, 22, 32)
BOUNDARY_VALIDATOR_COUNTS = (21,)
BUILDER_COUNTS = (1, 2, 3, 4)
LEGACY_CELLS = ((6, 1), (6, 4), (12, 1), (12, 4))


@dataclass(frozen=True)
class Cell:
    n_byz_validators: int
    n_byz_builders: int

    @property
    def beta_v(self) -> float:
        return self.n_byz_validators / N_VALIDATORS

    @property
    def beta_b(self) -> float:
        return self.n_byz_builders / N_BUILDERS

    def as_dict(self) -> dict:
        return {
            **asdict(self),
            "beta_v": self.beta_v,
            "beta_b": self.beta_b,
        }


def grid_cells(name: str) -> list[Cell]:
    """Return exact-count cells for a named Paper 2 grid."""
    if name == "legacy4":
        return [Cell(*pair) for pair in LEGACY_CELLS]
    if name == "main16":
        validators = MAIN_VALIDATOR_COUNTS
    elif name == "boundary4":
        validators = BOUNDARY_VALIDATOR_COUNTS
    elif name == "all20":
        validators = MAIN_VALIDATOR_COUNTS + BOUNDARY_VALIDATOR_COUNTS
    else:
        raise ValueError(f"unknown grid {name!r}")
    return [
        Cell(n_byz_validators=v, n_byz_builders=b)
        for v in validators for b in BUILDER_COUNTS
    ]


def make_cfg(cell: Cell, *, enable_ffg: bool = True) -> Tier2Config:
    return Tier2Config(
        num_validators=N_VALIDATORS,
        num_builders=N_BUILDERS,
        num_slots=N_SLOTS,
        committee_size=COMMITTEE_SIZE,
        enable_ffg=enable_ffg,
        byzantine_validators=set(range(cell.n_byz_validators)),
        byzantine_builders=set(range(cell.n_byz_builders)),
    )


def enumerate_templates() -> list[tuple[int, ...]]:
    return list(itertools.product(*(range(size) for size in HEAD_SIZES_L)))


def action_key(action: Sequence[int]) -> str:
    return ":".join(str(value) for value in validate_template(action))


def action_from_key(key: str) -> tuple[int, ...]:
    return validate_template(tuple(int(value) for value in key.split(":")))


def single_head_deviations() -> list[tuple[int, ...]]:
    return [
        action for action in enumerate_templates()
        if sum(value != 0 for value in action) == 1
    ]


def component_templates(action: Sequence[int]) -> list[tuple[int, ...]]:
    """Return one-head-at-a-time deviations that compose ``action``."""
    action = validate_template(action)
    out = []
    for head, value in enumerate(action):
        if value == 0:
            continue
        component = [0] * len(HEAD_SIZES_L)
        component[head] = value
        out.append(tuple(component))
    return out


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def consensus_specs_snapshot() -> dict:
    """Return the reviewed consensus-specs provenance embedded in artifacts."""
    payload = json.loads(SPEC_SNAPSHOT_PATH.read_text())
    return {
        "synced_on": payload["synced_on"],
        **payload["consensus_specs"],
    }


def artifact_metadata(
    *,
    schema_version: str = SCHEMA_VERSION,
    revision: str | None = None,
    **fields,
) -> dict:
    """Return the common provenance header for Paper 2 result artifacts."""
    return {
        "schema_version": schema_version,
        "git_commit": revision if revision is not None else git_commit(),
        "consensus_specs_snapshot": consensus_specs_snapshot(),
        **fields,
    }


def _coalition_utility(env: CoalitionEnvL) -> dict[str, float]:
    store = env.env.store
    proposer_credits = sum(
        store.proposer_credits.get(i, 0) for i in env.byz_v)
    builder_debits = sum(
        store.builder_debits.get(i, 0) for i in env.byz_b)
    proposer_rewards = sum(
        store.proposer_rewards.get(i, 0) for i in env.byz_v)
    # Phase H+ — byz committee attestation reward/penalty (cost of suppression).
    attester_net = sum(
        getattr(store, "attester_rewards", {}).get(i, 0) for i in env.byz_v)
    utility = proposer_credits - builder_debits + proposer_rewards + attester_net
    # Mirror CoalitionEnvL: mark pending-at-quorum payments to settlement so
    # the episode-truncation censoring artifact does not inflate stationary-scan
    # deltas (a withheld bid that reaches quorum WILL settle).
    if getattr(env, "mark_to_settlement", True):
        m_credit, m_debit = _marked_pending_flows(store, env.byz_v, env.byz_b)
        utility += m_credit - m_debit
    return {
        "coalition_utility_gwei": float(utility),
        "coalition_proposer_credits_gwei": float(proposer_credits),
        "coalition_builder_debits_gwei": float(builder_debits),
        "coalition_proposer_rewards_gwei": float(proposer_rewards),
    }


def _validator_payoffs(env: CoalitionEnvL) -> dict[str, float]:
    """BunnyFinder-style per-side validator pay-offs (R_B, R_h).

    Unlike ``coalition_utility`` (which is the byz_validators+byz_builders
    *coalition* profit — a Metric-I objective), these are the **per-side
    validator** net pay-offs in the sense of BunnyFinder (NDSS'26): the sum
    of proposer credits, proposer rewards, and attester reward/penalty over
    the byzantine vs honest validator sets. They are what lets us classify
    an attack's *effect* on the (R_B, R_h) plane instead of only its profit.

    We report both the coalition total and the per-validator average (the
    latter is fair-share comparable — byz and honest sets differ in size).

    ePBS-specific addendum: ``honest_builder_debit_gwei`` is the settlement
    paid out by honest builders. Standard-PoS BunnyFinder has no builder
    layer; in ePBS a Metric-III attack can additionally make honest builders
    pay while their payload is evicted, so we surface this channel too.
    """
    store = env.env.store
    byz_v = env.byz_v
    n_val = env.cfg_template.num_validators
    honest_v = set(range(n_val)) - byz_v
    att = getattr(store, "attester_rewards", {})

    def _vp(indices) -> float:
        return float(sum(
            store.proposer_credits.get(i, 0)
            + store.proposer_rewards.get(i, 0)
            + att.get(i, 0)
            for i in indices
        ))

    rb = _vp(byz_v)
    rh = _vp(honest_v)
    # Mark-to-settlement (must match coalition_utility's accounting): a pending
    # payment already at quorum WILL settle and credit its proposer, so count
    # it now. Without this, an episode-truncated pending credit to an honest
    # proposer is silently dropped, spuriously depressing R_h — the very
    # censoring artifact we removed from coalition utility. Payments below
    # quorum (e.g. genuinely suppressed by a byz committee) expire and are
    # correctly excluded, so a real honest-proposer loss still shows up.
    if getattr(env, "mark_to_settlement", True):
        quorum = bp.get_quorum_threshold(store.total_active_balance)
        for pay in store.builder_pending_payments.values():
            if pay.weight < quorum:
                continue
            if pay.proposer_index in byz_v:
                rb += pay.amount
            elif pay.proposer_index in honest_v:
                rh += pay.amount
    n_byz = max(1, len(byz_v))
    n_honest = max(1, len(honest_v))
    byz_b = env.byz_b
    honest_builder_debit = float(sum(
        store.builder_debits.get(b.index, 0)
        for b in env.env.bset.builders if b.index not in byz_b
    ))
    byz_builder_debit = float(sum(
        store.builder_debits.get(b.index, 0)
        for b in env.env.bset.builders if b.index in byz_b
    ))
    return {
        "byz_validator_payoff_gwei": rb,
        "honest_validator_payoff_gwei": rh,
        "byz_validator_payoff_per_gwei": rb / n_byz,
        "honest_validator_payoff_per_gwei": rh / n_honest,
        "honest_builder_debit_gwei": honest_builder_debit,
        "byz_builder_debit_gwei": byz_builder_debit,
    }


# Threshold (gwei, on coalition-total deltas) below which a pay-off change is
# treated as numerical noise rather than an incentive signal. Per-slot proposer
# rewards are ~26k gwei and the uniform-suppression attester cost is ~535k, so
# 1k gwei cleanly separates signal from float jitter on a 1-seed quick scan.
METRIC_EPS_GWEI = 1_000.0


def classify_metric(
    d_rb: float, d_rh: float, *,
    honest_abs_payoff: float | None = None,
    eps: float = METRIC_EPS_GWEI,
) -> str:
    """Map paired pay-off deltas to a BunnyFinder incentive-flaw metric.

    ``d_rb`` / ``d_rh`` are the byzantine / honest validators' pay-off change
    relative to the honest baseline (which *is* the fair share). We classify
    on fair-share-relative deltas rather than BunnyFinder's absolute-sign test
    (R_B>0, R_h<0): in this sim a validator's *absolute* net pay-off is almost
    never negative (rewards dominate the small attester penalty), so the
    absolute test is insensitive, whereas ``R_h < fair_share`` is exactly
    ``d_rh < 0``. Mapping:

      * Metric III — ``d_rb>0 and d_rh<0``: byz gains while honest is harmed
        (the most dangerous incentive flaw; staircase-II family).
      * Metric II  — ``d_rb<0 and d_rh<0`` with honest in absolute penalty:
        pyrrhic (both lose; here honest absolutely penalised).
      * Metric IV  — ``d_rb<0 and d_rh<0`` otherwise: pyrrhic where honest is
        merely below fair share but still net-positive.
      * Metric I   — ``d_rb>0 and d_rh>=0``: profit without external harm.
      * ``flat``   — no pay-off signal beyond ``eps``.
    """
    byz_gain = d_rb > eps
    byz_loss = d_rb < -eps
    honest_harm = d_rh < -eps
    if byz_gain and honest_harm:
        return "III"
    if byz_loss and honest_harm:
        if honest_abs_payoff is not None and honest_abs_payoff < 0:
            return "II"
        return "IV"
    if byz_gain and not honest_harm:
        return "I"
    return "flat"


# Finality-harm threshold: epochs of failed justification beyond the honest
# baseline above which liveness is treated as attacked rather than noise.
LIVENESS_EPS_EPOCHS = 0.5


def classify_liveness(
    failj_delta: float,
    byz_payoff_ci_low: float,
    byz_payoff_ci_high: float,
    *,
    eps_epochs: float = LIVENESS_EPS_EPOCHS,
) -> str:
    """Classify an attack on the orthogonal *liveness* axis.

    BunnyFinder's (R_B, R_h) plane only captures profit-type harm. A liveness
    attack harms the *system* (finality stalls) rather than any individual
    honest validator's reward, so it needs a separate axis: finality loss
    (``failj_delta`` = extra epochs that failed to justify vs the honest
    baseline) crossed with the byz validators' own payoff change.

      * ``profitable`` — finality destroyed AND byz validators strictly profit
        (payoff CI low > 0): a *free* liveness attack, the dangerous case. Its
        absence is the liveness-dimension safety property, mirroring the
        no-validator-lens-Metric-III result on the profit axis.
      * ``pyrrhic``   — finality destroyed but byz validators strictly net-lose
        (payoff CI high < 0): destroying finality forfeits the attesters' own
        reward, so the attack is self-harming.
      * ``ambiguous`` — finality destroyed, byz payoff CI straddles 0.
      * ``none``      — finality not harmed beyond ``eps_epochs``.
    """
    if failj_delta <= eps_epochs:
        return "none"
    if byz_payoff_ci_low > 0:
        return "profitable"
    if byz_payoff_ci_high < 0:
        return "pyrrhic"
    return "ambiguous"


def _episode_metrics(env: CoalitionEnvL) -> dict[str, float]:
    result = env.env.finalise_episode()
    slots = max(1, len(result.slot_events))
    resolved = result.settled_payments + result.expired_payments
    slot_stats = result.slot_stats
    role_slots = max(1, len(slot_stats))
    committee_total = sum(s.committee_total_weight for s in slot_stats)
    ptc_total = sum(s.ptc_total_weight for s in slot_stats)
    dropped_total = sum(result.dropped_messages.values())
    metrics = {
        **_coalition_utility(env),
        **_validator_payoffs(env),
        "num_slots": float(len(result.slot_events)),
        "empty_blocks": float(result.empty_blocks),
        "empty_rate": result.empty_blocks / slots,
        "num_full": float(result.num_full),
        "full_but_unsettled_blocks": float(result.full_but_unsettled_blocks),
        "full_but_unsettled_rate": result.full_but_unsettled_blocks / slots,
        "settled_payments": float(result.settled_payments),
        "expired_payments": float(result.expired_payments),
        "resolved_payments": float(resolved),
        "expiry_rate": result.expired_payments / resolved if resolved else 0.0,
        "settled_payment_amount_gwei": float(
            result.settled_payment_amount_gwei),
        "expired_payment_amount_gwei": float(
            result.expired_payment_amount_gwei),
        "pending_payments_at_end": float(result.pending_payments_at_end),
        "num_reorgs": float(result.num_reorgs),
        "failed_to_justify_epochs": float(result.num_epochs_failed_to_justify),
        "final_justified_epoch": float(result.final_justified_epoch),
        "final_finalized_epoch": float(result.final_finalized_epoch),
        "dropped_messages": float(dropped_total),
        "clipped_bid_slots": float(sum(
            ev.bid_was_clipped for ev in result.slot_events)),
        "byz_proposer_slot_rate": (
            sum(s.proposer_byz for s in slot_stats) / role_slots),
        "byz_builder_slot_rate": (
            sum(s.builder_byz for s in slot_stats) / role_slots),
        "committee_byz_weight_fraction": (
            sum(s.committee_byz_weight for s in slot_stats) / committee_total
            if committee_total else 0.0),
        "ptc_byz_weight_fraction": (
            sum(s.ptc_byz_weight for s in slot_stats) / ptc_total
            if ptc_total else 0.0),
    }
    return metrics


def run_fixed_episode(
    cfg: Tier2Config, action: Sequence[int], *, seed: int,
) -> dict[str, float]:
    """Run one paired-evaluation episode for a stationary action tuple."""
    env = CoalitionEnvL(replace(cfg))
    obs, _ = env.reset(seed=seed)
    done = False
    while not done:
        obs, _, done, _, _ = env.step(action)
    return _episode_metrics(env)


def mean_metrics(rows: Iterable[dict[str, float]]) -> dict[str, float]:
    rows = list(rows)
    if not rows:
        return {}
    return {
        key: float(statistics.mean(row[key] for row in rows))
        for key in rows[0]
    }


def paired_summary(deltas: Sequence[float]) -> dict[str, float | int]:
    deltas = [float(value) for value in deltas]
    n = len(deltas)
    mean = float(statistics.mean(deltas)) if deltas else 0.0
    std = float(statistics.stdev(deltas)) if n > 1 else 0.0
    half_width = 1.96 * std / math.sqrt(n) if n else 0.0
    return {
        "n_episodes": n,
        "delta_mean_gwei": mean,
        "delta_std_gwei": std,
        "delta_ci95_low_gwei": mean - half_width,
        "delta_ci95_high_gwei": mean + half_width,
        "p_rational": (
            sum(value > 0 for value in deltas) / n if n else 0.0),
    }


def evaluate_template(
    cfg: Tier2Config,
    action: Sequence[int],
    seeds: Sequence[int],
    *,
    honest_by_seed: dict[int, dict[str, float]] | None = None,
) -> dict:
    """Evaluate one stationary template with paired honest seeds."""
    action = validate_template(action)
    if honest_by_seed is None:
        honest_by_seed = {
            seed: run_fixed_episode(cfg, HONEST_TEMPLATE, seed=seed)
            for seed in seeds
        }
    strategy_rows = [
        run_fixed_episode(cfg, action, seed=seed) for seed in seeds
    ]
    honest_rows = [honest_by_seed[seed] for seed in seeds]
    deltas = [
        strategic["coalition_utility_gwei"] - honest["coalition_utility_gwei"]
        for strategic, honest in zip(strategy_rows, honest_rows)
    ]
    # BunnyFinder (R_B, R_h) channel — paired vs the honest baseline so the
    # honest row is the fair-share reference. Drives the Metric classifier,
    # which is orthogonal to ``classification`` (spec-feasibility) above.
    byz_payoff_deltas = [
        s["byz_validator_payoff_gwei"] - h["byz_validator_payoff_gwei"]
        for s, h in zip(strategy_rows, honest_rows)
    ]
    honest_payoff_deltas = [
        s["honest_validator_payoff_gwei"] - h["honest_validator_payoff_gwei"]
        for s, h in zip(strategy_rows, honest_rows)
    ]
    mean_byz_d = statistics.mean(byz_payoff_deltas) if byz_payoff_deltas else 0.0
    mean_honest_d = (
        statistics.mean(honest_payoff_deltas) if honest_payoff_deltas else 0.0)
    honest_abs = (
        statistics.mean(s["honest_validator_payoff_gwei"] for s in strategy_rows)
        if strategy_rows else None)
    # Two metric lenses (they can disagree — that disagreement is itself a
    # finding in ePBS): the validator lens uses the byz *validators'* R_B,
    # the coalition lens uses the byz_v+byz_b coalition profit. An attack can
    # be Metric IV by the validator lens (byz validators net-lose) yet
    # Metric III by the coalition lens (the coalition profits via builder
    # debit-evasion while honest validators are harmed).
    metric = classify_metric(
        mean_byz_d, mean_honest_d, honest_abs_payoff=honest_abs)
    coalition_d = statistics.mean(deltas) if deltas else 0.0
    metric_coalition = classify_metric(
        coalition_d, mean_honest_d, honest_abs_payoff=honest_abs)
    # Byzantine Advantage = R_B - R_h, per-validator, under the strategy.
    ba_values = [
        s["byz_validator_payoff_per_gwei"]
        - s["honest_validator_payoff_per_gwei"]
        for s in strategy_rows
    ]
    byzantine_advantage = statistics.mean(ba_values) if ba_values else 0.0
    classification = classify_template(action)
    return {
        "action": list(action),
        "action_key": action_key(action),
        "action_names": action_names(action),
        "seeds": list(seeds),
        "classification": classification.as_dict(),
        "spec_feasible": classification.label == SPEC_FEASIBLE,
        "paired_deltas_gwei": deltas,
        "summary": paired_summary(deltas),
        "byz_payoff_deltas_gwei": byz_payoff_deltas,
        "honest_payoff_deltas_gwei": honest_payoff_deltas,
        "byz_payoff_summary": paired_summary(byz_payoff_deltas),
        "honest_payoff_summary": paired_summary(honest_payoff_deltas),
        "metric": metric,
        "metric_coalition": metric_coalition,
        "byzantine_advantage_gwei": byzantine_advantage,
        "strategy_metrics": mean_metrics(strategy_rows),
        "honest_metrics": mean_metrics(honest_rows),
    }


def compute_interaction_from_deltas(
    joint_deltas: Sequence[float],
    component_delta_lists: Sequence[Sequence[float]],
) -> dict[str, float | int]:
    """Compute the paired interaction term for one joint template."""
    n = min(
        [len(joint_deltas)] + [len(values) for values in component_delta_lists]
    )
    values = [
        float(joint_deltas[i] - sum(component[i] for component in component_delta_lists))
        for i in range(n)
    ]
    summary = paired_summary(values)
    return {
        "n_episodes": summary["n_episodes"],
        "interaction_mean_gwei": summary["delta_mean_gwei"],
        "interaction_ci95_low_gwei": summary["delta_ci95_low_gwei"],
        "interaction_ci95_high_gwei": summary["delta_ci95_high_gwei"],
    }


def attach_interaction(record: dict, by_key: dict[str, dict]) -> None:
    """Mutate ``record`` to add interaction and combination-advantage data."""
    components = component_templates(record["action"])
    if len(components) < 2:
        record["interaction"] = None
        return
    component_records = [by_key[action_key(action)] for action in components]
    record["interaction"] = compute_interaction_from_deltas(
        record["paired_deltas_gwei"],
        [item["paired_deltas_gwei"] for item in component_records],
    )


def spec_feasible_records(records: Sequence[dict]) -> list[dict]:
    return [record for record in records if record["spec_feasible"]]
