"""Experiment 16 -- ePBS risk-aware utility evaluator.

This is the R1 gate for the DRL incentive-attack exploration branch.  It keeps
the ePBS simulator semantics unchanged, then evaluates policies under an
explicit risk-aware utility wrapper:

    U_risk = U_coalition - risk_costs

The goal is not to claim a new protocol vulnerability.  The goal is to check
whether adding timing pressure makes stationary all-attack suboptimal in one or
two anchor cells, creating a meaningful target for a later PPO/MAPPO learner.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Iterable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from epbs.rl.coalition_env import (
    B_REVEAL_HONEST,
    B_REVEAL_LATE,
    B_REVEAL_WITHHOLD,
    C_HONEST_ATTEST,
    C_VOTE_EMPTY,
    C_WITHHOLD_VOTE,
    H_B_BID,
    H_B_REVEAL,
    H_C,
    H_P,
    H_PTC,
    P_BUILD_ON_NONHEAD,
    P_FORCE_PICK_BYZ_BUILDER,
    P_HONEST_PROPOSE,
    PTC_FRAUD_ABSENT,
    PTC_HONEST,
    CoalitionEnvL,
)
from experiments.exp14_common import (
    Cell,
    _episode_metrics,
    action_key,
    artifact_metadata,
    make_cfg,
    paired_summary,
)
from experiments.exp14_state_conditional import MEV_PARAMS


SCHEMA_VERSION = "exp16-epbs-risk-aware-v1"
OUT_DIR = Path(__file__).resolve().parent / "figures" / "drl_risk_epbs"

HONEST = (
    P_HONEST_PROPOSE,
    0,
    B_REVEAL_HONEST,
    C_HONEST_ATTEST,
    PTC_HONEST,
)

# Paper-2 best spec-feasible stationary template: builder withhold plus
# committee suppression, without simulator-sensitive PTC fraud.
ATTACK_SPEC = (
    P_HONEST_PROPOSE,
    0,
    B_REVEAL_WITHHOLD,
    C_WITHHOLD_VOTE,
    PTC_HONEST,
)


@dataclass(frozen=True)
class RiskParams:
    """Analysis-layer risk parameters, in gwei unless noted otherwise."""

    lambda_streak_gwei: float = 10_000.0
    lambda_reveal_gwei: float = 5_000.0
    lambda_finality_gwei: float = 2_000.0
    lambda_capital_rate: float = 0.0
    finality_committee_fraction_threshold: float = 0.25


@dataclass(frozen=True)
class PolicyContext:
    slot: int
    slot_in_epoch: int
    value_gwei: float
    streak: int
    non_reveals: int
    pending_byz_builder_exposure_gwei: float
    proposer_byz: bool
    byz_builder_would_win: bool
    committee_byz_fraction: float
    ptc_byz_fraction: float


PolicyFn = Callable[[PolicyContext], tuple[int, ...]]


@dataclass(frozen=True)
class PolicySpec:
    name: str
    family: str
    param: str
    policy: PolicyFn
    adaptive: bool


@dataclass
class StepRisk:
    penalty_gwei: float = 0.0
    streak_penalty_gwei: float = 0.0
    reveal_penalty_gwei: float = 0.0
    finality_penalty_gwei: float = 0.0
    capital_penalty_gwei: float = 0.0
    effective_attack: bool = False
    builder_non_reveal: bool = False
    committee_suppression: bool = False
    finality_stress: bool = False


class RiskTracker:
    """Stateful risk-cost accumulator for one episode."""

    def __init__(self, params: RiskParams):
        self.params = params
        self.streak = 0
        self.non_reveals = 0
        self.total_penalty_gwei = 0.0
        self.streak_penalty_gwei = 0.0
        self.reveal_penalty_gwei = 0.0
        self.finality_penalty_gwei = 0.0
        self.capital_penalty_gwei = 0.0
        self.effective_attacks = 0
        self.builder_non_reveals = 0
        self.committee_suppressions = 0
        self.finality_stress_events = 0

    def context(
        self,
        env: CoalitionEnvL,
        roles: dict,
        slot: int,
    ) -> PolicyContext:
        committee = roles.get("committee", [])
        ptc = roles.get("ptc", [])
        committee_frac = _byz_fraction(committee, env.byz_v)
        ptc_frac = _byz_fraction(ptc, env.byz_v)
        return PolicyContext(
            slot=slot,
            slot_in_epoch=slot % config.SLOTS_PER_EPOCH,
            value_gwei=float(roles.get("default_effective_amount", 0.0)),
            streak=self.streak,
            non_reveals=self.non_reveals,
            pending_byz_builder_exposure_gwei=_pending_byz_builder_exposure(env),
            proposer_byz=roles.get("proposer") in env.byz_v,
            byz_builder_would_win=_byz_builder_would_win(env, ATTACK_SPEC, roles),
            committee_byz_fraction=committee_frac,
            ptc_byz_fraction=ptc_frac,
        )

    def apply_pre_step(
        self,
        env: CoalitionEnvL,
        action: Sequence[int],
        roles: dict,
    ) -> StepRisk:
        components = _action_components(
            env,
            action,
            roles,
            finality_fraction_threshold=(
                self.params.finality_committee_fraction_threshold
            ),
        )
        risk = StepRisk(
            effective_attack=components["effective_attack"],
            builder_non_reveal=components["builder_non_reveal"],
            committee_suppression=components["committee_suppression"],
            finality_stress=components["finality_stress"],
        )

        if components["effective_attack"]:
            self.streak += 1
            self.effective_attacks += 1
            risk.streak_penalty_gwei = (
                self.params.lambda_streak_gwei * (self.streak ** 2)
            )
        else:
            self.streak = 0

        if components["builder_non_reveal"]:
            self.non_reveals += 1
            self.builder_non_reveals += 1
            risk.reveal_penalty_gwei = (
                self.params.lambda_reveal_gwei * self.non_reveals
            )

        if components["committee_suppression"]:
            self.committee_suppressions += 1

        if components["finality_stress"]:
            self.finality_stress_events += 1
            risk.finality_penalty_gwei = (
                self.params.lambda_finality_gwei
                * components["committee_byz_fraction"]
            )

        risk.penalty_gwei = (
            risk.streak_penalty_gwei
            + risk.reveal_penalty_gwei
            + risk.finality_penalty_gwei
        )
        self._add(risk)
        return risk

    def apply_post_step(self, env: CoalitionEnvL, risk: StepRisk) -> None:
        exposure = _pending_byz_builder_exposure(env)
        risk.capital_penalty_gwei = self.params.lambda_capital_rate * exposure
        risk.penalty_gwei += risk.capital_penalty_gwei
        self.capital_penalty_gwei += risk.capital_penalty_gwei
        self.total_penalty_gwei += risk.capital_penalty_gwei

    def _add(self, risk: StepRisk) -> None:
        self.total_penalty_gwei += risk.penalty_gwei
        self.streak_penalty_gwei += risk.streak_penalty_gwei
        self.reveal_penalty_gwei += risk.reveal_penalty_gwei
        self.finality_penalty_gwei += risk.finality_penalty_gwei

    def as_dict(self) -> dict:
        return {
            "risk_penalty_gwei": self.total_penalty_gwei,
            "streak_penalty_gwei": self.streak_penalty_gwei,
            "reveal_penalty_gwei": self.reveal_penalty_gwei,
            "finality_penalty_gwei": self.finality_penalty_gwei,
            "capital_penalty_gwei": self.capital_penalty_gwei,
            "effective_attacks": self.effective_attacks,
            "builder_non_reveals": self.builder_non_reveals,
            "committee_suppressions": self.committee_suppressions,
            "finality_stress_events": self.finality_stress_events,
        }


def policy_honest(ctx: PolicyContext) -> tuple[int, ...]:
    return HONEST


def policy_all_attack(ctx: PolicyContext) -> tuple[int, ...]:
    return ATTACK_SPEC


def policy_mev_threshold(theta_gwei: float) -> PolicyFn:
    def _policy(ctx: PolicyContext) -> tuple[int, ...]:
        return ATTACK_SPEC if ctx.value_gwei >= theta_gwei else HONEST

    return _policy


def policy_epoch_suffix(k_slots: int) -> PolicyFn:
    def _policy(ctx: PolicyContext) -> tuple[int, ...]:
        start = config.SLOTS_PER_EPOCH - k_slots
        return ATTACK_SPEC if ctx.slot_in_epoch >= start else HONEST

    return _policy


def policy_risk_threshold(
    *,
    theta_gwei: float,
    max_streak: int,
    max_non_reveals: int,
) -> PolicyFn:
    def _policy(ctx: PolicyContext) -> tuple[int, ...]:
        if ctx.value_gwei < theta_gwei:
            return HONEST
        if ctx.streak > max_streak:
            return HONEST
        if ctx.non_reveals >= max_non_reveals:
            return HONEST
        return ATTACK_SPEC

    return _policy


def make_policy_specs(*, smoke: bool, policy_grid: str = "compact") -> list[PolicySpec]:
    specs = [
        PolicySpec("honest", "stationary", "honest", policy_honest, False),
        PolicySpec(
            "stationary_all_attack",
            "stationary",
            action_key(ATTACK_SPEC),
            policy_all_attack,
            False,
        ),
        PolicySpec(
            "best_spec_feasible_stationary",
            "stationary",
            action_key(ATTACK_SPEC),
            policy_all_attack,
            False,
        ),
    ]

    if smoke or policy_grid == "compact":
        mev_thresholds = [0.0, 1_000_000.0, 1_500_000.0, 2_000_000.0, math.inf]
        epoch_windows = [
            config.SLOTS_PER_EPOCH,
            config.SLOTS_PER_EPOCH // 2,
            config.SLOTS_PER_EPOCH // 4,
            0,
        ]
        risk_grid = [
            (1_000_000.0, 0, 2),
            (1_500_000.0, 0, 2),
            (1_500_000.0, 1, 3),
            (2_000_000.0, 0, 2),
        ]
    elif policy_grid == "broad":
        mev_thresholds = [
            0.0,
            750_000.0,
            1_000_000.0,
            1_250_000.0,
            1_500_000.0,
            2_000_000.0,
            3_000_000.0,
            5_000_000.0,
            math.inf,
        ]
        epoch_windows = [
            config.SLOTS_PER_EPOCH,
            3 * config.SLOTS_PER_EPOCH // 4,
            config.SLOTS_PER_EPOCH // 2,
            config.SLOTS_PER_EPOCH // 4,
            config.SLOTS_PER_EPOCH // 8,
            0,
        ]
        risk_grid = [
            (theta, max_streak, max_non_reveals)
            for theta in (750_000.0, 1_000_000.0, 1_250_000.0, 1_500_000.0, 2_000_000.0)
            for max_streak in (0, 1, 2)
            for max_non_reveals in (2, 4, 8)
        ]
    else:
        raise ValueError(f"unknown policy_grid {policy_grid!r}")

    for theta in mev_thresholds:
        label = "inf" if math.isinf(theta) else f"{theta:.0f}"
        specs.append(PolicySpec(
            f"mev_threshold_{label}",
            "mev_threshold",
            label,
            policy_mev_threshold(theta),
            True,
        ))

    for k in epoch_windows:
        specs.append(PolicySpec(
            f"epoch_suffix_{k}",
            "epoch_suffix",
            str(k),
            policy_epoch_suffix(k),
            True,
        ))

    for theta, max_streak, max_non_reveals in risk_grid:
        label = (
            f"theta={theta:.0f},max_streak={max_streak},"
            f"max_non_reveals={max_non_reveals}"
        )
        specs.append(PolicySpec(
            f"risk_threshold_{theta:.0f}_{max_streak}_{max_non_reveals}",
            "risk_threshold",
            label,
            policy_risk_threshold(
                theta_gwei=theta,
                max_streak=max_streak,
                max_non_reveals=max_non_reveals,
            ),
            True,
        ))
    return specs


def make_risk_cfg(cell: Cell, *, mev: bool):
    cfg = make_cfg(cell)
    if mev:
        cfg = replace(cfg, **MEV_PARAMS)
    return cfg


def run_policy(
    cfg,
    spec: PolicySpec,
    seeds: Sequence[int],
    risk_params: RiskParams,
) -> dict:
    rows = []
    for seed in seeds:
        env = CoalitionEnvL(replace(cfg))
        env.reset(seed=seed)
        tracker = RiskTracker(risk_params)
        done = False
        requested_attacks = 0
        slots = 0
        while not done:
            slot = env.env._cur_slot + 1
            roles = env._peek_next_slot_roles()
            if roles is None:
                action = HONEST
            else:
                ctx = tracker.context(env, roles, slot)
                action = spec.policy(ctx)
            requested_attacks += int(_requested_attack(action))
            slots += 1
            risk = tracker.apply_pre_step(env, action, roles or {})
            _, _, done, _, _ = env.step(action)
            tracker.apply_post_step(env, risk)

        metrics = _episode_metrics(env)
        raw_utility = metrics["coalition_utility_gwei"]
        risk_penalty = tracker.total_penalty_gwei
        rows.append({
            "seed": seed,
            "num_slots": slots,
            "raw_utility_gwei": raw_utility,
            "risk_penalty_gwei": risk_penalty,
            "risk_adjusted_utility_gwei": raw_utility - risk_penalty,
            "requested_attack_rate": requested_attacks / max(1, slots),
            **tracker.as_dict(),
        })
    return {
        "policy": spec.name,
        "family": spec.family,
        "param": spec.param,
        "adaptive": spec.adaptive,
        "rows": rows,
        "mean": _mean_row(rows),
    }


def evaluate_cell(
    cell: Cell,
    *,
    seeds: Sequence[int],
    risk_params: RiskParams,
    mev: bool,
    smoke: bool,
    policy_grid: str,
) -> dict:
    cfg = make_risk_cfg(cell, mev=mev)
    specs = make_policy_specs(smoke=smoke, policy_grid=policy_grid)
    by_policy = {
        spec.name: run_policy(cfg, spec, seeds, risk_params)
        for spec in specs
    }
    honest_rows = by_policy["honest"]["rows"]
    all_attack_rows = by_policy["stationary_all_attack"]["rows"]

    policy_rows = []
    for spec in specs:
        result = by_policy[spec.name]
        deltas = [
            row["risk_adjusted_utility_gwei"]
            - honest["risk_adjusted_utility_gwei"]
            for row, honest in zip(result["rows"], honest_rows)
        ]
        raw_deltas = [
            row["raw_utility_gwei"] - honest["raw_utility_gwei"]
            for row, honest in zip(result["rows"], honest_rows)
        ]
        vs_all = [
            row["risk_adjusted_utility_gwei"]
            - all_attack["risk_adjusted_utility_gwei"]
            for row, all_attack in zip(result["rows"], all_attack_rows)
        ]
        summary = paired_summary(deltas)
        raw_summary = paired_summary(raw_deltas)
        all_summary = paired_summary(vs_all)
        mean = result["mean"]
        policy_rows.append({
            "policy": spec.name,
            "family": spec.family,
            "param": spec.param,
            "adaptive": spec.adaptive,
            "action_key": action_key(ATTACK_SPEC) if spec.name != "honest" else action_key(HONEST),
            "raw_delta_summary": raw_summary,
            "risk_adjusted_delta_summary": summary,
            "paired_vs_all_attack_summary": all_summary,
            "risk_penalty_mean_gwei": mean["risk_penalty_gwei"],
            "requested_attack_rate": mean["requested_attack_rate"],
            "effective_attack_rate": mean["effective_attacks"] / max(1.0, mean["num_slots"]),
            "builder_non_reveals_mean": mean["builder_non_reveals"],
            "committee_suppressions_mean": mean["committee_suppressions"],
            "finality_stress_events_mean": mean["finality_stress_events"],
        })

    attack_row = next(
        row for row in policy_rows if row["policy"] == "stationary_all_attack"
    )
    candidate_rows = [
        row for row in policy_rows
        if row["policy"] != "honest"
    ]
    adaptive_rows = [row for row in candidate_rows if row["adaptive"]]
    best_overall = max(
        candidate_rows,
        key=lambda row: row["risk_adjusted_delta_summary"]["delta_mean_gwei"],
    )
    best_adaptive = max(
        adaptive_rows,
        key=lambda row: row["risk_adjusted_delta_summary"]["delta_mean_gwei"],
    )
    passes_gate = _passes_r1_gate(best_adaptive)
    return {
        "cell": cell.as_dict(),
        "mev": mev,
        "risk_params": asdict(risk_params),
        "n_seeds": len(seeds),
        "seeds": list(seeds),
        "policies": sorted(
            policy_rows,
            key=lambda row: row["risk_adjusted_delta_summary"]["delta_mean_gwei"],
            reverse=True,
        ),
        "stationary_all_attack": attack_row,
        "best_overall_policy": _policy_brief(best_overall),
        "best_adaptive_policy": _policy_brief(best_adaptive),
        "positive_adaptive_beats_all_attack": passes_gate,
        "all_attack_not_optimal": passes_gate,
        "r1_status": (
            "baseline-gate-passed"
            if passes_gate
            else "needs-risk-calibration-or-richer-policy"
        ),
    }


def run_experiment(
    *,
    cells: Sequence[Cell],
    seeds: Sequence[int],
    risk_params: RiskParams,
    mev: bool,
    smoke: bool,
    policy_grid: str,
) -> dict:
    cell_results = [
        evaluate_cell(
            cell,
            seeds=seeds,
            risk_params=risk_params,
            mev=mev,
            smoke=smoke,
            policy_grid=policy_grid,
        )
        for cell in cells
    ]
    passing = sum(
        1 for result in cell_results
        if result["positive_adaptive_beats_all_attack"]
    )
    return artifact_metadata(
        schema_version=SCHEMA_VERSION,
        mode="smoke" if smoke else "full",
        credibility_label="incentive-model-extension",
        consensus_semantics_changed=False,
        risk_params=asdict(risk_params),
        cells=[cell.as_dict() for cell in cells],
        seeds=list(seeds),
        mev=mev,
        policy_grid=policy_grid,
        results=cell_results,
        summary={
            "cells_passing_positive_adaptive_gate": passing,
            "n_cells": len(cell_results),
            "overall_status": (
                "r1-baseline-gate-passed"
                if passing > 0
                else "needs-work"
            ),
            "next_gate": (
                "train PPO/MAPPO only after adaptive handwritten baselines "
                "are fixed and reported"
            ),
        },
    )


def _action_components(
    env: CoalitionEnvL,
    action: Sequence[int],
    roles: dict,
    *,
    finality_fraction_threshold: float | None = None,
) -> dict:
    action = tuple(int(x) for x in action)
    byz_builder_wins = _byz_builder_would_win(env, action, roles)
    committee = roles.get("committee", [])
    ptc = roles.get("ptc", [])
    committee_frac = _byz_fraction(committee, env.byz_v)
    ptc_frac = _byz_fraction(ptc, env.byz_v)

    builder_non_reveal = (
        byz_builder_wins
        and action[H_B_REVEAL] in (B_REVEAL_WITHHOLD, B_REVEAL_LATE)
    )
    committee_suppression = (
        action[H_C] in (C_WITHHOLD_VOTE, C_VOTE_EMPTY)
        and committee_frac > 0.0
    )
    ptc_deviation = (
        action[H_PTC] == PTC_FRAUD_ABSENT
        and ptc_frac > 0.0
    )
    proposer_deviation = (
        action[H_P] in (P_FORCE_PICK_BYZ_BUILDER, P_BUILD_ON_NONHEAD)
        and roles.get("proposer") in env.byz_v
    )
    effective_attack = any([
        builder_non_reveal,
        committee_suppression,
        ptc_deviation,
        proposer_deviation,
    ])
    finality_stress = (
        committee_suppression
        and committee_frac >= (
            RiskParams().finality_committee_fraction_threshold
            if finality_fraction_threshold is None
            else finality_fraction_threshold
        )
    )
    return {
        "byz_builder_wins": byz_builder_wins,
        "builder_non_reveal": builder_non_reveal,
        "committee_suppression": committee_suppression,
        "ptc_deviation": ptc_deviation,
        "proposer_deviation": proposer_deviation,
        "effective_attack": effective_attack,
        "finality_stress": finality_stress,
        "committee_byz_fraction": committee_frac,
    }

def _byz_builder_would_win(
    env: CoalitionEnvL,
    action: Sequence[int],
    roles: dict,
) -> bool:
    action = tuple(int(x) for x in action)
    proposer_byz = roles.get("proposer") in env.byz_v
    force_pick_active = (
        action[H_P] == P_FORCE_PICK_BYZ_BUILDER
        and proposer_byz
        and bool(env.byz_b)
    )
    if force_pick_active:
        return True
    try:
        return bool(env._predict_byz_wins(action[H_B_BID]))
    except Exception:
        default_winner = roles.get("default_winner")
        return bool(getattr(default_winner, "is_byzantine", False))


def _byz_fraction(indices: Iterable[int], byz_v: frozenset[int]) -> float:
    indices = list(indices)
    if not indices:
        return 0.0
    return sum(1 for item in indices if item in byz_v) / len(indices)


def _pending_byz_builder_exposure(env: CoalitionEnvL) -> float:
    store = env.env.store
    return float(sum(
        payment.amount
        for payment in store.builder_pending_payments.values()
        if payment.builder_index in env.byz_b
    ))


def _requested_attack(action: Sequence[int]) -> bool:
    return tuple(int(x) for x in action) != HONEST


def _mean_row(rows: Sequence[dict]) -> dict[str, float]:
    if not rows:
        return {}
    return {
        key: float(statistics.mean(row[key] for row in rows))
        for key in rows[0]
        if key != "seed"
    }


def _policy_brief(row: dict) -> dict:
    return {
        "policy": row["policy"],
        "family": row["family"],
        "param": row["param"],
        "delta_mean_gwei": row["risk_adjusted_delta_summary"]["delta_mean_gwei"],
        "delta_ci95_low_gwei": row["risk_adjusted_delta_summary"]["delta_ci95_low_gwei"],
        "delta_ci95_high_gwei": row["risk_adjusted_delta_summary"]["delta_ci95_high_gwei"],
        "paired_vs_all_attack_ci95_low_gwei": row["paired_vs_all_attack_summary"]["delta_ci95_low_gwei"],
    }


def _passes_r1_gate(row: dict) -> bool:
    return (
        row["risk_adjusted_delta_summary"]["delta_ci95_low_gwei"] > 0
        and row["paired_vs_all_attack_summary"]["delta_ci95_low_gwei"] > 0
    )


def _format_m(value: float) -> str:
    return f"{value / 1_000_000.0:.3f}"


def _write_markdown(payload: dict, out_md: Path) -> None:
    lines = [
        "# Exp16 ePBS Risk-Aware Utility",
        "",
        f"Schema: `{payload['schema_version']}`",
        f"Mode: `{payload['mode']}`",
        f"Credibility label: `{payload['credibility_label']}`",
        f"Consensus semantics changed: `{payload['consensus_semantics_changed']}`",
        "",
        "Risk parameters:",
        "",
        "| parameter | value |",
        "|---|---:|",
    ]
    for key, value in payload["risk_params"].items():
        lines.append(f"| `{key}` | {value} |")
    lines.extend([
        "",
        "Best policies:",
        "",
        "| cell | status | best adaptive | adjusted delta M gwei | CI low M | vs all-attack CI low M |",
        "|---|---|---|---:|---:|---:|",
    ])
    for result in payload["results"]:
        cell = result["cell"]
        best = result["best_adaptive_policy"]
        lines.append(
            f"| ({cell['n_byz_validators']},{cell['n_byz_builders']}) | "
            f"{result['r1_status']} | "
            f"{best['policy']} `{best['param']}` | "
            f"{_format_m(best['delta_mean_gwei'])} | "
            f"{_format_m(best['delta_ci95_low_gwei'])} | "
            f"{_format_m(best['paired_vs_all_attack_ci95_low_gwei'])} |"
        )
    for result in payload["results"]:
        cell = result["cell"]
        lines.extend([
            "",
            f"## Cell ({cell['n_byz_validators']}, {cell['n_byz_builders']})",
            "",
            "| policy | family | param | raw delta M | risk penalty M | adjusted delta M | adjusted CI M | vs all-attack CI M | attack rate |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ])
        for row in result["policies"][:18]:
            summary = row["risk_adjusted_delta_summary"]
            raw = row["raw_delta_summary"]
            vs_all = row["paired_vs_all_attack_summary"]
            lines.append(
                f"| {row['policy']} | {row['family']} | `{row['param']}` | "
                f"{_format_m(raw['delta_mean_gwei'])} | "
                f"{_format_m(row['risk_penalty_mean_gwei'])} | "
                f"{_format_m(summary['delta_mean_gwei'])} | "
                f"[{_format_m(summary['delta_ci95_low_gwei'])}, "
                f"{_format_m(summary['delta_ci95_high_gwei'])}] | "
                f"[{_format_m(vs_all['delta_ci95_low_gwei'])}, "
                f"{_format_m(vs_all['delta_ci95_high_gwei'])}] | "
                f"{row['requested_attack_rate']:.3f} |"
            )
    lines.extend([
        "",
        "Interpretation:",
        "",
        "- This artifact is an incentive-model extension. It does not alter fork choice, builder-payment settlement, FFG, or attestation semantics.",
        "- Passing the R1 gate means at least one hand-written adaptive policy has positive risk-adjusted payoff and beats stationary all-attack under paired CI.",
        "- PPO/MAPPO discovery remains a later gate and must beat these adaptive baselines.",
    ])
    out_md.write_text("\n".join(lines) + "\n")


def _parse_cell(text: str) -> Cell:
    nv, nb = text.split(",")
    return Cell(int(nv), int(nb))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--cell", action="append", default=None,
                        help="Anchor cell as 'n_validators,n_builders'.")
    parser.add_argument("--mev", action="store_true", default=True)
    parser.add_argument("--no-mev", dest="mev", action="store_false")
    parser.add_argument("--lambda-streak-gwei", type=float, default=10_000.0)
    parser.add_argument("--lambda-reveal-gwei", type=float, default=5_000.0)
    parser.add_argument("--lambda-finality-gwei", type=float, default=2_000.0)
    parser.add_argument("--lambda-capital-rate", type=float, default=0.0)
    parser.add_argument("--policy-grid", choices=("compact", "broad"),
                        default="compact")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    if not (args.smoke or args.full):
        args.smoke = True
    if args.seeds is None:
        args.seeds = 8 if args.smoke else 30
    if args.cell is None:
        args.cell = ["12,4"] if args.smoke else ["12,4", "22,4"]
    return args


def main() -> None:
    args = _parse_args()
    risk_params = RiskParams(
        lambda_streak_gwei=args.lambda_streak_gwei,
        lambda_reveal_gwei=args.lambda_reveal_gwei,
        lambda_finality_gwei=args.lambda_finality_gwei,
        lambda_capital_rate=args.lambda_capital_rate,
    )
    cells = [_parse_cell(text) for text in args.cell]
    seeds = list(range(1_600_000, 1_600_000 + args.seeds))
    payload = run_experiment(
        cells=cells,
        seeds=seeds,
        risk_params=risk_params,
        mev=args.mev,
        smoke=args.smoke,
        policy_grid=args.policy_grid,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = "smoke" if args.smoke else "full"
    out_json = args.out_dir / f"exp16_epbs_risk_aware_{tag}.json"
    out_md = args.out_dir / f"exp16_epbs_risk_aware_{tag}.md"
    out_json.write_text(json.dumps(payload, indent=2))
    _write_markdown(payload, out_md)
    print(json.dumps(payload["summary"], indent=2))
    for result in payload["results"]:
        cell = result["cell"]
        best = result["best_adaptive_policy"]
        print(
            f"cell=({cell['n_byz_validators']},{cell['n_byz_builders']}) "
            f"status={result['r1_status']} best={best['policy']} "
            f"delta={best['delta_mean_gwei']/1e6:.3f}M "
            f"vs_all_ci_low={best['paired_vs_all_attack_ci95_low_gwei']/1e6:.3f}M"
        )
    print(f"Results -> {out_json}")
    print(f"Summary -> {out_md}")


if __name__ == "__main__":
    main()
