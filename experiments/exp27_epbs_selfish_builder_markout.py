"""Experiment 27 -- full-ePBS selfish builder markout/free-option probe.

Exp26 found a stylized non-collusive builder exit option.  Exp27 ports the
mechanism one step closer to the full simulator: it uses Tier2Environment's
auction, cover-bid checks, payload reveal path, same-slot builder-payment
quorum, epoch settlement, and telemetry.  The only analysis-layer extension is
an ex-ante/ex-post builder profit process:

* bids are based on noisy ex-ante expected execution profit;
* after winning, the selfish builder observes ex-post execution profit;
* the bid can still settle through ePBS builder_pending_payments;
* withholding only avoids negative execution markout and creates EMPTY payload
  harm.  It is not assumed to evade settlement.

This is a non-collusive incentive-defect probe, not a consensus exploit.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from epbs import builder_payments as bp
from epbs.adversary import BuilderAction, HonestAdversary
from epbs.env_tier2 import Tier2Config, Tier2Environment
from experiments.exp14_common import artifact_metadata, paired_summary
from experiments.exp16_epbs_risk_aware import OUT_DIR


SCHEMA_VERSION = "exp27-epbs-selfish-builder-markout-v1"
SELFISH_BUILDER_INDEX = 0


@dataclass(frozen=True)
class MarkoutParams:
    num_validators: int = 64
    num_builders: int = 8
    slots: int = 96
    committee_size: int = 8
    expected_profit_gwei: float = 3_000_000.0
    estimate_noise_frac: float = 0.20
    markout_sigma_frac: float = 1.15
    honest_bid_fraction: float = 0.82
    selfish_bid_fraction: float = 0.82


@dataclass(frozen=True)
class MarkoutPolicy:
    name: str
    reveal_cutoff_gwei: float
    exit_window: int
    reset_cost_gwei: float


HONEST_POLICY = MarkoutPolicy(
    name="honest_reveal",
    reveal_cutoff_gwei=-math.inf,
    exit_window=0,
    reset_cost_gwei=0.0,
)


def default_policy_grid(*, smoke: bool = False) -> list[MarkoutPolicy]:
    policies = [HONEST_POLICY]
    cutoffs = (-500_000.0, 0.0) if smoke else (-1_000_000.0, -500_000.0, 0.0, 500_000.0)
    exit_windows = (16, 64) if smoke else (8, 16, 32, 96)
    reset_costs = (0.0, 500_000.0) if smoke else (0.0, 250_000.0, 500_000.0, 1_000_000.0)
    for cutoff in cutoffs:
        for exit_window in exit_windows:
            for reset_cost in reset_costs:
                policies.append(MarkoutPolicy(
                    name=f"cut{cutoff:g}_exit{exit_window}_reset{reset_cost:g}",
                    reveal_cutoff_gwei=cutoff,
                    exit_window=exit_window,
                    reset_cost_gwei=reset_cost,
                ))
    return policies


class MarkoutHooks:
    def __init__(self, *, seed: int, params: MarkoutParams, policy: MarkoutPolicy):
        self.seed = seed
        self.params = params
        self.policy = policy

    def estimate_profit_gwei(self, builder_index: int, slot: int) -> float:
        rng = random.Random(self._mix("estimate", builder_index, slot))
        estimate = self.params.expected_profit_gwei * (
            1.0 + rng.gauss(0.0, self.params.estimate_noise_frac)
        )
        return max(0.0, estimate)

    def realised_profit_gwei(self, builder_index: int, slot: int) -> float:
        estimate = self.estimate_profit_gwei(builder_index, slot)
        rng = random.Random(self._mix("markout", builder_index, slot))
        shock = rng.gauss(
            0.0,
            self.params.expected_profit_gwei * self.params.markout_sigma_frac,
        )
        return estimate + shock

    def bid_value_override(self, builder_index: int, slot: int) -> int:
        fraction = (
            self.params.selfish_bid_fraction
            if builder_index == SELFISH_BUILDER_INDEX
            else self.params.honest_bid_fraction
        )
        return max(0, int(self.estimate_profit_gwei(builder_index, slot) * fraction))

    def forced_builder_action(self, builder_index: int, slot: int):
        if builder_index != SELFISH_BUILDER_INDEX:
            return None
        if self.policy is HONEST_POLICY:
            return None
        remaining = self.params.slots - slot + 1
        in_exit_window = remaining <= self.policy.exit_window
        if not in_exit_window:
            return None
        if self.realised_profit_gwei(builder_index, slot) < self.policy.reveal_cutoff_gwei:
            return BuilderAction.WITHHOLD
        return None

    def _mix(self, label: str, builder_index: int, slot: int) -> int:
        label_offset = 17 if label == "estimate" else 37
        return (
            self.seed * 1_000_003
            + builder_index * 10_007
            + slot * 101
            + label_offset
        )


def run_episode(*, seed: int, params: MarkoutParams, policy: MarkoutPolicy) -> dict:
    hooks = MarkoutHooks(seed=seed, params=params, policy=policy)
    cfg = Tier2Config(
        num_validators=params.num_validators,
        num_builders=params.num_builders,
        num_slots=params.slots,
        committee_size=params.committee_size,
        enable_ffg=True,
        byzantine_validators=set(),
        byzantine_builders={SELFISH_BUILDER_INDEX},
        seed=seed.to_bytes(32, "little"),
        bid_value_override_hook=hooks.bid_value_override,
        forced_builder_action_hook=hooks.forced_builder_action,
    )
    env = Tier2Environment(cfg, HonestAdversary())
    env.reset()
    env.begin_episode()
    execution_profit = 0.0
    reset_cost = 0.0
    selfish_wins = 0
    selfish_withholds = 0
    selfish_reveals = 0
    while not env.episode_done():
        event, _stat = env.advance_one_slot()
        if event.winner_builder_index != SELFISH_BUILDER_INDEX:
            continue
        selfish_wins += 1
        realised = hooks.realised_profit_gwei(SELFISH_BUILDER_INDEX, event.slot)
        if event.payload_revealed:
            selfish_reveals += 1
            execution_profit += realised
        else:
            selfish_withholds += 1
            reset_cost += policy.reset_cost_gwei
    result = env.finalise_episode()
    debit = _builder_debit_mark_to_settlement(env, SELFISH_BUILDER_INDEX)
    utility = execution_profit - debit - reset_cost
    slots = max(1, len(result.slot_events))
    return {
        "seed": seed,
        "selfish_utility_gwei": utility,
        "selfish_execution_profit_gwei": execution_profit,
        "selfish_builder_debit_gwei": debit,
        "selfish_reset_cost_gwei": reset_cost,
        "selfish_wins": selfish_wins,
        "selfish_reveals": selfish_reveals,
        "selfish_withholds": selfish_withholds,
        "selfish_withhold_rate": selfish_withholds / selfish_wins if selfish_wins else 0.0,
        "empty_blocks": result.empty_blocks,
        "empty_rate": result.empty_blocks / slots,
        "expired_payments": result.expired_payments,
        "expired_payment_amount_gwei": result.expired_payment_amount_gwei,
        "settled_payment_amount_gwei": result.settled_payment_amount_gwei,
        "pending_payments_at_end": result.pending_payments_at_end,
        "full_but_unsettled_blocks": result.full_but_unsettled_blocks,
    }


def evaluate_policy(
    *,
    policy: MarkoutPolicy,
    params: MarkoutParams,
    seeds: Sequence[int],
    baseline_rows: Sequence[dict] | None = None,
) -> dict:
    rows = [run_episode(seed=seed, params=params, policy=policy) for seed in seeds]
    if baseline_rows is None:
        baseline_rows = [
            run_episode(seed=seed, params=params, policy=HONEST_POLICY)
            for seed in seeds
        ]
    utility_deltas = [
        row["selfish_utility_gwei"] - base["selfish_utility_gwei"]
        for row, base in zip(rows, baseline_rows)
    ]
    empty_deltas = [
        row["empty_blocks"] - base["empty_blocks"]
        for row, base in zip(rows, baseline_rows)
    ]
    debit_deltas = [
        row["selfish_builder_debit_gwei"] - base["selfish_builder_debit_gwei"]
        for row, base in zip(rows, baseline_rows)
    ]
    return {
        "policy": asdict(policy),
        "rows": rows,
        "mean": _mean_row(rows),
        "selfish_utility_delta_summary": paired_summary(utility_deltas),
        "empty_block_delta_summary": paired_summary(empty_deltas),
        "builder_debit_delta_summary": paired_summary(debit_deltas),
        "classification": classify_policy(utility_deltas, empty_deltas),
    }


def classify_policy(
    utility_deltas: Sequence[float],
    empty_deltas: Sequence[float],
) -> str:
    utility = paired_summary(utility_deltas)
    empty = paired_summary(empty_deltas)
    if utility["delta_ci95_low_gwei"] > 0 and empty["delta_ci95_low_gwei"] > 0:
        return "rational-markout-withholding-with-empty-harm"
    if utility["delta_mean_gwei"] > 0 and empty["delta_mean_gwei"] > 0:
        return "positive-mean-markout-withholding"
    if empty["delta_mean_gwei"] > 0:
        return "harmful-but-not-rational"
    return "no-defect"


def run_search(
    *,
    params: MarkoutParams,
    policies: Sequence[MarkoutPolicy],
    seeds: Sequence[int],
) -> dict:
    baseline_rows = [
        run_episode(seed=seed, params=params, policy=HONEST_POLICY)
        for seed in seeds
    ]
    results = [
        evaluate_policy(
            policy=policy,
            params=params,
            seeds=seeds,
            baseline_rows=baseline_rows,
        )
        for policy in policies
    ]
    ranked = sorted(
        results,
        key=lambda result: (
            result["classification"] == "rational-markout-withholding-with-empty-harm",
            result["selfish_utility_delta_summary"]["delta_mean_gwei"],
            result["empty_block_delta_summary"]["delta_mean_gwei"],
        ),
        reverse=True,
    )
    rational = [
        result for result in results
        if result["classification"] == "rational-markout-withholding-with-empty-harm"
    ]
    positive = [
        result for result in results
        if result["classification"] in (
            "rational-markout-withholding-with-empty-harm",
            "positive-mean-markout-withholding",
        )
    ]
    return artifact_metadata(
        schema_version=SCHEMA_VERSION,
        mode="search",
        credibility_label="incentive-model-extension",
        consensus_semantics_changed=False,
        params=asdict(params),
        seeds=list(seeds),
        n_policies=len(policies),
        honest_baseline_mean=_mean_row(baseline_rows),
        results=ranked,
        summary={
            "n_policies": len(policies),
            "n_rational_markout_withholding": len(rational),
            "n_positive_mean_markout_withholding": len(positive),
            "best_classification": ranked[0]["classification"] if ranked else "none",
            "overall_status": (
                "selfish-builder-markout-defect-found"
                if rational
                else "positive-mean-markout-only"
                if positive
                else "no-markout-defect-found"
            ),
        },
    )


def _builder_debit_mark_to_settlement(env: Tier2Environment, builder_index: int) -> float:
    store = env.store
    debit = float(store.builder_debits.get(builder_index, 0))
    quorum = bp.get_quorum_threshold(store.total_active_balance)
    for payment in store.builder_pending_payments.values():
        if payment.builder_index == builder_index and payment.weight >= quorum:
            debit += payment.amount
    return debit


def _mean_row(rows: Sequence[dict]) -> dict[str, float]:
    if not rows:
        return {}
    return {
        key: float(statistics.mean(float(row[key]) for row in rows))
        for key in rows[0]
        if key != "seed"
    }


def _format_m(value: float) -> str:
    return f"{value / 1_000_000.0:.3f}"


def write_markdown(payload: dict, path: Path, *, top_k: int = 12) -> None:
    lines = [
        "# Exp27 Full-ePBS Selfish Builder Markout Probe",
        "",
        f"Schema: `{payload['schema_version']}`",
        f"Credibility label: `{payload['credibility_label']}`",
        f"Consensus semantics changed: `{payload['consensus_semantics_changed']}`",
        "",
        "Summary:",
        "",
        "| key | value |",
        "|---|---:|",
    ]
    for key, value in payload["summary"].items():
        lines.append(f"| `{key}` | {value} |")
    lines.extend([
        "",
        "Top policies:",
        "",
        "| rank | classification | cutoff M | exit window | reset cost M | utility delta M | utility CI low M | empty delta | empty CI low | debit delta M | withhold rate |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for idx, result in enumerate(payload["results"][:top_k], start=1):
        policy = result["policy"]
        utility = result["selfish_utility_delta_summary"]
        empty = result["empty_block_delta_summary"]
        debit = result["builder_debit_delta_summary"]
        lines.append(
            f"| {idx} | {result['classification']} | "
            f"{_format_m(policy['reveal_cutoff_gwei'])} | "
            f"{policy['exit_window']} | "
            f"{_format_m(policy['reset_cost_gwei'])} | "
            f"{_format_m(utility['delta_mean_gwei'])} | "
            f"{_format_m(utility['delta_ci95_low_gwei'])} | "
            f"{empty['delta_mean_gwei']:.3f} | "
            f"{empty['delta_ci95_low_gwei']:.3f} | "
            f"{_format_m(debit['delta_mean_gwei'])} | "
            f"{result['mean']['selfish_withhold_rate']:.3f} |"
        )
    lines.extend([
        "",
        "Interpretation:",
        "",
        "- This uses the full ePBS auction, cover-bid, pending-payment, and settlement path.",
        "- Withholding is not assumed to evade payment; builder debit deltas are reported separately.",
        "- A positive result means ex-post markout avoidance can rationally create EMPTY payload harm even when settlement remains active.",
    ])
    path.write_text("\n".join(lines) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--slots", type=int, default=None)
    parser.add_argument("--num-validators", type=int, default=64)
    parser.add_argument("--num-builders", type=int, default=8)
    parser.add_argument("--committee-size", type=int, default=8)
    parser.add_argument("--expected-profit-gwei", type=float, default=3_000_000.0)
    parser.add_argument("--estimate-noise-frac", type=float, default=0.20)
    parser.add_argument("--markout-sigma-frac", type=float, default=1.15)
    parser.add_argument("--honest-bid-fraction", type=float, default=0.82)
    parser.add_argument("--selfish-bid-fraction", type=float, default=0.82)
    parser.add_argument("--top-policy-only", action="store_true")
    parser.add_argument("--reveal-cutoff-gwei", type=float, default=0.0)
    parser.add_argument("--exit-window", type=int, default=None)
    parser.add_argument("--reset-cost-gwei", type=float, default=0.0)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    if args.seeds is None:
        args.seeds = 8 if args.smoke else 60
    if args.slots is None:
        args.slots = 40 if args.smoke else 96
    if args.exit_window is None:
        args.exit_window = args.slots
    return args


def main() -> None:
    args = _parse_args()
    params = MarkoutParams(
        num_validators=args.num_validators,
        num_builders=args.num_builders,
        slots=args.slots,
        committee_size=args.committee_size,
        expected_profit_gwei=args.expected_profit_gwei,
        estimate_noise_frac=args.estimate_noise_frac,
        markout_sigma_frac=args.markout_sigma_frac,
        honest_bid_fraction=args.honest_bid_fraction,
        selfish_bid_fraction=args.selfish_bid_fraction,
    )
    if args.top_policy_only:
        policies = [
            HONEST_POLICY,
            MarkoutPolicy(
                name=(
                    f"top_cut{args.reveal_cutoff_gwei:g}"
                    f"_exit{args.exit_window}"
                    f"_reset{args.reset_cost_gwei:g}"
                ),
                reveal_cutoff_gwei=args.reveal_cutoff_gwei,
                exit_window=args.exit_window,
                reset_cost_gwei=args.reset_cost_gwei,
            ),
        ]
    else:
        policies = default_policy_grid(smoke=args.smoke)
    payload = run_search(
        params=params,
        policies=policies,
        seeds=list(range(args.seeds)),
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    mode = "smoke" if args.smoke else "full"
    out_base = (
        f"exp27_epbs_selfish_builder_markout_{mode}"
        f"_s{args.seeds}"
        f"_t{params.slots}"
        f"_sigma{params.markout_sigma_frac:g}"
    )
    if params.num_builders != 8:
        out_base += f"_b{params.num_builders}"
    if (
        params.selfish_bid_fraction != 0.82
        or params.honest_bid_fraction != 0.82
    ):
        out_base += (
            f"_sf{params.selfish_bid_fraction:g}"
            f"_hf{params.honest_bid_fraction:g}"
        )
    if args.top_policy_only:
        out_base += (
            f"_top_cut{args.reveal_cutoff_gwei:g}"
            f"_exit{args.exit_window}"
            f"_reset{args.reset_cost_gwei:g}"
        )
    out_json = args.out_dir / f"{out_base}.json"
    out_md = args.out_dir / f"{out_base}.md"
    out_json.write_text(json.dumps(payload, indent=2))
    write_markdown(payload, out_md)
    print(json.dumps(payload["summary"], indent=2))
    best = payload["results"][0]
    print(
        "best="
        f"{best['classification']} "
        f"utility_delta={best['selfish_utility_delta_summary']['delta_mean_gwei']/1e6:.3f}M "
        f"empty_delta={best['empty_block_delta_summary']['delta_mean_gwei']:.3f}"
    )
    print(f"Results -> {out_json}")
    print(f"Summary -> {out_md}")


if __name__ == "__main__":
    main()
