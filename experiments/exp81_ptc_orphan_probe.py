"""Experiment 81 -- PTC-orphan decoupling probe (D1 make-or-break).

The one structural exception to non-collusive static-optimality is the free
option, which the regular-committee PAYMENT quorum (60% of a slot's committee,
scale-varying) gates -> Paper 2's ~40%-of-stake bound. But gloas ALSO has a
separate, FIXED-SIZE gate: the Payload Timeliness Committee (PTC_SIZE=512,
PAYLOAD_TIMELY_THRESHOLD=256). `payload_timeliness` (PTC-gated) drives
`should_extend_payload` / `should_build_on_full` -> whether a revealed payload
is accepted as FULL or ORPHANED. This is DECOUPLED from the payment path.

Attack premise (D1): a coalition controlling the PTC can force an honest
builder's timely payload to be declared not-timely -> orphan it -> the next
proposer re-captures the now-public MEV without paying, while the victim builder
STILL PAYS when the regular quorum is met. The protocol pivotal count is 257
PTC seats; its market price is swept separately.

This probe validates the load-bearing fact BEFORE any oracle/DRL:
    forcing PTC not-timely on a revealed payload =>
      (a) is the payload orphaned (head EMPTY despite reveal)?
      (b) does the builder still get debited (pays anyway)?

If (a) & (b) hold, the reveal-then-orphan / MEV-steal surface is real and
scale-robust. If not (e.g., proposer boost protects the payload, or orphaning
also cancels the payment), D1 is weaker than hoped.

Run: python experiments/exp81_ptc_orphan_probe.py [--seeds 12] [--slots 64]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from epbs import builder_payments as bp  # noqa: E402
from epbs.adversary import HonestAdversary  # noqa: E402
from epbs.env_tier2 import Tier2Config, Tier2Environment  # noqa: E402
from epbs.forkchoice import PayloadStatus  # noqa: E402
from experiments.exp27_epbs_selfish_builder_markout import (  # noqa: E402
    MarkoutParams, MarkoutHooks, HONEST_POLICY, SELFISH_BUILDER_INDEX,
    _builder_debit_mark_to_settlement,
)

OUT_DIR = Path(__file__).resolve().parent / "figures" / "drl_risk_epbs"


def _run(seed, params, attack: bool, ptc_false_fraction: float) -> dict:
    hooks = MarkoutHooks(seed=seed, params=params, policy=HONEST_POLICY)  # always reveal

    def ptc_vote_hook(ptc_idx, slot, local_view_present):
        # Byzantine PTC: on a revealed payload, a fraction of PTC members vote
        # "not present" to push it below the timely threshold. Deterministic by
        # index so a fixed fraction is controlled.
        if not attack or not local_view_present:
            return None
        # control the lowest `ptc_false_fraction` share of PTC seats -> vote False
        h = (ptc_idx * 2654435761) & 0xFFFFFFFF
        if (h / 0xFFFFFFFF) < ptc_false_fraction:
            return False
        return None

    cfg = Tier2Config(
        num_validators=params.num_validators, num_builders=params.num_builders,
        num_slots=params.slots, committee_size=params.committee_size, enable_ffg=True,
        byzantine_validators=set(), byzantine_builders={SELFISH_BUILDER_INDEX},
        seed=seed.to_bytes(32, "little"),
        bid_value_override_hook=hooks.bid_value_override,
        forced_builder_action_hook=hooks.forced_builder_action,
        ptc_vote_hook=ptc_vote_hook,
    )
    env = Tier2Environment(cfg, HonestAdversary())
    env.reset()
    env.begin_episode()
    events = []
    while not env.episode_done():
        event, _ = env.advance_one_slot()
        events.append(event)
    result = env.finalise_episode()
    debit = _builder_debit_mark_to_settlement(env, SELFISH_BUILDER_INDEX)

    # orphan = selfish revealed at slot t, but slot t+1 built on it as EMPTY
    selfish_wins = revealed_wins = evaluable_reveals = 0
    orphaned_reveals = right_censored_reveals = 0
    by_parent = {ev.parent_root: ev for ev in events}
    for ev in events:
        if ev.winner_builder_index != SELFISH_BUILDER_INDEX:
            continue
        selfish_wins += 1
        if not ev.payload_revealed:
            continue
        revealed_wins += 1
        child = by_parent.get(ev.block_root)
        if child is None:
            # A reveal in the final simulated slot has no observed child. It is
            # right-censored, not evidence that the payload survived the attack.
            right_censored_reveals += 1
            continue
        evaluable_reveals += 1
        if child.parent_status == PayloadStatus.EMPTY:
            orphaned_reveals += 1
    return {
        "selfish_wins": selfish_wins,
        "revealed_wins": revealed_wins,
        "evaluable_reveals": evaluable_reveals,
        "orphaned_reveals": orphaned_reveals,
        "not_orphaned_reveals": evaluable_reveals - orphaned_reveals,
        "right_censored_reveals": right_censored_reveals,
        "orphan_rate": orphaned_reveals / evaluable_reveals if evaluable_reveals else 0.0,
        "debit_gwei": debit,
        "empty_blocks": result.empty_blocks,
        "full_but_unsettled_blocks": getattr(result, "full_but_unsettled_blocks", 0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--slots", type=int, default=64)
    ap.add_argument("--num-builders", type=int, default=8)
    ap.add_argument("--expected-profit-gwei", type=float, default=30_000_000.0)
    ap.add_argument("--ptc-false-fractions", type=str, default="0.5,0.6,0.75,1.0")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    params = MarkoutParams(
        num_builders=args.num_builders, slots=args.slots,
        expected_profit_gwei=args.expected_profit_gwei,
    )
    seeds = list(range(3000, 3000 + args.seeds))

    def pooled(runs):
        revealed = sum(r["revealed_wins"] for r in runs)
        evaluable = sum(r["evaluable_reveals"] for r in runs)
        orphaned = sum(r["orphaned_reveals"] for r in runs)
        censored = sum(r["right_censored_reveals"] for r in runs)
        return {
            "revealed_wins": revealed,
            "evaluable_reveals": evaluable,
            "orphaned_reveals": orphaned,
            "not_orphaned_reveals": evaluable - orphaned,
            "right_censored_reveals": censored,
            "orphan_rate": orphaned / evaluable if evaluable else 0.0,
        }

    base = [_run(s, params, attack=False, ptc_false_fraction=0.0) for s in seeds]
    b_counts = pooled(base)
    b_orphan = b_counts["orphan_rate"]
    b_debit = statistics.mean(r["debit_gwei"] for r in base)
    b_empty = statistics.mean(r["empty_blocks"] for r in base)
    print(f"BASELINE (honest PTC): orphan_rate={b_orphan:.3f} "
          f"({b_counts['orphaned_reveals']}/{b_counts['evaluable_reveals']}, "
          f"censored={b_counts['right_censored_reveals']}) debit={b_debit/1e6:.1f}M "
          f"empty={b_empty:.1f} wins={statistics.mean(r['selfish_wins'] for r in base):.1f}")

    rows = [{"mode": "baseline", "ptc_false_fraction": 0.0,
             **b_counts, "debit_mean_gwei": b_debit}]
    for frac in (float(x) for x in args.ptc_false_fractions.split(",")):
        atk = [_run(s, params, attack=True, ptc_false_fraction=frac) for s in seeds]
        a_counts = pooled(atk)
        a_orphan = a_counts["orphan_rate"]
        a_debit = statistics.mean(r["debit_gwei"] for r in atk)
        a_empty = statistics.mean(r["empty_blocks"] for r in atk)
        a_still_pays = a_debit >= 0.5 * b_debit  # still pays despite orphan?
        decoupled = a_orphan > b_orphan + 0.05 and a_still_pays
        print(f"ATTACK ptc_false={frac:.2f}: orphan_rate={a_orphan:.3f} "
              f"({a_counts['orphaned_reveals']}/{a_counts['evaluable_reveals']}, "
              f"censored={a_counts['right_censored_reveals']}) debit={a_debit/1e6:.1f}M "
              f"empty={a_empty:.1f} still_pays={a_still_pays} DECOUPLED(orphan+pay)={decoupled}")
        rows.append({"mode": "attack", "ptc_false_fraction": frac,
                     **a_counts, "debit_mean_gwei": a_debit,
                     "still_pays": a_still_pays, "decoupled": decoupled})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"exp81_ptc_orphan_probe_s{args.seeds}_t{args.slots}.json"
    out.write_text(json.dumps({
        "schema": "exp81-v2",
        "aggregation": "pooled_over_reveals_with_observed_child",
        "right_censoring_note": "Final-slot reveals without an observed child are reported separately and excluded from the orphan-rate denominator.",
        "baseline_debit_gwei": b_debit,
        "rows": rows,
    }, indent=2))
    any_decoupled = any(r.get("decoupled") for r in rows)
    print(f"\nD1 PREMISE (PTC can orphan a revealed payload while builder still pays) = {any_decoupled}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
