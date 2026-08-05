"""Experiment 86 -- mitigation evaluation for the PTC-orphan attack.

The attack (exp81-85) exploits the DECOUPLING of the two gates: a bribed PTC
orphans a payload (timeliness gate) while the builder's payment settles anyway
(payment gate). We evaluate the natural defenses, each of which attacks a
different link in the chain, and confirm the headline one in-sim.

  M1  couple payment to canonicality: if the payload is orphaned (PTC not-timely),
      VOID the builder's pending payment instead of settling it. Directly removes
      the OUT-OF-POCKET griefing ("pays for an empty block"). Measured here in-sim.
  M2  make the PTC vote accountable: attach a penalty P (or slashing) to a
      timeliness vote later contradicted by the canonical outcome. Raises the
      per-seat recruitment cost from p_seat to p_seat + P; if P is stake-scaled the fixed
      257-vote bribe becomes stake-linked again. Evaluated analytically via the
      exp82 break-even.
  M3  stake-weight / enlarge the PTC: replace the fixed 512-seat majority with a
      stake-weighted or larger committee, so controlling the pivotal mass scales
      with total stake (like the payment gate). Analytic (same break-even, larger
      or stake-scaled pivotal mass).

Honest scope: M1 removes the unconditional out-of-pocket griefing but NOT the
theft channel (the next proposer still re-captures the now-public MEV) nor the
lost-profit DoS (the victim's block still does not land). M2/M3 attack the
economics so the orphan is no longer cheap, addressing theft and DoS at the root.

M1 accounting: under the mitigation the bid on every orphaned slot is refunded
(never settles), so the victim's realized debit drops by the sum of those bids.
We reuse the exp83 episode runner to identify orphaned slots and read each slot's
bid from the same hook that drives the simulator.

Run: python experiments/exp86_mitigation_eval.py [--seeds 16]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from epbs.committee import MAX_EFFECTIVE_BALANCE_GWEI  # noqa: E402
from epbs.adversary import HonestAdversary  # noqa: E402
from epbs.env_tier2 import Tier2Config, Tier2Environment  # noqa: E402
from epbs.forkchoice import PayloadStatus  # noqa: E402
from experiments.exp27_epbs_selfish_builder_markout import (  # noqa: E402
    MarkoutParams, MarkoutHooks, HONEST_POLICY, SELFISH_BUILDER_INDEX,
    _builder_debit_mark_to_settlement,
)

OUT_DIR = Path(__file__).resolve().parent / "figures" / "drl_risk_epbs"
GWEI_PER_ETH = 1e9


def _run(seed, params, attack: bool) -> dict:
    """One episode. Returns victim utility plus the bid refunded by mitigation M1
    (the sum of bids the victim paid on orphaned slots)."""
    hooks = MarkoutHooks(seed=seed, params=params, policy=HONEST_POLICY)

    def ptc_vote_hook(ptc_idx, slot, local_view_present):
        if not attack or not local_view_present:
            return None
        return False  # bribed PTC: vote not-present -> orphan

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
        ev, _ = env.advance_one_slot()
        events.append(ev)
    env.finalise_episode()
    debit = _builder_debit_mark_to_settlement(env, SELFISH_BUILDER_INDEX)

    by_parent = {ev.parent_root: ev for ev in events}
    victim_exec = 0.0
    refund_m1 = 0.0            # bids paid on orphaned slots (voided under M1)
    n_rev = n_orph = 0
    for ev in events:
        if ev.winner_builder_index != SELFISH_BUILDER_INDEX or not ev.payload_revealed:
            continue
        n_rev += 1
        v_t = hooks.realised_profit_gwei(SELFISH_BUILDER_INDEX, ev.slot)
        child = by_parent.get(ev.block_root)
        orphaned = child is not None and child.parent_status == PayloadStatus.EMPTY
        if orphaned:
            n_orph += 1
            refund_m1 += float(hooks.bid_value_override(SELFISH_BUILDER_INDEX, ev.slot))
        else:
            victim_exec += v_t
    return {
        "victim_utility": victim_exec - debit,          # baseline (no mitigation)
        "victim_utility_m1": victim_exec - debit + refund_m1,  # orphaned bids voided
        "refund_m1": refund_m1,
        "n_revealed": n_rev, "n_orphaned": n_orph,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--slots", type=int, default=64)
    ap.add_argument("--num-builders", type=int, default=8)
    ap.add_argument("--expected-profit-gwei", type=float, default=30_000_000.0)
    ap.add_argument("--markout-sigma-frac", type=float, default=1.15)
    # M2/M3 analytic: an exogenous baseline reservation price plus protocol-level
    # accountability penalties. The current audited protocol penalty is zero.
    ap.add_argument("--baseline-per-seat-price-eth", type=float, default=0.0014)
    ap.add_argument("--accountability-penalties-eth", type=str, default="0,0.001,0.01")
    ap.add_argument("--stake-penalty-fracs", type=str, default="0.001,0.01")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    params = MarkoutParams(
        num_builders=args.num_builders, slots=args.slots,
        expected_profit_gwei=args.expected_profit_gwei,
        markout_sigma_frac=args.markout_sigma_frac,
    )
    seeds = list(range(4000, 4000 + args.seeds))
    if args.baseline_per_seat_price_eth < 0:
        ap.error("--baseline-per-seat-price-eth must be non-negative")
    baseline_per_seat_gwei = args.baseline_per_seat_price_eth * GWEI_PER_ETH
    stake_gwei = float(MAX_EFFECTIVE_BALANCE_GWEI)  # 32 ETH validator stake
    pivotal = config.PAYLOAD_TIMELY_THRESHOLD + 1  # strict > threshold false votes

    honest = [_run(s, params, attack=False) for s in seeds]
    atk = [_run(s, params, attack=True) for s in seeds]

    h_util = statistics.mean(r["victim_utility"] for r in honest)
    a_util = statistics.mean(r["victim_utility"] for r in atk)
    a_util_m1 = statistics.mean(r["victim_utility_m1"] for r in atk)
    griefing = a_util - a_util_m1  # negative of the refunded out-of-pocket loss
    refund = statistics.mean(r["refund_m1"] for r in atk)
    orph_rate = statistics.mean(r["n_orphaned"] / r["n_revealed"] if r["n_revealed"] else 0 for r in atk)

    print("=== M1: couple payment to canonicality (in-sim) ===")
    print(f"victim util  honest        = {h_util/1e6:+8.1f}M gwei")
    print(f"victim util  attack, no M1 = {a_util/1e6:+8.1f}M gwei  (orphan_rate={orph_rate:.2f})")
    print(f"victim util  attack, M1 on = {a_util_m1/1e6:+8.1f}M gwei")
    print(f"  -> out-of-pocket griefing removed by M1 = {refund/1e6:.1f}M gwei/episode "
          f"({refund/GWEI_PER_ETH:.4f} ETH)")
    print(f"  -> residual (lost-profit DoS + theft channel) NOT fixed by M1 alone\n")

    print("=== M2: accountable PTC vote (penalty P raises per-seat recruitment cost) ===")
    print(f"exogenous baseline p_seat={args.baseline_per_seat_price_eth:.6f} ETH; "
          f"current protocol penalty=0; pivotal={pivotal} (fixed)")
    m2_rows = []
    for penalty_eth in (float(x) for x in args.accountability_penalties_eth.split(",")):
        if penalty_eth < 0:
            ap.error("--accountability-penalties-eth must contain non-negative values")
        per_seat_eth = args.baseline_per_seat_price_eth + penalty_eth
        total_cost_eth = pivotal * per_seat_eth
        m2_rows.append({
            "accountability_penalty_per_seat_eth": penalty_eth,
            "total_per_seat_cost_eth": per_seat_eth,
            "total_recruitment_cost_eth": total_cost_eth,
        })
        print(f"  P={penalty_eth:.4f} ETH/seat -> total recruitment cost "
              f"(break-even MEV)={total_cost_eth:.4f} ETH")
    m3_rows = []
    print("  stake-scaled penalty (breaks stake-independence):")
    for sf in (float(x) for x in args.stake_penalty_fracs.split(",")):
        per_seat_gwei = baseline_per_seat_gwei + sf * stake_gwei
        total_cost_eth = pivotal * per_seat_gwei / GWEI_PER_ETH
        m3_rows.append({
            "stake_penalty_frac": sf,
            "total_per_seat_cost_eth": per_seat_gwei / GWEI_PER_ETH,
            "total_recruitment_cost_eth": total_cost_eth,
        })
        print(f"  P = {sf:.3f} x 32ETH -> total recruitment cost = {total_cost_eth:.2f} ETH "
              f"(now scales with stake, like the payment gate)")

    print("\n=== M3: stake-weight / enlarge PTC ===")
    print("  pivotal mass becomes a stake share, not a fixed 257 votes -> cost scales")
    print("  with total stake (removes the fixed-size gate the attack relies on).")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"exp86_mitigation_eval_s{args.seeds}.json"
    out.write_text(json.dumps({
        "schema": "exp86-v3",
        "m1_victim_honest_gwei": h_util,
        "m1_victim_attack_no_mit_gwei": a_util,
        "m1_victim_attack_mit_gwei": a_util_m1,
        "m1_griefing_removed_gwei": refund,
        "orphan_rate": orph_rate,
        "pivotal_members": pivotal,
        "protocol_direct_penalty_eth": 0.0,
        "reservation_price_model": "exogenous_per_seat",
        "baseline_per_seat_price_eth": args.baseline_per_seat_price_eth,
        "stake_gwei": stake_gwei,
        "m2_penalty_rows": m2_rows, "m3_stake_penalty_rows": m3_rows,
        "note": "M1 removes out-of-pocket griefing but not theft/DoS; M2/M3 attack the economics.",
    }, indent=2))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
