"""Experiment 83 -- PTC-orphan theft: victim loss + MEV re-capture (D1 second crux).

exp81 validated the orphan mechanic; exp82 showed the bribe is cheap and scale-
robust. The remaining make-or-break is the THEFT side: exp82 used an upper bound
"captured MEV <= block value". Here we (a) MEASURE the victim's realised loss
in-sim and the size of the orphaned-MEV pool, and (b) model the attacker's actual
theft as a re-capture fraction rho_recap of that pool, net of the per-slot bribe,
under a SELECTIVE (state-conditional) attack that only strikes slots where it pays.

Value flows when slot-t payload is orphaned via a bribed PTC:
  * victim builder t: pays bid B_t (settles anyway) AND loses execution profit V_t
    (EMPTY block -> MEV not executed on-chain). Net vs honest: -V_t.
  * attacker (next proposer t+1 + bribed PTC): captures rho_recap * V_t, pays the
    257-seat recruitment cost. A rational attacker strikes slot t iff
        rho_recap * V_t > cost_per_slot (= 257 * p_seat).
    -> naturally STATE-CONDITIONAL on V_t (the DRL/search lever).

Correctness: for orphaned slots the victim's execution profit is voided (the
payload never entered the canonical chain), while its debit still settles.

Run: python experiments/exp83_ptc_theft_recapture.py [--seeds 16]
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from epbs.adversary import HonestAdversary  # noqa: E402
from epbs.env_tier2 import Tier2Config, Tier2Environment  # noqa: E402
from epbs.forkchoice import PayloadStatus  # noqa: E402
from experiments.exp27_epbs_selfish_builder_markout import (  # noqa: E402
    MarkoutParams, MarkoutHooks, HONEST_POLICY, SELFISH_BUILDER_INDEX,
    _builder_debit_mark_to_settlement,
)

OUT_DIR = Path(__file__).resolve().parent / "figures" / "drl_risk_epbs"
GWEI_PER_ETH = 1e9
DEFAULT_EXPECTED_PROFIT_GWEI = 30_000_000.0
HIGH_MEV_EXPECTED_PROFIT_GWEI = 1_000_000_000.0


def _scenario_label(expected_profit_gwei: float, explicit: str | None) -> str:
    if explicit:
        return explicit
    if math.isclose(expected_profit_gwei, DEFAULT_EXPECTED_PROFIT_GWEI):
        return "default"
    if math.isclose(expected_profit_gwei, HIGH_MEV_EXPECTED_PROFIT_GWEI):
        return "highmev"
    return f"profit{int(expected_profit_gwei)}gwei"


def _run(seed, params, attack: bool) -> dict:
    hooks = MarkoutHooks(seed=seed, params=params, policy=HONEST_POLICY)  # victim always reveals

    def ptc_vote_hook(ptc_idx, slot, local_view_present):
        # Bribed PTC: vote not-present on any revealed payload.
        # The attack hook flips all sampled seats; economics below charges only
        # the exact pivotal set, PAYLOAD_TIMELY_THRESHOLD + 1 = 257.
        if not attack or not local_view_present:
            return None
        return False

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
    result = env.finalise_episode()
    debit = _builder_debit_mark_to_settlement(env, SELFISH_BUILDER_INDEX)

    by_parent = {ev.parent_root: ev for ev in events}
    victim_exec = 0.0          # execution profit victim actually keeps
    orphaned_pool = 0.0        # V_t of orphaned slots (stealable pool)
    orphaned_vts = []          # per-orphaned-slot MEV, for selective-attack accounting
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
            orphaned_pool += v_t
            orphaned_vts.append(v_t)
        else:
            victim_exec += v_t
    return {
        "victim_utility": victim_exec - debit,
        "victim_exec": victim_exec, "debit": debit,
        "n_revealed": n_rev, "n_orphaned": n_orph,
        "orphaned_pool_gwei": orphaned_pool, "orphaned_vts": orphaned_vts,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--slots", type=int, default=64)
    ap.add_argument("--num-builders", type=int, default=8)
    ap.add_argument("--expected-profit-gwei", type=float, default=DEFAULT_EXPECTED_PROFIT_GWEI)
    ap.add_argument("--markout-sigma-frac", type=float, default=1.15)
    ap.add_argument("--recapture-fractions", type=str, default="0.25,0.5,0.75,1.0")
    ap.add_argument(
        "--per-seat-prices-eth", type=str, default="0.00001,0.0001,0.001",
        help="Comma-separated exogenous reservation prices per recruited PTC seat.",
    )
    ap.add_argument("--scenario-label", type=str, default=None,
                    help="Output label; defaults to 'default' or 'highmev' for known calibrations.")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    params = MarkoutParams(
        num_builders=args.num_builders, slots=args.slots,
        expected_profit_gwei=args.expected_profit_gwei,
        markout_sigma_frac=args.markout_sigma_frac,
    )
    seeds = list(range(4000, 4000 + args.seeds))
    pivotal = config.PAYLOAD_TIMELY_THRESHOLD + 1  # strict > threshold false votes
    per_seat_prices_eth = [float(x) for x in args.per_seat_prices_eth.split(",")]
    if not per_seat_prices_eth or any(p < 0 for p in per_seat_prices_eth):
        ap.error("--per-seat-prices-eth must contain non-negative values")
    scenario = _scenario_label(args.expected_profit_gwei, args.scenario_label)

    honest = [_run(s, params, attack=False) for s in seeds]
    atk = [_run(s, params, attack=True) for s in seeds]

    h_util = statistics.mean(r["victim_utility"] for r in honest)
    a_util = statistics.mean(r["victim_utility"] for r in atk)
    victim_loss = h_util - a_util
    orph_rate = statistics.mean(r["n_orphaned"] / r["n_revealed"] if r["n_revealed"] else 0 for r in atk)
    pool = statistics.mean(r["orphaned_pool_gwei"] for r in atk)
    print(f"VICTIM: honest util={h_util/1e6:.1f}M  attack util={a_util/1e6:.1f}M  "
          f"loss/episode={victim_loss/1e6:.1f}M  orphan_rate={orph_rate:.2f}  "
          f"orphaned_pool/episode={pool/1e6:.1f}M ({pool/GWEI_PER_ETH:.4f} ETH)")
    print(f"pivotal recruited-seat count={pivotal} (fixed); per-seat price is exogenous\n")

    # attacker economics: selective attack (strike slot iff rho*V > bribe_per_slot)
    all_vts = [v for r in atk for v in r["orphaned_vts"]]
    n_ep = len(seeds)
    rows = []
    for rho in (float(x) for x in args.recapture_fractions.split(",")):
        for per_seat_eth in per_seat_prices_eth:
            per_seat_gwei = per_seat_eth * GWEI_PER_ETH
            bribe_slot = pivotal * per_seat_gwei
            # non-selective: attack every orphanable slot
            theft_all = rho * sum(all_vts) - bribe_slot * len(all_vts)
            # selective: only slots where rho*V_t > bribe_slot
            hit = [v for v in all_vts if rho * v > bribe_slot]
            theft_sel = sum(rho * v - bribe_slot for v in hit)
            rows.append({
                "rho_recapture": rho,
                "per_seat_reservation_price_eth": per_seat_eth,
                "per_seat_reservation_price_gwei": per_seat_gwei,
                "recruitment_cost_per_slot_gwei": bribe_slot,
                "attacker_net_per_episode_all_gwei": theft_all / n_ep,
                "attacker_net_per_episode_selective_gwei": theft_sel / n_ep,
                "selective_hit_rate": len(hit) / len(all_vts) if all_vts else 0.0,
            })
            print(f"rho={rho:.2f} p_seat={per_seat_eth:.6f} ETH: "
                  f"net/episode  all={theft_all/n_ep/1e6:+7.1f}M  "
                  f"selective={theft_sel/n_ep/1e6:+7.1f}M  (hit {len(hit)}/{len(all_vts)} slots)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"exp83_ptc_theft_recapture_{scenario}_s{args.seeds}_t{args.slots}.json"
    out.write_text(json.dumps({
        "schema": "exp83-v3",
        "scenario": scenario,
        "expected_profit_gwei": args.expected_profit_gwei,
        "markout_sigma_frac": args.markout_sigma_frac,
        "victim_honest_util_gwei": h_util, "victim_attack_util_gwei": a_util,
        "victim_loss_per_episode_gwei": victim_loss, "orphan_rate": orph_rate,
        "orphaned_pool_per_episode_gwei": pool, "pivotal_members": pivotal,
        "protocol_direct_penalty_eth": 0.0,
        "reservation_price_model": "exogenous_per_seat",
        "per_seat_prices_eth": per_seat_prices_eth,
        "rows": rows,
    }, indent=2))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
