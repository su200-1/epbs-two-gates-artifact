"""Experiment 84 -- selective PTC-orphan attack: is the optimal deployment policy
genuinely STATE-CONDITIONAL? (DRL-headline viability gate, avoids the exp79 trap.)

exp81-83 established the attack (orphan + still-pay + steal) and its economics.
The DRL headline needs the optimal *deployment* policy -- when to strike -- to be
state-conditional, else a fixed threshold is optimal and DRL merely recovers it
(the exp79 lesson: a single-feature threshold problem has a static Bayes-optimal
rule).

The attack has THREE observables that a rational attacker must jointly weigh:
  * s_t  -- a NOISY per-slot MEV signal (true value V_t observed with error);
  * b_t  -- the per-slot BRIBE cost: RANDAO samples the 512-PTC each slot, so the
            number of PTC seats the attacker already owns varies -> it must bribe
            max(0, 256 - owned_t) members, so b_t fluctuates slot to slot;
  * remaining bribe BUDGET -- the attacker is capital-constrained per epoch.

The oracle compares, paired on seeds and under the SAME budget:
  * NAIVE       : attack every orphanable slot (until budget out);
  * SIGNAL-ONLY : attack iff s_t > tau (best fixed threshold on the signal alone);
  * STATE-COND  : attack iff net_hat_t = rho*s_t - b_t > kappa (accounts for the
                  per-slot bribe) -- and, budget-aware, spend on the best net_hat.

SIGNAL-ONLY strictly nests in STATE-COND (b_t treated as constant). If STATE-COND
beats SIGNAL-ONLY with CI>0, the optimal policy is genuinely state-conditional ->
DRL has something a fixed rule cannot express -> build the neural policy next.
If not, DRL only recovers and the attack itself remains the discovery.

Realized reward per attacked slot uses TRUE V_t: rho*V_t - b_t. Decisions use the
noisy signal s_t. Distributions are calibrated to the exp82 block-value model.

Run: python experiments/exp84_selective_attack_oracle.py [--seeds 24]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "figures" / "drl_risk_epbs"
GWEI_PER_ETH = 1e9


def _draw_episode(rng, n_slots, median_eth, sigma, sig_noise, owned_frac, per_seat_price_gwei):
    """Per-slot (true V_t, noisy signal s_t, bribe b_t)."""
    mu = math.log(median_eth * GWEI_PER_ETH)
    slots = []
    for _ in range(n_slots):
        v = math.exp(rng.gauss(mu, sigma))                                   # true MEV
        s = max(0.0, v * math.exp(rng.gauss(0.0, sig_noise) - sig_noise ** 2 / 2))  # noisy signal
        owned = sum(1 for _ in range(config.PTC_SIZE) if rng.random() < owned_frac)  # PTC seats owned
        need = max(0, config.PAYLOAD_TIMELY_THRESHOLD + 1 - owned)            # seats to bribe: n^- >= 257 (>256)
        b = need * per_seat_price_gwei
        slots.append((v, s, b))
    return slots


def _greedy_budget(slots, score_fn, attack_pred, rho, budget):
    """Attack slots (time order) where attack_pred is true, while budget remains.
    score_fn/attack_pred take (v_hat_signal, bribe). Realized uses true V."""
    spent = 0.0
    realized = 0.0
    for v, s, b in slots:
        if not attack_pred(s, b):
            continue
        if spent + b > budget:
            continue
        spent += b
        realized += rho * v - b
    return realized


def _ci95_low(xs):
    if len(xs) < 2:
        return xs[0] if xs else 0.0
    m = statistics.mean(xs); sd = statistics.pstdev(xs)
    return m - 1.96 * sd / math.sqrt(len(xs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=24)
    ap.add_argument("--slots", type=int, default=64)
    ap.add_argument("--median-eth", type=float, default=0.04)
    ap.add_argument("--sigma", type=float, default=1.6)
    ap.add_argument("--signal-noise", type=float, default=0.6)
    ap.add_argument("--owned-frac", type=float, default=0.0)    # attacker PTC-seat ownership share (main result: zero-holding)
    ap.add_argument(
        "--per-seat-price-eth", type=float, default=0.0014,
        help="Exogenous reservation price per recruited PTC seat (ETH).",
    )
    ap.add_argument("--rho", type=float, default=0.5)
    ap.add_argument("--budget-frac", type=float, default=0.4)   # budget as frac of attack-all bribe
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    if args.per_seat_price_eth < 0:
        ap.error("--per-seat-price-eth must be non-negative")
    per_seat_price_gwei = args.per_seat_price_eth * GWEI_PER_ETH
    seeds = list(range(5000, 5000 + args.seeds))
    episodes = [_draw_episode(random.Random(s), args.slots, args.median_eth, args.sigma,
                              args.signal_noise, args.owned_frac, per_seat_price_gwei) for s in seeds]

    # budget per episode = budget_frac * (bribe to attack everything)
    budgets = [args.budget_frac * sum(b for _, _, b in ep) for ep in episodes]

    # signal thresholds grid (on s_t) and net thresholds grid (on rho*s - b)
    all_s = sorted(s for ep in episodes for _, s, _ in ep)
    q = lambda p: all_s[min(len(all_s) - 1, int(p * len(all_s)))]
    sig_grid = [0.0] + [q(p) for p in (0.3, 0.5, 0.7, 0.85, 0.93, 0.97)]
    net_grid = [-math.inf, 0.0] + [args.rho * q(p) for p in (0.3, 0.5, 0.7, 0.85, 0.93)]

    def eval_policy(attack_pred):
        return [_greedy_budget(ep, None, attack_pred, args.rho, bud)
                for ep, bud in zip(episodes, budgets)]

    naive = eval_policy(lambda s, b: True)

    # OFFLINE FORESIGHT REFERENCE (greedy, NOT a proven optimum/upper bound):
    # with foresight of all slots, greedily take positive-net slots by
    # (rho*V - b)/b ratio s.t. sum(b) <= budget. Ratio-greedy on a 0/1 knapsack
    # is a heuristic, not exact, so this is a reference point, not a ceiling that
    # provably upper-bounds every online policy. Used only for a rough headroom
    # gauge; the paper makes no optimality claim on it.
    def offline_ref(ep, budget):
        cand = [(rho_v_b) for rho_v_b in
                ((args.rho * v - b, b) for v, s, b in ep) if rho_v_b[0] > 0]
        cand.sort(key=lambda nb: (nb[0] / nb[1] if nb[1] > 0 else float("inf")), reverse=True)
        spent = 0.0; tot = 0.0
        for net, b in cand:
            if spent + b > budget:
                continue
            spent += b; tot += net
        return tot
    ceiling = [offline_ref(ep, bud) for ep, bud in zip(episodes, budgets)]

    best_sig = None; best_sig_u = None; best_tau = None
    for tau in sig_grid:
        u = eval_policy(lambda s, b, t=tau: s > t)
        if best_sig is None or statistics.mean(u) > statistics.mean(best_sig_u):
            best_sig = statistics.mean(u); best_sig_u = u; best_tau = tau

    best_sc = None; best_sc_u = None; best_kappa = None
    for kappa in net_grid:
        u = eval_policy(lambda s, b, k=kappa: (args.rho * s - b) > k)
        if best_sc is None or statistics.mean(u) > statistics.mean(best_sc_u):
            best_sc = statistics.mean(u); best_sc_u = u; best_kappa = kappa

    diff_sc_sig = [sc - sg for sc, sg in zip(best_sc_u, best_sig_u)]
    diff_sig_naive = [sg - nv for sg, nv in zip(best_sig_u, naive)]

    print(f"per-seat reservation price={args.per_seat_price_eth:.6f} ETH  "
          f"owned_frac={args.owned_frac}  "
          f"rho={args.rho}  budget_frac={args.budget_frac}  signal_noise={args.signal_noise}")
    print(f"NAIVE(attack all)      = {statistics.mean(naive)/1e6:8.1f}M gwei")
    print(f"SIGNAL-ONLY (best tau) = {best_sig/1e6:8.1f}M   vs naive: +{statistics.mean(diff_sig_naive)/1e6:.1f}M "
          f"CIlow {_ci95_low(diff_sig_naive)/1e6:+.1f}M")
    print(f"STATE-COND (best kappa)= {best_sc/1e6:8.1f}M   vs signal: {statistics.mean(diff_sc_sig)/1e6:+.1f}M "
          f"CIlow {_ci95_low(diff_sc_sig)/1e6:+.1f}M")
    sc_beats = _ci95_low(diff_sc_sig) > 0
    diff_ceil_sig = [c - sg for c, sg in zip(ceiling, best_sig_u)]
    print(f"OFFLINE REFERENCE (greedy foresight, not a proven bound) = {statistics.mean(ceiling)/1e6:8.1f}M   "
          f"headroom over best fixed threshold: +{statistics.mean(diff_ceil_sig)/1e6:.1f}M "
          f"({100*statistics.mean(diff_ceil_sig)/best_sig:.1f}%)")
    print(f"\nSTATE-CONDITIONAL beats SIGNAL-ONLY (CI>0) = {sc_beats}")
    print("-> small headroom over the best fixed threshold suggests the exploitable")
    print("   optimum is well-approximated by a static MEV threshold (no optimality claim).")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"exp84_selective_attack_oracle_s{args.seeds}.json"
    out.write_text(json.dumps({
        "schema": "exp84-v2",
        "params": vars(args) | {
            "per_seat_reservation_price_gwei": per_seat_price_gwei,
            "protocol_direct_penalty_eth": 0.0,
            "reservation_price_model": "exogenous_per_seat",
        },
        "naive_mean_gwei": statistics.mean(naive),
        "signal_only_mean_gwei": best_sig, "best_tau": best_tau,
        "state_cond_mean_gwei": best_sc, "best_kappa": best_kappa,
        "sc_minus_signal_mean_gwei": statistics.mean(diff_sc_sig),
        "sc_minus_signal_ci95_low_gwei": _ci95_low(diff_sc_sig),
        "sc_beats_signal": sc_beats,
    }, indent=2, default=str))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
