"""Experiment 85 -- learned selective-attack policy: does a learned policy beat a
simple fixed threshold for the zero-holding PTC-orphan attacker? (Negative result.)

This is imitation / behaviour cloning (supervised MLP on offline labels), NOT DRL and
NOT DAgger: there is no reward-driven policy optimisation, no environment interaction,
no learner rollout, and no oracle relabel / dataset aggregation.

The teacher is a greedy foresight REFERENCE selection (ratio-greedy 0/1 knapsack with
foresight) -- a heuristic, not a proven optimum. The learned policy sees only causal
online features (current signal, current bribe, remaining budget, remaining slots).

Finding (main case = zero-holding, owned_frac=0): the learned policy does NOT beat a
simple fixed threshold; on separate evaluation seeds it fails to beat even the fixed signal
threshold, and a two-feature hand rule (rho*V - b > kappa) is the best non-oracle policy
across bribe premia. We report this as a negative methodological result: learning
recovers rather than discovers; the contribution is the attack surface, not the policy.

Run: python experiments/exp85_selective_attack_neural.py [--train-seeds 40] [--eval-seeds 40]
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

import torch
import torch.nn as nn

import config  # noqa: E402
from experiments.exp84_selective_attack_oracle import _draw_episode  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "figures" / "drl_risk_epbs"
GWEI_PER_ETH = 1e9
FEATURE_DIM = 6


def _offline_ref_set(ep, rho, budget):
    """Per-slot binary labels of a greedy foresight REFERENCE attack set
    (ratio-greedy 0/1 knapsack -- a heuristic, not a proven optimum)."""
    idx_net = sorted(
        (i for i in range(len(ep)) if rho * ep[i][0] - ep[i][2] > 0),
        key=lambda i: ((rho * ep[i][0] - ep[i][2]) / ep[i][2] if ep[i][2] > 0 else float("inf")),
        reverse=True,
    )
    spent = 0.0
    labels = [0] * len(ep)
    for i in idx_net:
        b = ep[i][2]
        if spent + b > budget:
            continue
        spent += b
        labels[i] = 1
    return labels


def _features(v_signal, bribe, rem_budget, rem_slots, total_budget, n_slots, rho, med_gwei):
    """Causal online features for a slot."""
    return [
        math.log1p(v_signal) - math.log1p(med_gwei),         # signal vs median (log)
        math.log1p(bribe) - math.log1p(med_gwei),            # bribe vs median (log)
        rem_budget / total_budget if total_budget > 0 else 0.0,   # remaining budget frac
        rem_slots / n_slots,                                  # remaining slots frac
        (rho * v_signal - bribe) / med_gwei,                  # net estimate (scaled)
        (rho * v_signal) / bribe if bribe > 0 else 10.0,      # value/bribe ratio (capped)
    ]


class AttackMLP(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FEATURE_DIM, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)


def _run_online(ep, decide, rho, budget):
    """decide(signal,bribe,rem_budget,rem_slots,total_budget)->bool. Realized uses true V."""
    spent = 0.0
    realized = 0.0
    n = len(ep)
    for i, (v, s, b) in enumerate(ep):
        if decide(s, b, budget - spent, n - i, budget) and spent + b <= budget:
            spent += b
            realized += rho * v - b
    return realized


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-seeds", type=int, default=40)
    ap.add_argument("--eval-seeds", type=int, default=40)
    ap.add_argument("--slots", type=int, default=64)
    ap.add_argument("--median-eth", type=float, default=0.04)
    ap.add_argument("--sigma", type=float, default=1.6)
    ap.add_argument("--signal-noise", type=float, default=0.9)
    ap.add_argument("--owned-frac", type=float, default=0.0)   # main result: zero-holding attacker
    ap.add_argument(
        "--per-seat-price-eth", type=float, default=0.0014,
        help="Exogenous reservation price per recruited PTC seat (ETH).",
    )
    ap.add_argument("--rho", type=float, default=0.5)
    ap.add_argument("--budget-frac", type=float, default=0.30)   # constrained-regime default
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    if args.per_seat_price_eth < 0:
        ap.error("--per-seat-price-eth must be non-negative")
    per_seat_price_gwei = args.per_seat_price_eth * GWEI_PER_ETH
    med_gwei = args.median_eth * GWEI_PER_ETH
    rho = args.rho

    def make(seeds):
        eps = [_draw_episode(random.Random(s), args.slots, args.median_eth, args.sigma,
                             args.signal_noise, args.owned_frac, per_seat_price_gwei) for s in seeds]
        buds = [args.budget_frac * sum(b for _, _, b in e) for e in eps]
        return eps, buds

    tr_eps, tr_bud = make(range(6000, 6000 + args.train_seeds))
    ev_eps, ev_bud = make(range(9000, 9000 + args.eval_seeds))

    # ---- build imitation dataset: features -> greedy-foresight-reference label ----
    X, Y = [], []
    for ep, bud in zip(tr_eps, tr_bud):
        labels = _offline_ref_set(ep, rho, bud)
        spent = 0.0; n = len(ep)
        for i, (v, s, b) in enumerate(ep):
            X.append(_features(s, b, bud - spent, n - i, bud, n, rho, med_gwei))
            Y.append(labels[i])
            if labels[i] == 1:      # teacher-forced budget trajectory
                spent += b
    X = torch.tensor(X, dtype=torch.float32); Y = torch.tensor(Y, dtype=torch.long)

    def train_one(seed):
        torch.manual_seed(seed)
        model = AttackMLP()
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        counts = torch.bincount(Y, minlength=2).float()
        w = (counts.sum() / (2 * counts.clamp(min=1)))
        lossf = nn.CrossEntropyLoss(weight=w)
        for _ in range(args.epochs):
            opt.zero_grad()
            loss = lossf(model(X), Y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        return model

    # baselines on eval set
    def eval_pol(decide):
        return [_run_online(ep, decide, rho, bud) for ep, bud in zip(ev_eps, ev_bud)]

    naive = eval_pol(lambda s, b, rb, rs, tb: True)
    all_s = sorted(s for ep in ev_eps for _, s, _ in ep)
    qs = [all_s[min(len(all_s) - 1, int(p * len(all_s)))] for p in (0, .3, .5, .7, .85, .93, .97)]
    best_sig_u = max((eval_pol(lambda s, b, rb, rs, tb, t=t: s > t) for t in qs),
                     key=statistics.mean)
    net_grid = [-math.inf, 0.0] + [rho * q for q in qs[1:]]
    best_net_u = max((eval_pol(lambda s, b, rb, rs, tb, k=k: (rho * s - b) > k) for k in net_grid),
                     key=statistics.mean)
    ceiling = [sum(rho * ep[i][0] - ep[i][2] for i, lab in
                   enumerate(_offline_ref_set(ep, rho, bud)) if lab)
               for ep, bud in zip(ev_eps, ev_bud)]

    # neural: multi-seed
    n_seeds = 5
    neural_seed_values = []
    for ts in range(n_seeds):
        model = train_one(ts)
        model.eval()

        def decide(s, b, rem_budget, rem_slots, total_budget, m=model):
            f = torch.tensor(_features(s, b, rem_budget, rem_slots,
                                       total_budget, args.slots, rho, med_gwei),
                             dtype=torch.float32).view(1, -1)
            with torch.no_grad():
                return bool(torch.argmax(m(f)).item() == 1)

        u = eval_pol(decide)
        neural_seed_values.append({"seed": ts, "mean_gwei": statistics.mean(u)})

    neural_means = [row["mean_gwei"] for row in neural_seed_values]
    neural_mean = statistics.mean(neural_means)
    neural_sd = statistics.stdev(neural_means)
    neural_min = min(neural_means)
    neural_max = max(neural_means)
    signal_mean = statistics.mean(best_sig_u)
    net_mean = statistics.mean(best_net_u)
    print(f"regime: budget_frac={args.budget_frac} "
          f"p_seat={args.per_seat_price_eth:.6f} ETH noise={args.signal_noise}")
    print(f"NAIVE           = {statistics.mean(naive)/1e6:8.1f}M")
    print(f"best SIGNAL thr = {signal_mean/1e6:8.1f}M")
    print(f"best NET thr     = {net_mean/1e6:8.1f}M")
    for row in neural_seed_values:
        print(f"NEURAL seed {row['seed']} = {row['mean_gwei']/1e6:8.1f}M")
    print(f"NEURAL all {n_seeds}: mean={neural_mean/1e6:8.1f}M  "
          f"sample SD={neural_sd/1e6:.1f}M  range=[{neural_min/1e6:.1f}M, {neural_max/1e6:.1f}M]")
    print(f"descriptive mean difference: vs signal {(neural_mean-signal_mean)/1e6:+.1f}M  "
          f"vs net {(neural_mean-net_mean)/1e6:+.1f}M")
    print(f"OFFLINE REFERENCE (greedy foresight, not a bound) = {statistics.mean(ceiling)/1e6:8.1f}M")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    price_tag = f"{args.per_seat_price_eth:g}".replace(".", "p").replace("-", "m")
    out = args.out_dir / f"exp85_selective_attack_neural_bf{args.budget_frac}_pseat{price_tag}.json"
    out.write_text(json.dumps({
        "schema": "exp85-v3",
        "params": ({k: v for k, v in vars(args).items() if k != "out_dir"} | {
            "per_seat_reservation_price_gwei": per_seat_price_gwei,
            "protocol_direct_penalty_eth": 0.0,
            "reservation_price_model": "exogenous_per_seat",
        }),
        "naive_gwei": statistics.mean(naive), "best_signal_gwei": signal_mean,
        "best_net_gwei": net_mean,
        "neural_seed_values": neural_seed_values,
        "neural_mean_gwei": neural_mean, "neural_sd_gwei": neural_sd,
        "neural_min_gwei": neural_min, "neural_max_gwei": neural_max,
        "foresight_reference_gwei": statistics.mean(ceiling),
        "comparison_note": "Descriptive negative evidence only: signal and net-value thresholds were tuned on the evaluation episodes; all five neural initializations are reported.",
        "n_seeds": n_seeds,
    }, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
