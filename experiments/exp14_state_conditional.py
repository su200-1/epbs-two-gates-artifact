"""Experiment 14 addendum: state-conditional optimality check (the S1 fix).

The stationary scan establishes that the best *stationary* template is the
all-slot WITHHOLD+committee-suppression attack. Completeness of that result over
the larger *state-conditional* policy space must not rest on the RL searcher
(which is an `incomplete-heuristic-searcher` — a circular argument). Instead we
check it directly and reliably.

We sweep two parameterized families of state-conditional deviations from the
stationary all-attack policy and test, with PAIRED 95% CIs at matched seeds,
whether any deviation beats all-attack:

  * MEV-threshold:  attack iff the slot's winning bid > theta (the *value*
    dimension — the most plausible timing edge: hit only high-MEV slots).
  * epoch-suffix:   attack only in the last k slots of each epoch (the *timing*
    dimension — captures any settlement/quorum coupling across the epoch
    boundary that the value dimension cannot see).

If no deviation significantly beats theta=0 / k=full, the stationary all-attack
dominates this parameterized state-conditional neighborhood without relying on
RL. This is a bounded optimality check, not an exhaustion of all history-
dependent policies. This also fixes S2 for this check by running multi-seed
with CIs.

Run:
    python experiments/exp14_state_conditional.py [--mev] [--seeds N]
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from epbs.env_tier2 import Tier2Config
from epbs.rl.coalition_env import CoalitionEnvL
from experiments.exp14_common import _episode_metrics, artifact_metadata

ATTACK = (0, 0, 1, 1, 1)   # WITHHOLD + WITHHOLD_VOTE + PTC_FRAUD_ABSENT
HONEST = (0, 0, 0, 0, 0)
OUT = (Path(__file__).resolve().parent / "figures" / "exp14"
       / "exp14_state_conditional.json")

MEV_PARAMS = dict(
    mev_enabled=True, mev_base_gwei=1_000_000, mev_slot_sigma=0.4,
    mev_builder_sigma=0.5,
    mev_builder_skill=(1.6, 1.3, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6), mev_seed=0)
CELLS = [(12, 4), (6, 1)]
THETAS = [0, 0.25e6, 0.5e6, 0.75e6, 1e6, 1.5e6, 2e6, 3e6, 5e6, 1e12]


def make_cfg(nv: int, nb: int, *, mev: bool) -> Tier2Config:
    kw = dict(num_validators=64, num_builders=8, num_slots=65,
              committee_size=8, enable_ffg=True,
              byzantine_validators=set(range(nv)),
              byzantine_builders=set(range(nb)))
    if mev:
        kw.update(MEV_PARAMS)
    return Tier2Config(**kw)


def _run(cfg: Tier2Config, decide, seed: int) -> tuple[float, float]:
    """Run one episode; ``decide(roles, slot)`` returns the 5-tuple action.
    Returns (coalition_utility_gwei, attack_fraction)."""
    env = CoalitionEnvL(replace(cfg))
    obs, _ = env.reset(seed=seed)
    done = False
    n_attack = n_total = 0
    while not done:
        slot = env.env._cur_slot + 1
        roles = env._peek_next_slot_roles()
        a = decide(roles, slot) if roles is not None else HONEST
        n_attack += int(a == ATTACK)
        n_total += 1
        obs, _, done, _, _ = env.step(a)
    return _episode_metrics(env)["coalition_utility_gwei"], n_attack / max(1, n_total)


def _mev_threshold(theta: float):
    return lambda roles, slot: (
        ATTACK if roles["default_effective_amount"] > theta else HONEST)


def _epoch_suffix(k: int):
    spe = config.SLOTS_PER_EPOCH
    return lambda roles, slot: (
        ATTACK if (slot % spe) >= (spe - k) else HONEST)


def _paired(du_theta: list[float], du_base: list[float]) -> tuple[float, float]:
    paired = [a - b for a, b in zip(du_theta, du_base)]
    pm = statistics.mean(paired)
    hw = 1.96 * statistics.stdev(paired) / math.sqrt(len(paired)) if len(paired) > 1 else 0.0
    return pm, hw


def _sweep(cfg, family: str, params: list, seeds: list[int]) -> dict:
    honest = {s: _run(cfg, lambda r, sl: HONEST, s)[0] for s in seeds}
    du, af, labels = {}, {}, {}
    for p in params:
        decide = _mev_threshold(p) if family == "mev_threshold" else _epoch_suffix(p)
        d, a = [], []
        for s in seeds:
            u, frac = _run(cfg, decide, s)
            d.append(u - honest[s])
            a.append(frac)
        du[p] = d
        af[p] = statistics.mean(a)
        labels[p] = (f"{p/1e6:.2f}M" if family == "mev_threshold" and p < 1e11
                     else ("inf" if p >= 1e11 else str(p)))
    base_key = params[0]           # theta=0 / k=full == all-attack
    base = du[base_key]
    rows, sig = [], None
    for p in params:
        pm, hw = _paired(du[p], base)
        rows.append({"param": labels[p], "mean_du_gwei": statistics.mean(du[p]),
                     "paired_vs_allattack_gwei": pm, "ci95_halfwidth_gwei": hw,
                     "attack_fraction": af[p]})
        if p != base_key and pm - hw > 0 and sig is None:
            sig = labels[p]
    return {"family": family, "rows": rows,
            "all_attack_optimal": sig is None, "first_significant_winner": sig}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mev", action="store_true", default=True)
    ap.add_argument("--no-mev", dest="mev", action="store_false")
    ap.add_argument("--seeds", type=int, default=8)
    args = ap.parse_args()
    seeds = list(range(999_000, 999_000 + args.seeds))
    spe = config.SLOTS_PER_EPOCH
    epoch_ks = [spe, spe * 3 // 4, spe // 2, spe // 4, spe // 8, 0]

    results = []
    for nv, nb in CELLS:
        cfg = make_cfg(nv, nb, mev=args.mev)
        for family, params in [("mev_threshold", THETAS),
                               ("epoch_suffix", epoch_ks)]:
            res = _sweep(cfg, family, params, seeds)
            res.update(n_byz_validators=nv, n_byz_builders=nb, mev=args.mev,
                       n_seeds=len(seeds))
            results.append(res)
            print(f"\n=== {family}  nv{nv} nb{nb} (k={len(seeds)} seeds, "
                  f"mev={args.mev}) ===")
            print(f"{'param':>8} {'mean dU(M)':>11} "
                  f"{'paired vs all-attack (M,95%CI)':>32} {'attack%':>8}")
            for r in res["rows"]:
                ci = ("(baseline)" if r["param"] in ("0.00M", str(spe))
                      else f"{r['paired_vs_allattack_gwei']/1e6:+.3f} +/- "
                           f"{r['ci95_halfwidth_gwei']/1e6:.3f}")
                print(f"{r['param']:>8} {r['mean_du_gwei']/1e6:>11.2f} "
                      f"{ci:>32} {r['attack_fraction']*100:>7.1f}%")
            verdict = ("all-attack optimal (no deviation significantly wins)"
                       if res["all_attack_optimal"]
                       else f"DEVIATION WINS: {res['first_significant_winner']}")
            print(f"  VERDICT: {verdict}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(artifact_metadata(
        mev=args.mev,
        n_seeds=len(seeds),
        seeds=seeds,
        cells=[{"n_byz_validators": nv, "n_byz_builders": nb}
               for nv, nb in CELLS],
        families=["mev_threshold", "epoch_suffix"],
        results=results,
        claim_scope=(
            "bounded state-conditional sweep over MEV-threshold and "
            "epoch-suffix deviations; not exhaustive over arbitrary "
            "history-dependent policies"
        ),
    ), indent=2))
    overall = all(r["all_attack_optimal"] for r in results)
    print(f"\nOVERALL: stationary all-attack optimal across all families/cells: "
          f"{overall}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
