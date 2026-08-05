"""Experiment 95 -- how much of the PTC pivotal set the OBSERVED professional
node-operator population can supply, with no invented tail distribution.

Why this exists
---------------
exp93 sampled committees under invented operator distributions (Zipf(128), a
hand-built 32-operator ladder).  Both are far more concentrated than mainnet --
the largest real node operator holds 7.45 % of stake, whereas Zipf(128) puts
18.4 % in its largest bucket -- so they understate how many counterparties an
attacker must recruit, biasing the estimate in the attacker's favour.

This experiment uses only measured shares.  Public attribution covers 37.2 % of
stake; the remaining 62.8 % (solo stakers, unlabelled entities, permissionless
modules) has no public composition, so instead of inventing one we report only
what the measured head determines:

  * the marginal seats the coalition must add,      (exact, from f)
  * the seats the whole attributed set supplies,    (measured)
  * and, where the head covers the need, how many
    of the largest attributed operators suffice.    (measured)

Below the crossover the honest statement is not a number but a structural fact:
the entire professional operator set is insufficient and the coalition must
reach into the unattributed remainder.

Data
----
Node-operator network-penetration shares, Rated Network mainnet explorer,
node-operator view (accessed 2026-07-22).  Node-operator granularity is the
right unit here: it is the entity that runs the validator keys and can be
solicited individually, not the staking-pool brand.  (Lido's ~37 curated
operators appear separately, each near 0.55 %, which is why the list is flat
in its middle.)

Metric
------
Largest-first recruitment, i.e. the attacker-optimal order, so any operator
count reported is a LOWER bound on counterparties.

Run: python experiments/exp95_empirical_operator_concentration.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).resolve().parent / "figures" / "drl_risk_epbs"
PTC_SIZE = 512
PIVOTAL = 257
N_COMMITTEES = 100_000
SEED = 20260722

# Rated Network, mainnet, node-operator view, accessed 2026-07-22.
RATED_OPERATORS: list[tuple[str, float]] = [
    ("Kraken", 7.45), ("Everstake", 2.66), ("P2P.org", 2.62), ("Stakefish", 2.02),
    ("Allnodes", 1.55), ("Bitcoin Suisse", 1.54), ("InfStones", 0.96),
    ("HashKey Cloud", 0.87), ("Consensys", 0.79), ("Ebunker", 0.78),
    ("DSRV", 0.75), ("Luganodes", 0.75), ("ParaFi Technologies", 0.68),
    ("Galaxy", 0.68), ("Figment", 0.68), ("Blockdaemon", 0.68), ("Stakely", 0.59),
    ("RockX", 0.58), ("SenseiNode", 0.56), ("Blockscape", 0.56), ("Gateway.fm", 0.55),
    ("Launchnodes", 0.55), ("RockawayX Infra", 0.55), ("Attestant", 0.55),
    ("Twinstake", 0.55), ("ChainSafe", 0.55), ("Kukis Global", 0.55),
    ("RockLogic GmbH", 0.55), ("ChainLayer", 0.55), ("Simply Staking", 0.55),
    ("Staking Facilities", 0.55), ("DARMA Capital", 0.49), ("Kiln", 0.49),
    ("Coinbase Cloud", 0.42), ("Validation Cloud", 0.40), ("Node.Monster", 0.35),
    ("Chainnodes", 0.25), ("Golem Foundation", 0.25), ("Nethermind", 0.18),
    ("Finoa Consensus Services", 0.13), ("Pier Two", 0.11), ("Chorus One", 0.09),
    ("EtherNodes", 0.09), ("ChainUp Cloud", 0.07), ("Stakin", 0.03), ("Meria", 0.02),
    ("cryptonative.systems", 0.01), ("XHash", 0.01), ("Lighthouse Client Team", 0.01),
    ("Prysm Client Team", 0.01), ("Nimbus Client Team", 0.01),
]

OWNED_FRACTIONS = (0.0, 0.10, 0.20, 0.25, 1 / 3)


def summarize(x: np.ndarray) -> dict:
    return {"median": float(np.median(x)),
            "p05": float(np.percentile(x, 5)),
            "p95": float(np.percentile(x, 95))}


def main() -> None:
    head = np.array([s for _, s in RATED_OPERATORS], dtype=float) / 100.0
    attributed = float(head.sum())
    unattributed = 1.0 - attributed

    rows = []
    for f in OWNED_FRACTIONS:
        rng = np.random.default_rng(SEED)
        # One bucket per attributed operator, one aggregate bucket for the
        # unattributed remainder, plus the attacker. The remainder is NOT split
        # into invented operators: it is only used to keep the draw well formed.
        probs = np.r_[f, (1.0 - f) * head, (1.0 - f) * unattributed]
        counts = rng.multinomial(PTC_SIZE, probs, size=N_COMMITTEES)
        owned = counts[:, 0]
        attr_seats = counts[:, 1:-1]
        marginal = np.maximum(PIVOTAL - owned, 0)

        head_supply = attr_seats.sum(axis=1)
        covered = head_supply >= marginal

        sorted_head = np.sort(attr_seats, axis=1)[:, ::-1]
        cumulative = np.cumsum(sorted_head, axis=1)
        need_k = (cumulative < marginal[:, None]).sum(axis=1) + 1
        need_k = np.where(marginal == 0, 0, need_k)

        row = {
            "owned_fraction": f,
            "owned_seats": summarize(owned),
            "marginal_seats": summarize(marginal),
            "seats_supplied_by_attributed_set": summarize(head_supply),
            "share_of_committees_head_suffices": float(covered.mean()),
        }
        if covered.mean() > 0.99:
            row["operators_to_coordinate"] = summarize(need_k[covered])
        else:
            row["operators_to_coordinate"] = None
            row["note"] = ("the whole attributed operator set is insufficient; "
                           "the coalition must also reach unattributed stake")
        rows.append(row)

    result = {
        "schema": "empirical-operator-supply-v1",
        "source": ("Rated Network mainnet explorer, node-operator view, "
                   "accessed 2026-07-22"),
        "attributed_operators": len(RATED_OPERATORS),
        "attributed_stake_share": attributed,
        "unattributed_stake_share": unattributed,
        "largest_operator_share": RATED_OPERATORS[0][1] / 100.0,
        "ptc_size": PTC_SIZE, "pivotal": PIVOTAL,
        "committees_per_cell": N_COMMITTEES, "seed": SEED,
        "scope": ("Uses measured shares only; the unattributed remainder is kept "
                  "as one aggregate bucket and never split into invented operators. "
                  "Operator counts use the attacker-optimal largest-first order and "
                  "are lower bounds. Not evidence that any operator would accept a "
                  "bribe."),
        "rows": rows,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "exp95_empirical_operator_concentration.json").write_text(
        json.dumps(result, indent=2) + "\n")

    def fmt(d):
        return f"{d['median']:.0f} [{d['p05']:.0f},{d['p95']:.0f}]"

    out = ["# Empirical PTC operator supply\n\n",
           f"Rated Network node-operator shares (accessed 2026-07-22): "
           f"{len(RATED_OPERATORS)} operators cover {attributed*100:.2f}% of stake; "
           f"largest {RATED_OPERATORS[0][1]}%.\n\n",
           "| attacker stake | marginal seats | seats the attributed set supplies | "
           "committees where it suffices | operators to coordinate |\n"
           "|---:|---:|---:|---:|---|\n"]
    for r in rows:
        ops = (fmt(r["operators_to_coordinate"]) if r["operators_to_coordinate"]
               else "--")
        out.append(f"| {r['owned_fraction']*100:.1f}% | {fmt(r['marginal_seats'])} | "
                   f"{fmt(r['seats_supplied_by_attributed_set'])} | "
                   f"{r['share_of_committees_head_suffices']*100:.2f}% | {ops} |\n")
    (OUT_DIR / "exp95_empirical_operator_concentration.md").write_text("".join(out))
    print("".join(out))


if __name__ == "__main__":
    main()
