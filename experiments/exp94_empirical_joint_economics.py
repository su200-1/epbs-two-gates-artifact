"""RETIRED -- retired: ran on the top-bid panel, which is no longer distributed, and backs
no claim in the paper.

Kept for provenance; not part of the reproduction path.
"""

"""Empirical block-value x recruitment-cost x recapture-ratio sensitivity.

Crosses attacker-owned PTC seat share, exogenous per-seat reservation price, and
re-capture ratio against the bundled block-value panel, reporting ex-post panel
coverage rather than a deployable targeting result.

Run: python experiments/exp94_empirical_joint_economics.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent / "figures" / "drl_risk_epbs"
PANEL = (
    Path(__file__).resolve().parent
    / "data"
    / "block_value_panel_24000000_24499999_min.parquet"
)
PTC_SIZE = 512
PIVOTAL = 257
OWNED_FRACTIONS = (0.0, 0.10, 0.20, 1 / 3)
PER_SEAT_PRICES = (0.00001, 0.0001, 0.001, 0.01)
RECAPTURE_RATIOS = (0.25, 0.50, 0.75, 1.00)


def main() -> None:
    df = pd.read_parquet(PANEL, columns=["block_number", "top_bid_eth"])
    values = df.loc[df.top_bid_eth > 0, "top_bid_eth"].to_numpy(float)
    total_value = float(values.sum())
    rows = []
    for owned_fraction in OWNED_FRACTIONS:
        # Expected-seat sensitivity; committee variance is reported separately in exp93.
        expected_owned = PTC_SIZE * owned_fraction
        marginal_seats = max(PIVOTAL - expected_owned, 0.0)
        for price in PER_SEAT_PRICES:
            recruitment_cost = marginal_seats * price
            for rho in RECAPTURE_RATIOS:
                recoverable = rho * values
                profitable = recoverable > recruitment_cost
                net = np.where(profitable, recoverable - recruitment_cost, 0.0)
                rows.append({
                    "owned_fraction": owned_fraction,
                    "expected_owned_seats": expected_owned,
                    "marginal_recruited_seats": marginal_seats,
                    "per_seat_price_eth": price,
                    "recruitment_cost_eth": recruitment_cost,
                    "recapture_ratio": rho,
                    "ex_post_fraction_blocks_clearing_cost": float(profitable.mean()),
                    "share_panel_value_in_clearing_blocks": float(values[profitable].sum() / total_value),
                    "counterfactual_mean_net_eth_per_panel_block": float(net.mean()),
                })

    result = {
        "schema": "empirical-joint-economics-v1",
        "panel": str(PANEL),
        "n_positive_bid_blocks": int(len(values)),
        "pivotal_seats": PIVOTAL,
        "cost_model": "(257 - 512*f) * exogenous per-seat price",
        "scope": (
            "Ex-post sensitivity using top_bid_eth as a value proxy. It does not "
            "establish ex-ante targetability, transaction replayability, market "
            "clearing, or realized attack profit."
        ),
        "rows": rows,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "exp94_empirical_joint_economics.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )

    focus = [r for r in rows if r["per_seat_price_eth"] == 0.001]
    lines = [
        "# Empirical joint economic sensitivity\n\n",
        "Selected slice at 0.001 ETH/seat. Percentages are ex-post panel "
        "coverage, not a deployable targeting result.\n\n",
        "| Owned stake | Marginal seats | Cost (ETH) | rho=0.25 | rho=0.50 | rho=0.75 | rho=1.00 |\n",
        "|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for f in OWNED_FRACTIONS:
        sub = [r for r in focus if r["owned_fraction"] == f]
        sub.sort(key=lambda r: r["recapture_ratio"])
        lines.append(
            f"| {100*f:.1f}% | {sub[0]['marginal_recruited_seats']:.1f} | "
            f"{sub[0]['recruitment_cost_eth']:.3f} | "
            + " | ".join(f"{100*r['ex_post_fraction_blocks_clearing_cost']:.2f}%" for r in sub)
            + " |\n"
        )
    (OUT_DIR / "exp94_empirical_joint_economics.md").write_text("".join(lines))
    print("".join(lines))


if __name__ == "__main__":
    main()
