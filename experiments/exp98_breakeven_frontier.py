"""Experiment 98 -- Figure: the break-even frontier.

Turns the paper's economic table (tab:economics / tab:theft) into a picture.
Cost model, identical to exp94:
    marginal seats  = max(257 - 512*f, 0)      f = attacker-owned stake fraction
    recruitment cost = marginal_seats * p_seat  p_seat = exogenous per-seat price
    block t clears break-even  <=>  rho * value_t > recruitment_cost
The curve is the ex-post share of panel blocks that clear that bar -- coverage,
not realized profit.

fig_breakeven_frontier.png : two panels (f = 0 and f = 1/3). x = per-seat price
(log), y = % of blocks clearing break-even, one line per re-capture ratio rho.
Reads the on-chain-verified fresh panel from exp96.

Run: python experiments/exp98_breakeven_frontier.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent / "data"
FIG = Path(__file__).resolve().parent / "figures" / "drl_risk_epbs"
PANEL = DATA / "block_value_panel_delivered_2026H1.parquet"

PTC_SIZE = 512
PIVOTAL = 257
PRICES = np.logspace(-6, -1, 120)          # per-seat reservation price (ETH)
RHOS = (0.25, 0.50, 0.75, 1.00)
OWNED = (0.0, 1 / 3)
MARK_PRICES = (1e-5, 1e-4, 1e-3, 1e-2)     # ticks matching exp94's table

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 11,
    "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
})


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    v = pd.read_parquet(PANEL, columns=["block_value_eth"])["block_value_eth"].to_numpy(float)
    v = v[v > 0]
    print(f"panel: {len(v):,} blocks")

    cmap = plt.cm.viridis(np.linspace(0.1, 0.85, len(RHOS)))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    export = {"schema": "breakeven-frontier-v1", "n_blocks": int(len(v)),
              "panel": str(PANEL), "curves": []}

    for ax, f in zip(axes, OWNED):
        marginal = max(PIVOTAL - PTC_SIZE * f, 0.0)
        for c, rho in zip(cmap, RHOS):
            cost = marginal * PRICES                       # ETH, per price
            share = np.array([(rho * v > k).mean() for k in cost])
            ax.plot(PRICES, 100 * share, color=c, lw=1.9,
                    label=fr"$\rho={rho:.2f}$")
            export["curves"].append({
                "owned_fraction": f, "recapture_ratio": rho,
                "marginal_seats": marginal,
                "prices_eth": PRICES.tolist(),
                "attackable_share_pct": (100 * share).tolist(),
            })
        ax.set_xscale("log")
        ax.set_xlabel("per-seat reservation price (ETH)")
        ax.set_title(fr"$f={f:.3f}$   (marginal seats $={marginal:.0f}$)")
        for mp in MARK_PRICES:
            ax.axvline(mp, color="0.8", ls=":", lw=0.8, zorder=0)
    axes[0].set_ylabel("ex-post break-even coverage (%)")
    axes[0].legend(title="re-capture", fontsize=9, loc="upper right")
    axes[0].set_ylim(0, 100)

    fig.suptitle("Break-even frontier: share of blocks where "
                 r"$\rho\,V_t > (257-512f)\,p_{\mathrm{seat}}$", y=1.02)
    fig.tight_layout()
    out = FIG / "fig_breakeven_frontier.png"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {out}")

    (FIG / "exp98_breakeven_frontier.json").write_text(json.dumps(export) + "\n")


if __name__ == "__main__":
    main()
