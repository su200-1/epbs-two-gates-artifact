"""Experiment 99 -- Figure: temporal stability of the block-value distribution.

Answers the reviewer question "is this an artifact of one window?" by showing
the distribution is stable across the whole 2026-02..07 span.

fig_temporal_stability.png : two stacked panels sharing the time axis.
  (top)    daily median / p95 / p99 of block value (log y);
  (bottom) daily share of blocks profitable to attack at a fixed operating
           point (f=0, p_seat=0.001 ETH, rho=1.0) -- one concrete slice of the
           break-even frontier tracked over time.

Reads the on-chain-verified fresh panel from exp96.

Run: python experiments/exp99_temporal_stability.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent / "data"
FIG = Path(__file__).resolve().parent / "figures" / "drl_risk_epbs"
PANEL = DATA / "block_value_panel_delivered_2026H1.parquet"

PTC_SIZE = 512
PIVOTAL = 257
OP_PRICE = 0.001     # ETH per seat
OP_RHO = 1.00
OP_F = 0.0

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 11,
    "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
})


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(PANEL, columns=["block_value_eth", "day"])
    df = df[df.block_value_eth > 0]
    cost = max(PIVOTAL - PTC_SIZE * OP_F, 0.0) * OP_PRICE

    g = df.groupby("day")["block_value_eth"]
    daily = pd.DataFrame({
        "median": g.median(),
        "p95": g.quantile(0.95),
        "p99": g.quantile(0.99),
        "attackable": g.apply(lambda s: float((OP_RHO * s.to_numpy() > cost).mean())),
    })
    daily.index = pd.to_datetime(daily.index)
    daily = daily.sort_index()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True,
                                   height_ratios=[3, 2])

    ax1.plot(daily.index, daily["p99"], color="#c0504d", lw=1.4, label="p99")
    ax1.plot(daily.index, daily["p95"], color="#e0a030", lw=1.4, label="p95")
    ax1.plot(daily.index, daily["median"], color="#1f4e79", lw=1.6, label="median")
    ax1.set_yscale("log")
    ax1.set_ylabel("block value (ETH)")
    ax1.set_title("Daily block-value quantiles are stable across the panel")
    ax1.legend(ncol=3, fontsize=9, loc="upper right")

    ax2.plot(daily.index, 100 * daily["attackable"], color="#4a7", lw=1.6)
    ax2.fill_between(daily.index, 0, 100 * daily["attackable"],
                     color="#4a7", alpha=0.18)
    ax2.set_ylabel("blocks profitable (%)")
    ax2.set_title(fr"Daily attackable share at $f=0,\ "
                  fr"p_{{\mathrm{{seat}}}}=0.001$ ETH, $\rho=1$ "
                  fr"(cost $={cost:.3f}$ ETH)")
    ax2.set_ylim(bottom=0)
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    fig.tight_layout()
    out = FIG / "fig_temporal_stability.png"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {out}")

    print("\nday-level summary (mean across days):")
    print(f"  median block value: {daily['median'].mean():.4f} ETH "
          f"(range {daily['median'].min():.4f}..{daily['median'].max():.4f})")
    print(f"  attackable share:   {100*daily['attackable'].mean():.2f}% "
          f"(range {100*daily['attackable'].min():.2f}"
          f"..{100*daily['attackable'].max():.2f}%)")


if __name__ == "__main__":
    main()
