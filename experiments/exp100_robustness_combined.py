"""Experiment 100 -- combined robustness figure (merges the former temporal and
reproducibility panels into one two-panel figure, to keep Section 3 lean).

fig_robustness_combined.pdf:
  (a) Reproducibility: CCDF of the older 493k panel (raw top bid, 2025-12..2026-02)
      overlaid on the primary on-chain-verified panel (delivered payment,
      2026-02..07). The tail shape coincides.
  (b) Temporal stability: daily median / p95 / p99 block value across the primary
      panel. Quantiles are stable across the five-month span.

Reads the fresh panel from exp96 and the older bundled panel.

Run: python experiments/exp100_robustness_combined.py
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
NEW_PANEL = DATA / "block_value_panel_fresh_2026H1.parquet"
OLD_PANEL = DATA / "block_value_panel_24000000_24499999_min.parquet"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 11,
    "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
})
INK = "#1f4e79"


def ccdf(v):
    x = np.sort(v)
    return x, 1.0 - np.arange(len(x)) / len(x)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    new = pd.read_parquet(NEW_PANEL, columns=["block_value_eth", "day"])
    new = new[new.block_value_eth > 0]
    vnew = new["block_value_eth"].to_numpy(float)

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11, 4.3))

    # (a) reproducibility
    if OLD_PANEL.exists():
        vold = pd.read_parquet(OLD_PANEL, columns=["top_bid_eth"])["top_bid_eth"].to_numpy(float)
        vold = vold[vold > 0]
        x, y = ccdf(vold)
        axa.loglog(x, y, color="#888888", lw=1.7,
                   label=f"older panel, raw top bid\n(2025-12..2026-02; "
                         f"med {np.median(vold):.3f}, p99.9 {np.percentile(vold,99.9):.2f})")
    x, y = ccdf(vnew)
    axa.loglog(x, y, color=INK, lw=1.8,
               label=f"primary panel, delivered\n(2026-02..07; "
                     f"med {np.median(vnew):.3f}, p99.9 {np.percentile(vnew,99.9):.2f})")
    axa.set_xlim(1e-4, 2e3)
    axa.set_ylim(1e-6, 1.3)
    axa.set_xlabel("block value (ETH)")
    axa.set_ylabel(r"P(value $>$ x)")
    axa.set_title("(a) Reproducibility across panels")
    axa.legend(fontsize=8, loc="lower left")

    # (b) temporal
    g = new.groupby("day")["block_value_eth"]
    daily = pd.DataFrame({"median": g.median(), "p95": g.quantile(0.95),
                          "p99": g.quantile(0.99)})
    daily.index = pd.to_datetime(daily.index)
    daily = daily.sort_index()
    axb.plot(daily.index, daily["p99"], color="#c0504d", lw=1.3, label="p99")
    axb.plot(daily.index, daily["p95"], color="#e0a030", lw=1.3, label="p95")
    axb.plot(daily.index, daily["median"], color=INK, lw=1.5, label="median")
    axb.set_yscale("log")
    axb.set_ylabel("block value (ETH)")
    axb.set_title("(b) Daily quantiles are stable")
    axb.legend(ncol=3, fontsize=9, loc="upper right")
    axb.xaxis.set_major_locator(mdates.MonthLocator())
    axb.xaxis.set_major_formatter(mdates.DateFormatter("%m"))
    axb.set_xlabel("month of 2026")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig_robustness_combined.{ext}", bbox_inches="tight")
    print(f"wrote {FIG / 'fig_robustness_combined.pdf'}")


if __name__ == "__main__":
    main()
