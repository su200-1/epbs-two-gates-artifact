"""Experiment 97 -- Figure: the empirical block-value distribution, and its
reproducibility across an independent fresh panel.

Backs the distribution numbers that the break-even argument rests on (currently
only stated as bare prose: median 0.012, mean 0.039, p99.9 2.8 ETH). Two figures:

  fig_block_value_distribution.png
    (left)  histogram of log10(block value) -- shows the bulk sits far below
            1 ETH;
    (right) complementary CDF on log-log axes -- shows the heavy tail: a thin
            fraction of blocks carries the value that makes an attack pay.

  fig_panel_reproducibility.png
    CCDF of the OLD panel (blocks 24.0M-24.5M, 2025-12..2026-02) overlaid on the
    NEW on-chain-verified panel (2026-02..07). If the headline shape reproduces
    on independent, fresher, more strictly cleaned data, the economic claim is
    not an artifact of one window.

Run (after exp96 builds the fresh panel):
    python experiments/exp97_block_value_distribution.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent / "data"
FIG = Path(__file__).resolve().parent / "figures" / "drl_risk_epbs"
NEW_PANEL = DATA / "block_value_panel_delivered_2026H1.parquet"
# The earlier top-bid panel this script used to compare against has been
# retired; the delivered panel is the only one the paper uses.

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200,
    "font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
})
INK = "#1f4e79"
INK2 = "#c0504d"


def ccdf(v: np.ndarray):
    x = np.sort(v)
    y = 1.0 - np.arange(len(x)) / len(x)
    return x, y


def load(panel: Path, col: str) -> np.ndarray:
    df = pd.read_parquet(panel, columns=[col])
    v = df[col].to_numpy(float)
    return v[v > 0]


def annot(v: np.ndarray) -> str:
    return (f"n={len(v):,}\nmedian={np.median(v):.4f}\nmean={np.mean(v):.4f}\n"
            f"p99={np.percentile(v,99):.3f}\np99.9={np.percentile(v,99.9):.3f}\n"
            f"max={v.max():.2f} ETH")


def fig_distribution(v: np.ndarray) -> None:
    fig, (axh, axc) = plt.subplots(1, 2, figsize=(11, 4.4))

    logv = np.log10(v)
    axh.hist(logv, bins=np.linspace(-4, 3, 80), color=INK, alpha=0.85)
    axh.set_xlim(-4, 3)
    axh.set_xlabel(r"block value  ($\log_{10}$ ETH)")
    axh.set_ylabel("blocks")
    axh.set_title("(a) Bulk: most blocks are cheap")
    ytop = axh.get_ylim()[1]
    for q, lab, ha, yf in [(np.median(v), "median", "right", 0.96),
                           (np.mean(v), "mean", "left", 0.80)]:
        axh.axvline(np.log10(q), color=INK2, ls="--", lw=1.2)
        side = " " if ha == "left" else ""
        axh.text(np.log10(q), ytop * yf,
                 f"{side}{lab}={q:.3f} ", color=INK2, fontsize=9,
                 va="top", ha=ha)

    x, y = ccdf(v)
    axc.loglog(x, y, color=INK, lw=1.8)
    axc.set_xlim(1e-4, 2e3)
    axc.set_ylim(1e-6, 1.3)
    axc.set_xlabel("block value  (ETH)")
    axc.set_ylabel(r"P(value $>$ x)")
    axc.set_title("(b) Tail: value concentrates in a thin slice")
    for p, yf in ((99, 3.0e-3), (99.9, 3.0e-4)):
        xv = np.percentile(v, p)
        axc.axvline(xv, color=INK2, ls=":", lw=1.0)
        axc.text(xv * 1.15, yf, f"p{p}={xv:.2f}", color=INK2, fontsize=8.5,
                 ha="left", va="center")
    axc.text(0.03, 0.03, annot(v), transform=axc.transAxes, fontsize=8.5,
             va="bottom", ha="left",
             bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))

    fig.suptitle("Empirical block-value distribution "
                 "(on-chain-verified RelayScan panel, 2026-02..07)", y=1.02)
    fig.tight_layout()
    out = FIG / "fig_block_value_distribution.png"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {out}")


def fig_reproducibility(v_new: np.ndarray) -> None:
    if not OLD_PANEL.exists():
        print(f"(skip reproducibility fig: {OLD_PANEL} not found)")
        return
    v_old = load(OLD_PANEL, "top_bid_eth")

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for v, c, lab in [
        (v_old, "#888888",
         f"old panel 2025-12..2026-02 (raw top bid)\n"
         f"med={np.median(v_old):.3f}, mean={np.mean(v_old):.3f}, "
         f"p99.9={np.percentile(v_old,99.9):.2f}"),
        (v_new, INK,
         f"new panel 2026-02..07 (on-chain delivered)\n"
         f"med={np.median(v_new):.3f}, mean={np.mean(v_new):.3f}, "
         f"p99.9={np.percentile(v_new,99.9):.2f}"),
    ]:
        x, y = ccdf(v)
        ax.loglog(x, y, color=c, lw=1.8, label=lab)
    ax.set_xlim(1e-4, 2e3)
    ax.set_ylim(1e-6, 1.3)
    ax.set_xlabel("block value (ETH)")
    ax.set_ylabel(r"P(value $>$ x)")
    ax.set_title("Reproducibility: the tail shape holds on independent fresh data")
    ax.legend(fontsize=8.5, loc="lower left")
    fig.tight_layout()
    out = FIG / "fig_panel_reproducibility.png"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {out}")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    v = load(NEW_PANEL, "block_value_eth")
    print(f"fresh panel: {len(v):,} blocks; " + annot(v).replace("\n", "  "))
    fig_distribution(v)
    fig_reproducibility(v)


if __name__ == "__main__":
    main()
