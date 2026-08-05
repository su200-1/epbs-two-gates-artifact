"""Experiment 88 -- gap-conditioned mainnet block-value calibration for the
bribery economics (upgrades exp87 from a single-relay 4.8k sample to an
8-relay, 493k-block auction panel, with an auxiliary descriptive stratification
by the contemporaneous public-mempool visibility gap).

Motivation
----------
exp82 fitted a log-normal to two published moments; exp87 replaced that with a
~4,800-block sample from ONE relay's ``proposer_payload_delivered`` API. Both
answer a single question: at each exogenous per-seat reservation price, what
share of blocks (and of total block value) clears the fixed PTC-orphan
break-even cost ``(PAYLOAD_TIMELY_THRESHOLD + 1) * p_seat``?

The PTC-orphan theft is *selective*: it only pays in the high-value tail. This
experiment recomputes the empirical profitable fraction on the complete 8-relay
panel. It also reports a contemporaneous visibility-gap-decile breakdown as a
descriptive auxiliary analysis. That breakdown is not a pre-slot predictor and
is not used as evidence that an attacker can identify profitable slots in advance.

Calibration source
------------------
An 8-relay mainnet block auction panel (``block_value_panel_*.parquet``): one row
per block, ``top_bid_eth`` = max builder bid across 8 major MEV-Boost relays (a
proposer-payment proxy, i.e. the same MEV-plus-priority-fee quantity a
next-proposer would re-capture), ``private_candidate_share`` = block-level
``visibility_gap_share`` (public + private shares sum to 1 exactly). Window:
blocks 24000000-24499999, 2025-12-13 -- 2026-02-20 UTC. The compact four-column
extract bundled with this artifact (see ``DEFAULT_PANEL_CANDIDATES`` below) is
sufficient to reproduce Table 3 of the paper with no external download.

Scope note
----------
This script only *re-calibrates the value distribution* that feeds the bribery
break-even. It does NOT re-run the consensus simulator; the attack mechanism,
the fixed 257-seat pivotal set, and the theft/griefing accounting are
unchanged from exp81-83. ``top_bid_eth`` is a cross-relay max BID (willingness to
pay), which can sit weakly above a single relay's DELIVERED value; we treat it as
a block-value proxy and report both panel-wide and gap-conditioned fractions.

Run:
    python experiments/exp88_gap_conditioned_block_value.py
    # or, to point at a different full panel with the same schema:
    python experiments/exp88_gap_conditioned_block_value.py --panel /path/to/panel.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "figures" / "drl_risk_epbs"
GWEI_PER_ETH = 1e9

# Default panel location: the compact four-column extract bundled with this
# artifact, sufficient on its own to reproduce Table 3. Pass --panel to point at
# a different, full panel (e.g. one that also carries the gap-conditioned
# auxiliary columns beyond what the bundled extract needs).
DEFAULT_PANEL_CANDIDATES = [
    Path(__file__).resolve().parent / "data" / "block_value_panel_24000000_24499999_min.parquet",
]


def _find_default_panel() -> Path | None:
    for p in DEFAULT_PANEL_CANDIDATES:
        if p.exists():
            return p
    return None


def _load_panel(panel_path: Path) -> pd.DataFrame:
    cols = ["block_number", "top_bid_eth", "private_candidate_share", "public_tx_share"]
    df = pd.read_parquet(panel_path, columns=cols)
    df = df.rename(columns={"private_candidate_share": "visibility_gap_share"})
    # Keep blocks with a positive proposer-payment proxy (the re-capture target).
    df = df[df["top_bid_eth"] > 0].copy()
    df = df.dropna(subset=["top_bid_eth", "visibility_gap_share"])
    return df


def _profitable_rows(values: np.ndarray, price_levels: dict[float, float]) -> list[dict]:
    """For each bribe level, the empirical share of blocks and of TOTAL value in
    blocks whose value clears the break-even bribe (1 - ECDF(bribe))."""
    n = len(values)
    total = float(values.sum())
    rows = []
    for per_seat_eth, total_cost_eth in price_levels.items():
        hit = values[values > total_cost_eth]
        rows.append({
            "per_seat_reservation_price_eth": per_seat_eth,
            "total_reservation_cost_eth": total_cost_eth,
            "break_even_eth": total_cost_eth,
            "empirical_frac_blocks": (len(hit) / n) if n else 0.0,
            "empirical_frac_value": (float(hit.sum()) / total) if total else 0.0,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", type=Path, default=None,
                    help="path to a full block-auction panel parquet file "
                         "(optional; the bundled compact extract is used by default)")
    ap.add_argument(
        "--per-seat-prices-eth", type=str,
        default="0,0.00001,0.0001,0.001,0.01,0.1",
        help="Comma-separated exogenous reservation prices per recruited PTC seat.",
    )
    ap.add_argument("--deciles", type=int, default=10)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    panel_path = args.panel or _find_default_panel()
    if panel_path is None or not Path(panel_path).exists():
        raise SystemExit(
            "panel not found; pass --panel /path/to/block_auction_panel.parquet")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = _load_panel(Path(panel_path))
    n = len(df)

    # ---- Reservation-cost break-even levels (identical accounting to exp87) --
    pivotal = config.PAYLOAD_TIMELY_THRESHOLD + 1  # strict > threshold => 257
    per_seat_prices_eth = [float(x) for x in args.per_seat_prices_eth.split(",")]
    if not per_seat_prices_eth or any(p < 0 for p in per_seat_prices_eth):
        ap.error("--per-seat-prices-eth must contain non-negative values")
    price_levels = {
        p: round(pivotal * p, 12) for p in per_seat_prices_eth
    }

    values = df["top_bid_eth"].to_numpy()
    vsorted = np.sort(values)
    median = float(np.median(values))
    mean = float(values.mean())
    top20 = float(vsorted[int(0.8 * n):].sum() / vsorted.sum())

    print(f"Panel: {panel_path}")
    print(f"blocks (top_bid>0): {n:,}")
    print(f"EMPIRICAL block value: median={median:.4f} ETH  mean={mean:.4f} ETH")
    for p in (0.5, 0.8, 0.9, 0.99, 0.999):
        print(f"  p{p*100:>5.1f}: {np.quantile(values, p):.4f} ETH")
    print(f"  top 20% of blocks hold {top20*100:.1f}% of total value")
    print(f"\npivotal={pivotal} (fixed); per-seat reservation price is exogenous")

    panel_rows = _profitable_rows(values, price_levels)
    print("\nPANEL-WIDE (all deciles): share of blocks attackable and share of "
          "total value held by those blocks:")
    for r in panel_rows:
        print(f"  p_seat={r['per_seat_reservation_price_eth']:.5f} ETH  "
              f"break-even={r['break_even_eth']:.5f} ETH"
              f"  -> {r['empirical_frac_blocks']*100:5.1f}% of blocks,"
              f" holding {r['empirical_frac_value']*100:5.1f}% of all value")

    # ---- Gap-conditioned breakdown -----------------------------------------
    q = args.deciles
    df["gap_decile"] = pd.qcut(df["visibility_gap_share"], q,
                               labels=list(range(1, q + 1)))
    total_value_all = float(values.sum())

    decile_summary = []
    decile_profit = []  # long form: (decile, m, frac_blocks, value_share_of_decile,
    #                                  share_of_panel_attackable_value)
    for d, g in df.groupby("gap_decile", observed=True):
        gv = g["top_bid_eth"].to_numpy()
        decile_summary.append({
            "gap_decile": int(d),
            "n_blocks": len(g),
            "gap_mean": float(g["visibility_gap_share"].mean()),
            "top_bid_mean": float(gv.mean()),
            "top_bid_median": float(np.median(gv)),
            "value_share_of_panel": float(gv.sum() / total_value_all),
        })
        for per_seat_eth, total_cost_eth in price_levels.items():
            hit = gv[gv > total_cost_eth]
            decile_profit.append({
                "gap_decile": int(d),
                "per_seat_reservation_price_eth": per_seat_eth,
                "total_reservation_cost_eth": total_cost_eth,
                "break_even_eth": total_cost_eth,
                "frac_blocks_in_decile": len(hit) / len(g) if len(g) else 0.0,
                "value_share_of_decile": float(hit.sum() / gv.sum()) if gv.sum() else 0.0,
                "attackable_value_eth": float(hit.sum()),
            })

    decile_df = pd.DataFrame(decile_summary).sort_values("gap_decile")
    profit_df = pd.DataFrame(decile_profit)

    # For each bribe level, what share of ALL attackable value sits in the top
    # gap decile / top-3 gap deciles? (the "profitable hunting ground" headline)
    concentration = []
    for per_seat_eth, total_cost_eth in price_levels.items():
        sub = profit_df[profit_df["per_seat_reservation_price_eth"] == per_seat_eth]
        tot = sub["attackable_value_eth"].sum()
        top1 = sub[sub["gap_decile"] == q]["attackable_value_eth"].sum()
        top3 = sub[sub["gap_decile"] >= q - 2]["attackable_value_eth"].sum()
        concentration.append({
            "per_seat_reservation_price_eth": per_seat_eth,
            "total_reservation_cost_eth": total_cost_eth,
            "break_even_eth": total_cost_eth,
            "attackable_value_share_top_gap_decile": (top1 / tot) if tot else 0.0,
            "attackable_value_share_top3_gap_deciles": (top3 / tot) if tot else 0.0,
        })
    conc_df = pd.DataFrame(concentration)

    print("\nGAP-CONDITIONED: concentration of attackable value in high-gap slots")
    print(f"(top decile = highest visibility_gap_share; deciles built on {n:,} blocks)")
    for r in concentration:
        print(f"  p_seat={r['per_seat_reservation_price_eth']:.5f} ETH  top gap decile holds"
              f" {r['attackable_value_share_top_gap_decile']*100:5.1f}% of attackable value;"
              f"  top-3 gap deciles hold"
              f" {r['attackable_value_share_top3_gap_deciles']*100:5.1f}%")

    # ---- Persist ------------------------------------------------------------
    decile_df.to_csv(args.out_dir / "exp88_gap_decile_summary.csv", index=False)
    profit_df.to_csv(args.out_dir / "exp88_gap_decile_profit.csv", index=False)
    conc_df.to_csv(args.out_dir / "exp88_attackable_value_concentration.csv", index=False)

    out = args.out_dir / "exp88_gap_conditioned_block_value.json"
    out.write_text(json.dumps({
        "schema": "exp88-v2",
        "source": "8-relay mainnet block auction panel (cross-relay max top_bid_eth)",
        "panel_path": str(panel_path),
        "window_blocks": "24000000-24499999",
        "n": n,
        "median_eth": median, "mean_eth": mean, "top20_value_share": top20,
        "pivotal_members": pivotal,
        "protocol_direct_penalty_eth": 0.0,
        "reservation_price_model": "exogenous_per_seat",
        "per_seat_prices_eth": per_seat_prices_eth,
        "panel_wide_rows": panel_rows,
        "decile_summary": decile_summary,
        "attackable_value_concentration": concentration,
    }, indent=2))
    print(f"\n-> {out}")

    _plot(decile_df, profit_df, price_levels, args.out_dir)


def _plot(decile_df: pd.DataFrame, profit_df: pd.DataFrame,
          price_levels: dict[float, float], out_dir: Path) -> None:
    """Render in the shared BCRA figure house style (see exp14_bcra_assets):
    default matplotlib color cycle, constrained layout, per-axis Title Case
    titles, faint grid, fontsize-8 legend, no dev-noise suptitle. Emits a
    vector PDF (preferred for the manuscript) plus a PNG preview."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"(skipping figure: {e})")
        return

    q = int(decile_df["gap_decile"].max())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2),
                                   constrained_layout=True)

    # Left: value concentration by gap decile. Highlight the two end deciles
    # (the high-gap peak and the low-gap secondary lump) to carry the J-shape.
    colors = ["C1" if d in (1, q) else "C0" for d in decile_df["gap_decile"]]
    ax1.bar(decile_df["gap_decile"], decile_df["value_share_of_panel"] * 100,
            color=colors)
    ax1.set_xlabel("Visibility-gap decile (10 = highest public-mempool gap)")
    ax1.set_ylabel("Share of panel block value (%)")
    ax1.set_title("Block Value Concentrates in High-Gap Slots")
    ax1.set_xticks(decile_df["gap_decile"])
    ax1.grid(alpha=0.25, axis="y")

    # Right: fraction of blocks attackable, by decile, for selected price levels.
    focus = [p for p in (0.00001, 0.0001, 0.001, 0.01) if p in price_levels]
    for per_seat_eth in focus:
        sub = profit_df[
            profit_df["per_seat_reservation_price_eth"] == per_seat_eth
        ].sort_values("gap_decile")
        ax2.plot(sub["gap_decile"], sub["frac_blocks_in_decile"] * 100,
                 marker="o",
                 label=f"{per_seat_eth:g} ETH/seat  (>{price_levels[per_seat_eth]:.3f} ETH)")
    ax2.set_xlabel("Visibility-gap decile")
    ax2.set_ylabel("Attackable blocks in decile (%)")
    ax2.set_title("Attackable Share by Gap Decile")
    ax2.set_xticks(decile_df["gap_decile"])
    ax2.grid(alpha=0.25)
    ax2.legend(title="Reservation price", fontsize=8)

    pdf_path = out_dir / "exp88_gap_conditioned_block_value.pdf"
    png_path = out_dir / "exp88_gap_conditioned_block_value.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=220)
    plt.close(fig)
    print(f"-> {pdf_path}")
    print(f"-> {png_path}")


if __name__ == "__main__":
    main()
