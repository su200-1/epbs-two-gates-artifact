"""Experiment 96 -- build a fresh, on-chain-verified block-value panel from the
2026 H1 RelayScan top-bid archives.

Why this exists
---------------
The earlier panel (blocks 24.0M-24.5M, 2025-12 to 2026-02) took the cross-relay
MAX of *received* bids and dropped obvious spam with a crude >=100 ETH cap. That
cap is arbitrary: it removes the 11,000 ETH garbage (all from one relay,
relay.ethgas.com, with an empty block_fee_recipient) but cannot remove a
plausible-looking bid that was never delivered on-chain.

This build uses the principled filter instead: a bid counts only if it was the
block that actually landed on-chain. We do not have per-bid block_hash on the
execution side, but XBlock-ETH gives us each block's coinbase (minerAddress),
which for an MEV-Boost block equals the winning builder's block_fee_recipient.
So for each block_number we keep only relay bids whose block_fee_recipient
matches the on-chain coinbase, and take the max of those -- that max is the
delivered block's value, i.e. the payment the proposer actually received.

Consequences, all correct:
  * every never-delivered phantom bid (incl. the ethgas spam) is dropped;
  * the value is the on-chain builder->proposer payment, not a raw offer;
  * locally-built blocks (no builder bid matches the coinbase) are dropped,
    which is right -- they carry no relay/builder payment.

Coverage
--------
Relay top-bid archives:  2026-02-21 .. 2026-07-20 (147 daily files).
XBlock coinbase truth:   blocks 24,500,000 .. 25,499,999 (2026-02-20 .. 07-10).
The panel is therefore the overlap: ~2026-02-21 .. 2026-07-10, ~1.0M blocks.
Relay days past 2026-07-10 have no on-chain truth yet and are skipped.

Output
------
experiments/data/block_value_panel_fresh_2026H1.parquet with columns
    block_number : int
    block_value_eth : float   # delivered (coinbase-matched) bid, = on-chain pay
    raw_top_eth : float       # cross-relay max after a <100 ETH spam cap, kept
                              # only so a delivered-vs-raw robustness plot is
                              # possible; NOT the panel's value column
    day : str                 # YYYY-MM-DD (UTC) of the relay archive

This script is optional for reproduction: its output,
experiments/data/block_value_panel_fresh_2026H1.parquet, is already bundled, so
exp97-exp100 run without it. Re-running it from scratch additionally needs the
raw RelayScan top-bid archives and XBlock-ETH block-info CSVs on local disk (not
bundled, too large); point RELAY_DIR and BLOCK_INFO_FILES below at your copies.

Run: python experiments/exp96_fresh_panel_build.py
"""
from __future__ import annotations

import glob
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RELAY_DIR = Path("/Volumes/sun/GitHub/dataset/relayscan.io")
XBLOCK_SHARDS = [
    "/Volumes/sun/GitHub/dataset/block/24500000to24749999_Block/24500000to24749999_Block_Info.csv",
    "/Volumes/sun/GitHub/dataset/block/24750000to24999999_Block/24750000to24999999_Block_Info.csv",
    "/Volumes/sun/GitHub/dataset/block/25000000to25249999_Block/25000000to25249999_Block_Info.csv",
    "/Volumes/sun/GitHub/dataset/block/25250000to25499999_Block/25250000to25499999_Block_Info.csv",
]
OUT = Path(__file__).resolve().parent / "data" / "block_value_panel_fresh_2026H1.parquet"
SPAM_CAP_ETH = 100.0          # only for the auxiliary raw_top_eth column
LAST_DAY = "2026-07-10"       # last day with XBlock on-chain truth
WEI = 1e18


def load_coinbase_map() -> dict[int, str]:
    """block_number -> on-chain coinbase (lowercased minerAddress)."""
    cb: dict[int, str] = {}
    for shard in XBLOCK_SHARDS:
        print(f"  reading coinbase truth: {os.path.basename(shard)}", flush=True)
        for chunk in pd.read_csv(shard, usecols=["blockNumber", "minerAddress"],
                                 dtype={"blockNumber": "Int64", "minerAddress": "string"},
                                 chunksize=500_000):
            chunk = chunk.dropna()
            for bn, addr in zip(chunk["blockNumber"].astype("int64"),
                                chunk["minerAddress"].str.lower()):
                cb[int(bn)] = addr
    print(f"  coinbase map: {len(cb):,} blocks", flush=True)
    return cb


def process_day(path: str, cb: dict[int, str]) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path, usecols=["block_number", "value", "block_fee_recipient"],
                         dtype={"block_number": "Int64", "value": "string",
                                "block_fee_recipient": "string"})
    except Exception as e:  # noqa: BLE001
        print(f"  !! skip {os.path.basename(path)}: {e}", flush=True)
        return None
    day = re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(path)).group(1)

    df["block_number"] = df["block_number"].astype("Int64")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["block_number", "value"])
    df["block_number"] = df["block_number"].astype("int64")
    df["value_eth"] = df["value"].astype(float) / WEI
    df["fee"] = df["block_fee_recipient"].str.lower()

    # on-chain coinbase for each block; rows with no truth are dropped
    df["coinbase"] = df["block_number"].map(cb)
    df = df.dropna(subset=["coinbase"])

    # (a) delivered value: bids whose fee_recipient matches the coinbase, max
    matched = df[df["fee"] == df["coinbase"]]
    delivered = (matched.groupby("block_number")["value_eth"].max()
                 .rename("block_value_eth"))

    # (b) auxiliary raw cross-relay max after a crude spam cap, for comparison
    capped = df[df["value_eth"] < SPAM_CAP_ETH]
    raw = (capped.groupby("block_number")["value_eth"].max().rename("raw_top_eth"))

    out = pd.concat([delivered, raw], axis=1).reset_index()
    out["day"] = day
    return out


def main() -> None:
    files = sorted(p for p in glob.glob(str(RELAY_DIR / "2026-*_top.csv"))
                   if os.path.isfile(p))
    files = [f for f in files
             if re.search(r"(\d{4}-\d{2}-\d{2})", f).group(1) <= LAST_DAY]
    print(f"relay day files to process (<= {LAST_DAY}): {len(files)}", flush=True)

    print("loading on-chain coinbase truth...", flush=True)
    cb = load_coinbase_map()

    parts = []
    for i, f in enumerate(files, 1):
        part = process_day(f, cb)
        if part is not None and len(part):
            parts.append(part)
            print(f"  [{i:3d}/{len(files)}] {os.path.basename(f)}: "
                  f"{len(part):,} delivered blocks", flush=True)

    panel = pd.concat(parts, ignore_index=True)
    panel = panel.dropna(subset=["block_value_eth"])
    panel = panel.sort_values("block_number").reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUT, index=False)

    v = panel["block_value_eth"].to_numpy()
    print("\n================ PANEL SUMMARY ================", flush=True)
    print(f"rows (delivered blocks): {len(panel):,}", flush=True)
    print(f"block range: {panel['block_number'].min():,} .. "
          f"{panel['block_number'].max():,}", flush=True)
    print(f"day range:   {panel['day'].min()} .. {panel['day'].max()}", flush=True)
    print(f"block_value_eth: median={np.median(v):.4f}  mean={np.mean(v):.4f}  "
          f"p95={np.percentile(v,95):.3f}  p99={np.percentile(v,99):.3f}  "
          f"p99.9={np.percentile(v,99.9):.3f}  max={v.max():.2f}", flush=True)
    print("paper's current panel:  median=0.012  mean=0.039  p99.9=2.8", flush=True)
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
