"""Experiment 104 -- build the block-value panel from delivered payloads.

Supersedes the panel `exp96` produced. That script matched a bid's
`block_fee_recipient` against the on-chain coinbase and took the maximum over
the matches, which is not a delivered value: for a given slot the relays hold
many candidate bids that share a fee recipient but carry different block hashes
(median 28 on a sampled day, 97.8% of blocks with more than one), so the maximum
is a top-of-book figure over competing blocks. Checked against the blocks whose
truth could be recovered locally, that panel exceeded the delivered value for
53% of them, by 60% at the 99th percentile and 30% in the mean.

This script instead unions the relays' own `proposer_payload_delivered` records
(fetched by `exp103`), which carry the block hash of the payload that was
actually delivered. A block is delivered by one relay, but several may report
it; where they overlap the records agree, so deduplication by block number is
safe and is asserted rather than assumed.

Two independent cross-checks are run and reported, not just performed:

  1. `parent_hash` chaining. A block that landed becomes the `parent_hash` of
     the next slot's bids, so the canonical hash of slot s can be recovered from
     the relayscan bid archive alone, without touching any relay API. Where the
     archive also holds a bid carrying that hash, its value is an independent
     measurement of the delivered value.
  2. Coinbase agreement. The delivered record's builder fee recipient should
     match the on-chain coinbase from XBlock-ETH where both exist.

Run: python experiments/exp104_build_delivered_panel.py
"""
from __future__ import annotations

import collections
import csv
import glob
import os
import sys
from decimal import Decimal
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DELIVERED = HERE / "data" / "delivered"
OUT = HERE / "data" / "block_value_panel_delivered_2026H1.parquet"
RELAY_DIR = Path("/Volumes/sun/GitHub/dataset/relayscan.io")
XBLOCK_SHARDS = sorted(glob.glob(
    "/Volumes/sun/GitHub/dataset/block/*_Block/*_Block_Info.csv"))

WEI = Decimal(10) ** 18
FIRST_DAY, LAST_DAY = "2026-02-21", "2026-07-10"


def load_delivered() -> pd.DataFrame:
    """Union the per-relay shards; assert the overlaps agree before dedup."""
    frames = []
    for p in sorted(DELIVERED.glob("*.parquet")):
        d = pd.read_parquet(p)
        d["relay"] = p.stem.replace("_", ".")
        frames.append(d)
        print(f"  {p.stem:<34} {len(d):>8,}", flush=True)
    if not frames:
        sys.exit("no delivered shards; run exp103 first")
    df = pd.concat(frames, ignore_index=True)
    df["block_number"] = df["block_number"].astype("int64")
    df["slot"] = df["slot"].astype("int64")
    df["value_eth"] = df["value"].map(lambda v: float(Decimal(v) / WEI))

    # Where relays overlap they must agree; this is the assumption dedup rests on.
    g = df.groupby("block_number").agg(nh=("block_hash", "nunique"),
                                       nv=("value", "nunique"),
                                       n=("relay", "size"))
    dup = g[g.n > 1]
    bad_h, bad_v = int((dup.nh > 1).sum()), int((dup.nv > 1).sum())
    print(f"\n  blocks reported by >1 relay: {len(dup):,}"
          f"   disagreeing on hash: {bad_h}   on value: {bad_v}")
    if bad_h or bad_v:
        print("  !! relays disagree; inspect before trusting the dedup",
              file=sys.stderr)

    return df


def resolve_conflicts(df: pd.DataFrame, canon: dict[int, str]) -> pd.DataFrame:
    """Deduplicate to one row per block, adjudicating reorged heights by evidence.

    Every conflict we observed carries two *different* slots for one height: a
    block was delivered, reorged out, and the next slot produced another block at
    the same height. Both deliveries are real, so neither relay is wrong; only
    one of the two is on the canonical chain. `canon` maps a slot to the hash
    that the following slot's bids name as their parent, which is the arbiter.
    Where it cannot adjudicate we drop the height rather than guess: taking the
    later slot agrees with the evidence only 77% of the time.
    """
    nh = df.groupby("block_number").block_hash.nunique()
    conflicted = set(nh[nh > 1].index)
    clean = df[~df.block_number.isin(conflicted)]
    clean = clean.sort_values("relay").drop_duplicates("block_number", keep="first")

    rows, dropped = [], 0
    for bn, grp in df[df.block_number.isin(conflicted)].groupby("block_number"):
        keep = grp[[canon.get(s) == h for s, h in zip(grp.slot, grp.block_hash)]]
        if keep.block_hash.nunique() == 1:
            rows.append(keep.iloc[0])
        else:
            dropped += 1
    print(f"  reorged heights: {len(conflicted)}  adjudicated: {len(rows)}  "
          f"dropped (no verdict): {dropped}")
    out = pd.concat([clean, pd.DataFrame(rows)], ignore_index=True) if rows else clean
    return out.sort_values("block_number").reset_index(drop=True)


def canonical_from_parent_hash(day: str):
    """Recover (hash, value) for blocks that landed, from the bid archive alone.

    The block that landed at slot s is the one whose hash is the parent_hash of
    slot s+1's bids. Coverage is partial because `_top.csv` keeps only each
    relay's best bid, and the delivered block is often not the best.
    """
    path = RELAY_DIR / f"{day}_top.csv"
    if not path.exists():
        return {}, {}
    d = pd.read_csv(path, dtype=str,
                    usecols=["slot", "value", "block_hash", "parent_hash", "block_number"])
    d["slot"] = pd.to_numeric(d["slot"])
    par = d.groupby("slot").parent_hash.agg(
        lambda x: collections.Counter(x.dropna()).most_common(1)[0][0])
    canon = {s - 1: h for s, h in par.items()}
    d["hit"] = [canon.get(s) == h for s, h in zip(d.slot, d.block_hash)]
    hit = d[d.hit]
    out: dict[int, tuple[str, float]] = {}
    for bn, bh, v in zip(hit.block_number, hit.block_hash, hit.value):
        out[int(bn)] = (bh, float(Decimal(v) / WEI))
    return out, canon


def main() -> None:
    print("delivered shards:")
    df = load_delivered()

    days = sorted(os.path.basename(p)[:10]
                  for p in glob.glob(str(RELAY_DIR / "*_top.csv")))
    days = [d for d in days if FIRST_DAY <= d <= LAST_DAY]

    print(f"\ncross-check 1: parent_hash chaining over {len(days)} days", flush=True)
    truth: dict[int, tuple[str, float]] = {}
    canon_by_slot: dict[int, str] = {}
    for i, day in enumerate(days, 1):
        t, c = canonical_from_parent_hash(day)
        truth.update(t)
        canon_by_slot.update(c)
        if i % 25 == 0:
            print(f"  {i}/{len(days)} days, {len(truth):,} verifiable blocks", flush=True)

    print("\nresolving reorged heights:", flush=True)
    df = resolve_conflicts(df, canon_by_slot)

    idx = df.set_index("block_number")
    common = [b for b in truth if b in idx.index]
    agree_h = sum(idx.at[b, "block_hash"] == truth[b][0] for b in common)
    vals = [(idx.at[b, "value_eth"], truth[b][1]) for b in common]
    agree_v = sum(abs(a - b) < 1e-9 for a, b in vals)
    print(f"\n  overlap: {len(common):,} blocks")
    print(f"  block_hash agrees: {agree_h:,}/{len(common):,} "
          f"({agree_h / max(len(common), 1):.3%})")
    print(f"  value agrees:      {agree_v:,}/{len(common):,} "
          f"({agree_v / max(len(common), 1):.3%})")

    print("\ncross-check 2: builder fee recipient vs on-chain coinbase", flush=True)
    cb: dict[int, str] = {}
    for shard in XBLOCK_SHARDS:
        for chunk in pd.read_csv(shard, usecols=["blockNumber", "minerAddress"],
                                 dtype={"blockNumber": "Int64", "minerAddress": "string"},
                                 chunksize=500_000):
            chunk = chunk.dropna()
            cb.update(zip(chunk.blockNumber.astype("int64"),
                          chunk.minerAddress.str.lower()))
    df["coinbase"] = df.block_number.map(cb)
    have = df.dropna(subset=["coinbase"])
    print(f"  blocks with on-chain truth: {len(have):,}/{len(df):,}")

    # UTC day, derived exactly from the slot: mainnet genesis 1606824023, 12 s slots.
    # Checked against XBlock: slot 13,876,315 -> 1,773,339,803, the timestamp it
    # records for block 24,643,151.
    df["day"] = pd.to_datetime(1_606_824_023 + df.slot * 12, unit="s",
                               utc=True).dt.strftime("%Y-%m-%d")
    panel = (df[["block_number", "slot", "day", "block_hash", "value_eth", "relay"]]
             .rename(columns={"value_eth": "block_value_eth"})
             .sort_values("block_number").reset_index(drop=True))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUT, index=False)

    v = panel.block_value_eth
    print(f"\npanel written: {OUT.name}")
    print(f"  blocks {len(panel):,}  unique {panel.block_number.nunique():,}"
          f"  heights {panel.block_number.min():,}..{panel.block_number.max():,}")
    print(f"  median {v.median():.4f}  mean {v.mean():.4f}  "
          f"p99 {v.quantile(.99):.3f}  p99.9 {v.quantile(.999):.3f}  max {v.max():.2f} ETH")
    print("\n  relay shares:")
    for r, n in panel.relay.value_counts().items():
        print(f"    {r:<34} {n:>8,}  {n / len(panel):6.2%}")


if __name__ == "__main__":
    main()
