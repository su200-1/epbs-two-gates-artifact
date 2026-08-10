"""Experiment 103 -- fetch the relays' `proposer_payload_delivered` records.

Why this exists
---------------
`exp96` built the block-value panel by matching a bid's `block_fee_recipient`
against the on-chain coinbase and taking the maximum over the matches. That is
not a delivered value: for a given slot the relays receive many candidate bids
that share a fee recipient but carry *different* block hashes, so the maximum is
a top-of-book figure over ~28 competing blocks rather than the payment that
actually settled. On 2026-03-12 the resulting panel value exceeded the truth for
53% of the blocks it could be checked against, by 60% at the 99th percentile.

Each relay publishes what it actually delivered, with the block hash, through
`/relay/v1/data/bidtraces/proposer_payload_delivered`. This script walks that
endpoint backwards by cursor for every relay and stores the raw records; a block
is delivered by exactly one relay, so the union across relays is the panel.

The endpoint caps `limit` at 200 and is walked by decreasing `cursor` (a slot).
Relays are independent hosts, so they are fetched concurrently; within a host we
keep the concurrency low deliberately.

Resumable: each relay writes `data/delivered/<relay>.parquet` plus a `.cursor`
checkpoint, so an interrupted run continues where it stopped.

Run: python experiments/exp103_fetch_delivered_payloads.py [--from SLOT --to SLOT]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

OUT_DIR = Path(__file__).resolve().parent / "data" / "delivered"

# The eight relays the paper's panel draws on. `relay.ethgas.com` is included
# for completeness; it delivers a small share.
RELAYS = [
    "relay.ultrasound.money",
    "boost-relay.flashbots.net",
    "titanrelay.xyz",
    "agnostic-relay.net",
    "aestus.live",
    "bloxroute.max-profit.blxrbdn.com",
    "bloxroute.regulated.blxrbdn.com",
    "relay.ethgas.com",
]

# Slot bounds of the panel window, 2026-02-21 .. 2026-07-10 inclusive,
# read off the relayscan day files.
SLOT_LO = 13_733_999   # 2026-02-21T00:00:11Z; 13_733_998 is still 2026-02-20
SLOT_HI = 14_741_998

PAGE = 200            # endpoint maximum for most relays
# bloxroute caps `limit` at 100 and answers 400 above it.
PAGE_OVERRIDE = {
    "bloxroute.max-profit.blxrbdn.com": 100,
    "bloxroute.regulated.blxrbdn.com": 100,
}
TIMEOUT = 25
RETRIES = 4
GAP_STEP = 200        # how far to skip back when a window returns nothing


def fetch_page(relay: str, cursor: int) -> list[dict]:
    """One page of delivered payloads at or below `cursor`."""
    page = PAGE_OVERRIDE.get(relay, PAGE)
    url = (f"https://{relay}/relay/v1/data/bidtraces/proposer_payload_delivered"
           f"?limit={page}&cursor={cursor}")
    last = None
    for attempt in range(RETRIES):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "epbs-artifact/1.0"})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                return json.loads(r.read().decode())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as e:
            last = e
            time.sleep(1.5 * (attempt + 1))
    print(f"  !! {relay} cursor={cursor}: {last}", file=sys.stderr, flush=True)
    return []


def walk(relay: str, lo: int, hi: int) -> int:
    """Walk one relay from `hi` down to `lo`, appending as we go. Returns rows."""
    safe = relay.replace(".", "_")
    out = OUT_DIR / f"{safe}.parquet"
    ck = OUT_DIR / f"{safe}.cursor"

    rows: list[dict] = []
    if out.exists():
        rows = pd.read_parquet(out).to_dict("records")
    cursor = int(ck.read_text()) if ck.exists() else hi
    if cursor < lo:
        print(f"  {relay}: already complete ({len(rows):,} rows)", flush=True)
        return len(rows)

    seen = {int(r["slot"]) for r in rows}
    t0, pages = time.time(), 0
    while cursor >= lo:
        page = fetch_page(relay, cursor)
        pages += 1
        if not page:
            cursor -= PAGE_OVERRIDE.get(relay, GAP_STEP)   # empty window: skip past it
        else:
            slots = [int(p["slot"]) for p in page]
            for p, s in zip(page, slots):
                if s not in seen and lo <= s <= hi:
                    seen.add(s)
                    rows.append(p)
            nxt = min(slots) - 1
            cursor = nxt if nxt < cursor else cursor - PAGE_OVERRIDE.get(relay, GAP_STEP)
        if pages % 50 == 0:
            pd.DataFrame(rows).to_parquet(out, index=False)
            ck.write_text(str(cursor))
            done = (hi - cursor) / max(hi - lo, 1)
            print(f"  {relay}: {done:5.1%}  {len(rows):>7,} rows  "
                  f"{pages:>5} pages  {time.time()-t0:6.0f}s", flush=True)

    pd.DataFrame(rows).to_parquet(out, index=False)
    ck.write_text(str(lo - 1))
    print(f"  {relay}: DONE {len(rows):,} rows in {time.time()-t0:.0f}s", flush=True)
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="lo", type=int, default=SLOT_LO)
    ap.add_argument("--to", dest="hi", type=int, default=SLOT_HI)
    ap.add_argument("--relays", default=",".join(RELAYS))
    a = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    relays = [r for r in a.relays.split(",") if r]
    print(f"slots {a.lo:,}..{a.hi:,} ({a.hi - a.lo:,} slots) across {len(relays)} relays",
          flush=True)
    with cf.ThreadPoolExecutor(max_workers=len(relays)) as ex:
        futs = {ex.submit(walk, r, a.lo, a.hi): r for r in relays}
        for f in cf.as_completed(futs):
            try:
                f.result()
            except Exception as e:  # noqa: BLE001
                print(f"  !! {futs[f]} failed: {e}", file=sys.stderr, flush=True)
    print("\nper-relay row counts:")
    for r in relays:
        p = OUT_DIR / f"{r.replace('.', '_')}.parquet"
        if p.exists():
            print(f"  {r:<34} {len(pd.read_parquet(p)):>8,}")


if __name__ == "__main__":
    main()
