"""Experiment 87 -- empirical mainnet block-value calibration for the bribery
economics (upgrades exp82 from a log-normal fit to a real distribution).

exp82 evaluated the profitable-attack fraction against a log-normal fit to two
published moments. Here we replace that parametric model with a real sample of
delivered block values pulled from a public MEV-Boost relay's bidtraces API
(`value` = the payment to the proposer, i.e. block MEV + priority fees, the same
quantity a next-proposer would re-capture). We sample several time windows to
avoid a single-window bias, cache the raw sample for reproducibility, and report
the EMPIRICAL profitable fraction 1 - ECDF(bribe) at each bribe level.

Pull once (writes a cached snapshot), then re-run offline from the cache:
    python experiments/exp87_empirical_block_value.py --pull
    python experiments/exp87_empirical_block_value.py            # uses cache
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "figures" / "drl_risk_epbs"
CACHE = OUT_DIR / "exp87_block_value_sample.json"
GWEI_PER_ETH = 1e9
WEI_PER_GWEI = 1e9
RELAY = "https://boost-relay.flashbots.net/relay/v1/data/bidtraces/proposer_payload_delivered"


def _pull(n_per_window: int, windows_back_slots: list[int]) -> dict:
    """Page the relay bidtraces API across several windows; return values in gwei."""
    values_gwei = []
    slots = []
    # discover the latest slot
    latest = json.loads(_get(f"{RELAY}?limit=1"))
    cursor0 = int(latest[0]["slot"])
    for back in windows_back_slots:
        cursor = cursor0 - back
        got = 0
        while got < n_per_window:
            batch = json.loads(_get(f"{RELAY}?limit=200&cursor={cursor}"))
            if not batch:
                break
            for row in batch:
                values_gwei.append(int(row["value"]) / WEI_PER_GWEI)
                slots.append(int(row["slot"]))
            got += len(batch)
            cursor = min(int(r["slot"]) for r in batch) - 1
            time.sleep(0.4)
    return {"values_gwei": values_gwei, "slots": slots,
            "source": "flashbots boost-relay proposer_payload_delivered",
            "latest_slot": cursor0, "n": len(values_gwei)}


def _get(url: str) -> str:
    req = urllib.request.Request(url, headers={"accept": "application/json",
                                               "user-agent": "epbs-research/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode()


def _pct(sorted_vals, p):
    if not sorted_vals:
        return 0.0
    i = min(len(sorted_vals) - 1, int(p * len(sorted_vals)))
    return sorted_vals[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", action="store_true", help="fetch fresh sample from relay")
    ap.add_argument("--n-per-window", type=int, default=1500)
    ap.add_argument(
        "--per-seat-prices-eth", type=str,
        default="0,0.00001,0.0001,0.001,0.01,0.1",
        help="Comma-separated exogenous reservation prices per recruited PTC seat.",
    )
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.pull or not CACHE.exists():
        # recent, ~1 week back, ~2 weeks back (7200 slots/day)
        data = _pull(args.n_per_window, [0, 50_400, 100_800])
        CACHE.write_text(json.dumps(data))
        print(f"pulled {data['n']} blocks from {data['source']} -> {CACHE}")
    else:
        data = json.loads(CACHE.read_text())
        print(f"loaded {data['n']} cached blocks ({data['source']})")

    vals = sorted(v for v in data["values_gwei"] if v > 0)
    n = len(vals)
    median = _pct(vals, 0.5)
    mean = statistics.fmean(vals)
    print(f"\nEMPIRICAL block value (n={n}): "
          f"median={median/GWEI_PER_ETH:.4f} ETH  mean={mean/GWEI_PER_ETH:.4f} ETH")
    for p in (0.5, 0.8, 0.9, 0.99, 0.999):
        print(f"  p{int(p*100) if p<0.99 else p*100:>4}: {_pct(vals,p)/GWEI_PER_ETH:.4f} ETH")
    # share of total value in the top 20%
    top20 = sum(vals[int(0.8*n):]) / sum(vals) if sum(vals) else 0
    print(f"  top 20% of blocks hold {top20*100:.1f}% of total value")

    pivotal = config.PAYLOAD_TIMELY_THRESHOLD + 1  # strict > threshold false votes
    per_seat_prices_eth = [float(x) for x in args.per_seat_prices_eth.split(",")]
    if not per_seat_prices_eth or any(p < 0 for p in per_seat_prices_eth):
        ap.error("--per-seat-prices-eth must contain non-negative values")
    print(f"\npivotal={pivotal} (fixed); per-seat reservation price is exogenous")
    print("EMPIRICAL, per reservation-price level: share of BLOCKS attackable, and (more to the")
    print("point, since the attack is selective) share of total VALUE in those blocks:")
    total_value = sum(vals)
    rows = []
    for per_seat_eth in per_seat_prices_eth:
        total_cost_eth = round(pivotal * per_seat_eth, 12)
        total_cost_gwei = total_cost_eth * GWEI_PER_ETH
        hit = [v for v in vals if v > total_cost_gwei]
        frac_blocks = len(hit) / n
        frac_value = sum(hit) / total_value if total_value else 0.0
        rows.append({"per_seat_reservation_price_eth": per_seat_eth,
                     "total_reservation_cost_eth": total_cost_eth,
                     "break_even_eth": total_cost_eth,
                     "empirical_frac_blocks": frac_blocks,
                     "empirical_frac_value": frac_value})
        print(f"  p_seat={per_seat_eth:.5f} ETH  break-even={total_cost_eth:.5f} ETH  "
              f"-> {frac_blocks*100:5.1f}% of blocks, holding {frac_value*100:5.1f}% of all value")

    out = args.out_dir / "exp87_empirical_block_value.json"
    out.write_text(json.dumps({
        "schema": "exp87-v3", "n": n, "source": data["source"],
        "latest_slot": data.get("latest_slot"),
        "median_eth": median / GWEI_PER_ETH, "mean_eth": mean / GWEI_PER_ETH,
        "top20_value_share": top20,
        "pivotal_members": pivotal,
        "protocol_direct_penalty_eth": 0.0,
        "reservation_price_model": "exogenous_per_seat",
        "per_seat_prices_eth": per_seat_prices_eth,
        "rows": rows,
    }, indent=2))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
