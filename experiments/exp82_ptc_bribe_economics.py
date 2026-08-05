"""Experiment 82 -- PTC-orphan reservation-price sensitivity.

exp81 validated the mechanic: controlling >50% of the fixed-size PTC (PTC_SIZE=512,
PAYLOAD_TIMELY_THRESHOLD=256) orphans an honest builder's timely payload while the
builder still pays (payment settles via the separate regular-committee quorum).
The next proposer then re-captures the now-public MEV.

The economic question is conditional: at what externally supplied per-seat price is
the attack cheaper than the MEV that can be re-captured?

Reservation-price model. The fork-choice predicate payload_timeliness(., timely=False) fires on
sum(not-present) > PAYLOAD_TIMELY_THRESHOLD, so orphaning via should_build_on_full
strictly needs PAYLOAD_TIMELY_THRESHOLD + 1 = 257 not-present votes -- one more than
the bare half PTC_SIZE - PAYLOAD_TIMELY_THRESHOLD = 256. We use the exact 257-member
pivotal count in all break-even calculations. Manipulating the payload-timeliness
signal is a SPEC-VALID, NON-SLASHABLE deviation: gloas defines slashing only for
equivocating proposers and attesters, and process_payload_attestation applies neither
slashing nor an in-function balance deduction to a PTC member (verified against
consensus-specs gloas/beacon-chain.md). The audited specification also does not define
a separate reward that is forfeited for casting a validly signed ``not present`` vote.
The protocol-implied direct monetary loss, and hence its cost lower bound, is therefore
zero. This does not imply that operators participate for free: coordination, reputation,
legal exposure, counter-bribes, and pivotal-seat holdout power can create a positive but
protocol-external reservation price. Because that price is not identified by the
specification, we sweep an exogenous per-seat price directly in ETH.

    pivotal      = PAYLOAD_TIMELY_THRESHOLD + 1                  (= 257, FIXED)
    p_seat       = exogenous per-seat reservation price          (ETH)
    total_cost   = pivotal * p_seat
    breakeven    = total_cost      (attack profitable iff captured MEV > total_cost)

Crucially `pivotal` is a FIXED 257 regardless of total stake -- unlike Paper 2's
settlement-evasion / reorg bounds which need ~1/3-40% of ALL stake. That is the
scale-robustness of this channel.

We then compare breakeven to a mainnet block-value (MEV + priority fee)
distribution and report the fraction of blocks a rational next-proposer would
find profitable to attack. The distribution is illustrative (lognormal anchored
to public MEV-Boost/RelayScan order-of-magnitude figures Paper 2 already cites)
and flagged for final data calibration; the headline is the ORDER OF MAGNITUDE
relationship between an exogenous 257-seat reservation cost and per-block MEV.

Caveats (honest): (i) the zero-price row is a protocol-implied lower bound, not a claim
that operators can actually be recruited for free; (ii) captured MEV <= block value
(re-capture fraction not yet modeled in-sim); (iii) assumes the attacker can identify
and recruit the RANDAO-sampled PTC members ahead of the slot; (iv) breakeven ignores
the attacker's own next-slot proposer reward, which only helps the attacker.

Run: python experiments/exp82_ptc_bribe_economics.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "figures" / "drl_risk_epbs"
GWEI_PER_ETH = 1e9


def _lognormal_cdf(x_gwei: float, median_eth: float, sigma: float) -> float:
    """P(block value <= x) for a lognormal with given median (ETH) and shape sigma."""
    if x_gwei <= 0:
        return 0.0
    mu = math.log(median_eth * GWEI_PER_ETH)
    return 0.5 * (1.0 + math.erf((math.log(x_gwei) - mu) / (sigma * math.sqrt(2.0))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--per-seat-prices-eth",
        type=str,
        default="0,0.00001,0.0001,0.001,0.01,0.1",
        help=(
            "Comma-separated exogenous reservation prices per recruited PTC seat, "
            "in ETH. Zero is the protocol-implied direct-cost lower bound."
        ),
    )
    # mainnet block-value distribution (ETH), log-normal calibrated to public relay
    # data: median MEV-Boost block value ~0.05 ETH, mean ~0.112 ETH -> sigma =
    # sqrt(2*ln(0.112/0.05)) ~ 1.27 (mevboost.pics / RelayScan / Rated). The
    # empirical tail (p99.99 ~ 29 ETH) is heavier than log-normal, so profitable
    # fractions are conservative.
    ap.add_argument("--mev-median-eth", type=float, default=0.05)
    ap.add_argument("--mev-sigma", type=float, default=1.27)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    ptc_size = config.PTC_SIZE
    threshold = config.PAYLOAD_TIMELY_THRESHOLD
    pivotal = threshold + 1  # strict > threshold false votes needed for not-timely
    per_seat_prices_eth = [float(x) for x in args.per_seat_prices_eth.split(",")]
    if not per_seat_prices_eth or any(price < 0 for price in per_seat_prices_eth):
        ap.error("--per-seat-prices-eth must contain one or more non-negative values")

    print(f"PTC_SIZE={ptc_size}  PAYLOAD_TIMELY_THRESHOLD={threshold}  "
          f"pivotal(recruited-seat count, FIXED)={pivotal}")
    print("protocol-level direct penalty/deduction for a valid not-present vote = 0 ETH")
    print("per-seat recruitment price is an exogenous reservation-price assumption")
    print(f"illustrative block-value dist: lognormal median={args.mev_median_eth} ETH, "
          f"sigma={args.mev_sigma}\n")

    rows = []
    for per_seat_eth in per_seat_prices_eth:
        per_seat_gwei = round(per_seat_eth * GWEI_PER_ETH, 6)
        breakeven_eth = round(pivotal * per_seat_eth, 12)
        total_cost_gwei = round(breakeven_eth * GWEI_PER_ETH, 6)
        frac_profitable = 1.0 - _lognormal_cdf(
            total_cost_gwei, args.mev_median_eth, args.mev_sigma
        )
        rows.append({
            "per_seat_reservation_price_eth": per_seat_eth,
            "per_seat_reservation_price_gwei": per_seat_gwei,
            "pivotal_members": pivotal,
            "total_reservation_cost_eth": breakeven_eth,
            "total_reservation_cost_gwei": total_cost_gwei,
            "breakeven_mev_eth": breakeven_eth,
            "frac_blocks_profitable": frac_profitable,
        })
        print(f"p_seat={per_seat_eth:.5f} ETH  "
              f"total reservation cost(=breakeven MEV)={breakeven_eth:.5f} ETH  "
              f"-> blocks profitably attackable = {frac_profitable*100:5.1f}%")

    # scale-robustness contrast vs Paper 2's stake-share bound
    print("\nSCALE CONTRAST:")
    print(f"  This channel: recruit {pivotal} PTC members (FIXED) -- independent of total stake.")
    print(f"  Paper 2 settlement-evasion / reorg: need ~1/3-40% of ALL stake (scales with network).")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "exp82_ptc_bribe_economics.json"
    out.write_text(json.dumps({
        "schema": "exp82-v3",
        "ptc_size": ptc_size, "payload_timely_threshold": threshold,
        "pivotal_members": pivotal,
        "protocol_direct_penalty_eth": 0.0,
        "reservation_price_model": "exogenous_per_seat",
        "per_seat_prices_eth": per_seat_prices_eth,
        "mev_median_eth": args.mev_median_eth, "mev_sigma": args.mev_sigma,
        "rows": rows,
        "caveats": [
            "zero per-seat price is a protocol-implied direct-cost lower bound, not an observed market price",
            "the specification does not determine operators' reservation prices",
            "reservation prices may reflect coordination, reputation, legal exposure, counter-bribes, and pivotal-seat holdout power",
            "captured MEV <= block value (re-capture fraction not yet modeled in-sim)",
            "assumes attacker can identify+recruit RANDAO-sampled PTC members ahead of slot",
            "block-value distribution is illustrative; flag for RelayScan/mevboost.pics calibration",
            "a validly signed not-present PTC vote triggers no direct penalty, balance deduction, or slashing under the audited specification",
        ],
    }, indent=2))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
