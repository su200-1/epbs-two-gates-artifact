"""Experiment 92 -- exact PTC-timeliness x builder-payment quorum boundary.

Deterministic executable-predicate probe at the real committee size
(PTC_SIZE = 512): it crosses not-present PTC seats (255/256/257/258) with the
independent regular-attestation payment weight (59/60/61 %) and records the
resulting parent payload status and payment outcome in each cell. Produces the
exact two-gate boundary table of the paper.

Run: python experiments/exp92_exact_ptc_payment_boundary.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from epbs import builder_payments as bp  # noqa: E402
from epbs import forkchoice as F  # noqa: E402
from epbs import primitives as p  # noqa: E402
from epbs.primitives import PayloadStatus  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "figures" / "drl_risk_epbs"
EFFECTIVE_BALANCE_GWEI = 32 * 10**9
N_VALIDATORS = 1024
PAYMENT_AMOUNT_GWEI = 100_000_000


def make_store(not_present: int) -> F.Store:
    root = 2
    store = F.Store(
        time_ms=2 * config.MILLIS_PER_SLOT,
        total_active_balance=N_VALIDATORS * EFFECTIVE_BALANCE_GWEI,
    )
    store.blocks[root] = p.BeaconBlock(
        root=root,
        parent_root=1,
        slot=1,
        bid=p.ExecutionPayloadBid(0, 1, 200, 99, 1, 1_000, PAYMENT_AMOUNT_GWEI),
    )
    store.payloads[root] = p.ExecutionPayloadEnvelope(99, root, 200)
    store.payload_timeliness_vote[root] = (
        [False] * not_present + [True] * (config.PTC_SIZE - not_present)
    )
    # 固定 data availability 为正，仅隔离 payload timeliness。
    store.payload_data_availability_vote[root] = [True] * config.PTC_SIZE
    return store


def run_cell(not_present: int, payment_pct: int) -> dict:
    store = make_store(not_present)
    root = 2
    not_timely = p.payload_timeliness(store, root, timely=False)
    build_on_full = F.should_build_on_full(
        store, F.ForkChoiceNode(root, PayloadStatus.FULL)
    )

    bp.enqueue_pending_payment(
        store, slot=1, builder_index=99, proposer_index=7,
        amount=PAYMENT_AMOUNT_GWEI,
    )
    quorum = bp.get_quorum_threshold(store.total_active_balance)
    per_slot_balance = store.total_active_balance // config.SLOTS_PER_EPOCH
    payment_weight = per_slot_balance * payment_pct // 100
    store.builder_pending_payments[1].weight = payment_weight
    settled = bp.process_settlement_at_epoch_boundary(store, current_epoch=1) == 1

    expected_full = not_present <= config.PAYLOAD_TIMELY_THRESHOLD
    expected_settled = payment_pct >= 60
    assert build_on_full == expected_full
    assert not_timely == (not expected_full)
    assert settled == expected_settled
    assert store.builder_debits.get(99, 0) == (
        PAYMENT_AMOUNT_GWEI if expected_settled else 0
    )

    return {
        "not_present_ptc_votes": not_present,
        "payment_weight_pct": payment_pct,
        "payload_not_timely": not_timely,
        "parent_status": "FULL" if build_on_full else "EMPTY",
        "payment_quorum_gwei": quorum,
        "payment_weight_gwei": payment_weight,
        "payment_settled": settled,
        "builder_debit_gwei": store.builder_debits.get(99, 0),
        "proposer_credit_gwei": store.proposer_credits.get(7, 0),
    }


def main() -> None:
    rows = [
        run_cell(n, pct)
        for n in (255, 256, 257, 258)
        for pct in (59, 60, 61)
    ]
    result = {
        "schema": "exact-ptc-payment-boundary-v1",
        "artifact_commit": "7939874",
        "ptc_size": config.PTC_SIZE,
        "payload_timely_threshold": config.PAYLOAD_TIMELY_THRESHOLD,
        "pivotal_not_present_votes": config.PAYLOAD_TIMELY_THRESHOLD + 1,
        "payment_threshold": "60% of one slot's expected active balance",
        "payment_amount_gwei": PAYMENT_AMOUNT_GWEI,
        "rows": rows,
        "evidentiary_boundary": (
            "Deterministic protocol-predicate experiment; does not estimate "
            "PTC recruitment probability or transaction-level MEV recapture."
        ),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / "exp92_exact_ptc_payment_boundary.json"
    md_path = OUT_DIR / "exp92_exact_ptc_payment_boundary.md"
    json_path.write_text(json.dumps(result, indent=2) + "\n")

    header = (
        "| Not-present PTC votes | Payment weight | Parent status | Payment | "
        "Builder debit (gwei) |\n|---:|---:|:---:|:---:|---:|\n"
    )
    body = "".join(
        f"| {r['not_present_ptc_votes']} | {r['payment_weight_pct']}% | "
        f"{r['parent_status']} | "
        f"{'settled' if r['payment_settled'] else 'expired'} | "
        f"{r['builder_debit_gwei']} |\n"
        for r in rows
    )
    md_path.write_text(
        "# Exact PTC × payment-quorum boundary\n\n" + header + body
        + "\n结论：payload continuation 在 257 个 not-present votes 处由 FULL "
        "跳变为 EMPTY；builder payment 在 60% regular-attestation weight 处结算。"
        "两条谓词相互独立。\n"
    )
    print(header + body)
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
