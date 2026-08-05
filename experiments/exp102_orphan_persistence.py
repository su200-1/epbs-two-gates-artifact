"""Experiment 102 -- does the EMPTY selection persist past the boost window?

Why this exists
---------------
Sections 2 and 3 of the paper establish two things: the committee can force the
slot-(t+1) payload-status decision to EMPTY, and the builder's payment liability
survives that decision. They do NOT establish that the EMPTY branch stays
canonical afterwards -- and the victim's realised loss depends on exactly that.
The paper marks this as tier 3 of its evidence table and as the main open gap.
This experiment closes it, or shows it cannot be closed.

The mechanism under test is subtle and worth stating before measuring it. The
PTC's route into fork choice, `get_payload_status_tiebreaker`, is gated by
`is_previous_slot_payload_decision`: it only fires while the block is the
*immediately previous* slot's. Once slot t+2 starts, that gate is false and the
FULL/EMPTY contest is decided by ordinary accumulated attestation weight
(`get_weight` over nodes produced by `get_supported_node`). Proposer boost, which
Path B additionally relies on, also lapses at the end of the slot.

So persistence is not automatic; it depends on what honest attesters of slot t+1
voted. Per validator.md they set `attestation.data.index` from their own
fork-choice view -- which, during slot t+1, was the manipulated one. If they
therefore vote payload_present=False, their weight books onto the EMPTY node and
locks it in after the PTC's influence expires. That is the hypothesis.

We test it by advancing the ported fork choice past slot t+1 and asking
`get_head` which payload status survives, under three conditions:

  honest        no PTC corruption                        -> expect FULL persists
  attack        257 not-present votes at slot t+1         -> expect EMPTY persists
  attack_rescue attack, but slot-(t+1) attesters vote     -> control: shows the
                payload_present=True against the head        lock-in is what
                                                             carries persistence

The third arm is the falsification control. If EMPTY persisted even when honest
attesters vote FULL, the result would be an artifact of our setup rather than the
attestation-weight mechanism we claim.

Run: python experiments/exp102_orphan_persistence.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from epbs import attestation as A  # noqa: E402
from epbs import forkchoice as FC  # noqa: E402
from epbs import primitives as P  # noqa: E402
from epbs.primitives import PayloadStatus  # noqa: E402

OUT = Path(__file__).resolve().parent / "figures" / "drl_risk_epbs"

SLOT_MS = config.SECONDS_PER_SLOT * 1000 if hasattr(config, "SECONDS_PER_SLOT") else 12000
N_VAL = 64                    # attesting validators (fork-choice weight carriers)
BAL = 32 * 10**9              # effective balance, gwei
FOLLOW_SLOTS = 6              # how far past t+1 to advance


def _bid(parent_block_hash: int, block_hash: int, slot: int):
    return P.ExecutionPayloadBid(
        parent_block_hash=parent_block_hash, parent_block_root=0,
        block_hash=block_hash, builder_index=0, slot=slot,
        value=1000, execution_payment=0,
    )


def _block(root, parent_root, slot, *, parent_full, parent_block_hash=None):
    """A block whose bid commits to a FULL or EMPTY parent (see
    `get_parent_payload_status`): FULL iff bid.parent_block_hash equals the
    parent's own bid.block_hash."""
    ph = (parent_block_hash if parent_full else 999_999)
    blk = P.BeaconBlock(root=root, parent_root=parent_root, slot=slot,
                        bid=_bid(ph, block_hash=root * 1000, slot=slot))
    setattr(blk, "proposer_index", 0)
    return blk


def _register(store, blk, *, verified: bool):
    store.blocks[blk.root] = blk
    store.block_timeliness[blk.root] = [True, True]
    store.payload_timeliness_vote[blk.root] = [None] * config.PTC_SIZE
    store.payload_data_availability_vote[blk.root] = [None] * config.PTC_SIZE
    if verified:
        store.payloads[blk.root] = object()


def _attest(store, *, validators, root, slot, payload_present):
    """Deliver LMD votes for `root` from `slot` carrying a payload_present flag.

    Mirrors validator.md: `data.index = 1` iff the attester's fork-choice view
    has the payload FULL; `update_latest_messages` derives payload_present.
    """
    for vi in validators:
        store.latest_messages[vi] = A.LatestMessage(
            slot=slot, root=root, payload_present=payload_present)
    FC.rebuild_score_index(store)


def run(scenario: str) -> dict:
    """End-to-end: build slot t+1 the way the spec tells the proposer to, then
    advance past the boost window and see which payload status survives."""
    balances = {i: BAL for i in range(N_VAL)}
    anchor = _block(0xA0, 0, 0, parent_full=True, parent_block_hash=0)
    store = FC.get_forkchoice_store(
        anchor, effective_balances=balances,
        total_active_balance=BAL * N_VAL, genesis_time_ms=0)

    # --- slot t: victim builder's block; payload revealed on time ---
    t = 1
    victim = _block(0xB1, 0xA0, t, parent_full=True,
                    parent_block_hash=anchor.bid.block_hash)
    _register(store, victim, verified=True)

    # --- PTC votes, cast end of slot t, read during slot t+1 ---
    if scenario == "honest":
        votes = [True] * config.PTC_SIZE
    else:
        pivotal = config.PAYLOAD_TIMELY_THRESHOLD + 1     # 257
        votes = [False] * pivotal + [None] * (config.PTC_SIZE - pivotal)
    store.payload_timeliness_vote[victim.root] = votes
    store.payload_data_availability_vote[victim.root] = [True] * config.PTC_SIZE

    # --- slot t+1: the proposer applies should_build_on_full (validator.md) ---
    FC.on_tick(store, (t + 1) * SLOT_MS)
    head_full = FC.ForkChoiceNode(root=victim.root, payload_status=PayloadStatus.FULL)
    build_full = FC.should_build_on_full(store, head_full)
    child = _block(0xC2, victim.root, t + 1, parent_full=build_full,
                   parent_block_hash=victim.bid.block_hash)
    _register(store, child, verified=True)
    store.payload_timeliness_vote[child.root] = [True] * config.PTC_SIZE
    store.payload_data_availability_vote[child.root] = [True] * config.PTC_SIZE

    # What the slot-t block's payload status now is, as implied by the child.
    selected = FC.get_parent_payload_status(store, child)

    # --- honest attesters of slot t+1 attest to the child; their data.index
    # follows their own fork-choice view of the parent's payload status ---
    if scenario == "attack_rescue":
        present_flag = True                    # control: vote FULL against the head
    else:
        present_flag = (selected == PayloadStatus.FULL)
    _attest(store, validators=range(N_VAL), root=victim.root,
            slot=t + 1, payload_present=present_flag)

    # --- advance past the window: boost lapses, tiebreaker gate closes ---
    after = []
    for k in range(2, 2 + FOLLOW_SLOTS):
        FC.on_tick(store, (t + k) * SLOT_MS)
        store.proposer_boost_root = 0
        full_n = FC.ForkChoiceNode(root=victim.root, payload_status=PayloadStatus.FULL)
        empty_n = FC.ForkChoiceNode(root=victim.root, payload_status=PayloadStatus.EMPTY)
        w_full, w_empty = FC.get_weight(store, full_n), FC.get_weight(store, empty_n)
        gate = FC.is_previous_slot_payload_decision(store, empty_n)
        after.append({"slot": t + k, "weight_FULL": w_full, "weight_EMPTY": w_empty,
                      "survivor": ("FULL" if w_full > w_empty else
                                   "EMPTY" if w_empty > w_full else "TIE"),
                      "ptc_gate_open": gate})

    return {"scenario": scenario,
            "should_build_on_full": build_full,
            "selected_at_t_plus_1": selected.name,
            "attesters_voted_present": present_flag,
            "after_window": after,
            "persists": all(a["survivor"] == selected.name for a in after)}


def sweep_follow_fraction() -> list[dict]:
    """How many slot-(t+1) attesters must follow the manipulated head?

    The lock-in is carried by attestation weight, so it should degrade
    gracefully: below half following, the FULL node out-weighs EMPTY and the
    payload is rescued. This locates the crossover rather than assuming it.
    """
    rows = []
    for frac in (0.0, 0.25, 0.40, 0.50, 0.51, 0.60, 0.75, 1.0):
        balances = {i: BAL for i in range(N_VAL)}
        anchor = _block(0xA0, 0, 0, parent_full=True, parent_block_hash=0)
        store = FC.get_forkchoice_store(
            anchor, effective_balances=balances,
            total_active_balance=BAL * N_VAL, genesis_time_ms=0)
        t = 1
        victim = _block(0xB1, 0xA0, t, parent_full=True,
                        parent_block_hash=anchor.bid.block_hash)
        _register(store, victim, verified=True)
        pivotal = config.PAYLOAD_TIMELY_THRESHOLD + 1
        store.payload_timeliness_vote[victim.root] = (
            [False] * pivotal + [None] * (config.PTC_SIZE - pivotal))
        store.payload_data_availability_vote[victim.root] = [True] * config.PTC_SIZE

        FC.on_tick(store, (t + 1) * SLOT_MS)
        child = _block(0xC2, victim.root, t + 1, parent_full=False,
                       parent_block_hash=victim.bid.block_hash)
        _register(store, child, verified=True)
        store.payload_timeliness_vote[child.root] = [True] * config.PTC_SIZE
        store.payload_data_availability_vote[child.root] = [True] * config.PTC_SIZE

        n_follow = round(frac * N_VAL)
        _attest(store, validators=range(n_follow), root=victim.root,
                slot=t + 1, payload_present=False)          # follow -> EMPTY
        _attest(store, validators=range(n_follow, N_VAL), root=victim.root,
                slot=t + 1, payload_present=True)           # dissent -> FULL

        FC.on_tick(store, (t + 3) * SLOT_MS)
        store.proposer_boost_root = 0
        wf = FC.get_weight(store, FC.ForkChoiceNode(
            root=victim.root, payload_status=PayloadStatus.FULL))
        we = FC.get_weight(store, FC.ForkChoiceNode(
            root=victim.root, payload_status=PayloadStatus.EMPTY))
        rows.append({"follow_fraction": frac, "n_follow": n_follow,
                     "weight_FULL": wf, "weight_EMPTY": we,
                     "survivor": "EMPTY" if we > wf else "FULL" if wf > we else "TIE"})
    return rows


def main() -> None:
    results = [run(s) for s in ("honest", "attack", "attack_rescue")]

    print(f"{'scenario':16} {'build_full':11} {'t+1 status':11} {'attest present':15} "
          f"{'survivor after':15} {'persists'}")
    for r in results:
        surv = {a["survivor"] for a in r["after_window"]}
        print(f"{r['scenario']:16} {str(r['should_build_on_full']):11} "
              f"{r['selected_at_t_plus_1']:11} {str(r['attesters_voted_present']):15} "
              f"{'/'.join(sorted(surv)):15} {r['persists']}")
    gates = {a["ptc_gate_open"] for r in results for a in r["after_window"]}
    print(f"\nPTC tiebreaker gate open after the window: {gates}  (expected {{False}})")
    a0 = results[1]["after_window"][0]
    print(f"attack, first slot after window: "
          f"weight FULL={a0['weight_FULL']:,} vs EMPTY={a0['weight_EMPTY']:,}")

    sweep = sweep_follow_fraction()
    print("\nlock-in sensitivity: fraction of slot-(t+1) attesters following the head")
    print(f"{'follow':>8} {'n':>4} {'survivor':>9}")
    for r in sweep:
        print(f"{r['follow_fraction']:8.0%} {r['n_follow']:4d} {r['survivor']:>9}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "exp102_orphan_persistence.json").write_text(
        json.dumps({"schema": "orphan-persistence-v1",
                    "n_validators": N_VAL,
                    "follow_slots": FOLLOW_SLOTS,
                    "pivotal": config.PAYLOAD_TIMELY_THRESHOLD + 1,
                    "results": results,
                    "follow_fraction_sweep": sweep}, indent=2) + "\n")


if __name__ == "__main__":
    main()
