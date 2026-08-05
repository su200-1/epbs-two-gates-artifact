"""Tests for `epbs.builder_payments` — Phase H1.

Covers:
* Constants match gloas spec (6/10 quorum numerator/denominator)
* `get_quorum_threshold` matches spec formula
* `enqueue_pending_payment` records correct payment with weight=0
* same-slot regular attestations bump weight once per validator
* `process_settlement_at_epoch_boundary` settles iff weight >= quorum
* End-to-end fixture: withholding alone settles; committee suppression expires

Run:  pytest difftest/test_builder_payments.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from epbs import builder_payments as bp
from epbs import forkchoice as F
from epbs import primitives as p
from epbs.adversary import (
    AttesterAction, HonestAdversary, PayloadLeverAdversary,
)
from epbs.env_tier2 import Tier2Config, Tier2Environment


class TestConstants:
    def test_quorum_numerator_matches_spec(self):
        # gloas/beacon-chain.md line 152
        assert bp.BUILDER_PAYMENT_THRESHOLD_NUMERATOR == 6

    def test_quorum_denominator_matches_spec(self):
        # gloas/beacon-chain.md line 153
        assert bp.BUILDER_PAYMENT_THRESHOLD_DENOMINATOR == 10

    def test_quorum_threshold_formula(self):
        # (total_active_balance // SLOTS_PER_EPOCH) * 6 // 10
        # spec gloas/beacon-chain.md:799-806
        total = 1024 * 32 * 10**9  # 1024 validators × 32 ETH
        per_slot = total // config.SLOTS_PER_EPOCH
        expected = per_slot * 6 // 10
        assert bp.get_quorum_threshold(total) == expected


class TestEnqueueAndAccumulate:
    def _store(self, total_active_balance=1_000_000_000_000):
        anchor = p.BeaconBlock(
            root=1, parent_root=0, slot=0,
            bid=p.ExecutionPayloadBid(0, 0, 100, 0, 0, 1000, 50),
        )
        return F.get_forkchoice_store(
            anchor, effective_balances={0: 32 * 10**9},
            total_active_balance=total_active_balance,
        )

    def test_enqueue_records_payment_at_slot(self):
        store = self._store()
        bp.enqueue_pending_payment(
            store, slot=5, builder_index=3, proposer_index=7, amount=50)
        assert 5 in store.builder_pending_payments
        payment = store.builder_pending_payments[5]
        assert payment.builder_index == 3
        assert payment.proposer_index == 7
        assert payment.amount == 50
        assert payment.weight == 0

    def test_enqueue_skips_zero_amount(self):
        store = self._store()
        bp.enqueue_pending_payment(
            store, slot=5, builder_index=3, proposer_index=7, amount=0)
        assert 5 not in store.builder_pending_payments

    def test_enqueue_skips_self_build_index(self):
        store = self._store()
        bp.enqueue_pending_payment(
            store, slot=5,
            builder_index=bp.BUILDER_INDEX_SELF_BUILD,
            proposer_index=7, amount=50,
        )
        assert 5 not in store.builder_pending_payments

    def test_accumulate_bumps_funded_payment(self):
        store = self._store()
        store.blocks[50] = p.BeaconBlock(
            root=50, parent_root=1, slot=5,
            bid=p.ExecutionPayloadBid(0, 1, 500, 3, 5, 50, 0),
        )
        bp.enqueue_pending_payment(
            store, slot=5, builder_index=3, proposer_index=7, amount=50)
        bp.accumulate_same_slot_attestation_weight(
            store, voted_block_slot=5, voted_block_root=50,
            attesting_validators=[0],
        )
        assert store.builder_pending_payments[5].weight == 32 * 10**9

    def test_accumulate_noop_when_no_payment(self):
        store = self._store()
        # No bid enqueued for slot 5
        bp.accumulate_same_slot_attestation_weight(
            store, voted_block_slot=5, voted_block_root=1,
            attesting_validators=[0],
        )
        assert 5 not in store.builder_pending_payments

    def test_accumulate_deduplicates_validator_and_rejects_historical_vote(self):
        store = self._store()
        store.blocks[50] = p.BeaconBlock(
            root=50, parent_root=1, slot=5,
            bid=p.ExecutionPayloadBid(0, 1, 500, 3, 5, 50, 0),
        )
        bp.enqueue_pending_payment(
            store, slot=5, builder_index=3, proposer_index=7, amount=50)
        for root in (50, 50, 1):
            bp.accumulate_same_slot_attestation_weight(
                store, voted_block_slot=5, voted_block_root=root,
                attesting_validators=[0],
            )
        assert store.builder_pending_payments[5].weight == 32 * 10**9


class TestSettlement:
    def _store_with_payment_in_epoch(
        self, slot: int, weight: int, amount: int = 50,
        total_active_balance=1024 * 32 * 10**9,
    ):
        anchor = p.BeaconBlock(
            root=1, parent_root=0, slot=0,
            bid=p.ExecutionPayloadBid(0, 0, 100, 0, 0, 1000, 50),
        )
        store = F.get_forkchoice_store(
            anchor, effective_balances={},
            total_active_balance=total_active_balance,
        )
        bp.enqueue_pending_payment(
            store, slot=slot, builder_index=99, proposer_index=0, amount=amount)
        store.builder_pending_payments[slot].weight = weight
        return store

    def test_settles_when_weight_meets_quorum(self):
        # quorum at 1024 × 32 ETH active balance:
        # per_slot = 1024*32e9 / 32 = 32e9 × 32 = 1.024e12; quorum = 0.6 × 1.024e12
        store = self._store_with_payment_in_epoch(
            slot=5,
            weight=bp.get_quorum_threshold(1024 * 32 * 10**9),  # exactly quorum
        )
        # `_store_with_payment_in_epoch` enqueues with builder=99 proposer=0.
        # Slot 5 is in epoch 0; settle by calling current_epoch=1.
        n = bp.process_settlement_at_epoch_boundary(store, current_epoch=1)
        assert n == 1
        # Per gloas/builder.md:135 the payment flows builder → proposer:
        assert store.proposer_credits[0] == 50, "proposer receives bid.value"
        assert store.builder_debits[99] == 50, "builder pays bid.value"
        assert 5 not in store.builder_pending_payments

    def test_no_settle_below_quorum(self):
        store = self._store_with_payment_in_epoch(
            slot=5,
            weight=bp.get_quorum_threshold(1024 * 32 * 10**9) - 1,
        )
        n = bp.process_settlement_at_epoch_boundary(store, current_epoch=1)
        assert n == 0
        # Neither side moves money on unsettled payment.
        assert 0 not in store.proposer_credits
        assert 99 not in store.builder_debits
        # Payment is consumed (dropped) — matches spec vector shift
        assert 5 not in store.builder_pending_payments

    def test_only_previous_epoch_settled(self):
        # Put a payment in epoch 1 (slot 35) with full weight; settle epoch 0
        store = self._store_with_payment_in_epoch(
            slot=35,
            weight=bp.get_quorum_threshold(1024 * 32 * 10**9),
        )
        n = bp.process_settlement_at_epoch_boundary(store, current_epoch=1)
        assert n == 0
        assert 35 in store.builder_pending_payments  # untouched

    def test_skips_before_epoch_1(self):
        store = self._store_with_payment_in_epoch(
            slot=5, weight=bp.get_quorum_threshold(1024 * 32 * 10**9))
        n = bp.process_settlement_at_epoch_boundary(store, current_epoch=0)
        assert n == 0
        # Payment still pending — settlement skipped
        assert 5 in store.builder_pending_payments

    def test_full_parent_settles_immediately_without_quorum(self):
        store = self._store_with_payment_in_epoch(slot=5, weight=0)
        before = bp.cover_bid_headroom(store, builder_index=99)
        assert bp.settle_parent_payment_immediately(
            store, slot=5, builder_index=99, proposer_index=0, amount=50,
        )
        assert 5 not in store.builder_pending_payments
        assert store.proposer_credits[0] == 50
        assert store.builder_debits[99] == 50
        assert bp.cover_bid_headroom(store, builder_index=99) == before


class TestEndToEndInEnv:
    """Integration: run an episode and check settlement matches behaviour."""

    def test_honest_baseline_settles_most_bids(self):
        cfg = Tier2Config(
            num_validators=512, num_builders=16,
            num_slots=config.SLOTS_PER_EPOCH * 3 + 1,
            committee_size=32, enable_ffg=True,
            strict_spec_validation=True,
        )
        env = Tier2Environment(cfg, HonestAdversary())
        env.reset()
        env.run_episode()
        # All-honest: every settled bid moves bid_value_gwei (1e6 default)
        # from builder to proposer. Conservation: total credits = total debits.
        total_credits = sum(env.store.proposer_credits.values())
        total_debits = sum(env.store.builder_debits.values())
        assert total_credits == total_debits > 0, (
            f"honest baseline should have credits == debits > 0; got "
            f"credits={total_credits}, debits={total_debits}"
        )

    def test_all_byz_withhold_still_settles_from_regular_attestations(self):
        byz_b = set(range(16))
        cfg = Tier2Config(
            num_validators=512, num_builders=16,
            num_slots=config.SLOTS_PER_EPOCH * 3 + 1,
            committee_size=32, enable_ffg=True,
            byzantine_builders=byz_b,
        )
        env = Tier2Environment(
            cfg,
            PayloadLeverAdversary(
                byzantine_builders=byz_b, byzantine_validators=set()),
        )
        env.reset()
        env.run_episode()
        assert sum(env.store.proposer_credits.values()) > 0
        assert sum(env.store.proposer_credits.values()) == sum(
            env.store.builder_debits.values(),
        )

    def test_withhold_plus_committee_suppression_settles_nothing(self):
        class SuppressCommittee(PayloadLeverAdversary):
            def attester_action(self, validator_index, slot, view_root, ctx):
                return AttesterAction.WITHHOLD_VOTE

        byz_b = set(range(16))
        env = Tier2Environment(
            Tier2Config(
                num_validators=512, num_builders=16,
                num_slots=config.SLOTS_PER_EPOCH * 3 + 1,
                committee_size=32, enable_ffg=True,
                byzantine_builders=byz_b,
            ),
            SuppressCommittee(
                byzantine_builders=byz_b, byzantine_validators=set(),
            ),
        )
        env.run_episode()
        assert sum(env.store.proposer_credits.values()) == 0
        assert sum(env.store.builder_debits.values()) == 0
        assert env.store.expired_payment_slots


# ===========================================================================
# Executable differential tests against the gloas pyspec
# ===========================================================================
_DEFAULT_PYSPEC_DIR = os.environ.get("EPBS_PYSPEC_DIR", "../consensus-specs/tests/core/pyspec")
_SPEC_MODULE_CANDIDATES = [
    "eth_consensus_specs.gloas.mainnet",
    "eth_consensus_specs.gloas.minimal",
]


def _load_spec():
    pyspec_dir = os.environ.get("EPBS_PYSPEC_DIR", _DEFAULT_PYSPEC_DIR)
    if os.path.isdir(pyspec_dir) and pyspec_dir not in sys.path:
        sys.path.insert(0, pyspec_dir)
    for name in _SPEC_MODULE_CANDIDATES:
        try:
            return __import__(name, fromlist=["*"])
        except ImportError:
            continue
    return None


@pytest.fixture(scope="module")
def spec():
    s = _load_spec()
    if s is None:
        pytest.skip("gloas pyspec not importable — build with "
                    "`python -m pysetup.generate_specs --fork gloas` or set "
                    "EPBS_PYSPEC_DIR")
    return s


class TestSettlementAgainstPyspec:
    """Executable differential tests for the builder-payment settlement layer.

    Upgrades the constant-transcription checks above to a true diff against the
    executable gloas pyspec, pinning that (a) the quorum threshold matches and
    (b) the settle/expire decision is purely ``weight >= quorum``.

    Point (b) is payload-independent in both sim and spec — in the spec,
    ``process_attestation`` (beacon-chain.md) accrues builder-payment weight
    from same-slot timely attestations that set source/target/head flags, with
    no payload-presence condition. That is why a withheld bid which reaches
    quorum still settles (cf. TestEndToEndInEnv); WITHHOLD *alone* is therefore
    not a settlement free option, and only genuine quorum failure (e.g.
    committee vote suppression) expires a payment.
    """

    @staticmethod
    def _mk_pyspec_state(spec, n_val: int, eff: int = 32 * 10**9):
        st = spec.BeaconState()
        far = spec.FAR_FUTURE_EPOCH
        for _ in range(n_val):
            st.validators.append(spec.Validator(
                effective_balance=spec.Gwei(eff),
                activation_eligibility_epoch=spec.Epoch(0),
                activation_epoch=spec.Epoch(0),
                exit_epoch=far, withdrawable_epoch=far,
            ))
            st.balances.append(spec.Gwei(eff))
        return st

    @pytest.mark.parametrize("n_val", [64, 128, 256])
    def test_quorum_threshold_matches_pyspec(self, spec, n_val):
        eff = 32 * 10**9
        st = self._mk_pyspec_state(spec, n_val, eff)
        py_q = int(spec.get_builder_payment_quorum_threshold(st))
        assert bp.get_quorum_threshold(n_val * eff) == py_q

    def test_settle_expire_decision_matches_pyspec(self, spec):
        eff = 32 * 10**9
        n_val = 64
        st = self._mk_pyspec_state(spec, n_val, eff)
        q = int(spec.get_builder_payment_quorum_threshold(st))
        # Weights relative to quorum, placed at previous-epoch slots
        # [0, SLOTS_PER_EPOCH) so process_builder_pending_payments processes them.
        weights = {0: q, 1: q - 1, 2: q + 1, 3: 0}
        settle_expected = {slot for slot, w in weights.items() if w >= q}
        expire_expected = {slot for slot, w in weights.items() if w < q}

        # --- pyspec: number of withdrawals produced == settled count ---
        for slot, w in weights.items():
            st.builder_pending_payments[slot] = spec.BuilderPendingPayment(
                weight=spec.Gwei(w),
                withdrawal=spec.BuilderPendingWithdrawal(
                    amount=spec.Gwei(1_000_000),
                    builder_index=spec.BuilderIndex(slot)),
            )
        n_before = len(st.builder_pending_withdrawals)
        spec.process_builder_pending_payments(st)
        assert len(st.builder_pending_withdrawals) - n_before == len(settle_expected)

        # --- sim: same settle/expire partition ---
        store = F.Store(total_active_balance=n_val * eff)
        for slot, w in weights.items():
            store.builder_pending_payments[slot] = bp.PendingPayment(
                slot=slot, builder_index=slot, proposer_index=0,
                amount=1_000_000, weight=w)
        bp.process_settlement_at_epoch_boundary(store, current_epoch=1)
        assert set(store.settled_payment_slots) == settle_expected
        assert set(store.expired_payment_slots) == expire_expected
        # the threshold the sim used matches the spec's
        assert bp.get_quorum_threshold(n_val * eff) == q
