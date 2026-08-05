"""Tier 2a multi-slot ePBS environment.

Wires committee + forkchoice + network + adversary into a discrete-event
simulation of a length-``num_slots`` chain.

The environment is **not** a `gym.Env` — Tier 2a is not RL-trained (per
plan). It exposes a `run_episode()` API for honest baseline + hardcoded
adversary experiments (exp6, exp7). RL integration is reserved for a
later plan (see plan file).

Slot loop (per slot, time-domain):
    1. on_tick to the slot boundary; proposer-boost root resets
    2. Adversary's `proposer_action` picks parent — block scheduled at +0 ms
    3. on_block fires; honest builder reveals payload (or adversary withholds)
    4. Builder's `builder_action` schedules `on_execution_payload_envelope`
    5. Beacon committee attestations at `get_attestation_due_ms`,
       advancing 1 slot before they take effect
    6. PTC votes at `get_payload_attestation_due_ms` (honest: vote per local view)
    7. End of slot — collect metrics
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from hashlib import sha256

import config
from epbs import attestation as A
from epbs import builder as B
from epbs import builder_payments as bp
from epbs import committee as C
from epbs import ffg
from epbs import forkchoice as F
from epbs import primitives as p
from epbs import rewards as R
from epbs import stats as S
from epbs.adversary import (
    AdvCtx, Adversary, AttesterAction, BuilderAction, HonestAdversary,
)
from epbs.network import Message, MessageBus, MessageKind
from epbs.primitives import PayloadStatus


@dataclass
class Tier2Config:
    """Tier 2a/b config (orthogonal to ``config.SimConfig``)."""

    num_validators: int = 64
    num_builders: int = 16           # independent builder pool
    num_slots: int = 12
    byzantine_validators: set[int] = field(default_factory=set)
    # Now indexes the builder pool (was fused with validator indices before
    # the Phase D fix). Use ``set(range(int(num_builders * frac)))`` to express
    # a Byzantine builder share.
    byzantine_builders: set[int] = field(default_factory=set)
    committee_size: int = 8          # honest committee size per slot
    seed: bytes = b"\x42" * 32
    enable_ffg: bool = False         # Tier 2b — run FFG at epoch boundaries
    enforce_ffg_target_consistency: bool = False  # Tier 2b — validate_on_attestation FFG check
    # When True, re-raise spec AssertionError instead of counting + dropping.
    # Use for honest-baseline sanity checks and diff tests.
    strict_spec_validation: bool = False
    # Phase H3 — RL coalition hooks. Either returns None to defer to the
    # default behaviour (auction max-bid; adversary.builder_action), or
    # overrides it. Set per-slot by the coalition_env wrapper based on the
    # PPO agent's action choice. Plain functions, not Adversary methods,
    # because they need ad-hoc per-slot wiring from the RL loop.
    #   preferred_builder_hook(proposer_index, slot) -> int | None
    #   forced_builder_action_hook(builder_index, slot) -> str | None
    preferred_builder_hook: object = None
    forced_builder_action_hook: object = None
    # Phase L-α — attack-discovery hooks.
    #   proposer_parent_hook(proposer_index, slot) -> (parent_root, PayloadStatus) | None
    #     Returns an alternative (parent_root, parent_status) the byz proposer
    #     wants to build on (BUILD_ON_NONHEAD primitive). None → adversary
    #     default (head-build for honest, or as configured).
    #   bid_value_override_hook(builder_index, slot) -> int | None
    #     Returns the **intended** bid amount in gwei for this builder this
    #     slot (e.g., 5× cfg.bid_value_gwei for INFLATE_LOW). None → use
    #     cfg.bid_value_gwei. Env passes intended → compute_effective_bid_value
    #     to get the accepted amount (zero when the cover-bid check rejects); the
    #     effective amount is what drives BOTH auction ranking and settlement.
    #   late_reveal_hook(builder_index, slot) -> bool
    #     When True, env runs a fixed late-arrival sequence: regular committee
    #     votes at 25%, PTC votes absent at 75%, then the payload envelope is
    #     delivered. Settlement remains regular-attestation driven.
    #   ptc_vote_hook(ptc_validator_index, slot, local_view_present) -> bool | None
    #     Override an individual byz PTC member's vote. None → honest vote
    #     (local_view_present). Used for FRAUD_ABSENT (return False on a
    #     revealed payload) and the dual fraud-present (return True on a
    #     withheld payload) primitives.
    proposer_parent_hook: object = None
    bid_value_override_hook: object = None
    late_reveal_hook: object = None
    ptc_vote_hook: object = None
    # Phase H — per-bid value (gwei). PER SPEC (gloas/builder.md:135), this is
    # the amount the **builder pays the proposer** if the bid is accepted —
    # NOT the proposer paying the builder. Default 1e6 gwei = 0.001 ETH
    # (modest realistic MEV-Boost block reward). Settles to
    # `store.proposer_credits[proposer]` and debits `store.builder_debits[builder]`
    # after FULL-parent processing or at an epoch boundary when same-slot
    # regular-attestation weight reaches the 60% quorum.
    # bid.execution_payment is hardcoded to 0 per spec (gloas/builder.md:138 —
    # "Set `bid.execution_payment` to zero. A non-zero value indicates a trusted
    # execution-layer payment").
    bid_value_gwei: int = 1_000_000
    # Phase M (MEV market, step 1) — opt-in heterogeneous, time-varying builder
    # value. mev_enabled=False (default) keeps the flat bid_value_gwei for every
    # (builder, slot), so all pre-M results reproduce bit-for-bit. When enabled,
    # the intended bid of builder b at slot t is the DETERMINISTIC function
    #     intended(b,t) = mev_base_gwei * slot_factor(t) * builder_skill(b)
    # with slot_factor a reproducible per-slot lognormal keyed by (mev_seed, t)
    # — deterministic so paired evaluation (attack vs honest at the same seed)
    # sees the SAME MEV opportunity each slot, keeping deltas meaningful.
    # builder_skill indexes mev_builder_skill (heterogeneity); out-of-range or
    # empty -> skill 1.0. mev_slot_sigma=0 -> slot_factor=1 (time-invariant but
    # still builder-heterogeneous). Keep mev_base*factor*skill below builder
    # cover-headroom or compute_effective_bid_value rejects the bid in full.
    mev_enabled: bool = False
    mev_base_gwei: int = 1_000_000
    mev_slot_sigma: float = 0.0          # lognormal sigma of COMMON per-slot MEV
    mev_builder_sigma: float = 0.0       # lognormal sigma of per-(builder,slot)
    mev_builder_skill: tuple = ()        # per-builder multiplier; () => all 1.0
    mev_seed: int = 0


@dataclass
class SlotEvent:
    """Per-slot trace entry. Used by metrics + experiment scripts."""

    slot: int
    proposer_index: int
    block_root: int
    parent_root: int
    parent_status: PayloadStatus
    payload_revealed: bool
    head_root_at_end: int
    head_status_at_end: PayloadStatus
    builder_action: str
    proposer_action: str
    reorged_a_block: bool = False  # set in post-pass
    # Paper 2 observability: auction commitment seen by the winner.
    winner_builder_index: int = -1
    intended_bid_value_gwei: int = 0
    effective_bid_value_gwei: int = 0
    bid_was_clipped: bool = False


@dataclass
class EpochEvent:
    """Per-epoch FFG outcome (Tier 2b)."""

    epoch: int
    justified_epoch_before: int
    justified_epoch_after: int
    finalized_epoch_before: int
    finalized_epoch_after: int
    target_balance: int
    total_active_balance: int
    justified_this_epoch: bool


@dataclass
class EpisodeResult:
    """Result of one episode (chain). Consumed by exp6/exp7/exp8/exp9."""

    slot_events: list[SlotEvent]
    epoch_events: list[EpochEvent] = field(default_factory=list)
    final_head_root: int = 0
    final_head_status: PayloadStatus = PayloadStatus.PENDING
    num_reorgs: int = 0              # blocks that fell out of the canonical chain
    empty_blocks: int = 0            # head_status==EMPTY ever
    num_full: int = 0
    # Tier 2b summary
    final_justified_epoch: int = 0
    final_finalized_epoch: int = 0
    num_epochs_failed_to_justify: int = 0  # staircase attack signal
    # Paper 2 payment telemetry. Pending entries at episode end are censored.
    settled_payments: int = 0
    expired_payments: int = 0
    settled_payment_amount_gwei: int = 0
    expired_payment_amount_gwei: int = 0
    pending_payments_at_end: int = 0
    full_but_unsettled_blocks: int = 0
    # Per-kind counter of spec-handler AssertionErrors caught during the run.
    # Non-empty under a non-strict config flags a modelling bug (something is
    # generating messages the spec validator rejects) rather than an attack
    # signal — investigate before reading other metrics.
    dropped_messages: dict[str, int] = field(default_factory=dict)
    first_drop_error: dict[str, str] = field(default_factory=dict)
    # Phase G — probabilistic adversary state per slot.
    # Aggregate via `S.summarise_episode(result.slot_stats)`.
    slot_stats: list[S.SlotStat] = field(default_factory=list)


class Tier2Environment:
    """Multi-slot ePBS chain simulation."""

    def __init__(self, cfg: Tier2Config, adversary: Adversary | None = None):
        self.cfg = cfg
        self.adversary = adversary or HonestAdversary()
        # Validator set: only `byzantine_validators` flags validators as
        # Byzantine. Builders live in their own pool below — fusing them
        # would re-create the bug Phase D fixes.
        self.vset = C.make_validator_set(
            cfg.num_validators,
            byzantine_indices=cfg.byzantine_validators,
        )
        # Builder pool, independent of validator indices.
        self.bset = B.make_builder_set(
            cfg.num_builders, byzantine_indices=cfg.byzantine_builders,
        )
        self._anchor_eff_balances = {
            v.index: v.effective_balance for v in self.vset.validators
        }
        self._total_active_balance = sum(self._anchor_eff_balances.values())
        # state is created in reset()
        self.store: F.Store | None = None
        self.bus: MessageBus | None = None
        self._next_root: int = 1000  # anchor is root=1, blocks start from 1000
        self.dropped_messages: dict[str, int] = {}
        self.first_drop_error: dict[str, str] = {}

    # ------------------------------------------------------------------
    def reset(self) -> None:
        # D10 fix: anchor bid uses execution_payment=0 per gloas/builder.md:138
        # (non-zero indicates trusted-EL-payment; gossiped bids MUST be 0).
        # The anchor is a zero-value self-build-style starting point.
        anchor = p.BeaconBlock(
            root=1, parent_root=0, slot=0,
            bid=p.ExecutionPayloadBid(0, 0, 100, 0, 0, 0, 0),
        )
        setattr(anchor, "proposer_index", 0)
        self.store = F.get_forkchoice_store(
            anchor,
            effective_balances=self._anchor_eff_balances,
            total_active_balance=self._total_active_balance,
        )
        # Anchor payload is "delivered" so children may claim FULL anchor.
        self.store.payloads[1] = p.ExecutionPayloadEnvelope(0, 1, 100)
        self.bus = MessageBus()
        self._wire_bus()
        self._next_root = 1000
        self.dropped_messages = {}
        self.first_drop_error = {}

    def _record_drop(self, kind: str, err: BaseException) -> None:
        """Centralise spec-handler AssertionError handling.

        Non-strict mode: bump per-kind counter and remember the first error
        message so the exp script can surface it. Strict mode: re-raise so the
        test/diff-test can fail loudly.
        """
        if self.cfg.strict_spec_validation:
            raise err
        self.dropped_messages[kind] = self.dropped_messages.get(kind, 0) + 1
        self.first_drop_error.setdefault(kind, str(err) or repr(err))

    def _wire_bus(self):
        store = self.store
        self.bus.register_handler(
            MessageKind.BLOCK, lambda blk: F.on_block(store, blk))
        self.bus.register_handler(
            MessageKind.PAYLOAD_ENVELOPE,
            lambda env: F.on_execution_payload_envelope(store, env))

    # ------------------------------------------------------------------
    # Run one episode = num_slots-slot chain
    # ------------------------------------------------------------------
    def begin_episode(self) -> None:
        """Phase H3 — expose slot-by-slot stepping for the RL coalition wrapper.

        Initialises per-episode state on the env instance so callers can
        invoke `advance_one_slot()` repeatedly until `episode_done()`. The
        legacy `run_episode()` is a thin wrapper that drives this loop to
        completion.
        """
        if self.store is None:
            self.reset()
        self._slot_events: list[SlotEvent] = []
        self._epoch_events: list[EpochEvent] = []
        self._slot_stats: list[S.SlotStat] = []
        self._pending_atts: list[A.Attestation] = []
        self._pending_payload_msgs: list[
            tuple[A.PayloadAttestationMessage, list[int]]
        ] = []
        self._cur_slot: int = 0  # next slot to run is _cur_slot + 1

    def advance_one_slot(self) -> tuple[SlotEvent, S.SlotStat | None]:
        """Run the next slot; return its (event, stat). Drives the same loop
        body as `run_episode` but in single-slot increments so the RL wrapper
        can set hooks between slots.
        """
        self._cur_slot += 1
        slot = self._cur_slot
        event, stat = self._run_one_slot(
            slot, self._pending_atts, self._pending_payload_msgs)
        self._slot_events.append(event)
        if stat is not None:
            self._slot_stats.append(stat)
        if (self.cfg.enable_ffg
                and slot % config.SLOTS_PER_EPOCH == config.SLOTS_PER_EPOCH - 1):
            self._epoch_events.append(self._process_epoch_boundary())
        return event, stat

    def episode_done(self) -> bool:
        return self._cur_slot >= self.cfg.num_slots

    def finalise_episode(self) -> EpisodeResult:
        """Build EpisodeResult from collected slot/epoch state. Idempotent."""
        return self._summarise(
            self._slot_events, self._epoch_events, self._slot_stats)

    def run_episode(self) -> EpisodeResult:
        """Drive the slot loop to completion. Backwards compatible."""
        self.begin_episode()
        while not self.episode_done():
            self.advance_one_slot()
        return self.finalise_episode()

    def peek_slot_roles(self, slot: int) -> dict:
        """Compute proposer/committee/PTC/default auction-winner for ``slot``
        WITHOUT mutating any state. Used by the RL coalition wrapper to
        construct an observation before deciding the agent's action for
        that slot.
        """
        active = C.get_active_validator_indices(
            self.vset, A.compute_epoch_at_slot(slot))
        proposer = C.get_beacon_proposer_index(
            self.vset, active, self.cfg.seed, slot)
        committee = C.get_beacon_committee(
            active, self.cfg.seed, slot, 0,
            slots_per_epoch=config.SLOTS_PER_EPOCH, committees_per_slot=1,
        )[: self.cfg.committee_size]
        ptc = C.get_ptc(active, self.cfg.seed, slot)
        # Phase L-α — run_auction now returns (winner, effective_amount) | None.
        # peek_slot_roles is observation-only; we report the winner under the
        # *default* bid_value (no INFLATE applied) so the obs sees the
        # auction outcome the default policy would face.
        default_effective = {
            b.index: bp.compute_effective_bid_value(
                self.store, b.index, self._mev_intended_bid(b.index, slot),
            )
            for b in self.bset.builders
        }
        auction = B.run_auction(self.bset, slot, default_effective)
        if auction is None:
            # No external bid passed the cover check: the proposer self-builds.
            default_winner = B.Builder(
                index=bp.BUILDER_INDEX_SELF_BUILD, bid_value=0,
            )
            default_effective_amount = 0
        else:
            default_winner, default_effective_amount = auction
        return {
            "proposer": proposer,
            "committee": committee,
            "ptc": ptc,
            "default_winner": default_winner,
            "default_effective_amount": default_effective_amount,
        }

    @staticmethod
    def _det_uniform(seed_int: int, slot: int, salt: int) -> float:
        """Deterministic uniform in (0, 1) from (seed_int, slot, salt)."""
        h = sha256(f"{seed_int}:{slot}:{salt}".encode()).digest()
        return (int.from_bytes(h[:8], "big") + 1) / (2 ** 64 + 1)

    def _det_normal(self, seed_int: int, slot: int, salt_base: int) -> float:
        """Deterministic standard normal via Box-Muller on two uniforms."""
        u1 = self._det_uniform(seed_int, slot, salt_base)
        u2 = self._det_uniform(seed_int, slot, salt_base + 1)
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    def _mev_intended_bid(self, builder_index: int, slot: int) -> int:
        """Intended bid value (gwei) for ``builder_index`` at ``slot``.

        Backward-compatible: with ``mev_enabled=False`` returns the flat
        ``cfg.bid_value_gwei`` (pre-M behaviour, bit-for-bit). With MEV on::

            intended(b,t) = mev_base * skill(b)
                            * exp(mev_slot_sigma    * z_common(t))     # shared
                            * exp(mev_builder_sigma * z_idio(b,t))     # private

        The COMMON factor is the per-slot market opportunity (every builder
        sees the same CEX-DEX/liquidation slot). The IDIOSYNCRATIC factor is
        each builder's per-slot capture — without it the highest-skill builder
        wins every slot (a degenerate fixed monopoly, not an auction). Both are
        DETERMINISTIC in (mev_seed, slot[, builder]) so paired evaluation sees
        identical MEV at a given seed/slot. Salt layout: common uses 0/1,
        builder b uses 2+2b / 3+2b (disjoint).
        """
        cfg = self.cfg
        if not cfg.mev_enabled:
            return cfg.bid_value_gwei
        # Mix the episode seed (cfg.seed, set per reset) into the MEV stream so
        # different episodes draw different MEV, while paired honest/strategy
        # runs at the SAME episode seed see IDENTICAL MEV (delta stays valid).
        seed_mix = cfg.mev_seed ^ int.from_bytes(cfg.seed[:8], "little")
        factor = 1.0
        if cfg.mev_slot_sigma > 0.0:
            factor *= math.exp(
                cfg.mev_slot_sigma * self._det_normal(seed_mix, slot, 0))
        if cfg.mev_builder_sigma > 0.0:
            factor *= math.exp(
                cfg.mev_builder_sigma
                * self._det_normal(seed_mix, slot, 2 + 2 * builder_index))
        skill = (cfg.mev_builder_skill[builder_index]
                 if builder_index < len(cfg.mev_builder_skill) else 1.0)
        return int(cfg.mev_base_gwei * skill * factor)

    def _queue_beacon_attestations(
        self, *, slot: int, active: list[int], adv_ctx: AdvCtx,
        pending_atts: list[A.Attestation],
    ) -> list[int]:
        """Generate regular beacon attestations at the gloas 25% deadline."""
        store = self.store
        att_due_ms = slot * config.MILLIS_PER_SLOT + p.get_attestation_due_ms()
        F.on_tick(store, att_due_ms)
        committee = C.get_beacon_committee(
            active, self.cfg.seed, slot, 0,
            slots_per_epoch=config.SLOTS_PER_EPOCH,
            committees_per_slot=1,
        )[: self.cfg.committee_size]

        head_for_vote = F.get_head(store)
        target_root = head_for_vote.root
        target_block = store.blocks.get(target_root)
        target_block_slot = target_block.slot if target_block else 0

        epoch = A.compute_epoch_at_slot(slot)
        epoch_first_slot = epoch * config.SLOTS_PER_EPOCH
        if target_block_slot >= epoch_first_slot:
            try:
                ffg_target_root = F.get_ancestor(
                    store, target_root, epoch_first_slot,
                ).root
            except KeyError:
                ffg_target_root = target_root
        else:
            ffg_target_root = target_root

        head_payload_revealed = p.is_payload_verified(store, target_root)
        full_idx, empty_idx = [], []
        for i, attester_idx in enumerate(committee):
            action = self.adversary.attester_action(
                attester_idx, slot, target_root, adv_ctx,
            )
            # Per-attester reward/penalty (altair get_flag_index_deltas): a
            # committee member that casts a timely attestation earns the
            # attestation reward; one that abstains (WITHHOLD_VOTE) incurs the
            # missed-attestation penalty. This is the cost a byz committee pays
            # to suppress a slot's builder-payment quorum.
            R.credit_attester(
                store, attester_idx,
                attested=(action != AttesterAction.WITHHOLD_VOTE), epoch=epoch,
            )
            if action == AttesterAction.WITHHOLD_VOTE:
                continue
            if action == AttesterAction.VOTE_EMPTY:
                empty_idx.append(i)
            elif action == AttesterAction.VOTE_FULL:
                if head_payload_revealed and target_block_slot < slot:
                    full_idx.append(i)
                else:
                    empty_idx.append(i)
            else:
                if head_payload_revealed and target_block_slot < slot:
                    full_idx.append(i)
                else:
                    empty_idx.append(i)

        def _attestation(indices: list[int], vote_full: bool) -> A.Attestation:
            bits = tuple(i in indices for i in range(len(committee)))
            return A.Attestation(
                data=A.AttestationData(
                    slot=slot,
                    index=1 if vote_full else 0,
                    beacon_block_root=target_root,
                    source=A.Checkpoint(0, 1),
                    target=A.Checkpoint(epoch, ffg_target_root),
                ),
                aggregation_bits=bits,
                committee_indices=tuple(committee),
            )

        if full_idx:
            pending_atts.append(_attestation(full_idx, vote_full=True))
        if empty_idx:
            pending_atts.append(_attestation(empty_idx, vote_full=False))
        return committee

    def _run_one_slot(
        self, slot: int,
        pending_atts: list[A.Attestation],
        pending_payload_msgs: list[tuple[A.PayloadAttestationMessage, list[int]]],
    ) -> tuple[SlotEvent, S.SlotStat | None]:
        store = self.store
        bus = self.bus
        slot_start_ms = slot * config.MILLIS_PER_SLOT

        # 1) tick to slot boundary
        F.on_tick(store, slot_start_ms)

        # 1a) flush previous-slot attestations now that they are 1 slot old
        included_atts: list[A.Attestation] = []
        for att in pending_atts:
            try:
                F.on_attestation(
                    store, att,
                    enforce_ffg_target=self.cfg.enforce_ffg_target_consistency,
                )
            except AssertionError as e:
                self._record_drop("attestation_invalid", e)
                continue
            included_atts.append(att)
            bp.accumulate_same_slot_attestation_weight(
                store,
                voted_block_slot=att.data.slot,
                voted_block_root=att.data.beacon_block_root,
                attesting_validators=A.get_attesting_indices(att),
            )
        pending_atts.clear()
        # 1b) flush PTC messages (these target previous slot's block)
        for msg, ptc in pending_payload_msgs:
            try:
                F.on_payload_attestation_message(store, msg, ptc)
            except AssertionError as e:
                self._record_drop("ptc_invalid", e)
                continue
        pending_payload_msgs.clear()

        # 2) proposer chooses parent
        active = C.get_active_validator_indices(self.vset, A.compute_epoch_at_slot(slot))
        proposer = C.get_beacon_proposer_index(self.vset, active, self.cfg.seed, slot)
        adv_ctx = AdvCtx(store=store, bus=bus, current_slot=slot, current_ms=slot_start_ms)
        # Phase L-α: proposer_parent_hook lets a byz proposer build on a
        # non-head parent (BUILD_ON_NONHEAD primitive). Returns None to
        # defer to the adversary class (which is usually honest = head).
        decision = None
        if self.cfg.proposer_parent_hook is not None:
            decision = self.cfg.proposer_parent_hook(proposer, slot)
        if decision is None:
            decision = self.adversary.proposer_action(proposer, slot, adv_ctx)
        if decision is None:
            decision = (F.get_head(store).root, F.get_head(store).payload_status)
        parent_root, parent_status = decision

        # 3) run the builder auction → pick winner whose bid the proposer
        # includes. Phase L-α: each builder's effective bid amount is
        # computed under the cover-bid check (accepted amount or zero);
        # auction ranking and settlement amount are bound to this same
        # number, closing the "high rank, low payment" sim-only escape.
        # Phase H3: if a coalition's preferred_builder_hook is set, byz
        # proposer can override the auction outcome to force-pick a byz
        # builder. Spec analogue: the proposer chooses any signed bid in
        # their mempool; in practice they pick max-value, but adversarial
        # proposers may deviate.
        intended_amounts: dict[int, int] = {}
        effective_amounts: dict[int, int] = {}
        for b in self.bset.builders:
            intended = self._mev_intended_bid(b.index, slot)
            if self.cfg.bid_value_override_hook is not None:
                ov = self.cfg.bid_value_override_hook(b.index, slot)
                if ov is not None:
                    intended = ov
            intended_amounts[b.index] = intended
            effective_amounts[b.index] = bp.compute_effective_bid_value(
                store, b.index, intended,
            )
        auction = B.run_auction(self.bset, slot, effective_amounts)
        if auction is None:
            # No external builder passed `can_builder_cover_bid`. Gloas lets
            # the proposer self-build with a zero-value sentinel bid.
            winner = B.Builder(index=bp.BUILDER_INDEX_SELF_BUILD, bid_value=0)
            effective_amount = 0
        else:
            winner, effective_amount = auction
        if self.cfg.preferred_builder_hook is not None:
            override_idx = self.cfg.preferred_builder_hook(proposer, slot)
            if (override_idx is not None and 0 <= override_idx < len(self.bset)
                    and effective_amounts.get(override_idx, 0) > 0):
                winner = self.bset.builders[override_idx]
                effective_amount = effective_amounts[override_idx]
        winner_intended_amount = intended_amounts.get(
            winner.index, 0)
        winner_bid_was_clipped = effective_amount < winner_intended_amount

        # 4) build the block
        new_root = self._next_root
        self._next_root += 1
        parent = store.blocks[parent_root]
        # bid commits FULL parent iff parent_status == FULL; otherwise EMPTY.
        # Spec: parent_block_hash == parent.bid.block_hash => FULL.
        if parent_status == PayloadStatus.FULL:
            bid_parent_hash = parent.bid.block_hash
        else:
            bid_parent_hash = 999_999  # any non-matching value
        # `bid.value` = what builder pays proposer (per gloas/builder.md:135).
        # `bid.execution_payment = 0` per gloas/builder.md:138 (non-zero
        # indicates a trusted execution-layer payment, not standard flow).
        # Phase L-α: `bid.value` uses the auction's `effective_amount`
        # (already accepted by the cover-bid check). This binds the amount the
        # builder will be debited at settlement to the amount that won the
        # auction — no sim-only "high-rank, low-commitment" escape.
        bid = p.ExecutionPayloadBid(
            parent_block_hash=bid_parent_hash, parent_block_root=parent_root,
            block_hash=new_root * 100, builder_index=winner.index, slot=slot,
            value=effective_amount,
            execution_payment=0,
        )
        block = p.BeaconBlock(
            root=new_root, parent_root=parent_root, slot=slot, bid=bid)
        setattr(block, "proposer_index", proposer)

        # 4a) deliver block immediately (synchronous baseline)
        try:
            F.on_block(store, block)
        except AssertionError:
            # block invalid (e.g., FULL parent without verified payload).
            # Adversary lost this slot — skip and continue.
            # D4 fix: enqueue_pending_payment moved BELOW on_block so failed
            # blocks don't leave a ghost payment in the store (spec invariant:
            # process_execution_payload_bid runs as part of process_block, so
            # a failed block reverts the enqueue).
            return (
                SlotEvent(
                    slot=slot, proposer_index=proposer, block_root=-1,
                    parent_root=parent_root, parent_status=parent_status,
                    payload_revealed=False,
                    head_root_at_end=F.get_head(store).root,
                    head_status_at_end=F.get_head(store).payload_status,
                    builder_action="block_invalid", proposer_action="failed",
                    winner_builder_index=winner.index,
                    intended_bid_value_gwei=winner_intended_amount,
                    effective_bid_value_gwei=effective_amount,
                    bid_was_clipped=winner_bid_was_clipped,
                ),
                None,
            )
        # Gloas `process_parent_execution_payload`: a child claiming FULL
        # settles its parent's builder payment immediately.
        if parent_status == PayloadStatus.FULL:
            bp.settle_parent_payment_immediately(
                store,
                slot=parent.slot,
                builder_index=parent.bid.builder_index,
                proposer_index=getattr(parent, "proposer_index", 0),
                amount=parent.bid.value,
            )
        # Enqueue the current bid after on_block succeeds so invalid blocks
        # do not leave ghost payments.
        bp.enqueue_pending_payment(
            store, slot=slot, builder_index=winner.index,
            proposer_index=proposer, amount=bid.value,
        )
        # The current block includes the previous-slot attestations. In the
        # spec, the including block's proposer receives this reward.
        for att in included_atts:
            R.credit_proposer_reward(
                store,
                proposer_index=proposer,
                attesting_validators=A.get_attesting_indices(att),
                attestation_epoch=att.data.target.epoch,
            )

        # 4b) builder reveals or withholds — routed by the auction winner's
        # builder-pool index, independent of the proposer's validator index.
        # Phase H3: forced_builder_action_hook lets the RL coalition force
        # WITHHOLD even if the adversary class would have revealed.
        # late_reveal_hook takes precedence: the regular committee and PTC
        # observe the payload as absent, then the deferred envelope arrives.
        late_reveal_active = False
        if self.cfg.late_reveal_hook is not None:
            late_reveal_active = bool(
                self.cfg.late_reveal_hook(winner.index, slot))
        builder_action_override = None
        if self.cfg.forced_builder_action_hook is not None:
            builder_action_override = self.cfg.forced_builder_action_hook(
                winner.index, slot)
        if late_reveal_active:
            # Late-reveal builder schedules envelope NOW but env will hold
            # delivery until after both protocol deadlines.
            env_msg = p.ExecutionPayloadEnvelope(
                builder_index=winner.index, beacon_block_root=new_root,
                block_hash=new_root * 1000,
            )
            bus.schedule(Message(
                sender=winner.index, kind=MessageKind.PAYLOAD_ENVELOPE,
                payload=env_msg, deliver_at_ms=slot_start_ms,
            ))
            builder_action = BuilderAction.REVEAL_LATE
        elif builder_action_override is not None:
            # Convey the force-withhold by simply not scheduling the envelope.
            # This bypasses the adversary's normal builder_action; the action
            # name still flows to the SlotEvent for tracing.
            builder_action = builder_action_override
        else:
            builder_action = self.adversary.builder_action(
                winner.index, new_root, adv_ctx)
        # Deliver scheduled messages now UNLESS late-reveal is active. Under
        # late_reveal we hold the envelope so the PTC vote step sees no
        # payload in store and votes absent.
        if not late_reveal_active:
            bus.deliver_due(slot_start_ms)
        payload_revealed = new_root in store.payloads

        # 5) Regular beacon attestations at the gloas 25% deadline.
        committee = self._queue_beacon_attestations(
            slot=slot, active=active, adv_ctx=adv_ctx,
            pending_atts=pending_atts,
        )

        # 6) PTC votes at the gloas 75% deadline — honest baseline: local view.
        # Phase L-α: ptc_vote_hook can override an individual PTC member's
        # vote (used by FRAUD_ABSENT to vote False on a revealed payload, or
        # the dual fraud-present to vote True on a withheld one). The hook
        # is queried with the honest local-view value so it can choose to
        # match or invert.
        #
        # Emit one message per unique PTC validator, not one per seat. A
        # validator with K seats publishes one signed message.
        # Fork-choice ``on_payload_attestation_message`` is per-validator
        # idempotent (sets every seat held by ``msg.validator_index``), so
        # one message per validator suffices.
        ptc_due_ms = slot_start_ms + p.get_payload_attestation_due_ms()
        F.on_tick(store, ptc_due_ms)
        ptc = C.get_ptc(active, self.cfg.seed, slot)
        seen_ptc_voters: set[int] = set()
        for ptc_idx in ptc:
            if ptc_idx in seen_ptc_voters:
                continue
            seen_ptc_voters.add(ptc_idx)
            # vote payload_present iff locally verified
            local_view_present = p.is_payload_verified(store, new_root)
            present = local_view_present
            if self.cfg.ptc_vote_hook is not None:
                override = self.cfg.ptc_vote_hook(
                    ptc_idx, slot, local_view_present)
                if override is not None:
                    present = bool(override)
            msg = A.PayloadAttestationMessage(
                validator_index=ptc_idx,
                data=A.PayloadAttestationData(
                    beacon_block_root=new_root, slot=slot,
                    payload_present=present, blob_data_available=True,
                ),
            )
            # Queue for next slot
            pending_payload_msgs.append((msg, list(ptc)))
        # LATE_REVEAL: flush the deferred envelope after both deadlines.
        if late_reveal_active:
            bus.deliver_due(ptc_due_ms + 1)
            payload_revealed = new_root in store.payloads

        # 7) end-of-slot snapshot
        head = F.get_head(store)
        event = SlotEvent(
            slot=slot, proposer_index=proposer, block_root=new_root,
            parent_root=parent_root, parent_status=parent_status,
            payload_revealed=payload_revealed,
            head_root_at_end=head.root, head_status_at_end=head.payload_status,
            builder_action=builder_action,
            proposer_action=f"build_on({parent_root}, {parent_status.name})",
            winner_builder_index=winner.index,
            intended_bid_value_gwei=winner_intended_amount,
            effective_bid_value_gwei=effective_amount,
            bid_was_clipped=winner_bid_was_clipped,
        )
        # Phase G — record per-slot Byzantine-realisation stats. Committee here
        # is the truncated slot committee actually used for voting; PTC is the
        # spec-derived list (may contain duplicates per balance-weighted
        # selection). Weights use the validator set's effective_balance lookup.
        byz_v_set = self.cfg.byzantine_validators
        eff = self.vset.effective_balance
        com_byz_indices = [i for i in committee if i in byz_v_set]
        com_total_w = sum(eff(i) for i in committee)
        com_byz_w = sum(eff(i) for i in com_byz_indices)
        ptc_byz_indices = [i for i in ptc if i in byz_v_set]
        ptc_total_w = sum(eff(i) for i in ptc)
        ptc_byz_w = sum(eff(i) for i in ptc_byz_indices)
        stat = S.SlotStat(
            slot=slot,
            proposer_index=proposer,
            proposer_byz=(proposer in byz_v_set),
            committee_size=len(committee),
            committee_byz_count=len(com_byz_indices),
            committee_byz_weight=com_byz_w,
            committee_total_weight=com_total_w,
            ptc_size=len(ptc),
            ptc_byz_count=len(ptc_byz_indices),
            ptc_byz_weight=ptc_byz_w,
            ptc_total_weight=ptc_total_w,
            builder_index=winner.index,
            builder_byz=winner.is_byzantine,
        )
        return event, stat

    # ------------------------------------------------------------------
    def _process_epoch_boundary(self) -> EpochEvent:
        """Tier 2b: run FFG at the end of an epoch.

        Tick to the last ms of the slot so attestations from that slot are
        visible in store.latest_messages before FFG counts them.
        """
        store = self.store
        current_slot = F.get_current_slot(store)
        epoch = current_slot // config.SLOTS_PER_EPOCH
        # Settle unresolved payments from epoch E-1 at the end of epoch E,
        # matching gloas `process_builder_pending_payments`. FULL-parent
        # payments normally settle earlier during child-block processing.
        bp.process_settlement_at_epoch_boundary(store, current_epoch=epoch)

        justified_before = (store.current_justified_checkpoint.epoch
                            if store.current_justified_checkpoint else 0)
        finalized_before = (store.finalized_checkpoint.epoch
                            if store.finalized_checkpoint else 0)

        # Compute current-epoch target balance for reporting (before FFG mutates state)
        cur_target_root = ffg.get_target_root_for_epoch(store, epoch)
        target_balance = ffg.compute_target_balance_for_root(
            store, epoch, cur_target_root)

        ffg.process_justification_and_finalization(store)

        justified_after = (store.current_justified_checkpoint.epoch
                           if store.current_justified_checkpoint else 0)
        finalized_after = (store.finalized_checkpoint.epoch
                           if store.finalized_checkpoint else 0)

        return EpochEvent(
            epoch=epoch,
            justified_epoch_before=justified_before,
            justified_epoch_after=justified_after,
            finalized_epoch_before=finalized_before,
            finalized_epoch_after=finalized_after,
            target_balance=target_balance,
            total_active_balance=store.total_active_balance,
            justified_this_epoch=(justified_after > justified_before),
        )

    # ------------------------------------------------------------------
    def _summarise(self, events: list[SlotEvent],
                   epoch_events: list[EpochEvent] | None = None,
                   slot_stats: list[S.SlotStat] | None = None) -> EpisodeResult:
        store = self.store
        head = F.get_head(store)
        # Trace canonical ancestry from final head
        canonical = set()
        cur = head.root
        while cur in store.blocks:
            canonical.add(cur)
            parent = store.blocks[cur].parent_root
            if parent == 0 or parent == cur:
                break
            cur = parent

        num_reorgs = 0
        empty_blocks = 0
        num_full = 0
        for ev in events:
            if ev.block_root != -1 and ev.block_root not in canonical:
                ev.reorged_a_block = True
                num_reorgs += 1
            if ev.head_status_at_end == PayloadStatus.EMPTY:
                empty_blocks += 1
            if ev.head_status_at_end == PayloadStatus.FULL:
                num_full += 1

        ee = epoch_events or []
        n_failed = sum(1 for x in ee if not x.justified_this_epoch
                       and x.epoch >= 2)  # FFG skipped before epoch 2
        expired_slots = set(store.expired_payment_slots)
        full_but_unsettled = sum(
            1 for ev in events
            if ev.slot in expired_slots
            and ev.head_status_at_end == PayloadStatus.FULL
        )

        return EpisodeResult(
            slot_events=events,
            epoch_events=ee,
            final_head_root=head.root,
            final_head_status=head.payload_status,
            num_reorgs=num_reorgs,
            empty_blocks=empty_blocks,
            num_full=num_full,
            final_justified_epoch=(store.current_justified_checkpoint.epoch
                                   if store.current_justified_checkpoint else 0),
            final_finalized_epoch=(store.finalized_checkpoint.epoch
                                   if store.finalized_checkpoint else 0),
            num_epochs_failed_to_justify=n_failed,
            settled_payments=len(store.settled_payment_slots),
            expired_payments=len(store.expired_payment_slots),
            settled_payment_amount_gwei=store.settled_payment_amount_gwei,
            expired_payment_amount_gwei=store.expired_payment_amount_gwei,
            pending_payments_at_end=len(store.builder_pending_payments),
            full_but_unsettled_blocks=full_but_unsettled,
            dropped_messages=dict(self.dropped_messages),
            first_drop_error=dict(self.first_drop_error),
            slot_stats=list(slot_stats) if slot_stats else [],
        )
