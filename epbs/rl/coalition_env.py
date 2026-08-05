"""Gymnasium wrapper around `Tier2Environment` for the RL coalition agent.

Phase H3 deliverable. The PPO agent (epbs/rl/ppo.py + actor_critic.py) drives
a single coalition entity that controls both byz_validators and byz_builders.
Each environment step = one slot. The agent picks one of 4 actions per slot:

    0 = HONEST                    — no override; default auction; honest reveal
    1 = FORCE_PICK_BYZ_BUILDER    — if proposer is byz, override auction to
                                     a byz builder (which still reveals)
    2 = BYZ_BUILDER_WITHHOLD      — if byz builder won the auction, override
                                     to withhold (proposer untouched)
    3 = FORCE_PICK + WITHHOLD     — both 1 and 2

Reward per step is the *delta* in coalition utility:

    U_coalition = sum(settled_payments[i] for i in byz_b)
                + sum(proposer_rewards[i] for i in byz_v)
    r_t = U_coalition_t - U_coalition_{t-1}

Episode terminates at slot ``num_slots``.

The observation is a 12-dim float vector (see ``_build_obs``); all features
are pre-step (computed from ``peek_slot_roles``) so the agent has the info
needed to decide whether action 1/2/3 would even apply to this slot.
"""
from __future__ import annotations

import numpy as np
from dataclasses import replace

try:
    import gymnasium as gym
    from gymnasium import spaces
    _HAS_GYM = True
except ImportError:
    _HAS_GYM = False
    gym = object  # type: ignore

import config
from epbs import builder as B
from epbs import builder_payments as bp
from epbs import forkchoice as F
from epbs.adversary import (
    AttesterAction, BuilderAction, HonestAdversary,
)
from epbs.env_tier2 import Tier2Config, Tier2Environment
from epbs.primitives import PayloadStatus


# Action enum (legacy 4-action CoalitionEnv — unchanged for exp11 reproducibility)
A_HONEST = 0
A_FORCE_PICK = 1
A_WITHHOLD = 2
A_FORCE_PICK_AND_WITHHOLD = 3

OBS_DIM = 12

# Phase L-α — multi-head action space layout.
# Heads = [P, B_bid, B_reveal, C, PTC]
HEAD_SIZES_L = (3, 2, 3, 3, 2)
H_P, H_B_BID, H_B_REVEAL, H_C, H_PTC = 0, 1, 2, 3, 4
# Sampling order: P, B_bid, C, PTC, then B_reveal (depends on B_bid).
SAMPLE_ORDER_L = [H_P, H_B_BID, H_C, H_PTC, H_B_REVEAL]
TOTAL_HEAD_DIMS_L = sum(HEAD_SIZES_L)  # 13

# P-head primitives
P_HONEST_PROPOSE = 0
P_FORCE_PICK_BYZ_BUILDER = 1
P_BUILD_ON_NONHEAD = 2
# B_bid primitives
B_BID_NORMAL = 0
B_BID_INFLATE_LOW = 1
INFLATE_MULTIPLIER = 5  # INFLATE_LOW intends 5× cfg.bid_value_gwei
# B_reveal primitives
B_REVEAL_HONEST = 0
B_REVEAL_WITHHOLD = 1
B_REVEAL_LATE = 2
# C primitives
C_HONEST_ATTEST = 0
C_WITHHOLD_VOTE = 1
C_VOTE_EMPTY = 2
# PTC primitives
PTC_HONEST = 0
PTC_FRAUD_ABSENT = 1

OBS_DIM_L = 21


def _marked_pending_flows(store, byz_v: frozenset, byz_b: frozenset):
    """Mark-to-settlement value of currently-pending builder payments.

    A pending payment whose accrued same-slot-attestation weight already meets
    the builder-payment quorum is economically *committed* — it will settle at
    its epoch boundary regardless of payload presence (see
    builder_payments.process_settlement_at_epoch_boundary and the spec
    process_attestation weight accrual, which is payload-independent). Counting
    such pending payments removes the episode-truncation/censoring artifact:
    without it, a withheld bid's delayed (epoch-boundary) settlement can fall
    outside the episode window and masquerade as "avoided", inflating coalition
    utility. Payments with weight < quorum (e.g. genuinely suppressed by a
    byzantine committee) will expire and are correctly NOT counted.

    Returns (marked_credit, marked_debit) in gwei for the coalition.
    """
    quorum = bp.get_quorum_threshold(store.total_active_balance)
    credit = debit = 0
    for pay in store.builder_pending_payments.values():
        if pay.weight < quorum:
            continue
        if pay.proposer_index in byz_v:
            credit += pay.amount
        if pay.builder_index in byz_b:
            debit += pay.amount
    return credit, debit


class CoalitionEnv(gym.Env if _HAS_GYM else object):  # type: ignore[misc]
    """Gymnasium env wrapping `Tier2Environment` for a coalition RL agent."""

    metadata = {"render_modes": []}

    def __init__(self, cfg: Tier2Config):
        """``cfg`` is a Tier2Config with byz_validators / byz_builders set.
        Other fields (num_slots, num_validators, ...) drive episode length.
        """
        self.cfg_template = cfg
        self.byz_v = frozenset(cfg.byzantine_validators)
        self.byz_b = frozenset(cfg.byzantine_builders)
        # Mark-to-settlement accounting (default ON). Counts pending payments
        # already at quorum as committed, removing the episode-truncation
        # censoring artifact. Set False to reproduce the pre-fix (settled-only)
        # utility for the artifact A/B comparison.
        self.mark_to_settlement = True

        if _HAS_GYM:
            self.action_space = spaces.Discrete(4)
            self.observation_space = spaces.Box(
                low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)

        self._pending_action: int = A_HONEST
        self._last_intended: int = A_HONEST
        self._last_executed: int = A_HONEST
        self.env: Tier2Environment | None = None
        self._u_last: int = 0
        self._u_total: int = 0

    # -- Gym API ----------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if _HAS_GYM:
            super().reset(seed=seed)
        # Fresh cfg per episode so the hooks (which reference self) stay
        # valid across resets.
        cfg = replace(
            self.cfg_template,
            seed=(seed or 0).to_bytes(32, "little"),
            preferred_builder_hook=self._preferred_builder_hook,
            forced_builder_action_hook=self._forced_builder_hook,
        )
        # Adversary doesn't matter — hooks override its decisions.
        self.env = Tier2Environment(cfg, HonestAdversary())
        self.env.reset()
        self.env.begin_episode()
        self._u_last = 0
        self._u_total = 0
        return self._build_obs(), {}

    def step(self, action: int):
        assert self.env is not None, "reset() first"
        if self.env.episode_done():
            raise RuntimeError("step() after episode_done; call reset()")

        # Phase I-4 — action masking via substitution. Compute the valid
        # action mask for the NEXT slot's roles; if the agent picked an
        # action whose precondition doesn't hold, silently fold it down to
        # the closest valid action. This keeps the action signal informative
        # without changing the PPO loop:
        #   FORCE_PICK requires byz proposer this slot
        #   WITHHOLD requires byz builder won the default auction this slot
        # Mask order matches the action enum.
        intended = int(action)
        mask = self._action_mask_for_next_slot()
        if not mask[intended]:
            # Downgrade in priority order: BOTH→FORCE_PICK or WITHHOLD,
            # FORCE_PICK→HONEST, WITHHOLD→HONEST.
            if intended == A_FORCE_PICK_AND_WITHHOLD:
                if mask[A_FORCE_PICK]:
                    intended = A_FORCE_PICK
                elif mask[A_WITHHOLD]:
                    intended = A_WITHHOLD
                else:
                    intended = A_HONEST
            else:
                intended = A_HONEST
        self._pending_action = intended
        self._last_intended = int(action)
        self._last_executed = intended

        event, _stat = self.env.advance_one_slot()

        u_now = self._coalition_utility()
        reward = float(u_now - self._u_last)
        self._u_last = u_now
        self._u_total = u_now

        terminated = self.env.episode_done()
        obs = (self._build_obs() if not terminated
               else np.zeros(OBS_DIM, dtype=np.float32))
        info = {
            "slot": event.slot,
            "u_coalition_running": u_now,
            "byz_proposer_this_slot": event.proposer_index in self.byz_v,
            "head_status": event.head_status_at_end.name,
            "action_intended": self._last_intended,
            "action_executed": self._last_executed,
            "masked": self._last_intended != self._last_executed,
        }
        return obs, reward, terminated, False, info

    def _action_mask_for_next_slot(self) -> list[bool]:
        """Return a 4-bool vector — True means the action is applicable
        given the next slot's roles. HONEST is always valid.
        """
        env = self.env
        next_slot = env._cur_slot + 1
        if next_slot > self.cfg_template.num_slots:
            return [True, False, False, False]
        roles = env.peek_slot_roles(next_slot)
        is_byz_proposer = roles["proposer"] in self.byz_v
        is_byz_default_winner = roles["default_winner"].is_byzantine
        return [
            True,                                       # HONEST
            is_byz_proposer and bool(self.byz_b),       # FORCE_PICK
            is_byz_default_winner,                      # WITHHOLD
            is_byz_proposer and bool(self.byz_b),       # FORCE_PICK+WITHHOLD
        ]

    # -- Hooks consumed by Tier2Environment -------------------------------
    def _preferred_builder_hook(self, proposer_index: int, slot: int):
        """Return a byz builder index iff byz proposer + action requests
        force-pick. None otherwise (default auction outcome stands).
        """
        if proposer_index not in self.byz_v or not self.byz_b:
            return None
        if self._pending_action in (A_FORCE_PICK, A_FORCE_PICK_AND_WITHHOLD):
            return min(self.byz_b)  # deterministic pick
        return None

    def _forced_builder_hook(self, builder_index: int, slot: int):
        """Return 'withhold' iff byz builder + action requests it. None
        means defer to the env's normal adversary.builder_action path.
        """
        if builder_index not in self.byz_b:
            return None
        if self._pending_action in (A_WITHHOLD, A_FORCE_PICK_AND_WITHHOLD):
            return BuilderAction.WITHHOLD
        return None

    # -- Utility computation ---------------------------------------------
    def _coalition_utility(self) -> int:
        """Joint utility of coalition (byz_validators + byz_builders).

        Per gloas/builder.md, ``bid.value`` is the amount the **builder pays
        the proposer** when their bid is accepted. So:

          U_byz_validators = (proposer_credits received by byz_v)
                           + (proposer_rewards earned by byz_v as proposers
                              for attestation inclusion)
          U_byz_builders   = -(builder_debits paid out by byz_b)

        Within the coalition, a settled payment from byz_builder to
        byz_validator is an internal transfer (gain + loss cancel). The
        net coalition utility from settlements comes from cross-boundary
        flows: gain when honest builder pays byz proposer; loss when byz
        builder pays honest proposer; zero (within) when both endpoints
        are coalition members.
        """
        store = self.env.store
        proposer_credits = sum(
            store.proposer_credits.get(i, 0) for i in self.byz_v
        )
        builder_debits = sum(
            store.builder_debits.get(i, 0) for i in self.byz_b
        )
        proposer_rewards = sum(
            store.proposer_rewards.get(i, 0) for i in self.byz_v
        )
        # Phase H+ — byz committee's own attestation reward/penalty. Abstaining
        # to suppress a builder-payment quorum forfeits the reward and incurs a
        # penalty, so this term is the modelled cost of the suppression attack.
        attester_net = sum(
            getattr(store, "attester_rewards", {}).get(i, 0) for i in self.byz_v
        )
        u = (proposer_credits - builder_debits + proposer_rewards
             + attester_net)
        if self.mark_to_settlement:
            m_credit, m_debit = _marked_pending_flows(
                store, self.byz_v, self.byz_b)
            u += m_credit - m_debit
        return int(u)

    # -- Observation ------------------------------------------------------
    def _build_obs(self) -> np.ndarray:
        env = self.env
        next_slot = env._cur_slot + 1
        if next_slot > self.cfg_template.num_slots:
            return np.zeros(OBS_DIM, dtype=np.float32)

        roles = env.peek_slot_roles(next_slot)
        proposer = roles["proposer"]
        committee = roles["committee"]
        ptc = roles["ptc"]
        default_winner = roles["default_winner"]

        slot_in_epoch = next_slot % config.SLOTS_PER_EPOCH
        # Head status (last known)
        if env._slot_events:
            head_status = env._slot_events[-1].head_status_at_end
        else:
            head_status = PayloadStatus.PENDING

        # Committee byz weight fraction (count proxy since uniform balance)
        com_size = max(1, len(committee))
        com_byz_frac = sum(1 for vi in committee if vi in self.byz_v) / com_size

        # PTC byz weight fraction
        ptc_size = max(1, len(ptc))
        ptc_byz_frac = sum(1 for vi in ptc if vi in self.byz_v) / ptc_size

        # Pending payments still unsettled (count, scaled)
        n_pending = len(env.store.builder_pending_payments)
        # Cap at 64 to keep in [0, 1]
        pending_scaled = min(1.0, n_pending / 64.0)

        # Justified epoch / total epochs
        max_epochs = max(1, self.cfg_template.num_slots // config.SLOTS_PER_EPOCH)
        cur_just = (env.store.current_justified_checkpoint.epoch
                    if env.store.current_justified_checkpoint else 0)
        just_frac = cur_just / max_epochs

        # Reward accumulator scaled. Bug fix (2026-05-25): clip BOTH
        # bounds to match the declared Box(low=-1, high=1); pre-fix only
        # clipped above, so a -20M coalition return scaled to -200.
        u_scaled = float(np.clip(self._u_total / 100_000.0, -1.0, 1.0))

        is_byz_proposer = float(proposer in self.byz_v)
        is_byz_default_winner = float(default_winner.is_byzantine)

        obs = np.array([
            slot_in_epoch / config.SLOTS_PER_EPOCH,            # 0
            1.0 if head_status == PayloadStatus.EMPTY else 0.0,  # 1
            1.0 if head_status == PayloadStatus.FULL else 0.0,   # 2
            1.0 if head_status == PayloadStatus.PENDING else 0.0,  # 3
            u_scaled,                                            # 4
            just_frac,                                           # 5
            is_byz_proposer,                                     # 6
            is_byz_default_winner,                               # 7
            com_byz_frac,                                        # 8
            ptc_byz_frac,                                        # 9
            pending_scaled,                                      # 10
            # spare: byz_b share (constant across episode but lets the same
            # policy generalise across cells if we later share params)
            len(self.byz_b) / max(1, self.cfg_template.num_builders),  # 11
        ], dtype=np.float32)
        return obs


# ===========================================================================
# Phase L-α — multi-head coalition env (MultiDiscrete[3, 2, 3, 3, 2])
# ===========================================================================
class _CoalitionAdversaryL:
    """Custom adversary used by CoalitionEnvL.

    Translates the C-head (committee) primitive into per-byz-attester
    ``AttesterAction``. Honest attesters always vote per local view.
    Non-attester methods defer to HonestAdversary.

    The adversary reads ``coalition._pending_action`` (a 5-tuple stored on
    the env wrapper) so the same per-slot action drives every byz committee
    member. PTC and proposer/builder paths use Tier2Config hooks rather
    than this adversary because those endpoints are hook-gated in env_tier2.
    """

    def __init__(self, coalition: "CoalitionEnvL"):
        self.coalition = coalition
        self._honest = HonestAdversary()

    def builder_action(self, builder_index, block_root, ctx):
        # Builder reveal/withhold/late-reveal is handled entirely via
        # env_tier2 hooks (forced_builder_action_hook + late_reveal_hook),
        # so the adversary's default path is honest reveal.
        return self._honest.builder_action(builder_index, block_root, ctx)

    def attester_action(self, validator_index, slot, view_root, ctx):
        if validator_index not in self.coalition.byz_v:
            return AttesterAction.HONEST
        c_action = self.coalition._pending_action[H_C]
        if c_action == C_WITHHOLD_VOTE:
            return AttesterAction.WITHHOLD_VOTE
        if c_action == C_VOTE_EMPTY:
            return AttesterAction.VOTE_EMPTY
        return AttesterAction.HONEST

    def proposer_action(self, proposer_index, slot, ctx):
        # Parent selection is hook-driven (proposer_parent_hook); fallback
        # to honest head-build.
        return self._honest.proposer_action(proposer_index, slot, ctx)


class CoalitionEnvL(gym.Env if _HAS_GYM else object):  # type: ignore[misc]
    """Phase L-α multi-head coalition env.

    Action space: MultiDiscrete([3, 2, 3, 3, 2]) = (P, B_bid, B_reveal, C, PTC).
    13 primitives total, 108 joint tuples.

    Reward: ΔU_coalition per step (same definition as legacy CoalitionEnv).

    Observation: 21-dim float vector — see ``_build_obs``. All features
    are computed from honest-validator-visible state (Store public fields
    + peek_slot_roles) so the policy never reads hook-only state. See
    plan §12 obs static audit gate.

    Sampling is **autoregressive** via ``mask_for_head``: the agent's
    runtime (PPO.act) queries this callback in SAMPLE_ORDER_L. The
    B_reveal mask depends on B_bid's choice (whether byz will actually
    win the auction under the chosen bid magnitude), so it's sampled
    last. Plan §2.
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: Tier2Config):
        self.cfg_template = cfg
        self.byz_v = frozenset(cfg.byzantine_validators)
        self.byz_b = frozenset(cfg.byzantine_builders)
        # Mark-to-settlement accounting (default ON). Counts pending payments
        # already at quorum as committed, removing the episode-truncation
        # censoring artifact. Set False to reproduce the pre-fix (settled-only)
        # utility for the artifact A/B comparison.
        self.mark_to_settlement = True
        if _HAS_GYM:
            from gymnasium.spaces import MultiDiscrete
            self.action_space = MultiDiscrete(list(HEAD_SIZES_L))
            self.observation_space = spaces.Box(
                low=-1.0, high=1.0, shape=(OBS_DIM_L,), dtype=np.float32)
        # Default action: all-HONEST 5-tuple.
        self._pending_action: tuple[int, ...] = (
            P_HONEST_PROPOSE, B_BID_NORMAL, B_REVEAL_HONEST,
            C_HONEST_ATTEST, PTC_HONEST,
        )
        self._last_intended: tuple[int, ...] = self._pending_action
        self._last_executed: tuple[int, ...] = self._pending_action
        self.env: Tier2Environment | None = None
        self._u_last: int = 0
        self._u_total: int = 0
        # Diagnostics propagated to caller via info dict
        self._last_was_late_reveal: bool = False
        # Curriculum gate: per-head boolean list (length n_heads); each
        # entry corresponds to whether the head's *non-HONEST* primitives
        # are unlocked yet. Default = all-unlocked (no curriculum).
        # Training loop (exp13) sets this to restrict action space in
        # early rollouts and progressively widen. See plan §risks.
        self._curriculum_unlocked: list[bool] = [True] * len(HEAD_SIZES_L)

    def set_curriculum_unlocked(self, unlocked_per_head: list[bool]) -> None:
        """Phase L-α curriculum control. ``unlocked_per_head[i] = False``
        forces head ``i``'s mask to allow ONLY the HONEST primitive
        (index 0 of each head), regardless of state-based mask rules.
        ``True`` defers to the normal ``mask_for_head`` rules.

        This is a *training-time* tool to seed the policy on the
        HONEST baseline value function before opening up the larger
        action space. Per plan §risks ("5-head + autoregressive PPO 比
        单头不稳 — exp11 超参 baseline; curriculum: 前 5 rollout 把所
        有非 HONEST primitive 的 mask 强制 False").
        """
        if len(unlocked_per_head) != len(HEAD_SIZES_L):
            raise ValueError(
                f"unlocked_per_head must be length {len(HEAD_SIZES_L)}"
            )
        self._curriculum_unlocked = [bool(b) for b in unlocked_per_head]

    # -- Gym API ----------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if _HAS_GYM:
            super().reset(seed=seed)
        cfg = replace(
            self.cfg_template,
            seed=(seed or 0).to_bytes(32, "little"),
            preferred_builder_hook=self._preferred_builder_hook,
            forced_builder_action_hook=self._forced_builder_hook,
            proposer_parent_hook=self._proposer_parent_hook,
            bid_value_override_hook=self._bid_value_override_hook,
            late_reveal_hook=self._late_reveal_hook,
            ptc_vote_hook=self._ptc_vote_hook,
        )
        self.env = Tier2Environment(cfg, _CoalitionAdversaryL(self))
        self.env.reset()
        self.env.begin_episode()
        self._u_last = 0
        self._u_total = 0
        self._last_was_late_reveal = False
        return self._build_obs(), {}

    def step(self, action):
        """``action`` is a 5-tuple/array (P, B_bid, B_reveal, C, PTC)."""
        assert self.env is not None, "reset() first"
        if self.env.episode_done():
            raise RuntimeError("step() after episode_done; call reset()")
        a = tuple(int(x) for x in action)
        if len(a) != len(HEAD_SIZES_L):
            raise ValueError(f"action must be length {len(HEAD_SIZES_L)}, got {len(a)}")
        self._pending_action = a
        self._last_intended = a
        self._last_executed = a
        self._last_was_late_reveal = (a[H_B_REVEAL] == B_REVEAL_LATE)

        event, _stat = self.env.advance_one_slot()

        u_now = self._coalition_utility()
        reward = float(u_now - self._u_last)
        self._u_last = u_now
        self._u_total = u_now

        terminated = self.env.episode_done()
        obs = (self._build_obs() if not terminated
               else np.zeros(OBS_DIM_L, dtype=np.float32))
        info = {
            "slot": event.slot,
            "u_coalition_running": u_now,
            "byz_proposer_this_slot": event.proposer_index in self.byz_v,
            "head_status": event.head_status_at_end.name,
            "action": a,
            "builder_action": event.builder_action,
            "proposer_action": event.proposer_action,
        }
        return obs, reward, terminated, False, info

    # -- Autoregressive mask callback (consumed by PPO.act) ---------------
    def mask_for_head(self, head_idx: int,
                      prev_actions: list[int | None]) -> list[bool]:
        """Return the legal-primitive mask for ``head_idx`` given any
        previously sampled head actions in ``prev_actions``.

        Called once per head in PPO.act per env step. Plan §2 mask rules.

        Curriculum gate: if ``self._curriculum_unlocked[head_idx]`` is
        False, the returned mask is ``[True, False, False, ...]`` —
        forcing HONEST regardless of state. See ``set_curriculum_unlocked``.
        """
        # Curriculum gate fires FIRST — short-circuit before any state read.
        # Justification: when locked, no state-dependent variation is
        # needed (HONEST is the only legal primitive). This also avoids
        # spending peek_slot_roles compute during locked-head sampling.
        if not self._curriculum_unlocked[head_idx]:
            return [True] + [False] * (HEAD_SIZES_L[head_idx] - 1)
        roles = self._peek_next_slot_roles()
        if roles is None:
            # Past episode end → all-HONEST trivially legal
            return [True] + [False] * (HEAD_SIZES_L[head_idx] - 1)

        proposer_byz = roles["proposer"] in self.byz_v
        committee = roles["committee"]
        ptc = roles["ptc"]
        n_byz_committee = sum(1 for vi in committee if vi in self.byz_v)
        n_byz_ptc = sum(1 for vi in ptc if vi in self.byz_v)
        any_byz_builder_has_cover = self._any_byz_builder_has_cover()

        if head_idx == H_P:
            # BUILD_ON_NONHEAD additionally needs a non-head candidate; we
            # do NOT gate the mask on candidate existence (avoid mask
            # leaking store ancestry); hook fallback to HONEST if empty.
            # Plan §2 final paragraph.
            return [True, proposer_byz, proposer_byz]
        if head_idx == H_B_BID:
            return [True, any_byz_builder_has_cover]
        if head_idx == H_C:
            return [True, n_byz_committee > 0, n_byz_committee > 0]
        if head_idx == H_PTC:
            return [True, n_byz_ptc > 0]
        if head_idx == H_B_REVEAL:
            # Conditional on (P, B_bid). Both heads come before B_reveal in
            # SAMPLE_ORDER_L = [P, B_bid, C, PTC, B_reveal].
            #
            # Bug fix (2026-05-25): when P=FORCE_PICK_BYZ_BUILDER and the
            # proposer is byzantine and a byz builder exists, the
            # preferred_builder_hook forces byz_b to win regardless of the
            # auction outcome — so WITHHOLD/LATE_REVEAL are meaningful even
            # if the auction itself wouldn't pick byz_b under the chosen
            # B_bid. Without this branch the mask would gate FORCE_PICK +
            # WITHHOLD/LATE_REVEAL combinations out of the agent's reach.
            p_choice = prev_actions[H_P]
            force_pick_active = (
                p_choice == P_FORCE_PICK_BYZ_BUILDER
                and proposer_byz
                and bool(self.byz_b)
            )
            if force_pick_active:
                byz_wins = True
            else:
                b_bid_choice = prev_actions[H_B_BID]
                byz_wins = self._predict_byz_wins(
                    b_bid_choice if b_bid_choice is not None else B_BID_NORMAL
                )
            return [True, byz_wins, byz_wins]
        raise ValueError(f"unknown head_idx {head_idx}")

    def _peek_next_slot_roles(self) -> dict | None:
        env = self.env
        next_slot = env._cur_slot + 1
        if next_slot > self.cfg_template.num_slots:
            return None
        return env.peek_slot_roles(next_slot)

    def _any_byz_builder_has_cover(self) -> bool:
        store = self.env.store
        for b in self.env.bset.builders:
            if not b.is_byzantine:
                continue
            if bp.cover_bid_headroom(store, b.index) > 0:
                return True
        return False

    def _predict_byz_wins(self, b_bid_choice: int) -> bool:
        """Would a byz builder win the auction this slot under
        ``b_bid_choice``? Pure-read; uses the same compute_effective_bid_value
        env._run_one_slot will use, so it's exact, not approximate."""
        env = self.env
        next_slot = env._cur_slot + 1
        if next_slot > self.cfg_template.num_slots:
            return False
        store = env.store
        eff: dict[int, int] = {}
        for b in env.bset.builders:
            # Base each builder's intended bid on this slot's MEV value (flat
            # returns cfg.bid_value_gwei, so pre-MEV behaviour is unchanged).
            # INFLATE multiplies the builder's own MEV value, not a fixed
            # constant, so its "over-bid" semantics hold at any MEV magnitude.
            intended = env._mev_intended_bid(b.index, next_slot)
            if b.is_byzantine and b_bid_choice == B_BID_INFLATE_LOW:
                intended = INFLATE_MULTIPLIER * intended
            eff[b.index] = bp.compute_effective_bid_value(store, b.index, intended)
        auction = B.run_auction(env.bset, next_slot, eff)
        if auction is None:
            return False
        winner, _ = auction
        return winner.is_byzantine

    # -- Hook callbacks consumed by Tier2Environment ----------------------
    def _preferred_builder_hook(self, proposer_index: int, slot: int):
        if proposer_index not in self.byz_v or not self.byz_b:
            return None
        if self._pending_action[H_P] == P_FORCE_PICK_BYZ_BUILDER:
            return min(self.byz_b)
        return None

    def _forced_builder_hook(self, builder_index: int, slot: int):
        if builder_index not in self.byz_b:
            return None
        if self._pending_action[H_B_REVEAL] == B_REVEAL_WITHHOLD:
            return BuilderAction.WITHHOLD
        return None

    def _proposer_parent_hook(self, proposer_index: int, slot: int):
        if proposer_index not in self.byz_v:
            return None
        if self._pending_action[H_P] != P_BUILD_ON_NONHEAD:
            return None
        candidate = self._pick_nonhead_parent()
        return candidate  # may be None → env falls back to honest

    def _bid_value_override_hook(self, builder_index: int, slot: int):
        if builder_index not in self.byz_b:
            return None
        if self._pending_action[H_B_BID] != B_BID_INFLATE_LOW:
            return None
        # Over-bid relative to THIS slot's MEV value (flat -> 5x cfg.bid_value,
        # unchanged); a fixed constant would invert to an under-bid at high MEV.
        return INFLATE_MULTIPLIER * self.env._mev_intended_bid(builder_index, slot)

    def _late_reveal_hook(self, builder_index: int, slot: int) -> bool:
        if builder_index not in self.byz_b:
            return False
        return self._pending_action[H_B_REVEAL] == B_REVEAL_LATE

    def _ptc_vote_hook(self, ptc_idx: int, slot: int,
                       local_view_present: bool):
        if ptc_idx not in self.byz_v:
            return None
        if self._pending_action[H_PTC] == PTC_FRAUD_ABSENT:
            return False
        return None  # honest local view

    def _pick_nonhead_parent(self):
        """BUILD_ON_NONHEAD candidate set (plan §4 P.2): blocks within the
        last 2 slots, root != head, on-block validated, payload status
        non-PENDING. Highest fork-choice score wins (proxied by accumulated
        attestation weight). Returns (parent_root, parent_status) or None.
        """
        env = self.env
        store = env.store
        current_slot = env._cur_slot + 1
        head_root = F.get_head(store).root
        candidates = []
        for root, block in store.blocks.items():
            if root == head_root:
                continue
            if block.slot < current_slot - 2:
                continue
            # Payload status: FULL iff a child block could commit to its
            # payload (root in store.payloads); else EMPTY. PENDING shouldn't
            # apply to blocks already in store.blocks unless mid-slot.
            status = (PayloadStatus.FULL if root in store.payloads
                      else PayloadStatus.EMPTY)
            candidates.append((root, status, block.slot))
        if not candidates:
            return None
        # Pick the one whose block.slot is highest (most recent)
        candidates.sort(key=lambda t: -t[2])
        root, status, _ = candidates[0]
        return (root, status)

    # -- Utility (same definition as legacy CoalitionEnv) -----------------
    def _coalition_utility(self) -> int:
        store = self.env.store
        proposer_credits = sum(
            store.proposer_credits.get(i, 0) for i in self.byz_v
        )
        builder_debits = sum(
            store.builder_debits.get(i, 0) for i in self.byz_b
        )
        proposer_rewards = sum(
            store.proposer_rewards.get(i, 0) for i in self.byz_v
        )
        # Phase H+ — byz committee's own attestation reward/penalty. Abstaining
        # to suppress a builder-payment quorum forfeits the reward and incurs a
        # penalty, so this term is the modelled cost of the suppression attack.
        attester_net = sum(
            getattr(store, "attester_rewards", {}).get(i, 0) for i in self.byz_v
        )
        u = (proposer_credits - builder_debits + proposer_rewards
             + attester_net)
        if self.mark_to_settlement:
            m_credit, m_debit = _marked_pending_flows(
                store, self.byz_v, self.byz_b)
            u += m_credit - m_debit
        return int(u)

    # -- Observation (21 dim) ---------------------------------------------
    def _build_obs(self) -> np.ndarray:
        env = self.env
        next_slot = env._cur_slot + 1
        if next_slot > self.cfg_template.num_slots:
            return np.zeros(OBS_DIM_L, dtype=np.float32)

        roles = env.peek_slot_roles(next_slot)
        proposer = roles["proposer"]
        committee = roles["committee"]
        ptc = roles["ptc"]
        default_winner = roles["default_winner"]

        slot_in_epoch = next_slot % config.SLOTS_PER_EPOCH
        if env._slot_events:
            last_event = env._slot_events[-1]
            head_status = last_event.head_status_at_end
            last_head_full = (head_status == PayloadStatus.FULL)
            last_was_late_reveal = (
                last_event.builder_action == BuilderAction.REVEAL_LATE
            )
        else:
            head_status = PayloadStatus.PENDING
            last_head_full = False
            last_was_late_reveal = False

        com_size = max(1, len(committee))
        n_byz_committee = sum(1 for vi in committee if vi in self.byz_v)
        com_byz_frac = n_byz_committee / com_size

        ptc_size = max(1, len(ptc))
        n_byz_ptc = sum(1 for vi in ptc if vi in self.byz_v)
        ptc_byz_frac = n_byz_ptc / ptc_size

        n_pending = len(env.store.builder_pending_payments)
        pending_scaled = min(1.0, n_pending / 64.0)

        max_epochs = max(1, self.cfg_template.num_slots // config.SLOTS_PER_EPOCH)
        cur_just = (env.store.current_justified_checkpoint.epoch
                    if env.store.current_justified_checkpoint else 0)
        just_frac = cur_just / max_epochs

        # Bug fix (2026-05-25): clip BOTH bounds; pre-fix only clipped
        # above, so a typical -20M gwei coalition return scaled to -200 —
        # 200× outside the declared Box low=-1.0. This broke obs-audit
        # invariants and hurt training stability.
        u_scaled = float(np.clip(self._u_total / 100_000.0, -1.0, 1.0))
        is_byz_proposer = float(proposer in self.byz_v)
        is_byz_default_winner = float(default_winner.is_byzantine)

        # L-α additions
        is_byz_in_committee = 1.0 if n_byz_committee > 0 else 0.0
        is_byz_in_ptc = 1.0 if n_byz_ptc > 0 else 0.0
        # Cover-bid ratio: min over byz builders of (headroom / DEFAULT_STAKE).
        # If no byz builders, ratio = 0.
        if self.byz_b:
            byz_headrooms = [
                bp.cover_bid_headroom(env.store, b.index)
                for b in env.bset.builders if b.is_byzantine
            ]
            min_cover_ratio = (
                min(byz_headrooms) / B.DEFAULT_BUILDER_STAKE_GWEI
            )
        else:
            min_cover_ratio = 0.0

        has_nonhead = float(self._pick_nonhead_parent() is not None)

        obs = np.array([
            slot_in_epoch / config.SLOTS_PER_EPOCH,            # 0
            1.0 if head_status == PayloadStatus.EMPTY else 0.0,  # 1
            1.0 if head_status == PayloadStatus.FULL else 0.0,   # 2
            1.0 if head_status == PayloadStatus.PENDING else 0.0,  # 3
            u_scaled,                                            # 4
            just_frac,                                           # 5
            is_byz_proposer,                                     # 6
            is_byz_default_winner,                               # 7
            com_byz_frac,                                        # 8
            ptc_byz_frac,                                        # 9
            pending_scaled,                                      # 10
            len(self.byz_b) / max(1, self.cfg_template.num_builders),  # 11
            is_byz_in_committee,                                 # 12
            is_byz_in_ptc,                                       # 13
            com_byz_frac,                                        # 14 (n_byz_committee / size)
            ptc_byz_frac,                                        # 15 (n_byz_ptc / size)
            min_cover_ratio,                                     # 16
            1.0 if last_was_late_reveal else 0.0,                # 17
            1.0 if last_head_full else 0.0,                      # 18
            has_nonhead,                                         # 19
            # 20 — current-slot MEV signal: the winning effective bid the
            # auction would settle this slot. Without it the agent sees only
            # WHETHER a byz builder wins (idx 7), not HOW LARGE the payment is,
            # so it cannot time WITHHOLD to high-MEV slots. peek already
            # computes it; normalise at 5M gwei (flat bid -> constant 0.2).
            min(1.0, roles["default_effective_amount"] / 5_000_000.0),
        ], dtype=np.float32)
        return obs
