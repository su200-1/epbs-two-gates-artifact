"""Gymnasium environment for the ePBS free-option problem (Tier 1).

One episode = one slot's commit->reveal window. The single agent is a strategic
**Builder**: after the bid is committed (a sunk cost), it watches the external
price move and chooses when/whether to reveal the execution payload. Withholding
exercises the put option — the slot goes empty (Metric V).

Honest Proposer + PTC are deterministic environment code (not agents):
- honest Proposer: includes the bid as committed;
- honest PTC: votes the payload timely iff revealed in time.

Architecture (reset/step/_get_state) mirrors BunnyFinder's `rl/env/` *pattern*
only; none of that (buggy) code is reused.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    _HAS_GYM = True
except ImportError:  # keep importable without gymnasium
    _HAS_GYM = False
    gym = object  # type: ignore

import config
from epbs import economics, primitives
from epbs.primitives import PayloadStatus

# actions
HOLD, REVEAL, WITHHOLD = 0, 1, 2
_ACTION_NAMES = {HOLD: "HOLD", REVEAL: "REVEAL", WITHHOLD: "WITHHOLD"}


@dataclass
class EpisodeOutcome:
    """Per-episode result, consumed by `epbs.metrics`."""

    builder_payoff: float
    proposer_payoff: float
    payload_status: PayloadStatus
    empty_block: bool
    r_used: float          # external return realised at the decision point
    pi_tau: float          # realised block profit at the decision point
    decision_unit: int     # time unit at which the builder acted
    action: int            # REVEAL or WITHHOLD


class EPBSFreeOptionEnv(gym.Env if _HAS_GYM else object):  # type: ignore[misc]
    """Single-Builder ePBS free-option environment.

    Args:
        cfg: simulation knobs (volatility, reveal-window units, ...).
        pi_0: block value at commit time (positive => profitable block).
        y: CEX-DEX position size carried in the block.
        execution_payment: unconditional payment bid->proposer (sunk cost).
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: config.SimConfig | None = None,
                 pi_0: float | None = None,
                 y: float | None = None,
                 execution_payment: float = 0.02):
        self.cfg = cfg or config.DEFAULT
        # default block: option-free value `mu`, overshooting position `y*`
        self.pi_0 = self.cfg.mu if pi_0 is None else float(pi_0)
        self.y = (economics.optimal_position(self.cfg.sigma, self.cfg.dex_liquidity,
                                             self.cfg.initial_cex_dex_gap)
                  if y is None else float(y))
        self.execution_payment = float(execution_payment)
        self.units = self.cfg.reveal_window_units

        # Builder payoff when it withholds the payload. Spec-style conditional
        # payment => the bid never settles, so withholding costs nothing (0).
        # Sunk-cost model => the bid is already paid, so withholding costs it.
        self.withhold_payoff = (0.0 if self.cfg.conditional_payment
                                else -self.execution_payment)

        from epbs.price import PriceProcess
        self._price = PriceProcess(self.cfg.sigma, self.cfg.drift, self.units)

        if _HAS_GYM:
            self.action_space = spaces.Discrete(3)
            # obs: [remaining_frac, pi_0, y, cum_return_now, pi_t, sigma]
            high = np.array([1.0, 1e3, 1e6, 10.0, 1e6, 10.0], dtype=np.float32)
            self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self._rng = np.random.default_rng(self.cfg.seed)
        self._path: np.ndarray | None = None
        self._t = 0
        self._done = True

    # -- gym API -----------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if _HAS_GYM:
            super().reset(seed=seed)       # seeds gymnasium's self.np_random
            self._rng = self.np_random
        elif seed is not None:
            self._rng = np.random.default_rng(seed)
        self._path = self._price.sample_path(self._rng)  # cum returns per unit
        self._t = 0
        self._done = False
        return self._obs(), {}

    def step(self, action: int):
        if self._done:
            raise RuntimeError("step() called on a finished episode; call reset()")

        if action == HOLD:
            self._t += 1
            if self._t >= self.units:
                # deadline reached without revealing => payload withheld
                return self._finish(WITHHOLD, self.units - 1)
            return self._obs(), 0.0, False, False, {}

        if action in (REVEAL, WITHHOLD):
            return self._finish(action, self._t)

        raise ValueError(f"invalid action {action!r}")

    # -- helpers -----------------------------------------------------------
    def _obs(self) -> np.ndarray:
        idx = min(self._t, self.units - 1)  # clamp: terminal obs after deadline
        r_now = float(self._path[idx])
        pi_t = economics.realized_profit(self.pi_0, self.y, r_now)
        remaining = max(0.0, 1.0 - idx / max(1, self.units - 1))
        return np.array([remaining, self.pi_0, self.y, r_now, pi_t, self.cfg.sigma],
                        dtype=np.float32)

    def _finish(self, action: int, unit: int):
        r_used = float(self._path[unit])
        pi_tau = economics.realized_profit(self.pi_0, self.y, r_used)

        if action == REVEAL:
            status = PayloadStatus.FULL
            builder_payoff = pi_tau          # honest delivery: eats the sign
            proposer_payoff = self.execution_payment
        else:  # WITHHOLD — exercise the option
            status = PayloadStatus.EMPTY
            builder_payoff = self.withhold_payoff
            # conditional payment => withheld bid never settles (proposer gets
            # nothing); sunk-cost model => the proposer keeps the bid.
            proposer_payoff = (0.0 if self.cfg.conditional_payment
                               else self.execution_payment)

        empty = status != PayloadStatus.FULL
        outcome = EpisodeOutcome(
            builder_payoff=builder_payoff,
            proposer_payoff=proposer_payoff,
            payload_status=status,
            empty_block=empty,
            r_used=r_used,
            pi_tau=pi_tau,
            decision_unit=unit,
            action=action,
        )
        self._done = True
        info = {"outcome": outcome, "action": _ACTION_NAMES[action]}
        return self._obs(), float(builder_payoff), True, False, info


# --------------------------------------------------------------------------
# Reference policies (honest baseline + optimal strategic) — not RL agents
# --------------------------------------------------------------------------
def honest_policy(env: EPBSFreeOptionEnv, obs: np.ndarray) -> int:
    """Honest Builder: always deliver — HOLD to the deadline, then REVEAL."""
    remaining = obs[0]
    return REVEAL if remaining <= 1e-6 else HOLD


def optimal_strategic_policy(env: EPBSFreeOptionEnv, obs: np.ndarray) -> int:
    """Optimal free-option Builder: at the deadline, reveal iff profitable.

    Reveal iff revealing (payoff ``pi_t``) beats withholding (payoff
    ``env.withhold_payoff``). Under conditional payment the threshold is 0;
    under the sunk-cost model it is ``-execution_payment`` (withhold less).
    This is the analytically optimal Tier 1 strategy and the target the RL
    agent should rediscover.
    """
    remaining, pi_t = obs[0], obs[4]
    if remaining > 1e-6:
        return HOLD
    return REVEAL if pi_t >= env.withhold_payoff else WITHHOLD
