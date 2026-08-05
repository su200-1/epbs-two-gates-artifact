"""SSF-style strategy representation for the ePBS free-option environment.

Mirrors BunnyFinder's Strategy Specification Format (JSON; see
`bf_workspace/code/attacker_v5/ai/prompt.txt` and `strategy/`): a strategy is a
list of per-slot ``actions`` keyed by *inject point*. The strategy generator
(SG, later) emits these; the executor turns one into a policy the env runs.

A single static SSF cannot express price-conditional behaviour — exactly as in
BunnyFinder, conditional logic emerges from the SG/RL producing many instances.
The ``conditionalWithhold`` action is provided as the one analytically-optimal
Tier 1 primitive.

ePBS Tier 1 inject point & actions
----------------------------------
``PayloadBeforeReveal``:
    ``reveal``                 -- deliver the payload at the deadline (honest)
    ``withhold``               -- exercise the put option (slot goes empty)
    ``delayWithDuration:x``    -- hold x time units, then reveal
    ``conditionalWithhold``    -- at the deadline, reveal iff block is profitable
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from epbs.env import HOLD, REVEAL, WITHHOLD

INJECT_PAYLOAD_BEFORE_REVEAL = "PayloadBeforeReveal"

VALID_ACTIONS = ("reveal", "withhold", "conditionalWithhold", "delayWithDuration")


@dataclass
class SlotStrategy:
    """Actions the strategic Builder applies in one slot."""

    slot: int
    actions: dict[str, str] = field(default_factory=dict)


@dataclass
class Strategy:
    """An SSF strategy instance — a list of per-slot strategies."""

    slots: list[SlotStrategy] = field(default_factory=list)

    # -- (de)serialisation -------------------------------------------------
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(
            {"slots": [{"slot": s.slot, "actions": s.actions} for s in self.slots]},
            indent=indent,
        )

    @classmethod
    def from_json(cls, text: str) -> "Strategy":
        data = json.loads(text)
        slots = [SlotStrategy(slot=int(s["slot"]), actions=dict(s.get("actions", {})))
                 for s in data.get("slots", [])]
        return cls(slots=slots)

    def slot_strategy(self, slot: int) -> SlotStrategy | None:
        for s in self.slots:
            if s.slot == slot:
                return s
        return None


def _parse_action(spec: str) -> tuple[str, int | None]:
    """Split ``name`` or ``name:param`` into (name, int param or None)."""
    if ":" in spec:
        name, param = spec.split(":", 1)
        return name, int(param)
    return spec, None


def build_policy(strategy: Strategy, slot: int = 0):
    """Turn one slot's SSF action into a policy callable for `EPBSFreeOptionEnv`.

    Returned signature matches the env's reference policies:
    ``policy(env, obs) -> action``.
    """
    ss = strategy.slot_strategy(slot)
    spec = (ss.actions.get(INJECT_PAYLOAD_BEFORE_REVEAL, "reveal")
            if ss is not None else "reveal")
    name, param = _parse_action(spec)
    if name not in VALID_ACTIONS:
        raise ValueError(f"unknown PayloadBeforeReveal action: {spec!r}")

    def policy(env, obs):
        remaining = obs[0]
        at_deadline = remaining <= 1e-6
        if name == "withhold":
            return WITHHOLD
        if name == "reveal":
            return REVEAL if at_deadline else HOLD
        if name == "conditionalWithhold":
            if not at_deadline:
                return HOLD
            # Align with env.optimal_strategic_policy: under sunk-cost payment
            # the withhold floor is -execution_payment, not 0.
            return REVEAL if obs[4] >= env.withhold_payoff else WITHHOLD
        if name == "delayWithDuration":
            # hold `param` units (within the window), then reveal
            units = env.units
            target_unit = min(param or 0, units - 1)
            return REVEAL if env._t >= target_unit else HOLD
        raise AssertionError(name)

    return policy


# -- convenience constructors ----------------------------------------------
def honest_strategy(slot: int = 0) -> Strategy:
    return Strategy([SlotStrategy(slot, {INJECT_PAYLOAD_BEFORE_REVEAL: "reveal"})])


def optimal_free_option_strategy(slot: int = 0) -> Strategy:
    return Strategy([SlotStrategy(slot,
                     {INJECT_PAYLOAD_BEFORE_REVEAL: "conditionalWithhold"})])
