"""Paper 2 metadata for stationary joint-action templates.

The classifier is intentionally conservative: it describes how a template may
be used in paper-facing claims under the simulator's documented abstractions.
It does not change action execution or assert that an attack is profitable.
"""
from __future__ import annotations

from dataclasses import dataclass

from epbs.rl.coalition_env import (
    B_BID_INFLATE_LOW,
    B_REVEAL_LATE,
    B_REVEAL_WITHHOLD,
    C_VOTE_EMPTY,
    C_WITHHOLD_VOTE,
    HEAD_SIZES_L,
    P_BUILD_ON_NONHEAD,
    P_FORCE_PICK_BYZ_BUILDER,
    PTC_FRAUD_ABSENT,
)

SPEC_FEASIBLE = "spec-feasible-upper-bound"
SIM_SENSITIVE = "sim-sensitive"
SIM_ARTIFACT = "sim-artifact"
MODEL_LIMITED = "model-limited"

HEAD_NAMES = ("P", "B_bid", "B_reveal", "C", "PTC")
PRIMITIVE_NAMES = (
    ("HONEST_PROPOSE", "FORCE_PICK_BYZ_BUILDER", "BUILD_ON_NONHEAD"),
    ("NORMAL_BID", "INFLATE_LOW"),
    ("HONEST_BUILD", "WITHHOLD", "LATE_REVEAL"),
    ("HONEST_ATTEST", "WITHHOLD_VOTE", "VOTE_EMPTY"),
    ("HONEST_PTC", "FRAUD_ABSENT"),
)

_SEVERITY = {
    SPEC_FEASIBLE: 0,
    SIM_SENSITIVE: 1,
    MODEL_LIMITED: 2,
    SIM_ARTIFACT: 3,
}


@dataclass(frozen=True)
class TemplateClassification:
    label: str
    caveats: tuple[str, ...]

    def as_dict(self) -> dict:
        return {"label": self.label, "caveats": list(self.caveats)}


def validate_template(action: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    """Return ``action`` as a validated 5-tuple."""
    action = tuple(int(x) for x in action)
    if len(action) != len(HEAD_SIZES_L):
        raise ValueError(f"action must contain {len(HEAD_SIZES_L)} heads")
    for head, (choice, n) in enumerate(zip(action, HEAD_SIZES_L)):
        if not 0 <= choice < n:
            raise ValueError(f"action[{head}]={choice} outside range(0, {n})")
    return action


def action_names(action: tuple[int, ...] | list[int]) -> list[str]:
    action = validate_template(action)
    return [PRIMITIVE_NAMES[h][choice] for h, choice in enumerate(action)]


def classify_template(
    action: tuple[int, ...] | list[int],
) -> TemplateClassification:
    """Classify a stationary action template for paper-facing use."""
    p, b_bid, b_reveal, c, ptc = validate_template(action)
    labels = [SPEC_FEASIBLE]
    caveats: list[str] = []

    if p == P_FORCE_PICK_BYZ_BUILDER:
        caveats.append("adversarial proposer bid selection")
    if p == P_BUILD_ON_NONHEAD:
        labels.append(MODEL_LIMITED)
        caveats.append("BUILD_ON_NONHEAD degenerates to honest head-build "
                       "(candidate set empty at production: only head ancestors "
                       "within 2 slots); zero reorgs, see exp14_reorg_probe")
    if b_bid == B_BID_INFLATE_LOW:
        caveats.append("cover-bid rejection uses parameterised builder balances")
    if b_reveal == B_REVEAL_WITHHOLD:
        caveats.append("payload withholding does not suppress same-slot payment quorum")
    if b_reveal == B_REVEAL_LATE:
        labels.append(SIM_SENSITIVE)
        caveats.append("fixed late-arrival ordering after the 75% PTC deadline")
    if c in (C_WITHHOLD_VOTE, C_VOTE_EMPTY):
        caveats.append("committee-side spec-valid non-slashable deviation")
    if ptc == PTC_FRAUD_ABSENT:
        caveats.append("PTC absent vote affects fork choice, not settlement quorum")

    label = max(labels, key=lambda value: _SEVERITY[value])
    return TemplateClassification(label=label, caveats=tuple(dict.fromkeys(caveats)))


def is_spec_feasible(action: tuple[int, ...] | list[int]) -> bool:
    """Whether the template belongs in the spec-feasible-only headline view."""
    return classify_template(action).label == SPEC_FEASIBLE
