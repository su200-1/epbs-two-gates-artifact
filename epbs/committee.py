"""Validator-set committees — ported from `consensus-specs/specs/phase0` and
`specs/gloas`.

This is part of the Tier 2a fork-choice layer. The shuffle (``compute_shuffled_index``,
``compute_committee``) is ported verbatim from `phase0/beacon-chain.md` and is
**directly diff-tested** against the pyspec — these are pure functions of
``(indices, seed, index, count)``.

``get_beacon_committee`` and ``get_ptc`` in pyspec read from cached BeaconState
fields (``state.proposer_lookahead``, ``state.ptc_window``) that are populated
at epoch-boundary processing. To avoid building a full BeaconState/epoch-
processor (out of scope per MODEL_SCOPE.md), Tier 2a exposes simplified helpers
that take ``active_indices + seed`` directly and call the same underlying
shuffle. By construction these return the same indices that the pyspec cache
would, provided callers pass the spec's ``(epoch, domain_type)``-derived seed.

Spec sources:
    phase0/beacon-chain.md::compute_shuffled_permutation, compute_shuffled_index,
        compute_committee, compute_proposer_index, get_active_validator_indices
    gloas/beacon-chain.md::get_ptc (simplified: bypasses state.ptc_window cache)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from hashlib import sha256

import config

# --- shuffle constants (pinned to consensus-specs) -------------------------
# Diff-tested against pyspec in difftest/test_committee.py.
SHUFFLE_ROUND_COUNT: int = 10
MAX_EFFECTIVE_BALANCE_GWEI: int = 32_000_000_000  # 32 ETH (phase0 default)
# Gloas uses electra's compounding-validator cap as the rejection-sampling
# normaliser. Pinned to electra mainnet/minimal presets (same value).
MAX_EFFECTIVE_BALANCE_ELECTRA: int = 2_048_000_000_000  # 2048 ETH

# Domain types (phase0/beacon-chain.md). Used by `derive_epoch_seed` so
# different epochs and different roles (attester, proposer, PTC) get
# independent shuffles even from a single base seed. Pre-Phase-I-2 sim
# bug: get_beacon_committee was called with the raw base seed regardless
# of epoch, so committees at slot K and slot K+SLOTS_PER_EPOCH were
# identical (verified empirically before fix).
DOMAIN_BEACON_ATTESTER: bytes = bytes([0x01, 0x00, 0x00, 0x00])
DOMAIN_BEACON_PROPOSER: bytes = bytes([0x00, 0x00, 0x00, 0x00])
DOMAIN_PTC_ATTESTER: bytes = bytes([0x0C, 0x00, 0x00, 0x00])  # gloas-specific


# --------------------------------------------------------------------------
# Validator type — slimmed (no pubkey/withdrawal etc.). Only fields the
# fork-choice / committee / attestation layer reads.
# --------------------------------------------------------------------------
@dataclass
class Validator:
    """Minimal validator record. Spec analogue: `class Validator`."""

    index: int
    effective_balance: int = MAX_EFFECTIVE_BALANCE_GWEI
    activation_epoch: int = 0
    exit_epoch: int = (1 << 64) - 1
    slashed: bool = False
    is_byzantine: bool = False        # Tier 2a sim flag — not in spec


@dataclass
class ValidatorSet:
    """Pool of validators with helper indexers. Tier 2a-specific container."""

    validators: list[Validator] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.validators)

    def byzantine_indices(self) -> list[int]:
        return [v.index for v in self.validators if v.is_byzantine]

    def effective_balance(self, idx: int) -> int:
        return self.validators[idx].effective_balance


# --------------------------------------------------------------------------
# Shuffle  (phase0/beacon-chain.md)
# --------------------------------------------------------------------------
def _hash(b: bytes) -> bytes:
    """Spec `hash` — SHA-256, 32-byte output."""
    return sha256(b).digest()


@lru_cache(maxsize=128)
def compute_shuffled_permutation(index_count: int, seed: bytes) -> tuple[int, ...]:
    """Port of phase0 `compute_shuffled_permutation` (swap-or-not shuffle).

    Swap-or-not (Khovratovich-Bonneau): permutes 0..index_count-1 using
    ``SHUFFLE_ROUND_COUNT`` rounds, each round using a hash-derived pivot.

    Returns a tuple (immutable) so the lru_cache cannot be poisoned by a
    caller that mutates the result. All callers do read-only indexing.
    """
    indices = list(range(index_count))
    for current_round in range(SHUFFLE_ROUND_COUNT):
        round_bytes = current_round.to_bytes(1, "little")
        pivot = int.from_bytes(_hash(seed + round_bytes)[0:8], "little") % index_count
        source_by_bucket: dict[int, bytes] = {}
        for i in range(index_count):
            flip = (pivot + index_count - indices[i]) % index_count
            position = max(indices[i], flip)
            position_bucket = position // 256
            if position_bucket not in source_by_bucket:
                source_by_bucket[position_bucket] = _hash(
                    seed + round_bytes + position_bucket.to_bytes(4, "little")
                )
            source = source_by_bucket[position_bucket]
            byte_val = source[(position % 256) // 8]
            bit = (byte_val >> (position % 8)) % 2
            indices[i] = flip if bit else indices[i]
    return tuple(indices)


def compute_shuffled_index(index: int, index_count: int, seed: bytes) -> int:
    """Port of phase0 `compute_shuffled_index`."""
    assert 0 <= index < index_count
    return compute_shuffled_permutation(index_count, seed)[index]


def compute_committee(
    indices: list[int], seed: bytes, index: int, count: int
) -> list[int]:
    """Port of phase0 `compute_committee`.

    Returns the ``index``-th of ``count`` committees drawn from ``indices``.
    """
    n = len(indices)
    start = (n * index) // count
    end = (n * (index + 1)) // count
    return [indices[compute_shuffled_index(i, n, seed)] for i in range(start, end)]


def compute_proposer_index(
    vset: ValidatorSet, indices: list[int], seed: bytes
) -> int:
    """Port of phase0 `compute_proposer_index` (effective-balance-weighted).

    Note: Tier 2a uses uniform effective_balance=32 ETH by default, so this
    reduces to ``indices[compute_shuffled_index(0, ..., seed)]`` in expectation.
    """
    assert len(indices) > 0
    MAX_RANDOM_BYTE = 2**8 - 1
    i = 0
    total = len(indices)
    while True:
        candidate_index = indices[compute_shuffled_index(i % total, total, seed)]
        random_byte = _hash(seed + (i // 32).to_bytes(8, "little"))[i % 32]
        effective_balance = vset.effective_balance(candidate_index)
        if effective_balance * MAX_RANDOM_BYTE >= MAX_EFFECTIVE_BALANCE_GWEI * random_byte:
            return candidate_index
        i += 1


# --------------------------------------------------------------------------
# Active set  (phase0/beacon-chain.md::get_active_validator_indices)
# --------------------------------------------------------------------------
def get_active_validator_indices(vset: ValidatorSet, epoch: int) -> list[int]:
    """Port of phase0 `get_active_validator_indices`.

    A validator is active in ``epoch`` iff activation_epoch <= epoch < exit_epoch.
    """
    return [
        v.index
        for v in vset.validators
        if v.activation_epoch <= epoch < v.exit_epoch
    ]


# --------------------------------------------------------------------------
# Simplified committee accessors (avoid building a full BeaconState)
# --------------------------------------------------------------------------
def derive_epoch_seed(base_seed: bytes, epoch: int, domain: bytes) -> bytes:
    """Compose a per-(epoch, domain) seed from the experiment's base seed.

    Spec analogue: ``get_seed(state, epoch, domain_type)`` which hashes the
    epoch's RANDAO mix together with the domain type. Sim has no RANDAO,
    so we substitute the base seed — the result still makes shuffles for
    different epochs and different roles statistically independent, which
    is what multi-epoch tail-probability experiments require.

    Pre-Phase-I-2 bug: ``get_beacon_committee`` consumed the raw base seed
    without mixing in epoch, so slot=1 and slot=33 produced identical
    committees (same ``slot % SLOTS_PER_EPOCH``). The fix moves seed
    derivation into the committee accessors.
    """
    return _hash(base_seed + epoch.to_bytes(8, "little") + domain)


def get_beacon_committee(
    active_indices: list[int],
    seed: bytes,
    slot: int,
    committee_index: int,
    *,
    slots_per_epoch: int = config.SLOTS_PER_EPOCH if hasattr(config, "SLOTS_PER_EPOCH") else 8,
    committees_per_slot: int = 1,
) -> list[int]:
    """Simplified `get_beacon_committee` — calls `compute_committee` directly.

    pyspec routes through ``state.proposer_lookahead`` / cache, ours computes
    the same committee from the shuffle. The ``seed`` argument is the
    experiment's base seed; internally we derive a per-epoch seed via
    `derive_epoch_seed(seed, epoch, DOMAIN_BEACON_ATTESTER)`, matching
    spec's ``get_seed(state, epoch, DOMAIN_BEACON_ATTESTER)``.

    ``committee_index`` is the per-slot committee number; the overall index
    into `compute_committee` is ``(slot % slots_per_epoch) * committees_per_slot
    + committee_index``, and total count is ``slots_per_epoch *
    committees_per_slot``.
    """
    slot_in_epoch = slot % slots_per_epoch
    overall_index = slot_in_epoch * committees_per_slot + committee_index
    overall_count = slots_per_epoch * committees_per_slot
    epoch = slot // slots_per_epoch
    epoch_seed = derive_epoch_seed(seed, epoch, DOMAIN_BEACON_ATTESTER)
    return compute_committee(active_indices, epoch_seed, overall_index, overall_count)


def compute_balance_weighted_selection(
    indices: list[int],
    seed: bytes,
    size: int,
    *,
    shuffle_indices: bool,
    effective_balance_of: callable = None,
) -> list[int]:
    """Port of gloas/beacon-chain.md `compute_balance_weighted_selection`.

    Spec accepts a ``state`` and pulls effective_balance off ``state.validators``;
    Tier 2 passes ``effective_balance_of(idx) -> gwei`` directly to avoid a
    BeaconState model. With uniform 32 ETH balances the rejection always
    accepts (32 ETH = 0x06f05b59d3b20000; threshold is sampled against
    MAX_EFFECTIVE_BALANCE_ELECTRA = 2048 ETH, so a 32 ETH validator is
    accepted with prob 1/64 — many rejections per accepted index, but
    correct distribution). Returns ``size`` indices (with possible duplicates).
    """
    MAX_RANDOM_VALUE = 2**16 - 1
    total = len(indices)
    assert total > 0
    eff_of = effective_balance_of or (lambda _idx: MAX_EFFECTIVE_BALANCE_GWEI)
    effective_balances = [eff_of(idx) for idx in indices]
    selected: list[int] = []
    i = 0
    random_bytes = b""
    while len(selected) < size:
        offset = (i % 16) * 2
        if offset == 0:
            random_bytes = _hash(seed + (i // 16).to_bytes(8, "little"))
        next_index = i % total
        if shuffle_indices:
            next_index = compute_shuffled_index(next_index, total, seed)
        weight = effective_balances[next_index] * MAX_RANDOM_VALUE
        random_value = int.from_bytes(random_bytes[offset:offset + 2], "little")
        threshold = MAX_EFFECTIVE_BALANCE_ELECTRA * random_value
        if weight >= threshold:
            selected.append(indices[next_index])
        i += 1
    return selected


def get_ptc(
    active_indices: list[int],
    seed: bytes,
    slot: int,
    *,
    slots_per_epoch: int = config.SLOTS_PER_EPOCH if hasattr(config, "SLOTS_PER_EPOCH") else 8,
    ptc_size: int = config.PTC_SIZE,
    committees_per_slot: int = 1,
    effective_balance_of: callable = None,
) -> list[int]:
    """Port of gloas/beacon-chain.md `compute_ptc` (state-free).

    Spec routes through an epoch-cached `state.ptc_window`; Tier 2 recomputes
    on demand using `compute_balance_weighted_selection` over the union of
    the slot's beacon committees. With possible duplicates and balance-
    weighted (per electra rejection sampling) — distribution matches spec.

    Per-epoch seed derived via `derive_epoch_seed(seed, epoch, DOMAIN_PTC_ATTESTER)`
    (post Phase I-2 — pre-fix used the raw base seed without epoch, so PTC
    repeated across epochs the same way beacon committees did).
    """
    epoch = slot // slots_per_epoch
    indices: list[int] = []
    for ci in range(committees_per_slot):
        indices.extend(get_beacon_committee(
            active_indices, seed, slot, ci,
            slots_per_epoch=slots_per_epoch,
            committees_per_slot=committees_per_slot,
        ))
    if not indices:
        return []
    # Spec: hash(get_seed(state, epoch, DOMAIN_PTC_ATTESTER) + uint_to_bytes(slot))
    epoch_seed = derive_epoch_seed(seed, epoch, DOMAIN_PTC_ATTESTER)
    slot_seed = _hash(epoch_seed + slot.to_bytes(8, "little"))
    return compute_balance_weighted_selection(
        indices, slot_seed, ptc_size,
        shuffle_indices=False,
        effective_balance_of=effective_balance_of,
    )


def get_beacon_proposer_index(
    vset: ValidatorSet, active_indices: list[int], seed: bytes, slot: int
) -> int:
    """Simplified `get_beacon_proposer_index` — bypasses ``state.proposer_lookahead``.

    Per spec the proposer is drawn slot-by-slot by `compute_proposer_index`
    with a slot-specific seed. Tier 2a mixes the slot into the seed directly.
    """
    slot_seed = _hash(seed + slot.to_bytes(8, "little"))
    return compute_proposer_index(vset, active_indices, slot_seed)


# --------------------------------------------------------------------------
# Helpers for sim setup
# --------------------------------------------------------------------------
def make_validator_set(
    n: int, *, byzantine_indices: set[int] | None = None
) -> ValidatorSet:
    """Build a uniform validator set of size ``n``.

    All validators have ``effective_balance = 32 ETH``; ``byzantine_indices``
    marks the Byzantine subset (Tier 2a flag; not in spec).
    """
    byzantine = byzantine_indices or set()
    return ValidatorSet(
        validators=[
            Validator(index=i, is_byzantine=(i in byzantine)) for i in range(n)
        ]
    )
