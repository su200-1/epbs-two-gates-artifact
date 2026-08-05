"""Lightweight ePBS (EIP-7732 / gloas) simulation.

Tier 1 (single-slot free-option environment):
    primitives  -- consensus primitives ported from consensus-specs/specs/gloas
    price       -- external (CEX) price process sigma_t
    economics   -- block value, net option value, exercise probability
    env         -- Gymnasium environment (single strategic Builder vs honest env)
    metrics     -- Metric I-V incentive/liveness flaw classifier
    strategy    -- SSF-style JSON strategy representation
    rl          -- PPO + LSTM strategy optimizer

Tier 2a (multi-slot ePBS fork choice + adversaries):
    committee   -- validator pool, shuffle, beacon committee, PTC
    attestation -- Attestation / PayloadAttestation + validate / update_latest_messages
    forkchoice  -- LMD-GHOST + payload-status fork choice, Store, handlers
    network     -- MessageBus (non-spec; invariant-tested)
    adversary   -- Honest / ExAnteReorg / PayloadLever / Staircase adversaries
    env_tier2   -- multi-slot chain simulation (FFG opt-in via Tier2Config)

Tier 2b (Casper FFG):
    ffg         -- weigh_/process_justification_and_finalization,
                   update_checkpoints (epoch-boundary FFG processing)
"""

__version__ = "0.3.0"
