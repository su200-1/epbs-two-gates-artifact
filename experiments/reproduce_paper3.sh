#!/usr/bin/env bash
# Reproduce the headline results of Paper 3 (Two Fixed-Seat Paths to Payload
# Orphaning in ePBS).
# Entry point: docs/PAPER3_ARTIFACT_README.md. Environment: pbs-gym (py3.10 + torch).
# Usage: bash experiments/reproduce_paper3.sh
set -u
PY="${PY:-/opt/anaconda3/envs/pbs-gym/bin/python}"
cd "$(dirname "$0")/.." || exit 1

hr() { printf '\n========== %s ==========\n' "$1"; }

hr "Exact two-gate boundary (exp92: 512-seat PTC 255/256/257/258 x 59/60/61% payment)"
# Deterministic executable-predicate matrix at the real committee size. This is the
# paper's primary mechanism evidence: EMPTY at 257 not-present seats while the
# independent payment predicate still settles at its 60% quorum.
$PY experiments/exp92_exact_ptc_payment_boundary.py

hr "Scaled decoupling sanity check (exp81: orphan while builder still pays)"
# Scaled 64-validator end-to-end probe. The paper reports only the honest and
# full-corruption endpoints; a scaled committee does not represent the exact
# 256/257 mainnet boundary, which exp92 covers.
$PY experiments/exp81_ptc_orphan_probe.py --seeds 8 --slots 64

hr "Economics (exp82: fixed 257-seat set, exogenous reservation price vs MEV)"
$PY experiments/exp82_ptc_bribe_economics.py

hr "Griefing (exp83 default: victim pays for empty block)"
$PY experiments/exp83_ptc_theft_recapture.py

hr "Conditional re-capture (exp83 high-MEV variant)"
$PY experiments/exp83_ptc_theft_recapture.py --expected-profit-gwei 1000000000

hr "Operator supply (exp95: observed node-operator shares vs the pivotal set)"
# Uses measured Rated Network node-operator shares; the unattributed remainder is
# kept as one aggregate bucket rather than split into invented operators.
$PY experiments/exp95_empirical_operator_concentration.py

hr "Operator concentration (exp93: SUPERSEDED by exp95, retained for provenance)"
# exp93's invented profiles are far more concentrated than mainnet and are no
# longer cited by the paper; kept only so the earlier analysis stays reproducible.
$PY experiments/exp93_ptc_operator_concentration.py

hr "Fresh on-chain-verified panel (exp96: 2026 H1 RelayScan + XBlock coinbase match)"
# Rebuilds an 817k-block value panel from the 2026-02..07 RelayScan top-bid
# archives, keeping per block only the bid whose block_fee_recipient matches the
# on-chain coinbase (XBlock minerAddress) -- i.e. the block that actually landed.
# This replaces the earlier >=100 ETH spam cap with a delivery test and yields
# the on-chain builder->proposer payment. Optional: the output is already bundled
# at experiments/data/block_value_panel_fresh_2026H1.parquet, so exp97-exp100 run
# without this step; re-running it from scratch needs the raw relay/on-chain
# archives on local disk (see the script's docstring), and self-skips if absent.
$PY experiments/exp96_fresh_panel_build.py || echo "(skip: raw datasets not present)"

hr "Figure: block-value distribution + reproducibility (exp97)"
$PY experiments/exp97_block_value_distribution.py || echo "(skip: fresh panel not built)"

hr "Figure: break-even frontier (exp98)"
$PY experiments/exp98_breakeven_frontier.py || echo "(skip: fresh panel not built)"

hr "Figure: temporal stability (exp99)"
$PY experiments/exp99_temporal_stability.py || echo "(skip: fresh panel not built)"

hr "Figure: robustness, combined (exp100)"
$PY experiments/exp100_robustness_combined.py || echo "(skip: fresh panel not built)"

hr "Mitigations (exp86: re-coupling gates and accountable-vote penalties)"
$PY experiments/exp86_mitigation_eval.py

hr "Economics panel (exp88: 493,330-block, 8-relay break-even; self-contained)"
# exp88 reads the compact 4-column extract bundled at
# experiments/data/block_value_panel_24000000_24499999_min.parquet, so this
# reproduces self-contained. Override with --panel for the full external panel.
$PY experiments/exp88_gap_conditioned_block_value.py

hr "Joint economic sensitivity (exp94: owned stake x reservation price x rho)"
# Crosses attacker-owned seat share, per-seat price, and re-capture ratio against
# the same bundled panel. Ex-post coverage, not realized attack profit.
$PY experiments/exp94_empirical_joint_economics.py

hr "Feasibility (exp87: superseded single-relay n=4800 sample; retained for provenance)"
# exp87 is the earlier, smaller single-relay sample that exp88's 493k panel replaced;
# the paper's economics table uses exp88, not exp87. Kept for historical comparison.
$PY experiments/exp87_empirical_block_value.py

hr "Superseded deployment probes (exp84/exp85: retained for provenance only)"
# The manuscript no longer reports a learned-deployment result; Section
# "Deployment Requirements and Evidentiary Boundary" states the preconditions
# instead. These two scripts back no table or claim in the paper and are kept
# only so the earlier selective-attack exploration stays reproducible.
$PY experiments/exp84_selective_attack_oracle.py
$PY experiments/exp85_selective_attack_neural.py

hr "Mutation audit of the differential suite (exp101)"
# Tests the tests: mutates the ported predicate on exactly the properties the
# paper asserts (strict >, threshold location, silence semantics, short-circuit
# direction) and checks each mutant is killed. Reported as Table "mutation".
$PY experiments/exp101_mutation_audit.py

hr "Orphan persistence past the boost window (exp102)"
# Advances the ported fork choice past slot t+1 with proposer boost cleared and
# asks which payload status survives. Includes a rescue control (attesters vote
# against the manipulated head) and a follow-fraction sweep locating the 50%
# crossover. Backs the persistence claim in Section "Limitations and Scope".
$PY experiments/exp102_orphan_persistence.py

hr "Spec-faithfulness audit (differential tests)"
# The pyspec differential layer needs the generated gloas spec (see README).
# If absent, TestAgainstPyspec self-skips; the sim-internal tests still run.
$PY -m pytest difftest/ -q

hr "DONE — headline JSON under experiments/figures/drl_risk_epbs/"
