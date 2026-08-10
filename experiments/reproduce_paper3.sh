#!/usr/bin/env bash
# Reproduce the headline results of Paper 3 (Two Fixed-Seat Paths to Payload
# Orphaning in ePBS).
# Entry point: docs/PAPER3_ARTIFACT_README.md. Environment: pbs-gym (py3.10 + torch).
#
# Usage:  bash experiments/reproduce_paper3.sh
#
# By default this runs exactly the experiments the paper cites. Two opt-ins:
#   PANEL=1        rebuild the delivered-payload panel from the relays (~85 min
#                  of fetching); the panel it produces is bundled, so the figure
#                  steps below run without it.
#   PROVENANCE=1   also run the scripts retained for provenance that back no
#                  claim in the paper (exp82/83/84/85/87/93).
set -u
PY="${PY:-/opt/anaconda3/envs/pbs-gym/bin/python}"
cd "$(dirname "$0")/.." || exit 1

hr() { printf '\n========== %s ==========\n' "$1"; }

# ---------------------------------------------------------------- mechanism --

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

hr "Orphan persistence past the boost window (exp102)"
# Advances the ported fork choice past slot t+1 with proposer boost cleared and
# asks which payload status survives. Includes a rescue control (attesters vote
# against the manipulated head) and a follow-fraction sweep locating the 50%
# crossover. Backs the persistence claim of Section 2.1.
$PY experiments/exp102_orphan_persistence.py

# ---------------------------------------------------------------- economics --

hr "Operator supply (exp95: observed node-operator shares vs the pivotal set)"
# Uses measured Rated Network node-operator shares; the unattributed remainder is
# kept as one aggregate bucket rather than split into invented operators.
$PY experiments/exp95_empirical_operator_concentration.py

if [ "${PANEL:-0}" = "1" ]; then
  hr "Delivered-payload panel (exp103 fetch + exp104 merge/adjudicate/verify)"
  # Walks the eight relays' proposer_payload_delivered endpoints (~85 min,
  # resumable), then merges the shards, adjudicates reorganised heights against
  # the public bid archive, and runs the two cross-checks the paper reports.
  $PY experiments/exp103_fetch_delivered_payloads.py
  $PY experiments/exp104_build_delivered_panel.py
else
  hr "Delivered-payload panel: using the bundled copy (set PANEL=1 to rebuild)"
  ls -l experiments/data/block_value_panel_delivered_2026H1.parquet
fi

hr "Figure: block-value distribution (exp97)"
$PY experiments/exp97_block_value_distribution.py

hr "Figure: break-even frontier (exp98)"
$PY experiments/exp98_breakeven_frontier.py

hr "Figure: temporal stability (exp99)"
$PY experiments/exp99_temporal_stability.py

# ----------------------------------------------------------------- defences --

hr "Mitigations (exp86: re-coupling gates and accountable-vote penalties)"
$PY experiments/exp86_mitigation_eval.py

# ---------------------------------------------------------------- soundness --

hr "Mutation audit of the differential suite (exp101)"
# Tests the tests: mutates the ported predicate on exactly the properties the
# paper asserts (strict >, threshold location, silence semantics, short-circuit
# direction) and checks each mutant is killed. Reported as Table "mutation".
$PY experiments/exp101_mutation_audit.py

hr "Spec-faithfulness audit (differential tests)"
# The pyspec differential layer needs the generated gloas spec (see README).
# If absent, the pyspec tests self-skip; the sim-internal tests still run.
$PY -m pytest difftest/ -q

# --------------------------------------------------------------- provenance --

if [ "${PROVENANCE:-0}" = "1" ]; then
  hr "Provenance only: scripts that back no claim in the paper"
  # exp93 is superseded by exp95 (measured operator shares rather than invented
  # profiles); the manuscript reports no learned deployment policy, so nothing in
  # exp84/exp85 backs a claim; exp87 is the earlier single-relay n=4800 sample;
  # exp82/exp83 are toy-scale economics kept from the earlier draft.
  for s in exp82_ptc_bribe_economics exp83_ptc_theft_recapture \
           exp87_empirical_block_value exp93_ptc_operator_concentration \
           exp84_selective_attack_oracle exp85_selective_attack_neural; do
    hr "provenance: $s"
    $PY "experiments/$s.py"
  done
fi

hr "DONE — headline JSON and figures under experiments/figures/drl_risk_epbs/"
