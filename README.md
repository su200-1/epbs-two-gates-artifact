# Two Fixed-Seat Paths to Payload Orphaning in ePBS — reproducibility artifact

Artifact for *Two Fixed-Seat Paths to Payload Orphaning in ePBS*. The manuscript is
not distributed here while it is under review; the accepted version will be linked
once it is available.

The paper reports a specification-level incentive flaw in Ethereum's enshrined
Proposer-Builder Separation (ePBS, EIP-7732 / `gloas`): the builder-payment predicate
never reads the Payload Timeliness Committee's (PTC) vote, and the payload-timeliness
predicate never reads the payment. That decoupling admits two routes to discarding a
punctually revealed payload: the next proposer's build rule is overridden by 257
validly signed *not-present* votes (needing no cooperation from that proposer), while
fork choice's payload tiebreaker is defeated by merely *withholding* the corresponding
affirmation — 256 silent seats suffice — combined with control of the next slot's
proposer. Both counts are fixed, independent of total stake; the builder is still
debited for the block either way.

## Quick start

```bash
conda create -n epbs python=3.10 && conda activate epbs
pip install numpy pandas pyarrow pytest torch
bash experiments/reproduce_paper3.sh
```

The whole run takes about two minutes — the differential suite is a few seconds
once the pyspec is built — except `exp96` (fresh-panel rebuild from raw
relay/on-chain archives, optional, see below). Every headline result is written to
`experiments/figures/drl_risk_epbs/`.

## What maps to what

`docs/PAPER3_ARTIFACT_README.md` is the authoritative claim → command map: each paper
section is listed with the command that reproduces it and the numbers to expect.

| Paper element | Command |
|---|---|
| Exact two-gate boundary (Table 2) | `python experiments/exp92_exact_ptc_payment_boundary.py` |
| Scaled end-to-end sanity check (Table 3) | `python experiments/exp81_ptc_orphan_probe.py --seeds 8 --slots 64` |
| Operator supply vs. the pivotal set (Table 4) | `python experiments/exp95_empirical_operator_concentration.py` |
| Reservation-price sensitivity (Table 5) | `python experiments/exp88_gap_conditioned_block_value.py` |
| Joint economic sensitivity (Table 6) | `python experiments/exp94_empirical_joint_economics.py` |
| Mitigation stress test (Table 7) | `python experiments/exp86_mitigation_eval.py` |
| On-chain-verified panel build (817,528 blocks) | `python experiments/exp96_fresh_panel_build.py` (optional — output bundled) |
| Block-value distribution + reproducibility figure | `python experiments/exp97_block_value_distribution.py` |
| Break-even frontier figure | `python experiments/exp98_breakeven_frontier.py` |
| Temporal stability figure | `python experiments/exp99_temporal_stability.py` |
| Robustness figure (combined) | `python experiments/exp100_robustness_combined.py` |
| Mutation audit of the differential suite | `python experiments/exp101_mutation_audit.py` |
| Orphan persistence past the boost window | `python experiments/exp102_orphan_persistence.py` |
| Spec-faithfulness audit | `python -m pytest difftest/ -q` |

`exp84` and `exp85` are retained for provenance only; `exp93` is superseded by `exp95`
(measured node-operator shares rather than invented profiles). The manuscript reports
no learned deployment policy; none of `exp84`/`exp85`/`exp93` back a table or claim in
the paper.

## Layout

```
config.py                     protocol constants (PTC_SIZE, thresholds, ...)
epbs/                         spec-anchored simulator port
experiments/                  experiments cited by the paper
  data/                       bundled block-value panels (~25 MB, two vintages)
  figures/drl_risk_epbs/      generated results (JSON/CSV tracked; PNG/PDF regenerated)
  reproduce_paper3.sh         one-shot reproduction
difftest/                     differential tests for the load-bearing predicates
docs/PAPER3_ARTIFACT_README.md  claim → command map
docs/DISCLOSURE.md            responsible-disclosure record
```

## Differential tests against the executable specification

The conformance layer compares this port against the executable `pyspec` generated from
`consensus-specs` at the audited snapshot `015d72704fea321e95bb74631e34be17e1104e86`
(2026-07-02), the commit the paper quotes line numbers from. Build it once and point the
tests at it:

```bash
git clone https://github.com/ethereum/consensus-specs
cd consensus-specs && git checkout 015d72704fea321e95bb74631e34be17e1104e86
pip install uv && make _pyspec
export EPBS_PYSPEC_DIR=$PWD/tests/core/pyspec
```

Then `python -m pytest difftest/ -q` reports **293 passed, 0 skipped**. 206 of those
tests execute the generated spec; the rest are simulator-internal conformance checks.
The load-bearing predicates are run side by side with the spec rather than only
compared on constants:

- `payload_timeliness` and `payload_data_availability` over the 257/256 boundary grid,
  the absent-seat (`None`) cells, and an exhaustive 0–512 sweep of the not-present and
  withheld-affirmation counts
  (`difftest/test_primitives.py::TestTimelinessPredicatesAgainstPyspec`)
- `should_build_on_full`, `should_extend_payload` and `get_payload_status_tiebreaker`
  over both slot relations, both head states, and all four proposer-boost shapes
  (`difftest/test_forkchoice.py::TestPayloadGateConsumersAgainstPyspec`)
- the payment quorum and the settle/expire decision of
  `process_builder_pending_payments`
  (`difftest/test_builder_payments.py::TestSettlementAgainstPyspec`)

If `EPBS_PYSPEC_DIR` is unset and no sibling checkout is found, the pyspec layer
self-skips and the simulator-internal tests still run.

### Running against a newer consensus-specs

The suite also runs against `master`. Checked at `46d3d3513` (2026-08-05, 80 commits
past the audited snapshot): **291 passed, 2 failed**, and both failures are the same
genuine drift rather than a portability problem —

- `test_payload_due_bps_matches_spec` / `test_payload_due_ms_matches_spec`:
  `PAYLOAD_DUE_BPS` moved from `7500` to `5000` upstream (#5414, the builder's reveal
  deadline moved from 9 s to 6 s into the slot). `config.py` is pinned to the audited
  snapshot, so these assertions are the drift detector doing its job. Nothing the paper
  claims depends on this constant: the PTC attestation deadline (`7500`) and the regular
  attestation deadline (`2500`) are unchanged.

Everything else passes, including all 160 predicate-level differentials and the 257/256
boundary. The suite is tolerant of the upstream renames that landed in the same window
(`uint*` → `Uint*` #5469, `boolean` → `Boolean` #5466) and of `should_build_on_full`
gaining an explicit `slot` parameter (#5497), so no edit is needed to try a newer
snapshot.

## Data provenance

Two block-value panels are bundled under `experiments/data/`, so nothing here needs an
external download:

- `block_value_panel_fresh_2026H1.parquet` (817,528 blocks, 2026-02-21–2026-07-10):
  the calibration panel for the manuscript's main text (Figures~valuedist/breakeven/
  robust). Built by `exp96` from eight relays' top-bid archives via RelayScan,
  cross-referenced against on-chain coinbases from XBlock-ETH — a block's bid counts
  only if `block_fee_recipient` matches the on-chain coinbase, i.e. only bids that
  actually landed.
- `block_value_panel_24000000_24499999_min.parquet` (493,330 blocks, block heights
  24,000,000–24,499,999): the earlier eight-relay sample, still used by `exp88`'s
  reservation-price sensitivity table.

Both derive from the public `builder_blocks_received` relay archives via RelayScan,
with on-chain block fields from XBlock-ETH (Zheng, Zheng, Wu and Dai,
*IEEE Open Journal of the Computer Society* 1 (2020) 95–106,
[doi:10.1109/OJCS.2020.2990458](https://doi.org/10.1109/OJCS.2020.2990458)).

## Scope

The artifact establishes a protocol-level capability. `exp102` establishes, in
simulation and conditional on a slot-\(t{+}1\) attester majority, that the orphaned
branch stays canonical past the boost window rather than being rescued; it does not
establish that PTC seats are cheap, that a majority can be recruited inside the
lookahead window, or that orphaned value can be reconstructed profitably — reservation
prices and the re-capture ratio are swept as exogenous parameters rather than measured.
See the paper's Section "Limitations and Scope".

## License

MIT — see `LICENSE`.
