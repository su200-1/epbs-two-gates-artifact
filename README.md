# Two Fixed-Seat Paths to Payload Orphaning in ePBS — reproducibility artifact

Artifact for *Two Fixed-Seat Paths to Payload Orphaning in ePBS*. The manuscript is
not distributed here while it is under review; the accepted version will be linked
once it is available.

The paper prices a specification-level design trade-off in Ethereum's enshrined
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
once the pyspec is built — except `exp103` (re-fetching the delivered-payload
shards, ~85 min, optional: the panel it feeds is bundled). Every headline result is
written to `experiments/figures/drl_risk_epbs/`.

## What maps to what

`docs/PAPER3_ARTIFACT_README.md` is the authoritative claim → command map: each paper
section is listed with the command that reproduces it and the numbers to expect.

| Paper element | Command |
|---|---|
| Exact two-gate boundary (Table `tab:boundary`) | `python experiments/exp92_exact_ptc_payment_boundary.py` |
| Scaled end-to-end sanity check (§2, orphan-and-still-pay) | `python experiments/exp81_ptc_orphan_probe.py --seeds 8 --slots 64` |
| Orphan persistence past the boost window (§2.5) | `python experiments/exp102_orphan_persistence.py` |
| Non-participation asymmetry (Table `tab:participation`) | — analytic; no experiment |
| Operator shares behind §3.1's 37.2 % / 7.45 % | `python experiments/exp95_empirical_operator_concentration.py` |
| Block-value distribution (Figure `fig:valuedist`) | `python experiments/exp97_block_value_distribution.py` |
| Break-even frontier (Figure `fig:breakeven`) | `python experiments/exp98_breakeven_frontier.py` |
| Temporal stability (Figure `fig:robust`) | `python experiments/exp99_temporal_stability.py` |
| Mitigation stress test (Table `tab:mit`) | `python experiments/exp86_mitigation_eval.py` |
| Mutation audit (Table `tab:mutation`) | `python experiments/exp101_mutation_audit.py` |
| Spec-faithfulness audit (§4) | `python -m pytest difftest/ -q` |
| Delivered-payload panel build (935,314 blocks) | `python experiments/exp103_fetch_delivered_payloads.py && python experiments/exp104_build_delivered_panel.py` (optional — panel bundled) |

Floats are keyed by LaTeX label rather than by number, because the numbering shifts as the
manuscript is revised.

Retained but **not cited by the paper**: `exp82`, `exp83`, `exp84`, `exp85`, `exp87`,
`exp93`. Kept for provenance — `exp93` is superseded by `exp95` (measured node-operator
shares rather than invented profiles), and the manuscript reports no learned deployment
policy, so nothing in `exp84`/`exp85` backs a claim.

**Retired**, marked as such in their own docstrings and no longer runnable because the
panels they read are no longer distributed: `exp88`, `exp94`, `exp96`, `exp100`.
They are excluded from `reproduce_paper3.sh` and exit with a missing-file error.

## Layout

```
config.py                     protocol constants (PTC_SIZE, thresholds, ...)
epbs/                         spec-anchored simulator port
experiments/                  experiments cited by the paper
  data/                       bundled delivered-payload panel (72 MB)
  figures/drl_risk_epbs/      generated results (JSON/CSV tracked; PNG/PDF regenerated)
  reproduce_paper3.sh         one-shot reproduction
difftest/                     differential tests for the load-bearing predicates
docs/PAPER3_ARTIFACT_README.md  claim → command map
docs/DISCLOSURE.md            responsible-disclosure record
```

## Differential tests against the executable specification

The conformance layer compares this port against the executable `pyspec` generated from
`consensus-specs` at the audited snapshot `5366cb59e`
(2026-08-05), the commit the paper quotes line numbers from. Build it once and point the
tests at it:

```bash
git clone https://github.com/ethereum/consensus-specs
cd consensus-specs && git checkout 5366cb59e
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

The suite also runs against `master`. Checked at `015d72704` (2026-07-02, the previous pin): **291 passed, 2 failed**, and both failures are the same
genuine drift rather than a portability problem —

- `test_payload_due_bps_matches_spec` / `test_payload_due_ms_matches_spec`:
  `PAYLOAD_DUE_BPS` moved from `7500` to `5000` upstream (#5414, the builder's reveal
  deadline moved from 9 s to 6 s into the slot). `config.py` now tracks the current
  value, so running against the older snapshot trips the drift detector instead. Nothing the paper
  claims depends on this constant: the PTC attestation deadline (`7500`) and the regular
  attestation deadline (`2500`) are unchanged.

Everything else passes, including all 160 predicate-level differentials and the 257/256
boundary. The suite is tolerant of the upstream renames that landed in the same window
(`uint*` → `Uint*` #5469, `boolean` → `Boolean` #5466) and of `should_build_on_full`
gaining an explicit `slot` parameter (#5497), so no edit is needed to try a newer
snapshot.

## Data provenance

One panel is bundled under `experiments/data/`, so nothing here needs an external
download:

- `block_value_panel_delivered_2026H1.parquet` — **935,314 blocks**, 2026-02-21 to
  2026-07-10 (UTC), the calibration panel for Figures `valuedist` / `breakeven` /
  `robust`. Each row is a payload a relay reports having delivered, carrying the block
  hash and the value the builder paid.

Built by `exp103` (fetch) and `exp104` (merge, adjudicate, verify) from the eight relays'
`proposer_payload_delivered` endpoints. Delivery shares: Ultra Sound 30.9%, bloXroute
max-profit 24.3%, Titan 18.0%, Aestus 14.6%, bloXroute regulated 6.4%, Agnostic 2.2%,
EthGas 1.8%, Flashbots 1.8%.

Two independent checks, both reported by `exp104`:

- **`parent_hash` chaining.** A block that lands becomes the `parent_hash` of the next
  slot's bids, so the canonical hash of a slot can be recovered from the public RelayScan
  bid archive alone. Over the 172,088 blocks where that reconstruction and the delivered
  records overlap, block hashes agree on **100.0%** and values on 99.5%.
- **Builder coinbase.** Where the bid archive also carries the delivered block's hash,
  its `block_fee_recipient` is the builder's coinbase for that block. Compared against
  the on-chain miner address from XBlock-ETH (Zheng, Zheng, Wu and Dai, *IEEE Open
  Journal of the Computer Society* 1 (2020) 95–106,
  [doi:10.1109/OJCS.2020.2990458](https://doi.org/10.1109/OJCS.2020.2990458)), the two
  agree on **99.995%** of the 145,509 blocks where both exist, with 8 mismatches. Note
  that the delivered records carry `proposer_fee_recipient`, a different address, which
  is why the comparison is routed through the archive rather than taken from them.

314 heights carry two delivered blocks at two different slots — reorganisations. 216 are
adjudicated by the `parent_hash` reconstruction; the remaining 98 are dropped rather than
guessed, since preferring the later slot agrees with the evidence only 77% of the time.

The raw per-relay shards `exp103` writes are not distributed (467 MB); re-fetching them
takes about 85 minutes.

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
