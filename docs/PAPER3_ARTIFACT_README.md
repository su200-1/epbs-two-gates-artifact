# Paper 3 Artifact README

Reproducibility entry point for *Two Fixed-Seat Paths to Payload Orphaning in
ePBS*.

The repository is the artifact source of truth; the manuscript's
"Data and Code Availability" cites this file. Every headline number is produced by
one command below and written to a tracked JSON under
`experiments/figures/drl_risk_epbs/`.

## Snapshot

- Repository branch: `codex/drl-incentive-attack-exploration`
- Spec snapshot: `SPEC_SNAPSHOT.json`
- Pinned `consensus-specs`: `ec1c01f5f10fb60636a022ae8944c912b6da35f8`
  (`v1.7.0-alpha.10`, 2026-06-01). The load-bearing PTC/timeliness/payment functions
  are unchanged across the full 73-commit range to the current spec HEAD `015d72704`
  (2026-07-02), verified two ways: (i) source-level `git diff` of `payload_timeliness`,
  `should_build_on_full`, `should_extend_payload`, `process_builder_pending_payments`,
  and the same-slot payment-weight accrual — only a behavior-preserving `is`→`==` and a
  semantically-equivalent branch reorder, with no new PTC slashing and no
  orphan-voids-payment coupling; and (ii) the executable differential suite re-run
  against the pyspec generated at `015d72704`. The functions are inherited unchanged by
  the successor heze / EIP-7805 fork.

## Environment

```bash
PY=/opt/anaconda3/envs/pbs-gym/bin/python
```

All experiments run in `pbs-gym` (imports `torch` for exp85). One-shot
reproduction of the simulator headlines, plus exp88 when `PAPER1_PANEL` is set or
one of the companion panel's documented default paths is available:

```bash
bash experiments/reproduce_paper3.sh
```

## Claim → command map

Each row is a paper section, its command, and the headline it reproduces.

### §4 The Payload-Timeliness Orphaning Attack (exact two-gate boundary)

```bash
$PY experiments/exp92_exact_ptc_payment_boundary.py
```

The paper's primary mechanism evidence, at the real committee size
(`PTC_SIZE = 512`, 1024 validators).  A deterministic executable-predicate matrix
crosses not-present PTC seats `255/256/257/258` with the independent
regular-attestation payment weight `59/60/61 %`.  The parent payload status flips
`FULL` → `EMPTY` strictly above 256 not-present votes (i.e. at 257), while payment
settles at or above the 60 % quorum in the same cell: every settling cell records a
builder debit and proposer credit of `10^8` gwei, every expiring cell records zero.
Output schema `exact-ptc-payment-boundary-v1`, written to
`exp92_exact_ptc_payment_boundary.json` and `.md`.

### §4 The Payload-Timeliness Orphaning Attack (scaled end-to-end sanity check)

```bash
$PY experiments/exp81_ptc_orphan_probe.py --seeds 8 --slots 64
```

Corrupting enough PTC seats to vote "not timely" orphans the honest builder's
revealed payload through the real ported fork-choice, while the settled debit is
unchanged.  Rates are pooled over revealed payloads whose child is observed:
orphan_rate `0.000` (honest) → `0.545`, `0.606`, `0.712`, and `1.000` at false-vote
fractions `0.50`, `0.60`, `0.75`, and `1.00`, respectively.  Under full PTC
corruption, all `66/66` evaluable reveals are orphaned; one final-slot reveal has no
observed child and is reported as right-censored rather than included in the
denominator.  The builder debit remains `259.0M` gwei and
`DECOUPLED (orphan + pay) = True`.  The output uses schema `exp81-v2`, records the
pooled aggregation rule and censoring note, and is written to
`exp81_ptc_orphan_probe_s8_t64.json`.

### §5 Reservation-Price Sensitivity: A Fixed Pivotal Set

```bash
$PY experiments/exp82_ptc_bribe_economics.py
```

The pivotal set is a fixed `257` PTC seats, because the spec uses a strict
`> PAYLOAD_TIMELY_THRESHOLD` test with `PAYLOAD_TIMELY_THRESHOLD=256`.
A validly signed `not present` vote triggers no direct protocol-level penalty,
balance deduction, or slashing under the audited specification. The protocol-implied
direct-cost lower bound is therefore zero, but the specification does not determine
an operator's market reservation price. exp82 consequently treats the per-seat price
`p_seat` as an exogenous sensitivity parameter rather than deriving it from
`SIM_BASE_REWARD_GWEI` or a presumed forgone reward.

For default per-seat prices `0`, `0.00001`, `0.0001`, `0.001`, `0.01`, and `0.1`
ETH, the total 257-seat break-even costs are `0`, `0.00257`, `0.0257`, `0.257`,
`2.57`, and `25.7` ETH. Under the illustrative log-normal block-value distribution
(median 0.05 ETH, σ=1.27), the corresponding shares of profitably attackable blocks
are `100.0%`, `99.0%`, `70.0%`, `9.9%`, `0.096%`, and `0.000044%`. The zero-price
row is a protocol-cost lower bound, not a claim that operators can be recruited for
free; positive prices may reflect coordination, reputation, legal exposure,
counter-bribes, and pivotal-seat holdout power. The `exp82-v3` output is
`exp82_ptc_bribe_economics.json`.

### §6 Victim Harm and MEV Re-capture

Unconditional griefing (default calibration):

```bash
$PY experiments/exp83_ptc_theft_recapture.py
```

Victim: honest util `+59.7M` vs attacked `-256.8M`; loss/episode `316.5M` gwei
(`0.32` ETH), orphan_rate `0.99` — the builder pays for an empty block regardless of
re-capture. → `exp83_ptc_theft_recapture_default_s16_t64.json`.

Conditional theft (high-MEV calibration, mean ≈ 1 ETH):

```bash
$PY experiments/exp83_ptc_theft_recapture.py --expected-profit-gwei 1000000000
```

Selective theft is profitable on the high-value calibration under a conservative
25% re-capture across the three explicitly assumed per-seat prices. At ρ=0.25,
`p_seat = 0.00001 / 0.0001 / 0.001 ETH` (total 257-seat cost
`0.00257 / 0.0257 / 0.257 ETH`) gives
`+2755M / +2586M / +1202M` per episode. These prices are sensitivity inputs, not
protocol-derived validator losses.
→ `exp83_ptc_theft_recapture_highmev_s16_t64.json`.

### §7 Spec-Faithfulness Audit

```bash
$PY -m pytest difftest/ -q                       # 293 passed, 0 skipped
# the two predicate-level differentials the paper's boundary claims rest on:
$PY -m pytest \
  difftest/test_primitives.py::TestTimelinessPredicatesAgainstPyspec \
  difftest/test_forkchoice.py::TestPayloadGateConsumersAgainstPyspec \
  -q -rs                                         # 160 passed, 257/256 boundary vs pyspec
```

Of the 293 tests, 206 execute the generated gloas spec. The four functions the paper's
claims rest on — `payload_timeliness`, `payload_data_availability`,
`should_build_on_full`, `should_extend_payload` (plus `get_payload_status_tiebreaker`) —
are executed side by side with the spec on vote vectors built from shared parameters, at
the mainnet committee size. The sweep is exhaustive over the seat count: for every
not-present count in 0–512 the port and the spec agree, the least count satisfying
`timely=False` is 257 in both, and the least number of withheld affirmations that
defeats `timely=True` is 256 in both.

The differential layer requires the executable gloas pyspec. Build it once:

```bash
cd $CONSENSUS_SPECS            # $CONSENSUS_SPECS
pip install uv && PATH="$(dirname $(which python)):$PATH" make _pyspec
# runtime deps for importing the generated gloas spec into pbs-gym (py3.10):
$PY -m pip install eth-remerkleable==0.1.31 trie==3.1.0 lru-dict==1.4.1 \
    frozendict==2.4.7 eth-typing eth-utils eth-hash hexbytes rlp cytoolz toolz \
    sortedcontainers ckzg cachebox pycryptodome py-arkworks-bls12381
```

If the pyspec is absent, the `TestAgainstPyspec` layer self-skips with
`gloas pyspec not importable`; the sim-internal behavioral tests still run. Override
the generated-spec path with `EPBS_PYSPEC_DIR`.

### §Deployment Requirements and Evidentiary Boundary

The manuscript states four deployment preconditions (recruitment, verifiable
conditional payment, ex-ante targeting, re-capture) and reports **no** learned
deployment policy.  The two scripts below back no table or claim in the paper; they
are retained only so the earlier selective-attack exploration stays reproducible.

```bash
$PY experiments/exp84_selective_attack_oracle.py     # SUPERSEDED: not cited by the paper
$PY experiments/exp85_selective_attack_neural.py     # SUPERSEDED: not cited by the paper
```

exp84 uses `p_seat=0.0014 ETH` by default. Selectivity beats indiscriminate attack
by a wide margin. The cost-aware state-conditional rule has a higher point estimate
than the best fixed signal threshold (`836.3M` versus `793.7M` gwei), but its paired
CI includes zero (`CI_low=-21.0M`), so `sc_beats_signal=False` and no superiority
claim is made. exp85 reports all five neural initializations rather
than selecting the best run. None exceeds the tuned signal or two-feature net-value
threshold: neural mean `-329.6M` gwei (sample SD `122.6M`, range `-411.4M` to
`-114.0M`) versus signal `+812.7M` and net-value `+811.6M`. Because those
thresholds are selected on the evaluation episodes, this is descriptive negative
evidence rather than a formal out-of-sample superiority test: learning does not
strengthen the attack claim. → `exp84_selective_attack_oracle_s*.json`,
`exp85_selective_attack_neural_bf0.3_pseat0p0014.json`.

### §Mitigations

```bash
$PY experiments/exp86_mitigation_eval.py
```

Mitigations that re-couple the two gates remove the fixed-set, unaccountable-vote
property: an accountable-vote penalty (M2) is added to the exogenous per-seat
reservation price; a stake-scaled penalty or
a stake-weighted / enlarged PTC (M3) turns the fixed 257-vote pivotal set back into a
stake share, restoring the payment gate's stake-dependence.
→ `exp86_mitigation_eval_s16.json`.

### §Feasibility (empirical block-value calibration)

```bash
$PY experiments/exp87_empirical_block_value.py
```

Upgrades exp82's log-normal to a cached sample of `n=4800` mainnet relay blocks
(median `0.0087`, mean `0.0317` ETH; top 20% of blocks hold `79.8%` of value).
Because the attack is selective, the load-bearing quantity is *value* attackable, not
*blocks*. At `p_seat=0.0001 ETH`, the total break-even is `0.0257 ETH`; `10.1%`
of blocks clear it and those blocks hold `73.6%` of sampled value. At
`p_seat=0.001 ETH`, only `0.8%` of blocks clear the `0.257 ETH` break-even but
they hold `59.1%` of sampled value. These are ex-post distributional sensitivities,
not evidence of pre-slot identification. Cached sample: `exp87_block_value_sample.json`;
result: `exp87_empirical_block_value.json`.

```bash
$PY experiments/exp88_gap_conditioned_block_value.py \
    --panel /path/to/block_auction_panel.parquet
```

Further upgrades exp87 from a single-relay 4.8k sample to an 8-relay,
**493,330-block** auction panel (`block_auction_panel.parquet`;
`top_bid_eth` = cross-relay max builder bid; median `0.0118`, mean `0.0393` ETH; top 20%
of blocks hold `76.5%` of value). At `p_seat=0.0001 ETH`, `21.5%` of blocks clear
the `0.0257 ETH` break-even and hold `77.5%` of panel value; at `p_seat=0.001 ETH`,
`1.2%` clear the `0.257 ETH` break-even and hold `46.6%` of panel value.
The script retains a visibility-gap-decile breakdown as an auxiliary descriptive
stratification only. The gap is contemporaneous and the analysis does not provide or
validate a pre-slot predictor, so it is not used as evidence that an attacker can
identify profitable slots within the decision window. This experiment does not re-run
the simulator; only the value distribution feeding the break-even changes.
Outputs: `exp88_gap_conditioned_block_value.json`, `exp88_gap_decile_summary.csv`,
`exp88_gap_decile_profit.csv`, `exp88_attackable_value_concentration.csv`,
`exp88_gap_conditioned_block_value.png`.

```bash
$PY experiments/exp94_empirical_joint_economics.py
```

Crosses attacker-owned PTC seat share (`0 / 10 / 20 / 33.3 %`), exogenous per-seat
reservation price, and re-capture ratio `rho` against the same bundled panel.  At
`0.001 ETH/seat` and zero pre-owned seats the marginal recruitment is `257` seats
(`0.257 ETH`) and `1.22%` of panel blocks clear at `rho=1.00`, falling to `0.28%` at
`rho=0.25`; a `33.3%` pre-owned share cuts the marginal recruitment to `86.3` seats
(`0.086 ETH`) and raises coverage to `4.79%` / `0.87%`.  These are ex-post panel
coverage figures, not a deployable targeting result or realized attack profit.
Output: `exp94_empirical_joint_economics.json` and `.md`.

### §Feasibility (PTC operator concentration)

```bash
$PY experiments/exp93_ptc_operator_concentration.py
```

`100,000` synthetic 512-seat committees per cell, reporting medians with 5–95 %
intervals: how many additional seats and how many *distinct operators* the 257-seat
predicate requires under different operator-concentration profiles.  With no
pre-owned stake the attacker must add `257` seats; under a `Zipf(128, alpha=1.0)`
operator distribution those seats sit with a median of `8` distinct operators, and at
a `33.3%` pre-owned share only `86` further seats spread over `2` operators.  This is
a synthetic coordination envelope, **not** a mainnet validator-to-operator census and
not evidence that any operator would accept a bribe.
Output: `exp93_ptc_operator_concentration.json` and `.md`.

## Main artifacts (tracked JSON)

Under `experiments/figures/drl_risk_epbs/`:
`exp81_ptc_orphan_probe_s8_t64.json`, `exp82_ptc_bribe_economics.json`,
`exp83_ptc_theft_recapture_default_s16_t64.json`,
`exp83_ptc_theft_recapture_highmev_s16_t64.json`, `exp84_selective_attack_oracle_s24.json`,
`exp85_selective_attack_neural_bf0.3_pseat0p0014.json`, `exp86_mitigation_eval_s16.json`,
`exp87_block_value_sample.json`, `exp87_empirical_block_value.json`,
`exp88_gap_conditioned_block_value.json`, `exp88_gap_decile_summary.csv`,
`exp88_gap_decile_profit.csv`, `exp88_attackable_value_concentration.csv`.

Legacy note: `exp83_ptc_theft_recapture_s16_t64.json` is a superseded unlabelled
output name. Use the `default` and `highmev` files above so the griefing and
high-MEV theft calibrations cannot overwrite each other.

## Simulator and spec anchor

- Simulator package: `epbs/` (fork-choice `forkchoice.py`, PTC/timeliness
  `primitives.py`, committee sampling `committee.py`, tier-2 environment
  `env_tier2.py`, payments `builder_payments.py`, rewards `rewards.py`).
- Top-level config: `config.py` (`PTC_SIZE=512`, `PAYLOAD_TIMELY_THRESHOLD=256`,
  `PROPOSER_SCORE_BOOST=40`).
- exp81–88 depend on `experiments/exp27_epbs_selfish_builder_markout.py`
  (reveal/markout hooks) and exp85 imports helpers from exp84.

## Manuscript

Not distributed with this artifact while the paper is under review. The accepted
version will be linked from the top-level `README.md` once it is available; this
document is written to stand on its own in the meantime, mapping each claim to the
command that reproduces it.
