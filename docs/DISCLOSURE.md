# Responsible disclosure

| | |
|---|---|
| **Status** | **Reported; answered.** |
| Venue | `ethereum/consensus-specs` issue #5516 |
| Date sent | 2026-08-05 |
| Link | https://github.com/ethereum/consensus-specs/issues/5516 |
| Response | 2026-08-05, `potuz` (maintainer; EIP-7732 co-author): *"This is a deliberate tradeoff, this issue should be closed."* |
| Follow-up | 2026-08-05, ours: asked whether "deliberate" also covers the evidentiary asymmetry (that denying the affirmation requires no signed object). Unanswered as of this writing. |
| Cross-post | *pending — ethresear.ch topic 16054* |

Reported to the specification authors before the paper was submitted. The text below is
the full write-up the issue links to; the shorter text actually posted as the issue is
reproduced at the end of this file. Both are given unchanged.

**Reading the response.** The reply disputes no part of the mechanism — not the predicate
reading, not the seat counts, not the line references. It states that the behaviour is a
deliberate trade-off, which is what this work assumes throughout: EIP-7732's rationale
says the omission of PTC penalties was chosen for "simplicity of implementation," and
unconditional settlement is documented as closing the builder free option. The paper
prices a deliberate choice rather than reporting a defect, and the reply is consistent
with that framing. What the reply does not address is whether the *consequence* the issue
asked about --- that because an absent seat counts for neither side, the denial direction
produces no object a vote-accountability rule could adjudicate --- was itself weighed.
That question is the subject of the follow-up above.

---

## gloas: the builder-payment predicate and the payload-timeliness predicate are decoupled — a fixed-seat cost to orphan a punctually revealed payload

*Line numbers and the runnable check below are against `master` @ `46d3d3513`
(2026-08-05). Our manuscript audits the `015d7270` (2026-07-02) snapshot; we re-checked
every function cited here against current `master` and none of them changed
semantically.*

---

## Summary

We have been auditing the gloas (EIP-7732) payload-timeliness and builder-payment paths
for a paper, and would like to check our reading with the people who designed them
before we publish.

Two observations, both of which we believe follow directly from the current spec text:

1. A coalition holding a majority of one slot's PTC seats can cause a payload that was
   revealed on time to be treated as `EMPTY` — and the spec exposes **two** routes to
   this, with different seat counts and different evidentiary footprints.
2. The builder's payment obligation is untouched by that outcome. The victim builder
   pays its winning bid for a block whose execution content is discarded.

Neither observation contradicts anything in the EIP; unconditional settlement is a
deliberate choice made to close the builder free option, and the PTC's honest-majority
assumption is stated. What we think is worth a second look is the **quantity** that has
to be corrupted: a fixed number of committee seats that does not grow with the validator
set.

We are not reporting an exploit against anything deployed — gloas is not on mainnet, and
this is a design question rather than an implementation bug, which is why we are raising
it here rather than through the bug bounty.

## What we checked

Line numbers are from `master` @ `46d3d3513`.

- `payload_timeliness` — `specs/gloas/fork-choice.md:274-289`
- `payload_data_availability` — `specs/gloas/fork-choice.md:295-310`
- `should_build_on_full` — `specs/gloas/fork-choice.md:423-433`
- `should_extend_payload` — `specs/gloas/fork-choice.md:445-457`
- `get_payload_status_tiebreaker` — `specs/gloas/fork-choice.md:463-473`
- `get_head` — `specs/gloas/fork-choice.md:571-594`
- vote-array initialisation `[None] * PTC_SIZE` — `specs/gloas/fork-choice.md:1003`
- `PAYLOAD_TIMELY_THRESHOLD = PTC_SIZE // 2` — `specs/gloas/fork-choice.md:79`
- `process_attestation` (builder-payment weight accrual) — `specs/gloas/beacon-chain.md:1947-1952`
- `process_builder_pending_payments` — `specs/gloas/beacon-chain.md:1222-1233`
- `process_payload_attestation` — `specs/gloas/beacon-chain.md:1974-1985`

Three properties of `payload_timeliness` that the rest of this depends on:

- **It is not complementary.** It counts votes *equal to the queried value*, so
  `timely=True` and `timely=False` are two separate strict-majority tests. A 256/256
  split satisfies neither.
- **Silence is not a `not present` vote.** The tally is initialised as
  `[None] * PTC_SIZE`, and `None` equals neither `True` nor `False`, so an absent seat
  contributes to neither side.
- **There is a local short-circuit** when the payload never arrived. Our threat model
  excludes it: the victim is a builder that *did* reveal on time.

## Two routes to an orphan

**Route A — override the next proposer's build rule.** `should_build_on_full` is
fail-open: it returns `False` only if `payload_timeliness(..., timely=False)` holds,
which needs **257 signed `not present` votes**. This needs no cooperation from the next
proposer — an honest proposer, reading the corrupted tally, builds `EMPTY` itself. The
cost is 257 seats and the signature trail is permanent.

**Route B — mute the tiebreaker.** `should_extend_payload` is a four-way disjunction
whose first conjunct is `payload_timeliness(..., timely=True)`, needing 257 *affirmative*
votes. Denying that affirmation takes only **256 seats staying silent** — and by the
property above, silence is enough. But the remaining conjuncts mean this route
additionally requires a boosted proposer that builds `EMPTY` on the target, so it costs
one fewer seat and one proposer.

The two are not equivalent in what they buy: Route A buys 257 attributable signatures,
Route B buys 256 non-events that are indistinguishable from latency or downtime. Route B
does relocate rather than erase the evidence — its proposer is an identifiable on-chain
deviation — but no accountability rule written over *cast votes* reaches its seats.

## The payment side

`process_builder_pending_payments` settles on `payment.weight >= quorum` alone and never
references any PTC vote; `payload_timeliness` never references the accrued weight. The
weight accrues in `process_attestation`, gated on `is_attestation_same_slot`.

We note that #5514 makes this cleaner to state than it was when we started. Now that
`get_attestation_participation_flag_indices` takes payload status into account, the
same-slot branch pins it explicitly:

```python
if is_attestation_same_slot(state, data):
    assert data.index == 0
    payload_matches = True
```

Since builder-payment weight accrues only for same-slot attestations, payload status
still cannot perturb the payment quorum — and that is now an assertion in
`beacon-chain.md` rather than something a reader has to derive from `validator.md`. So
under either route above, the payload is not extended while the pending payment is
untouched, and it settles at the epoch boundary.

This is the part we would most like confirmed as intended. We understand the rationale
for unconditional settlement (closing the builder free option), and we are not proposing
to reverse it — but the case it was reasoned about seems to be the builder's *own*
withholding, not a third party imposing the same loss on a builder that behaved
honestly.

## One concrete note on the 2/3 threshold in the rationale

EIP-7732's rationale suggests that PTC equivocation "could be mitigated by setting
`PAYLOAD_TIMELY_THRESHOLD` to be 2/3 of the PTC." Separating the two routes makes
visible that this change is direction-sensitive. For a committee of `N` seats and
threshold `T`:

```
Route A (override) = T + 1        Route B (deny) = N - T
```

At `N=512`, moving `T` from 256 to 341 raises Route A from 257 to 342 seats and lowers
Route B from 256 to 171 — the same 85 seats, in opposite directions. Against the
equivocating-builder case the rationale has in view, the change is a real strengthening;
against an adversary that mutes the committee and supplies its own proposer, it is a
weakening. We do not read this as an argument against the change, only that the
threshold cannot be tuned against one direction in isolation.

## What we are *not* claiming

- **Not that the seats are cheap.** The protocol attaches neither penalty nor reward to
  a payload attestation — `process_payload_attestation` is three asserts, with no
  `increase_balance` or `decrease_balance` on any control path, and there is still no
  payload-attestation slashing anywhere in the spec. So the price of a seat is entirely
  exogenous. We sweep it as a parameter; we have no evidence about what an operator
  would actually charge, and at any price we would consider plausible the attack does
  not clear on our block-value panel.
- **Not that recruitment is practical.** 257 assigned seats opting in before their
  attestation deadline, with no implemented conditional-payment primitive, is the
  binding constraint and we have not demonstrated it.
- **Not that value can be re-captured profitably.** That is swept, not measured.

The part we do think is solid is the protocol-level statement: the pivotal quantity is a
fixed seat count, it is non-slashable, and the payment survives it.

## Reproducing the boundary against your own pyspec

Nothing here depends on our simulator. This runs against the generated gloas spec —
we ran it against `master` @ `46d3d3513`:

```python
import eth_consensus_specs.gloas.mainnet as spec

def R(i): return spec.Root(i.to_bytes(32, "little"))

def store(not_present, present):
    return spec.Store(
        time=spec.Uint64(2 * int(spec.config.SLOT_DURATION_MS) // 1000),
        genesis_time=spec.Uint64(0),
        justified_checkpoint=spec.Checkpoint(), finalized_checkpoint=spec.Checkpoint(),
        unrealized_justified_checkpoint=spec.Checkpoint(),
        unrealized_finalized_checkpoint=spec.Checkpoint(),
        proposer_boost_root=spec.Root(), equivocating_indices=set(),
        blocks={R(2): spec.BeaconBlock(slot=spec.Slot(1))},
        block_states={}, block_timeliness={}, checkpoint_states={},
        latest_messages={}, unrealized_justifications={},
        payloads={R(2): spec.ExecutionPayloadEnvelope()},
        payload_timeliness_vote={R(2): [False] * not_present + [True] * present
                                 + [None] * (512 - not_present - present)},
        payload_data_availability_vote={R(2): [True] * 512})

head = spec.ForkChoiceNode(root=R(2), payload_status=spec.PAYLOAD_STATUS_FULL)

print("Route A - signed `not present` votes vs the build rule:")
for n in (256, 257):
    print("  %3d -> should_build_on_full = %s"
          % (n, spec.should_build_on_full(store(n, 512 - n), head, spec.Slot(2))))

print("Route B - withheld affirmations vs the tiebreaker's first conjunct:")
for silent in (255, 256):
    print("  %3d silent -> payload_timeliness(timely=True) = %s"
          % (silent, spec.payload_timeliness(store(0, 512 - silent), R(2), timely=True)))
    print("               should_build_on_full             = %s"
          % spec.should_build_on_full(store(0, 512 - silent), head, spec.Slot(2)))
```

Output:

```
Route A - signed `not present` votes vs the build rule:
  256 -> should_build_on_full = True
  257 -> should_build_on_full = False
Route B - withheld affirmations vs the tiebreaker's first conjunct:
  255 silent -> payload_timeliness(timely=True) = True
               should_build_on_full             = True
  256 silent -> payload_timeliness(timely=True) = False
               should_build_on_full             = True
```

The last two lines are the point of separating the routes: withholding does **not**
reach the build rule — an honest proposer still builds `FULL` — which is why Route B
additionally needs the next slot's proposer.

(On the `015d7270` snapshot the same script runs with `spec.uint64` instead of
`spec.Uint64` and `should_build_on_full(store, head)` without the slot argument, per
#5466 and #5497. The outputs are identical.)

We also ran a differential suite (293 tests, 206 of them executing the generated spec)
that sweeps every not-present count from 0 to 512 against an independent port of these
predicates; it is pinned to `015d7270` and we are happy to share it.

## Questions

1. Is the fixed-seat pivotal cost — 257 of 512, independent of total stake — an accepted
   property of the design, or is committee sizing expected to change before shipping?
2. Was the third-party case of the payment/timeliness decoupling evaluated, i.e. an
   honest builder that reveals on time, is orphaned by others, and still pays?
3. Is the 2/3 threshold suggestion in the rationale intended to be read together with an
   accountability rule, given that on its own it lowers the withhold-route cost?

## Coordination

We have a manuscript describing this and would rather it be accurate than early. We are
happy to delay submission, share the draft, or reword anything you consider misleading.
If you would prefer this discussed somewhere other than a public issue, tell us where
and we will move it.

---

# Companion: `ethereum/consensus-specs` issue

Posted as [ethereum/consensus-specs#5516](https://github.com/ethereum/consensus-specs/issues/5516)
on 2026-08-05, reproduced here verbatim. It is a short version built around what holds
in the specification as it stands today, linking back to this file for the rest;
everything in it is present-tense and assumes nothing about future parameter choices.
Line numbers are for `master` @ `46d3d3513`.

**Title**

```
gloas: denying should_extend_payload's PTC affirmation takes 256 withheld votes, not 257 signed ones
```

**Body**

```markdown
*Filing this publicly rather than via bounty@ethereum.org: `gloas` is not deployed,
nothing in the bounty programme's scope is affected, and this concerns a stated design
trade-off rather than an implementation bug. Happy to move it if you disagree.*

While auditing `gloas` we noticed that the PTC's verdict is read by two functions that
query it in **opposite directions**, so the two have different pivotal costs and, more
importantly, different evidentiary footprints. At the current
`PAYLOAD_TIMELY_THRESHOLD = PTC_SIZE // 2`:

- `should_build_on_full` (fork-choice.md L423-433) is fail-open. To make the next
  proposer build EMPTY on a punctually revealed payload you must satisfy
  `payload_timeliness(..., timely=False)` — that is, **257 seats must sign a
  `not present` vote**. Those signatures are permanent, attributable objects.
- `should_extend_payload` (fork-choice.md L445-457) queries the committee the other way,
  `timely=True`. Defeating that affirmation needs only **256 seats to withhold their
  vote**: the tally is initialised `[None] * PTC_SIZE` and the predicate counts
  `vote == timely`, so an absent seat counts toward neither side. There is nothing to
  sign, and a withheld payload attestation is indistinguishable from latency, a client
  fault, or ordinary downtime. (This route does additionally require the next slot's
  proposer to build EMPTY on the target, which is itself an attributable deviation.)

So the cheaper-to-hide route costs one seat *less* than the loud one, and no rule
written over cast votes can reach it — an accountability or slashing condition on
payload attestations would bind the first route and not the second.

Reproducible against the generated spec (checked at 46d3d3513):

    256 not-present -> should_build_on_full = True     257 -> False
    255 silent      -> payload_timeliness(timely=True) = True    256 -> False

Two questions:

1. Was this asymmetry intended? Treating a `None` seat as neutral is clearly deliberate
   for the affirmative query; the consequence that the *denial* is therefore reachable
   without producing any evidence seems worth stating explicitly somewhere.
2. Separately: `process_builder_pending_payments` settles on attestation weight alone
   and never reads the PTC vote, so a builder that revealed on time still pays when its
   payload is orphaned by either route. We understand unconditional settlement is
   deliberate (closing the builder free option); was the third-party case — an honest
   builder griefed by others — evaluated, or only the builder's own withholding?

One consequence worth flagging in case the threshold is ever revisited: for a committee
of `N` seats and threshold `T`, the first route costs `T+1` and the second `N-T`, so the
two move in opposite directions. EIP-7732's rationale mentions `2/3` as a possible
mitigation for PTC equivocation; at `N = 512` that would raise the first cost from 257
to 342 and lower the second from 256 to 171. We raise this only as a property of the
parameterisation, not as a claim that such a change is planned.

Full write-up, threat model, and a runnable script: <[epbs-two-gates](https://github.com/su200-1/epbs-two-gates-artifact/blob/main/docs/DISCLOSURE.md)>
```
