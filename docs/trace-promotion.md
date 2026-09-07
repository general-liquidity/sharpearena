# Producer replay and trace promotion

`sharpearena.promotion/2.0.0` gold cases reconstruct a Gym environment, replay
every captured action from reset, and apply a named invariant to the new output.
The same case should fail with the defective producer and pass after its repair.
Gold evaluation does not require repaired output to equal the old output.

The only supported producer is `sharpearena.gym-actions/1`, implemented by
[`promotion_replay.py`](../crates/sharpearena-py/python/sharpearena/promotion_replay.py#L1).
It runs the installed `SharpeArenaEnv` and native engine with supplied target-weight
actions. It makes no model calls and does not load callables or plugins from case
data. LOB environments, arbitrary wrappers and model/scaffold execution are
outside this adapter's scope.

## Capture and review

Capture while the original producer still reproduces the defect. Supply the
original unwrapped `SharpeArenaEnv` and its complete action list, including every
action before the failing step. The source trace must use the adapter's pre-action
observations and include its effective construction config in metadata.

The following assumes `original_env` and `actions` belong to that source capture:

```python
from pathlib import Path

from sharpearena.promotion_replay import gym_replay_inputs
from sharpearena.trace_promotion import (
    SilverStore, blocking_failures, build_silver_candidate,
    load_trace_strict, run_promotion_checks,
)

trace = load_trace_strict(Path("private/source.jsonl"))
inputs = gym_replay_inputs(original_env, actions)
failures = blocking_failures(run_promotion_checks(trace))
if not failures:
    raise ValueError("source trace has no blocking failure")
candidate = build_silver_candidate(trace, failures[0], replay_inputs=inputs)
SilverStore(Path("private/silver-v2.jsonl")).append(candidate)
```

`gym_replay_inputs` captures every constructor field: dimensions, seed, window,
CSV text, weight limits, shorting, distribution and jump/volatility controls,
train/eval mode, and `env_kwargs`. For example, fees use
`env_kwargs={"fee_bps": 3.0}`. Use the actual recorded configuration; do not infer
missing parameters from rewards or observations. Actions must contain at least
two finite numeric vectors with the environment's exact shape and bounds.
The caller owns and closes `original_env`; each replay closes its new environment.

`build_silver_candidate` reruns these inputs and compares the complete source
steps and metadata before building the candidate. Mismatch raises
`PromotionError`. Only the silver diagnostic excerpt is minimized; gold keeps the
full replay history. An output-only silver candidate can be triaged but cannot
be promoted. Adding replay inputs changes its ID and requires review of that new
candidate.

After review, record the operator's decision against `candidate.candidate_id`:

```python
import time

from sharpearena.trace_promotion import OperatorDecision, promote_to_gold

decision = OperatorDecision(
    candidate_id=candidate.candidate_id,
    decision="promote",
    operator=reviewer_id,
    rationale=review_rationale,
    decided_at_unix_ns=time.time_ns(),
)
gold = promote_to_gold(candidate, decision)
gold.write(Path("private/gold") / f"{gold.case_id}.json")
```

`reviewer_id` must be nonblank and `review_rationale` must contain at least 20
characters after trimming surrounding whitespace. A rejection or a decision
naming another candidate cannot promote it. Gold has its own content ID and
retains the approved candidate ID in its decision record.

## Evaluate after repair

```python
from sharpearena.trace_promotion import evaluate_gold_case, load_gold_case

case = load_gold_case(Path("private/gold") / f"{gold.case_id}.json")
outcome = evaluate_gold_case(case)
assert outcome.passed, f"{outcome.check_id}: {outcome.detail}"
```

`evaluate_gold_case` returns a `GoldOutcome` with `case_id`, `check_id`, `passed`
and `detail`. Loading and evaluation revalidate the case identity. Capture,
promotion, record export and evaluation copy nested payloads so edits to caller
inputs or exported records do not change their source. Direct nested edits to a
candidate or case invalidate its ID. Malformed files raise `TraceIntegrityError`;
identity, replay or promotion failures raise `PromotionError`. Constructor and
action validation errors also propagate to the caller.

## V1 migration and limits

V1 silver queues and output-only gold files are refused by V2 readers. Keep them
as historical diagnostics. To migrate, recover the original construction inputs
and complete actions, reproduce the source under the old producer, build a new
V2 silver candidate, and obtain a decision for its new ID. Changing the schema
label or preserving only the actions around a failure cannot migrate a case.
If the inputs or original reproduction are unavailable, it remains diagnostic
evidence and cannot become an executable gold regression.

Constructor settings and default selections are captured; future library behavior
is not pinned by the recipe. Testing changed producer behavior is the purpose of
gold replay. Lineage metadata such as model and dataset digests is supplied by the
caller. The adapter recomputes construction config and step count, but does not
independently authenticate those other claims. Fingerprints summarize lineage;
candidate and gold IDs separately bind case content. Hashes detect changed
content under an ID, not authenticated authorship. An `OperatorDecision` records
a declaration, not a signature or proof of actual human approval.

Promotion and gold evaluation guard ordinary Python `socket.socket` construction.
The guard changes process-global state: do not run concurrent replays in one
process. It does not block existing sockets, native I/O or subprocess networking
and supplies no OS isolation. Direct `replay_gym_inputs(inputs, meta)` returns a
new `StrictTrace` without installing that guard.

Replay inputs contain private seeds and may contain the complete CSV, including
future prices. Keep silver queues, gold files and exported records operator-only;
never give them to evaluated agents or publish them without inspecting and
authorizing their contents. Content validation does not enforce access controls.

Implementation: [`trace_promotion.py`](../crates/sharpearena-py/python/sharpearena/trace_promotion.py#L1).
Regression coverage: [`test_trace_promotion.py`](../crates/sharpearena-py/tests/test_trace_promotion.py#L1).
See also the [capability map](capabilities.md#agent-operations).
