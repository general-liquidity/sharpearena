# C01 to C07 mapped onto the types that already own them (INT-01)

The integration plan asks for shared contracts before adapter proliferation, and it
asks for them without a parallel type system. Most of C01 to C07 is already owned by
something in this package. This page names the owner for each clause, so an adapter
fills an existing structure instead of inventing one, and it lists the three things
that had no owner and are now in `sharpearena.integrations.contracts`.

Read `sharpearena/integrations/parity.py` next: it is the mechanism that checks an
adapter against these clauses rather than asserting them in prose.

## C01: effective configuration and identity

| Clause | Owner |
|---|---|
| Read the configuration back out of what consumed it | `effective_config.env_effective_config`, `check_env_effective_config` |
| Distinguish requested from effective | `EffectiveConfigError` names every disagreement; the readback carries `verified` |
| Bind the data and generator identity | `scenario_fingerprint` (FNV-1a/64 over the generated panel) and the `dataset_fnv1a64` readback |
| Bind the native package and spec identity | `_spec_hash.EXPECTED_SPEC_HASH`, `check_spec_hash`, checked at package import |
| Bind the scorer version | `kernel_score` and the SharpeBench pin it carries |
| Bind the wrapper order | `effective_config.wrapper_stack`, carried as `wrapper_stack` in the readback and compared by `check_env_effective_config(wrappers=...)` |
| Fold per-seed readbacks into one arm | `merge_effective_configs`, which refuses when the seeds did not run one configuration |
| Reject rather than fall back | Every path above raises; none warns and continues |

Not owned: the upstream library revision. An adapter records it in its own evidence
block. INT-03 onward should put it beside the effective config rather than in a separate
file.

Decision on wrapper order: it belongs in the readback, so one check and one merge rule
cover it. `wrapper_stack` lists qualified class names, innermost first, read off the
object the caller passes in. `check_env_effective_config` compares it against the
`wrappers` argument, whose default is none, so a wrapped environment has to be declared.
`merge_effective_configs` refuses seeds that ran different stacks, and refuses a block
with no `wrapper_stack` because absent is not empty. The merged block carries the key only
when the stack is non-empty, so a bare arm keeps the shape of the frozen evidence, whose
arms all ran the bare environment.

The stack binds classes and their order. It does not bind constructor arguments, such as
the `TimeLimit` cap or the `FrameStack` depth, or a normalizer's `training` flag and
statistics. `describe_preprocessing` discloses the arguments for the canonical pipeline.

## C02: observation and action mapping

| Clause | Owner |
|---|---|
| Keys, shapes, dtype, bounds | `gym.SharpeArenaEnv.observation_space` / `action_space`, `vector.single_observation_space`, `spaces.flatten_obs` |
| The wire shapes behind the spaces | `observation_space_schema()` and `decision_space()` on the env |
| Batching dimension | `vector.SharpeArenaVectorEnv`, `batch_space` |
| Intended versus validated actions | `_action_validation.validated_action` |
| No silent transformation | `validated_action` refuses shape, kind, finiteness and bound violations; it does not reshape, clip or coerce |
| Causal preprocessing, fitted on training data only | `preprocessing.CANONICAL_PREPROCESSING`, `wrappers.CausalNormalizeObservation` |
| Normalizer persistence and evaluation freeze | `state_dict`, `load_state_dict` and `training` on `CausalNormalizeObservation`, `CausalNormalizeReward` and `VectorCausalNormalizeObservation` |
| The action dtype rule | `contracts.ACTION_SPACE_DTYPE`, `contracts.ENGINE_ACTION_DTYPE`, `contracts.is_float32_exact` |

The dtype rule, stated once so no adapter decides it again:

1. The action space stays `float32`. A learner sends `float32`.
2. `step` takes any real numeric array that passes `validated_action`, and the value
   reaches the engine as `float64` without being rounded to the advertised dtype. Widening
   a `float32` value is exact, so a `float32` action and its widening are one engine input.
3. A `float64` action that is not `float32` representable is still accepted. The checkpoint
   layer depends on this to replay the bytes it recorded. Such an action is not a member of
   the space, because `Box.contains` casts safely, so an adapter must not gate actions on
   `contains`. Use `validated_action`, which is what `step` applies.
4. Two paths are comparable only when the fixed action sequence is `float32` representable.
   `is_float32_exact` checks that, and the parity fixtures quantise to it.

Advertising a `float64` space was rejected because it changes the dtype every learner
samples from. Requiring `float32` at `step` was rejected because it breaks checkpoint
replay. Neither change was needed, so default behaviour stays as it was. The bounds check
compares the `float64` value against the `float32` bound, which for a limit that `float32`
cannot represent admits values within that bound's rounding error.

Normalizer statistics stay in Arena's wrappers rather than a learner's own normalizer,
because those wrappers carry the causal guarantee. The contract:

- `state_dict()` returns JSON-serialisable data that shares no memory with the wrapper, so
  a checkpoint needs no pickle. `load_state_dict` validates everything before changing
  anything, and raises `NormalizerStateError` for a wrong wrapper, format, key, shape, lane
  count or a non-finite value.
- `training=False` stops every update. An emitted value is then a pure function of that bar
  and the frozen statistics, including on the first bar, which `training=True` emits as
  zeros. A freeze with fewer than two samples raises `NormalizerNotFittedError` before the
  engine advances, rather than emitting zeros or passing rewards through.
- Loading never changes the flag. An evaluator sets `training=False`, on every normalizer in
  its stack. `get_wrapper_attr` finds only the outermost one, so walk `.env` to reach both.
- The reward wrapper and the vector wrapper clear their accumulators at each episode
  boundary, which would leave a checkpoint taken there with nothing to freeze. They keep
  the last cleared accumulator with at least two samples as `retired`, which never affects
  a training-mode value. Frozen scale is `retired` where it exists, otherwise the live
  accumulator. So the frozen reward scale comes from one episode, not from all of training.
- Vector state is per lane and refuses a different `num_envs`.

C02's substantive requirement is the one about probability: a wrapper that squashes,
projects or discretises changes the decision problem, and the learner's log probability
must be computed under the parameterisation it actually samples from. This package
refuses out-of-bound actions rather than clipping them, which keeps the problem
unchanged; an adapter that adds a squash owns the correction and must say so.

## C03: lifecycle and return targets

| Clause | Owner |
|---|---|
| Reset and explicit reseed | `SharpeArenaEnv.reset(seed=...)` rebuilds the scenario; `seed=None` keeps it |
| Terminal versus truncated | `gym.step`: out of bars is truncation, `nav <= 0` is termination |
| Autoreset modes | `vector.SharpeArenaVectorEnv(autoreset_mode=...)`, all three of `next_step`, `same_step`, `disabled` |
| Final observation | `infos["final_obs"]` and `infos["final_info"]` under `same_step` only |
| Time limits | `wrappers.TimeLimit`, and the registered `max_episode_steps` backstop |
| Typed outcomes for invalid input and crashes | The `SharpeArenaError` taxonomy with `[CODE]` prefixes at the pyo3 boundary |
| Failure classification | `failure_taxonomy.classify_episode_failure`, `episode_outcomes.reward_eligible` |

Terminal liquidation and horizon end are already separate here, which is the trap C03
names: the window ending is not an absorbing state. `integrations.contracts`
restates the supported modes as `SUPPORTED_AUTORESET_MODES` and
`AUTORESET_MODES_WITH_FINAL_OBS`, and the test for them constructs the vector env
rather than reading the upstream enum, so the list cannot drift from behaviour.

## C04: training reward versus benchmark verdict

| Clause | Owner |
|---|---|
| Shaped per-step reward | `rewards.REWARD_SCHEMES` and the functions it names |
| Episode outcome | `episode_outcomes`, `failure_taxonomy.FailureRollup` |
| Diagnostic statistics, never the rank key | `metrics.RunMetrics` |
| Benchmark verdict | `kernel_score.score_run` and the SharpeBench kernel behind it |
| Unavailable rather than zero | `KernelScoreUnavailable`, `is_kernel_score_unavailable`, `kernel_score_or_unavailable` |
| Refusal semantics to reuse | `decision_parser`, `counterfactual` dispositions (`risk_refused`, `not_submitted`) |

The default scoring method is unchanged by any of this work. A learner's episode scalar
is not a benchmark input, and the typed unavailability already exists, so a HUD or
Harbor reward mapping that must return a scalar on failure exports the failure status
separately rather than reporting a clean zero.

## C05: seed, split and resource scheduling

| Clause | Owner |
|---|---|
| Scenario seed and execution seed, kept separate | `gym._resolve_seeds`, restated in `effective_config.resolved_scenario_seed` and `parity.resolved_seeds` |
| Band boundary, one definition | `_seed_bands.EVAL_SEED_BASE` |
| Frozen evaluation set | `eval_seeds.EVAL_SEEDS`, `evaluate_eval_set`, `assert_no_regression` |
| Private evaluation seeds | `eval_seeds.sealed_eval_seeds`, salt keyed |
| Disjoint train and eval bands | `generalization.train_test_seeds`, `dataset.seed_ranges_disjoint` |
| Explicit curriculum schedule | `curriculum.CurriculumEnv`, `regime_curriculum`, `AdaptiveScheduler` |
| Worker and lane identity | `vector.SharpeArenaVectorEnv(seeds=...)`, one scenario seed per lane, fixed at construction |

One behaviour an adapter must not assume away: `SharpeArenaVectorEnv.reset(seed=...)`
ignores the seed. Lane scenarios are chosen at construction. Repeated `reset(seed=None)`
is therefore not a training-diversity schedule here either, which is what C05 says.

Not owned: worker allocation independent of completion order, and resource accounting
for a distributed executor. INT-06 owns both. Note for that ticket: Ray tasks are
at-least-once by default, so an attempt identity must be assigned before scheduling and
ledger writes must be idempotent under it, and Ray resources are logical, so a thread
cap for the Rust engine is a contract obligation rather than something the scheduler
enforces.

## C06: trajectories, attempts and manifests

| Clause | Owner |
|---|---|
| Ordered observations, actions, rewards, done flags | `trace.RolloutTraceWriter`, `load_trace`, `trace_to_returns` |
| Episode and agent identity | `trace` meta records, `pettingzoo_env` agent ids |
| Process events | `event_contract`, `episode_outcomes.is_process_block` |
| Task configuration and terminal outcome | `trace` meta, `edge_manifest.EdgeManifest` |
| Attempt ledger with duration and token accounting | `bench_bridge`, which refuses a journal that disagrees with itself |
| Unknown usage stays unknown | `bench_bridge` duration sources `unavailable` and `mixed` |
| No private chain of thought required | `decision_parser` reads the structured decision only |
| Raw returns kept for the existing scorer | `trace_to_returns` into `kernel_score` |

Added here: `contracts.AttemptCounts`. `bench_bridge` counts attempts for the local
model field, but nothing let an RL or harness adapter state expected, attempted,
completed, refused, failed and retried for a rollout batch and have the arithmetic
checked. `AttemptCounts` refuses a record where the dispositions do not sum to the
attempts, where more trials completed than were planned, or where a count is missing,
because absent is not zero. `is_complete` requires zero refused and zero failed, so a
partial run cannot read as a whole one relative to its declared expected count.
An empty workload is valid arithmetic, not evidence of an experiment. Consumers
must call `require_complete(expected=<independent plan count>)` to enforce that
count; empty workloads require `allow_empty=True`. This does not authenticate the
plan or establish that its count was independently committed.

## C07: reproducibility and trust class

| Clause | Owner |
|---|---|
| Identical configuration and actions preserve engine outputs | `integrations.parity`, `check_env.check_determinism_across_constructors`, `tests/test_equivalence.py` |
| Engine build identity | `check_spec_hash` |
| Cross-runtime byte identity | `test_python_golden.py`, `test_wire_conformance.py`, `test_canonical_json.py` |
| Restricted artifacts kept out of agent-visible state | `lookahead_guard`, `trace._reject_leaky`, `test_secret_leak_probe.py` |

Added here: `contracts.ExactnessDomain` and `contracts.TrustClass`.

The exactness domain makes the boundary explicit instead of implicit. `ENGINE_EXACTNESS`
declares reward, nav, observation digest, the two terminal flags, the outcome label and
the step count exact, with no tolerances, because one deterministic engine build replays
one tape under one action sequence on both sides of the comparison. A field in neither
list has no declared rule and `tolerance_for` refuses it, so a comparison cannot happen
under a rule nobody wrote down. That domain deliberately does not extend to trained
policy parameters, GPU reductions or distributed optimiser state, and it should stay
narrower than any upstream framework's seeding language implies: Gymnasium does not
claim seed independence across vector lanes, and RLlib's seeding statement does not
scope itself for differing worker counts, restarts or asynchronous sampling.

Tolerance values must be finite and positive. The domain snapshots the supplied
mapping so later caller mutations cannot change the comparison rules. Parity
also requires finite numeric reward and NAV on successful transitions, even when
both records contain the same invalid value or the field has a tolerance.
Environment conformance guards remain active under `python -O`; cross-instance
determinism checks include both terminal flags, not only observations and reward.

`TrustClass` separates provenance from integrity. A digest establishes that bytes did
not change; it does not make deserialising them safe and says nothing about who produced
them. `TRUSTED_LOCAL` is the only class that answers yes to
`safe_to_deserialize_locally`.

## What INT-01 could not settle

These need an owner decision and block only the routes named:

1. ~~**Wrapper order as part of identity.**~~ Settled: it is in the effective-config
   readback. See the decision under C01.
2. ~~**The dtype boundary.**~~ Settled: the action space stays `float32` and `step` keeps
   taking an exact `float64`. See the rule under C02.
3. **What L2 completion means per route**, which the plan itself lists as unresolved.
   `AttemptCounts` gives the shape of the evidence but not the threshold.
4. ~~**The `verifiers` version.**~~ Settled by INT-10: the module now states 0.3.1,
   matching the CI pin, and `pyproject.toml`'s `verifiers` extra carries a matching
   `>=0.3.1,<0.4` bound. A version-pin drift test guards against this reopening
   silently.
