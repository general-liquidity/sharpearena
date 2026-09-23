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
| Fold per-seed readbacks into one arm | `merge_effective_configs`, which refuses when the seeds did not run one configuration |
| Reject rather than fall back | Every path above raises; none warns and continues |

Not owned: the upstream library revision and the wrapper order. An adapter records
both in its own evidence block. INT-03 onward should put them beside the effective
config rather than in a separate file.

## C02: observation and action mapping

| Clause | Owner |
|---|---|
| Keys, shapes, dtype, bounds | `gym.SharpeArenaEnv.observation_space` / `action_space`, `vector.single_observation_space`, `spaces.flatten_obs` |
| The wire shapes behind the spaces | `observation_space_schema()` and `decision_space()` on the env |
| Batching dimension | `vector.SharpeArenaVectorEnv`, `batch_space` |
| Intended versus validated actions | `_action_validation.validated_action` |
| No silent transformation | `validated_action` refuses shape, kind, finiteness and bound violations; it does not reshape, clip or coerce |
| Causal preprocessing, fitted on training data only | `preprocessing.CANONICAL_PREPROCESSING`, `wrappers.CausalNormalizeObservation` |

Two dtype facts an adapter must carry rather than rediscover. The action space is
`float32` and the engine consumes `float64`, so any fixed action sequence used on both
sides must be `float32` representable or the two paths receive different numbers. And
`Box.contains` casts safely, so a `float64` array is not a member of a `float32` space;
the decision belongs in the contract, not in each adapter.

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
partial run cannot read as a whole one.

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

`TrustClass` separates provenance from integrity. A digest establishes that bytes did
not change; it does not make deserialising them safe and says nothing about who produced
them. `TRUSTED_LOCAL` is the only class that answers yes to
`safe_to_deserialize_locally`.

## What INT-01 could not settle

These need an owner decision and block only the routes named:

1. **Wrapper order as part of identity.** C01 requires the wrapper order to be bound,
   and nothing records it today. Whether it belongs in the effective-config readback or
   in a separate adapter block changes what `merge_effective_configs` has to accept.
   Blocks nothing until INT-03 records its first wrapper stack.
2. **The dtype boundary.** The action space is `float32` and the engine is `float64`.
   Parity fixtures quantise, which is correct for a fixture, but whether the adapter
   should advertise a `float64` action space, or whether learners should be required to
   send `float32`, is a contract decision, not a fixture detail. Affects INT-03, INT-05
   and INT-07.
3. **What L2 completion means per route**, which the plan itself lists as unresolved.
   `AttemptCounts` gives the shape of the evidence but not the threshold.
4. **The `verifiers` version.** The module states 0.1.14 and CI pins 0.3.1. Until INT-10
   reconciles them, no verifiers-dependent claim should name a single supported version.
