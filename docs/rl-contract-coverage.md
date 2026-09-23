# RL-contract coverage matrix

Which parts of the declared reinforcement-learning contract are covered by a test, where
that test lives, and what remains open. Written for ticket P17, against
`crates/sharpearena-py` at the v0.31.0 tree.

This is engineering coverage of the environment contract. It says nothing about what a
learner would learn in this environment, and it is not evidence for any learning claim.

Two upstream definitions the matrix uses. Gymnasium separates `terminated`, the task
completing or failing, from `truncated`, an external limit such as the end of a window
([basic usage](https://gymnasium.farama.org/introduction/basic_usage/)). Gymnasium
defines exactly three vector autoreset modes, `NEXT_STEP`, `SAME_STEP` and `DISABLED`,
and warns that learner support depends on which one is active
([vector autoreset mode](https://farama.org/Vector-Autoreset-Mode)). SharpeArena
implements all three; `crates/sharpearena/src/vec_env.rs` carries the enum and
`AutoresetMode::from_label` accepts `next_step`, `same_step` and `disabled` only.

## Matrix

Existing checks were reused as they stand. Nothing in the "existing coverage" column was
rewritten, duplicated or re-counted as new work.

| Contract item | Existing coverage reused | Added in this ticket | Status |
|---|---|---|---|
| Reset returns an observation inside the declared space, with split-seed info | `test_gym.py::test_reset_returns_observation`, `test_reset_info_carries_split_seeds`, `test_conformance.py::test_check_env_passes` | none | covered before |
| Reseeding through `reset(seed=k)` selects a reproducible scenario | `test_gym.py::test_reset_seed_selects_scenario`, `test_seed_bands.py::test_gym_env_places_an_eval_seed_in_the_held_out_band` | none | covered before |
| Same seed reproduces the trajectory; different seeds diverge | `test_gym.py::test_determinism_same_seed_identical_rewards`, `test_conformance.py::test_same_seed_identical_reset_obs`, `test_different_seed_reset_obs_differ` | none | covered before |
| Observation and action spaces match the dataset and hold under the Gymnasium checker | `test_conformance.py::test_gymnasium_env_checker_passes`, `test_vector.py::test_vector_wrapper_reset_step_shapes`, `test_registration.py::test_make_resolves_each_tier` | none | covered before |
| Invalid actions are refused before any lane advances | `test_action_validation.py` (all cases), `test_checkpoint.py::test_checkpoint_wrapper_rejects_invalid_actions_before_advance`, `test_gym.py::test_native_binding_rejects_bad_json` | none | covered before |
| Scenario and execution seeds are distinct streams resolved from one user seed | `test_gym.py::test_reset_info_carries_split_seeds`, `test_equivalence.py::test_batched_equals_scalar_under_execution_noise` | `test_rl_contract.py::test_execution_seed_moves_fills_without_moving_the_price_path`, `test_scenario_seed_moves_the_price_path` | gap closed |
| `terminated` is a blow-up, reached inside the horizon | none: every prior `terminated` assertion ran against a stub env, never the engine | `test_rl_contract.py::test_insolvency_terminates_inside_the_horizon_without_truncating` | gap closed |
| `truncated` is the window ending, and the terminal observation a learner sees | `test_gym.py::test_full_episode_finite_rewards_and_terminates` (that an episode ends), `test_conformance.py::test_time_limit_truncates_at_cap` (the wrapper cap) | `test_rl_contract.py::test_running_out_of_bars_truncates_and_repeats_the_last_bar` | gap closed |
| Scalar and vector trajectories agree under aligned seeds | `test_equivalence.py::test_batched_equals_scalar`, `test_batched_equals_scalar_under_execution_noise`, `test_vector.py::test_b1_matches_scalar_engine` | none | covered before |
| Each supported autoreset mode behaves as declared | `test_vector.py::test_next_step_defers_reset_to_following_step`, `test_same_step_surfaces_final_obs`, `test_disabled_never_resets`, `test_unknown_autoreset_mode_rejected` | `test_rl_contract.py::test_metadata_records_the_gymnasium_autoreset_mode`, `test_only_the_three_upstream_modes_are_accepted` | gap closed for the recorded mode |
| Lanes hold independent state | `test_vector.py::test_auto_reset_keeps_batch_running` (uniform actions only) | `test_rl_contract.py::test_each_lane_matches_a_solo_env_under_per_lane_actions` | gap closed |
| Checkpoint and restore continuity, including the RNG-implied state | `test_checkpoint.py::test_restore_returns_to_snapshot_point`, `test_checkpoint_preserves_every_scenario_control`, `test_replay_checkpoint_after_native_restore_keeps_the_entire_prefix`, `test_native_checkpoint_matches_replay`, `test_two_branches_same_actions_identical` | none | covered before |
| Per-step reward, episode accounting and the reported evaluation metric are one chain | `test_metrics_panel.py` (panel arithmetic on supplied series), `test_eval_seeds_gate.py` (the pinned snapshot) | `test_rl_contract.py::test_reward_is_the_nav_return_the_metrics_panel_reconstructs`, `test_the_reported_eval_metric_scores_exactly_the_reward_series` | gap closed |
| Wrapper behaviour and reward scaling stay causal and preserve the 5-tuple | `test_conformance.py::test_wrapper_preserves_5_tuple`, `test_causal_normalize_obs_is_prefix_stable`, `test_causal_normalize_reward_prefix_stable`, `test_frame_stack_shape`, `test_record_episode_statistics_injects_episode` | none | covered before |
| Failed and incomplete episodes keep their own accounting | `test_verifiers_episode_outcomes.py`, `test_finish_reasons.py`, `test_risk.py` | none | covered before |
| The environment runs with no optional dependency installed | `test_gym.py::test_verifiers_import_guarded` (the `verifiers` import only) | `test_rl_contract.py::test_the_adapter_runs_with_every_optional_dependency_blocked` | gap closed |
| The packaged adapter, not only the source tree | `ci.yml` job `wheel-install-import` (native `TradingEnv`, one bar) | `scripts/check-packaged-adapter.py`, run by the same CI job | gap closed |

## What the added tests pin

`test_insolvency_terminates_inside_the_horizon_without_truncating` drives one fully
shorted symbol under persistent large jumps at seed 2. NAV crosses zero at bar 165 of a
200-bar window, so the step reports `terminated` true and `truncated` false with bars
still left. Before this, no Python test reached a `terminated` step from the engine; the
existing risk-layer tests raise the flag from stub environments.

`test_running_out_of_bars_truncates_and_repeats_the_last_bar` pins what the truncating
step hands a learner. The episode spans exactly one step per bar in the window, the final
step reports `truncated` true with a positive NAV, and its observation repeats the
previous step's closes because there is no further bar to advance to. That step still
earns the NAV return of the bar, so it is an accounting step rather than padding, which
is what makes bootstrapping past it correct.

`test_execution_seed_moves_fills_without_moving_the_price_path` holds the scenario seed
fixed and varies the execution seed: every observed close stays byte-identical and the
rewards move. `test_scenario_seed_moves_the_price_path` is its control in the other
direction. Existing tests bind the two streams together to prove scalar and vector agree;
neither shows that the streams are separable.

`test_each_lane_matches_a_solo_env_under_per_lane_actions` gives the lanes different
actions and compares each against a single-lane environment on the same seed, then
repeats with the seed order and the action rows both reversed. Existing vector tests send
the same action to every lane, which a transposed or broadcast dispatch would survive.

`test_metadata_records_the_gymnasium_autoreset_mode` checks `metadata["autoreset_mode"]`
against Gymnasium's own enum member for each of the three modes. A learner reads the mode
from the metadata, and upstream support is mode-dependent, so the recorded value is part
of the contract; the existing tests assert behaviour and the private attribute instead.

`test_reward_is_the_nav_return_the_metrics_panel_reconstructs` walks a real episode and
asserts each reward equals the step's NAV ratio minus one, then shows that feeding
`RunMetrics` the rewards and feeding it the NAV series produce the same panel.
`test_the_reported_eval_metric_scores_exactly_the_reward_series` reproduces every named
held-out seed's `mean_return` and `deflated_sharpe` from a hand-rolled reward series, so
a rescaling introduced anywhere between the step reward and the reported metric fails.

`test_the_adapter_runs_with_every_optional_dependency_blocked` installs an import blocker
for `verifiers`, `minari`, `pettingzoo`, `mcp`, `torch` and `jax` in a subprocess before
importing `sharpearena`, then constructs, resets and steps both the scalar and the vector
adapter. `scripts/check-packaged-adapter.py` does the same against an installed
distribution rather than the source tree, and refuses to run if any of those packages is
importable or if the import resolved outside the interpreter's package directory.

## Isolated-cause controls

Each added test was checked against a deliberate defect in an isolated copy of the
package, never in the working tree. Every control changed one thing and failed the
intended test only.

| Mutation | Failing tests |
|---|---|
| `SharpeArenaEnv.step` always reports `terminated` false | the insolvency test only |
| `SharpeArenaEnv.step` doubles the reward | the truncation test and the reward/NAV chain test only |
| `eval_seeds._rollout_returns` rescales the collected returns | the reported-metric test only |
| `SharpeArenaVectorEnv` stops forwarding `env_kwargs` to the native batch | the execution-seed test, alongside the pre-existing `test_equivalence.py` cases that bridge the two streams |
| `LaneConfig::build` seeds the scenario generator from the execution seed, rebuilt and installed as a wheel | both stream-separation tests only |
| `SharpeArenaVectorEnv` dispatches lane 0's action to every lane | the lane-independence test only |
| The autoreset label table maps `next_step` to `SAME_STEP` | the metadata test for `next_step` only |
| The dependency blocker is pointed at a required dependency | the blocked-dependency subprocess, confirming the blocker bites |

A spec-hash or import mismatch cannot stand in for these failures. The suite already
fails loudly and separately on either: `test_spec_hash.py` and `scripts/check-packaged-spec.py`
cover the cross-surface handshake, and `test_exports.py` covers the import surface. The
added tests assert numeric and flag-level behaviour that those checks do not produce.

## Unresolved boundaries

- `SharpeArenaVectorEnv.reset(seed=...)` accepts a seed and ignores it. Lane seeds are
  fixed by the constructor and `scenario_seeds` is unchanged after such a call, so
  reseeding a batch means constructing a new one. Gymnasium's vector API expects
  `reset(seed=...)` to reseed. This is left as found; resolving it changes environment
  semantics, which P17 does not authorize. It is deliberately not pinned by a test, so
  that a later fix is not blocked by a regression blessing the current behaviour.
- `SharpeArenaEnv` derives its execution seed from the user seed and rejects an
  `env_kwargs={"exec_seed": ...}` override with a `TypeError` from the duplicate keyword.
  The vector surface accepts the same override. The stream-separation tests therefore run
  on the vector surface. The asymmetry is recorded, not changed.
- The `CheckpointState.include_rng` flag is documented as reserved for a future
  stochastic-fill mode and does not change behaviour today. Nothing here establishes a
  contract for it.
- The insolvency case is one reproducible configuration, not a characterization of when
  the engine terminates. It shows that the terminating transition exists, is reachable
  inside the horizon and is flagged distinctly from truncation.
- Coverage of the packaged adapter comes from the wheel built by CI on Linux and, for
  this ticket, a locally built Windows wheel. Other platforms are covered by the existing
  cross-platform jobs, which do not run the adapter script.
