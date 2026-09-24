# Integration inventory and version matrix (INT-00)

What the tree actually contains on `b9fdf30`, read against the integration plan's
claims rather than assumed from them. Every row below was checked against the source
file named in it. Nothing is listed as missing that was not grepped for first. The
original pass was checked against `d33b249`; this update reread every row against the
current commit rather than trusting the earlier one, and the one row that had gone
stale (HUD and Harbor, below) is corrected.

## What already exists

| Ecosystem | Present | Where | Upstream import |
|---|---|---|---|
| <picture><source media="(prefers-color-scheme: dark)" srcset="../assets/logos/gymnasium-dark.svg"><img src="../assets/logos/gymnasium.svg" alt="" height="16"></picture> Gymnasium scalar | Yes | `crates/sharpearena-py/python/sharpearena/gym.py` (`SharpeArenaEnv`) | Hard; `gymnasium>=1.0` is a base dependency |
| <picture><source media="(prefers-color-scheme: dark)" srcset="../assets/logos/gymnasium-dark.svg"><img src="../assets/logos/gymnasium.svg" alt="" height="16"></picture> Gymnasium vector | Yes | `vector.py` (`SharpeArenaVectorEnv` over native `VecTradingEnv`) | Hard, with a guarded `AutoresetMode` import that falls back to the string label |
| <picture><source media="(prefers-color-scheme: dark)" srcset="../assets/logos/gymnasium-dark.svg"><img src="../assets/logos/gymnasium.svg" alt="" height="16"></picture> Gymnasium wrappers | Yes | `wrappers.py`, `wrappers_vector.py`, `spaces.py` | Hard |
| <picture><source media="(prefers-color-scheme: dark)" srcset="../assets/logos/gymnasium-dark.svg"><img src="../assets/logos/gymnasium.svg" alt="" height="16"></picture> Gymnasium registration | Yes | `registration.py`, IDs `SharpeArena/<Tier>[-Eval]-v1` with scalar and vector entry points | Hard |
| Conformance checker | Yes | `check_env.py` (`check_env`, `check_determinism_across_constructors`) | None; hand written, numpy only |
| <picture><source media="(prefers-color-scheme: dark)" srcset="../assets/logos/pettingzoo-dark.svg"><img src="../assets/logos/pettingzoo.svg" alt="" height="16"></picture> PettingZoo | Yes | `pettingzoo_env.py`, `lob_env.py`, `market_env.py` | Guarded; `ParallelEnv` falls back to `object` and construction raises a named error |
| <picture><source media="(prefers-color-scheme: dark)" srcset="../assets/logos/minari-dark.svg"><img src="../assets/logos/minari.svg" alt="" height="16"></picture> Minari | Yes | `minari_export.py` (`to_minari`, `to_minari_train_test`, `seed_band_metadata`) | Guarded behind `_require_minari()` |
| <picture><source media="(prefers-color-scheme: dark)" srcset="../assets/logos/prime-intellect-dark.png"><img src="../assets/logos/prime-intellect.png" alt="" height="11"></picture> verifiers / Prime-RL | Yes | `verifiers_env.py`, `examples/prime-rl/` | Guarded; the class only exists when `verifiers` imports |
| SharpeBench bridge | Yes | `bench_bridge.py` | Standard library only |
| <a href="https://modelcontextprotocol.io"><picture><source media="(prefers-color-scheme: dark)" srcset="../assets/logos/mcp-dark.svg"><img src="../assets/logos/mcp.svg" alt="" height="11"></picture></a> MCP | Yes | `mcp_server.py` | Guarded behind `FastMCP is None` |
| Functional view | Yes | `functional.py` (`SharpeArenaFuncEnv`) | Three-way probe ending in a local shim, so the module always imports |
| CleanRL | No | Nothing in the tree | |
| <a href="https://www.ray.io"><img src="../assets/logos/ray.svg" alt="" height="12"></a> Ray, RLlib | No | Only a literature citation in `paper/review/environment-genre-study-2026.md` | |
| <picture><source media="(prefers-color-scheme: dark)" srcset="../assets/logos/hud-dark.svg"><img src="../assets/logos/hud.svg" alt="" height="14"></picture> HUD | Yes, as a local feasibility fixture, not a supported integration | `examples/hud/`, `crates/sharpearena-py/tests/test_hud_local.py`. Only `LocalRuntime` and `SubprocessRuntime` were exercised; `DockerRuntime`, `ModalRuntime`, and `HUDRuntime` were not. Report: `docs/integrations/INT-08-hud-local-feasibility.md`. | Not imported by the package; the fixture depends on the `hud` PyPI package |
| <picture><source media="(prefers-color-scheme: dark)" srcset="../assets/logos/harbor-dark.png"><img src="../assets/logos/harbor.png" alt="" height="13"></picture> Harbor | Yes, as a local feasibility fixture, not a supported integration | `integrations/harbor/` (task package, fixtures, tampering jobs). Ran on Docker Desktop over WSL2. Report: `docs/integrations/INT-09-harbor-local-feasibility.md`. | Not imported by the package; the fixture depends on the `harbor` PyPI package |
| <img src="../assets/logos/stable-baselines3.png" alt="" height="16"> Stable-Baselines3 | No | Only the phrase "SB3-style MLP feature extractors" in `spaces.py:5`, a docstring | |
| <a href="https://pytorch.org/rl"><img src="../assets/logos/torchrl.png" alt="" height="10"></a> <img src="../assets/logos/envpool.svg" alt="" height="10"> TorchRL, PufferLib, EnvPool | No | Nothing in the tree | |

Every row above with an upstream project of its own carries that project's logo;
see [`docs/assets/logos/`](../assets/logos/) for each file's source and the terms
governing its use. A logo credits the project this tree talks to. It is not a
support claim: the "Present" column beside it, and
[`support-status.md`](support-status.md), are where that is stated, and two of
the rows carrying a logo are local feasibility fixtures rather than supported
integrations. Rows for ecosystems this tree does not implement carry no logo.

The declared optional extras are exactly four: `verifiers`, `minari`, `pettingzoo`,
`mcp` (`crates/sharpearena-py/pyproject.toml`). The plan's warning holds: generic
Gymnasium and Farama compatibility is real and already tested, and the gap is the
seven named consumer routes, not the interfaces they consume.

### Two claims with thin test backing

`docs/capabilities.md` lists both, and both have a module behind them, so neither is a
false claim. Neither has a direct test:

- `wrappers_vector.py` (`VectorCausalNormalizeObservation`, `VectorRecordEpisodeStatistics`)
  has no test file, and no test names either class.
- `spaces.py` (`flatten_obs`, `unflatten_obs`, `flat_dim`, `FlattenObservation`) is
  exercised only indirectly, through `PreprocessingConfig(flatten=True)` in
  `tests/test_preprocessing.py`.

Both belong to INT-03, which is where the Gymnasium surface is brought to contract
coverage.

## Autoreset modes this environment supports

Gymnasium defines three, and this package supports all three. `SharpeArenaVectorEnv`
accepts `next_step`, `same_step` and `disabled` (`vector.py`), maps them onto
`gymnasium.vector.AutoresetMode` when that enum is importable, and reports the chosen
mode in `metadata["autoreset_mode"]`.

| Mode | Behaviour here | Final observation |
|---|---|---|
| `next_step` | Default. The terminal step is returned verbatim; the lane resets on the following step | None. The terminal observation is the step return itself |
| `same_step` | The lane resets in place, so the returned batch already holds the new episode's first observation | `infos["final_obs"]` and `infos["final_info"]`, object arrays with `None` for lanes that did not finish |
| `disabled` | The lane never auto-resets and stays at its terminal bar | None |

Two consequences for adapter work. The key is `final_obs`, not Gymnasium's
`final_observation`, and `wrappers_vector.py` exists because the stock vector wrappers
assume `NEXT_STEP` and assert the metadata is an `AutoresetMode` enum rather than a
string. An adapter that assumes the stock wrapper stack works unchanged under
`same_step` is wrong in both respects. The mode belongs in the recorded configuration
alongside the Gymnasium version, because learner support for it is mode dependent.

Termination and truncation are distinct here and the distinction is not the usual one:
running out of bars at the end of the point-in-time window is **truncation**, and
bankruptcy (`nav <= 0`) is **termination**, the absorbing state. A value estimate
bootstraps past the first and must not bootstrap past the second (`gym.py:243`).

## Version matrix

Two columns, because they differ and the difference matters. "CI pin" is what
`crates/sharpearena-py/ci-requirements.txt` installs. "Verified here" is what was
actually installed in the environment these INT-00 to INT-02 checks ran in; the full
suite passed under it (1893 passed, 2 skipped).

| Component | Declared floor | CI pin | Verified here |
|---|---|---|---|
| Python | `>=3.9` (`pyproject.toml`) | not pinned | 3.12.6, CPython, MSC v.1940 |
| NumPy | `>=1.21` | 2.5.3 | 2.5.1 |
| Gymnasium | `>=1.0` | 1.3.0 | 1.2.1 |
| PettingZoo | extra, unpinned | 1.27.0 | 1.26.1 |
| Minari | extra, `minari[create,hdf5]` plus `pillow` | 0.5.3 | 0.5.3 |
| h5py | via the Minari extra | 3.16.0 | 3.13.0 |
| jax / jaxlib | via the Minari extra | 0.11.1 | 0.6.1 |
| pillow | via the Minari extra | 12.3.0 | 11.2.1 |
| verifiers | extra, `>=0.3.1,<0.4` | 0.3.1 | 0.1.14 (this row only; see the INT-10 caveat below for the separate 0.3.1 pass) |
| mcp | extra, unpinned | 1.30.0 | not installed |
| pytest | not shipped | 9.1.1 | 8.4.2 |
| OS / architecture | not constrained | Linux runners | Windows 11 26220, x86-64 (AMD64) |

Caveats a later ticket must not read past:

- The Python floor is the base package's. It does not carry to any optional learner,
  and each extra needs its own range.
- INT-10 is closed: `verifiers_env.py` now states it was verified against `verifiers`
  0.3.1, matching the CI pin. The module targets 0.3.1's `verifiers.legacy` v0 API
  (`MultiTurnEnv`, `stop`/`cleanup`, `Rubric`), which 0.3.1 aliases transparently onto
  the top-level `verifiers.*` names — the same surface the module was previously
  verified against at 0.1.14 — so the fix was re-verifying and re-pinning, not a
  rewrite. The 0.3.1 pass built a separate isolated venv (`verifiers==0.3.1`, plus
  `numpy`, `gymnasium`, `pytest`, `jsonschema`, but not `pettingzoo`/`minari`/`mcp`,
  which this narrower venv did not need) against this tree's already-built extension
  and ran the full `tests/` suite except `test_hud_local.py` (unrelated to `verifiers`,
  needs the separate `hud` package and Docker): 1811 passed, 65 skipped, no source
  changes, no deprecation warnings. The 65 skips are the `pettingzoo`/`minari`/`mcp`
  tests this venv could not exercise, not anything about `verifiers`; the two
  `verifiers`-specific files (`test_verifiers.py`, `test_verifiers_episode_outcomes.py`,
  82 tests) are part of that 1811. This is a separate, narrower-dependency venv from
  the one the "Verified here" column above documents (which predates this pass and
  used `verifiers` 0.1.14 among a wider set of extras), so that column keeps its
  0.1.14 value rather than implying that exact environment was rerun.
  `tests/test_verifiers.py::test_verified_version_matches_ci_pin` fails CI if the
  module's claimed version and the pin ever diverge again, and
  `UnsupportedVerifiersAPIError` (a structural capability probe, not a version-string
  check) fails loudly if a future `verifiers` release drops or reshapes that surface.
- Gymnasium's three autoreset modes exist as an enum only from 1.1. The guarded import
  in `vector.py` is what keeps 1.0 working, and a pinned combination must record the
  Gymnasium version and the mode together.
- Minari's PyPI release is 0.5.3 while 0.5.4 is tagged upstream, and their NumPy floors
  differ. Pin what is installed, not what a changelog describes.
- No Linux or macOS run backs this row. The verified column is one machine.

The INT-02 tests were additionally run against an installed wheel
(`sharpearena-0.31.0-cp312-cp312-win_amd64.whl`, built from this tree) in a clean
virtual environment outside the checkout holding only NumPy 2.5.3, Gymnasium 1.3.0 and
pytest, which is the CI pin for both. All 30 passed, and the base environment imported,
reset and stepped there with no PettingZoo, Minari, verifiers or MCP installed.

## Existing parity and equivalence checkers

Four, and they check different things. INT-02 adds a fifth rather than replacing any.

| Checker | Compares |
|---|---|
| `tests/test_equivalence.py` | The batched path against B scalar envs, lane for lane, on identical actions, including under execution noise |
| `check_env.py::check_env` | One env against the Gymnasium contract plus exact reset determinism, and optionally that distinct seeds diverge |
| `check_env.py::check_determinism_across_constructors` | Two identically built envs against each other, step by step, on exact observation and reward equality |
| Cross-runtime goldens (`test_python_golden.py`, `test_wire_conformance.py`, `test_spec_hash.py`, `test_canonical_json.py`) | Python against Rust and WASM on fingerprints, wire payloads, the spec hash and canonical bytes |
| `integrations/parity.py` (new, INT-02) | The native engine against an adapter for one effective configuration and one action sequence, including refused and incomplete cases |

The gap the new one fills: nothing compared an adapter against the engine underneath
it. `test_equivalence.py` compares two adapters to each other, so a mapping error
shared by both is invisible to it. The upstream Gymnasium checkers do not verify reward
correctness, the semantics of `terminated`, info contents, action validity, episode
boundaries or autoreset, and there is no vector checker at all, so for the seven
consumer routes this is the only gate that exists.

## Handoff notes

For the branch working on RL contract coverage (`test_gym.py`, `test_vector.py`,
`test_checkpoint.py`, `test_rewards_risk_aware.py`, `test_seed_bands.py`,
`test_eval_seeds_gate.py`), which this work did not edit:

1. INT-03 should assert Gymnasium parity through `integrations.parity.check_adapter_parity`
   rather than hand rolling an engine comparison. The fixtures are in `CORE_FIXTURES`
   and take an adapter factory.
2. Fixture actions must be quantised to `float32` before use. The action space is
   `float32` while the engine takes `float64`, so an unquantised weight reaches the two
   paths as two different numbers and the difference reads as a mapping bug.
3. A parity fixture must go through the engine's resolved `(scenario, execution)` seed
   pair. `SharpeArenaEnv` splits one user seed into both; the native constructor takes
   them directly. `parity.resolved_seeds` restates the split.
4. `wrappers_vector.py` and the `spaces.py` flatten helpers are the two untested
   Gymnasium exports listed above.
