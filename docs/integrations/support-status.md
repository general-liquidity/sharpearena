# Support status (INT integration surface)

This page states, per ecosystem, how far each route has actually been checked. It
does not restate the inventory or the contract mapping; read
[`inventory.md`](inventory.md) for what exists and where, and
[`contracts.md`](contracts.md) for which module owns each contract clause.

## Status levels

A route is marked at the highest level it has direct evidence for, not the level its
design implies:

| Level | What it means |
|---|---|
| Specified | A contract clause has an owning type or module, with no dedicated conformance or workflow test found for it. |
| Contract-tested | A dedicated automated test or checker verifies the route against the ecosystem's own interface contract (a Gymnasium `check_env`-style pass, a wire-shape test, the adapter parity checker). |
| Workflow-tested | A runnable example or training recipe exercises the route end to end, beyond a unit test. |
| Empirically evaluated | The route produced a reported, reproducible quantitative result (a paper artifact or an `EVALUATION.md` number). |
| Operationally exercised | The route ran against a real external target outside this repository's own dev checkout, such as an installed wheel in a clean environment or a real container runtime, with the run's outcome recorded. |

## Present in this tree

| Ecosystem | Status | Evidence |
|---|---|---|
| <picture><source media="(prefers-color-scheme: dark)" srcset="../assets/logos/gymnasium-dark.svg"><img src="../assets/logos/gymnasium.svg" alt="" height="16"></picture> Gymnasium, scalar and vector | Operationally exercised | `check_env.py` and `check_determinism_across_constructors` enforce the Gymnasium contract; `tests/test_gym.py`, `tests/test_vector.py`, `tests/test_integration_parity.py` cover it in CI. The built wheel was additionally installed and stepped in a clean virtual environment outside the checkout (`inventory.md`, version matrix section). |
| <picture><source media="(prefers-color-scheme: dark)" srcset="../assets/logos/pettingzoo-dark.svg"><img src="../assets/logos/pettingzoo.svg" alt="" height="16"></picture> PettingZoo (`pettingzoo_env.py`, `lob_env.py`, `market_env.py`) | Contract-tested | `tests/test_pettingzoo.py`. The import is guarded; the class only exists when `pettingzoo` is installed. |
| <picture><source media="(prefers-color-scheme: dark)" srcset="../assets/logos/minari-dark.svg"><img src="../assets/logos/minari.svg" alt="" height="16"></picture> Minari (`minari_export.py`) | Contract-tested | `tests/test_minari.py` exercises `to_minari`, `to_minari_train_test`, and `seed_band_metadata`. |
| <picture><source media="(prefers-color-scheme: dark)" srcset="../assets/logos/prime-intellect-dark.png"><img src="../assets/logos/prime-intellect.png" alt="" height="11"></picture> `verifiers` / Prime-RL | Workflow-tested | `tests/test_verifiers.py` and `tests/test_verifiers_episode_outcomes.py`, plus the runnable `examples/prime-rl/rl.toml` recipe. `verifiers_env.py` states it was verified against `verifiers` 0.1.14; the CI pin is 0.3.1, and reconciling the two is open (`inventory.md`). |
| <a href="https://modelcontextprotocol.io"><picture><source media="(prefers-color-scheme: dark)" srcset="../assets/logos/mcp-dark.svg"><img src="../assets/logos/mcp.svg" alt="" height="11"></picture></a> MCP (`mcp_server.py`) | Contract-tested | `tests/test_mcp_server.py`. The import is guarded behind `FastMCP is None`. |
| Functional view (`functional.py`, `SharpeArenaFuncEnv`) | Contract-tested | `tests/test_functional.py`. |
| SharpeBench bridge (`bench_bridge.py`) | Contract-tested | Imported and exercised by `tests/test_confidence_pairing.py`, `tests/test_finish_reasons.py`, and `tests/test_local_agents.py`. |
| Integration contracts and parity checker (`integrations.contracts`, `integrations.parity`) | Contract-tested | `tests/test_integration_parity.py` tests the checker's own failure modes: a deliberately broken adapter mapping is caught by name, and the three ways a check could report parity without comparing anything are made to raise. |
| <img src="../assets/logos/stable-baselines3.png" alt="" height="16"> Stable-Baselines3 (`sb3_env.py`, `SharpeArenaSB3VecEnv`) | Workflow-tested | `tests/test_sb3.py` covers the autoreset and terminal-observation mapping (`test_autoreset_is_same_step_and_not_next_step`, which fails if the mapping is inverted), the `TimeLimit.truncated` encoding, engine parity through `integrations.parity`, and a PPO learn-then-evaluate pass on a tiny CPU budget. `test_reference_semantics_still_match_installed_sb3` pins the module against the installed `DummyVecEnv.step_wait` source so an upstream change to the mapping fails a test rather than silently changing what the route computes. Targeted against `stable-baselines3` 2.9.0, the version installed and tested here; the import is guarded and construction raises `SB3Unavailable` without it. `examples/sb3/train_ppo.py` is the runnable recipe. |

The five optional extras that gate these routes are `verifiers`, `minari`, `pettingzoo`,
`mcp`, `sb3` (`crates/sharpearena-py/pyproject.toml`). Nothing above reaches "empirically
evaluated": no paper or `EVALUATION.md` result currently runs through any of these
adapters. The paper's evidence uses the Rust engine and the SharpeBench bridge
directly, not a Gymnasium, PettingZoo, `verifiers`, or Stable-Baselines3 rollout.

## Not present in this tree

| Ecosystem | Status | Where recorded |
|---|---|---|
| CleanRL | Not supported | [CleanRL: dependency ranges don't intersect](#cleanrl-dependency-ranges-dont-intersect), below. |
| <a href="https://www.ray.io"><img src="../assets/logos/ray.svg" alt="" height="14"></a> Ray, RLlib | Not supported | `inventory.md`: no code, only a literature citation in `paper/review/environment-genre-study-2026.md`. |
| <a href="https://pytorch.org/rl"><img src="../assets/logos/torchrl.png" alt="" height="12"></a> TorchRL | Not supported | `inventory.md`: nothing in the tree. |
| <picture><source media="(prefers-color-scheme: dark)" srcset="../assets/logos/hud-dark.svg"><img src="../assets/logos/hud.svg" alt="" height="14"></picture> HUD | Local feasibility only, not a supported integration | [HUD: a local feasibility check, now in the tree](#hud-a-local-feasibility-check-now-in-the-tree), below. |
| <picture><source media="(prefers-color-scheme: dark)" srcset="../assets/logos/harbor-dark.png"><img src="../assets/logos/harbor.png" alt="" height="13"></picture> Harbor | Local feasibility only, not a supported integration | `integrations/harbor/`, recorded in [INT-09](INT-09-harbor-local-feasibility.md). A local job ran end to end under Harbor 0.23.0 on Docker Desktop over WSL2 with a separate-container verifier, and seven of eight tampering fixtures are refused. The evidence covers that one host: it is not evidence against kernel-level container escape, and network egress is untested because `no-network` is unavailable there. Harbor's built-in job aggregation counts a missing or crashed reward as 0, so a custom metric is required before any job-level number means what it appears to. |
| PufferLib | Ruled out | [Ruled out: PufferLib and EnvPool](#ruled-out-pufferlib-and-envpool), below. |
| <img src="../assets/logos/envpool.svg" alt="" height="12"> EnvPool | Ruled out | [Ruled out: PufferLib and EnvPool](#ruled-out-pufferlib-and-envpool), below. |

## Ruled out: PufferLib and EnvPool

Both were evaluated against the native Rust engine and rejected. The evaluation is
`SHARPE-HANDOFF-SUPPORT/product-planning/integration-refs/NATIVE-ACCEL-REFERENCE.md`,
read against the live PufferLib and EnvPool documentation on 2026-09-23.

**PufferLib 5.0.** PufferLib's own FAQ states that its fast path requires the
environment's dynamics to be written in C: "Where did all the Python/third-party
stuff go? It was all 100x+ slower than PufferLib is now." Its documented contract is a
plain C struct and plain C functions. A Rust `staticlib` exporting `extern "C"`
symbols could in principle satisfy that shape, but the route is undocumented,
unsupported, requires editing PufferLib's own `build.sh`, and still requires
flattening this package's nested, variable-length observation into the fixed layout
PufferLib's C side expects, which means rewriting the environment's output contract in
C regardless of the binding mechanism. Reimplementing the trading dynamics is rejected
by this project's own scope, so PufferLib is out.

**EnvPool 1.2.7.** EnvPool's contract is a C++ class deriving from a template
instantiated on a compile-time spec type; Rust cannot implement that directly. The
only route is a hand-written C++ shim class holding an opaque handle to the Rust
engine, calling `extern "C"` entry points into it, and copying results into EnvPool's
pool-allocated arrays. This keeps the finance engine in Rust, unlike the PufferLib
route, so it is feasible under this project's scope rather than rejected outright. It
is not built: the shim does not exist, and building it is out of scope for the current
integration ticket. EnvPool is therefore "no go under current scope," not "unhealthy
upstream" or "infeasible in principle."

Neither vendor's published throughput number (PufferLib's "60,000,000 steps/second";
EnvPool's own benchmark claims) is measured against a comparable workload to this
package's; both figures come from each project's own micro-benchmark harness on its
own reference environments.

## CleanRL: dependency ranges don't intersect

CleanRL pins `gymnasium==0.29.1` and declares `requires-python >=3.8,<3.11`. This
package declares `gymnasium>=1.0` (`crates/sharpearena-py/pyproject.toml`; the CI pin
is `gymnasium==1.3.0`, `crates/sharpearena-py/ci-requirements.txt`). Both pins were
checked directly against each project's published package metadata on 2026-09-23; they
were not restated from memory. `0.29.1` and `>=1.0` do not overlap, so no single
Python environment can install CleanRL and this package's Gymnasium-facing surface
together today. This is a fact about the current dependency sets, not a statement that
either project is unwilling to move; no fix is planned here.

## HUD: a local feasibility check, now in the tree

HUD is not a supported integration. There is no adapter and no import of `hud`
anywhere under `crates/sharpearena-py/python/sharpearena/`. What the tree holds is
the fixture from a local feasibility and isolation-boundary check, merged in PR #102:
`examples/hud/`, which wraps one bounded SharpeArena episode behind three HUD MCP
tools, and `crates/sharpearena-py/tests/test_hud_local.py`. The full record is in
[INT-08](INT-08-hud-local-feasibility.md). Nothing in the package depends on HUD.

What the check established: `hud` 0.6.18 installs and runs locally with no account,
no deployment and no provider call, and all eleven tests pass deterministically. Two
runtimes were exercised, and neither is isolation. `LocalRuntime` shares one Python
interpreter between the grader, the episode state and the test process, with only the
fixture's own tool surface between them. `SubprocessRuntime` adds a process boundary
over loopback TCP but shares the host filesystem and network namespace.
`DockerRuntime`, `ModalRuntime` and `HUDRuntime` were not exercised, because each
needs Docker execution, an account or a deployment.

What it does not cover: container or cloud isolation, the control channel's
authentication surface at the wire level, training-client export, and replay through
the canonical engine or the SharpeBench bridge.

## Harbor: a local feasibility check, now in the tree

Harbor is not a supported integration. The task package from the feasibility and
verifier-boundary check now lives in `integrations/harbor/` (merged in PR #99), and it
is standalone: it does not touch this package's engine, wire contract or SharpeBench
bridge, and nothing in the package depends on Harbor or exposes a Harbor adapter. The
full record is in [INT-09](INT-09-harbor-local-feasibility.md).

What the check established, on one host and no more: a local job ran end to end on Docker Desktop over WSL2 with Harbor
v0.23.0, completing in about 83 seconds with the recorded reward. A separate verifier
container ran, and in the fixtures that tested it, the private evaluator files and
grader were unreachable from the agent's own container. Of eight tampering fixtures,
seven were refused or scored no better than an honest run, four of those by scoring
exactly the honest reward. One fixture (path traversal) was refused by an unhandled
exception during artifact re-upload rather than by the grader's own reason-coded
contract, which that check records as a real gap, not a pass.

That check does not cover native Linux, Windows containers, or any cloud provider; it
is Docker Desktop on WSL2 only. It is not evidence against kernel-level container
escape, since these are ordinary Docker containers with no gVisor, Firecracker, or Kata
isolation. Network egress is untested, because `network_mode = "no-network"` is
unavailable on that host and the task ran at Docker's default network policy. The
installed package's `schema_version` default is `"1.4"`, not the `"1.3"` an earlier
planning reference recorded.

Until that work is merged into this tree, Harbor's status here stays "not
integrated." Treat the paragraph above as a pointer to PR #99, not as documentation of
a capability this package ships.
