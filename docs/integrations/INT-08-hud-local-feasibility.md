# INT-08: HUD local feasibility and boundary map

Scope per the integration plan, sections 12 and 15: feasibility and the
restricted-evaluation boundary only. This does not build the full adapter
the plan describes (grader-as-verifier-environment, training-client export,
the SharpeBench bridge, DockerRuntime/ModalRuntime/HUDRuntime execution)
because the shared evidence contract those steps depend on is still in
flight. No account was created, nothing was deployed, no paid provider was
called, and no trace was uploaded anywhere in this work.

All findings below are from an actual local run on this machine, not from
reading the vendor docs. Where a claim in
`SHARPE-HANDOFF-SUPPORT/product-planning/integration-refs/HUD-HARBOR-REFERENCE.md`
is confirmed, contradicted, or extended by that run, this says so
explicitly.

## 1. Runtime boundary this evidence covers

- Host: Windows 11 Pro, this repository's worktree.
- HUD: PyPI package **`hud`**, resolved to **v0.6.18**. `pip install
  hud-python` (the name the reference doc's own example and most existing
  tutorials still use) fails outright: PyPI's `hud-python` build backend
  now refuses to install and prints its own migration notice, `'hud-python'
  is now 'hud'`. `pip install hud` is the correct, current command; the
  reference doc's provenance table cites `0.6.13` as an example value, so
  `0.6.18` is newer than anything quoted there.
- A real installation obstacle, reproduced once: installing `hud` into a
  virtualenv created under this session's default deeply nested scratch
  path failed with `OSError: [Errno 2] No such file or directory` on one
  of `anthropic`'s generated type files
  (`beta_managed_agents_self_hosted_resources_unsupported_deployment_paused_reason_error.py`),
  with pip's own hint pointing at Windows long-path support. Moving the
  venv to a short path (`C:\hudvenv`) fixed it immediately, same package,
  same versions. `hud`'s dependency tree (`anthropic`, `fastmcp`, `mcp`,
  `google-genai`, `httpx`/`httpx2`, ...) generates file paths long enough
  to hit `MAX_PATH` on a Windows host without long-path support enabled,
  independent of anything HUD does. Worth flagging for CI: a nested
  worktree/venv path (this repo's `wt-*` worktree convention plus a deep
  scratch directory) can reproduce this.
- Agent: five deterministic `hud.agents.base.Agent` subclasses
  (`examples/hud/agents.py`) driving a published `market` MCP capability.
  No model was called anywhere in this session.

Everything below is a property of this Windows 11 / Python 3.12 / `hud`
0.6.18 boundary. It is not evidence about a Linux host (where HUD's `bwrap`
sandbox activates), a Docker/Modal/HUD-hosted runtime, or a cloud provider,
none of which were exercised here.

## 2. Does HUD install and run a local job here, no account, no key

Yes. `pip install hud` (0.6.18, short-path venv) is the only external call
in this session besides the one-time PyPI resolution itself. With
`HUD_TELEMETRY_ENABLED=false`, `HUD_CLI_ANALYTICS_ENABLED=false`,
`HUD_FILE_TRACKING_ENABLED=false` set and no `HUD_API_KEY` in the
environment (confirmed absent, not merely unset by the test - the fixture
pops it explicitly), 11 of 11 tests in
`crates/sharpearena-py/tests/test_hud_local.py` pass end to end:

```
11 passed in 40.31s
```

covering a completed episode, an agent that lies about its own outcome
(`agent_report` is carried through as diagnostic only, never scored), a
malformed decision that is refused without burning a bar, a duplicate
submission, an early exit that takes the floor rather than a favorable
prefix, a `rollout_timeout` that produces a distinct `is_error` run rather
than a zero folded into the mean, `SubprocessRuntime` serving the same
environment out of process, a two-scenario `Taskset`, repeated rollouts of
one scenario for determinism (`group=2`, identical return series both
times), and an interrupted-export test proving a crash mid-write never
replaces or truncates a prior evidence export (temp-file-then-`replace`
pattern in `examples/hud/grading.py`).

Confirmed locally, matching `~/.hud/spans` under `os.environ` inspection:
spans are written to disk (`~/.hud/spans/*.jsonl`) as the reference doc
describes even with telemetry disabled, but with no `HUD_API_KEY` present
there is no code path that uploads them - nothing left this machine.

## 3. Runtimes usable locally with no account, deployment, or provider call

| Runtime | Exercised | Result |
|---|---|---|
| `LocalRuntime` | Yes | Works. Serves the fixture's custom `market` MCP capability over `http://127.0.0.1:<ephemeral-port>/mcp` in the same Python process as the test runner. |
| `SubprocessRuntime` | Yes | Works. `run.runtime` reads `tcp://127.0.0.1:58321`-shaped - a real separate-process, loopback-only control channel, confirmed by inspecting the field, not assumed from the docstring. |
| `DockerRuntime` | No | Out of scope per this ticket's hard rules (no `service_access=True` under any circumstances; Docker execution itself was not attempted this pass). |
| `ModalRuntime` | No | Requires a Modal account - out of scope (no account creation). |
| `HUDRuntime` | No | Requires a deployed environment on hud.ai - out of scope (no deployment). |

Both exercised runtimes needed no account, no deployment, and no provider
call. `LocalRuntime`/`SubprocessRuntime` are the only two the plan's own
acceptance language ("Local L2 completion does not claim HUD-hosted
deployment... those require their own runs and approvals") permits at this
stage, and both are confirmed working.

## 4. Deterministic no-model-call fixture: runs end to end

Yes, all 11 cases. The fixture (`examples/hud/{env.py,episode.py,agents.py,grading.py}`,
test at `crates/sharpearena-py/tests/test_hud_local.py`) wraps one bounded
SharpeArena episode as a HUD `@env.template`. The engine, the scenario seed,
and the mandate live inside `episode.py`'s private `Episode` object; the
agent reaches it only through three MCP tools (`spec`, `observe`,
`submit_decision`) published by a `FastMCP` server the environment starts
on `env.initialize`. Grading (`grading.py:grade_episode`) reads the
episode's own recorded `returns`/`attempts`/`events`, never the agent's
self-report, and calls the real `sharpearena.sharpearena_py.score_run` /
`episode_outcomes.is_process_block` / `mandate.mandate_breach` - the actual
scoring APIs, not a stand-in. This deliberately avoids the two hardest
documented HUD-on-Windows traps: `LLMJudgeGrader` (network call by
construction - not used, grading is plain Python) and `BashGrader` (scores
`0.0` on native Windows with `/bin/bash not found` - not used either, for
the same reason). No test in this suite is an all-zero grading result
standing in for a real pass.

Reruns in this session were mostly 15-20s; one back-to-back rerun took over
90s and was killed by the outer timeout, with no error captured (the
following rerun passed clean at 19s). That single slow run is recorded here
rather than dropped; it did not reproduce on either the run before or the
two runs after it, and is most plausibly leftover OS socket state from
several manual reproduction scripts run in quick succession against the
same loopback ports immediately before it, not a fixture-level hang. It is
flagged as an open question rather than resolved.

## 5. A real bug this exercise found and fixed, matching a documented hazard

The `Taskset` test (`test_two_scenarios_run_as_one_taskset_without_crossing_state`)
originally hung indefinitely on this host - confirmed by running it in
isolation under a 60-second `pytest-timeout` kill, twice, with an
`asyncio`/`ProactorEventLoop` stack trace and `ERROR: ASGI callable
returned without completing response.` on stderr each time. This is not a
flake: every isolated rerun reproduced it.

Root cause, isolated by direct experiment outside pytest: the two `Task`
rows in the original fixture did not set `slug`. The reference doc warns
about exactly this - `"slug is the only identifier HUD calls stable... Set
slug explicitly on every generated task row. Without it, Job.results cannot
key runs back to tasks"` - but understates the failure mode on this host:
it is not a silent positional-zip degradation, it is a deadlock. Minimal
reproduction:

- Two unslugged `Task` rows in one `Taskset.run(...)`, `LocalRuntime`,
  `max_concurrent=1`: **hangs**, killed after 60s.
- Same two tasks with explicit `slug="calm-a"` / `slug="fat-a"`,
  `max_concurrent=1`: **completes** (each rollout binds a distinct
  loopback port, `51005`/`51006`, confirmed in the log).
- Same two tasks with explicit slugs at default concurrency: **completes**,
  `job.results.keys() == {"calm-a", "calm-b"}`.

Fixed in this pass: `crates/sharpearena-py/tests/test_hud_local.py`'s
`_task()` helper now sets `slug=scenario` on every minted `Task`, per the
reference doc's own recommendation. This is a one-line correction to the
feasibility fixture itself, not adapter work.

A second, unrelated bug surfaced by exercising the two-scenario path at
all: `examples/hud/episode.py`'s `SCENARIO_SEEDS["fat-a"]` declared
`distribution_mode: "fat"`, which is not a value the Rust engine accepts
(`InvalidArgument: unknown distribution_mode "fat" (expected calm | hard |
extreme | cointegrated_pairs | regime_shift)` - confirmed via
`crates/sharpearena/src/scenario_gen.rs`, where `Hard` is the fat-tail
amplification tier). Fixed by changing it to `"hard"`. Neither bug was
reachable through the 7 single-task tests that were run and passing before
this pass; both needed the multi-task path actually exercised to surface.

## 6. Isolation verdict per runtime, with residual limits

- **`LocalRuntime`**: **no isolation**, confirmed structurally, not just
  documented. The grader (`grading.py`), the private `Episode` state
  (scenario seed, mandate, future bars), and the test process share one
  Python interpreter. The only boundary between the agent double and that
  state is the fixture's own MCP tool surface (`spec`/`observe`/
  `submit_decision`) - a cooperative API contract, not a security boundary.
  This fixture does not use `env.workspace()` / the `ssh` capability at
  all, so the `bwrap`-isolated shell the reference doc describes as "the
  only agent-facing isolation primitive HUD owns" never enters the
  picture either way; a real HUD adapter that did use `env.workspace()`
  would additionally need to confirm the `bwrap` sandbox is Linux-only - 
  per `/v6/reference/capabilities`, quoted in the reference doc: *"Sandbox
  isolation (`bwrap`) is Linux-only - unisolated elsewhere, isolated in a
  built image."* On this Windows host, any HUD `ssh`/workspace capability
  is unconfined by construction; this fixture sidesteps that entirely by
  not exposing a shell to the agent at all, only three typed tools.
- **`SubprocessRuntime`**: process boundary only, confirmed via
  `run.runtime == "tcp://127.0.0.1:<port>"` - a real separate OS process
  reached over loopback TCP, not an in-process call. It shares the host
  filesystem and network namespace; this is not a filesystem or network
  boundary, matching the reference doc's characterization exactly.
- **`DockerRuntime`, `ModalRuntime`, `HUDRuntime`**: not exercised. Per the
  non-negotiable constraints for this ticket, `DockerRuntime` was never run
  with `service_access=True` - the reference doc is explicit that this
  setting mounts the **host's own Docker daemon** into the container on
  local `DockerRuntime`, handing code in `main` control of every container
  on the machine, and that flag was never set anywhere in this session.
  `DockerRuntime` without that flag runs, per the reference doc, with a
  documented **default-allow seccomp profile** with
  `systempaths=unconfined` so its inner `Workspace` sandbox can nest - 
  this is a stated relaxation of Docker's own hardening, not a hardening,
  and this pass did not attempt to verify it further since running any
  `DockerRuntime` fixture was out of scope. `ModalRuntime` and
  `HUDRuntime` were not exercised (no account, no deployment).
- **Control channel**: confirmed bound to loopback only
  (`127.0.0.1:<ephemeral>`), never `0.0.0.0`, in every run this pass
  performed. The reference doc's finding that this channel has no
  documented authentication was not independently re-derived (that would
  require probing the raw TCP protocol, out of scope for a feasibility
  pass) but nothing here contradicts it, and every fixture in this repo
  binds loopback by construction (`socket.bind(("127.0.0.1", 0))` in
  `examples/hud/episode.py`'s `serve_market`), never an explicit host
  override.

## 7. HUD run/trace identity mapped to our run and attempt identities

Confirmed directly from a live local run (`Task.slug="calm-a"`,
`LocalRuntime`, `CompletingAgent`), not from the docs alone:

| HUD field | Observed value this pass | Maps to |
|---|---|---|
| `Task.slug` | `"calm-a"` | Our scenario/episode-template id. Stable and must be set explicitly - see §5. |
| `Run.trace.trace_id` | `"3d009f08fe3a4bab8a46173ff6338f61"` (fresh UUID-hex per rollout) | Our per-attempt id. |
| `Run.job_id` | `"677cff0f61414d0987c615920a820271"` | Our run-batch id. Locally generated, not `None`/undocumented as the reference doc worried might be the case for a fully offline run - confirmed to be a real value with no platform round trip. |
| `Run.group_id` | `"58697fdc1ba84830bcc2da7945b8a2ca"` | GRPO/repeat-group id, distinct from `job_id`; relevant only if group rollouts (`group=N`) are used. |
| `Run.slug` | `"calm-a"` (echoes `Task.slug`) | The join key `Job.results` uses to key runs back to tasks without positional zip - confirmed by the §5 experiment. |
| `Job.id` | same as `run.job_id` above | Batch identity for a whole `Task.run`/`Taskset.run` call. |
| `Job.taskset_id` | `None` (single `Task.run`, not a synced `Taskset.from_api`) | `None` locally for anything that isn't a platform-synced taskset; expected per the reference doc. |
| `Run.trace.status` | `"completed"` | Terminal status; also seen `"error"` (rollout-timeout fixture) in this pass. |
| `Run.trace.stop_reason` | `None` on a clean completion | Would read `"timeout"`/`"max_steps"`/etc. on a truncated run; confirmed via the rollout-timeout fixture (`run.trace.stop_reason == "timeout"`, `run.trace.is_error == True`, `job.errors == [run]`). |
| `Run.grade.is_error` | `False` on a clean run | Confirmed present and independently readable from `run.trace.is_error`, per the reference doc's warning to aggregate from these two fields and never from `Job.reward`. |

The round trip this pass actually tested: a crashed/timed-out rollout comes
back as a distinct errored `Run` inside `job.errors`, excluded from
`job.runs`' normal completion path, and `job.reward`'s mean would silently
fold a genuine zero-score run and a never-graded errored run together if
read naively - the fixture's own `test_rollout_timeout_is_an_error_not_a_zero_score`
asserts `job.errors == [run]` specifically to keep that distinction
explicit, per the reference doc's instruction to aggregate on
`run.trace.status` / `run.grade.is_error`, never `Job.reward`.

## 8. What this does not cover

- **Container or cloud isolation.** No `DockerRuntime`, `ModalRuntime`, or
  `HUDRuntime` fixture was run. The seccomp/`service_access` findings in
  §6 are read from the reference doc's already-quoted vendor lines, not
  independently re-verified against a running container this pass.
- **The control channel's actual authentication surface.** Loopback
  binding was confirmed; the absence of an auth handshake was not
  independently probed at the wire level.
- **Training-client export** (I06 implementation item 8 - group/task
  identities, response/action boundaries, policy-likelihood masks). Not
  attempted; explicitly out of scope pending the shared evidence contract.
- **Replay through the canonical engine / SharpeBench bridge.** Not
  attempted, for the same reason as INT-09's equivalent note: the
  promotion-replay and bridge pieces this would depend on are still in
  flight.
- **Harbor's HUD adapter** (`hud/integrations/harbor/`). Not exercised;
  the reference doc's own read is that building the Harbor task and the
  HUD task natively is the lower-risk path than generating one from the
  other, and this pass did not need to touch it.

## Summary

| Question | Answer |
|---|---|
| SDK version that installs cleanly | `hud` 0.6.18 (not `hud-python`, which PyPI now refuses to build under that name). Requires a short filesystem path on Windows - a deeply nested venv path reproduced a real `OSError`/long-path failure once, unrelated to HUD's own code. |
| Runtimes usable locally, no account/deployment/provider call | `LocalRuntime` and `SubprocessRuntime`, both confirmed working. `DockerRuntime`, `ModalRuntime`, `HUDRuntime` not exercised (each needs Docker execution, an account, or a deployment - out of scope). |
| Deterministic no-model-call fixture runs end to end | Yes - 11/11 tests pass, covering completion, lying-agent immunity, malformed decision, duplicate submission, early exit, rollout timeout, subprocess runtime, multi-task taskset, repeat-group determinism, and interrupted-export durability. Avoids both documented Windows traps (`LLMJudgeGrader`'s network call, `BashGrader`'s native-Windows all-zero result) by construction. |
| Isolation verdict | `LocalRuntime`: none - grader and private episode state share the test process; this fixture never opens the `bwrap`-gated shell capability at all. `SubprocessRuntime`: process boundary only, confirmed via a real loopback TCP `run.runtime` value; not a filesystem or network boundary. Docker/Modal/HUD-hosted: unevaluated. |
| HUD run/trace identity vs. our identities | `Task.slug` ↔ scenario id (must be set explicitly - its absence deadlocked a multi-task `Taskset` on this host, not merely degraded to positional zip); `Run.trace.trace_id` ↔ attempt id; `Run.job_id`/`Job.id` ↔ run-batch id; `Run.slug` is the actual `Job.results` join key. Aggregate always from `run.trace.status`/`run.grade.is_error`, never `Job.reward`. |
| Blocker found | A real, reproducible deadlock in this repo's own fixture (missing `Task.slug`), not an SDK-side blocker - fixed in this pass, all 11 tests now pass. No blocker prevented a working local route. |
