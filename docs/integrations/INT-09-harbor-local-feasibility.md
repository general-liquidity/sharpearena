# INT-09: Harbor local feasibility and verifier boundary

Scope per the integration plan, sections 13 and 15: feasibility and the
restricted-evaluation boundary only. This does not build the full task
package the plan describes (private scenario delivery, canonical-engine
replay, the SharpeBench bridge, provider/remote-runtime execution) because
the shared evidence contract those steps depend on is still in flight. No
public Hub submission, no external storage, no credentials, no paid
provider call were made anywhere in this work.

All findings below are from an actual local run on this machine, not from
reading the vendor docs. Where a claim in
`SHARPE-HANDOFF-SUPPORT/product-planning/integration-refs/HUD-HARBOR-REFERENCE.md`
is confirmed or contradicted by that run, this says so explicitly.

## 1. Runtime boundary this evidence covers

- Host: Windows 11 Pro, this repository's worktree.
- Harbor: `pip install harbor` resolved to **v0.23.0**, the version pinned
  in the reference doc's provenance table.
- Container runtime: Docker Desktop 29.7.2, `OSType: linux` (WSL2-backed
  Linux containers, not Windows containers).
- Agent: the built-in **oracle** agent (`solution/solve.sh` / a scripted
  shell command) for the honest fixture, and eight deterministic
  `BaseAgent` subclasses (`integrations/harbor/fixtures/sa_harbor_fixtures`)
  for the tampering fixtures. No model was called anywhere in this session.

Everything below is a property of this Docker-Desktop-on-WSL2 boundary. It
is not evidence about a native Linux host, a Windows-container host, or any
cloud sandbox provider (Modal, Daytona, E2B, GKE, ...), which Harbor's own
docs explicitly decline to make isolation claims about.

## 2. Does Harbor install and run a local job here

Yes, unmodified, no account, no network egress attempted beyond the initial
`pip install` and the base-image pulls already cached from prior Docker use.

```
HARBOR_TELEMETRY=off harbor run -p tasks/sa-replay-feasibility -a oracle
```

produced a completed trial, `verifier/reward.txt` = `1.0`, in ~1m23s. The
task package is four files plus a dedicated verifier image, exactly the
minimum shape the reference doc describes:
`integrations/harbor/tasks/sa-replay-feasibility/{instruction.md,task.toml,environment/Dockerfile,tests/{Dockerfile,test.sh,grade.py}}`.

**`network_mode = "no-network"` does not work on this host.** Setting it and
rerunning raised, at validation time, before any container started:

```
ValueError: network_mode='no-network' is not supported by
EnvironmentType.DOCKER environment. Environment providers must enforce the
requested network policy or reject the task.
```

This is the fail-closed behavior the reference doc credits Harbor with
("Harbor rejects the trial at validation time rather than running with a
weaker policy") and it is real, not asserted. It also means the task's
current `[environment]` section deliberately does **not** set
`network_mode`, and runs at Docker's default (public) network policy for
the agent phase on this host. That is the honest state of this
feasibility check, not an oversight: reaching `no-network` here needs a
Linux host (or WSL2 distro) whose kernel carries the nftables features
Harbor's egress-control sidecar needs, which this Docker Desktop VM does
not expose. A real deployment must either move the job to such a host or
accept public network on the agent phase and rely on the network-level
controls Gordon already runs at the process boundary instead.

Two extra job directories from this `no-network` probe
(`jobs/2026-09-23__14-48-50`, `jobs/2026-09-23__14-49-03`) are local-only;
`integrations/harbor/jobs/.gitignore` excludes all timestamped job output
so a fresh run doesn't collide with what's committed.

## 3. Does the separate verifier run and receive only declared artifacts

Yes, confirmed by direct inspection of eight trials, not by reading the
`environment_mode = "separate"` documentation and trusting it.

`task.toml` sets `[verifier] environment_mode = "separate"` with an empty
`[verifier.environment]`, which per the config model builds the verifier
from `tests/Dockerfile` (`COPY . /tests`) rather than uploading the agent's
`tests/` copy. No `[[artifacts]]` entry is declared; Harbor's own always-on
`/logs/artifacts` convention is the only channel and the grader reads
exactly one path out of it (`ARTIFACT = /logs/artifacts/actions.jsonl`).

Evidence, from `jobs/tampering.yaml` (imports
`sa_harbor_fixtures.agents:*`, run with
`PYTHONPATH=fixtures harbor run -c jobs/tampering.yaml`):

| Fixture | Attempts | Agent-container probe (`tests_dir_exists`, `private_readable`) | Verifier `refusal` code | Reward |
|---|---|---|---|---|
| honest | legitimate 8-step submission | false, false | none (`accepted: true`) | 0.87 |
| forged_reward | writes `/logs/verifier/reward.txt=1.0`, `rewards.json`, a fake `completion.json`; submits no real actions | false, false | `artifact_missing` | 0.0 |
| writable_grader_path | overwrites its own container's `/tests/grade.py` and `/tests/private/future_prices.json`, then submits honest actions | **true, true** (see note below) | none (`accepted: true`) | 0.87 (unaffected) |
| missing_final_action | 8 records, last one is `"buy"` not `"close"` | false, false | `missing_final_action` | 0.0 |
| path_traversal | symlinks `/logs/artifacts/actions.jsonl → /tests/private/future_prices.json`, `.../shadow.jsonl → /etc/shadow`, writes outside `/logs/artifacts` via `../../escape` | false, false | trial crashed before grading (see §5) | ungraded, 0 counted |
| stale_outputs | plants `/logs/verifier/reward.txt=1.0` and a forged `report.json` from an "earlier pass," then submits honest actions | false, false | none (`accepted: true`) | 0.87 (unaffected) |
| cross_task_state | writes `/app/SA_CROSS_TASK_MARKER`, then submits honest actions | false, false | none (`accepted: true`) | 0.87 (unaffected) |
| private_evaluator_read | `cat`s `/tests/private/future_prices.json` and `/tests/grade.py` into its own `/logs/artifacts/{stolen.json,stolen-grader.py}`, then submits honest actions | false, false | none (`accepted: true`) | 0.87 (unaffected); both stolen files are **0 bytes** |

Note on `writable_grader_path`: its probe shows `tests_dir_exists: true`
and echoes back `{returns: [1,1,1,1,1,1,1,1]}` - this is **not** a leak.
That fixture's own script runs `mkdir -p /tests/private && cat >
/tests/grade.py` before probing, i.e. it is reading back the fake files it
just wrote inside its own container. The verifier still scored 0.87 (the
real honest-action result), because the separate verifier never sees that
container at all; it is built fresh from `tests/Dockerfile` in its own
image. All six other fixtures show `tests_dir_exists: false` from a stock
container, which is the actual answer to "can the agent read `/tests` or
the private evaluator file": **no** - the private future-return file used
for scoring was never mounted, copied, or otherwise reachable from the
agent's own container in this configuration.

So: the four fixtures that combine an attack with a legitimate submission
(`writable_grader_path`, `stale_outputs`, `cross_task_state`,
`private_evaluator_read`) all score exactly the honest-action reward,
proving the attack achieved nothing beyond what honest behavior would have
earned. `forged_reward` and `missing_final_action` are refused with a
distinct, attributable code from the grader itself
(`_require_clean_verifier_dir` / step validation in
`tests/grade.py`). `path_traversal` is refused too, but not by the
grader - see below.

## 4. Aggregation: missing rewards read as zero unless you write a custom metric

Confirmed against the reference doc's own claim. Harbor's `harbor.job_finished`
and `harbor.trial_finished` mean-reward computation treats a missing or
`null` reward as `0` in the average. This project's grader already sidesteps
the sharper version of that problem (a crash vs. a deliberate `reward=0.0`)
by writing a `report.json` with an explicit `accepted`/`refusal` field next
to `reward.txt`, so a human or a script reading `verifier/report.json` can
tell "genuinely scored zero" from "never reached the grader" per trial. But
**no custom `metric.py` was written for this task**, so Harbor's own
built-in job-level aggregate still cannot make that distinction - a crashed
trial (see `path_traversal` below) and a trial that honestly earned 0.0 both
count as `0` in `harbor.job_finished`'s mean. For anything beyond this local
feasibility check, aggregation must go through `report.json`'s per-trial
`accepted`/`refusal` field (or a `metric.py` that reads it), never through
`Job.reward` or the printed job-summary table's `Mean` column.

## 5. A real gap the docs don't mention: symlink artifacts crash the trial

`path_traversal`'s attack was refused - it earned nothing - but not the way
the other refusals were. The grader in `tests/grade.py` has an explicit
check (`ARTIFACT.is_symlink()` → `artifact_not_regular_file`) for exactly
this case, and it never ran. Instead, Harbor's own artifact pipeline threw
an uncaught `RuntimeError` while re-uploading the collected artifact
directory from the host into the fresh verifier container:

```
docker compose cp failed; retrying upload with tar stream: ...
GetFileAttributesEx ...\tests\jobs: The system cannot find the file specified.
...
OSError: [Errno 22] Invalid argument: '...\artifacts\logs\artifacts\actions.jsonl'
RuntimeError: Docker compose cp failed, and tar upload fallback also failed.
```

What actually happened, confirmed by inspecting the host-side collected
directory: Harbor's collection step preserved the agent's dangling symlinks
literally on the host filesystem (`actions.jsonl -> /tests/private/future_prices.json`,
`shadow.jsonl -> /etc/shadow`), and the *second* copy - host to the fresh
separate-verifier container - is what broke, because neither `docker compose
cp` nor the tar fallback can transfer a dangling symlink cleanly through
this Windows/WSL2 boundary. The trial ends as an uncaught exception
(`Trials: 0, Exceptions: 1` in the job summary), not the artifact-manifest
"soft failure" the reference doc describes ("A collection failure is
recorded in the manifest without failing the trial"). The manifest in this
trial's directory does record the `/logs/artifacts` directory collection as
`"status": "ok"` - the failure is downstream of that, in the
verifier-side re-upload, which the manifest does not cover.

Net effect for trust: the path-traversal attempt still could not produce a
reward, and could not read anything (`/etc/shadow`'s target was never
resolved through the dangling symlink on this container-to-host boundary).
But it is refused by an infrastructure crash rather than by the grader's
own name-and-reason contract this task otherwise honors, and it currently
takes the whole trial down as an exception rather than leaving a
`report.json` with a `path_traversal_attempt` code. If this fixture must
report failures the same way the others do, `test.sh` (or the grader)
needs to run a symlink/traversal check as the very first thing, before
Harbor's own artifact-collection re-upload has a chance to choke on it -
or the task needs `[[artifacts]]` entries narrow enough that a symlinked
path is excluded from collection in the first place. That is scoped out
of this feasibility pass; recorded here as a real, reproduced finding, not
a hypothetical.

## 6. Reference-doc drift confirmed against the installed package

- **`schema_version`**: the reference doc's "follow the reference, not the
  tutorial" guidance names `"1.3"` as current for Harbor v0.23.0. The
  *installed* v0.23.0 package's `TaskConfig.schema_version` Pydantic
  default is **`"1.4"`**
  (`harbor/models/task/config.py:802`). `task.toml` in this task package
  uses `"1.4"` to match the code that actually validates it, not the
  documentation page, which is itself stale relative to the release it
  claims to describe. `schema_version` is an unconstrained `str` field
  (no enum/`Literal`), so `"1.3"` would also have parsed - this is a
  drift worth flagging, not a task-breaking one.
- **ATIF version**: grepping the installed package's agent trajectory
  writers (`harbor/agents/installed/*.py`) confirms `schema_version="ATIF-v1.7"`
  is what ships for current-generation agents (Claude Code, Codex, Gemini
  CLI, Cursor CLI, Devin, Eve, ...), matching the reference doc's
  instruction to emit `ATIF-v1.7`, not the changelog's `v1.8`. This
  project does not currently emit trajectories of its own; recorded here
  so a future ATIF exporter targets the right version.

## 7. `[verifier.env]` confirmation prompt

Not exercised - this task declares no `[verifier.env]` secrets, so the
prompt never fires in this run. The reference doc is correct that no
documented CI bypass exists for it. **This blocks unattended automation**
for any task that does need verifier-side credentials (a real
SharpeBench-bridge task almost certainly will, even if only for a
model-provider key on the agent side, which is a separate confirmation
path). Flagging as an open blocker for I07/I08 rather than working around
it with an undocumented flag.

## 8. Isolation verdict and residual limits

Within the boundary in §1:

- The separate verifier's grading code and its one private input
  (`tests/private/future_prices.json`) were never present in, copied to,
  or readable from the agent's own container in any of the eight
  fixtures that didn't fabricate the paths themselves. Confirmed by
  direct probe, not by trusting `environment_mode = "separate"`.
- Tampering that only touches the agent's own container (forged reward
  files, a locally rewritten grader, planted stale verifier output, a
  cross-trial marker file, reading files that were never mounted) has
  **zero effect** on the actual reward. The four "attack + honest
  submission" fixtures all score exactly the honest result.
- A dangling-symlink artifact is not silently trusted either, but the
  refusal path for it is an unhandled Harbor-side exception rather than
  the grader's own reason-coded refusal (§5) - a real gap, reproduced
  once, not closed by this task package as it stands.

What this does **not** cover, and must not be read as covering:

- **Container escape.** This ran in ordinary Docker Desktop / WSL2
  containers with no additional hardening (no gVisor, no Firecracker/Kata,
  no seccomp/AppArmor profile beyond Docker's defaults). Containerization
  here is a filesystem and process boundary for a cooperative-but-adversarial
  scripted agent, not a claim about resistance to a kernel-level container
  escape. Neither Harbor's docs nor this feasibility check make or imply
  that stronger claim.
- **Network isolation**, since `no-network` is unavailable on this host
  (§2) - the agent phase ran with Docker's default (public) network
  policy. Nothing here demonstrates egress containment.
- **Any cloud sandbox provider** (Modal, Daytona, E2B, GKE, ...) or a
  native-Linux / Windows-container host - untested, out of scope for this
  evidence.
- **Multi-tenant / concurrent-job isolation** - every run in this session
  was one job at a time, sequentially.

## 9. Replay through the canonical engine / SharpeBench bridge

Not attempted. The plan's own acceptance language for this route requires
the shared evidence contract and the promotion-replay/SharpeBench pieces
named in plan section 15 ("Inspect `promotion_replay.py` and its documented
limits before reuse"), which are explicitly still in flight per this
ticket's scope note. The `actions.jsonl` shape this task's grader consumes
(`step`, `action`, `size`, terminal `close`) is a placeholder deterministic
contract for exercising the verifier boundary, not a claim that it already
maps onto the canonical Rust engine's action space or that captured
trajectories from a real run would replay through it. That mapping is
follow-on work, not part of this feasibility/boundary check.

## Summary

| Question | Answer |
|---|---|
| Harbor installs and runs a local job here | Yes - pip-installed v0.23.0, one Docker Desktop/WSL2 job, no account, no network beyond package/image pulls |
| Toolchain required on Windows | Python 3.12 + `pip install harbor` + Docker Desktop with the WSL2 Linux-container backend; `no-network` mode is unavailable on this host (confirmed, not assumed) |
| Separate verifier runs and receives only declared artifacts | Yes - confirmed by probe evidence across 8 trials, not by documentation alone |
| Private state / future inputs stay outside the agent boundary | Yes, when the agent doesn't fabricate the paths itself (1 of 8 fixtures did, self-evidently, and still scored the honest result) |
| Tampering fixtures refused | 7 of 8 cleanly (distinct reward/refusal code or provably zero-effect); 1 of 8 (`path_traversal`) refused via an unhandled infrastructure exception, not the grader's own contract - a genuine, reproduced gap |
| Replay through canonical engine + SharpeBench bridge | Not attempted - out of scope per this ticket, blocked on the shared evidence contract |
| Isolation claim | Container/process boundary only; explicitly not a claim against kernel-level escape, network egress, or any non-Docker-Desktop provider |
