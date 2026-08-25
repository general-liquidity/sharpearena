# Local Agent Sandboxes and Trading-Agent Environments: 2026 Architecture Review

**Evidence current through 2026-08-26**
**Scope:** local execution on the current Windows 11 / WSL2 workstation; SharpeArena as the environment and SharpeBench as the scoring and evidence consumer; no cloud deployment; no real-capital order execution.

## Executive decision

The products do not need a new monolithic “sandbox platform.” They need a layered local execution architecture with different boundaries for different kinds of entrants:

1. **Data-only local models:** keep the deterministic SharpeArena environment and evaluator in a trusted supervisor; keep Ollama or another model server in a separate trusted host process with GPU access; accept only the canonical, schema-constrained `Decision` object. A model completion is data, not executable code.
2. **Third-party agent executables:** use the existing digest-pinned SharpeBench Docker launcher, which already removes networking, capabilities, host mounts, device access, and writable root state and applies CPU, memory, PID, file-descriptor, user, and tmpfs limits. Strengthen it with a required live negative-test suite and bounded diagnostics before calling the field ready.
3. **Generated strategies:** keep a closed, validated strategy DSL as the default. If arbitrary generated code is later admitted, execute each candidate in a fresh no-network Docker or Wasm/WASI boundary; never execute generated Python or shell on the host.
4. **Forward paper trading:** run it as a separate, explicitly non-replayable evidence class. The agent never receives a broker credential. A trusted supervisor validates the decision, applies the native risk gate, and alone calls a paper-only broker adapter through an allowlisted egress proxy.

This preserves the right product relationship without creating a cyclic library dependency:

- **SharpeArena owns the environment contract:** observations, decisions, transitions, rewards, seeds, reset/snapshot semantics, data-access rules, multi-agent mechanics, and local execution profiles.
- **SharpeBench owns evaluation:** field composition, trial accounting, pass\(^{k}\), deflation, process and cost gates, evidence records, attestations, and leaderboard publication.
- Their integration boundary is a **versioned environment/trajectory artifact**, not mutual imports. SharpeArena can depend on shared protocol/statistics crates; SharpeBench consumes signed or hashed run artifacts. Each package remains independently testable and releasable.

The current workstation is sufficient for this architecture. Its NVIDIA RTX A4000 has 16,376 MiB VRAM; model inference should remain outside the untrusted container so the GPU driver is not added to that container’s attack surface. Model caches and immutable images may live on `D:` if space on `C:` becomes tight, but sandboxes must receive only a narrow per-run workspace—not a bind mount of the drive.

## 1. Research question and evidence standard

This review asks four questions:

1. Which local isolation boundary is appropriate for each SharpeArena entrant type?
2. Which controls must surround the boundary to address credentials, network and data egress, resource denial, persistence, model/GPU access, and evidence integrity?
3. Do SharpeArena’s environment APIs satisfy current agent/RL environment practice, and what gaps would bias the first model field?
4. How should SharpeArena and SharpeBench compose without conflating environment determinism with agent determinism or retrospective evidence with forward paper trading?

Sources are weighted as follows:

- **Tier 1 — normative or implementation-primary:** Linux kernel, Docker, NVIDIA, gVisor, Firecracker, Wasmtime/WASI, and Farama documentation; inspected source and tests in these repositories.
- **Tier 2 — research evidence:** peer-reviewed work where available; 2026 preprints are labeled unreviewed and are not treated as settled security results.
- **Tier 3 — vendor engineering reports:** useful for architectures and operational lessons, but product claims and internal metrics are not independent evidence.
- **Tier 4 — practitioner and secondary essays:** useful for failure cases and design vocabulary, not proof of isolation.

The user-supplied articles are all included in the annotated source ledger. Absolute claims such as “complete safety” are not adopted. A security boundary is judged by its mechanism, tested configuration, and threat model—not by the word *sandbox*.

## 2. Current system and product inspection

### 2.1 Local host

The inspection was performed from WSL2 on Windows 11:

| Component | Observed state | Architectural consequence |
|---|---|---|
| WSL2 kernel | `5.15.167.4-microsoft-standard-WSL2`, x86_64 | Linux containers and `/dev/kvm` are possible, but later Landlock network mediation is not available on this kernel. |
| GPU | NVIDIA RTX A4000, 16,376 MiB, Ampere compute capability 8.6 | Suitable for local quantized inference; keep it in the trusted model service, not the entrant container. |
| Docker Desktop | Installed; Windows CLI at `C:\Program Files\Docker\Docker\resources\bin\docker.exe` | At audit time the Linux engine was stopped/unreachable and `docker` was not on the WSL `PATH`. This is a transient readiness failure, not an architectural absence. |
| KVM | `/dev/kvm` present in WSL2 | Firecracker experimentation may be possible, but `/dev/kvm` alone does not establish a supported, hardened production setup inside WSL2. |
| Storage | `C:` workspace; `D:` available if needed | Put large model/image caches on `D:`; never expose a whole host drive to an entrant. |

Before any field run, start Docker Desktop, enable its WSL integration, make the CLI visible in WSL, and record `docker info` including security options. The field runner should refuse to start unless a live sandbox probe passes. A CI test that skips when Docker is unavailable proves only that the code can skip.

### 2.2 Existing SharpeBench sandbox

`sharpebench/crates/sharpebench-arena/src/sandbox.rs` already launches untrusted entrants using:

- digest-pinned images by default (`repository@sha256:<digest>`), `--pull never`;
- `--network none`, `--ipc none`, no host bind mounts, no Docker socket, and no GPU/device flag;
- a read-only root filesystem;
- `--cap-drop ALL` and `--security-opt no-new-privileges=true` while retaining Docker’s default seccomp profile;
- UID/GID `65532:65532`;
- 1 GiB memory and swap total, 1 CPU, 128 PIDs, and 256 file descriptors;
- `noexec,nosuid,nodev` tmpfs mounts at `/tmp` and `/run` with size ceilings;
- `--init`, `--rm`, stdin/stdout protocol transport, and no Docker log driver;
- a hard refusal when Docker is unavailable, except for an explicit owner-authored local-development opt-in.

That is a strong local single-tenant baseline. It aligns with Docker’s official guidance to combine namespaces, cgroups, a non-root identity, a default seccomp allowlist, capability removal, and guarded daemon access ([Docker Engine security](https://docs.docker.com/engine/security/), [Docker seccomp](https://docs.docker.com/engine/security/seccomp/), [resource constraints](https://docs.docker.com/engine/containers/resource_constraints/)). It is not a microVM: an ordinary Docker Desktop `docker run` container shares the Linux kernel of Docker Desktop’s VM with other Linux containers.

Important remaining gaps are operational rather than a missing container boundary:

1. The live Docker smoke test skips if Docker is absent. A release can therefore be green without ever exercising the actual isolation path.
2. There is no hostile-fixture suite demonstrating that host files, Docker control, devices, loopback services, and network endpoints are unreachable.
3. `--log-driver none` prevents unbounded container logging, but also removes useful forensics. The host transport should retain a bounded, redacted stderr/error record.
4. Image digest pinning identifies bytes, not trust. The image build recipe, base-image provenance, SBOM, and vulnerability status should be recorded.
5. A container restart/reset policy must be tied to evaluation independence. Reusing a stateful agent process across cells can leak memory between seeds even if the environment resets correctly.

### 2.3 Existing SharpeArena environment

SharpeArena already implements most of the environment substrate a frontier local-agent field needs:

- a deterministic Rust transition/reward engine with thin Python adapters;
- a canonical closed-object `Decision{orders, reasoning, cost}` wire contract;
- strict Python parsing that rejects malformed JSON, legacy dialects, unknown symbols, duplicate orders, and out-of-range weights instead of converting failures to a flat position;
- Gymnasium `Env`, native vector stepping, and a `SharpeArenaVectorEnv` with declared autoreset modes;
- PettingZoo parallel multi-agent environments and API conformance tests;
- Minari export and environment recovery;
- disjoint training/evaluation seed bands, sealed evaluation seeds, provenance manifests, and deterministic golden traces;
- native clone/restore plus replay-based checkpointing;
- external-process and HTTP transports with timeouts, circuit breaking, and explicit health/error state;
- Docker execution of untrusted entrant images;
- retrospective historical/generated environments and a separately framed paper-trading path.

The scan found one integration inconsistency and the implementation pass closed it. The strict `decision_parser.py` already rejected unknown and duplicate symbols, while `mcp_server.py::_decision_to_weights` ignored unknown symbols and let later duplicate orders overwrite earlier ones. Because schema validation cannot know the current episode's symbol set, a syntactically valid but semantically invalid MCP decision could silently become a different action. The MCP path now calls the same canonical fail-closed parser as the local-model and `verifiers` paths; an invalid decision returns a typed error with `environment_advanced: false`. Focused regression tests cover unknown symbols, duplicates and action/weight inconsistency. The remaining acceptance question is runtime isolation, not parser equivalence.

## 3. Threat model

### 3.1 Workload classes

| Class | Entrant | Main risk | Required boundary |
|---|---|---|---|
| T0 | Local model produces constrained `Decision` data; no code/tool execution | malformed or adversarial output, prompt injection through observation text, inference nondeterminism, host client accidentally carrying secrets or broad egress | trusted supervisor + strict schema/semantic validation; model service isolated from evaluation secrets |
| T1 | Owner-authored local agent process | accidental file/network damage, dependency compromise, loops/resource exhaustion | hardened container by default; narrowly audited host mode only for development |
| T2 | Third-party image or model-generated executable code | intentional exfiltration, host mutation, kernel/runtime escape attempts, resource denial, persistence, evidence tampering | hardened fresh container or Wasm/WASI; no network, mounts, secrets, socket, devices, or GPU; hostile-fixture testing |
| T3 | Hostile multi-tenant/cloud workload | cross-tenant compromise, kernel zero-days, side channels, operator separation | microVM or equivalently strong dedicated isolation; outside this local single-user deployment |

### 3.2 Protected assets

- Windows and WSL host files on `C:` and `D:`;
- Git, GitHub, package registry, broker, exchange, market-data, and model-service credentials;
- Docker/WSL control sockets and privileged host APIs;
- sealed evaluation salts, unrevealed forward-window data, and hidden test cases;
- model weights and local inference endpoint;
- GPU device/driver and other host devices;
- evidence rows, traces, scores, provenance manifests, and release attestations;
- the availability of the workstation.

### 3.3 Attack surfaces

The most likely failures are configuration and authority leaks, not cinematic kernel escapes: writable host mounts, Docker socket exposure, inherited environment variables, broad egress, shared caches, permissive host functions, stale process state, and a supervisor that accepts malformed output. Kernel/runtime vulnerabilities, dependency supply-chain attacks, PID/memory/disk exhaustion, terminal-control output, and GPU-driver attack surface remain in scope.

The security goal is to prevent entrants from reading or modifying host state, credentials, hidden evaluation material, other cells, or evidence; prevent uncontrolled egress and resource denial; and preserve a truthful record of faults. It does **not** defend against a hostile host administrator, physical access, compromised Windows/WSL/Docker base systems, or all hardware side channels.

## 4. Boundary comparison

| Boundary | Isolation mechanism | Reset/snapshot | Network and secrets | Resource controls | GPU implication | Best use here | Key limitation |
|---|---|---|---|---|---|---|---|
| Hardened Docker | Linux namespaces/cgroups, seccomp, capabilities, LSMs; shares Docker VM’s Linux kernel | fresh image + ephemeral writable layers; `--rm`; volumes only if explicitly added | `--network none`; no secret/env/mount by default | mature CPU, memory, PID, file, tmpfs controls | no device unless explicitly exposed | default for T1/T2 local executable entrants and arbitrary generated code | shared kernel; daemon/socket are highly privileged; configuration is the real boundary |
| Rootless Docker Engine | daemon and containers run in a user namespace without host root | same container lifecycle | same policies; daemon compromise has reduced host privilege | supports cgroups when host is configured; verify actual enforcement | rootless GPU support can add operational complexity; unnecessary here | optional Linux/WSL hardening if validated | not the same as merely setting container UID; Docker Desktop integration and cgroup behavior must be tested |
| gVisor (`runsc`) | user-space Sentry intercepts guest system API and reduces direct host-kernel surface | OCI/container lifecycle | still relies on container network and secret policy | relies on host cgroups for resource control | GPU/application compatibility is narrower; keep inference outside | optional stronger T2 backend on supported Linux after conformance/throughput bakeoff | syscall compatibility and performance; not protection from every side channel or host policy error |
| Firecracker microVM | KVM hardware virtualization, minimal VMM; jailer adds chroot, UID/GID, cgroups and privilege reduction | fast memory/device/disk snapshots with copy-on-write restore | distinct guest kernel; still needs explicit network/secret policy | jailer/host cgroups and host hardening | GPU passthrough is not the intended simple path | future T3/high-threat service on a dedicated Linux host | substantially more operations; snapshot state is sensitive; WSL2 presence of `/dev/kvm` is not production validation |
| Wasm/WASI | memory/control-flow isolation plus explicit imports/capability-oriented host interfaces | instantiate fresh module cheaply; snapshot at application layer | no ambient filesystem/network authority unless host grants it | fuel/epoch interruption and host quotas; runtime-specific | no raw GPU unless a host API grants it | generated strategy DSL/plugins and portable deterministic tools | cannot transparently run arbitrary Python/shell; overbroad host functions defeat the model |
| Landlock / `nono` | unprivileged process-tree restriction of filesystem and, on newer ABIs, network access | process restart; no VM/container filesystem snapshot | policy denies ambient paths/network; proxy can mediate credentials | Landlock is not a CPU/memory/PID quota system | host process retains only devices explicitly reachable | defense-in-depth for trusted local tools/supervisor on supported kernels | shared kernel; feature support depends on kernel ABI; current WSL2 kernel is too old for Landlock network mediation |
| Git worktree / separate OS user | repository/lifecycle separation, optionally Unix DAC | delete/recreate worktree | no intrinsic egress or secret isolation | no intrinsic quotas | unchanged | concurrent development and human review | **not a security sandbox**; writable shared repository/credentials remain reachable unless separately restricted |

### 4.1 Docker and rootless mode

Docker’s default seccomp profile is an allowlist that blocks roughly 44 of more than 300 Linux syscalls and is explicitly described as “moderately protective”; it should not be disabled ([Docker seccomp](https://docs.docker.com/engine/security/seccomp/)). Seccomp alone is not enough. SharpeBench’s capability drop, `no-new-privileges`, read-only root, non-root UID, no network, no mounts, and cgroup limits are the important composition.

Rootless mode runs both the daemon and containers without host root inside a user namespace; that is stronger than setting `--user` inside a rootful daemon and differs from `userns-remap`, where the daemon remains root ([Docker rootless mode](https://docs.docker.com/engine/security/rootless/)). It is worth evaluating if the project moves to a native Linux/WSL Docker Engine. It is not necessary for the first field because Docker Desktop is already the installed runtime, the entrant receives no Docker socket, and the model/GPU stays outside. Record rather than assume whether CPU/memory/PID controls work in the chosen rootless/WSL configuration.

Docker Desktop’s newer **Docker Sandboxes** product is a different path from ordinary `docker run`. Docker reports a dedicated microVM per sandbox on macOS/Windows, network policies, and synchronized workspaces, while the Linux implementation uses a legacy container; the feature was still evolving when announced ([Docker vendor engineering report](https://www.docker.com/blog/building-ai-teams-docker-sandboxes-agent/)). SharpeBench’s current Rust launcher invokes ordinary `docker run` and gains none of that product’s per-sandbox-microVM claims. Do not describe the current implementation as a Docker Sandbox or microVM.

### 4.2 gVisor

gVisor’s Sentry implements the application-facing Linux system API in userspace and aims to minimize the host system API available to a workload. It does not simply pass every guest syscall to the host. It still depends on host cgroups and container network policy, and its own security guide does not claim to solve all side channels ([gVisor security model](https://gvisor.dev/docs/architecture_guide/security/)). Docker can select it with the `runsc` OCI runtime ([gVisor Docker quick start](https://gvisor.dev/docs/user_guide/quick_start/docker/)).

This is the most plausible stronger local backend for third-party images, but not the default until the actual SharpeArena agent image passes: protocol conformance, subprocess/stdio behavior, dependency imports, timeouts, filesystem expectations, throughput, and hostile-fixture tests. A runtime label is not a substitute for that bakeoff.

### 4.3 Firecracker

Firecracker supplies a much stronger kernel boundary, but safe operation includes the `jailer` or an equivalent: close inherited file descriptors and environment, use chroot, unique UID/GID, cgroups, resource limits, and a hardened host ([Firecracker jailer](https://github.com/firecracker-microvm/firecracker/blob/main/docs/jailer.md), [production host setup](https://github.com/firecracker-microvm/firecracker/blob/main/docs/prod-host-setup.md)). Guest inputs remain part of the operator’s trust decision.

Snapshots improve reset speed, but they are not innocuous cache files. Firecracker documents memory/state/disk restore, copy-on-write memory mapping, the need to authenticate/encrypt snapshot files, and the risk of cloned unique state or network identity; snapshot CRCs only detect accidental corruption ([Firecracker snapshot support](https://github.com/firecracker-microvm/firecracker/blob/main/docs/snapshotting/snapshot-support.md)). Never capture a base snapshot after injecting a credential, evaluation salt, or persistent model session.

Although `/dev/kvm` exists in the current WSL2 instance, Firecracker should be a future dedicated-Linux backend for T3 or deliberately hostile entrant research, not the first Windows-local field. Its operating burden would add little to data-only model decisions and duplicate a working Docker boundary for ordinary entrants.

### 4.4 Wasm/WASI

WebAssembly modules can access external resources only through imports the host provides; Wasmtime describes isolation of linear memory and control flow and a capability-oriented WASI filesystem model ([Wasmtime security](https://docs.wasmtime.dev/security.html)). WASI 0.3 describes no ambient authority and explicit host grants ([WASI](https://wasi.dev/)).

That is a good fit for generated indicators, transforms, or a restricted strategy runtime. It is not a transparent sandbox for arbitrary Python packages or shell tools. The host ABI must be tiny: market observation in, target decision out, deterministic clock/RNG if needed, bounded memory/fuel, and no general filesystem/network/process imports. Even text output must be treated as untrusted; Wasmtime’s security guidance calls out terminal escape/control sequences, so evidence viewers should encode or sanitize them.

### 4.5 Landlock and `nono`

Landlock lets an unprivileged process restrict its own process tree’s ambient filesystem and, in newer ABI versions, network rights ([Linux Landlock userspace API](https://docs.kernel.org/userspace-api/landlock.html)). `nono` packages this pattern into profiles and path/network rules, using Landlock on Linux and Seatbelt on macOS, with irreversible restriction and canonicalized paths ([`nono` implementation documentation](https://github.com/nolabs-ai/nono)). Those are useful mechanisms, but `nono`’s own materials acknowledge that it shares the host kernel and is not a cgroup resource limiter.

The current WSL2 5.15 kernel may provide early filesystem Landlock if enabled, but not the later network mediation needed for a complete agent policy. Runtime support must be probed (`nono why` or an equivalent ABI test); network denial must remain Docker `none`, a firewall, or an allowlisting proxy. `nono` is therefore optional defense-in-depth for trusted local host tools, not the replacement for Docker around T2 code.

Two 2026 preprints are informative but unreviewed. Sandlock proposes static kernel-enforced policy plus a narrow supervisor for dynamic effects and reports low overhead ([Sandlock preprint, May 2026](https://arxiv.org/abs/2605.26298)). SandboxEscapeBench seeds configuration and vulnerability flaws and tests whether adversarial agents exploit them, supporting continuous negative testing rather than the proposition that every container is broken ([SandboxEscapeBench preprint, v3 August 2026](https://arxiv.org/abs/2603.02277)). Their architectural direction supports SharpeArena’s fail-closed static boundary plus a minimal host supervisor; their performance/security numbers should not be treated as independently replicated facts.

### 4.6 Worktrees and development sandboxes

Git worktrees are valuable for concurrent agent development and review, but they isolate branch state, not authority. Mike McQuaid’s setup places agents under an unprivileged account, gives each a worktree, and reserves review/push for the human’s normal account ([practitioner report](https://mikemcquaid.com/sandboxed-agent-worktrees-my-coding-and-ai-setup-in-2026/)). INNOQ describes a Lima VM in which the host retains Git/network credentials while the agent sees a shared code mount; the authors explicitly leave network restriction as follow-up work ([practitioner report](https://www.innoq.com/en/blog/2025/12/dev-sandbox/)).

These are good development workflows. A writable worktree or shared mount can still alter everything exposed by that mount, and unrestricted egress can still exfiltrate it. Field entrants should not receive Git metadata, repository credentials, or a development worktree at all.

## 5. Recommended Windows 11 / WSL2 architecture

```text
Windows 11 host
├── GPU model service (trusted: Ollama/other local server)
│   ├── model weights/cache (C: or D:)
│   ├── loopback-only listener
│   └── no broker credentials or sealed evaluation salt
│
└── WSL2 trusted supervisor
    ├── SharpeArena environment + deterministic reward/transition engine
    ├── canonical Decision schema + semantic validator
    ├── run scheduler, timeouts, evidence writer, hashes
    ├── SharpeBench scorer (consumes completed run artifacts)
    ├── data-only model adapter ───────► model service
    │
    └── executable entrant adapter
        └── fresh digest-pinned Docker container
            ├── stdin observation / stdout Decision only
            ├── no network, mounts, secrets, devices or GPU
            ├── read-only root + bounded tmpfs
            └── CPU/RAM/PID/fd/time/output ceilings
```

### 5.1 Trust and data flow

The supervisor is the sole owner of datasets, sealed salts, environment state, reward calculation, model/broker routing, and evidence. An entrant sees only the current observation. It returns a decision. The supervisor validates syntax and episode-specific semantics before stepping.

For local open-weight models, call the trusted loopback model server from the supervisor. The model does not need access to files, the Docker socket, the environment object, or the GPU device through an entrant container. This is Browser Use’s “isolate the dangerous tool” pattern when the model produces only data; submitted images and generated code use its “isolate the whole agent” pattern, where the external control plane owns credentials and state ([Browser Use vendor architecture report](https://browser-use.com/posts/two-ways-to-sandbox-agents)).

### 5.2 Network and credential policy

**Retrospective field:** no entrant egress. The model adapter may call only the trusted loopback inference endpoint. It must reject redirects, use fixed destination and port configuration, impose request/response byte and time ceilings, and never inherit unrelated proxy or credential environment variables. Package download occurs in a separate image/model preparation phase, never during an evaluated cell.

**Forward paper trading:** a separate allowlisting proxy or supervisor adapter reaches only named market-data and paper-broker endpoints. Broker credentials remain in the supervisor’s credential store; the agent receives neither files nor environment variables containing them. Use scoped, short-lived, paper-only credentials where the provider supports them. A decision passes canonical validation, mandate/risk limits, position/open-order reconciliation, and postcondition checks before the supervisor submits it. Real-capital order execution is out of scope.

The container gets no network merely to “call Ollama.” That would require opening host networking and making policy harder. Keeping inference outside preserves both GPU performance and the entrant boundary.

### 5.3 GPU policy

NVIDIA’s container runtime exposes devices through `--gpus`/`NVIDIA_VISIBLE_DEVICES` and selects driver capabilities separately ([NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html)). The current entrant launcher supplies neither, which is correct.

Raw GPU passthrough adds the large kernel driver/API surface to the sandbox and weakens reset and resource-isolation assumptions. Keep the RTX A4000 accessible only to the trusted model service. If a future entrant genuinely needs acceleration, expose a narrow inference service, not `/dev/nvidia*`. If raw GPU becomes unavoidable, use a separate host/runtime profile and treat it as a different security class with exact device, memory, process, and driver-version evidence.

### 5.4 Storage on `D:`

Moving bulk data to `D:` is safe when it is host-managed:

- model weights, Ollama blobs, immutable OCI layers, archived datasets, and completed evidence may live there;
- create an explicit per-run staging directory with a quota and no credentials;
- prefer streaming observations over stdin or mounting a single immutable dataset artifact read-only;
- never bind `D:\`, the workspace root, the user profile, Docker’s state directory, or model cache into an entrant;
- destroy ephemeral scratch after the run and retain only hashed evidence/diagnostics;
- keep the sealed salt and forward-window secrets outside any entrant-visible path.

Storage location changes capacity, not the trust model.

### 5.5 Reset, snapshots, and independence

Environment reset and sandbox reset are separate requirements:

- reset SharpeArena with the committed seed and declared Gymnasium autoreset mode;
- create a fresh entrant process/container for each statistically independent run, or explicitly declare persistent scaffold memory as an experimental axis;
- keep the model server warm for throughput, but clear model conversation/KV state between cells unless memory is the tested treatment;
- use native environment clone/restore only for authorized tree-search/training code. Evaluation entrants must not restore privileged branches;
- never create a reusable container/VM snapshot after credentials, hidden seeds, or entrant state are present;
- hash the reset configuration and record termination versus truncation distinctly.

### 5.6 Required live negative tests

Before the first field—and on every runtime/configuration change—the actual configured backend should run a deliberately hostile fixture proving:

1. a host sentinel file, Git credential, Windows mount, Docker socket, model cache, and sealed-salt path cannot be read;
2. DNS, public TCP, WSL host gateways, Docker bridge peers, and the Ollama port cannot be reached from the entrant container;
3. `/dev/nvidia*`, other sensitive devices, IPC namespaces, and host processes are absent;
4. writes succeed only in the two bounded tmpfs mounts and disappear after exit;
5. setuid/capability escalation, namespace creation outside the profile, and representative blocked syscalls fail;
6. a fork bomb hits the PID limit; memory, file, stdout/stderr, CPU, and wall-time exhaustion are bounded;
7. timeout, crash, malformed JSON, duplicate/unknown symbols, oversized weights, and silent output become recorded faults—not fabricated holds;
8. restart removes all entrant state; two cells cannot communicate through files, process state, caches, or network;
9. the exact launch manifest, Docker/runtime versions, `docker info` security options, image digest, and negative-test results enter provenance.

SandboxEscapeBench motivates adversarial regression testing; it does not provide a turnkey guarantee for this configuration. The fixture must test this project’s exact command line and host.

## 6. Agent/RL environment SOTA and SharpeArena mapping

### 6.1 Environment contract

Gymnasium’s core contract distinguishes **termination** (a terminal state under the task’s MDP) from **truncation** (an external horizon or limit), because bootstrapping across them differs; it also defines seeded reset and explicit action/observation spaces ([Gymnasium `Env`](https://gymnasium.farama.org/api/env/)). SharpeArena should preserve this distinction across Rust, Gymnasium, MCP, local model, evidence, and Minari paths. A timeout or protocol failure is neither an economic terminal state nor a successful truncation; record it as a transport/process fault alongside any training-compatible transition.

Patronus’s 2026 guide usefully emphasizes structured actions, deterministic/verifiable rewards, reset semantics, and defined invalid-action behavior, but it is a vendor guide rather than a standard ([Patronus practitioner/vendor guide](https://www.patronus.ai/guide-to-rl-environments)). Veris frames an agent environment as a partially observed stochastic system and highlights sim-to-real gaps in tool latency/error, user behavior, scenario coverage, and task postconditions; it too is a vendor essay, not independent validation ([Veris practitioner/vendor essay](https://veris.ai/blog/building-an-agent-you-need-an-environment)). These recommendations align with, but do not supersede, the Farama APIs and SharpeArena’s tested contract.

### 6.2 Vectorization

Gymnasium `VectorEnv` batches independent environment copies and returns batched observations, rewards, terminations, truncations, and info; autoreset behavior is an explicit mode ([Gymnasium vector API](https://gymnasium.farama.org/api/vector/)). SharpeArena’s native vector engine is the correct substrate for local model evaluation, but environment batching and model batching must remain conceptually separate.

Required disclosures and tests:

- exact number of lanes and autoreset mode;
- seed derivation for each lane and proof that lanes do not share environment state;
- whether model requests are batched or continuously scheduled;
- no shared conversation/KV memory across lanes unless scaffold memory is the named treatment;
- per-lane fault accounting—a malformed decision in lane 7 must not convert or cancel the actions in other lanes;
- throughput reported separately for environment steps, model tokens, and end-to-end decisions.

Continuous GPU batching can be nondeterministic even at nominal temperature zero. That is agent-side variability for pass\(^{k}\) to measure; it must not be described as environment nondeterminism.

### 6.3 Multi-agent environments

PettingZoo’s Parallel API models simultaneous observations and actions in a partially observable stochastic game and returns dictionaries keyed by live agent ([PettingZoo Parallel API](https://pettingzoo.farama.org/api/parallel/)). Its test suite includes `parallel_api_test` and seed determinism checks ([PettingZoo environment tests](https://pettingzoo.farama.org/content/environment_tests/)).

SharpeArena’s market-making, competition, endogenous-market, and LOB environments fit this API. Maintain simultaneous step semantics—do not let model-call completion order become economic action order. Run both API and seed tests for every parallel environment, not only the competition configuration. If centralized training exposes a privileged global `state()`, label it training-only; evaluation entrants receive only their declared observation.

### 6.4 Offline RL and Minari

Minari datasets retain observations, actions, rewards, terminations, truncations, infos, reset seeds/options, action/observation spaces, environment specification, episode/step counts, algorithm identity, code permalink, and version/dependency metadata; `recover_environment` and dataset splitting support reproducible reuse ([Minari dataset standards](https://minari.farama.org/main/content/dataset_standards/), [Minari basic usage](https://minari.farama.org/main/content/basic_usage/)).

SharpeArena’s exporter should extend—not replace—these fields with:

- dataset and scenario hashes;
- contract/schema, reward, mandate, cost, and scoring configuration hashes;
- generator/canonical-tape identity and train/evaluation band;
- model digest, quantization, sampler, prompt/scaffold, decision cadence, and inference server version for model rollouts;
- sandbox image/runtime/launch manifest and host/GPU/driver metadata;
- protocol/transport fault flags and cost reports;
- a statement of whether the episode was retrospective, generated, or forward paper trading.

An offline dataset is evidence of the behavior policy that produced it. It is not evidence that the same policy would act identically under a different model build, batch scheduler, simulator version, or live market.

### 6.5 Deterministic, verifiable rewards

Eligibility rewards and gates must remain native, deterministic, and independently recomputable from the trajectory. An LLM judge may supply supplementary qualitative analysis, but never the rank-eligible reward, pass/fail gate, or unobserved market fact. The evidence record should allow a verifier to recompute every reward and final score without calling a model service.

For every reward term, specify:

- mathematical definition, units, observation timing, and allowed state;
- treatment of costs, partial fills, latency, bankruptcy, mandate breach, invalid action, termination, and truncation;
- whether the term is dense training shaping or the terminal evaluation measure;
- invariance/units tests, boundary tests, positive and negative controls, and a golden trajectory.

This is where the two-product division is useful: SharpeArena proves trajectory/reward correctness; SharpeBench proves selection correction, reliability, process/cost gates, and field-level scoring.

### 6.6 Invalid actions

One semantics should apply across JSON, stdio/HTTP, Gymnasium, MCP, vector, and paper-broker paths:

1. parse the canonical closed-object schema;
2. validate current-episode symbols, uniqueness, finiteness, weight bounds, shorting/mandate constraints, and action/weight consistency;
3. return a typed fault carrying layer, reason, model/run/cell, and raw-output hash;
4. never silently replace an invalid decision with zero weights or a hold;
5. in training APIs that require a continuing step, use an explicit invalid-action transition/penalty and mark it in `info`; in rank evaluation, follow the precommitted fault policy, which may abort or disqualify the cell.

The current strict parser implements this principle, but the MCP helper’s ignore/overwrite behavior must be unified before use.

### 6.7 Partial observability and leakage

SharpeArena is naturally partially observable: the entrant sees a trailing market/portfolio view, while the simulator owns latent future paths, generator state, hidden seeds, other-agent internals, and final outcomes. Document the privileged-state/observation split as part of each environment spec.

Leak tests should establish that:

- no observation, info field, exception, filename, cache, timestamp, order, or episode length encodes future bars or the sealed salt;
- reset options and Minari metadata do not reveal evaluation identities before reveal;
- centralized critic/global state is unavailable to evaluation policies;
- model prompts contain no filesystem path or metadata from which a public pretrained model can trivially identify the exact held-out interval;
- external historical-data familiarity is acknowledged as a model-pretraining limitation. Local weights and no egress close runtime leakage, not pretraining contamination.

### 6.8 Sim-to-real and forward paper trading

A simulated environment is not validated merely because it contains transaction costs, latency, an order book, or recognizable API names. Veris’s practitioner formulation is useful here: validation should cover task success/postconditions, tool/API error and latency distributions, scenario coverage, and user/market interaction—not only nominal state/action shape ([Veris](https://veris.ai/blog/building-an-agent-you-need-an-environment)).

Precommit a bridge table rather than claiming “realism” globally:

| Gap | Retrospective/generator measure | Forward paper-trading measure |
|---|---|---|
| Returns/dynamics | stylized-fact diagnostics, regime coverage, cross-asset dependence | rolling distribution drift relative to observed feed |
| Execution | spread, impact, partial-fill, delay, queue/slippage model controls | realized paper fill latency/slippage and reject/cancel rates |
| Agent interaction | endogenous-market and PettingZoo policies | market is external; measure action-to-market response without causal overclaim |
| Operational tools | injected timeout/error scenarios | actual endpoint latency, throttling, downtime, stale data |
| Safety/postcondition | mandate/risk/order invariants in deterministic tests | positions, cash, open orders, rejected decisions, and reconciliation after every cycle |
| Result stability | seed/window/scaffold/model pass\(^{k}\) | preregistered forward windows; no retrospective reselection |

Forward evidence is prospective and valuable precisely because it is not replayable. Give it a separate schema and headline. Do not merge it into byte-identity or historical leaderboard claims.

## 7. SharpeArena–SharpeBench composition

The intuitive statement—“the sandbox and environment are SharpeArena, and SharpeBench tests them”—is directionally right but should be expressed as layers:

```text
Entrant/model
    │ canonical Decision / typed fault
    ▼
SharpeArena execution profile
    ├── environment/scenario/mandate
    ├── sandbox or data-only model adapter
    ├── deterministic transition/reward
    └── trajectory + environment provenance
    │ versioned run artifact
    ▼
SharpeBench evaluation
    ├── trial/field accounting and clone handling
    ├── deflation and bootstrap/reliability
    ├── process and cost gates
    └── attested result/leaderboard
```

Do not make both packages mutually dependent. A cycle makes independent conformance, versioning, and reuse harder and can allow evaluation policy to leak into environment mechanics. Instead:

1. publish a shared protocol/run-artifact schema with semantic versioning;
2. put environment identity, scenario hash, reward/config hash, entrant/model identity, scaffold, sandbox manifest, faults, costs, and trajectory digest in it;
3. have SharpeArena emit/validate it;
4. have SharpeBench ingest and independently recompute every score it can;
5. conformance-test both sides against fixed fixtures and reject unknown breaking versions;
6. link evidence using content hashes, not implicit filesystem state.

This arrangement also permits other environments to target SharpeBench and other evaluators to consume SharpeArena without weakening either paper’s claim.

## 8. Implementation sequence and acceptance criteria

### Phase 0 — runtime readiness

- Start Docker Desktop, enable WSL2 integration, expose the CLI in WSL, and record runtime/security information.
- Add a required local/self-hosted sandbox readiness command; generic hosted CI may keep a non-blocking unit test, but a field cannot begin on a skipped live test.
- Build and run the hostile fixture suite in Section 5.6 against the exact image/runtime.
- Unify MCP semantic decision validation with the canonical strict parser.

**Exit:** all negative tests pass; Docker absence, missing limits, mutable image tags, or runtime drift aborts the field.

### Phase 1 — data-only local model field

- Trusted supervisor calls a loopback-only Ollama/model endpoint.
- JSON-schema-constrained canonical decisions plus strict episode semantics.
- Scheduler over model × environment × seed × repetition × cadence/scaffold.
- Record model digest, weights/quantization, prompt, sampler/thinking settings, server/runtime, hardware, faults, and costs.
- Environment and model batching have independent lane/session tests.

**Exit:** one complete field reproduces environment trajectories and scores from stored decisions; model outputs may vary, but every variation is attributed and pass\(^{k}\)-visible.

### Phase 2 — executable entrants and strategy generation

- Keep the closed strategy DSL first.
- Add Docker-per-candidate or Wasm/WASI executor only for arbitrary code.
- Count every generated candidate and selection attempt in SharpeBench’s observed trial field.
- Fresh state per candidate; no network/package install; deterministic clock/RNG; bounded stdout/stderr/time/memory/PIDs/files.
- Optional `runsc` bakeoff against the same conformance, negative, and throughput tests.

**Exit:** generated code cannot reach host/network/secrets/devices; trial count is observed, not declared; selected-strategy evidence replays from stored candidates.

### Phase 3 — forward paper trading

- Paper-only adapter, exact endpoint allowlist, supervisor-held credential, native risk/mandate gate, final-state reconciliation, kill switch, and bounded open orders/notional.
- Commit forward windows, configs, model/scaffold identities, and decision cadence before data arrive.
- Separate evidence type and UI; no deterministic/replay badge.

**Exit:** a dry-run and paper-only canary prove that no agent path can submit a real-capital order and every intended/blocked/cancelled order is reconciled.

### Phase 4 — optional stronger isolation

- Compare Docker default vs gVisor on compatibility, hostile tests, throughput, and reproducibility.
- Evaluate Firecracker only if threat class expands to T3 or a dedicated Linux host is available.
- Probe `nono`/Landlock capabilities after a WSL/kernel upgrade; never infer network protection from filesystem-only support.

**Exit:** adopt a stronger backend only if it passes the same contract and materially improves the named threat without making the field operationally unreproducible.

## 9. Findings from the supplied practitioner literature

The supplied articles converge on several sound principles, but at different evidence levels:

- Browser Use separates isolating one dangerous tool from isolating the whole agent, and keeps credentials/state in an external control plane. This maps cleanly to data-only model decisions versus executable entrants ([vendor architecture report](https://browser-use.com/posts/two-ways-to-sandbox-agents)).
- Cursor reports a Linux design using Landlock, seccomp, and overlay filesystems and a Windows path through WSL2; it also notes that the agent harness must understand the sandbox to avoid futile permission retries. Its reported productivity metric is internal vendor evidence, not a security benchmark ([vendor engineering report](https://cursor.com/blog/agent-sandboxing)).
- Docker describes per-sandbox microVMs on Windows/macOS, workspace synchronization, and network policies in Docker Sandboxes. This is relevant future technology, not the implementation SharpeBench currently invokes ([vendor engineering report](https://www.docker.com/blog/building-ai-teams-docker-sandboxes-agent/)).
- Rafael Ben-Ari reports agents in one repository crossing role/file boundaries and uses separate non-root containers with only required resources visible. The case motivates separation; the example Dockerfile is not a complete hardening profile ([practitioner report](https://medium.com/@rafaelbenari/custom-containerized-sandboxes-for-ai-agents-dd2cd2603b3b)).
- Alan West advocates layered non-root Docker, user namespaces, seccomp, narrow mounts, and guarded sockets/network. The illustrative custom seccomp profile should not be copied as a production allowlist without workload-specific testing ([practitioner report](https://dev.to/alanwest/how-to-sandbox-ai-coding-agents-without-crippling-them-116c)).
- Justin Lam shows practical `nono` profiles and low-friction filesystem/network restriction while acknowledging the shared kernel and absence of cgroup resource limits ([practitioner report](https://www.justinmklam.com/posts/2026/05/sandboxing-with-nono/)).
- Luis Cardoso’s useful synthesis is **boundary + policy + lifecycle**: containers/gVisor/microVM/runtime are boundaries; files/network/process/device/quota/interface rules are policy; fresh/workspace/snapshot modes are lifecycle. It correctly notes that microVMs still need policy and GPU passthrough changes the failure surface ([secondary field guide](https://www.luiscardoso.dev/blog/sandboxes-for-ai)).
- Mike McQuaid and INNOQ demonstrate separating agent worktrees/VMs from the human’s push credentials. These are effective development practices but incomplete runtime sandboxes, especially when writable mounts and egress remain ([McQuaid practitioner report](https://mikemcquaid.com/sandboxed-agent-worktrees-my-coding-and-ai-setup-in-2026/), [INNOQ practitioner report](https://www.innoq.com/en/blog/2025/12/dev-sandbox/)).
- Veris and Patronus emphasize that an agent needs a specified environment, structured actions, reset/fault semantics, deterministic/verifiable reward where possible, and validation beyond happy-path trajectories. Their pieces are vendor guidance, not normative RL APIs ([Veris](https://veris.ai/blog/building-an-agent-you-need-an-environment), [Patronus](https://www.patronus.ai/guide-to-rl-environments)).

The consistent lesson is that a container, VM, worktree, LSM policy, or environment API is only one layer. The quality of the system depends on the authority left outside it, the lifecycle between trials, and negative evidence that the exact configuration fails closed.

## 10. Non-goals and residual risks

- No local boundary makes a compromised Windows/WSL/Docker/kernel/GPU stack trustworthy.
- Rootless Docker, gVisor, Landlock, Firecracker, and Wasm reduce different surfaces; none replaces credential, egress, resource, reset, supply-chain, and evidence policy.
- Local open weights prevent API drift and runtime provider leakage, but cannot prove the model did not memorize public historical data during pretraining.
- Schema-constrained output prevents many syntax faults, not economically incoherent decisions or prompt injection.
- A deterministic simulator and reward do not make continuous-batched GPU inference deterministic. Treat agent variance as measured evidence.
- Paper-trading fills and data are provider- and time-dependent. They cannot carry SharpeArena’s byte-replay guarantee.
- Resource ceilings prevent one entrant from exhausting the workstation only if the chosen Docker/WSL/cgroup configuration actually enforces them; the live fixture must prove this.
- Firecracker and Docker Sandboxes are not justified merely because they are stronger-sounding. They should be adopted only against a named expanded threat model.

## 11. Annotated source ledger

### Normative and implementation-primary

- [Docker Engine security](https://docs.docker.com/engine/security/) — namespaces, cgroups, daemon authority, capabilities, kernel attack surface.
- [Docker rootless mode](https://docs.docker.com/engine/security/rootless/) — daemon and containers in a user namespace without host root.
- [Docker seccomp](https://docs.docker.com/engine/security/seccomp/) — default syscall allowlist and blocked syscall rationale.
- [Docker resource constraints](https://docs.docker.com/engine/containers/resource_constraints/) — CPU and memory controls.
- [NVIDIA Container Toolkit: specialized Docker configuration](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html) — GPU/device and driver-capability exposure.
- [gVisor security model](https://gvisor.dev/docs/architecture_guide/security/) and [Docker quick start](https://gvisor.dev/docs/user_guide/quick_start/docker/) — Sentry boundary, residual dependencies, OCI integration.
- [Firecracker jailer](https://github.com/firecracker-microvm/firecracker/blob/main/docs/jailer.md), [production host setup](https://github.com/firecracker-microvm/firecracker/blob/main/docs/prod-host-setup.md), and [snapshot support](https://github.com/firecracker-microvm/firecracker/blob/main/docs/snapshotting/snapshot-support.md) — operational hardening and snapshot trust.
- [Wasmtime security](https://docs.wasmtime.dev/security.html) and [WASI](https://wasi.dev/) — capability imports, memory/control isolation, WASI authority model.
- [Linux Landlock userspace API](https://docs.kernel.org/userspace-api/landlock.html) — kernel-mediated unprivileged process restrictions and ABI variation.
- [`nono`](https://github.com/nolabs-ai/nono) — primary implementation documentation for its Landlock/Seatbelt wrapper; project claims are not independent audit evidence.
- [Gymnasium `Env`](https://gymnasium.farama.org/api/env/) and [Vector API](https://gymnasium.farama.org/api/vector/) — reset/seed, termination/truncation, batch/autoreset semantics.
- [PettingZoo Parallel API](https://pettingzoo.farama.org/api/parallel/) and [environment tests](https://pettingzoo.farama.org/content/environment_tests/) — simultaneous multi-agent contract and conformance/seed tests.
- [Minari dataset standards](https://minari.farama.org/main/content/dataset_standards/) and [basic usage](https://minari.farama.org/main/content/basic_usage/) — offline dataset schema, metadata, splitting, and environment recovery.

### 2026 preprints — unreviewed

- [Sandlock: Protecting AI Agents with Information-Flow Control](https://arxiv.org/abs/2605.26298) — static kernel policy plus narrow dynamic supervisor; author-reported performance.
- [SandboxEscapeBench](https://arxiv.org/abs/2603.02277) — adversarial-agent testing of seeded sandbox weaknesses; supports hostile regression fixtures.

### Vendor engineering and guidance

- [Browser Use: Two Ways to Sandbox Agents](https://browser-use.com/posts/two-ways-to-sandbox-agents) — tool-only versus whole-agent isolation and external control plane.
- [Cursor: Agent Sandboxing](https://cursor.com/blog/agent-sandboxing) — vendor account of Landlock/seccomp/overlay and WSL2 architecture; internal performance claims.
- [Docker: Building AI Teams with Docker Sandboxes](https://www.docker.com/blog/building-ai-teams-docker-sandboxes-agent/) — vendor description of evolving Docker Sandboxes/microVM behavior.
- [Veris: Building an Agent? You Need an Environment](https://veris.ai/blog/building-an-agent-you-need-an-environment) — practitioner POMDP/environment and sim-to-real framing.
- [Patronus: Guide to RL Environments](https://www.patronus.ai/guide-to-rl-environments) — practitioner/vendor checklist for actions, rewards, reset, and invalid states.

### Practitioner and secondary reports

- [Mike McQuaid: Sandboxed Agent Worktrees](https://mikemcquaid.com/sandboxed-agent-worktrees-my-coding-and-ai-setup-in-2026/) — unprivileged account/worktree development workflow.
- [Rafael Ben-Ari: Custom Containerized Sandboxes for AI Agents](https://medium.com/@rafaelbenari/custom-containerized-sandboxes-for-ai-agents-dd2cd2603b3b) — role/file-crossing failure case and container separation.
- [Luis Cardoso: Sandboxes for AI](https://www.luiscardoso.dev/blog/sandboxes-for-ai) — secondary synthesis of boundary, policy, lifecycle, GPU, and control planes.
- [Alan West: How to Sandbox AI Coding Agents Without Crippling Them](https://dev.to/alanwest/how-to-sandbox-ai-coding-agents-without-crippling-them-116c) — layered Docker practitioner guidance.
- [Justin Lam: Sandboxing with `nono`](https://www.justinmklam.com/posts/2026/05/sandboxing-with-nono/) — practical capability-policy experience and stated limitations.
- [INNOQ: Dev Sandbox](https://www.innoq.com/en/blog/2025/12/dev-sandbox/) — Lima VM, host-owned credentials, shared workspace, and acknowledged network-policy gap.

## 12. Research disclosure

This review was assembled with AI-assisted web research and direct repository/system inspection. Current or security-sensitive claims were checked against official runtime/kernel/API documentation where available. Every supplied URL was read as a vendor, practitioner, or secondary source and labeled accordingly. No penetration test, formal verification, independent runtime audit, or peer review was performed. The local Docker engine was unavailable at the audit moment, so its runtime security options and negative-test behavior remain acceptance work rather than verified facts. Recommendations distinguish inspected code from proposed controls and do not claim that any boundary is escape-proof.
