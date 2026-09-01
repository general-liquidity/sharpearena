# Gordon architecture and portability assessment

**Assessment date:** 2026-08-26

**Source reviewed:** `work/gordon-work`

**Targets:** SharpeArena and SharpeBench
**Scope:** architecture and portability only. This assessment does not port Gordon code.

**Current-status note (2026-09-01):** this document began as a pre-implementation
assessment. The four native candidates it identified have since shipped:

| Candidate | Current disposition |
|---|---|
| `EdgeManifest` | Implemented as a closed, unit-typed strategy-evidence schema with host-counted raw trial ordinals and held-out kill-condition evaluation. |
| Strict trace promotion | Implemented as immutable silver candidates, explicit operator promotion, deterministic fingerprints, and offline gold regressions. |
| Paper reconciliation | Implemented with a crash-persistent `submission_unknown` state, deterministic client-order identity, query-before-retry, and no production endpoint. Partial-fill state remains the broker's cumulative last-write view rather than an independently accumulated fill ledger. |
| Evidence coverage | Implemented in SharpeBench as a machine-readable covered/excluded-field inventory with a drift test and secret redaction. |

The historical code observations and rejection reasoning below are retained; any
sentence phrased as a proposed port should be read with this status table.

## Executive decision

Gordon should not become the runtime underneath SharpeArena or SharpeBench. It is a large, stateful trading application whose orchestration, permissions, memory, broker integrations, and proactive services are deliberately coupled around an interactive operator. SharpeArena and SharpeBench need a much smaller and more stable experimental boundary: a fixed environment produces typed trajectories; a scorer evaluates them. Importing Gordon wholesale would make the scaffold part of the treatment, weaken reproducibility, introduce a circular package boundary, and bring in several fail-open or permissive paths that are unacceptable in a benchmark.

The scan nevertheless found three Gordon ideas worth preserving as native concepts. The assessment itself did not port them; later Sharpe-suite work implemented product-native versions without a Gordon dependency.

1. **An edge manifest for generated strategies.** Gordon's `EdgeSpec` usefully binds a hypothesis to a mechanism, regimes, invariants, kill conditions, and a verification plan. SharpeArena now carries that concept as typed evidence attached to every generated candidate without importing Gordon's parser or monitor.
2. **A production-trace-to-regression promotion queue.** Gordon has a sound silver-to-gold workflow for turning a flagged real trace into a frozen regression scenario. SharpeArena now provides a strict, operator-approved native workflow over its typed traces.
3. **An explicit uncertain-execution state for paper trading.** Gordon's termination/reconciliation work captures the important distinction between “not submitted,” “submission outcome unknown,” and “broker acknowledged.” The paper-only arm now models that state explicitly and persists it before reconciliation.

A fourth group belongs only in a **later heavy-scaffold experimental arm**: multi-agent orchestration, tool loops, context compaction, offloaded tool results, rejection memory, and richer process checks. The same model must be evaluated under the minimal scaffold and the heavy scaffold on identical cells. Otherwise the benchmark would report Gordon-plus-model as model performance.

Everything else is either already present in a stronger form or should be rejected. In particular, do not port Gordon's strategy-generator fallbacks, custom trial/DSR implementation, genetic auto-promotion loop, adaptive trust trajectory, heuristic risk classifier, sandbox wrappers, duplicated HMAC audit stacks, permissive dry-run evaluator, live-result cache, or application-wide SessionRuntime.

## Decision vocabulary

| Decision | Meaning |
|---|---|
| **Implemented native concept** | The assessment supplied the invariant; a later change implemented it within the Sharpe product boundary without a Gordon dependency. |
| **Later heavy-scaffold arm** | Useful only as an explicit treatment arm after the minimal local-model field is stable. It must be configurable, disclosed, and scored separately. |
| **Already exists** | SharpeArena/SharpeBench already implement the capability, generally with a cleaner experimental boundary. Extend only if a concrete gap remains. |
| **Reject** | The component is application-specific, duplicates stronger target code, weakens the guarantees, contains unsafe fallbacks, or has no valid trading-benchmark analogue. |

## Method and evidence boundary

This was an end-to-end architectural scan, not a filename inventory. It began with Gordon's onboarding and invariants, then followed the main construction paths from session creation through orchestration, tools, memory, permissions, trading adapters, evidence, evaluation, and background services. For each requested subsystem, the scan traced the factory or registry into representative leaf implementations and tests, and compared it against the corresponding SharpeArena/SharpeBench implementation.

Gordon itself warns against literal coding-agent analogy: it is a trading agent, not a coding agent, and designs must be translated into the trading domain before reuse (`work/gordon-work/CLAUDE.md:5-9`; `work/gordon-work/AGENTS.md:7-20`). It also documents that the orchestrator/executor/researcher split is a safety boundary, not an accidental inefficiency (`work/gordon-work/CLAUDE.md:71-73`). Those two constraints govern the recommendations below.

## Existing product boundary: keep it directed, not circular

The desired conceptual composition is already the right one:

```text
model/scaffold
      |
      v
SharpeArena environment + transport + sandbox
      |
      v
typed trajectory/evidence contract
      |
      v
SharpeBench statistics + process gates + leaderboard
```

SharpeArena already depends on SharpeBench's neutral crates and re-exports the shared simulator/protocol surface (`sharpearena/Cargo.toml:11-15`; `sharpearena/crates/sharpearena/src/lib.rs:79-125`). That is a directed implementation dependency supporting the conceptual pipeline “Arena runs; Bench scores.” SharpeBench must not import the full SharpeArena package in return. Shared contracts should remain in small neutral crates (`sharpebench-protocol`, `sharpebench-core`, and the existing simulator layer), preventing a circular dependency and allowing either product to evolve its presentation layer independently.

The local-model and strategy-generation work also exists already:

- `sharpearena/crates/sharpearena-py/python/sharpearena/local_agents.py` distinguishes deterministic environment behavior from nondeterministic model behavior, records model identity and generation parameters, uses loopback Ollama, supports constrained output, batching, sharding, resumability, and preserves typed process events (`local_agents.py:1-6`, `57-133`, `350-523`, `630-937`).
- `sharpearena/crates/sharpearena-py/python/sharpearena/strategy_generation.py` uses a non-executable JSON DSL, records every raw candidate before validation or deduplication, and separates validation from test data (`strategy_generation.py:1-7`, `49-136`, `520-552`).
- `sharpearena/crates/sharpearena-py/python/sharpearena/paper_trading.py` already supplies public/paper connectors, a fixed risk guard, deterministic client-order IDs, all-orders preflight before any submission, a journal, forward commitments, and a separate evidence class.
- `sharpebench/crates/sharpebench-arena/src/sandbox.rs` already provides a fail-closed Docker boundary with no network, a read-only filesystem, a non-root identity, dropped capabilities, resource limits, and digest pinning (`sandbox.rs:1-18`, `95-190`).
- SharpeBench process scoring is typed rather than text-pattern based (`sharpebench/crates/sharpebench-core/src/process.rs:1-105`).

Therefore Gordon is a source of a few design ideas and an optional future scaffold treatment, not a missing foundation.

## Architecture map and portability verdict

| Gordon subsystem | Principal local evidence | What it actually does | Decision |
|---|---|---|---|
| Session runtime | `src/runtime/session/SessionRuntimeFactory.ts:86-199`; `SessionRuntime.ts:41-183` | Composes persistence, transcript, permissions, compaction, replay, plugins, worker registry, and UI-facing runtime state | **Reject** wholesale |
| Agent orchestration | `src/infra/agents/orchestrator.ts:236-335,401-731`; `definitions/gordon.ts:337-454` | Builds grounded prompts, selects tools/agents, streams Mastra execution, records usage, and triggers post-turn work | **Later heavy-scaffold arm** |
| Role routing | `src/infra/agents/orchestrator/toolAgentMap.ts:28-185`; `HandoffCoordinator.ts:62-260` | Maps state-changing operations to Executor, research to Researcher, budgets handoffs, redacts secrets/account context | **Later heavy-scaffold arm** |
| Context compaction | `src/infra/domain/memory/summarizer.ts:81-99,259-330`; `compactionHandler.ts:38-90`; `contextCollapse.ts:1-170` | Five pressure stages, summary/collapse projection, recent-message preservation | **Later only after repair** |
| Memory | `src/infra/domain/memory/memoryFactory.ts:24-199`; `memoryGate.ts:37-185` | Small durable hot tier, opt-in writes, separate sub-agent memories, provenance labels | **Later heavy arm; concept only** |
| Tool result offload | `src/infra/context/toolResultStorage.ts`; `withSpill.ts`; `runtimeHarness.ts:110-135` | Redacts, spills large results to protected local files, returns bounded previews | **Later heavy arm; reimplement content-addressed** |
| Loop detection | `src/infra/agents/harness/runtimeHarness.ts:62-105,503-526,536+` | Detects repeated tool fingerprints and cycles; caps output by family | **Later heavy arm** |
| Strategy generation | `src/infra/agents/tools/strategy-generator.ts:172-359,460-582,642-718` | Natural language to intent/DSL/backtest with recovery fallbacks | **Reject implementation** |
| Recipe primitives | `src/core/strategies/recipes/` | Pure signal-state transforms and exposure/time gates | **Already covered / reference entrants only** |
| Trial tracker / DSR | `src/core/strategies/multipleTestingTracker.ts:1-290` | Persists caller-scoped attempts and computes a custom DSR | **Reject** |
| EdgeSpec | `src/core/edge/types.ts:1-52`; `parser.ts:1-20,87-104`; `monitor.ts:18-72` | Structured hypothesis, mechanism, regimes, invariants, kill and verification fields | **Candidate concept — not ported** |
| Strategy genome | `src/core/genome/types.ts`; `evolution-loop.ts:1-440`; `fitness.ts:8-110`; `optimization-tier.ts:130-143` | Lineage, mutation, heuristic fitness, scheduled search, and auto-promotion | **Reject automation; limited later concepts** |
| Eval generation | `src/infra/domain/evals/harness/generator/`; `CLAUDE.md:75-106` | Derives deterministic scenarios from specs with provenance | **Already exists in typed target tests** |
| Process checks | `harness/process/processChecks.ts:1-334` | Checks tool-call order, approvals, denylist, loops, lookahead recall, and duplicate calls | **Already exists for trading core; later heavy arm for tools** |
| pass^k | `harness/process/passK.ts:41-83`; `kRunProducer.ts:35-58` | All/majority/rate aggregation across sequential runs | **Already exists; target implementation is stronger** |
| Trace promotion | `harness/traces/traceAdapter.ts`; `traceScorer.ts`; `promotionQueue.ts`; `CLAUDE.md:104` | Converts real traces, flags failures, queues silver cases, operator-promotes to frozen gold | **Candidate concept — not ported** |
| Audit chain | `src/core/audit/signing.ts:6-190,340-466` | Redacted HMAC hash chain with local key | **Already exists in stronger target provenance; reject code** |
| Platform audit log | `src/infra/platform/audit/audit-log.ts:25-144,292-369` | A second, independent per-entry HMAC log | **Reject duplicate** |
| Permission engine | `src/runtime/permissions/PermissionEngine.ts:21-117,223+` | Deny/allow/prompt rules, patterns, hooks, queue | **Paper arm already has fixed guards; reject benchmark port** |
| Adaptive trust | `src/runtime/permissions/trustTrajectory.ts:1-29,87-122,167+` | Learns auto-approval from repeated operator approvals with hard exclusions | **Reject** |
| Risk classifier | `src/infra/trading/risk/riskClassifier.ts:9-14,204-263,356-363` | Multi-dimensional heuristic pre-trade classification | **Reject** |
| Execution reconciliation | `src/infra/trading/execution/terminationLayers.ts:7-29` | Distinguishes runtime retry, session reconciliation, and system recovery | **Candidate concept — not ported, paper arm only** |
| Broker adapters | `src/infra/broker/types.ts:265-314`; `factory.ts:18-67`; `adapters/alpaca.ts:27-30,185-253` | Normalized brokerage surface, paper defaults, credential handling | **Already exists sufficiently; later adapters only** |
| Exchange adapters | `src/infra/exchange/factory.ts:104-157` | CCXT venue construction with sandbox/live gates | **Reject dependency port** |
| OHLCV cache | `src/infra/data/ohlcvCache.ts:129-210` | Immutable first-write-wins, as-of candle cache with hashes | **Concept useful only for ingestion; never scored live reads** |
| Proactive observer | `src/infra/proactive/engine/observer.ts:1-315`; `producers/index.ts:11-40` | Interval scheduler over news, regime, volatility, funding, and account producers | **Reject benchmark; limited paper-arm scheduling later** |
| Result cache | `src/infra/agents/tooling/toolResultCache.ts:79-255`; `withResultCache.ts:8-39` | TTL cache keyed by tool/input | **Reject in scored paths** |
| Prefix cache | `src/infra/agents/context/sharedPrefixCache.ts:1-55` | Stable shared prompt-prefix caching metadata | **Later heavy arm** |
| Observability | `src/infra/observability/tracing.ts:1-10,62-82,219-223`; `metrics.ts` | Local metrics/logging; external tracing is disabled/no-op | **Already covered by evidence metrics; do not overclaim** |
| Subprocess sandbox | `src/infra/safety/subprocessSandbox.ts:1-28,128-140,262-307` | Optional bwrap/Seatbelt wrapper; unsupported/unavailable paths run unsandboxed | **Reject** |
| Eval sandbox | `src/infra/domain/evals/harness/live/sandbox.ts` | Redirects files, database, and environment for eval runs | **Reject as a security boundary** |
| Strategy sandbox | `src/core/strategies/strategySandbox.ts:1-18` | Isolated in-memory paper portfolio state | **Reject the “sandbox” label; useful simulation state already exists** |

## 1. Runtime and orchestration

### What Gordon has

`SessionRuntimeFactory` is the composition root for an interactive application. It creates a runtime store, transcript and scratchpad services, permission engine, bridge/history, compaction and replay managers, transcript projector, worker registry, and plugin manager, hydrates persisted state, then installs debounced persistence (`work/gordon-work/src/runtime/session/SessionRuntimeFactory.ts:86-199`). `SessionRuntime` applies policy and permission checks before tool execution and exposes runtime controls to the host application (`work/gordon-work/src/runtime/session/SessionRuntime.ts:41-183`).

The agent layer is a centralized three-role topology. The Gordon orchestrator assembles grounded context and a large tool surface, can run separate thinking and critique phases, delegates to restricted Executor and Researcher agents, streams a multi-step Mastra response, and performs post-turn accounting/compaction (`work/gordon-work/src/infra/agents/orchestrator.ts:236-335,401-731`). The role registry assigns state-changing operations to Executor and research operations to Researcher, budgets delegations, and filters handoff context (`work/gordon-work/src/infra/agents/orchestrator/toolAgentMap.ts:28-185`; `HandoffCoordinator.ts:62-260`). Gordon's own instructions explain that this split is intentional error containment and permission separation (`work/gordon-work/CLAUDE.md:71-73`).

### Why it does not belong in the minimal benchmark

This runtime is appropriate for an operator-facing trading assistant. It is not a neutral agent adapter. Its hidden treatment variables include prompt assembly, tool availability, agent routing, handoff summarization, memory state, thinking/critique calls, persistence, permission history, and post-turn compaction. Comparing two models through it would compare a compound system, while comparing a Gordon-wrapped model with a stateless model would be uninterpretable.

**Decision:** reject `SessionRuntime` and the application shell. Preserve the current minimal local-agent interface as the default. Later, expose Gordon-like orchestration as a named scaffold treatment, for example `scaffold=minimal|heavy`, and run paired cells with identical model digest, data, seed, cadence, and sampling settings. The heavy arm must record every intermediate model call and handoff, not flatten them into one decision.

## 2. Compaction, memory, and tool-result offload

### Valuable ideas

Gordon has a thoughtful pressure schedule: 70/80/90/94/99 percent context pressure maps to masking, pruning, aggressive pruning, collapse, and full summarization, with both proportional thresholds and absolute headroom floors (`work/gordon-work/src/infra/domain/memory/summarizer.ts:81-99,259-330`). Its durable hot memory is deliberately small, semantic recall is not ambient, working-memory writes are capped, and untrusted/sensitive writes are gated (`work/gordon-work/src/infra/domain/memory/memoryFactory.ts:24-199`; `memoryGate.ts:37-185`; also documented at `CLAUDE.md:45-57`). Large tool outputs are redacted and spilled to owner-protected local files with a preview, while family-specific limits and loop fingerprints prevent repeated output explosions (`work/gordon-work/src/infra/context/toolResultStorage.ts`; `withSpill.ts`; `runtimeHarness.ts:62-135,503+`).

These are useful in a long-horizon, tool-using scaffold. They are unnecessary in the minimal per-bar Markov scaffold, where the observation already contains the environment state needed for the next decision.

### Defects and ambiguity that prohibit a direct port

The full-summary handler computes a summary but returns a prefix slice of the original messages instead of the summary projection (`work/gordon-work/src/infra/domain/memory/compactionHandler.ts:38-90`, especially `86-89`). That can discard the very summary the stage was meant to preserve. Collapse reinflation depends on an in-memory block list (`contextCollapse.ts:104-170`), so a process restart cannot necessarily reconstruct collapsed content from a durable content address. The orchestration path also needs a conservation test proving that a post-turn compacted state is the state persisted and restored; the scan did not find enough evidence to assert that invariant end to end.

**Decision:** do not port the compaction implementation. In a future heavy arm, reimplement the ideas with:

- transcript-conservation tests across every stage and restart;
- content-addressed offload artifacts in the evidence bundle rather than ephemeral path references;
- an explicit compaction event containing input hash, output hash, policy version, retained ranges, and model identity;
- an ablation comparing no compaction, deterministic truncation, and model summary;
- no ambient memory in the minimal arm.

## 3. Strategy generation, recipes, trial accounting, EdgeSpec, and genomes

### Gordon's generation pipeline

The generator advertises a natural-language-to-intent-to-DSL-to-validation-to-backtest-to-iteration pipeline (`work/gordon-work/src/infra/agents/tools/strategy-generator.ts:172-181`). The implementation is not suitable as benchmark ground truth:

- parse failures silently default fields (`strategy-generator.ts:460-519`);
- DSL failures fall back to a generated substitute (`525-582`);
- a no-exchange path can return a mock backtest (`183-359`);
- attempts are recorded after a successful backtest, not at the moment every raw candidate is generated (`642-718`).

Those recovery behaviors are reasonable user-experience choices in an assistant. In a search benchmark they change the search process, hide generator failures, and undercount the trials that must enter deflation.

The pure recipe primitives under `work/gordon-work/src/core/strategies/recipes/` are much cleaner: state-in/state-out signal gates and exposure/time controls can serve as published reference entrants. They do not justify importing the generator. SharpeArena's current constrained, non-executable DSL and raw-candidate-first trial ledger already implement the important search-accounting invariant more strongly.

### Trial tracker and DSR

`multipleTestingTracker.ts` appends attempts under a caller-provided family and computes its own DSR (`work/gordon-work/src/core/strategies/multipleTestingTracker.ts:1-290`). Family scope therefore remains an honor-system decision, and the generator wiring omits candidates that fail before a successful backtest. SharpeBench's corrected statistical kernel and SharpeArena's observed candidate ledger are authoritative. Maintaining a second DSR implementation would recreate the exact units and pooling risks the papers have worked to eliminate.

**Decision:** reject Gordon's tracker and DSR implementation.

### EdgeSpec: the strongest immediate idea

Gordon's `EdgeSpec` binds a strategy to a hypothesis, causal/mechanical story, intended regimes and instruments, invariants, kill conditions, and a verification plan (`work/gordon-work/src/core/edge/types.ts:1-52`). This is a useful complement to SharpeArena's generated DSL because it makes falsifiability part of the candidate artifact rather than prose added after selection.

Do not port the Gordon parser or monitor. Its monitor degrades on a missing invariant but does not necessarily retire on a missing kill condition (`work/gordon-work/src/core/edge/monitor.ts:18-72`), which is too permissive for a benchmark contract.

**Decision: implemented natively — concept only.** The closed `EdgeManifest` schema records:

- `hypothesis` and `mechanism`;
- target assets, timeframe, and regime claims;
- expected signal direction and invalidating observations;
- invariants checked before selection;
- quantitative kill conditions with evaluation horizon;
- verification plan and data split identity;
- generator model/scaffold identity and raw candidate ordinal.

Missing required fields must invalidate the candidate, not synthesize defaults. Kill-condition evaluation belongs to held-out evidence and must never feed candidate selection.

### Genome and evolutionary loop

Gordon's genome layer records lineage and mutation, which is useful provenance (`work/gordon-work/src/core/genome/types.ts`). The surrounding optimizer is not scientifically suitable: it combines arbitrary weighted fitness targets (`fitness.ts:8-110`), can mix live, paper, and backtest observations, performs random parameter nudges, weakens an empty filter by returning the original population, and automatically promotes winners on a schedule (`evolution-loop.ts:1-440`; `optimization-tier.ts:130-143`).

**Decision:** reject the fitness, scheduled evolution, random mutation, and automatic promotion. Later, the heavy strategy-generation arm may borrow only immutable lineage links, mutation-operator identity, and rejection memory. Selection remains a predeclared experiment; promotion requires held-out evidence and operator approval.

## 4. Evals, pass^k, process checks, and trace promotion

### What maps cleanly

Gordon deterministically derives scenarios from authoritative specifications and attaches provenance, rather than maintaining hand-written fixtures (`work/gordon-work/CLAUDE.md:75-106`). That is a sound regression-testing principle. Its deterministic process checker covers risk-before-order, approval/denylist order, loops, inconsistent outcomes, duplicates, future-looking recall, and poisoned memory (`work/gordon-work/src/infra/domain/evals/harness/process/processChecks.ts:1-334`). Its pass^k helper supports all, majority, and rate semantics (`harness/process/passK.ts:41-83`).

SharpeBench already has typed trading process events and gates, and the local-agent runner preserves those exact events. Its pass^k implementation is the one used by the statistical contract. Do not port duplicate helpers.

Gordon process checks inspect a coding/assistant-style sequence of named tool calls and often match tool names or substrings. That does not map directly to a trading environment whose canonical output is a typed `Decision`. The target analogue is a typed state machine over observation, decision, risk evaluation, submission, acknowledgment, fill, and reconciliation events. String-pattern checks would be brittle and would reward naming conventions rather than behavior.

### Trace promotion was a genuine gap and is now implemented

Gordon converts a real audit trace into normalized process/judge views, scores recent traces, appends flagged cases to a promotion queue, and requires operator silver-to-gold triage before freezing a regression scenario (`work/gordon-work/src/infra/domain/evals/harness/traces/traceAdapter.ts`; `traceScorer.ts`; `promotionQueue.ts`; `CLAUDE.md:104`).

SharpeArena already had an append-only point-in-time trace writer and replay surface (`sharpearena/crates/sharpearena-py/python/sharpearena/trace.py:1-18,83-216`), but did not have this promotion workflow when assessed. Its permissive loader can skip malformed records and substitute a zero reward for missing reward data; that remains useful for exploratory reading but is not used for promotion.

**Decision: implemented natively — concept only.** The promotion path:

1. reads in strict mode and rejects malformed/incomplete traces;
2. fingerprints the environment, model, scaffold, contract, data, and process-event sequence;
3. applies deterministic process checks before any model judge;
4. writes immutable silver candidates with the triggering check and source-trace hash;
5. requires explicit operator review to become gold;
6. freezes a minimal scenario plus expected invariant, not an overfit full transcript;
7. runs gold cases in CI without network or live model calls.

The k-run live producer in Gordon is sequential (`kRunProducer.ts:35-58`), while SharpeArena already has vectorized/sharded execution. The target implementation is stronger. Gordon's eval “sandbox” only redirects files, database, and environment; it is not an operating-system security boundary. Its dry-run synthesis and loose recent-trace matching must not enter evidence.

## 5. Audit and provenance

Gordon has two separate HMAC audit systems. The core system hashes redacted content with a previous signature and a local HMAC key, persists atomically, and can verify the chain (`work/gordon-work/src/core/audit/signing.ts:6-190,340-466`). The platform audit log independently signs individual canonical entries without the same chain (`work/gordon-work/src/infra/platform/audit/audit-log.ts:25-144,292-369`). The duplication creates two notions of “audited.” At least one field group (`absorptions`) is intentionally outside the signed material, so consumers must know the signed-field inventory rather than infer it from the serialized object.

HMAC proves consistency to a party holding the same secret. It does not provide public third-party attestation or non-repudiation; the key holder can rewrite the history and recompute the chain. SharpeBench already has an attestation crate, evidence hashes, trajectory capture/replay, and release provenance designed for externally verifiable research artifacts.

**Decision:** do not port either Gordon audit stack. Retain the target attestation chain. Borrow only two concepts if they are not yet explicit:

- redact secrets before hashing/signing so verification never depends on secret material;
- publish a machine-readable inventory of which fields are and are not covered by each evidence digest.

Any generated-strategy or paper-trading evidence should link its `EdgeManifest`, model digest, prompt/scaffold hash, trace hash, and environment commitment through the existing target provenance system.

## 6. Permissions, adaptive trust, risk, and execution safety

### Permission engine and trust trajectory

Gordon's permission engine supports allow, deny, and human queues, pattern rules, hook ordering, and extra protection for safety-critical tools (`work/gordon-work/src/runtime/permissions/PermissionEngine.ts:21-117,223+`). That is suitable for an interactive application where an operator grants authority over time. A deterministic benchmark should grant no mutable authority: the environment contract itself defines the allowed actions.

The adaptive trust trajectory learns auto-approval from repeated operator approvals while retaining hard exclusions (`work/gordon-work/src/runtime/permissions/trustTrajectory.ts:1-29,87-122,167+`). This makes safety behavior depend on historical operator interactions. It is therefore neither reproducible across hosts nor desirable in paper trading.

**Decision:** reject both for the benchmark. Keep SharpeArena's fixed, versioned paper-risk guard. If a human approval workflow is later added to paper trading, record approvals as evidence but do not let approval history weaken the guard.

### Risk classifier defect

The risk classifier validates inputs fail-closed and attempts a broad multi-dimensional pre-trade score (`work/gordon-work/src/infra/trading/risk/riskClassifier.ts:9-14,204-263`). The scan found a concrete calculation inconsistency: `drawdownRoom` subtracts current drawdown from `config.maxPositionPct`, while the adjacent explanation refers to `portfolio.maxDrawdownPct` (`riskClassifier.ts:356-363`). That mixes a position-size limit with a drawdown limit. The classifier also combines many heuristic dimensions into an application-specific composite.

**Decision:** do not port it. SharpeArena's risk rules should remain small, dimensionally typed, fixed by an evidence-versioned config, and tested at every boundary.

### Execution uncertainty and reconciliation

`terminationLayers.ts` identifies three distinct recovery layers and the important “query order state before retry” rule (`work/gordon-work/src/infra/trading/execution/terminationLayers.ts:7-29`). The module describes those layers but explicitly does not enforce all of them. The concept is sound and should be completed in the paper arm.

**Decision: port now — concept only.** Add an enforced paper-execution state machine:

```text
prepared -> submitted -> acknowledged -> partially_filled -> filled
                    \-> submission_unknown -> reconciled_(accepted|absent)
                    \-> rejected
```

On timeout after submission, never issue a replacement until the broker is queried using the deterministic client-order ID. Persist each transition and the raw broker acknowledgment hash. Batch preflight remains all-or-nothing before the first submission. This applies only to forward paper trading and carries no deterministic replay claim.

## 7. Brokers, exchanges, execution, and market data

Gordon has a normalized broker interface and factories for several broker/exchange providers (`work/gordon-work/src/infra/broker/types.ts:265-314`; `factory.ts:18-67`; `src/infra/exchange/factory.ts:104-157`). The Alpaca adapter defaults to paper endpoints and centralizes requests/redaction (`work/gordon-work/src/infra/broker/adapters/alpaca.ts:27-30,185-253`). Its OHLCV cache is a useful immutable, first-write-wins, venue/symbol/time/as-of store with content hashes (`work/gordon-work/src/infra/data/ohlcvCache.ts:129-210`).

SharpeArena already has the two connectors needed for its first local/forward program (Binance public market data and Alpaca paper trading), an in-memory broker, deterministic IDs, journal, risk guard, and a distinct forward evidence class. Importing Gordon's CCXT/broker layer would increase credentials and dependency surface without improving the experiment.

**Decision:** already exists for the planned scope. Add future venues one at a time behind SharpeArena's neutral connector interface only when a predeclared forward experiment needs them. An immutable as-of candle cache is reasonable for data ingestion, but the scored historical datasets remain frozen artifacts. A live-result or TTL cache must never decide which observation an agent sees.

Real-capital execution is outside this architecture. Paper connectors must make production endpoints unrepresentable or require an explicit build/runtime capability that is absent by default.

## 8. Proactive scheduling

Gordon's proactive observer is a durable application event loop: interval-driven producers emit news, regime, volatility, funding, and account events, with wake/reset handling, persistence, per-producer timeouts, health, proposition judging, backpressure, and outcome tracking (`work/gordon-work/src/infra/proactive/engine/observer.ts:1-315`; `producers/index.ts:11-40`).

This does not belong in a scored historical field. Adding RSS, news sentiment, account state, or an LLM proposition judge changes the information environment and makes Gordon's producer selection part of the treatment.

**Decision:** reject the proactive radar and its producers for benchmarking. A forward paper arm may later borrow only:

- a fixed, versioned schedule;
- idempotent missed-tick recovery;
- producer heartbeat/health evidence;
- strict separation between market-data acquisition and decision execution.

The schedule must be part of the forward commitment. No LLM judge decides whether a scheduled evaluation occurs.

## 9. Caching and observability

Gordon's tool-result cache uses per-tool TTLs and a hash of tool/input (`work/gordon-work/src/infra/agents/tooling/toolResultCache.ts:79-255`). Its wrapper comments describe opt-in behavior, but `isEnabled()` returns `true` unconditionally (`withResultCache.ts:8-39`). That discrepancy is especially dangerous for market data: two nominally identical calls can observe different real-world times, while a cache silently makes one reuse the other.

**Decision:** reject tool-result caching in scored and forward decision paths. Cache only immutable artifacts identified by content hash or complete as-of key. Prefix/KV caching may be used in a heavy local-model arm for efficiency, but it is a host optimization: disclose its configuration, record prompt hashes, and do not treat a reported provider cache hit as a scientific guarantee.

Gordon's observability is mostly local structured logging and in-memory metrics. Its external tracing surface is deliberately disabled/no-op (`work/gordon-work/src/infra/observability/tracing.ts:1-10,62-82,219-223`). It should not be described as distributed tracing. SharpeArena's evidence already records latency, token use, model identity, transport failures, and outcome data.

**Decision:** already exists for experimental metrics. A future heavy arm should add scaffold-specific counts—model calls, handoffs, compactions, offloaded bytes, loop interventions, and cache settings—to the same evidence row rather than adopt Gordon's application telemetry.

## 10. Sandboxing

Gordon uses “sandbox” for three materially different things:

1. `subprocessSandbox.ts` is an optional bwrap/Seatbelt wrapper. It is off by default, has no Windows implementation, and when enabled but unavailable can continue without isolation (`work/gordon-work/src/infra/safety/subprocessSandbox.ts:1-28,128-140,262-307`). This is fail-open.
2. The eval sandbox redirects file/database/environment paths. It isolates test state, not system calls, process creation, or networking (`work/gordon-work/src/infra/domain/evals/harness/live/sandbox.ts`).
3. `strategySandbox.ts` is an in-memory paper portfolio separated from production state; its “worktree” analogy describes state isolation, not hostile-code containment (`work/gordon-work/src/core/strategies/strategySandbox.ts:1-18`).

None should be ported as a security boundary. SharpeBench's Docker sandbox is stronger and fail-closed. It should remain the only untrusted-code execution boundary, with digest-pinned images, no network, read-only root, non-root user, dropped capabilities, explicit CPU/memory/PID limits, and no unsandboxed fallback.

Model-generated strategies should remain in SharpeArena's non-executable DSL for the primary experiment. If model-written code is later studied, it becomes a separate high-risk treatment inside the existing Docker boundary, with no broker credentials, no host write mount, and no network. The environment process—not the generated code—writes the authoritative evidence.

## 11. Duplication and coupling hazards

### Duplication found in Gordon

- Two HMAC audit systems define overlapping but different integrity claims.
- Multiple memory/context layers (Mastra memory, hot working memory, transcript, scratchpad, compaction projection, spill files) require conservation invariants that are hard to prove.
- Risk decisions occur across permissions, trust, risk classifier, tool hooks, and execution termination layers.
- “Sandbox” names three distinct guarantees.
- Strategy trial accounting exists beside generator fallbacks that can prevent failed raw candidates from being counted.

These are understandable in a mature application, but importing them would duplicate target systems and blur the research contract.

### Coupling rule for SharpeArena and SharpeBench

Keep **one-way dependency and two-way evidence compatibility**, not interdependent packages:

- SharpeArena may use neutral SharpeBench protocol/core/statistics crates.
- SharpeBench accepts Arena evidence through a versioned schema; it does not import the full Arena runtime.
- The evidence schema contains environment identity, scenario/data hash, model/scaffold identity, process events, cost, and outcome.
- Each product can validate the shared schema independently.
- Release compatibility is tested as a matrix, not enforced by circular imports.

This gives the user-facing composition they want—SharpeArena is the sandbox/environment and SharpeBench is the evaluator—without making the packages technically circular or forcing every scorer user to install an RL environment and local-model runtime.

## 12. Coding-agent patterns that do not map literally to trading

| Coding-agent pattern or analogy | Why literal reuse is wrong | Trading-native translation |
|---|---|---|
| Git worktree sandbox | A branch/worktree protects source files, not capital, credentials, network calls, or broker idempotency | Frozen simulation state for historical runs; fail-closed container for untrusted code; fixed paper-risk and broker reconciliation for forward runs |
| File permission prompts | File writes are usually reversible and local; orders are external, asynchronous, and may partially fill | Predeclared action contract, all-orders preflight, deterministic client-order ID, broker acknowledgment state machine |
| Doom loop over repeated tool calls | Repeated tool names can be valid during polling or multi-bar trading | Typed loop conditions over identical observation/decision fingerprints, unresolved broker state, and lack of state progress |
| Tool-call substring process checks | Tool names are scaffold-specific | Typed `ProcessEvent` ordering and invariants independent of tool naming |
| Context compaction as a default | A coding task accumulates repository context; the minimal trading observation is intentionally Markovian | No memory in minimal arm; explicit, separately scored memory/scaffold treatment |
| Orchestrator/researcher/executor as “the agent” | The topology contributes capability and cost | Named scaffold axis; same base model and cells in paired comparison |
| Auto-approve from trust trajectory | Repeated approval does not reduce market or execution risk | Fixed, versioned guard; human approval may add authority but never relax system limits |
| Automatic winning-strategy promotion | CI patches can be reviewed and reverted; strategy selection creates multiple-testing and deployment risk | Observed trial ledger, held-out selection, signed evidence, operator promotion only |
| Mock/fallback result to keep UX moving | A placeholder may be acceptable in an interactive coding draft | Any parse, generation, transport, or backtest failure is a typed failed cell, never a hold or synthetic success |
| Tool-result cache | Source/tool results are often immutable enough for convenience caching | Only content-addressed frozen data or complete as-of market-data keys; never TTL reuse in a scored decision path |

## 13. Implementation disposition

This began as an assessment backlog rather than a code port plan. Every item is
native to SharpeArena/SharpeBench; the D1–D4 candidates are now implemented and
the P1/P2 items remain conditional future treatments.

### Implemented candidate set

The assessment document itself implemented none of these. Subsequent work built
all four with the scope and caveats recorded in the current-status table above.

#### D1 EdgeManifest for generated strategies — implemented

**Source idea:** Gordon `EdgeSpec`.

**Owner:** SharpeArena strategy-generation evidence.
**Acceptance criteria:**

- closed schema with the fields listed in section 3;
- candidate invalid if a required field is missing;
- raw candidate and manifest recorded before parsing/validation;
- manifest hash linked to candidate, trial ordinal, model digest, scaffold hash, and split plan;
- kill conditions evaluated only out of selection sample;
- no Gordon runtime or parser dependency.

#### D2 Strict trace promotion — implemented

**Source idea:** Gordon silver-to-gold trace promotion.

**Owner:** SharpeArena trace/eval layer with SharpeBench typed checks.
**Acceptance criteria:**

- strict trace reader for promotion;
- deterministic fingerprint and reason for flagging;
- immutable silver queue;
- explicit human promotion record;
- frozen minimal gold scenario runnable offline in CI;
- regression scenario carries source trace and environment hashes.

#### D3 Paper-execution reconciliation state machine — implemented with a partial-fill caveat

**Source idea:** Gordon termination layers.

**Owner:** SharpeArena paper trading.
**Acceptance criteria:**

- explicit `submission_unknown` state;
- query by deterministic client-order ID before retry;
- idempotent reconciliation after restart;
- partial-fill representation;
- raw acknowledgment and transition hashes in forward evidence;
- no production endpoint or real-capital authority.

#### D4 Evidence coverage declaration — implemented

**Source idea:** lesson from Gordon's signed/unsigned audit fields.

**Owner:** shared evidence/attestation schema.
**Acceptance criteria:** machine-readable signed-field inventory and explicit exclusions; secrets redacted before hashing; schema tests fail when a new evidence field is neither covered nor explicitly excluded.

### P1 — after the first minimal local-model field is stable

#### P1.1 Heavy-scaffold treatment

Implement a separate scaffold with bounded tool use, optional researcher/executor roles, and complete intermediate-call evidence. Run a paired design:

- same model digest and quantization;
- same cell schedule, observation cadence, data, seed, and sampling settings;
- minimal versus heavy scaffold as the only intended difference;
- report capability, reliability, cost, latency, parse/transport failures, and process-gate outcomes.

Do not call this a model leaderboard column unless scaffold identity is part of the entrant identity.

#### P1.2 Deterministic context controls and offload

Reimplement bounded history and content-addressed result offload. First establish conservation tests and no-compaction baselines. Never copy the current Gordon full-summary handler.

#### P1.3 Typed scaffold-process checks

Add checks only for behaviors the heavy scaffold introduces: risk-before-submission, approval evidence, duplicated orders, unresolved execution state, poisoned/untrusted memory use, no-progress loops, and missing result dereference. Use typed events, not tool-name substrings.

#### P1.4 Lineage and rejection memory

Record parent candidate, mutation operator, rejected reason, and prior test exposure. Do not implement automatic evolutionary promotion or weighted Gordon fitness.

### P2 — forward paper arm only

- fixed scheduling and missed-tick recovery;
- connector health and acquisition latency evidence;
- additional paper brokers only as experiments require;
- immutable as-of data ingestion cache;
- optional human approval evidence that cannot relax fixed limits.

No proactive news/sentiment producer, adaptive trust, or LLM proposition judge belongs in the canonical field unless the information environment itself becomes a declared experimental axis.

## 14. Explicit rejection register

The following should not be revisited without new evidence:

1. **Gordon as the common runtime.** It makes the benchmark scaffold-dependent and couples UI/operator state into experiments.
2. **Circular SharpeArena/SharpeBench packages.** Use a neutral protocol/evidence layer and directed dependencies.
3. **Generator fallbacks or mock backtests.** Failures are evidence, not holds or substitute strategies.
4. **Gordon's trial tracker/DSR.** The target statistical kernel is authoritative; raw candidates must be counted before validation.
5. **Genome auto-promotion and heuristic fitness.** They contaminate selection, mix evidence classes, and weaken held-out claims.
6. **Adaptive trust.** Historical operator behavior must not relax a research or trading safety boundary.
7. **Risk-classifier port.** It is heuristic, application-coupled, and contains a position/drawdown dimension bug.
8. **Any Gordon sandbox as hostile-code isolation.** The subprocess path is fail-open, the eval path redirects state only, and the strategy path is a paper portfolio.
9. **Either Gordon HMAC stack.** They duplicate stronger target provenance and do not provide public attestation.
10. **TTL caches in scored or forward decision paths.** Only immutable/content-addressed or complete as-of caching is admissible.
11. **Application proactive radar in the canonical benchmark.** It changes the information environment and introduces external drift.
12. **Coding-agent string checks as trading process gates.** Use typed trading lifecycle events.

## 15. Final architecture recommendation

The complete local system should have four explicit layers:

```text
Layer 1: entrant
  model weights + quantization + sampling + named scaffold

Layer 2: SharpeArena
  prompt/contract adapter -> deterministic environment -> typed trajectory
  trusted local-model process selected by the operator
  separate nondeterministic forward paper arm

Layer 3: shared evidence contract
  model/scaffold identity, environment/data commitments, process events,
  costs, trace hashes, EdgeManifest, signed-field coverage

Layer 4: SharpeBench
  statistics, deflation, pass^k, process gates, audit, leaderboard/reporting
  fail-closed digest-pinned Docker boundary for untrusted entrants
```

Gordon contributes design hypotheses to layers 1 and 3 and may later define one layer-1 scaffold. It should not sit between SharpeArena and SharpeBench, own their evidence, or become a package dependency. This keeps the environment falsifiable, the scorer independent, and the effect of sophisticated agentic scaffolding measurable rather than silently baked into every result.

The current build keeps Gordon out of the runtime and package graph. It has
implemented `EdgeManifest`, strict trace promotion, paper-order reconciliation,
and explicit evidence coverage as native Sharpe-suite concepts. No local-model
field has been run or admitted as evidence. A Gordon-inspired heavy scaffold
should be considered only after a minimal-field result exists, and then only as
a paired experimental arm.
