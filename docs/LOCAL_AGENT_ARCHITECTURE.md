# Local open-weight agent architecture

SharpeArena owns the point-in-time world and the agent interaction. SharpeBench owns
field-level statistical judgment. They compose through a validated evidence artifact,
not a cyclic package dependency:

```text
sharpebench protocol/sim/core (published leaf crates)
                    ↓
       SharpeArena environment + local model policy
                    ↓
  append-only cell journals + exact decisions/process trace
                    ↓
       sharpearena-compile-bench (artifact boundary)
                    ↓
        SharpeBench score / audit / leaderboard
```

SharpeArena depends on the published SharpeBench leaf crates so the environment and
scorer share one wire contract and one native execution model. SharpeBench does not
import SharpeArena back into its core. `sharpearena-compile-bench` validates a complete
Cartesian field and emits ordinary SharpeBench submissions. This keeps the dependency
graph acyclic while making the two products operationally interdependent.

## Guarantee boundary

- The market environment, costs, windows, and recorded-decision replay are
  deterministic.
- Model inference is not claimed to be byte deterministic. Its artifact digest,
  server version, sampling settings, inference seed, cadence, response hashes, token
  counts, and per-request latency are evidence. Every duration states whether it came
  from the backend or a host monotonic clock.
- In the direct-policy scheduler, an inference or schema failure fails that cell and
  cannot become a scoreable return series. The stdio compatibility adapter necessarily
  returns a flagged hold because the upstream `Agent` trait has no error variant;
  `run_backtest_checked` is the explicit gate that refuses its recorded transport fault.
- The local monetary cost is recorded as zero self-reported USD, with token counts kept
  separately. The harness does not mislabel tokens or wall time as dollars.
- A field entry needs a public source URL, exact source revision, and license identifier. Locally modified or
  untraceable models are host or unverified-local entries, never independent field
  evidence.

## Direct-policy field

`sharpearena.local_agents` supplies:

- a scaffold-neutral, stateless prompt renderer;
- a loopback-only Ollama client with redirect refusal and the published Decision JSON
  Schema as its constrained-output format;
- native `VecTradingEnv` batch stepping;
- a Cartesian scheduler over model, dataset/tier, seed, and repetition;
- stable cell ordinals and hashes, deterministic sharding, and append-only resume;
- exact native process events, canonical decisions, observation/response hashes,
  return series, and per-cell score diagnostics. Model identity also records the
  quantizer/converter, server commit, decoding/parser stack, cache and parallelism
  settings, accelerator/runtime identity, and labels unavailable backend facts as
  unresolved rather than inferring them.

The plan schema is closed. Unknown keys, duplicate model arms, duplicate dataset IDs,
invalid seeds, unsafe windows, and invalid cost controls fail before inference.

Inspect the no-network smoke plan:

```bash
python -m sharpearena.local_field_cli \
  --plan examples/local-agents/field-plan-smoke.json \
  --evidence local-evidence/smoke.jsonl \
  --inspect
```

Run it only after the model inventory and plan are frozen:

```bash
python -m sharpearena.local_field_cli \
  --plan examples/local-agents/field-plan-smoke.json \
  --evidence local-evidence/smoke.jsonl
```

Shards use the same plan hash. Change only `shard_index` and `shard_count`, write each
shard to a separate journal, then compile all journals together:

```bash
python -m sharpearena.bench_bridge \
  local-evidence/shard-0.jsonl local-evidence/shard-1.jsonl \
  --output-dir local-evidence/bench
```

The bridge refuses incomplete grids, failed cells, coordinate collisions, conflicting
duplicates, bad return hashes, or misaligned confidence/outcome sequences. A failed
attempt followed by one completed retry is valid; the source-journal hash commits to
both attempts. A completion followed by another conflicting record is not valid.
Raw-field schema 2 also requires nonnegative inference accounting, one duration sample
per scheduled model call, exact agreement between the samples and total duration, and
an observation source. Bridge schema 2 publishes nearest-rank p50/p95 duration, token
totals, reasoning-token provenance, and retries for each model with `rank_input: false`.
These fields support operational diagnosis and capacity planning; they cannot alter the
score submission emitted beside the manifest.

## Stdio compatibility path

`python -m sharpearena.ollama_shim --model TAG` translates the SharpeBench newline
wire protocol to constrained local inference. It writes one Decision per observation
and exits nonzero on any fault. SharpeBench's
`local_open_weight_field_eval` example exercises this path directly against its frozen
historical panels. That example is a compatibility and scorer regression path; the
canonical product composition is the Arena journal and Bench artifact bridge above.

## Strategy generation with observed trials

`sharpearena.strategy_generation` never executes generated Python or shell code. A
model emits a closed JSON DSL containing bounded combinations of price, SMA, EMA,
momentum, RSI, and volatility conditions. The host counts every raw candidate before
validation or deduplication. That observed count is the DSR trial count.

Accepted candidates are evaluated on a validation split. The deterministic selection
rule is median per-seed DSR with candidate ID as the tie-break. Only the winner touches
the disjoint test split. If validation and test use the same underlying bytes, both
half-open windows must be explicit and non-overlapping. The evidence retains the exact
prompt, raw generation response, accepted and rejected candidates, split metadata,
seeds, score records, and hashes.

Each ledger row also carries a host-derived strategy-family digest, the exact generator
identity, and resolved lineage. A candidate may name only an earlier candidate as a
parent and may cite only an idea-source digest registered by the operator in the search
plan. Source records bind exact content bytes and may add a URL or DOI, immutable
revision, authors, and license. The host resolves candidate IDs to raw-candidate digests
before writing the row, then binds the family, ancestry, generator, sources, and raw
candidate in a separate lineage hash. The strict reader recomputes every v2 digest and
rejects forward ancestry, source mismatch, or tampering.

Family grouping is diagnostic. It can show whether one conceptual signal is robust
across retuned windows, thresholds, or exposure, but it never merges proposals and
never lowers the DSR trial count. Invalid proposals and duplicates remain observed
trials. Plans and responses without source or lineage fields remain valid, so the v1
workflow continues to run while emitting the richer v2 evidence.

```bash
python -m sharpearena.strategy_cli \
  --plan examples/local-agents/strategy-search-smoke.json \
  --evidence local-evidence/strategy-search.jsonl \
  --inspect
```

This closes one specific weakness: trials generated inside this harness are observed,
not declared. It also makes ancestry and cited research auditable for candidates created
inside the run. It does not reveal searches performed before an entrant was submitted,
prove that a cited source caused an idea, or make family membership a scoring input.

## Isolation model

There are two trust zones:

1. The model server and minimal inference adapter are trusted local services. The
   shipped Ollama adapter accepts only a loopback origin (`127.0.0.1`, `::1`, or
   `localhost`) and refuses redirects.
2. Untrusted executable agents run through SharpeBench's Docker sandbox: digest-pinned
   image, no network or IPC, read-only root, non-root UID, all capabilities dropped,
   no-new-privileges, Docker's default seccomp policy, bounded CPU/RAM/PIDs/file
   descriptors, and small `noexec,nosuid,nodev` tmpfs mounts. Docker absence is an
   error by default; the sibling launcher exposes a separately named, owner-authored
   local-development opt-in for unsandboxed execution.

The strategy-generation path deliberately avoids executable code, so it does not need
zone 2. A future code-writing scaffold must cross that boundary and must never receive
broker credentials or host paths. See `SANDBOX_ENVIRONMENT_RESEARCH_2026.md` for the
threat-model comparison and upgrade path.

## Paper-only forward arm

Forward evidence is a different class from deterministic backtests. It explicitly
carries no replay guarantee. `sharpearena.paper_trading` includes:

- unauthenticated read-only Binance spot bars;
- authenticated read-only Alpaca stock bars;
- an in-memory paper fill adapter;
- an Alpaca adapter hard-pinned to `https://paper-api.alpaca.markets`;
- a deny-first host-side risk guard with symbol, notional, gross exposure, daily-loss,
  drawdown, shorting, and kill-switch controls;
- batch preflight before the first remote paper order;
- append-only decision, market snapshot, proposed-order, refusal, and paper-response
  evidence;
- a byte-compatible SharpeBench forward-window commitment plus a separate private
  reveal preimage.

The CLI accepts no credentials in its plan. It reads the fixed Alpaca environment
variables only when an Alpaca adapter is selected. Remote paper submission additionally
requires `--allow-remote-paper-submit`. There is no real-capital endpoint or override.

```bash
python -m sharpearena.paper_cli execute \
  --plan examples/local-agents/paper-plan-safe.json \
  --decision examples/local-agents/hold-decision.json \
  --evidence local-evidence/forward-paper.jsonl \
  --inspect
```

Prepare a public commitment without exposing the salt:

```bash
export SHARPEARENA_FORWARD_SALT='<random value without | or newlines>'
python -m sharpearena.paper_cli commit \
  --agent-id local-open-weight-policy \
  --target-window window-002 \
  --artifact-manifest local-evidence/agent-manifest.json \
  --commitment local-evidence/public-commitment.json \
  --private-preimage private/forward-preimage.json
```

Never commit the private preimage before the reveal deadline.

After the deadline, combine the scored submission, published commitment, and private
preimage into the exact `RevealedEntry` array accepted by `sharpebench arena score`:

```bash
python -m sharpearena.paper_cli reveal \
  --submission local-evidence/submission.json \
  --commitment local-evidence/public-commitment.json \
  --private-preimage private/forward-preimage.json \
  --output local-evidence/revealed-entry.json
```

## Model-server diversity

The scheduler depends on the `DecisionModel` protocol, not the Ollama class. Ollama is
the first backend because it is installed on the Windows host and supports constrained
output. The shipped OpenAI-compatible backend covers llama.cpp, vLLM, SGLang and
compatible local servers through an explicit closed identity manifest. Both backends
apply the same fail-closed Decision validation; backend name, version, quantization and
offload remain experimental axes because changing engines silently would confound the
field.

## CI boundary

CI uses deterministic model doubles and injected HTTP transports. It validates schema
closure, prompt/cadence behavior, sharding/resume, exact process traces, failure
semantics, strategy trial counting and split isolation, paper-only origin enforcement,
risk refusal, commitment compatibility, and the Arena-to-Bench bridge. It does not
download weights, call a model server, access live market data, or run a substantive
field.
