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
  counts, and latency are evidence.
- An inference or schema failure fails that cell. It is never converted into a hold or
  a flat return series.
- The local monetary cost is recorded as zero self-reported USD, with token counts kept
  separately. The harness does not mislabel tokens or wall time as dollars.
- A field entry needs a public source URL and license identifier. Locally modified or
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
  return series, and per-cell score diagnostics.

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

```bash
python -m sharpearena.strategy_cli \
  --plan examples/local-agents/strategy-search-smoke.json \
  --evidence local-evidence/strategy-search.json \
  --inspect
```

This closes one specific weakness: trials generated inside this harness are observed,
not declared. It does not reveal searches performed before an entrant was submitted.

## Isolation model

There are two trust zones:

1. The model server and minimal inference adapter are trusted local services. The
   shipped Ollama adapter accepts only a bare loopback origin and refuses redirects.
2. Untrusted executable agents run through SharpeBench's Docker sandbox: digest-pinned
   image, no network or IPC, read-only root, non-root UID, all capabilities dropped,
   no-new-privileges, Docker's default seccomp policy, bounded CPU/RAM/PIDs/file
   descriptors, and small `noexec,nosuid,nodev` tmpfs mounts. Docker absence is an
   error, never an unsandboxed fallback.

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
- a deny-first native risk guard with symbol, notional, gross exposure, daily-loss,
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

## Model-server diversity

The scheduler depends on the `DecisionModel` protocol, not the Ollama class. Ollama is
the first shipped backend because it is already installed on the Windows host and
supports constrained output. Backend identity belongs in each model record. A future
llama.cpp, vLLM, SGLang, or Transformers adapter must preserve the same fail-closed
Decision validation and record the engine/version/quantization as an experimental axis;
changing inference engines silently would confound the field.

## CI boundary

CI uses deterministic model doubles and injected HTTP transports. It validates schema
closure, prompt/cadence behavior, sharding/resume, exact process traces, failure
semantics, strategy trial counting and split isolation, paper-only origin enforcement,
risk refusal, commitment compatibility, and the Arena-to-Bench bridge. It does not
download weights, call a model server, access live market data, or run a substantive
field.
