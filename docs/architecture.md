# Architecture

SharpeArena produces point-in-time trajectories; SharpeBench scores and attests
them. The split is directed so there is one execution model without a package
cycle.

```text
agent (any language)
        │ Observation → Decision
        ▼
SharpeArena
  scenario + PIT cursor + execution + process trace
        │ append-only decisions, returns, effective config
        ▼
SharpeBench
  deflation + pass^k + significance + process/mandate gates
        │
        ▼
attested leaderboard / forward window
```

SharpeArena depends on the small published SharpeBench protocol, simulator, and
scoring crates. SharpeBench does not depend on the full SharpeArena package.
`sharpearena-compile-bench` validates the completed field grid and emits ordinary
SharpeBench submissions.

## Package topology

```text
sharpebench-protocol + sharpebench-sim + sharpebench-core
                         │
                  crates/sharpearena
                 /          |          \
 crates/sharpearena-wasm  pyo3 core   Rust users
             │               │
 npm/sharpearena      crates/sharpearena-py
                              │
             Gymnasium · PettingZoo · verifiers · MCP · field CLIs
```

| Surface | Responsibility |
|---|---|
| `sharpearena` | `TradingEnv`, vector stepping, procedural scenarios, mandates, execution noise, market clearing, LOB, contract, transport-fault gate, and reference agents. |
| `sharpearena-wasm` | JSON/wasm-bindgen entry points over the same native kernels. |
| `@general-liquidity/sharpearena` | Typed TypeScript wrapper over the committed WASM package. |
| `sharpearena` on PyPI | pyo3 engine plus Gymnasium, PettingZoo, `verifiers`, Minari, wrappers, local-field, strategy-search, paper-trading, and evidence adapters. |
| observation/decision JSON | Language-neutral external-agent contract; reference programs in Rust, Python, and TypeScript demonstrate the contract. |

## Identity and compatibility

`CONTRACT_VERSION` governs the additive observation/decision wire. Closed JSON
Schemas reject unknown caller fields, and conformance tests compare serialized
keys in both directions.

`SPEC_HASH` is separate: it fingerprints the exact files and dependency pins that
define generated tape semantics. Rust, pyo3, WASM, and npm expose the value; a
wrapper refuses an engine built against another hash instead of returning a
plausible but wrong number. Canonical pre-hash JSON makes a golden drift readable
before the compact fingerprint assertion fires.

## Deterministic and adapter layers

The determinism-critical path is Rust: data, cursor, scenario generation,
mandates, execution noise, market clearing, scoring inputs, and replay. Ecosystem
adapters stay in the language whose interface they implement. Training rewards
shape optimization only; published ranking remains the SharpeBench kernel.

The effective-configuration gate reads dimensions, windows, seeds, and scenario
fingerprints back from the environment that consumed them. This prevents a
driver from labelling one arm while silently running another—a failure that
output hashes alone cannot detect.

## Release topology

One workspace version covers Rust, npm/WASM, and Python. The release driver cuts
from freshly fetched `origin/main` in a temporary worktree, promotes pushed and
non-empty notes, runs the version replacements, commits the clean candidate,
regenerates provenance, validates the annotated tag, and atomically pushes branch
and tag. Registry jobs check out the exact validated object and smoke-test the
published surfaces.
