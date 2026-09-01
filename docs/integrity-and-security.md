# Integrity and security

SharpeArena makes several distinct guarantees. Keeping them separate prevents
“sandbox” from becoming an umbrella claim stronger than the code.

| Property | Mechanism | Scope |
|---|---|---|
| Point-in-time leak freedom | The environment owns the cursor; observations expose history only; `LookaheadGuard` refuses future reads. | Agent/environment information flow. |
| Decision integrity | Raw decisions are journaled and replayed through the frozen engine; transport faults become failed cells rather than empty holds. | A pinned engine, data/config artifact, and trajectory. |
| Cross-surface semantics | `SPEC_HASH`, exact engine dependency pins, canonical JSON goldens, native/WASM/npm/Python parity tests. | Published wrappers and the engine they load. |
| Arm identity | Effective configuration is read back from the constructed environment and compared before evidence is written. | Evidence-producing scenario arms. |
| Seed custody | `SealedSalt` enforces a 16-byte floor, redacts debug output, avoids serialization/display/deref, and derives disjoint held-out seeds. | Commit-reveal protocol; not a certified cryptographic PRF or custody service. |
| Input closure | JSON Schemas, `deny_unknown_fields`, path containment, and typed boundary errors. | Caller-controlled configuration and plans. |
| Artifact integrity | Atomic provenance generation, code-owned scopes, exact artifact hashes, clean-generation Git blob checks, offline pack/import tests. | Repository/release evidence and packages. |

## Leak freedom is not containment

SharpeArena runs the model process selected by the operator in the operator's
environment. It does not isolate that process from the filesystem, network,
credentials, or kernel and should only run trusted local models and scaffolds.

The sibling SharpeBench product owns the fail-closed OCI path for untrusted
entrants. Its `--image <repository@sha256:...>` route is exercised in a
Docker-enabled CI job with a pinned fixture. That is a real container boundary,
but still not a microVM, a proof against kernel escape, or a hosted multi-tenant
service. See the [SharpeBench arena chapter](https://github.com/general-liquidity/sharpebench/blob/main/docs/book/src/arena.md).

## Failure is not a hold

An empty order set is a deliberate action that carries the existing position.
A timeout, malformed response, unknown/duplicate symbol, invalid action/weight
pair, or transport loss therefore cannot be converted into empty orders without
corrupting the return series. Python's field scheduler records a typed failed
cell. Rust wire consumers use `run_backtest_checked` with `TransportHealth` to
obtain the same rule; the unchecked in-process engine remains available for
trusted policies.

## Provenance scope

The committed manifest binds source text with CRLF/LF canonicalization and
result artifacts byte-for-byte. The checker owns the admitted scopes and
exclusions, rejects empty scopes, validates claimed clean-generation bytes from
Git, and treats the manifest as the sole digest authority. The binding commit
describes the branch/PR tip it is generated for; intermediate source commits can
carry a manifest that has not yet been rebound.

No model artifact appears in the current manifest. An empty model-artifact list
means no model result has entered the evidence, not that model identity was
omitted.

## Security non-claims

- `sealed_seed` is a deterministic keyed derivation used in a bounded
  commit-reveal protocol, not a formally analyzed MAC.
- Synthetic-market red-team results are finite-grid diagnostics, not proofs over
  every impact model or schedule.
- CI package smoke tests prove installation and one public call, not every
  downstream environment.
- A clean provenance check establishes tree/artifact consistency, not the truth
  of a scientific interpretation. The paper and its evidence scripts carry that
  burden.
