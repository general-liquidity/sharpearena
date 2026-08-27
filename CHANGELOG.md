# Changelog

All notable changes to SharpeArena are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). One workspace
version covers the Rust crates, the npm package and the PyPI package; each
section is one `v*` tag and links the commits it was built from. The wire
contract has stayed at `CONTRACT_VERSION` 1.0 throughout.

[Unreleased]: https://github.com/general-liquidity/sharpearena/compare/v0.19.0...HEAD

## [Unreleased]

### Changed
- release: `cargo release` no longer applies the tag. The pre-release hook necessarily runs while the version bump is uncommitted, so the manifest it writes into the release commit binds that commit's tree but can only record a dirty generation. The tag now goes on the rebind commit that follows, whose manifest names the release commit and is checked against it. `RELEASING.md` carries the two commands and the reason.

### Fixed
- paper: the appendix and the README stated the tamper-evidence gate without stating its scope. It establishes a property of the commit the job runs on, the tip of a pushed branch and the head of a pull request, and not of every commit in the history: the manifest is bound in a commit of its own after the work it describes, so a commit that moved a source file carries a manifest that has not caught up until the binding commit lands. The appendix now says that, and says what a dirty generation leaves unobservable.
- paper: `check-provenance.py` holds a manifest that records `generated_at_head_dirty: false` to what that claims. Every recorded digest is compared against the bytes committed at `generated_at_head`, read back out of the repository. Flipping the flag by hand previously produced a file indistinguishable from one regenerated on a clean checkout, which was the one field in the manifest that nothing could contradict.

## [0.19.0] - 2026-08-27

### Added
- test: the WebAssembly leg of the byte-identity guarantee is executed rather than assumed. The canonical and clustered scenario pins live in `crates/sharpearena/contract/attestation/scenario-goldens.json`, and three gates read that one file: the native suite, `wasm-pack test --node`, which runs the exported entry point inside a WebAssembly module, and the npm suite, which runs it against the committed `pkg/sharpearena_bg.wasm` the package publishes. Before this the goldens were plain `#[test]` compiled for the host, CI only built the wasm32 target, and `npm test` had no fingerprint assertion, so the shipped binary was published without any gate having recomputed it.
- test: both provenance scripts have tests. They run end to end against a throwaway checkout and cover the unresolved-digest rejection, the line-ending case and the dirty-flag case; the model-artifact scope holds no data in this repository, so neither half had been exercised.
- test: three `plan_sha256` digests are pinned by moving one bound field at a time, so dropping a field from a hashed payload fails instead of passing.
- ci: the evidence-provenance check runs on every push and pull request, and the wasm32 scenario goldens run on the ubuntu leg.

### Changed
- npm: `pkg/package.json` carried version 0.0.6 against a 0.18.0 wrapper. It now carries the crate version, and a test pins both it and the wrapper to the workspace version, so the published wasm package cannot drift from the crate it was built from.
- py: `pip install sharpearena[minari]` declared only `minari`, which reproduces the environment CI had while every Minari test skipped: h5py sits behind minari's `hdf5` extra, jax behind `create`, and its HDF5 storage imports PIL without declaring pillow at all. The extra is now `minari[create,hdf5]` plus pillow, and tests hold it and `ci-requirements.txt` in agreement.

### Fixed
- paper: provenance digests are taken with CRLF collapsed to LF for text files, so the manifest validates on a `core.autocrlf` checkout. Twenty files read as mismatches on Windows and none of them were real drift.
- paper: `check-provenance.py` shares one rule module with `make-provenance.py`, so a model identity file whose checkpoint digest, quantization, server or server version reads as unresolved is now refused by the validator as well as the writer. It also re-derives `generated_at_head_dirty` instead of believing it, and fails a manifest claiming a clean generation against a tree with uncommitted work.
- test: every strategy-DSL indicator is pinned to its exact value rather than to a one-sided inequality. The 0.18.0 pins had up to 4.7x slack, so a volatility returning a sample rather than population deviation, or an RSI returning 99 rather than 100, passed unchanged. Boolean operators now carry a false-side case.
- test: the assertions that could not fail are gone. `generated_code_executed is False` asserted a literal written by the evidence builder itself and is now backed by an AST scan of the modules on the path from model output to decision plus a refused executable indicator name; the credential test asserted a secret was absent from a dict built out of eight named keys and now pins that key set with the secret genuinely on the wire; and four truthiness-only digest assertions are recomputed independently.

## [0.18.0] - 2026-08-26

### Added
- py: the local-model record now binds an immutable publisher revision, model artifact and runtime identity, context and optional thinking budgets, accelerator details, raw responses and hashes, provider-token availability, retry count, and explicit unresolved values for backend facts that cannot be observed. The release provenance manifest has a separately validated model-artifact scope that exposes checkpoint digest, quantization, server and server version. The scope is empty in this repository, so an empty list means no model result has entered the paper rather than that model provenance was omitted.
- py: the forward paper arm now opens its public commitment into the exact SharpeBench `RevealedEntry` shape. A shared fixture is checked by both Python and the published Rust `sharpebench-attest` primitive, so delimiter or hashing drift fails cross-language CI.
- test: real loopback transport tests cover invalid JSON, HTTP failure, redirect refusal, unreachable endpoints, concurrent lane ordering and per-lane faults. Field shards are executed and recombined, every strategy-DSL indicator and Boolean operator is exercised against a one-sided threshold, and all terminal field-failure branches are exercised. (The thresholds were not tight; see Unreleased, where they became equalities.)

### Changed
- py: sparse `Decision.orders` now preserve the current weight of omitted symbols. Confidence is recorded only when the model supplied it; no synthetic 0.5 enters calibration evidence. Independent field evidence requires a public URL, immutable source revision and license, while host and unverified entries remain runnable but cannot cross the SharpeBench bridge.
- py: generated-strategy evidence is append-only JSONL. Every raw trial first enters a manifest-bound ordinal ledger, selection candidates cross the benchmark boundary, and repetitions remain the explicit pass^k axis while execution noise stays one seed per recorded run.
- ci: Python optional conformance dependencies are pinned and installed in CI, eliminating environment-dependent silent skips.
- py: about twenty-five names became public API through `__all__`, among them `LocalAgentError` and the `ModelHttpError` / `ModelTransportError` / `ModelResponseError` / `DecisionResponseError` split, `DecisionModel`, `FieldCell`, `InferenceOutcome`, `InferenceResult`, `OpenAICompatibleClient`, `load_identity_manifest`, `CandidateRejection`, `GenerationResult`, `StrategyGenerator`, `StrategyProtocolError`, `evaluate_condition`, `parse_generated_pool`, `strategy_decision`, `make_forward_commitment`, `prepare_forward_window_reveal` and `target_weights_to_orders`.
- mcp: `step` before `reset` is a named state. The tool returns `{"error": "episode_not_reset", "environment_advanced": false}` instead of acting on an absent observation, which is a new error contract on a shipped tool surface.
- py: the `nonpositive-equity` refusal is removed from the paper-execution risk guard. It was unreachable: `AccountSnapshot.__post_init__` requires finite and strictly positive equity and `reconcile_account` is the only mutator, `_projected_gross` keeps its own `equity <= 0` guard, and the daily-loss branch fires at zero equity regardless. A removed deny-list branch belongs in a changelog even when it was dead.
- cli: `sharpearena-ollama-shim` and the OpenAI-compatible shim take `--context-tokens`.

### Fixed
- py: `sharpearena.__version__` reported `0.16.0`. The v0.17.0 release therefore shipped a package that identified itself as two versions older than its own metadata; it now reports the workspace version.
- py: model-server failures retain typed HTTP, transport, response or Decision fault classes and the raw-response hash where bytes existed. Unsupported numeric thinking budgets fail before a request is sent.
- py: Binance and Alpaca market-data adapters reject malformed payloads consistently, and their complete OHLCV mappings and exact request normalization are tested.
- docs/paper: containment, host-side risk, target/action semantics, forward commitment, OpenAI-compatible serving and reproducibility claims now match the shipped boundaries.

## [0.17.0] - 2026-08-26

### Added
- py: `EdgeManifest`, a closed schema binding a generated candidate to its hypothesis, the mechanism it claims, the regimes and instruments it claims them in, its invariants, quantitative kill conditions and a verification plan. Falsifiability is part of the candidate artifact instead of prose written after selection. The schema is closed: a missing required field invalidates the candidate rather than synthesizing a default, and every threshold carries an explicit unit drawn from a closed enum, so a table cannot mix basis points, dollars and unit fractions in one untyped column. The manifest is recorded with the raw candidate and its trial ordinal, before validation or deduplication, and kill conditions are evaluated only outside the selection sample ([89c77cf](https://github.com/general-liquidity/sharpearena/commit/89c77cf)).
- py: strict silver-to-gold trace promotion, which turns a flagged production trace into a frozen regression scenario. A malformed or incomplete trace is rejected rather than skipped. The fingerprint is deterministic over environment, model, scaffold, contract, data and the process-event sequence. Deterministic checks run before anything else, silver candidates are immutable and carry the triggering check with the source trace hash, promotion to gold requires a recorded operator decision, and what is frozen is a minimal scenario plus its expected invariant rather than a transcript. The existing permissive reader is untouched: strictness is a separate mode, not a tightening of the exploratory path ([94197b3](https://github.com/general-liquidity/sharpearena/commit/94197b3)).
- py: a counterfactual ledger used by the paper-execution arm. Every intended paper order gets a record whether it was executed, resized, refused or not submitted, so the gap between what was considered and what was executed is measurable rather than invisible. The historical local-field scheduler does not yet instantiate it because the native target-weight step surface does not expose the per-order fill quantities the ledger requires ([12c1d94](https://github.com/general-liquidity/sharpearena/commit/12c1d94)).
- py: `submission_unknown` is a representable paper-execution state. A forward order could previously be submitted, acknowledged, filled or rejected, but there was no way to say the submission got no verdict at all. The state is reachable only from `submitted` and leaves only to `reconciled_accepted` or `reconciled_absent`; ack latency is `None` until a real acknowledgment arrives and is computed from two recorded timestamps. A replacement is permitted only after the broker confirms absence, queried by the deterministic client order id, and exactly once; a broker that cannot be queried raises rather than resolving anything, so an unanswerable query can never be mistaken for a confirmed absence. The store is written before the submit call, on the unknown transition and after every reconciliation, so an order left unresolved by a crash is still unresolved after a restart rather than silently resubmitted. Account state, previously read from the plan file and never reconciled, is now reconciled against the broker while keeping the session anchor and ratcheting peak equity. Every transition and the hash of each raw broker acknowledgment reaches the forward evidence, which stays distinct from deterministic backtest evidence and carries no replay guarantee. No real-capital path is added: the origin is validated and then reassigned unconditionally to the paper host, both new calls are GETs that read state, and the deny-first risk guard, redirect refusal and market-only restriction are untouched ([ee32c33](https://github.com/general-liquidity/sharpearena/commit/ee32c33)).

### Changed
- paper: `paper/evidence/provenance.json` is re-pinned over this work and passes at 117 sources and 29 artifacts ([4c5cb88](https://github.com/general-liquidity/sharpearena/commit/4c5cb88)).

### Fixed
- transport: a masked hold is a failed cell, never a return series. The external transports cannot signal an error through the `Agent` trait, so a wire fault returned an empty-orders hold and recorded the fault into a `TransportHealth`. Against the native engine an empty-orders decision is a **true hold**: the position persists. A wedged agent therefore rode its last position for the rest of the window, the run completed, and it yielded a return series indistinguishable from a deliberately conservative agent's. The health record was the only mitigation, and the crate re-exported neither `TransportDiagnostics` nor `TransportHealth`, so no consumer could name the type, let alone read it; a repository-wide search found zero readers. `transport_gate` is the reader: `run_backtest_checked` runs an external agent and converts any recorded fault into a typed failed `CellOutcome`, and the diagnostics types are re-exported alongside the transports they describe. A failure is evidence, never a hold ([7d633d7](https://github.com/general-liquidity/sharpearena/commit/7d633d7)).

### Known limitations
- py: partial-fill accumulation in the paper arm is last-write-wins rather than summing fill deltas, so a sequence of partial fills records the last reported quantity instead of the accumulated one.
- py: `reconcile_all` has no retry. An unanswerable broker query raises and halts the run, which is the safe direction, but it stops a session on a flaky broker.

## [0.16.0] - 2026-08-26

### Added
- py: a local open-weight field runner with a fixed prompt scaffold, constrained Ollama inference, strict canonical `Decision` validation, native vector stepping, cadence/thinking controls, stable sharding, append-only resume, exact process traces and model/runtime provenance; `sharpearena-compile-bench` validates complete journals and emits ordinary SharpeBench submissions without introducing a reverse package dependency.
- py: a closed, non-executable strategy DSL whose host records every raw generated candidate, derives the deflation trial count from observed generation, selects on a validation split and evaluates only the winner on a disjoint test split.
- py: a separate forward paper-trading evidence class with read-only Binance/Alpaca market data, in-memory or Alpaca-paper-only submission, deny-first host-side risk preflight, append-only decision/order/refusal evidence and a SharpeBench-compatible commitment with a separate private reveal preimage. No real-capital endpoint or override exists.
- docs: local-agent architecture, August 2026 frontier-model/server matrix, sandbox-and-environment research, and an end-to-end Gordon portability assessment. The Gordon report recommends no code port into the default scaffold.

### Changed
- docs/paper: state the directed composition explicitly: SharpeArena owns the environment and sandbox, a validated artifact crosses the boundary, and SharpeBench owns field-level scoring. The build is distinguished from the still-unrun local-model experiment.
- paper: the companion citation reads SharpeBench version 0.11.0.
- paper: `paper/evidence/provenance.json` is regenerated against the current tree and its source snapshot excludes build outputs (`target/`, `.venv/`, `__pycache__/`), so the manifest is reproducible across machines; `paper/src/check-provenance.py` validates it and exits nonzero on any mismatch.
- paper: `04-contract.tex` states the per-runtime golden coverage exactly (native and Python assert all seven, WebAssembly asserts the canonical and clustered scenarios); the F1 pass^k difference half-width is corrected to 0.35 at the stated p = 1/2 convention; the positive control notes that the fourth familywise-positive cell clears zero at the boundary.
- market_making: the closed-form reference quoting policy is exported as `closed_form_reference_policy`; `analytically_optimal_policy` is kept as a deprecated alias, so the public symbol no longer asserts an optimality the docstrings retract.

### Fixed
- py: MCP, `verifiers`, and local-model execution now share the canonical fail-closed decision parser. Unknown or duplicate symbols, malformed actions, and non-finite or out-of-range weights return typed faults without advancing the environment instead of silently becoming a different or flat action. The signed target is authoritative: an action label never invents a sign constraint, because `sell` may reduce a long and `buy` may cover a short.

## [0.15.0] - 2026-08-25

### Added
- py: two pyo3 entry points expose the native scenario serialization, and `crates/sharpearena-py/tests/test_python_golden.py` recomputes FNV-1a in pure Python to assert all seven committed scenario goldens plus determinism through the bindings ([c4b5785](https://github.com/general-liquidity/sharpearena/commit/c4b5785)).
- paper: the witness artifact serializes all five noise paths per cell under `noise_replicates`, which is what the witness table summarizes ([c4b5785](https://github.com/general-liquidity/sharpearena/commit/c4b5785)).

### Changed
- paper: the re-review residuals are closed. No site classifies an F3 cell by the single realized 0.34 gap; cells are read only through their seed-paired bootstrap intervals. The experiments preamble gains a findings map (ten findings across fifteen subsections), scoping statement 1 separates the tier-generator findings from F2, F5 and F6, the positive control separates its three granularities, F2 reports its minimum detectable effect, F8 drops "more often than not" for raw and resolved counts, pass^k and clean-rate differences carry binomial widths, and the provenance hash is stated once ([c4b5785](https://github.com/general-liquidity/sharpearena/commit/c4b5785)).
- README: the witness threshold and the market-making cost figures are restated from `witness.json` and `f2-regret.json` ([c4b5785](https://github.com/general-liquidity/sharpearena/commit/c4b5785)).
- paper: the companion citation reads SharpeBench version 0.9.0 ([c4b5785](https://github.com/general-liquidity/sharpearena/commit/c4b5785)).

## [0.14.0] - 2026-08-25

### Added
- adverse_selection: an endogenous arm (`EndogenousImpact`, `compare_endogenous_arms`) in which informed meta-order flow moves the mid through the engine's permanent-impact law (cross-checked against the native clearing engine to 1e-12); makers still profit against informed flow at the default calibration, the sampled informed level changes sign between the lambda 0.2 and 0.3 grid points, and the informed-uninformed gap survives the whole sweep ([b750c51](https://github.com/general-liquidity/sharpearena/commit/b750c51)).
- manipulation: `side` on the asymmetric schedules (mirrored short round trips) and the 60-cell extended sweep over push size, lambda, leg length, hold, short side and follower gain, corrected over the global 195-cell family; the strongest sampled cell is the 45:5 slow-accumulate schedule under concave impact, and followers destroy the manipulator's profit ([b750c51](https://github.com/general-liquidity/sharpearena/commit/b750c51)).
- manipulation: a fixed-cell 32-fresh-seed confirmation of the selected positive-control schedule (+26.3e-4, [23.7, 28.9]e-4) ([b750c51](https://github.com/general-liquidity/sharpearena/commit/b750c51)).
- realism: the absolute-return autocorrelation check now uses a reproducible IID-null calibration of the finite-panel estimator instead of a fixed threshold at zero, and aggregational Gaussianity tests reduction in absolute excess kurtosis so platykurtic tape converging to Gaussian is not failed by definition; Fano stays exploratory ([b750c51](https://github.com/general-liquidity/sharpearena/commit/b750c51)).
- scenario_gen, paper: the Calm-tier tail calibration sweep (99 cells, pre-declared selection rule, disjoint confirmation band); under the calibrated checks no Calm configuration qualifies as certified and the candidate knobs are shipped as `CALM_CALIBRATION_CANDIDATE_KNOBS` with a test that they do not masquerade as a preset ([b750c51](https://github.com/general-liquidity/sharpearena/commit/b750c51)).

### Changed
- paper: F4 recalibrated (canonical panels 1/24), F3 reports bootstrap intervals instead of a single-gap noise floor, the witness and F3 seed bands are named as the gap-band subset they are, the abstract no longer claims Python bindings are byte-pinned, and the F5 extension intervals use the global family; README aligned ([b750c51](https://github.com/general-liquidity/sharpearena/commit/b750c51)).

## [0.13.0] - 2026-08-25

### Added
- scenario_gen: opt-in clustered jump bursts (`jump_burst_probability`, `jump_burst_persistence`, `jump_burst_size`), a deterministic post-pass that raises the Fano intermittency statistic while the zero-knob golden path stays byte-identical; threaded through the pyo3 constructors, `SharpeArenaEnv`, the vector env and `build_scenario_dataset` ([2e9c8b4](https://github.com/general-liquidity/sharpearena/commit/2e9c8b4)).
- python: `splitmix_inversion`, exact SplitMix64 finalizer inversion primitives that make the generator-inversion boundary reproducible: one full finalizer output inverts exactly, while the published 53-bit unit leaves 2^11 candidate states before the price transform ([2e9c8b4](https://github.com/general-liquidity/sharpearena/commit/2e9c8b4)).
- paper: `paper/evidence/provenance.json` records the Git parent, a content hash over the source snapshot and the digest of every evidence and figure artifact ([7348c82](https://github.com/general-liquidity/sharpearena/commit/7348c82)).

### Changed
- paper: re-derived claims and evidence. The nonlinear-impact branch applies the exponent to dimensionless crowd flow with a scale-invariance regression test; the positive control reports Bonferroni familywise intervals over its 135-cell search alongside pointwise ones; the realism gate is the conjunction of three calibrated directional checks with Fano reported as an exploratory proxy (2 of 24 canonical panels pass); the witness producer enforces coarse-grid monotonicity before bisection ([7348c82](https://github.com/general-liquidity/sharpearena/commit/7348c82)).
- paper: the companion citation reads SharpeBench version 0.8.0 ([aec2e48](https://github.com/general-liquidity/sharpearena/commit/aec2e48)).

### Fixed
- py: the binding satisfies the clippy gate ([9f16e9f](https://github.com/general-liquidity/sharpearena/commit/9f16e9f)).

## [0.12.0] - 2026-08-24
### Added
- scenario_gen: `sealed_seed(salt, slot)`, an opt-in keyed derivation of held-out evaluation seeds into `[EVAL_SEED_BASE, 2^64)` from a secret salt (FNV-1a digest plus SplitMix64 finalizer rounds, PRF-style, not a certified MAC), exposed through pyo3 and as `sealed_eval_seeds(salt)` / `evaluate_eval_set(salt=...)` in Python; disjointness from the train band holds by construction without the salt, and the public set, its version string and its regression snapshot are untouched. Paper: the predictability probe's band scan recovers 16/16 public seeds and 0/16 sealed seeds, and the revealed salt replays 16/16 ([93b0a73](https://github.com/general-liquidity/sharpearena/commit/93b0a73)).
- manipulation: `AsymmetricSchedule` (leg durations plus a block fraction, the symmetric schedule recovered as a member) and the paper's positive control: under concave impact at exponent 0.5 a slow-accumulate, block-liquidate round trip is profitable with a CI clear of zero on the follower-free arm, exponent 0.7 shows the theory's ordering without profit, and all 45 linear-impact points are negative with time-reversal symmetry; the existing `linear` and `concave` evidence keys are byte-identical ([93b0a73](https://github.com/general-liquidity/sharpearena/commit/93b0a73)).
- paper: rank-eligibility existence witness. An out-of-band oracle signal of controlled strength, consumed by `sign_follow` and `deadband_hold`, opens the kernel's gates on every tier; pass^k on every seed is the binding leg at all ten attained boundaries, and Calm every-bar trading cannot pass on costs alone even for a perfect oracle ([93b0a73](https://github.com/general-liquidity/sharpearena/commit/93b0a73)).

### Changed
- paper: the three fragments are integrated into the experiments section; the abstract, introduction, limitations, command appendix and README reflect the positive control, sealed seeds and witness ([93b0a73](https://github.com/general-liquidity/sharpearena/commit/93b0a73)).
- paper: the companion citation reads SharpeBench version 0.7.0 ([b7193fa](https://github.com/general-liquidity/sharpearena/commit/b7193fa)).
- chore: build artifacts and bytecode untracked and ignored ([5a94d70](https://github.com/general-liquidity/sharpearena/commit/5a94d70), [18a37da](https://github.com/general-liquidity/sharpearena/commit/18a37da)).

## [0.11.1] - 2026-08-24

### Fixed
- release: the workspace-excluded registry manifests (`sharpearena-py` Cargo.toml and pyproject, `__version__`, npm `package.json`) are tracked at the release version ([92dd3bf](https://github.com/general-liquidity/sharpearena/commit/92dd3bf)).
- release: the workflow accepts a `release_tag` input so a recovery or verify run targets one existing tag; every job checks out that tag ([7fdc7ec](https://github.com/general-liquidity/sharpearena/commit/7fdc7ec)).

## [0.11.0] - 2026-08-24

### Added
- market: `clear_bar_concave` and `MarketClearing::step_concave` take an `impact_exponent` applied to the permanent (Kyle) flow term only; `1.0` is bit-identical to the linear path and never touches the golden hashes, below one gives concave impact on a separate `powf` path. Exposed as `impact_exponent` on `PyMarketClearing`, `EndogenousMarketEnv` and the manipulation probe ([f8840f2](https://github.com/general-liquidity/sharpearena/commit/f8840f2)).
- paper: concave-impact ablation of the manipulation probe (exponents 0.5 and 0.7) and a generator predictability probe (unconditional baseline, ridge AR(5) adversary, known-seed oracle, and a bounded seed-band scan), with evidence, figures and the two section fragments ([f8840f2](https://github.com/general-liquidity/sharpearena/commit/f8840f2)).
- docs: `EVALUATION.md` states that disjoint is not secret (a public bounded seed band is enumerable) and that the canonical setting is `impact_exponent=1.0` ([f8840f2](https://github.com/general-liquidity/sharpearena/commit/f8840f2)).

### Changed
- paper: five-seat panel revision. The eight findings are framed as instrument calibrations on tape the realism gate fails 23 of 24 times at the canonical configuration; leak-freedom claims are scoped to the interface boundary; the ecology probe replicated over eight seeds retracts the single-seed dethronement headline (the shock-driven winner replacement occurs on 1 of 8 seeds); confidence intervals added to F2, F3, F5 and F6 ([cce2460](https://github.com/general-liquidity/sharpearena/commit/cce2460)).

## [0.10.0] - 2026-08-24

### Added
- scenario_gen: opt-in `vol_clustering` field on `ScenarioSpec` (serde default `0.0`, byte-identical off) applying an EMA persistence state over realized absolute deviations with mul/add/div/max arithmetic; threaded through `LaneConfig`, the pyo3 constructors, `SharpeArenaEnv`, the vector env and `build_scenario_dataset`; a sixth golden pins the clustered Hard panel ([e9088b7](https://github.com/general-liquidity/sharpearena/commit/e9088b7)).
- test: the eval-seeds regression gate is armed with a committed reference snapshot (eight named held-out seeds, three tiers) ([eece1ac](https://github.com/general-liquidity/sharpearena/commit/eece1ac)).

## [0.9.0] - 2026-08-24

### Added
- py: `PyMarketClearing` takes `lambda_radius` / `eta_radius` / `correlation` for the elliptic uncertainty set and steps through `step_robust`; `EndogenousMarketEnv` gains an opt-in `uncertainty` kwarg ([34ae9c7](https://github.com/general-liquidity/sharpearena/commit/34ae9c7)).
- py: ten stranded public names re-exported (ecology `classify_outcomes`, the deferred-claims trio, the risk-aware reward trio, the behavioral counterparties) and the `mcp` optional extra declared ([fe03149](https://github.com/general-liquidity/sharpearena/commit/fe03149)).
- paper: scaffold, then the evidence run and finished draft with eight findings, titled "SharpeArena: The Point-in-Time (PIT) RL Environment for Trading Agents" ([a34837b](https://github.com/general-liquidity/sharpearena/commit/a34837b), [70fb02c](https://github.com/general-liquidity/sharpearena/commit/70fb02c)).

### Changed
- kernel: sharpebench-sim/protocol/core upgraded from 0.0.8 to 0.5.0; `leaderboard_ci` mirrors the kernel's units fix (annualized prior converted per period once). `EVALUATION.md` regenerated: every baseline had scored 0.0000 deflated Sharpe as an artifact; under the corrected kernel drift saturates DSR on Calm and nothing is rank-eligible on pass^k ([34ae9c7](https://github.com/general-liquidity/sharpearena/commit/34ae9c7)).
- docs: every stale claim corrected against the tree (nine undocumented capabilities added to the README table, the false eval-leak caveat removed from the training docs, an npm registry README) ([b84d9f5](https://github.com/general-liquidity/sharpearena/commit/b84d9f5)).
- ci: per-ref concurrency, job timeouts, caches, a three-OS matrix for the Rust job, a cargo-deny leg for the excluded `sharpearena-py` tree, and PyPI build provenance ([f686e4a](https://github.com/general-liquidity/sharpearena/commit/f686e4a), [4719d13](https://github.com/general-liquidity/sharpearena/commit/4719d13), [46d94aa](https://github.com/general-liquidity/sharpearena/commit/46d94aa)).

## [0.8.0] - 2026-08-20

### Added
- py: `risk_aware` and `time_inhomogeneous_vol_aversion` reward schemes (per-bar risk pricing from a causal EMA conditional volatility); `DispositionEffectPolicy` and `OverconfidentPolicy` behavioral counterparties in a separate `BEHAVIORAL_POLICIES` registry ([ddf0a5b](https://github.com/general-liquidity/sharpearena/commit/ddf0a5b)).
- py: adverse-selection scenario (informed meta-orders against a paired uninformed control, exact markout decomposition) and the manipulation red-team probe with impact-boundary and size-response sweeps ([88cbadf](https://github.com/general-liquidity/sharpearena/commit/88cbadf)).
- market: opt-in `EllipticUncertaintySet` over the Kyle / Almgren-Chriss coefficients; `clear_bar_robust` clears at the closed-form worst case, and `clear_bar` is the `None` case, proven byte-identical ([ae116fa](https://github.com/general-liquidity/sharpearena/commit/ae116fa)).
- py: `run_ecology` replicator population dynamics with innovation and shocks, `classify_outcomes`, `detect_coalitions`; `DeferredDesk` deferred-claims commitment and `resolve_claims` scoring ([49e9d6c](https://github.com/general-liquidity/sharpearena/commit/49e9d6c)).

### Changed
- ci: the excluded `sharpearena-py` crate is linted with rustfmt and clippy from inside its directory ([94be03c](https://github.com/general-liquidity/sharpearena/commit/94be03c), [a866502](https://github.com/general-liquidity/sharpearena/commit/a866502)).

### Fixed
- release: version rewrites for `package.json`, `pyproject.toml`, the pyo3 crate and `__version__`; manifest-versus-tag assertions before publish; a verify job over crates.io, npm and PyPI ([d682af7](https://github.com/general-liquidity/sharpearena/commit/d682af7), [880f8f5](https://github.com/general-liquidity/sharpearena/commit/880f8f5), [0d6ba04](https://github.com/general-liquidity/sharpearena/commit/0d6ba04)).

## [0.7.0] - 2026-07-20

### Added
- scenario_gen, py: `cross_regime_split` / `cross_regime_transfer` zero-shot transfer protocol (seed band fixed, tier swapped) ([956d32e](https://github.com/general-liquidity/sharpearena/commit/956d32e), [826aeb0](https://github.com/general-liquidity/sharpearena/commit/826aeb0)).
- curriculum, py: `AdaptiveCurriculum` / `AdaptiveScheduler` PLR difficulty targeting by online solve rate ([ab1e371](https://github.com/general-liquidity/sharpearena/commit/ab1e371), [826aeb0](https://github.com/general-liquidity/sharpearena/commit/826aeb0)).
- eval, py: `leaderboard_ci` seed-paired bootstrap CI and paired-difference test on the deflated Sharpe; `bootstrap_dsr_ci` / `paired_dsr_diff` bindings; `run_baselines` attaches a CI per row and `pairwise_significance` labels adjacent pairs ([e73100e](https://github.com/general-liquidity/sharpearena/commit/e73100e), [64ffc10](https://github.com/general-liquidity/sharpearena/commit/64ffc10), [647844e](https://github.com/general-liquidity/sharpearena/commit/647844e)).
- market, py: `ObservationRichness` information-disclosure axis (`data_poor` / `standard` / `data_rich`) orthogonal to regime; `standard` is byte-identical to the historical stream ([ec718c5](https://github.com/general-liquidity/sharpearena/commit/ec718c5), [b654e1d](https://github.com/general-liquidity/sharpearena/commit/b654e1d)).
- docs: `EVALUATION.md` sections for cross-regime transfer, statistical confidence and information disclosure ([7e36488](https://github.com/general-liquidity/sharpearena/commit/7e36488), [dc9e605](https://github.com/general-liquidity/sharpearena/commit/dc9e605), [6bdeb1a](https://github.com/general-liquidity/sharpearena/commit/6bdeb1a)).

### Changed
- **Breaking rename**: OpenOutcry becomes SharpeArena across the crate, the Python distribution, the wasm/npm package and every identifier. The wire contract is unchanged, so conformant agents stay conformant ([80ff730](https://github.com/general-liquidity/sharpearena/commit/80ff730), [43fe55e](https://github.com/general-liquidity/sharpearena/commit/43fe55e)).

### Fixed
- wasm: artifacts rebuilt from source after the rename corrupted the committed binary ([03bf1ce](https://github.com/general-liquidity/sharpearena/commit/03bf1ce)).
- deps: crossbeam-epoch 0.9.20 for RUSTSEC-2026-0204 ([cfaff11](https://github.com/general-liquidity/sharpearena/commit/cfaff11)).

## [0.6.1] - 2026-07-03

### Added
- lob, py: `OrderBook::uncross` single-price call-auction clearing and the read-only `sweep_cost` walk, exposed on `PyOrderBook` ([11856c8](https://github.com/general-liquidity/sharpearena/commit/11856c8), [4dd55b6](https://github.com/general-liquidity/sharpearena/commit/4dd55b6)).
- realism: `stylized_facts` / `certify_realism` diagnostic (fat tails, volatility clustering, Zumbach asymmetry, gain/loss skew, aggregational Gaussianity, Fano intermittency) with a Hard/Extreme-fatter-than-Calm test ([07bf407](https://github.com/general-liquidity/sharpearena/commit/07bf407), [df39719](https://github.com/general-liquidity/sharpearena/commit/df39719)).
- risk: `CrossSectionalDeleverage` breadth-based circuit breaker ([8386ac0](https://github.com/general-liquidity/sharpearena/commit/8386ac0)).
- eval: `classify_episode_failure` / `rollup_failure_modes` deterministic episode-failure taxonomy ([6db4a4c](https://github.com/general-liquidity/sharpearena/commit/6db4a4c)).
- obs: `KalmanSpreadObservation` (delta-Kalman hedge ratio for the cointegrated-pairs scenario) and `KalmanTrendObservation` (constant-velocity filter) ([aa6378b](https://github.com/general-liquidity/sharpearena/commit/aa6378b), [4f36a8b](https://github.com/general-liquidity/sharpearena/commit/4f36a8b)).

## [0.6.0] - 2026-06-27

### Added
- market: deterministic integer-tick limit-order-book matching engine (price-time priority, market/limit/cancel/modify, depth ladder, microprice, queue imbalance) with `PyOrderBook` and the PettingZoo `LOBMarketEnv` ([affe9cf](https://github.com/general-liquidity/sharpearena/commit/affe9cf), [30c8f1a](https://github.com/general-liquidity/sharpearena/commit/30c8f1a)).
- env: sharpebench-sim 0.0.8 consumed; `clone_state` / `restore_state` give `CheckpointableEnv` an O(1) native fast path that agrees byte-for-byte with replay ([e23f155](https://github.com/general-liquidity/sharpearena/commit/e23f155)).

## [0.5.0] - 2026-06-26

### Added
- env: `LiquidationCascadeEnv` margin-call / forced-reduce / cascade-impact wrapper ([867aa8e](https://github.com/general-liquidity/sharpearena/commit/867aa8e)).
- eval: reward-misspecification negative-control track (raw PnL, win rate, indicator-shaped, recency) with `demonstrate_punishment` ([366b698](https://github.com/general-liquidity/sharpearena/commit/366b698)).

## [0.4.0] - 2026-06-26

### Added
- env: `MarketMakingEnv` with the closed-form Avellaneda-Stoikov optimal policy as a committed baseline and `mm_regret`; `ExecutionEnv` (VWAP/TWAP implementation shortfall); `PortfolioEnv` (simplex allocation) ([cf4eeab](https://github.com/general-liquidity/sharpearena/commit/cf4eeab), [c97e277](https://github.com/general-liquidity/sharpearena/commit/c97e277), [831efdb](https://github.com/general-liquidity/sharpearena/commit/831efdb)).
- eval: `calibrated_forecast` and `forecast_skill_curve`, a forecast-quality axis with exactly known OOS R2 (eval-only) ([55018ee](https://github.com/general-liquidity/sharpearena/commit/55018ee)).
- obs: `MultiTimescaleMomentum`, `RollingCovarianceObservation`, `TimeToHorizonObservation`, `CounterfactualInfo` ([6f216c3](https://github.com/general-liquidity/sharpearena/commit/6f216c3)).
- data: gap-aware contiguous-block episode sampling for frozen CSVs ([55e40b0](https://github.com/general-liquidity/sharpearena/commit/55e40b0)).
- market: `vol_scale` widens temporary impact with trailing realized variance (`0` is bit-identical) ([0255af9](https://github.com/general-liquidity/sharpearena/commit/0255af9)).

## [0.3.0] - 2026-06-26

### Added
- env: `step_async` / `step_wait` on the vector env with a batched-equals-scalar equivalence test; `CurriculumEnv` / `regime_curriculum`; `PreprocessingConfig` with `CANONICAL_PREPROCESSING`; `EVAL_SEEDS` named held-out set with `assert_no_regression`; `to_minari_train_test` over disjoint seed bands ([d6b87f1](https://github.com/general-liquidity/sharpearena/commit/d6b87f1), [65c48af](https://github.com/general-liquidity/sharpearena/commit/65c48af), [4740cdf](https://github.com/general-liquidity/sharpearena/commit/4740cdf), [b3007a9](https://github.com/general-liquidity/sharpearena/commit/b3007a9), [7dc1e4d](https://github.com/general-liquidity/sharpearena/commit/7dc1e4d)).
- train: reward-scheme registry (differential Sharpe, Sortino, drawdown-penalized, turnover-penalized, loss-averse) via `load_environment(reward_scheme=)`; `PairsConvergence` mandate style and `max_inventory` cap ([d0dae5f](https://github.com/general-liquidity/sharpearena/commit/d0dae5f)).
- obs: `CausalIndicatorObservation` (RSI/SMA/EMA/MACD/Bollinger/realized vol), `SpreadObservation`, `SyntheticNewsObservation` ([5c2419f](https://github.com/general-liquidity/sharpearena/commit/5c2419f), [8f6a2ba](https://github.com/general-liquidity/sharpearena/commit/8f6a2ba), [1cfc245](https://github.com/general-liquidity/sharpearena/commit/1cfc245)).
- scenario: `CointegratedPairs` and `RegimeShift` generator modes, golden-hashed ([8f6a2ba](https://github.com/general-liquidity/sharpearena/commit/8f6a2ba)).
- eval: full risk/profit metrics panel on `RunMetrics`; `MinVariance` / `MaxSharpe` / `KellyVolTarget` baselines, `evaluate_per_regime`, `radar_score` ([cab0162](https://github.com/general-liquidity/sharpearena/commit/cab0162), [0d9492f](https://github.com/general-liquidity/sharpearena/commit/0d9492f)).
- env: `DrawdownStopper` and `TurbulenceHalt` risk wrappers; `DiscreteAction` adapter ([2b0aea8](https://github.com/general-liquidity/sharpearena/commit/2b0aea8), [8aeea26](https://github.com/general-liquidity/sharpearena/commit/8aeea26)).

## [0.2.0] - 2026-06-26

### Added
- env: trainable `verifiers` `MultiTurnEnv` stepping the market per bar, the Procgen-style seeded scenario generator with `train_test_split` and a cross-runtime golden hash, gym conformance (`terminated` vs `truncated`, `validate_decision_json`, `check_env`), point-in-time-safe wrappers, `generalization_gap` ([b90f427](https://github.com/general-liquidity/sharpearena/commit/b90f427)).
- env: `VecTradingEnv` (rayon lockstep lanes), the prime-rl GRPO example, the MCP server, `LookaheadGuard`, versioned JSONL rollout traces and `RunMetrics` ([9072f4d](https://github.com/general-liquidity/sharpearena/commit/9072f4d)).
- scenario: Hard / Extreme volatility-and-jump tiers with mul/add/div/max arithmetic; `reset(seed)` selects a scenario ([f93bdf7](https://github.com/general-liquidity/sharpearena/commit/f93bdf7)).
- env: Gymnasium registration with versioned IDs, `AutoresetMode`, split seeding, Minari export, vector wrappers, `ExecutionNoiseWrapper`, the PettingZoo competition env, per-scenario mandates with a mandate-conditioned rubric, and `EVALUATION.md` with a baseline leaderboard ([81a4f98](https://github.com/general-liquidity/sharpearena/commit/81a4f98)).
- env: `CheckpointableEnv` replay-based clone/restore/branch and the stateless `FuncEnv` ([897247f](https://github.com/general-liquidity/sharpearena/commit/897247f), [df1c781](https://github.com/general-liquidity/sharpearena/commit/df1c781)).
- market: the endogenous price-impact market (Kyle permanent plus Almgren-Chriss temporary impact) as `PyMarketClearing` and `EndogenousMarketEnv`; mandate sampling and execution noise moved into the Rust core ([79ed8c3](https://github.com/general-liquidity/sharpearena/commit/79ed8c3)).

### Changed
- docs: README reflects the full feature set with no em-dashes, and adds TS/npm, vectorized and wrapper examples ([7b783c9](https://github.com/general-liquidity/sharpearena/commit/7b783c9), [5d7da98](https://github.com/general-liquidity/sharpearena/commit/5d7da98), [54cdfdc](https://github.com/general-liquidity/sharpearena/commit/54cdfdc), [b37309f](https://github.com/general-liquidity/sharpearena/commit/b37309f)).

### Fixed
- ci: maturin sdist no longer leaks the sccache wrapper ([a9f973b](https://github.com/general-liquidity/sharpearena/commit/a9f973b)).
- release: npm repository URL corrected (it pointed at sharpebench and failed provenance validation); PyPI publish is `skip-existing` ([0c8cc18](https://github.com/general-liquidity/sharpearena/commit/0c8cc18)).

## [0.1.0] - 2026-06-26

First published release, as OpenOutcry.

### Added
- The env and the governed `Observation` / `Decision` wire contract extracted from the SharpeBench workspace, depending on the published sharpebench engine; JSON Schemas, a conformance kit and reference agents in Rust, TypeScript and Python; the wasm crate, the npm package and the pyo3 binding with a Gymnasium adapter ([246fa5f](https://github.com/general-liquidity/sharpearena/commit/246fa5f)).
- `score_run` from the binding, so the `verifiers` rubric is scored by the real SharpeBench kernel ([91328e6](https://github.com/general-liquidity/sharpearena/commit/91328e6)).
- OIDC trusted-publishing pipeline for crates.io, npm and PyPI on a `v*` tag ([bfdaeed](https://github.com/general-liquidity/sharpearena/commit/bfdaeed)).

### Changed
- pyo3 0.29, Node 24 in CI, README in the house style ([14ceeee](https://github.com/general-liquidity/sharpearena/commit/14ceeee), [378f9da](https://github.com/general-liquidity/sharpearena/commit/378f9da)).

### Fixed
- ci: toolchain pinned to 1.96.0 for the wasm32 target; a virtualenv for maturin ([bb0ee4d](https://github.com/general-liquidity/sharpearena/commit/bb0ee4d)).

[0.19.0]: https://github.com/general-liquidity/sharpearena/compare/v0.18.0...v0.19.0
[0.18.0]: https://github.com/general-liquidity/sharpearena/compare/v0.17.0...v0.18.0
[0.17.0]: https://github.com/general-liquidity/sharpearena/compare/v0.16.0...v0.17.0
[0.16.0]: https://github.com/general-liquidity/sharpearena/compare/v0.15.0...v0.16.0
[0.15.0]: https://github.com/general-liquidity/sharpearena/compare/v0.14.0...v0.15.0
[0.14.0]: https://github.com/general-liquidity/sharpearena/compare/v0.13.0...v0.14.0
[0.13.0]: https://github.com/general-liquidity/sharpearena/compare/v0.12.0...v0.13.0
[0.12.0]: https://github.com/general-liquidity/sharpearena/compare/v0.11.1...v0.12.0
[0.11.1]: https://github.com/general-liquidity/sharpearena/compare/v0.11.0...v0.11.1
[0.11.0]: https://github.com/general-liquidity/sharpearena/compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com/general-liquidity/sharpearena/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/general-liquidity/sharpearena/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/general-liquidity/sharpearena/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/general-liquidity/sharpearena/compare/v0.6.1...v0.7.0
[0.6.1]: https://github.com/general-liquidity/sharpearena/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/general-liquidity/sharpearena/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/general-liquidity/sharpearena/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/general-liquidity/sharpearena/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/general-liquidity/sharpearena/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/general-liquidity/sharpearena/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/general-liquidity/sharpearena/releases/tag/v0.1.0
