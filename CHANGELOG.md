# Changelog

All notable changes to SharpeArena are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). One workspace
version covers the Rust crates, the npm package and the PyPI package; each
section is one `v*` tag and links the commits it was built from. The wire
contract has stayed at `CONTRACT_VERSION` 1.0 throughout.

[Unreleased]: https://github.com/general-liquidity/sharpearena/compare/v0.11.1...HEAD

## [Unreleased]

### Added
- scenario_gen: `sealed_seed(salt, slot)`, an opt-in keyed derivation of held-out evaluation seeds into `[EVAL_SEED_BASE, 2^64)` from a secret salt (FNV-1a digest plus SplitMix64 finalizer rounds, PRF-style, not a certified MAC), exposed through pyo3 and as `sealed_eval_seeds(salt)` / `evaluate_eval_set(salt=...)` in Python; disjointness from the train band holds by construction without the salt, and the public set, its version string and its regression snapshot are untouched. Paper: the predictability probe's band scan recovers 16/16 public seeds and 0/16 sealed seeds, and the revealed salt replays 16/16 (commit pending).
- manipulation: `AsymmetricSchedule` (leg durations plus a block fraction, the symmetric schedule recovered as a member) and the paper's positive control: under concave impact at exponent 0.5 a slow-accumulate, block-liquidate round trip is profitable with a CI clear of zero on the follower-free arm, exponent 0.7 shows the theory's ordering without profit, and all 45 linear-impact points are negative with time-reversal symmetry; the existing `linear` and `concave` evidence keys are byte-identical (commit pending).
- paper: rank-eligibility existence witness. An out-of-band oracle signal of controlled strength, consumed by `sign_follow` and `deadband_hold`, opens the kernel's gates on every tier; pass^k on every seed is the binding leg at all ten attained boundaries, and Calm every-bar trading cannot pass on costs alone even for a perfect oracle (commit pending).

### Changed
- paper: the three fragments are integrated into the experiments section; the abstract, introduction, limitations, command appendix and README reflect the positive control, sealed seeds and witness (commit pending).

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
