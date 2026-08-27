<!-- prettier-ignore -->
<div align="center">

# SharpeArena

### The agentic point-in-time environment for trading agents, and the contract they speak

*Whoever defines the agent interface owns the ecosystem. SharpeArena is the leak-free trading floor every agent runs on, and the language-agnostic contract that makes any agent scorable.*

[![Crates.io](https://img.shields.io/crates/v/sharpearena?style=flat-square&logo=rust&color=DEA584&label=crates.io)](https://crates.io/crates/sharpearena)
[![npm](https://img.shields.io/npm/v/@general-liquidity/sharpearena?style=flat-square&logo=npm&color=CB3837)](https://www.npmjs.com/package/@general-liquidity/sharpearena)
[![PyPI](https://img.shields.io/pypi/v/sharpearena?style=flat-square&logo=pypi&logoColor=white&color=3776AB)](https://pypi.org/project/sharpearena/)
[![docs.rs](https://img.shields.io/docsrs/sharpearena?style=flat-square&logo=docsdotrs&label=docs.rs)](https://docs.rs/sharpearena)
[![CI](https://img.shields.io/github/actions/workflow/status/general-liquidity/sharpearena/ci.yml?style=flat-square&label=CI)](https://github.com/general-liquidity/sharpearena/actions)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue?style=flat-square)](#license)
[![Unsafe](https://img.shields.io/badge/unsafe-forbidden-success?style=flat-square)](#architecture)

**[Why](#why) · [Local agents](#run-local-open-weight-agents) · [Quickstart](#quickstart) · [Train an agent](#train-an-agent) · [Surfaces](#use-it-from-anywhere) · [The contract](#the-agent-contract) · [Architecture](#architecture) · [Tech stack](#tech-stack)**

</div>

---

## Why

An eval is useless without an environment. A benchmark scores *trajectories*; something has to **produce** them. SharpeArena is that producer: a leak-free, point-in-time market environment wrapped in a dead-simple, language-agnostic agent contract. **The harness sends an `Observation`, the agent returns a `Decision`, repeat.**

The agent on the other side of that loop is increasingly a language model rather than a policy network, so the environment is built to drive one: a fixed prompt scaffold, constrained decoding, strict host-side validation of the canonical `Decision`, model/checkpoint and runtime provenance with unresolved facts labeled as such, and a typed fault whenever the model's output or its transport fails. Around that loop sit two adjacent arms, a strategy generator whose deflation trial count comes from candidates the **host** counted rather than the model reported, and a forward paper arm that submits only to a paper endpoint. Classical RL agents run through the identical stepping surface; nothing about the Gymnasium, PettingZoo or `verifiers` paths changed to make room for the agentic one.

Three properties make it trustworthy rather than a toy:

1. **Look-ahead is structurally impossible.** The environment owns the time cursor and the data layer has *no API to read a future bar*, so an agent cannot peek by construction, not by policing. A `LookaheadGuard` refuses agent operations that read future data, and the deferred-claims desk holds no dataset, no env and no path to future data at all.
2. **Trajectories are recompute-from-raw-decisions.** A run records only the agent's decisions; a separate verifier replays them against the frozen data to recompute a **byte-identical** result. A tampered trajectory recomputes differently, so an agent cannot lie about its returns.
3. **A failure is evidence, never a hold.** A wire fault, a malformed decision, an unknown symbol or a timeout produces a typed failed cell. It is never flattened into an empty-orders decision, which against this engine is a *true hold* that silently carries the previous position forward and yields a return series indistinguishable from a deliberately conservative agent's.

The strategic bet is **interface ownership**: if every trading agent in the open ecosystem conforms to SharpeArena's `Observation`/`Decision` contract, then [SharpeBench](https://crates.io/crates/sharpebench-core) is the natural scorer and the whole funnel (env, trajectory, score, leaderboard) runs on one standard. This is the OpenAI-Gym moment for trading agents. The interface *is* the product; the simulator is the credibility behind it.

> An agent is just a program that reads an observation and writes a decision, in any language. Conform to the contract, and you are scorable everywhere.

## Status: published, active (pre-1.0)

Published to **crates.io**, **npm**, and **PyPI** (the badges above show the live versions), depending on the **published** sharpebench engine (not a vendored copy). CI is green across four surfaces: Rust (`fmt`, `clippy -D warnings`, tests, a WASM target build, and the scenario goldens executed on `wasm32`), `cargo-deny`, the npm package, and the Python wheel (`maturin` + `pytest`).

**What the evidence shows.** The paper, *SharpeArena: The Point-in-Time (PIT) RL Environment for Trading Agents* ([`paper/`](paper/)), commits ten experiments (every number from a script, JSON evidence, and fixed seeds) and frames them as calibrations of the suite's instruments, not market findings. Among them:

- **The corrected deflation prior rewrites the baseline board.** The pre-0.5.0 kernel scored every baseline **0.0000** deflated Sharpe on every tier: a unit bug, an annualized prior applied per period, a bar near annualized Sharpe 18. Corrected, drift saturates the deflated Sharpe at **1.0000** on Calm, and still **nothing is rank-eligible**: no baseline passes pass^k on every seed (best rate 0.75), and Hard/Extreme collapse the score to 0.11/0.02. The eligibility conjunction, not a mis-united floor, is what gates.
- **The realism gate fails the canonical tape, and says so.** The three calibrated stylized-facts checks fail the canonical configuration on **23 of 24** seeded panels (0/8 Calm, 0/8 Hard, 1/8 Extreme). The ACF gate uses a fixed 95th-percentile IID null for its exact short-panel estimator, while the finite-panel Fano ratio is conditional-IID-calibrated but exploratory. At `vol_clustering=0.5`, three-check pass counts are 0/8, 8/8 and 4/8; the 99-cell Calm follow-up has no setting that meets its diagnostic, confirmation and 25% volatility rule. At 0.0, the leaderboard configuration, the tape is byte-identical, so certification is reported beside the results rather than required by them.
- **Market-making agents are scored on regret against a closed-form reference.** The env ships the closed-form Avellaneda-Stoikov reference policy; its regret is **0.0 by construction**, and fixed-spread quoting pays a U-shaped regret: **84.6** at a 0.05 half-spread, a minimum of **0.037** at 0.5 whose CI [-8.7, 8.8] makes it indistinguishable from the reference, back up to **58.8** at 4.0.
- **Cross-regime transfer exposes what the within-tier gap cannot.** The calm-trained reference loses **0.88** of deflated Sharpe zero-shot on Hard and **0.97** on Extreme, while its within-tier generalization gap sits near zero (+0.0012 on Calm), the expected control for an unfitted policy.
- **Linear-impact manipulation is a finite-grid consistency check, not a theorem.** The red-team pump-and-unwind probe has negative means in all **45 sampled** linear-impact cells (Kyle lambda 0 to 0.8, eta 0 to 0.8, follower gain 0 to 120). The size-response sweep is bounded and decreasing on its sampled range. This finite-grid result does not cover other parameter interactions, transient impact, or nonlinear permanent impact.
- **The concave ablation is an honest null, and the positive control demonstrates falsifiability.** At normalized permanent-impact exponents **0.5 and 0.7**, every sampled symmetric-pump mean is negative; 7 of 23 stored intervals cross zero at 0.5 (five distinct configurations), and 0 of 23 cross at 0.7. An exploratory 135-cell asymmetric search finds a profitable sampled schedule at exponent **0.5**: nine bars of slow accumulation followed by a one-bar block liquidation earns an impact-attributable **+21.8e-4**, with pointwise 95% interval [17.4, 26.2]e-4 and Bonferroni familywise interval [10.0, 33.6]e-4. A 60-cell one-axis extension uses independently normalized unequal legs and, because its anchor comes from the first grid on the same seeds, global 195-cell familywise intervals: 23 cells are positive, 31 negative and 6 unresolved; the largest sampled 45:5 cell is **+135.0e-4**, [122.1, 148.0]e-4. All **45 sampled** linear-impact means are negative, and the mirror values are numerically near-symmetric rather than bit-identical. These are finite-grid results, not a universal theorem or optimum; interactions, other exponents, flow scales, transient impact and overshooting trips remain outside the search.
- **Bounded public seed bands are invertible; salt secrecy defeats that scanner.** A causal AR adversary finds no undisclosed stressed-tier edge, but a scan of a public **2^16** seed band recovers **16/16** tested seeds from one observed bar in about one second. The opt-in `sealed_seed` / `sealed_eval_seeds` derivation recovers **0/16** under that same bounded scanner while a high-entropy salt is withheld, and revealing the salt replays **16/16**. This is a simulated commit-reveal result against one enumeration budget, not a cryptographic proof or evidence that any live evaluation uses sealed seeds. A forward protocol can commit a salt hash before entry; custody and deployment remain operator responsibilities.
- **Rank-eligibility is attainable, and pass^k is what binds.** An out-of-band oracle (the seed replayed one bar ahead, not an entrant, never scored) mixed with noise at controlled strength opens the gates on every tier: the eligibility set is non-empty, and at every attained crossing of all ten eligible cells, replicated over five independent noise paths each, the binding leg is the per-run PSR gate on every seed, which the pooled deflated-Sharpe and bootstrap legs clear long before. On Calm an every-bar sign follower never passes even with a perfect oracle: at zero signal its every-bar turnover costs **14.4 bp/bar**, and even at full oracle strength its net mean return is only **2.7 bp/bar**, which leaves 5 of 16 seeds short of the PSR gate. A turnover-rationing variant does pass, from a mean signal-truth correlation of **0.848** over five noise paths (range [0.777, 0.930]). `flat` fails 16/16 because a zero-dispersion series carries no evidence of edge (PSR 0.5), not because inactivity is penalized.
- **Ecology is a distribution, not one trajectory.** Across eight replicator seeds per schedule, shock-driven replacement of the volatility targeter occurs on **1 of 8** seeds, while a `kelly_vol_target` lineage ends dominant on 7 of 8 shocked seeds. The probe reports this outcome distribution rather than treating one trajectory as a general conclusion.

The rest (the generalization-gap bootstrap CIs, adverse-selection markouts, failure-mode distributions) is in the paper's Experiments section, with the exact command per number in its appendix. The scoring kernel's own paper, *SharpeBench: A Luck-Robust Benchmark for Trading Agents*, is the companion to this one.

Beyond the core `reset`/`step` lifecycle, the environment now ships a full **reinforcement-learning training surface**:

| Capability | What it is |
|:--|:--|
| **Procedural scenarios** | A seeded generator (`ScenarioSpec` / `generate_scenario`) using Procgen's integer-seed-interval model, with `Calm` / `Hard` / `Extreme` volatility-and-jump tiers and provably disjoint `train_test_split`, cross-runtime golden-hashed for byte-identical generation, plus an opt-in `vol_clustering` persistence driver on `ScenarioSpec` (default 0.0, byte-identical off; threaded through the Gymnasium and vector envs) pinned by its own golden hash. |
| **Generalization gap** | `generalization_gap` measures train-vs-held-out deflated Sharpe over disjoint seed bands, turning "did it overfit" into one number scored by the SharpeBench kernel. |
| **`verifiers` training env** | A PrimeIntellect `verifiers` `MultiTurnEnv` that steps the market bar-by-bar, over a multi-row scenario `Dataset`, with an `XMLParser` decision protocol and a GRPO-safe bounded reward scored by the real SharpeBench `score_run` (deflated Sharpe, pass^k, process checks). |
| **Pluggable reward schemes** | `load_environment(reward_scheme=...)` selects from a registry: an online differential Sharpe (Moody-Saffell, aligns the training signal with the deflated-Sharpe score), Sortino, drawdown-penalized, turnover-penalized, loss-averse, plus two risk-aware schemes that price risk per bar instead of at episode end (`risk_aware` charges a causally-estimated EMA conditional volatility at a fixed aversion; `time_inhomogeneous_vol_aversion` varies that aversion across the horizon). Schemes shape *training only*; the rank key stays the SharpeBench kernel. |
| **Reward-misspecification probes** | A registry of deliberately flawed rewards (raw PnL, win-rate, indicator-shaped, recency-biased) run as *negative controls*: greedy proxy maximizers of each flawed reward score below a clean baseline on the SharpeBench kernel, making "the scorer punishes over-leverage / overfit / churn" falsifiable. Never registered as valid schemes; never feed the rank key. |
| **More env tasks** | Beyond the position env: a `PortfolioEnv` (simplex allocation, log-return), an `ExecutionEnv` (VWAP/TWAP implementation-shortfall MDP), and a `MarketMakingEnv` (Avellaneda-Stoikov) that ships the model's **closed-form quoting policy** as a fixed reference (the source model's asymptotic closed form, not a proven optimum for this env's reward), so agents are scored on regret against a fixed analytical reference. |
| **Scenario families** | Calm/Hard/Extreme vol-jump tiers plus `CointegratedPairs` (genuine mean-reverting spread), `RegimeShift` (trend-to-whipsaw), curriculum chaining (`CurriculumEnv`/`regime_curriculum`), and a frozen named held-out eval-seed regression set. |
| **Rich observations** | Opt-in, causal, leak-free obs augmentations computed from the point-in-time history: technical indicators (RSI/MACD/Bollinger/...), multi-timescale momentum, rolling covariance, a spread z-score, a **recursive Kalman hedge-ratio spread** (a `KalmanSpreadObservation` for the cointegrated-pairs scenario emitting the innovation-normalized z, leak-free by construction, superseding the rolling-OLS approximation), a **Kalman constant-velocity trend** obs (filtered velocity + sign), a synthetic seed-derived news/sentiment channel, and time-to-horizon. Declarative via `PreprocessingConfig`. |
| **Risk + eval axes** | Drawdown stop-out, turbulence-halt, liquidation-cascade, and a **cross-sectional deleveraging circuit-breaker** (flattens the oversold subset when universe-wide breadth flips oversold, distinct from the single-asset breakers) wrappers; a forecast-quality calibrated eval axis (FinPILOT), per-regime breakdown, efficient-frontier/Kelly baselines, a deterministic **episode-failure taxonomy** + suite rollup (clean / bankrupt / stopped-out / cascade-wiped / mandate-breach), and a full risk/profit metrics panel (Calmar/Sortino/VaR/CVaR/tail/turnover). |
| **Deferred-claims resolution** | Predictions scored *after* the episode: an agent commits a claim (point, probability, direction, interval) with a resolution horizon via a `DeferredDesk` that holds no dataset, no env, and no path to future data, so leak-freedom is structural. `resolve_claims` scores later against outcomes stamped with when the datum first existed, refusing any outcome available at or before commit; scoring covers Brier, sign accuracy, and interval coverage, and claims round-trip through JSON so commit and resolution can be separate processes or days. |
| **Vectorized rollouts** | `VecTradingEnv` runs B scenario lanes in lockstep (rayon, structure-of-arrays JSON, current-Gymnasium `AutoresetMode`, async `send`/`recv`), exposed as a `gymnasium.vector` env. |
| **Point-in-time-safe wrappers** | Causal normalize (no future-bar leak), `TimeLimit`, `FrameStack`, `RecordEpisodeStatistics`, vector-env variants, and `flatten`/`unflatten` Dict-obs helpers, plus a `check_env` conformance harness that *proves* seed-determinism (and adopts Gymnasium's own `check_env`). |
| **Real-data blocks + discrete actions** | Gap-aware contiguous-block episode sampling for frozen real-data CSVs (every sampled window lies inside one continuous block, so a reset never straddles a listing gap or exchange downtime and fabricates a jump), and a `DiscreteAction` wrapper that maps `Discrete`/`MultiDiscrete` heads (long/flat/short or binned) onto the canonical target-weight `Box` for value-based learners. |
| **Gymnasium registration** | Versioned, namespaced IDs: `gymnasium.make("SharpeArena/Hard-v1")` and `make_vec(...)` route to the scalar and vector envs, with `-Eval-v1` variants on a disjoint held-out seed band. |
| **Multi-agent markets** | A PettingZoo `MultiAgentSharpeArenaEnv` (batched competition: N agents on one frozen scenario, SharpeBench-ranked), an `EndogenousMarketEnv` (a shared-book market where aggregate flow *moves* the cleared price via Kyle + Almgren-Chriss impact), and an `LOBMarketEnv` over a real **deterministic limit-order-book matching engine** (integer ticks, price-time priority, market/limit/cancel/modify, depth-ladder + microprice + queue-imbalance obs), now with a single-price **call-auction uncross** (`uncross`, the max-matched-volume open/close equilibrium the CDA book lacked) and a read-only **walk-the-book `sweep_cost`** query (average fill price + slippage without mutating the book or the tape). |
| **Ecological population dynamics** | `run_ecology` plays generations of a strategy *population* on the endogenous market (FinEvo): discrete-replicator selection moves share toward above-field-average payoff, new variants are bred from the leader on a fixed cadence, and a `Shock` schedule rotates the regime tier and scales the impact coefficients (a liquidity shock). Output is the full generation-by-generation share/fitness trajectory plus `classify_outcomes` (dominant / collapsed / extinct / persistent / marginal, labeling the *shape* of each share trajectory) and `detect_coalitions`, RNG-free so a run reproduces exactly. |
| **Adverse-selection scenario** | An **informed trader** whose meta-order flow precedes the price move, sliced into child orders, against a paired *uninformed* control (identical schedule, sizes, and path; side drawn independently), so the two modes differ only in whether the counterparty's direction is correlated with what happens next. Measured by per-fill **markout decomposition**: `spread_capture + adverse_drift = markout`, an exact partition, with `toxic_fill_rate` and a stated `picked_off` verdict. |
| **Manipulation red-team probe** | A diagnostic aimed at the simulator, not a strategy: a crude pump-and-dump schedule, an `AsymmetricSchedule` family (leg durations plus a block fraction, containing the symmetric schedule as a member), impact-boundary and size-response sweeps, paired zero-impact references, and an opt-in permanent-impact exponent (`impact_exponent=1.0` linear, below 1 concave). The unit exponent preserves the canonical byte-identical path; non-unit exponents are explicit ablations outside the cross-runtime golden guarantee. |
| **Sealed evaluation seeds** | Opt-in, commit-reveal: `sealed_seed(salt, slot)` (Rust, pyo3, and `sealed_eval_seeds(salt)` / `evaluate_eval_set(salt=...)` in Python) derives each named held-out slot from a secret salt into the `[EVAL_SEED_BASE, 2^64-1)` band, so disjointness from the train band stays publicly checkable without the salt while the concrete seeds are unenumerable. The evaluator publishes SHA-256(salt) before the run and reveals the salt after; anyone then recomputes the seeds and replays byte for byte. A keyed PRF-style derivation (FNV-1a plus SplitMix64 finalizer rounds), not a certified MAC; the public set and its regression snapshot are untouched. |
| **Robust market clearing** | An opt-in **elliptic uncertainty set** over the Kyle/Almgren-Chriss impact coefficients (`EllipticUncertaintySet`): the robust entry points clear each bar at the closed-form worst case inside the set (the support function of the ellipse in the cost direction). With no set supplied the robust path executes the identical float operations as the point-estimate path, byte for byte, so published results and golden hashes never move. |
| **Market realism diagnostic** | A `stylized_facts` / `certify_realism` finite-panel diagnostic (Python, off the byte-identical hot path). Its calibrated conjunction gates positive excess kurtosis, absolute-return autocorrelation above an estimator-specific IID null, and aggregation toward zero absolute excess kurtosis. Gain/loss skew, Zumbach asymmetry and conditional-IID-calibrated Fano are exploratory; the canonical tape fails the conjunction on 23 of 24 seeded panels, and the diagnostic is reported rather than used to rank. |
| **Per-scenario mandates** | Each scenario samples a trading mandate: one of five structural styles (long-only, market-neutral, momentum, unconstrained, pairs-convergence) plus up to three optional constraints layered on top (a realized-drawdown cap, a gross-exposure cap, a benchmark to beat). The `verifiers` rubric is mandate-conditioned, so wrong-objective behavior is penalized, not just unrewarded. |
| **Offline-RL + checkpointing** | `to_minari` exports rollouts as a Farama [Minari](https://minari.farama.org) dataset (leak-safe, `recover_environment`-ready); `CheckpointableEnv` clones/restores/branches market state for tree search (O(1) native engine snapshot or replay-from-decisions); `SharpeArenaFuncEnv` is a stateless `gymnasium.functional.FuncEnv` view. |
| **Benchmark protocol** | A committed [`EVALUATION.md`](EVALUATION.md): the canonical eval contract, the disjoint train/held-out split, and a baseline leaderboard (no baseline is rank-eligible: drift saturates the deflated Sharpe on Calm but no baseline passes pass^k on every seed, and pass^k degrades Calm to Hard to Extreme). |
| **Behavioral counterparties** | Deliberately biased policies (`DispositionEffectPolicy` trims winners early and holds/tops up losers; `OverconfidentPolicy` over-sizes a 3-bar signal and churns the book every step) as counterparties and a realism floor, not strategies. Kept out of the ranked reference set on purpose, so the deflation trial count the leaderboard is scored against stays honest. |
| **Harness integration** | An MCP server (`reset` / `step` / `spec` tools) so any MCP agent harness drives an episode with zero glue, a `LookaheadGuard` that refuses agent operations reading future data, versioned JSONL rollout traces that re-score offline through the SharpeBench kernel, and a cost-adjusted `RunMetrics` block for leaderboard ranking. |
| **Transport fault gate** | `run_backtest_checked` is the explicit checked entry point for an external agent: it converts any fault recorded in `TransportHealth` into a typed failed `CellOutcome`, and `TransportDiagnostics` / `TransportHealth` are re-exported so a consumer can name the types at all. The plain engine call remains public for in-process agents; the type system does not force a wire consumer to choose the checked function, so callers must do so deliberately. The shipped Python local-field scheduler has its own strict failed-cell path. |
| **Agentic candidate discipline** | An `EdgeManifest` per generated candidate (hypothesis, mechanism, claimed regimes and instruments, invariants, unit-typed kill conditions, verification plan; closed schema, missing required field invalidates), a counterfactual ledger that records every decision whether or not it was acted on, and strict silver-to-gold promotion of flagged traces into frozen minimal scenarios behind a recorded operator decision. |

The determinism-critical core (the engine, scoring, scenario generation, mandates, the market-clearing model, the execution-noise integrity knob) lives in **Rust** so a published number is byte-identical across every surface; the per-ecosystem adapters (gymnasium, PettingZoo, Minari, verifiers, MCP) are thin and live in the language each ecosystem speaks.

**Not yet shipped:** the [PrimeIntellect](https://app.primeintellect.ai) Environments-Hub listing. Gordon is not a package dependency or the default scaffold: its architecture has been assessed as an optional future treatment arm, while the reproducible field keeps a minimal, fixed scaffold. See [`docs/GORDON_PORT_ASSESSMENT.md`](docs/GORDON_PORT_ASSESSMENT.md).

## Run local open-weight agents

The local field runner is built, but no model result is reported yet. SharpeArena owns the point-in-time environment, canonical `Decision` parsing, execution and process trace. Completed journals cross a validated artifact boundary into SharpeBench, which owns field-level statistics and ranking:

```text
local model + fixed scaffold
            ↓
SharpeArena environment: point-in-time stepping, canonical
decision parsing, execution and process trace
            ↓
append-only decisions, process trace and returns
            ↓
sharpearena-compile-bench
            ↓
SharpeBench scorer and leaderboard
```

This is operational interdependence without a package cycle. SharpeArena uses the small published SharpeBench protocol/simulator/kernel crates; SharpeBench does not depend on the full SharpeArena package. The bridge refuses incomplete grids, failed cells, coordinate collisions, conflicting completions and invalid return hashes.

**On containment.** SharpeArena provides no process or container isolation and does not claim any. It runs the model process the operator points it at, in the operator's own environment. The fail-closed OCI containment path for *untrusted* entrants lives in the sibling `sharpebench-arena` crate. That boundary is now exercised: its smoke test skipped invisibly as a pass for its whole life until it was marked ignored, and a CI job on a Docker-enabled runner has since run it and the hostile probe inside a live container. The evidence is narrow, a single runner image and one pinned fixture tag, and no hostile entrant has ever been put inside it. What SharpeArena earns is a different property, structural leak-freedom: the time cursor belongs to the environment, the data layer has no API for a future bar, and the `LookaheadGuard` refuses agent operations that read future data. Leak-freedom is not containment. Run only models and scaffolds you trust.

The shipped local path includes constrained Ollama inference, strict host-side validation, native vector stepping, cadence and thinking controls, stable sharding, append-only resume, and model/runtime provenance that explicitly says when Ollama's CPU/GPU split was unresolved because a model was not loaded at identity capture. Syntax-constrained output is not trusted as semantic validation: malformed JSON, unknown or duplicate symbols, action/weight contradictions, timeouts and transport failures are recorded as faults and never flattened into a hold. The Python scheduler enforces that rule directly; external Rust-agent consumers get the same outcome only when they use `run_backtest_checked`. CI uses deterministic model doubles; it downloads no weights and reports no performance result.

```bash
sharpearena-local-field \
  --plan examples/local-agents/field-plan-smoke.json \
  --evidence local-evidence/field.jsonl \
  --inspect

sharpearena-compile-bench local-evidence/field.jsonl \
  --output-dir local-evidence/bench
```

Two adjacent paths are also built.

**Strategy generation on host-counted trials.** The generator emits a closed, non-executable JSON DSL. Every raw candidate is counted before validation or deduplication, validation and test windows are disjoint, and the observed candidate count, not a self-reported one, is what feeds the deflation calculation. Each candidate carries an **`EdgeManifest`**: a closed schema binding it to a hypothesis, the mechanism it claims, the regimes and instruments it claims them in, its invariants, quantitative kill conditions and a verification plan. Falsifiability is part of the artifact rather than prose written after selection. Because the schema is closed, a missing required field invalidates the candidate instead of synthesizing a default, and every threshold carries an explicit unit from a closed enum, so a table cannot silently mix basis points, dollars and unit fractions in one untyped column. Kill conditions are evaluated only outside the selection sample.

**A forward paper arm.** It accepts read-only market data and submits only to an in-memory broker or Alpaca's fixed paper endpoint, after a deny-first host-side risk check. There is no real-capital endpoint and no override: the origin is validated and then reassigned unconditionally to the paper host, and redirects are refused. Its provider-time-dependent evidence is explicitly non-replayable and stays in a separate class from deterministic backtest evidence. The execution state machine treats "we never heard back" as its own state rather than collapsing it into a rejection: `submission_unknown` is reachable only from `submitted` and leaves only to a reconciled-accepted or reconciled-absent verdict, ack latency stays `None` until a real acknowledgment arrives, a replacement is permitted only after the broker confirms absence by deterministic client order id and exactly once, and a broker that cannot be queried raises rather than resolving anything. State is persisted before the submit call, on the unknown transition and after every reconciliation, so an order left unresolved by a crash is still unresolved after a restart rather than silently resubmitted. Account state is reconciled against the broker instead of read from the plan file, so limits are not measured against a stale book after the first remote fill.

Two limitations of that arm are worth stating up front: partial-fill accumulation is last-write-wins rather than summing fill deltas, and `reconcile_all` has no retry, so an unanswerable broker query halts the run. Halting is the safe direction, but it will stop a session on a flaky broker.

**The paper arm records everything considered, not just what was executed.** Its counterfactual ledger records every intended order whether it was executed, resized, refused, below minimum notional, or never submitted after an earlier refusal, so the gap between the considered and executed sets is measurable. The historical local-field scheduler does not yet instantiate this ledger: its native target-weight step surface does not expose the per-order intended-versus-filled quantities the ledger requires, and inferring them from portfolio snapshots would conflate slippage, participation caps, and carried fills.

**Flagged traces can become frozen regressions.** Strict silver-to-gold promotion turns a flagged production trace into a permanent scenario. A malformed or incomplete trace is rejected rather than skipped; the fingerprint is deterministic over environment, model, scaffold, contract, data and the process-event sequence; deterministic checks run before anything else; silver candidates are immutable and carry the triggering check with the source trace hash; promotion to gold requires a recorded operator decision; and what is frozen is a minimal scenario plus its expected invariant, not a transcript with a prose rubric attached. The existing permissive reader is untouched, so strictness is a separate mode rather than a tightening of the exploratory path.

The August 2026 model study distinguishes frontier availability from local feasibility. It covers Gemma 4, Qwen3.8, Kimi K3, DeepSeek V4, GLM-5.2, Inkling and Ornith 1.5, plus llama.cpp, vLLM, SGLang, TensorRT-LLM, Transformers and Ollama. On the current 16 GB RTX A4000, the trillion-parameter flagships are comparison targets rather than local downloads; feasible field arms must be selected by exact checkpoint, quantization and measured placement. Of the tags pulled on that machine, `ornith:9b` at 5.24 GB is the only one that fits the card; `qwen3.6:27b` at 16.22 GB, `ornith:35b` at 19.71 GB and `qwen3.6:35b` at 22.29 GB all exceed 16 GB before the KV cache and will spill to CPU. Size an arm by the pulled size `ollama list` prints, not by the parameter count. See [`docs/LOCAL_AGENT_ARCHITECTURE.md`](docs/LOCAL_AGENT_ARCHITECTURE.md), [`docs/LOCAL_MODEL_MATRIX_2026.md`](docs/LOCAL_MODEL_MATRIX_2026.md) and [`docs/SANDBOX_ENVIRONMENT_RESEARCH_2026.md`](docs/SANDBOX_ENVIRONMENT_RESEARCH_2026.md).

## Quickstart

```bash
cargo add sharpearena        # the Rust crate (re-exports the engine + the wire contract)
```

```rust
use sharpearena::{TradingEnv, Dataset, CostModel, Window, BuyAndHold, Agent};

let data = Dataset::synthetic(4, 120, 1);
let mut env = TradingEnv::new(data, Window { start: 20, end: 120 }, CostModel::default(), 7);
let mut agent = BuyAndHold;
let mut obs = env.reset();
loop {
    let decision = agent.decide(&obs);
    let step = env.step(decision);     // -> { observation, reward, done, info }
    obs = step.observation;
    if step.done { break; }
}
```

Both stepping surfaces (the open-loop `TradingEnv` and the closed-loop `run_backtest`) call one shared per-step body, so a trajectory the env produces is **byte-identical** to the equivalent backtest, enforced by a test.

```python
import gymnasium, sharpearena          # a first-class Gymnasium env
env = sharpearena.SharpeArenaEnv(n_symbols=4, n_days=120, seed=1)
obs, info = env.reset(seed=1)         # the seed selects the scenario
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

## Train an agent

The Python package is a PrimeIntellect `verifiers` environment, so an agent trains over a distribution of leak-free scenarios scored by the SharpeBench kernel:

```python
import sharpearena

# A multi-turn env that steps the market bar-by-bar over many seeded scenarios.
env = sharpearena.load_environment(n_windows=256, n_symbols=4, n_days=120)

# Measure overfitting directly: train vs a provably disjoint held-out seed band.
gap = sharpearena.generalization_gap(
    lambda seed: sharpearena.SharpeArenaEnv(n_symbols=4, n_days=120, seed=seed),
    n_train=64, n_test=64,
)
print(gap["gap_deflated_sharpe"])
```

**Scale with vectorized rollouts.** `SharpeArenaVectorEnv` steps `B` independent scenario lanes in lockstep (rayon under the hood), exposing the standard `gymnasium.vector` API:

```python
import numpy as np, sharpearena

vec = sharpearena.SharpeArenaVectorEnv(num_envs=64, n_symbols=4, n_days=120)
obs, infos = vec.reset()
actions = np.full((vec.num_envs, len(vec.symbols)), 0.25, dtype=np.float32)
obs, rewards, terminated, truncated, infos = vec.step(actions)   # arrays of length 64
```

**Compose point-in-time-safe wrappers.** Standard gym wrappers, but the normalizers are *causal*, so no future bar ever leaks into the running statistics:

```python
import sharpearena
from sharpearena import TimeLimit, CausalNormalizeObservation, RecordEpisodeStatistics

env = RecordEpisodeStatistics(            # info["episode"] carries Sharpe + max drawdown
    CausalNormalizeObservation(
        TimeLimit(sharpearena.SharpeArenaEnv(n_symbols=4, n_days=120), max_episode_steps=64)
    )
)
```

**Register it like any Gymnasium env.** Importing `sharpearena` registers versioned IDs, so the whole RL ecosystem reaches it through muscle memory:

```python
import gymnasium, sharpearena
env = gymnasium.make("SharpeArena/Hard-v1")            # difficulty + version pinned in the ID
vec = gymnasium.make_vec("SharpeArena/Hard-v1", num_envs=8)
```

**Trade a real multi-agent market.** In `EndogenousMarketEnv` the agents' aggregate flow *moves* the price they all see (Kyle permanent + Almgren-Chriss temporary impact), not a frozen path:

```python
from sharpearena import EndogenousMarketEnv     # a PettingZoo ParallelEnv

market = EndogenousMarketEnv(n_agents=3, n_symbols=4, n_days=120, seed=1)
obs, infos = market.reset(seed=1)
# every agent submits a target-weight order; the book clears once and the cleared
# price reflects everyone's flow, so one agent's size moves the others' fills.
obs, rewards, terminations, truncations, infos = market.step(
    {a: market.action_space(a).sample() for a in market.agents}
)
```

A one-command [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) GRPO training config lives in [`examples/prime-rl/`](examples/prime-rl/); see [`docs/training.md`](docs/training.md) for the full loop (install, `vf-eval` baseline, `uv run rl`).

## Use it from anywhere

One Rust engine, scored identically across every surface, because they run the same code.

| Surface | Get it | What it is |
|:--|:--|:--|
| <img height="14" align="top" src="https://cdn.simpleicons.org/rust/DEA584" />&nbsp; **Rust crate** | `cargo add sharpearena` | The env, the procedural scenario generator, the batched `VecTradingEnv`, the mandate / execution-noise / market-clearing cores, and the governed wire contract, re-exporting the leak-free engine. |
| <img height="14" align="top" src="https://cdn.simpleicons.org/pypi/3776AB" />&nbsp; **Python** | `pip install sharpearena` | Gymnasium (`Env` + `vector` + registered IDs), PettingZoo (competition + endogenous market), the `verifiers` training env, Minari export, checkpointing, `FuncEnv`, point-in-time-safe wrappers, traces, and an MCP server, over the pyo3 binding. |
| <img height="14" align="top" src="https://cdn.simpleicons.org/npm/CB3837" />&nbsp; **npm** | `npm i @general-liquidity/sharpearena` | A typed JS/TS API over the engine compiled to WASM. |
| <img height="14" align="top" src="https://cdn.simpleicons.org/webassembly/654FF0" />&nbsp; **WASM** | `sharpearena-wasm` | The wasm-bindgen bridge the npm package and Gordon (Bun) embed. |

```ts
import { runBaseline } from "@general-liquidity/sharpearena";

// The identical Rust engine, in the browser or Bun: run a baseline over a seeded panel.
const run = runBaseline({
  agent: "momentum",
  dataset: { synthetic: { n_symbols: 4, n_days: 120, seed: 1 } },
  seed: 7,
});
console.log(run.returns.length, run.cost);   // per-period returns + realized execution cost
```

The agent itself can be written in **any** language: a conforming agent is a program that reads `MarketObservation` JSON (stdin or `POST /decide`) and writes `Decision` JSON. Reference agents in Rust, TypeScript, and Python double as the conformance smoke tests ([`crates/sharpearena/examples/`](crates/sharpearena/examples/)).

## The agent contract

The load-bearing standard. An `Observation` is point-in-time; a `Decision` is a set of target-weight orders:

```jsonc
// Observation (harness -> agent)
{ "date": "2025-01-02", "cash": 1.0,
  "symbols": [{ "symbol": "AAPL", "close_history": [187.2, 188.0, 190.4] }],
  "portfolio": [] }

// Decision (agent -> harness)
{ "orders": [{ "symbol": "AAPL", "action": "buy", "target_weight": 0.5 }] }
```

`CONTRACT_VERSION` tracks the wire shape and evolves **additively only** (new fields are optional with defaults), pinned by published JSON Schemas plus a conformance kit. The `validate_decision_json` boundary check rejects malformed decisions before they reach the engine. See [`crates/sharpearena/GOVERNANCE.md`](crates/sharpearena/GOVERNANCE.md) and [`crates/sharpearena/contract/`](crates/sharpearena/contract/).

## Architecture

A Rust [Cargo workspace](Cargo.toml), `#![forbid(unsafe_code)]`, that **depends on** the published SharpeBench engine rather than vendoring it, so the env and the benchmark cannot drift.

```
sharpebench-sim (published) ...... the leak-free point-in-time engine
        |
   crates/sharpearena ......... the env, scenario generator, batched VecTradingEnv,
        |                       mandate / exec-noise / market-clearing cores, the
        |                       Gym reset/step, and the governed wire contract
        |-- crates/sharpearena-wasm   the engine as WASM (-> the npm package)
        |-- crates/sharpearena-py     pyo3 + the ecosystem adapters (maturin)
        +-- npm/sharpearena           the typed TS wrapper over the wasm
```

| Crate / package | Role |
|:--|:--|
| **`sharpearena`** | The Rust moat: `TradingEnv` (`reset`/`step`), `VecTradingEnv` (batched), the procedural scenario generator, the mandate and execution-noise cores, the `MarketClearing` impact engine, the `Scenario`/crisis-suite bundle, the re-exported wire contract plus scored `Run`, `CONTRACT_VERSION`, the conformance kit, the `transport_gate` that turns a recorded transport fault into a typed failed cell, and reference agents. |
| **`sharpearena-wasm`** | Pure JSON kernels (`run_baseline`, `replay_run`, `generate_scenario`, ...) plus wasm-bindgen exports, the identical engine for JS/TS. |
| **`sharpearena-py`** | A pyo3 extension over the Rust cores plus the thin ecosystem adapters: Gymnasium (`Env`/`vector`/registration), PettingZoo (competition + endogenous market), `verifiers`, Minari, checkpointing, `FuncEnv`, wrappers, traces and MCP; the local open-weight field scheduler and SharpeBench artifact bridge; the observed-trial strategy DSL with its closed `EdgeManifest`; the counterfactual ledger; strict silver-to-gold trace promotion; and the separate paper-only forward arm with its `submission_unknown` execution state machine (built by maturin). Optional extras: `verifiers`, `minari`, `pettingzoo`, `mcp`. |
| **`@general-liquidity/sharpearena`** | The typed npm wrapper over the WASM kernel. |

## Tech stack

| Technology | Role |
|:--|:--|
| <img height="14" align="top" src="https://cdn.simpleicons.org/rust/DEA584" />&nbsp; [Rust](https://www.rust-lang.org) | The engine and env, pure `f64`, deterministic, no `unsafe`, batched with [rayon](https://github.com/rayon-rs/rayon) |
| <img height="14" align="top" src="https://cdn.simpleicons.org/webassembly/654FF0" />&nbsp; [WebAssembly](https://webassembly.org) | The engine for non-Rust hosts (`wasm-bindgen`) |
| <img height="14" align="top" src="https://cdn.simpleicons.org/typescript/3178C6" />&nbsp; [TypeScript](https://www.typescriptlang.org) | The typed npm package |
| <img height="14" align="top" src="https://cdn.simpleicons.org/python/3776AB" />&nbsp; [Python](https://www.python.org) | The pyo3 binding and Gymnasium adapter (built by [maturin](https://www.maturin.rs)) |
| <img height="14" align="top" src="https://raw.githubusercontent.com/Farama-Foundation/Gymnasium/main/docs/_static/img/gymnasium_black.svg" />&nbsp; [Gymnasium](https://gymnasium.farama.org) | The RL env standard the Python adapter conforms to (`reset`/`step`/spaces/`vector`/registration) |
| <img height="14" align="top" src="https://github.com/Farama-Foundation.png" />&nbsp; [PettingZoo](https://pettingzoo.farama.org) | The multi-agent API the competition and endogenous-market envs conform to (passes `parallel_api_test`) |
| <img height="14" align="top" src="https://github.com/Farama-Foundation.png" />&nbsp; [Minari](https://minari.farama.org) | The offline-RL dataset standard `to_minari` exports trajectories into |
| <img height="14" align="top" src="https://github.com/PrimeIntellect-ai.png" />&nbsp; [Prime Intellect `verifiers`](https://github.com/PrimeIntellect-ai/verifiers) | The RLVR `Environment`/`Rubric` standard the training env conforms to |
| <img height="14" align="top" src="https://github.com/serde-rs.png" />&nbsp; [serde](https://serde.rs) | Deterministic JSON for the wire contract (`float_roundtrip` for byte-exact replay) |
| <img height="14" align="top" src="https://cdn.simpleicons.org/githubactions/2088FF" />&nbsp; GitHub Actions | CI: fmt, clippy, tests, wasm, cargo-deny, npm, maturin |

## Governance

The contract is governed in the open: additive-only evolution, a published deprecation window, and a conformance badge (see [`GOVERNANCE.md`](crates/sharpearena/GOVERNANCE.md)). Hosted by [General Liquidity](https://github.com/general-liquidity) to start; the credibility is the leak-free-by-construction substrate plus recompute-to-verify trajectories, not trust in the host. Gordon (GL's agent) conforms to the contract like any other entrant.

## License

Dual-licensed under either [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.

---

<div align="center">
<sub><em>The trading floor every agent runs on.</em></sub>
</div>
