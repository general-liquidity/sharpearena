# Capability map

This page is the current-main feature inventory. It groups related surfaces so
the root README can stay task-oriented.

## Environment core

| Capability | Public surface |
|---|---|
| Point-in-time lifecycle | `TradingEnv.reset/step`, `run_backtest`, capture and replay over one shared step body |
| Procedural scenarios | `ScenarioSpec`, Calm/Hard/Extreme, cointegrated pairs, regime shifts, disjoint train/eval seed bands, sealed evaluation seeds |
| Batched rollouts | Native `VecTradingEnv` and Gymnasium `SharpeArenaVectorEnv` with explicit autoreset modes |
| Execution model | Fees, slippage, impact, financing, seeded execution noise, mandates, drawdown/stop/cascade failure taxonomy |
| Observations | Causal indicators, multi-timescale momentum, covariance, Kalman spread/trend, synthetic news, horizon, flatten/unflatten helpers |
| Real data | Gap-aware contiguous-block sampling that cannot reset across an exchange/listing discontinuity |
| Checkpointing | Native snapshots, Python clone/restore/branch, and functional environment view |

## Training and evaluation

| Capability | Public surface |
|---|---|
| Gymnasium | Scalar/vector envs, registered difficulty and held-out IDs, causal wrappers, `check_env` determinism harness |
| `verifiers` / RLVR | Multi-turn environment, scenario dataset, XML decision parser, bounded reward over the SharpeBench score |
| Reward shaping | Differential Sharpe, Sortino, drawdown, turnover, loss aversion, and causal risk-aware schemes; never the rank key |
| Negative controls | Incomplete research rewards and distinct hand-written proxy policies; diagnostic comparisons, not trained optimizers or guaranteed underperformance |
| Generalization | Disjoint train/eval bands, cross-regime transfer, bootstrap confidence intervals, frozen eval set |
| Offline RL | Minari export and train/test dataset helpers |
| Evaluation contract | [`EVALUATION.md`](../EVALUATION.md) fixes seeds, gates, baselines, and reporting semantics |

## Market tasks

| Task | Surface |
|---|---|
| Portfolio allocation | Simplex weights and log-return reward |
| Execution | VWAP/TWAP implementation shortfall |
| Market making | Avellaneda–Stoikov closed-form reference and seed-paired regret (`mm_regret` raises `UnpairedMidPathError` when the two arms' mids, arrival counts or generator positions differ after any step, as they do at high arrival rates, also when `sigma = 0` keeps every mid equal); per-step reward split into spread capture, inventory mark-to-market, running penalty and liquidation cost (`mm_pnl_split`, diagnostic only), and the same split of a paired regret under the same refusal (`mm_regret_split`) |
| Shared endogenous market | PettingZoo parallel env with Kyle/Almgren–Chriss impact |
| Limit-order book | Integer ticks, price-time matching, limit/market/cancel/modify, call auction, depth and sweep-cost queries; PettingZoo `LOBMarketEnv` values each agent's inventory at the mid of other controllers' resting quotes, or at the last fill between two different controllers while those quotes lack a side (`mark="ex_own_mid"`, the default; `controllers` groups the seats one entrant runs; `"book_mid"` replays the pre-2026-09-16 mark) and offers an opt-in seeded same-bar seat shuffle (`priority="seeded_shuffle"`; default `"agent_index"`) |
| Ecology | Deterministic population selection, mutation, regime/liquidity shocks, outcome and coalition classification |
| Adverse selection | Paired informed/uninformed meta-order arms and exact markout decomposition |
| Manipulation diagnostics | Symmetric/asymmetric schedules, impact boundary and size-response sweeps, explicit finite-grid scope, rank-neutral per-follower seat-removal externality (live P&L minus P&L with the manipulator held flat on the same seed) |
| Impact robustness | Paired point-estimate against worst-case report over an opt-in elliptic uncertainty set (`impact_misspecification_gap`: returns mark the held position at the exogenous mid and report the arm's own impact mark beside them, refusing unpaired arms or a cleared mid at or below zero; `sign_guaranteed` holds only for an eta-only set with identical weights in both arms), and the meta-order impact-shape probe (`meta_order_impact_shape`: execution exponent, post-execution relaxation ratio, duration exponent); both rank-neutral |

## Agent operations

| Capability | Surface |
|---|---|
| External contract | Observation JSON in, validated Decision JSON out, by stdio or HTTP |
| Local open-weight field | Fixed scaffold, Ollama/OpenAI-compatible shims, stable shards, append-only resume, model/runtime identity, strict failed cells, and source-labelled per-request inference accounting |
| SharpeBench bridge | Complete-grid validation, ordinary score submissions, and a rank-neutral operational profile with nearest-rank p50/p95 latency, token totals, reasoning-token provenance, and retries |
| Strategy generation | Closed non-executable DSL, host-counted trial footprint, disjoint selection/test windows, unit-typed `EdgeManifest` kill conditions |
| Paper-only forward arm | Read-only data, in-memory or fixed Alpaca paper endpoint, deny-first risk guard, crash-persistent unknown-submission reconciliation |
| Deferred claims | Commit now, resolve later through a desk with no dataset or future-data path |
| Trace promotion | [V2 producer replay](trace-promotion.md): content-bound silver candidates, recorded operator decisions, and gold checks over fresh Gym/native output from the complete action prefix |
| MCP | Episode `reset`, `step`, and `spec` tools over the Python environment |

## Known limits

- One prospective field is retained as a superseded engineering pilot: three
  older, already-cached local model snapshots, one fixed forecast scaffold, 24
  binary contracts, and six settlement-clock blocks. It validates lifecycle
  plumbing but is excluded from model evaluation, benchmark rank, and the
  paper's empirical conclusions. CI still uses deterministic model doubles and
  downloads no weights.
- SharpeArena provides structural point-in-time leak freedom, not process or
  container containment. Use SharpeBench's digest-pinned `--image` path for an
  untrusted entrant.
- The paper-trading lifecycle stores the broker's cumulative `filled_qty` view;
  it does not independently sum fill deltas. `reconcile_all` stops on an
  unanswerable broker query rather than retrying or guessing.
- The local-field target-weight surface does not have the intended-versus-filled
  per-order quantities required to populate the forward arm's counterfactual
  ledger without inference, so it does not pretend to do so.
- Trace promotion supports one built-in Gym/actions producer, without model calls.
  Replay inputs contain private seeds and possibly full CSV data. Its Python
  socket guard is process-global and provides neither concurrent replay safety
  nor OS isolation; hashes and operator metadata do not authenticate authorship.
- The limit-order book has no self-trade prevention, and its canonical batch
  order queues lower agent indices first at a shared tick. In `LOBMarketEnv` a
  self-trade needs a step that left one book side empty (2 of 27,552 fills over
  400 seeded random configurations) and nets to zero in the agent's cash and
  inventory. Under the default `priority="agent_index"`, seat 0 of two identical
  quoters was filled more on 32 of 32 seeds; `priority="seeded_shuffle"` makes
  the seats exchangeable (15 of 32) without changing the engine, `SPEC_HASH` or
  the golden fill tape. The rule lives in the Python environment only.
- The `LOBMarketEnv` inventory mark leaves out an agent's own quotes but not their
  effect on the one reference mid every seat quotes around. With two agents and
  the noise trader off, an agent quoting `(1, 20)` moved its own mark from 1000
  to 1001 ticks with no fill. Seats that one entrant runs stay out of each other's
  marks only when `controllers` declares them; the environment cannot infer it.
  A lone agent is marked at its last fill against the noise trader, the only
  counterparty it has.
- A PrimeIntellect Environments-Hub listing is not yet shipped.
