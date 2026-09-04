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
| Negative controls | Misspecified reward/proxy pairs that must underperform the clean scorer |
| Generalization | Disjoint train/eval bands, cross-regime transfer, bootstrap confidence intervals, frozen eval set |
| Offline RL | Minari export and train/test dataset helpers |
| Evaluation contract | [`EVALUATION.md`](../EVALUATION.md) fixes seeds, gates, baselines, and reporting semantics |

## Market tasks

| Task | Surface |
|---|---|
| Portfolio allocation | Simplex weights and log-return reward |
| Execution | VWAP/TWAP implementation shortfall |
| Market making | Avellaneda–Stoikov closed-form reference and regret |
| Shared endogenous market | PettingZoo parallel env with Kyle/Almgren–Chriss impact |
| Limit-order book | Integer ticks, price-time matching, limit/market/cancel/modify, call auction, depth and sweep-cost queries |
| Ecology | Deterministic population selection, mutation, regime/liquidity shocks, outcome and coalition classification |
| Adverse selection | Paired informed/uninformed meta-order arms and exact markout decomposition |
| Manipulation diagnostics | Symmetric/asymmetric schedules, impact boundary and size-response sweeps, explicit finite-grid scope |

## Agent operations

| Capability | Surface |
|---|---|
| External contract | Observation JSON in, validated Decision JSON out, by stdio or HTTP |
| Local open-weight field | Fixed scaffold, Ollama/OpenAI-compatible shims, stable shards, append-only resume, model/runtime identity, strict failed cells |
| Strategy generation | Closed non-executable DSL, host-counted trial footprint, disjoint selection/test windows, unit-typed `EdgeManifest` kill conditions |
| Paper-only forward arm | Read-only data, in-memory or fixed Alpaca paper endpoint, deny-first risk guard, crash-persistent unknown-submission reconciliation |
| Deferred claims | Commit now, resolve later through a desk with no dataset or future-data path |
| Trace promotion | Strict silver-to-gold conversion of flagged traces into immutable minimal regressions after an operator decision |
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
- A PrimeIntellect Environments-Hub listing is not yet shipped.
