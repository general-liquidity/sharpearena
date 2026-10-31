<!-- prettier-ignore -->
# SharpeArena Evaluation Protocol (v1)

This is the canonical, version-pinned protocol for evaluating a trading agent on
SharpeArena. The benchmark wins the way ALE won: one config is canonical, the env-ID
is version-pinned, the numbers to beat are published, and every entrant agrees to
**report these settings**. Deviate from the canonical config and you are no longer on
the leaderboard; you are running a private experiment.

The single most important rule: **rank on deflated Sharpe plus process checks, never
on raw return.** Raw return-rank is luck. A policy can top any one seed by drawing a
lucky price path, so a return-ranked leaderboard rewards overfitting and survivorship.
The deflated Sharpe discounts for the breadth of the search; pass^k demands the edge
hold on every run, not on average; the process checks reject runs that broke a rule on
the way to the number.

## Canonical environment

| Setting | Value |
|---|---|
| Env-ID | `SharpeArena/Calm-v1`, `SharpeArena/Hard-v1`, `SharpeArena/Extreme-v1` |
| Tier kwarg | `distribution_mode="calm" \| "hard" \| "extreme"` |
| Symbols | `n_symbols=4` |
| Episode length / window | `n_days=120` point-in-time bars (the engine truncates at end-of-window) |
| Action | target-weight vector over symbols, `max_weight=1.0`, shorting allowed |
| Reward | per-bar portfolio return (the series scored by SharpeBench) |

The three tiers are the same point-in-time market under increasing volatility-and-jump
stress (Calm / Hard / Extreme), matching the procedural `ScenarioSpec` generator. An
agent that clears Calm but collapses on Extreme is not robust; report all three.

## Train / held-out split (operator's responsibility)

Seeds are integer-interval bands, and the bands are **provably disjoint** (Procgen's
overfitting-is-measurable thesis, ported to a leak-free market). The operator owns the
split and must not train on test seeds.

| Band | Interval | Count |
|---|---|---|
| Train | `[0, 256)` | 256 |
| Gap | `[256, 10256)` | 10000 (never sampled) |
| Held-out test | `[10256, 10512)` | 256 |

The gap is wide on purpose: train can later grow by up to `gap` seeds without ever
touching the held-out band. The split is produced by
`train_test_seeds(n_train=256, n_test=256, seed_start=0, gap=10000)`, which raises
`ValueError` rather than returning overlapping bands. The check is a `raise`, not an
`assert`, so it is still there under `python -O`; the Rust `train_test_split` refuses the
same two cases with a typed `SplitError` in every build profile. Asking Rust for zero
held-out levels is accepted and means *unbounded*, not *empty*: `num_levels == 0` is
`ScenarioSpec`'s Procgen convention, so the held-out family spans the seed space above the
gap (still disjoint from train). Python's `train_test_seeds` materializes a `range` and so
returns an empty list for the same argument. Report the held-out count you actually
evaluated.

Disjoint is not secret. Because scenarios are deterministic, publishing a bounded
held-out band lets an adversary enumerate it: the paper's predictability probe
recovers 16/16 seeds from a public band of width `2^16` using one observed bar.
Leaderboard operators must sample held-out seeds from a high-entropy private space,
commit to the seed set or its sealed artifact before submissions close, and reveal
the seeds only with the final replay artifacts. The intervals above document the
canonical split; they are suitable for local research comparisons, not a secret
hosted evaluation after publication.

## What you must report

A leaderboard entry is incomplete unless it states all of:

1. **`distribution_mode`** (Calm / Hard / Extreme) and the env-ID.
2. **Slippage and fee model** in force (the cost model the engine charged; default if
   unchanged).
3. **Leakage mode** confirmed: leak-free point-in-time (the `LookaheadGuard` clean,
   no lookahead violations). Any relaxation is a different benchmark.
4. **Deflated Sharpe** (the rank key), **pass^k rate**, **mean return** (context only).
5. **Generalization gap**: `generalization_gap(make_env_for_seed, n_train, n_test)`
   reporting `train` vs `test` deflated Sharpe and `gap_deflated_sharpe`. A large
   positive gap is overfit; near zero generalizes. An entry with a strong train number
   and no reported test number is presumed overfit.
6. **Confidence interval on the deflated Sharpe** and, when comparing entries, the
   **paired-difference verdict** (see the next section). A ranked number with no interval,
   or an "A beats B" claim a paired test calls tied, is a dashboard, not a result.
7. **Declared trial count** (`n_trials`), the number the deflated Sharpe is deflated by.
   Count every configuration whose reward you looked at before choosing the entry: each
   training run and random seed, each hyperparameter setting, each reward shaping, and
   each checkpoint you evaluated to pick the one you submitted. An entry that trained
   ten runs and submits the best checkpoint of the best run has more trials than one,
   and declaring one makes its deflated Sharpe look better than the search that produced
   it. The SharpeBench bridge carries this count into scoring as `in_sample_trials`.

## Cross-regime transfer (a stronger robustness signal)

The generalization gap varies the *seed band* inside one `distribution_mode`, so a policy
that only works in calm markets but is scored solely on calm seeds still passes. The
cross-regime transfer metric closes that hole: it holds the seed band fixed and varies the
*regime*, scoring a policy in-distribution on one tier and **zero-shot** out-of-distribution
on another.

`cross_regime_transfer(make_env_for_seed_and_mode, train_mode, test_mode, seeds)` reports
`in_distribution` and `out_of_distribution` aggregates plus `transfer_gap_deflated_sharpe`
(in-distribution minus out-of-distribution). Because the seed band is identical on both
sides, `train_mode == test_mode` reuses byte-identical envs and the gap is exactly `0` by
construction; a large positive gap on `calm -> extreme` is a regime-specific overfit a
within-tier gap cannot see. The Rust core exposes the protocol primitive
`cross_regime_split(train_spec, test_mode)` (the seed-band-preserving, regime-swapping
sibling of `train_test_split`). Reporting a `calm -> hard` and a `calm -> extreme` transfer
gap alongside the within-tier generalization gap is strictly stronger evidence of robustness.

## Information disclosure (an orthogonal difficulty axis)

`distribution_mode` sets how adversarial the price *path* is; it says nothing about how
much of that market the agent is allowed to *see*. Those are independent axes. A Calm panel
shown through a 3-bar window with no fundamentals or news can be genuinely harder than an
Extreme panel shown through a 50-bar history with full context: the first measures
**information efficiency** and robustness to **data poverty** (the real trading axis of
"information edge") which the regime tiers structurally cannot probe. A scenario is
therefore a point on a 2-D grid: `(distribution_mode × observation_richness)`.

The shared-book market (`PyMarketClearing` / `EndogenousMarketEnv`) takes a `richness`
tier orthogonal to `distribution_mode`:

| Tier | Trailing lookback | Fundamentals | News |
|---|---|---|---|
| `data_poor` | 3 bars | no | no |
| `standard` | 20 bars | no | no |
| `data_rich` | 50 bars | yes | yes |

`standard` is the historical default disclosure, so a scenario built without a `richness`
setting is byte-identical to `standard`: the axis is strictly additive (no
`CONTRACT_VERSION` bump). Richer disclosure only ever surfaces **more past / contextual**
information: the extra bars, the derived `fundamentals` (`trailing_return`, `window_high`,
`window_low`), and the `news` headline are all computed from the same leak-free trailing
closes (`<= t`), so the point-in-time invariant holds at every tier, and no tier reveals a
future bar. Sweeping the same regime across the three tiers isolates how much of an agent's
edge is real signal-processing versus a dependence on being handed a rich observation; a
policy whose deflated Sharpe collapses from `data_rich` to `data_poor` on an otherwise
identical panel is riding disclosure, not skill. Report the tier alongside the
`distribution_mode` (Rust: `MarketClearing::from_dataset_with_richness` /
`RichnessTier`; Python: `PyMarketClearing(..., richness=...)`).

The generator also carries an opt-in `vol_clustering` knob (a deterministic
volatility-clustering post-pass on the tape); it is outside the canonical config, which
stays `vol_clustering=0`.

The endogenous market also exposes an opt-in `impact_exponent`. The canonical
setting is explicitly `impact_exponent=1.0` (linear permanent impact). Values below
one run on a separate `powf` path and are not covered by the canonical byte-identity
goldens; reports using them must state the exponent in every result row.

## Impact robustness diagnostics (optional, rank-neutral)

`sharpearena.impact_diagnostics` holds two diagnostics for the shared-book market. Neither
feeds SharpeBench, the kernel score, this leaderboard, pass^k or any other rank or gate.
Both drive the published `PyMarketClearing` binding with its existing arguments, so the
clearing arithmetic, `SPEC_HASH` and every golden file are untouched. An entry may quote
them next to its score. They are not required.

**Impact-misspecification gap.** The market can clear each bar at the worst case inside an
opt-in elliptic uncertainty set over the impact coefficients `(lambda, eta)`. That set is
adapted from Ma and Huang (2025), "Robust Reinforcement Learning in Finance", and differs
from their construction. Clearing that way does not show whether a policy's profit
depended on impact being exactly the point estimate. The paper judges robustness by how far
a policy's value moves when impact changes, and `impact_misspecification_gap` reports the
same kind of number:

```python
from sharpearena.impact_diagnostics import MarketSettings, impact_misspecification_gap

report = impact_misspecification_gap(
    make_policy,                      # zero-argument factory, called once per arm
    seeds=range(10256, 10280),
    uncertainty={"lambda_radius": 0.05, "eta_radius": 0.02, "correlation": 0.0},
    settings=MarketSettings(n_symbols=4, n_days=120),
)
```

Each seed runs the policy alone in the book twice on the linear kernel
(`impact_exponent=1.0`): once at the point estimate and once against the declared set.
The report gives per-seed and mean total return and per-bar Sharpe under each arm. The
returns mark the final position at the exogenous mid, and each row reports separately
what the engine's own mark, at a cleared mid that carries the arm's permanent impact,
adds (`point_own_impact_mark`, `robust_own_impact_mark`). The report also gives
`return_gap` and `sharpe_gap` (point minus worst case, so a positive return gap is value
the worst case cost), the number of seeds, and a t-based 95% interval on each gap over
the per-seed vector. The paper's relative portfolio gap compares a market with impact to
one without (p. 9), so `return_gap` is an adaptation of that measure, not the same number.

- **Pairing is proved, not assumed.** Each arm's market object is first replayed with flat
  orders, which exposes its exogenous mids exactly. It is then reset and the policy is run.
  The report raises `UnpairedArmsError` unless:
  - the two replays and the two first observations are bitwise identical;
  - each arm's traded tape equals `exogenous_mid * M` bit for bit, with `M` rebuilt from
    that arm's reported flow and applied `lambda`.
- **Degenerate inputs are typed.**
  - A zero-radius set, or a policy that never trades, leaves the arms bitwise identical.
    `return_gap` is then exactly `0.0`.
  - A reward track with all values exactly equal reports `Unavailable("constant_track")`
    instead of a Sharpe ratio. A Sharpe gap needs both ratios, so it is
    `Unavailable("sharpe_unavailable")` whenever either arm lacks one, identical arms
    included, and the mean and interval name the seeds without one.
  - One seed reports `Unavailable("insufficient_seeds")` instead of an interval.
  - A cleared mid at or below zero, which a large flow against the linear multiplier can
    produce, raises `NonPositiveMidError`.
- **The sign is guaranteed only for an eta-only set, and the report says when.**
  - With `lambda_radius = 0` and the same weights sent in both arms, both arms clear at
    identical mids and sizes, and only the fill cost rises, so the gap is never negative.
    Each row records `identical_actions`; a policy that reads `cash` or `avg_price`, or
    draws from random state kept outside its factory, can send different weights. The
    report's `sign_guaranteed` is true only for an eta-only set with identical weights on
    every seed.
  - With `lambda_radius > 0` no sign is guaranteed. The engine's mark would make a held
    position richer under the worst case (after ten equal buys, on 32 of 32 seeds whatever
    the path did); the exogenous mark removes that channel, and the same scale-ins then
    gave a positive gap on every seed. The report does not clamp a negative gap.

**Meta-order impact shape.** Permanent impact in this market compounds bar by bar,
`M_{t+1} = M_t * (1 + lambda * sign(Q/V) * |Q/V|**exponent)`, and never decays. An
exponent below one makes each bar's increment concave in that bar's flow (the nonlinear
regime Huberman and Stanzl analyse for manipulation). It does not reproduce the empirical
square-root law of meta-order impact, which is concave in the total executed quantity and
decays after execution ends.

`meta_order_impact_shape(impact_exponent=...)` measures this on one symbol against a
flat replay of the same path. It reports three numbers:

| Measurement | Square-root law | This kernel |
|---|---|---|
| Log-log exponent of impact against executed quantity, during a constant-rate buy | about 0.5 | about 1, at every exponent |
| Impact after execution ends, as a ratio to impact when it ends | below 1 | exactly 1 |
| Log-log exponent of end-of-execution impact against duration, for a fixed total | about 0 | about `1 - exponent` |

At exponents 1.0 and 0.5 the default probe measures execution exponents within 0.05 of 1
and duration exponents within 0.05 of `1 - exponent`, and the tests pin both.

## Execution settings (optional, versioned, not calibrated)

By default the shared-book market clears every decision against the close it was
made on, compounds permanent impact without decay, and values residual shares at
the last cleared mid. `MarketExecution` (wire schema `sharpearena.market-execution.v1`,
passed as `EndogenousMarketEnv(execution=...)` or `PyMarketClearing(execution=...)`)
changes that. Absent settings leave every published path and golden unchanged.
Every coefficient and decay rate is a declared setting chosen by the caller. None is
fitted to any market, so a result under these settings is not a calibrated claim.

- **Timing.** `fixed_delay_bars=n` uses SharpeBench's `sharpebench.fixed-decision-delay.v1`
  schema and counts bars from the last close a decision observed to its fill. With
  `n=1`, a decision made on bar `t` fills on `t+1` throughout: the first traded close is
  withheld from the first observation. Larger `n` queues decisions first in, first out,
  as in Bench. A queued decision is priced and sized at eligibility, is never replaced
  by a later one, and never fills past the window. `TradingEnv(fixed_delay_bars=n)`
  carries the same FIFO around the pinned engine, which predates the setting. One stated
  difference: an invalid decision is reported when it becomes eligible, not when it was
  submitted.
- **Transient impact.** `transient_coefficient` and `transient_decay` add a displacement
  `I = coefficient * h(J)` that sits beside the permanent multiplier. `J` is the decayed
  cumulative normalized flow, `J <- decay * (J + Q/V)`. The cleared mid is
  `exo * M * (1 + I)`. `M` never decays and `I` never enters `M`. Fills on the net-flow
  side pay the increment in `I` that their bar causes. Fills against the net flow pay
  nothing and receive nothing.
- **Square-root shape.** `transient_shape="square_root"` sets `h(J) = sign(J) sqrt(|J|)`.
  The probe reports this only where the arithmetic produces it (one symbol, seed 0,
  `kyle_lambda=0`, coefficient 0.1):

  | Decay per bar | Execution exponent | Duration exponent | Impact 20 bars after the end, as a ratio |
  |---|---|---|---|
  | 0.5 | 0.06 | -0.48 | 0.001 |
  | 0.9 | 0.28 | -0.28 | 0.35 |
  | 0.99 | 0.47 | -0.03 | 0.90 |
  | 0.999 | 0.50 | 0.00 | 0.99 |

  An exponent near 0.5 with no duration dependence appears only when execution is short
  against the decay horizon, and then relaxation is slow. Fast decay gives quick
  relaxation and no square-root growth. No single setting delivers both, so this is not
  the empirical square-root law, only a concave kernel whose shape the probe measures.
  The linear shape relaxes geometrically at the decay rate and grows about linearly.
- **Settlement.** Order sizes are `capital * (w - prev_w) / mid`. A round trip that is
  flat in target weight is therefore not flat in shares once the mid has moved. On
  seeds 0 to 7 of the default manipulation probe, residual positions under the default
  settings were worth -1.2% to +1.5% of capital. `settlement="cash_liquidation"` sells
  every residual on the last bar as one further clearing at the post-flow price, through
  the same impact terms, so terminal NAV is cash and no residual shares remain.

Every setting is bound into execution identity. `PyMarketClearing.effective_config`
reads the live coefficients, exponent, uncertainty set and execution settings back from
the engine, and `execution_fnv1a64` fingerprints them. A setter call or any changed
setting moves the fingerprint. For `TradingEnv`, a delay adds `fixed_delay` and an
`execution_fnv1a64` over costs and timing to the existing readback. Without a delay the
readback is byte-identical.

**What the manipulation probe found.** Each cell runs a round trip against a paired
zero-impact reference with the same timing and settlement, under cash liquidation, on a
finite grid. A finite grid is evidence about that grid, not a no-arbitrage theorem.

- With no followers and `kyle_lambda = 0` (`eta` 0 or 0.05), no seed of 16
  paid across 768 cells: linear and square-root shapes, decay 0, 0.5, 0.9 and 0.99,
  coefficients 0.05 and 0.2, six asymmetric schedules, both sides and two sizes. A
  hill-climbing search over arbitrary 40-bar weight paths, started from the best ramp,
  found no profitable path under any kernel setting tested (best seed-mean
  `impact_pnl` between -2.9e-5 and -2.6e-3). Under the same search the concave
  permanent exponent 0.5 positive control paid +7.7e-3, so the search could find a
  profit where one existed.
- With three momentum followers, an earlier kernel paid fills against the net flow the
  displacement increment as a rebate. Under the square-root shape the increment's slope
  is unbounded near `J = 0`, and pushing into the followers paid on 53 of 144 cells (best
  seed mean +1.9e-2, 7 of 8 seeds). The engine no longer pays that rebate, and the same
  grid now gives 0 of 288 transient cells with a positive seed mean.
- Under half the increment, in a diagnostic build only, the square-root kernel became
  profitable (+4.2e-2 at decay 0.99). The full-increment charge is what keeps it
  unprofitable, and the result depends on that convention.

The tests pin the grid without followers, the follower cell that found the rebate, the
positive control, and cash-settlement accounting.

## Statistical confidence (is A > B beyond seed noise?)

Deflation handles overfit-luck and pass^k handles per-run reliability, but a leaderboard has
one more thing to defend when two entries are close: is A's deflated Sharpe really higher
than B's, or did A just draw a kinder held-out band? That is a Ch. 19 A/B-testing question
(Advances in Financial Machine Learning) that neither the deflated Sharpe nor pass^k answers.
Two self-contained, deterministic tools close it, both keyed on a fixed resample seed, so a
confidence report replays bit-for-bit.

- **Seed-paired bootstrap CI on the deflated Sharpe.** The held-out seeds are the independent
  sampling units. `deflated_sharpe_ci(per_seed_returns, n_trials)` resamples them with
  replacement, recomputes the deflated Sharpe on each resample, and returns the percentile
  interval `{point, lo, hi, width}`. The `point` is exactly the number the leaderboard ranks
  on (the deflation footprint is matched), so the CI brackets it; the interval widens for a
  noisier or shorter track, where fewer seeds carry the headline. `run_baselines` attaches
  this to every row as `deflated_sharpe_ci`, and `leaderboard_markdown(rows, show_ci=True)`
  prints it as a column.

- **Paired-difference significance test.** `pairwise_significance(rows)` runs `paired_dsr_diff`
  down the ranked board: each bootstrap draw feeds the **same** resampled seeds to both
  neighbours, so the price-path variation the two share is removed from the difference.
  That narrows the interval; it does not isolate skill, because the part of each entry's
  result that depends on how its own strategy meets those paths remains, and the estimate
  is still the deflated Sharpe with its assumptions. When the difference CI straddles zero
  the verdict is `tied`, which means a difference was **not established** under this
  resampling, not that the entries are equivalent; otherwise the difference is established
  at the stated confidence for this seed band. The adjacent comparisons are exploratory
  and not adjusted for multiple testing. `significance_markdown` renders one verdict per
  adjacent pair.

Reproduce over the baselines with:

```bash
cd crates/sharpearena-py
python -c "from sharpearena.baselines import run_baselines, leaderboard_markdown; \
from sharpearena.confidence import pairwise_significance, significance_markdown; \
rows = run_baselines(n_symbols=4, n_days=120, seeds=range(16), distribution_mode='calm'); \
print(leaderboard_markdown(rows, show_ci=True)); print(); \
print(significance_markdown(pairwise_significance(rows)))"
```

The Rust core (`sharpearena::leaderboard_ci`) exposes the same primitives,
`bootstrap_dsr_ci` and `paired_dsr_diff`, over per-seed return series, with the deflated
Sharpe math ported self-contained (Bailey & López de Prado) so no extra dependency is pulled
in to draw the interval.

## Adaptive curriculum (training side)

For *training* (this is a training aid, not a leaderboard rule), an adaptive curriculum
targets difficulty by the agent's online success rate instead of a fixed tier rotation.
`AdaptiveScheduler` / `AdaptiveCurriculumEnv` (Python) and `AdaptiveCurriculum` (Rust) score
each candidate level by the zone-of-proximal-development weight `p * (1 - p)`, up-weighting
levels the agent solves 30-70% of the time (the richest learning signal) and down-weighting
the trivially-solved and hopeless tails. Selection is a pure deterministic function of the
recorded outcome history (Prioritized Level Replay), so a curriculum run replays identically
from its outcome log.

## Baseline leaderboard (numbers to beat)

These are the reference policies every entrant must clear: a do-nothing `flat`, a
buy-and-hold-analog `equal_weight_long`, a one-step `momentum` tilt, and the three
portfolio baselines `min_variance`, `max_sharpe`, and `kelly_vol_target`. They are
produced by `run_baselines` and ranked by `leaderboard_markdown`, so the table below is
fully reproducible.

Reproduce with:

```bash
cd crates/sharpearena-py
python -c "from sharpearena.baselines import run_baselines, leaderboard_markdown; \
print(leaderboard_markdown(run_baselines(n_symbols=4, n_days=120, seeds=range(16), distribution_mode='calm')))"
```

### `SharpeArena/Calm-v1` (n_symbols=4, n_days=120, seeds=range(16))

| Rank | Policy | Deflated Sharpe | pass^k rate | Mean return |
|---|---|---|---|---|
| 1~ | kelly_vol_target | 1.0000 | 0.75 | 0.000337 |
| 2~ | equal_weight_long | 1.0000 | 0.62 | 0.000356 |
| 3 | min_variance | 0.9849 | 0.44 | 0.000240 |
| 4 | flat | 0.0007 | 0.00 | 0.000000 |
| 5= | momentum | 0.0000 | 0.00 | -0.000869 |
| 5= | max_sharpe | 0.0000 | 0.06 | -0.001002 |

`=` exact tie in the ranked estimator (order within a tie is arbitrary); `~` separated only beyond the four displayed decimals. Neither is a claim of statistical separation; use `show_ci=True` and `pairwise_significance`.

Read this honestly: **on Calm, drift is free deflated Sharpe, and rank-eligibility is
what the baselines fail.** The long-drift policies saturate the deflated Sharpe on the
calm tier (the kernel states its 0.5 deflation prior annualized and converts it at 252
bars/year, so steady drift over 16 pooled seeds clears the expected-maximum bar), yet
none of them passes the per-run gate on every seed (best pass^k rate 0.75), so none is
rank-eligible. `momentum` and `max_sharpe` are net losers. The bar is exactly the right
height: a real agent has to produce a positive, process-clean, deflated number that
survives **every** held-out seed, not a lucky mean return.

Earlier revisions of this table showed 0.0000 deflated Sharpe for every baseline on
every tier. That was an artifact of a unit bug in the pre-0.5.0 kernel: the annualized
deflation prior was applied per period unconverted, which set the deflation benchmark
near an annualized Sharpe of 18 on daily bars, a bar nothing clears.

### `SharpeArena/Hard-v1` (n_symbols=4, n_days=120, seeds=range(16))

| Rank | Policy | Deflated Sharpe | pass^k rate | Mean return |
|---|---|---|---|---|
| 1 | equal_weight_long | 0.1114 | 0.31 | 0.000243 |
| 2 | kelly_vol_target | 0.0277 | 0.31 | 0.000118 |
| 3 | min_variance | 0.0024 | 0.19 | 0.000051 |
| 4 | flat | 0.0007 | 0.00 | 0.000000 |
| 5~ | max_sharpe | 0.0000 | 0.00 | -0.000928 |
| 6~ | momentum | 0.0000 | 0.00 | -0.000844 |

`=` exact tie in the ranked estimator (order within a tie is arbitrary); `~` separated only beyond the four displayed decimals. Neither is a claim of statistical separation; use `show_ci=True` and `pairwise_significance`.

### `SharpeArena/Extreme-v1` (n_symbols=4, n_days=120, seeds=range(16))

| Rank | Policy | Deflated Sharpe | pass^k rate | Mean return |
|---|---|---|---|---|
| 1 | min_variance | 0.0243 | 0.06 | 0.000585 |
| 2 | equal_weight_long | 0.0234 | 0.06 | 0.000473 |
| 3 | kelly_vol_target | 0.0029 | 0.12 | 0.000139 |
| 4 | flat | 0.0007 | 0.00 | 0.000000 |
| 5~ | momentum | 0.0000 | 0.06 | -0.000306 |
| 6~ | max_sharpe | 0.0000 | 0.00 | -0.001578 |

`=` exact tie in the ranked estimator (order within a tie is arbitrary); `~` separated only beyond the four displayed decimals. Neither is a claim of statistical separation; use `show_ci=True` and `pairwise_significance`.

The deflated Sharpe to beat collapses from saturated on Calm to ~0.1 on Hard and ~0.02
on Extreme, and the **pass^k rate degrades monotonically with difficulty** for the long
baseline (Calm 0.62, Hard 0.31, Extreme 0.06): the harder vol-and-jump tiers strip out
the easy drift, exactly as intended. Regenerate any tier with:

```bash
python -c "from sharpearena.baselines import run_baselines, leaderboard_markdown; \
print(leaderboard_markdown(run_baselines(n_symbols=4, n_days=120, seeds=range(16), distribution_mode='extreme')))"
```

## The social contract

Report the canonical env-ID, the tier, the cost and leakage model, and the
generalization gap alongside your deflated Sharpe (with its bootstrap CI) and pass^k rate.
Rank on the deflated, process-checked number, and when you claim one entry beats another,
back it with the paired-difference verdict. A score with no held-out gap is a dashboard, not
a result; an "A > B" with no significance test is a coin flip dressed as one.
