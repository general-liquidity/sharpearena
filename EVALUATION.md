<!-- prettier-ignore -->
# OpenOutcry Evaluation Protocol (v1)

This is the canonical, version-pinned protocol for evaluating a trading agent on
OpenOutcry. The benchmark wins the way ALE won: one config is canonical, the env-ID
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
| Env-ID | `OpenOutcry/Calm-v1`, `OpenOutcry/Hard-v1`, `OpenOutcry/Extreme-v1` |
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
touching the held-out band. The split is produced (and disjointness asserted) by
`train_test_seeds(n_train=256, n_test=256, seed_start=0, gap=10000)`.

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
  neighbours, so the price-path luck common to both cancels and the difference isolates skill.
  When the difference CI straddles zero the two entries are **statistically tied**; otherwise
  the higher-ranked one wins **beyond seed noise**. `significance_markdown` renders one verdict
  per adjacent pair.

Reproduce over the baselines with:

```bash
cd crates/openoutcry-py
python -c "from openoutcry.baselines import run_baselines, leaderboard_markdown; \
from openoutcry.confidence import pairwise_significance, significance_markdown; \
rows = run_baselines(n_symbols=4, n_days=120, seeds=range(16), distribution_mode='calm'); \
print(leaderboard_markdown(rows, show_ci=True)); print(); \
print(significance_markdown(pairwise_significance(rows)))"
```

The Rust core (`openoutcry::leaderboard_ci`) exposes the same primitives,
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

These are the trivial reference policies every entrant must clear: a do-nothing `flat`,
a buy-and-hold-analog `equal_weight_long`, and a one-step `momentum` tilt. They are
produced by `run_baselines` and ranked by `leaderboard_markdown`, so the table below is
fully reproducible.

Reproduce with:

```bash
cd crates/openoutcry-py
python -c "from openoutcry.baselines import run_baselines, leaderboard_markdown; \
print(leaderboard_markdown(run_baselines(n_symbols=4, n_days=120, seeds=range(16), distribution_mode='calm')))"
```

### `OpenOutcry/Calm-v1` (n_symbols=4, n_days=120, seeds=range(16))

| Rank | Policy | Deflated Sharpe | pass^k rate | Mean return |
|---|---|---|---|---|
| 1 | flat | 0.0000 | 0.00 | 0.000000 |
| 2 | equal_weight_long | 0.0000 | 0.62 | 0.000356 |
| 3 | momentum | 0.0000 | 0.00 | -0.000869 |

Read this honestly: **the deflated Sharpe to beat is 0.0.** None of the trivial
baselines establish a deflated edge. `equal_weight_long` drifts up on average (positive
mean return, pass^k on 62% of seeds) but the SharpeBench kernel floors its deflated
Sharpe to zero once the search breadth is accounted for, and `momentum` is a net loser.
The bar is exactly the right height: a real agent has to produce a positive,
process-clean, deflated number that survives the held-out band, not a lucky mean return.

### `OpenOutcry/Hard-v1` (n_symbols=4, n_days=120, seeds=range(16))

| Rank | Policy | Deflated Sharpe | pass^k rate | Mean return |
|---|---|---|---|---|
| 1 | flat | 0.0000 | 0.00 | 0.000000 |
| 2 | equal_weight_long | 0.0000 | 0.31 | 0.000243 |
| 3 | momentum | 0.0000 | 0.00 | -0.000844 |

### `OpenOutcry/Extreme-v1` (n_symbols=4, n_days=120, seeds=range(16))

| Rank | Policy | Deflated Sharpe | pass^k rate | Mean return |
|---|---|---|---|---|
| 1 | flat | 0.0000 | 0.00 | 0.000000 |
| 2 | equal_weight_long | 0.0000 | 0.06 | 0.000473 |
| 3 | momentum | 0.0000 | 0.06 | -0.000306 |

The deflated Sharpe to beat is 0.0 on every tier, but note the **pass^k rate degrades
monotonically with difficulty** for the long baseline (Calm 0.62, Hard 0.31, Extreme
0.06): the harder vol-and-jump tiers strip out the easy drift, exactly as intended.
Regenerate any tier with:

```bash
python -c "from openoutcry.baselines import run_baselines, leaderboard_markdown; \
print(leaderboard_markdown(run_baselines(n_symbols=4, n_days=120, seeds=range(16), distribution_mode='extreme')))"
```

## The social contract

Report the canonical env-ID, the tier, the cost and leakage model, and the
generalization gap alongside your deflated Sharpe (with its bootstrap CI) and pass^k rate.
Rank on the deflated, process-checked number, and when you claim one entry beats another,
back it with the paired-difference verdict. A score with no held-out gap is a dashboard, not
a result; an "A > B" with no significance test is a coin flip dressed as one.
