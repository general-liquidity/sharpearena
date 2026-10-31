# Learner generalization protocol (acceptance rows P18-E and P18-S)

Status: frozen at version 4 before any pilot. Constants live in `protocol.json`, whose SHA-256 every
artifact records. A change to either file is a new protocol version.

## Question

Does the seen-band versus unseen-band gap move for a policy able to overfit? The published
generalization measurement used an equal-weight reference that cannot overfit, so a gap near
zero there says nothing. The anti-claim to rule out is a flat gap caused by a policy that
learned nothing.

The primary cell trains and evaluates on `martingale_v1`, a tier with no edge by
construction: each symbol is an independent martingale, so any weights chosen from past data
earn zero expected return before costs (`EVALUATION.md`, generator calibration). On that tier
the expected held-out excess over equal weight is zero or below for every policy, so a positive
gap can only come from fitting the seen tapes. That makes the gap a clean read of overfitting,
which the default tiers cannot give: Calm carries tradable momentum (lag-1 autocorrelation 0.23
on long paths, one-bar momentum gross Sharpe near 5.7) and Hard keeps a diluted copy of it, so a
learner there can also learn a real edge and the gap mixes the two.

Scope: a within-generator measurement on two tiers. Learning the synthetic generator is not
learning markets. No claim transfers to real data, and no claim about RL in general follows
from one algorithm and one small search.

## Unit and design

The unit is the independent training run. Episodes from one checkpoint are not replications.

Runs are nested in seen bands: `K` bands of `n_seen` seeds, `m` runs per band differing only
in training seed. Bands are the source of market-history dispersion and runs the source of
training dispersion, and both are reported. `K` is fixed by the pilot rule below, `m` is 2.

| Cell | Role | Tier | Network | Seen seeds | Declared expectation |
|---|---|---|---|---|---|
| A | primary | `martingale_v1` | [256, 256] | 16 | no held-out edge |
| B | capacity contrast | `martingale_v1` | linear | 16 | no held-out edge |
| C | level-count contrast | `martingale_v1` | [256, 256] | 256 | no held-out edge |
| D | declared signal cell | `calm` | [256, 256] | 16 | held-out edge from momentum |

All cells get the same environment-step budget and report wall time and hardware. Cell D always
runs beside A in the main stage, with A's configuration and band count; it is not searched or
sized on its own. Cells B and C run only if cell A passes its capacity check.

Cell D is the positive case. On Calm a learner is expected to pick up the momentum and keep it
on held-out seeds, so a held-out edge there is the expected result, not a finding, and Calm's
gap is not read as overfitting alone. If D shows no held-out edge, the learner failed to learn a
signal known to exist, and that is reported as a limit on what cell A can show.

## Environment

`SharpeArenaEnv` at 4 symbols and 120 days, with each cell's tier from the table above and
the default cost model, read back with
`check_env_effective_config`. A single-agent task with no co-adapting opponents. Each lane is
assigned seen seeds and a reset replays the next of them, so memorising is possible. Training
uses 16 lanes in every cell. In cells A and B each lane holds one seed; in cell C seeds are dealt
to lanes and cycle at each reset, so rollout size and optimiser updates match across cells. The
seen band is the training seed list; dev, pilot-unseen and held-out bands never enter training. For cell
D, `hard` and `extreme` are evaluated zero-shot on the held-out seed numbers as a secondary
transfer read. The martingale cells have no transfer read: their tier carries no direction to
transfer.

`martingale_v1` daily volatility is about four times Calm's. The fixed reward scale is kept for
every cell, so the reward magnitude differs by tier; this is stated rather than corrected,
because a per-tier scale would be one more searched constant. Outcomes are in ledger basis
points and comparable within a cell, not across tiers.

Observation: the normalised `closes`, `positions` and `cash`. Holdings are observed, which
matters because costs depend on them. The normaliser is fitted once, before training, on the
seen band under a seeded random allocator with `training=True`. It is then frozen with
`training=False` for all of training and evaluation through `state_dict` and
`load_state_dict`, so both bands see one transform.

**Merge dependencies.** `martingale_v1` exists only on `feat/generator-calibration`, so no
stage can run on a tree without it. The normaliser interface needs `feat/normalizer-freeze`. The producer refuses to run `pilot`
or `main` without that interface. `smoke` alone may use the shim in `produce.py`, and the
artifact records `normalizer_backend`.

Reward: the environment's per-bar return times 100, for training only. Reward is not the
ledger. All outcomes come from the unscaled evaluation returns.

The study does not use `CausalNormalizeReward`. The training scale is the constant above, so
it depends on no data. On `feat/normalizer-freeze` (091f9de) a frozen reward normaliser divides
by the std of the last finished episode only. That is not acceptable for this study and is
harmless only because the study does not use it. One 120-bar episode is a noisy estimate of
scale, and which episode finishes last depends on lane order and the point of the freeze, so
the effective reward scale would change with the training seed and the seen band. That is an
uncounted hyperparameter that varies inside the unit of replication. A later version that wants
a learned scale needs one fitted over the whole seen band, and is a new protocol version.

## Algorithm and search

SB3 PPO on CPU, one torch thread, `MultiInputPolicy`, `protocol.json` hyperparameters, budget
2,000,000 environment steps, ten evenly spaced checkpoints.

Search is four configurations, frozen in `protocol.json`. It runs in the pilot only, on pilot
bands, and picks the configuration with the highest seen-band excess among those not flagged
inactive. It never reads the held-out band. Should no configuration pass capacity at 2 million
steps, the budget may escalate once to 8 million for all configurations. Every configuration
and escalation is counted as a trial. A framework switch would be a new protocol version and
counts as development search. Results are not a claim about RL defaults in either direction.

## Outcomes

For run `r` and seed `s`, let `x(r,s)` be mean bar return in basis points minus equal weight's
on the same seed. Differencing removes band difficulty.

**Primary:** `G = mean x over seen - mean x over held-out` at the final checkpoint.

Secondary: `G` at the best checkpoint; `gap_deflated_sharpe` as EVALUATION.md requires; the
held-out excess; zero-shot transfer; cells B and C contrasts.

**Final and best are both reported.** Best is picked on the dev band only. Selection is search:
`n_trials` for final is `configs_tried * runs_in_cell`, and for best it is that times
`checkpoints`. Failed runs stay in the count.

Analysis: mean of `G` over bands with a t interval on `K - 1` degrees of freedom over band
means. Variance components for band and run are reported beside it.

## Pilot and sample size

Six bands with two runs each on pilot bands fix the design. Let `s` be the band-mean SD of
`G`. It is inflated to its 80 percent upper confidence limit,
`s * sqrt((n - 1) / chi2(0.20, n - 1))`. `K` is the least integer with
`t(0.975, K - 1) * s_upper / sqrt(K) <= 0.5` basis points per bar, capped at 32. If the cap
binds the half-width achieved is stated. Pilot runs are never reused in the main study. The
result goes in `pilot_decision.json`, committed before `main`. The target half-width is half
the smallest effect of interest, 1 bp per bar, about 2.5 percent a year at 252 bars.

## Controls

Evaluated on the same seen, dev and held-out seeds as the learner:

| Control | Purpose |
|---|---|
| equal weight (`EqualWeightLongPolicy`) | reference for excess, rebalanced each bar |
| buy-and-hold | closes-weighted drift from equal weight at t0, no rebalance |
| flat (`FlatPolicy`) | null, and band difficulty read |
| `kelly_vol_target` | causal trailing allocator |
| zero-skill floor | 64 seeded random fully invested allocators per band, quantile 0.95 of the band mean. A floor, not one draw |
| exposure-matched timing null | see below |
| memoriser | instrument check: foresight on each seen tape, per symbol the best path over short, flat and long after the engine's own cost, flat elsewhere. Must show a gap whose interval excludes zero, or the instrument is broken |

Timing null: 200 draws over the first 32 held-out seeds, replayed through the engine. It mirrors
SharpeBench `timing_null`: holding periods at gross above 1e-9 keep their contents and are
moved to random bars. It is valid here because own orders do not move the closes, which the
producer asserts. A continuous policy can be invested on every bar, which Bench reports as
`always_invested`. That stays typed, and the producer then applies a predeclared circular shift
of the action sequence, labelled as such.

## Inactivity and failure

Each evaluation records mean gross exposure, mean turnover and action standard deviation over
time. Below 0.05, 0.001 or 0.001 respectively the run is flagged. Flagged runs stay in the
denominator. Failed, crashed or non-finite runs stay in the denominator with a typed status.
A run is never rerun for its result. Rerunning after an infrastructure failure is allowed and
recorded. Exclusions are not allowed.

## Decision rule

1. **Instrument:** the memoriser gap interval excludes zero. Otherwise `unavailable:instrument`.
2. **Capacity:** the lower 95 percent bound of seen-band excess exceeds 1 bp per bar and the
   mean seen excess beats the floor. Otherwise `unavailable:no_capacity`, which means the gap
   carries no generalization information.
3. **Gap:** `overfit_detected` if the interval is above zero; `within_margin` if it is inside
   plus or minus 1 bp; otherwise `inconclusive`.

Beating a control is a separate question, read on held-out excess against each control with
Holm correction over the controls. It never changes the gap verdict.

The memoriser's cost is the engine's cost per unit of weight traded, measured once per tier
on the reserved seed 79900 by alternating a full long and a full short book. It is a
measurement of the engine, not a fitted constant, and no scored seed is used.

## Declared-signal read

Reported beside the gap verdict for every cell and never replacing it. `heldout_edge` if the
95 percent interval of held-out excess is above zero, `no_heldout_edge` if it lies inside plus
or minus 1 bp, otherwise `inconclusive`. Each cell carries its declared expectation from the
table, and a mismatch is reported as such. On `martingale_v1` a `heldout_edge` result would mean
luck or an artifact, and is reported with its seed count.

## Amendments

Version 2, before any run: a fixed 16 lanes, with seeds cycling in cell C. Version 1 tied the lane
count to the seed count, which would have changed rollout size and updates per environment step
between cells and broken the equal budget.

Version 3, before any run: records that no reward normaliser is used and why the last-episode
frozen scale on `feat/normalizer-freeze` would not be acceptable. No outcome, threshold or
budget moved.

Version 4, before any pilot or graded run: the primary cell moved from `hard` to
`martingale_v1`, Calm became the declared cell D, and the declared-signal read was added. Branch
`feat/generator-calibration` (0cd9e50) measured that the default tiers carry tradable momentum,
which would have let a learner earn a real held-out edge and blurred the gap. Seed bands,
budget, thresholds and the decision rule did not move. The new tier is opt-in under a frozen
name, so this protocol depends on that branch merging before any graded run.

Also in version 4: the memoriser became cost-aware. The version 3 smoke run on Calm showed the
next-bar sign rule earning 31.5 bp gross and losing 4.1 bp net per bar, because engine costs
exceed Calm's small moves, so cell D's instrument check would have failed for a reason that has
nothing to do with the gap. This was found on smoke seeds only, and the rule it changes is the
instrument check, not an outcome.

## Publication

Publish all runs, failed and unfinished included, training curves, trial counts, turnover and
exposure, seed dispersion and band dispersion. A seed-averaged return is a statistic about
runs, not a deployable policy.

## Stages

`smoke`, `pilot`, `pilot-decide`, `main`. `main` needs a committed `pilot_decision.json`, a
clean tree and the freeze interface. Final runs happen on the merged engine.
