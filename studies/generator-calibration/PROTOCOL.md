# Generator calibration protocol (acceptance row 38)

Status: frozen at version 1 before any candidate was generated or scored. Constants live
in `protocol.json`. A change to either file after the first grid run is a new version on
new seeds, and results under this version keep their label.

## Question

Does the default generator hand an agent a tradable signal, and can a named tier remove
that signal while keeping clustered volatility and fat tails?

`baseline.json` answers the first half on the unchanged generator, seeds `[0, 256)`. Calm
carries it: lag-1 return autocorrelation 0.18 on 119 returns and 0.23 on long paths, and
gross Sharpe near 5.7 for one-bar momentum. The Calm recursion and the per-symbol drift
are the cause. Those two terms are not touched, because the default stays byte-identical.

This is a development calibration. It is not the frozen realism study (rows P13-R and
P14-R, `studies/realism/` on `feat/study-protocols`), which grades other arms on seed bands
from 20000 up. The two share no seeds, and nothing here changes a gate or a threshold.

## Candidate tier: `martingale_v1`

Each symbol is independent and starts at 100. For each bar, on one SplitMix64 stream:

```text
z_t    = U1 + U2 + U3 + U4 - 2                  four uniform draws, bounded in [-2, 2)
z_t    = k * z_t  when a fifth draw is below p  symmetric scale mixture, the fat tail
r_t    = s_t * z_t
price  = price * (1 + r_t)
s_t+1  = min(w + a * |r_t| + b * s_t, cap * sbar)
```

with `E|z| = m = (7/15) * ((1 - p) + p * k)`, `w = sbar * (1 - a * m - b)`, and a burn-in of
`burn_in_bars` so that the first recorded bar is drawn from near the stationary scale.

This is an absolute-value GARCH recursion, chosen because it needs no square root and
therefore stays inside the generator's mul/add/div/max rule. Exactly five draws are
consumed per bar whatever the parameters, so one seed gives one set of uniforms for every
cell of the grid.

`z_t` is independent of the past and symmetric, and `s_t` is known at `t - 1`, so
`E[r_t | past] = 0`. That is the design claim: no drift and no autocorrelation by
construction, for any parameter value. The grid only sets the volatility dynamics. The
`max(-0.95)` floor must be unreachable, which is checked as `cap * sbar * 2 * k < 0.95`,
otherwise the floor would add a positive conditional mean.

Not modelled, by design: a common market factor, the leverage effect, gain/loss skew,
overnight structure, and any relation between volume and price.

## Fit

Seeds `[0, 128)` fit. Seeds `[128, 256)` report. Nothing computed on the report seeds
feeds the choice.

The target comes only from `us-indices-1d.csv` and `commodities-1d.csv` in SharpeBench
`data/`, read from the existing checkout and checked against the `.sha256` sidecar. The
crypto and FX files are the real-tape controls of the frozen realism study, so they are
kept out of the fit and shown only as a comparison. `rates-1d.csv` is a yield level rather
than a price and is excluded, as are the other bar sizes, which resample the same history.

Each series is cut into consecutive 120-close windows from its first bar, so that a window
gives 119 log returns, the same length as a generated panel. A window with a non-positive
close is dropped by rule; the April 2020 WTI print is the expected case. The three targets
are the medians, over windows, of per-series excess kurtosis, mean `|r|` autocorrelation
over lags 1 to 10, and daily volatility. Medians are taken over per-series values so that
one estimator is applied to both sides.

For each grid cell, on fit seeds:

```text
J = ((median kurtosis - target) / target)^2 + ((median abs ACF - target) / target)^2
```

Volatility is not on the grid. The recursion is homogeneous in `sbar`, so `sbar` is set
after the shape is chosen, by scaling to the volatility target on fit seeds and rounding to
four significant figures.

Rule: among cells with a fit-seed gate pass share of at least 0.75, take the smallest `J`.
If no cell reaches 0.75, take the smallest `J` overall and report that the bar was missed.
Ties break toward lower `a`, then lower `p`, then lower `k`. The 0.75 is the bar the frozen
realism study uses for a tier.

The fit needs many parameter values, and `ScenarioSpec` exposes none, so `fit.py` holds an
exact port of the recursion. The port must reproduce the Rust tier bit for bit on the
report seeds; if it does not, the fit is void.

## Pass criteria, report seeds only

Every interval is a t-based 95% interval across seeds. The seed is the replicate, with 128
of them. A criterion below requires the whole interval to sit inside its margin, so each
is an equivalence test and passing all of them is an intersection-union test, which needs
no multiplicity adjustment.

| ID | Statistic | Pass when |
|---|---|---|
| C1 | lag-1 autocorrelation, 119 returns | interval inside `[-0.03, 0.03]` |
| C2 | mean daily return | interval inside `[-0.0005, 0.0005]` |
| C3 | buy-and-hold Sharpe, mean over symbols | interval inside `[-0.5, 0.5]` |
| C4 | gross Sharpe, equal-weight buy-and-hold, one-bar momentum, one-bar reversal, 20-bar trend | each interval inside `[-0.5, 0.5]` |
| C5 | median excess kurtosis | above zero and above Calm |
| C6 | mean `|r|` autocorrelation | interval above the gate's IID null for this geometry |
| C7 | gate pass count | at least 75% of panels, a development measurement |

The margins were set from the sampling spread alone: 128 seeds of a four-symbol panel give
a Sharpe standard error near 0.07 to 0.13, so a margin of 0.5 can be met by a true zero and
cannot be met by anything near Calm's 1.7 to 5.7.

A failed criterion is reported as failed. The report seeds are not refit, and a retune is
`martingale_v2` on new seeds.

## Threats

- The rules in C4 are four named examples, so passing them does not prove no rule works.
  The argument for that is the conditional mean above, which covers every rule that sees
  only the past. It does not cover costs, which only subtract.
- Zero mean means zero expected per-bar portfolio return for any adapted weights. It does
  not mean an agent has nothing to learn: volatility is forecastable, and risk can be
  managed. `vol_forecast_corr` measures that directly.
- Two target markets over one history each, and one window length. The fit does not claim
  coverage of any other market or horizon.
- Sampling variance of a Sharpe on 119 bars stays large on one seed. A lucky seed is still
  lucky, which is why a score here needs its seed count stated.
