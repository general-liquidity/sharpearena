# Realism control protocol (acceptance rows P13-R and P14-R)

Status: frozen before any graded run. Constants live in `protocol.json`, whose SHA-256
every artifact records. A change to either file is a new protocol version, and results
under the old version keep their label.

## Question

`certify_realism` refused the canonical generator on 23 of 24 panels in F4, with no pass
rate declared in advance and no known controls (`paper/sections/07-validation.tex`). This
study asks two things in order.

1. Can this gate separate panels known to carry the gated facts from panels known not to,
   at the 120 by 4 geometry, against a threshold fixed first?
2. If it can, where does each generator tier fall against the same threshold?

The claim is scoped to three stylized facts: excess kurtosis, absolute-return
autocorrelation above the fixed IID null, and aggregation toward Gaussian. A pass does not
establish predictive, behavioural or execution realism.

## Price realism arms

Each simulated arm is graded on 256 diagnostic seeds and 256 disjoint confirmation seeds.

| Arm | Known status | Source |
|---|---|---|
| `pos_garch` | carries the facts by construction | independent GARCH(1,1) Student-t lanes, alpha 0.08, beta 0.90, nu 5 |
| `pos_real` | market tape | non-overlapping windows from two underlyings |
| `neg_gauss` | no fat tails, no clustering | IID Gaussian |
| `neg_iid_t` | fat tails, no clustering, the hardest negative | IID Student-t, nu 4 |
| `neg_shuffled_real` | real marginals, time order destroyed | real windows with each symbol permuted independently |
| `calm`, `hard`, `extreme` | under test | zero-knob generator, flat policy, as F4 |
| `*_vc05` | descriptive only | same tiers with `vol_clustering=0.5` |

Tier panels are collected as F4 collects them: flat policy, 121 closes, 120 returns.

### Real panels

Only two daily series have four or more symbols and distinct underlyings:
`crypto-majors-1d.csv` and `fx-majors-1d.csv`. Each is read from an existing SharpeBench
checkout, its SHA-256 is checked against this protocol, and nothing is fetched. Windows are
consecutive 120-return blocks from the first bar, using the first four symbols in sorted
order. That gives 8 crypto and 34 FX windows. The other crypto bar sizes resample the same
history and are not counted. The trailing partial crypto bar falls outside the last window.

These counts cannot estimate a pass rate. They are read per underlying, by the observed
share, and are never pooled into one rate or padded with overlapping windows.

Artifacts hold derived statistics and the input hash, never an observation. See Open
decision below.

## Price decision rule

All bounds are one-sided 95 percent Clopper-Pearson bounds. Tier bounds use alpha 0.05/3.

1. The gate is **valid** only if, on both seed bands, the `pos_garch` lower bound is at least
   0.75 and every negative arm's upper bound is at most 0.10, and each real underlying has an
   observed pass share of at least 0.75.
2. If the gate is valid, a tier is `market_like_under_gate` when its lower bound is at least
   0.75 on both bands, `not_market_like_under_gate` when its upper bound is below 0.75 on both
   bands, and `inconclusive` otherwise.
3. If the gate is not valid, every tier verdict is `unavailable:gate_failed_control`. Rates are
   still published. A gate that fails on real tape is a finding about calibration at 120 bars,
   not a verdict on the generator.

The failing facts are named for every arm. Nothing is tuned after a look, and there is no grid.

### Why these numbers

Design calibration, run before freezing, used only synthetic panels on 512 throwaway seeds,
no generator and no real data. Observed pass rates: Gaussian 0.010, IID t(4) 0.047, GARCH
(0.08, 0.90, nu 5) 0.861. At n = 256 a true rate of 0.86 gives a lower bound near 0.82, so
0.75 leaves room for the positive control. A true negative rate of 0.05 gives an upper bound
near 0.08. The bar 0.75 is below the 7 of 8 used for the Calm calibration because that rule
was an operational screen on eight seeds, and this one bounds a rate.

### Threats

- `_abs_acf_iid_upper` is calibrated on Gaussian panels, so heavy tails may shift its size.
  `neg_iid_t` and `neg_shuffled_real` measure that directly.
- GARCH is a control for the statistic, not a claim about markets. `pos_real` is what
  connects the gate to tape.
- Two underlyings and one history each. A real-tape pass supports non-vacuity, not coverage.
- A scenario seed band is public. This study does not need secrecy.

## Execution realism

Execution realism is a separate construct, reported in a separate block with no combined
verdict. It runs on `LOBMarketEnv` with `priority="seeded_shuffle"` and records `mark` and
`controllers` in every row. Same seeds are not same paths across arms, so comparisons are of
distributions unless stated.

| Check | Known good | Known bad | Rule |
|---|---|---|---|
| E1 seat exchangeability | two identical quoters, `seeded_shuffle` | `agent_index` | the 90 percent CI of mean fill share minus 0.5 lies inside plus or minus 0.0025; the bad arm's lies entirely above it |
| E2 mark walk | `mark="ex_own_mid"` | `mark="book_mid"` | absolute mean of reward minus final value is at most 5 percent of mean final value; the bad arm exceeds it |
| E3 impact shape | none shipped | none shipped | descriptive: `meta_order_impact_shape` at exponents 1 and 0.5, compared with an exponent in [0.4, 0.6] and relaxation ratio at most 0.9 |

E3 is expected to fail. The kernel is reported as linear and non-relaxing at both settings,
and exponent choice alone does not produce relaxation. E3 is reported, not gated.

## Stages

`smoke` proves plumbing on band 29000 and a synthetic fixture. It never reads a real series
into the gate, never touches the diagnostic or confirmation seeds, and its numbers inform no
threshold. `diagnostic` runs next. `confirmation` refuses to start until a diagnostic summary
exists. Both refuse a dirty tree or an unknown dataset hash. Final runs happen on the merged
engine.

## Reporting

Publish every arm, every band, per-fact pass counts, failed jobs and the complete
denominators. An arm that could not run is `unavailable`, never zero.

## Open decision

`trading/sharpebench/data/RIGHTS.md` records no redistribution grant for the Binance
observations and says current FRED terms prohibit using FRED content to develop a software
system. This study reads the two files locally and publishes statistics only, but the owner
should confirm that the FX arm is acceptable before the final run. If it is not, the study
drops to one underlying and rule 1 is evaluated on crypto alone, which must then be stated.
