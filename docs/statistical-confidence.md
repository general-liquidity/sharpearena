# Baseline scores and statistical confidence

`run_baselines` reports the pooled deflated Sharpe and a separate per-seed
pass rate. Its Markdown table sorts available deflated Sharpe scores; it is
not SharpeBench's full eligibility board. Raw mean return is descriptive.

If the scoring kernel reports an error, the affected score or pass-rate
summary contains an explicit unavailability reason instead of a numeric floor.
Unavailable scores appear after measured scores, have no numeric rank, and
are excluded from adjacent paired-score comparisons. This applies with or
without confidence reporting enabled.

## Confidence requirements

The Rust `bootstrap_dsr_ci` and `paired_dsr_diff` APIs return
`Result<_, ConfidenceError>`. Native Python calls raise `InvalidArgument`
on refused requests; the Python convenience layer also validates argument types.

Both APIs require:

- At least two seed units, each containing at least two finite returns.
- At least two bootstrap draws and `0 < alpha < 1`.
- Finite, nonnegative trial dispersion and finite computed estimates.
- For paired differences, equal seed counts and equal return lengths within
  each pair. No input is silently reduced to a shared prefix.

Integer parameters are not rounded from floats or converted from booleans.
`run_baselines` refuses duplicate seed IDs. With only one seed it can still
report an available point estimate, but both interval fields are `None` and
`confidence_status` records why between-seed uncertainty is unavailable.

An available Arena interval is attached as `deflated_sharpe_ci` only when
the scoring kernel reproduces its point estimate exactly. A failed kernel,
unsupported resampling request, or differing point estimate withholds that
attachment. `arena_deflated_sharpe_ci` remains separately named when available.

## Interpretation

These are necessary input checks, not a coverage theorem. Two draws are a
minimum for a distribution, not a recommended precision target; choose a
resampling budget appropriate to the tail probability being reported.
Seed independence and positional identity alignment remain caller assumptions.
Distinct seed IDs alone do not prove independence. A zero-width interval can
still occur for identical seed results or a saturated score.

Paired resampling retains shared-path covariance. It does not isolate skill or
remove all luck. Adjacent comparisons are exploratory and not adjusted into a
simultaneous ranking guarantee. Historical paper evidence is unchanged.
