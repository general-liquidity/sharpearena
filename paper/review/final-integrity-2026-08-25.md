# Stage 4.5 Final Integrity Report — 2026-08-25

## Verdict

**PASS for the current, explicitly bounded claims.** This supersedes the
2026-08-24 report. It is not a certificate of market realism, cryptographic
security, or external trading performance.

## Corrections re-audited

| Finding | Resolution verified |
|---|---|
| Nonlinear-impact dimensions | The nonlinear branch applies the exponent to dimensionless crowd flow `Q/V`; the temporary term remains `eta*q_i/V`; the `kappa=1` branch retains the original arithmetic. A regression test rescales flow and volume together and obtains the same impact fraction. |
| Positive-control selection | The search family is declared as 135 cells. The selected-cell intervals include both pointwise Student-t intervals and two-sided Bonferroni familywise 95% intervals over all 135 cells. The stored critical value, 6.391202695754376 at 7 degrees of freedom, agrees with the t quantile at `1 - .05/(2*135)`. |
| Finite-grid scope | The abstract and body say all 45 **sampled** linear cells have negative means; they make no universal linear-impact claim and identify the unswept axes. |
| Concave counts | The manuscript reports 7/23 stored `kappa=.5` slots crossing zero (five unique configurations) and 0/23 at `kappa=.7`, matching the JSON. |
| Eligibility witness | One standardized noise path is reused across strengths (common random numbers). The producer checks coarse-grid monotonicity and bisects only monotone first crossings. All ten reported thresholds are monotone; the two unattained sign-follow/Calm cases have no reported boundary. |
| Fano statistic | It uses sample variance (`ddof=1`) but is explicitly exploratory and ungated because its threshold is estimated from the same 121-bar panel and the finite-sample null is not calibrated. F4's conjunction contains only the three stated directional checks. |
| Predictability uncertainty | Every plus/minus value is labelled across-seed standard deviation. No hypothesis-test or equivalence conclusion is inferred from those SDs. |
| Sealed seeds | The text calls the result a simulated commit-reveal replay that defeats one bounded-band scanner while the salt is withheld. It makes no universal injectivity, deployed-custody, or cryptographic-security claim; the current companion window is explicitly excluded. |
| Evidence provenance | `paper/evidence/provenance.json` records the Git parent, hashes of the complete producer/source snapshot, and hashes of every JSON and figure artifact. F1 is identified by its serialized 0.9.0 version; missing legacy runtime labels are not reconstructed. |

## Verification record

- `cargo fmt --all -- --check`: clean.
- `cargo clippy --all-targets --all-features -- -D warnings`: clean.
- `cargo test --workspace`: 152 Rust/Wasm tests passed, zero failed.
- `pytest crates/sharpearena-py/tests -q`: 643 passed, 2 skipped, zero failed.
- `latexmk -pdf -jobname=integrity -interaction=nonstopmode -halt-on-error main.tex`:
  29 pages, zero TeX errors, zero undefined-reference warnings, zero overfull boxes.
- TeX source sweep: no literal-tab-stripped command pattern and no ditto-style
  table cells.
- Bibliography-key audit inherited from the fresh review: every bibliography
  key is cited and every citation key resolves. This is a key-completeness
  check, not a new external metadata audit.

## Remaining named research, not defects in this candidate

The paper still calls for a trained entrant and tape-level analytic recovery
from quantized prices. Those are explicitly future experiments. The canonical
generator remains uncertified on 22/24 panels; the paper conditions its claims
on that failure rather than presenting the environment as market-valid.
