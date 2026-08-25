# Final-integrity addendum — 2026-08-25 post-repair

**Status: PASS for the frozen manuscript/evidence handoff.** This addendum supersedes the PASS verdicts in `final-integrity-2026-08-24.md` and `final-integrity-2026-08-25.md` only for claims affected by the 2026-08-25 mathematical and rigor repairs. It does not supersede their bibliographic or source-code coverage outside that scope.

## Frozen evidence checked

| Surface | Evidence | Result carried into manuscript |
|---|---|---|
| F4 realism | `paper/evidence/f4-realism.json`; `paper/src/make-f4-realism.py` | Canonical panels: Calm 0/8, Hard 0/8, Extreme 1/8; 1/24 pass the three calibrated checks. Fano is conditional-IID-calibrated but exploratory. No Calm preset qualifies: nearest candidate is 7/8 diagnostic and 5/8 confirmation at volatility ratio 1.1497; the 8/8 and 7/8 alternative has ratio 1.28945 and violates the cap. |
| F5 linear, concave and asymmetric search | `paper/evidence/f5-manipulation.json`; `paper/src/make-f5-manipulation.py` | All 45 sampled linear means are negative. The symmetric concave arm has no positive sampled mean. The 135-cell selected no-follower, zero-temporary-impact cell is +21.8e-4, pointwise [17.4, 26.2]e-4 and 135-cell Bonferroni [10.0, 33.6]e-4; its fixed-cell 32-seed confirmation is +26.3e-4 [23.7, 28.9]e-4. |
| F5 extension | `paper/evidence/f5-manipulation.json` key `extended_sweeps`; `paper/sections/manipulation-sweeps-fragment.tex` | The 60-cell extension is conditional on an anchor selected from the 135 initial cells, so reported extension intervals use the global 195-cell family. Counts are 23 positive, 31 negative and 6 crossing; the largest sampled 45:5 cell is +135.0e-4, global-familywise [122.1, 148.0]e-4. |
| F6 endogenous impact | `paper/evidence/f6-adverse-selection.json` key `endogenous`; `paper/src/make-f6-adverse-selection.py` | The six-value impact sweep is exploratory with pointwise t intervals only. The manuscript reports sampled sign brackets rather than an interpolated threshold, and calls the result a paired estimated gap rather than a certification. |
| Seed predictability | `paper/evidence/predictability.json`; `paper/evidence/sealed-seeds.json`; `paper/src/make-predictability.py`; `paper/src/make-sealed-seeds.py` | Public $2^{16}$-band scan: 16/16; simulated sealed scan with withheld salt: 0/16; revealed-salt replay: 16/16. Claims are scoped to that scanner and workflow. |
| Witness | `paper/evidence/witness.json`; `paper/src/make-witness.py` | The witness uses a documented $[0,16)$ / $[10016,10032)$ split and common random numbers. Text calls the latter a gap-band witness subset, not the canonical held-out namespace; crossings are observed brackets, not global thresholds. |

## Manuscript consistency repairs

- `paper/sections/00-abstract.tex` identifies native/WebAssembly determinism rather than falsely saying Python bindings are pinned, expands PSR once, and keeps the 23/24 F4 and 135-cell F5 values aligned with evidence.
- `paper/sections/03-environment.tex`, `05-experiments.tex`, and `arena-witness-fragment.tex` distinguish canonical seed namespaces from the 16-seed gap-band witness/F3 instantiation; report Fano construction and calibrated-versus-exploratory roles; and specify global 195-cell FWER for the extension.
- `concave-fragment.tex`, `positive-control-fragment.tex`, `predictability-fragment.tex`, and `sealed-seeds-fragment.tex` now state the normalized-flow scale caveat, finite-grid/selection scope, figure references, scanner-limited seed conclusion, and the exact derived-seed modulus $2^{64}-1-\texttt{EVAL\_SEED\_BASE}$.
- `07-limitations.tex` removes a stale assertion about a current companion window and specifies the operational conditions a public bounded-band evaluation must meet before it can resist the documented scanner.
- `README.md` now treats linear F5 as a finite-grid consistency check, calls the realism mechanism a diagnostic, and removes revision-relative/current-window language.

## Zero-issue criteria met for this handoff

1. No F4 prose claims a certified Calm configuration; the canonical result is the negative 1/24 outcome.
2. No F5 extension claim uses a 60-cell-only family after reusing a 135-cell-selected anchor; all such claims specify global 195-cell correction.
3. No F6 lambda sign bracket is presented as a calibrated boundary, multiplicity-adjusted sweep result, or universal impact law.
4. No manipulation result is presented as an optimum or theorem outside its stated sampled grid.
5. No sealed-seed result is presented as cryptographic proof or evidence of a deployed live workflow.
6. No table relies on ditto cells for numeric/symbol values; every table cell is explicit.
7. No remaining manuscript claim depends on paid LLM evaluation.

## Deliberate limitations, not integrity failures

- The F6 sweep remains exploratory; a sweep-wide multiplicity-adjusted claim would require a separately specified analysis.
- The realism diagnostic is a finite-panel calibrated instrument, not market validation; F2, F5 and F6 use market-model processes that are not themselves submitted to F4.
- The cited SharpeBench companion version is intentionally not changed here; its version locator must be updated only after the corresponding SharpeBench release.
- This report is a manuscript/evidence audit. It does not assert that full native LaTeX or cross-platform code gates were executed in this WSL handoff.
