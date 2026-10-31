# Evidence-freshness and applicability register

Ticket P00-E. This register maps every table, every figure and every empirical
claim in the inventoried manuscript scope of the two products to the artifact
that produced it, the command and version behind that artifact, the changes that
landed afterwards, and what the claim supports today.

The register maps existing evidence. It authorizes no experiment, and no frozen
evidence file, golden or manuscript number was changed to produce it. A rescore
or a measurement a row cites was written by its own committed producer to a new
file beside the frozen one. Where an artifact could not be located the row
records `unresolved` rather than a guess, and unknown historical versions stay
unknown.

`docs/evidence-register.jsonl` is the machine-readable authority; this file is
generated from it by `scripts/make-evidence-register.py`.
`scripts/check-evidence-register.py` fails on a missing artifact, a claim with no
disposition, a manuscript table or figure with no row, and a status change with
no recorded rationale.

Each product keeps its own register in this form, and SharpeBench's copy of this
generator and of its checker is the source of both. A row marked with an external
repository mirrors a row of the companion product's register, which is the row of
record and which that row names. Its artifact paths are relative to that
repository's root. The checker reports coverage only for what it can read, so it
opens those artifacts only when it is given the companion checkout:
`python scripts/check-evidence-register.py --companion <path>` opens every mirrored
artifact in both directions and fails on a moved digest, a disposition that
differs between the two registers, or a generator or checker that is no longer
byte-identical.

## Changing a row

Edit the JSONL row, then run `python scripts/make-evidence-register.py` to
rewrite this file and `python scripts/check-evidence-register.py` to validate
both. A disposition that changes gets a new `status_history` entry with its own
rationale; repeating the previous entry's text fails the check. Every entry
names its recorder in `recorded_by` (`human` or `model` with an `id`, or
`unknown` when the history does not say) and states in `independent` whether
the recorder was someone other than the author of the change under review. An
unknown recorder is never independent, and the checker prints how many entries
have one. An artifact's
`sha256` is over the file's bytes, and the checker prints the digest it computed
when a recorded one does not match.

## Dispositions

| Disposition | Meaning and action |
|---|---|
| historical-only | Validly records an older experiment. The claim is retained and bound to that version. It does not validate changed current behavior. |
| still-applicable | A documented comparison establishes that the computation, the inputs and the claim remain applicable. The check is recorded, not inferred. |
| needs-rescore | Recorded inputs may suffice to compute a current-method result. Sufficiency is verified first; any output is separately versioned and never overwrites the old one. |
| needs-new-experiment | Changed generation, policy, execution or measurement, or missing inputs, prevent a defensible rescore. A new protocol and budget request is required. |
| unresolved | Evidence is insufficient to choose a disposition. What is missing is recorded and no current validation is implied. |

## Counts

| Disposition | Rows |
|---|---|
| historical-only | 12 |
| still-applicable | 18 |
| needs-rescore | 0 |
| needs-new-experiment | 3 |
| unresolved | 0 |
| **total** | **33** |

## Priority queue

The headline claims whose present applicability is not established, worst first. The queue is bounded at 12 rows so that it stays a working list rather than a second copy of the register; every other claim is tracked by its row alone, and no historical audit is reopened for it.

| Claim | Product | Disposition | What it would take |
|---|---|---|---|
| `SA-predictability` | SharpeArena | needs-new-experiment | P13-R/P14-R: re-run the predictability producer under the corrected pin and report it as a new experiment beside the frozen artifact. |
| `SA-sealed-seeds` | SharpeArena | needs-new-experiment | P13-R/P14-R: re-run the sealed-seed reveal under the full-tape check and under the AP4 commitment ordering, reported as a new measurement. |
| `SA-tab-witness` | SharpeArena | needs-new-experiment | P13-R/P14-R: re-run the rank-eligibility witness under the corrected pin as a declared new experiment. |

## Summary

| Claim | Product | Kind | Source | Disposition | Owner | Follow-up |
|---|---|---|---|---|---|---|
| `SA-tab-f4` | SharpeArena | table | paper/sections/07-validation.tex:18, paper/sections/07-validation.tex:42, paper/sections/07-validation.tex:11 | historical-only | SharpeArena paper/evidence owner | P13-R/P14-R: the F4 positive control, which the manuscript names as available and unrun. |
| `SA-calm-calibration` | SharpeArena | claim | paper/sections/07-validation.tex:45, paper/sections/07-validation.tex:55 | historical-only | SharpeArena paper/evidence owner | none |
| `SA-tab-f5` | SharpeArena | table | paper/sections/07-validation.tex:69, paper/sections/07-validation.tex:104, paper/sections/07-validation.tex:110, paper/sections/07-validation.tex:58 | historical-only | SharpeArena paper/evidence owner | P13-R: re-verify the frozen F5 grid against the traded-bar validation rule, or record that the check was run. |
| `SA-f5-concave` | SharpeArena | claim | paper/sections/07-validation.tex:115, paper/sections/07-validation.tex:130 | historical-only | SharpeArena paper/evidence owner | none |
| `SA-f5-positive-control` | SharpeArena | claim | paper/sections/07-validation.tex:135, paper/sections/07-validation.tex:151 | still-applicable | SharpeArena paper/evidence owner | none |
| `SA-f5-extended-sweeps` | SharpeArena | claim | paper/sections/07-validation.tex:156, paper/sections/07-validation.tex:169 | historical-only | SharpeArena paper/evidence owner | none |
| `SA-tab-f6` | SharpeArena | table | paper/sections/07-validation.tex:179, paper/sections/07-validation.tex:204, paper/sections/07-validation.tex:172 | historical-only | SharpeArena paper/evidence owner | none |
| `SA-f6-endogenous` | SharpeArena | claim | paper/sections/07-validation.tex:207, paper/sections/07-validation.tex:222, paper/sections/07-validation.tex:245, paper/sections/07-validation.tex:270 | historical-only | SharpeArena paper/evidence owner | P15: name the test behind the 1e-12 native cross-check, or record that none exists. |
| `SA-predictability` | SharpeArena | claim | paper/sections/07-validation.tex:273, paper/sections/07-validation.tex:285 | needs-new-experiment | SharpeArena paper/evidence owner | P13-R/P14-R: re-run the predictability producer under the corrected pin and report it as a new experiment beside the frozen artifact. |
| `SA-seed-search` | SharpeArena | claim | paper/sections/07-validation.tex:273 | historical-only | SharpeArena paper/evidence owner | none |
| `SA-sealed-seeds` | SharpeArena | claim | paper/sections/07-validation.tex:290 | needs-new-experiment | SharpeArena paper/evidence owner | P13-R/P14-R: re-run the sealed-seed reveal under the full-tape check and under the AP4 commitment ordering, reported as a new measurement. |
| `SA-tab-f1` | SharpeArena | table | paper/sections/08-findings.tex:22, paper/sections/08-findings.tex:63, paper/sections/08-findings.tex:7 | still-applicable | SharpeArena paper/evidence owner | Owner decision: cite the rescore in the manuscript and print the flat rows as unavailable, and update paper/sections/10-limitations.tex:15. |
| `SA-tab-witness` | SharpeArena | table | paper/sections/08-findings.tex:78, paper/sections/08-findings.tex:99, paper/sections/08-findings.tex:66 | needs-new-experiment | SharpeArena paper/evidence owner | P13-R/P14-R: re-run the rank-eligibility witness under the corrected pin as a declared new experiment. |
| `SA-f2-regret` | SharpeArena | claim | paper/sections/08-findings.tex:102, paper/sections/08-findings.tex:114 | still-applicable | SharpeArena paper/evidence owner | none |
| `SA-tab-f3` | SharpeArena | table | paper/sections/08-findings.tex:124, paper/sections/08-findings.tex:117 | still-applicable | SharpeArena paper/evidence owner | none |
| `SA-tab-f7` | SharpeArena | table | paper/sections/08-findings.tex:158, paper/sections/08-findings.tex:180, paper/sections/08-findings.tex:151 | still-applicable | SharpeArena paper/evidence owner | P15: serialize a runtime version in the artifact so the 0.29.0 attribution does not rest on prose. |
| `SA-tab-f8` | SharpeArena | table | paper/sections/08-findings.tex:192, paper/sections/08-findings.tex:224, paper/sections/08-findings.tex:183 | historical-only | SharpeArena paper/evidence owner | none |
| `SA-tab-throughput` | SharpeArena | table | paper/sections/06-protocol.tex:85, paper/sections/06-protocol.tex:78 | historical-only | SharpeArena paper/evidence owner | none |
| `SA-orderbook-mark` | SharpeArena | claim | paper/sections/03-environment.tex:112 | still-applicable | SharpeArena paper/evidence owner | none |
| `SA-seeded-shuffle` | SharpeArena | claim | paper/sections/03-environment.tex:110 | still-applicable | SharpeArena paper/evidence owner | none |
| `SA-nonempirical-tables` | SharpeArena | table | paper/sections/03-environment.tex:20, paper/sections/03-environment.tex:55, paper/sections/06-protocol.tex:36, paper/sections/09-related.tex:12 | still-applicable | SharpeArena paper/evidence owner | P15: consider a per-cell provenance sheet for the related-work table, as the companion product keeps. |
| `SA-sec-leakfree` | SharpeArena | claim | paper/sections/03-environment.tex:5, paper/sections/00-abstract.tex:2 | still-applicable | SharpeArena paper/evidence owner | none |
| `SA-sec-determinism` | SharpeArena | claim | paper/sections/03-environment.tex:82, paper/sections/04-contract.tex:29 | still-applicable | SharpeArena paper/evidence owner | none |
| `SA-sec-replay` | SharpeArena | claim | paper/sections/03-environment.tex:88, paper/sections/00-abstract.tex:2 | still-applicable | SharpeArena paper/evidence owner | none |
| `SA-sec-wire-contract` | SharpeArena | claim | paper/sections/04-contract.tex:5 | still-applicable | SharpeArena paper/evidence owner | none |
| `SA-sec-transport-fault` | SharpeArena | claim | paper/sections/04-contract.tex:21 | still-applicable | SharpeArena paper/evidence owner | none |
| `SA-sec-agentic-layer` | SharpeArena | claim | paper/sections/05-agentic.tex:5, paper/sections/05-agentic.tex:35, paper/sections/05-agentic.tex:45, paper/sections/05-agentic.tex:53, paper/sections/06-protocol.tex:66 | still-applicable | SharpeArena paper/evidence owner | none |
| `SA-sec-prospective-protocol` | SharpeArena | claim | paper/sections/05-agentic.tex:19, paper/sections/05-agentic.tex:27 | still-applicable | SharpeArena paper/evidence owner | none |
| `SA-forecast-pilot` | SharpeArena | claim | paper/sections/12-reproducibility.tex:9, paper/sections/01-introduction.tex:19, paper/sections/00-abstract.tex:2 | historical-only | SharpeArena paper/evidence owner | none |
| `SA-sec-protocol-seeds` | SharpeArena | claim | paper/sections/06-protocol.tex:15 | still-applicable | SharpeArena paper/evidence owner | P15: state the regret figure as about 13, or name the calculation. |
| `SA-sec-throughput-derived` | SharpeArena | claim | paper/sections/00-abstract.tex:2, paper/sections/06-protocol.tex:99, paper/sections/06-protocol.tex:17, paper/sections/10-limitations.tex:7 | historical-only | SharpeArena paper/evidence owner | none |
| `SA-sec-absences` | SharpeArena | claim | paper/sections/01-introduction.tex:19, paper/sections/10-limitations.tex:25 | still-applicable | SharpeArena paper/evidence owner | none |
| `SA-cited-figures` | SharpeArena | claim | paper/sections/01-introduction.tex:3, paper/sections/02-principles.tex:11, paper/sections/03-environment.tex:7, paper/sections/05-agentic.tex:33, paper/sections/06-protocol.tex:62 | still-applicable | SharpeArena paper/evidence owner | none |

## SharpeArena

### `SA-tab-f4`

**Claim.** Stylized-facts pass rates over eight seeds per tier: the canonical generator passes 1 of 24 seeded panels, with gated pass rates 0.000 Calm, 0.000 Hard and 0.125 Extreme.

**Source.** `paper/sections/07-validation.tex:18`, `paper/sections/07-validation.tex:42`, `paper/sections/07-validation.tex:11`

**Labels.** `tab:f4`, `fig:f4`

**Artifacts.**
- `paper/evidence/f4-realism.json`, sha256 `a929cf0e1cae81d399f7ee81f25dec218193834b8c0942cb3f8b565194f44cda`, digest source: computed from the file on 2026-10-31 at sharpearena 7769ef5, equal to the digest paper/evidence/provenance.json records, tiers.<tier>.mean_facts, per_seed and pass_rate; every table cell matches to displayed precision
- `paper/figures/f4-realism.pdf`, sha256 `33d3a9ef026a84fed48d9248397c1a41a236b0aece8450b7ee9b10aea3e10cd4`, digest source: computed from the file on 2026-10-31 at sharpearena 7769ef5, equal to the digest paper/evidence/provenance.json records

**Producer command.** python paper/src/make-f4-realism.py (paper/sections/A-commands.tex); re-render only with python paper/src/make-figures.py

**Producing commit.** Unknown. paper/sections/A-commands.tex:5 is the paper's single provenance statement: only f1-baselines.json serializes a runtime version (sharpearena 0.9.0), no evidence file serializes a SharpeBench version, and the paper declines to reconstruct one from creation dates. No artifact carries a producing commit; the manifest records only the commit it was generated at.

**Effective configuration.** config = {n_symbols: 4, n_days: 120, seeds: 0 to 7, max_steps: 512, tiers: calm, hard, extreme, vol_clustering: 0.5}. No effective_config readback block.

**Missing provenance.**
- Runtime version, SharpeBench version, producing commit and host platform are all unserialized.
- No effective-configuration readback block, so the requested configuration is recorded and the realized one is not.

**Relevant later changes.**
- Apart from F7 the committed F1 to F8 artifacts were not rerun after effective-configuration readback, SPEC_HASH, canonical pre-hash fixtures, typed FFI errors, strict input decoding, path-containment checks or the prospective forecast contract were added. The manuscript states those additions strengthen present executions without changing the observations of artifacts that were not rerun (paper/sections/10-limitations.tex:13).
- The Python reductions behind the realism, manipulation, adverse-selection, failure, ecology and predictability evidence are off the golden-hash path, so their floating-point reproducibility is the platform's rather than the paper's (paper/sections/10-limitations.tex:11).
- The computation-changing group at paper/sections/10-limitations.tex:15 concerns deflated Sharpes and episode metrics; F4 reads neither.

**Present applicability.** The reported statistics do not pass through the scorer, so the moment repair does not reach them. The manuscript records that the gate has no positive control and that the control is available and has not been run (paper/sections/07-validation.tex:5), which bounds what the 23 of 24 failure establishes.

**Disposition.** `historical-only`. The artifact validly records the run and no later change moves its numbers, but nothing re-establishes it against the current tree: the producer is off the golden-hash path and the artifact was not rerun.

**Owner.** SharpeArena paper/evidence owner

**Follow-up.** P13-R/P14-R: the F4 positive control, which the manuscript names as available and unrun.

**Status history.**

| Date | Disposition | Recorded by | Independent | Rationale |
|---|---|---|---|---|
| 2026-09-23 | historical-only | unknown | no | Initial entry: The artifact validly records the run and no later change moves its numbers, but nothing re-establishes it against the current tree: the producer is off the golden-hash path and the artifact was not rerun. |
| 2026-10-31 | historical-only | unknown | no | Rechecked against SharpeBench #118, #171 and #172, SharpeArena #118 and the foundation branch through 7769ef5: no evidence producer steps a vector environment, so SharpeArena #118 reaches no artifact here. The artifact digests were opened against the sharpearena tree and match. This row moved here from the SharpeBench register, which keeps a mirror of it. |

### `SA-calm-calibration`

**Claim.** A 99-configuration Calm calibration sweep qualifies no cell, so there is no certified Calm preset.

**Source.** `paper/sections/07-validation.tex:45`, `paper/sections/07-validation.tex:55`

**Labels.** `fig:calm-calibration`

**Artifacts.**
- `paper/evidence/f4-realism.json`, sha256 `a929cf0e1cae81d399f7ee81f25dec218193834b8c0942cb3f8b565194f44cda`, digest source: computed from the file on 2026-10-31 at sharpearena 7769ef5, equal to the digest paper/evidence/provenance.json records, key calm_calibration, with 99 cells, n_qualifying 0 and chosen null
- `paper/figures/f4-calm-calibration.pdf`, sha256 `a196cd0593aff98f0ded31cf94ae7375e2e7e86cb7755646aa24acfe78827737`, digest source: computed from the file on 2026-10-31 at sharpearena 7769ef5, equal to the digest paper/evidence/provenance.json records

**Producer command.** python paper/src/make-f4-realism.py, whose same invocation runs the 99-configuration sweep and serializes it under calm_calibration (paper/sections/A-commands.tex)

**Producing commit.** Unknown. paper/sections/A-commands.tex:5 is the paper's single provenance statement: only f1-baselines.json serializes a runtime version (sharpearena 0.9.0), no evidence file serializes a SharpeBench version, and the paper declines to reconstruct one from creation dates. No artifact carries a producing commit; the manifest records only the commit it was generated at.

**Effective configuration.** As SA-tab-f4; the sweep's grid, seed bands and rule are serialized under calm_calibration.

**Missing provenance.**
- As SA-tab-f4.

**Relevant later changes.**
- Apart from F7 the committed F1 to F8 artifacts were not rerun after effective-configuration readback, SPEC_HASH, canonical pre-hash fixtures, typed FFI errors, strict input decoding, path-containment checks or the prospective forecast contract were added. The manuscript states those additions strengthen present executions without changing the observations of artifacts that were not rerun (paper/sections/10-limitations.tex:13).
- The Python reductions behind the realism, manipulation, adverse-selection, failure, ecology and predictability evidence are off the golden-hash path, so their floating-point reproducibility is the platform's rather than the paper's (paper/sections/10-limitations.tex:11).

**Present applicability.** A negative result recorded in the artifact as n_qualifying 0 and chosen null. No later change bears on it.

**Disposition.** `historical-only`. The artifact validly records the sweep and its negative outcome; nothing re-establishes it against the current tree.

**Owner.** SharpeArena paper/evidence owner

**Follow-up.** none

**Status history.**

| Date | Disposition | Recorded by | Independent | Rationale |
|---|---|---|---|---|
| 2026-09-23 | historical-only | unknown | no | Initial entry: The artifact validly records the sweep and its negative outcome; nothing re-establishes it against the current tree. |
| 2026-10-31 | historical-only | unknown | no | Rechecked against SharpeBench #118, #171 and #172, SharpeArena #118 and the foundation branch through 7769ef5: no evidence producer steps a vector environment, so SharpeArena #118 reaches no artifact here. The artifact digests were opened against the sharpearena tree and match. This row moved here from the SharpeBench register, which keeps a mirror of it. |

### `SA-tab-f5`

**Claim.** Manipulation boundary sweeps on three axes find no profitability boundary, and impact-attributable PnL is negative and decreasing in push size.

**Source.** `paper/sections/07-validation.tex:69`, `paper/sections/07-validation.tex:104`, `paper/sections/07-validation.tex:110`, `paper/sections/07-validation.tex:58`

**Labels.** `tab:f5`, `fig:f5`, `fig:f5-size`

**Artifacts.**
- `paper/evidence/f5-manipulation.json`, sha256 `19b08f0cb184aeca94020336153c2bf9f2b8ec6d392ada88cde0da04e8ed29e9`, digest source: computed from the file on 2026-10-31 at sharpearena 7769ef5, equal to the digest paper/evidence/provenance.json records, keys boundaries, size_response, dispersion, ci_convention
- `paper/figures/f5-boundaries.pdf`, sha256 `78d3a1c3a3626febad9edf9aeec19cf9720873769059a8173378a50f00844c04`, digest source: computed from the file on 2026-10-31 at sharpearena 7769ef5, equal to the digest paper/evidence/provenance.json records
- `paper/figures/f5-size-response.pdf`, sha256 `686649a200c0c12613d8cbfd00db2d4175388a1a9eb8f6a7c29d0768280e65b8`, digest source: computed from the file on 2026-10-31 at sharpearena 7769ef5, equal to the digest paper/evidence/provenance.json records

**Producer command.** python paper/src/make-f5-manipulation.py (paper/sections/A-commands.tex)

**Producing commit.** Unknown. paper/sections/A-commands.tex:5 is the paper's single provenance statement: only f1-baselines.json serializes a runtime version (sharpearena 0.9.0), no evidence file serializes a SharpeBench version, and the paper declines to reconstruct one from creation dates. No artifact carries a producing commit; the manifest records only the commit it was generated at.

**Effective configuration.** config = {seeds: 0 to 7, base_params: {n_symbols: 1, n_days: 80, n_followers: 3, kyle_lambda: 0.1, eta: 0.05, push_weight: 0.8, follower_gain: 30.0, impact_exponent: 1.0}, concave_exponents: [0.5, 0.7]}; t-based 95 percent intervals over per-seed impact PnL at df 7.

**Missing provenance.**
- As SA-tab-f4, plus no record of which impact-engine revision produced the artifact.

**Relevant later changes.**
- Release 0.31.0 validates the manipulation round trip against traded bars rather than calendar days, so some previously accepted configurations now raise. The committed config uses n_days 80, which is a validity change rather than a value change, and no re-verification against the frozen grid was found.
- The manipulation probe gained per-follower seat-removal externality fields that the committed F5 artifact does not carry (paper/sections/10-limitations.tex:15, paper/sections/03-environment.tex:130).
- The Python reductions behind the realism, manipulation, adverse-selection, failure, ecology and predictability evidence are off the golden-hash path, so their floating-point reproducibility is the platform's rather than the paper's (paper/sections/10-limitations.tex:11).

**Present applicability.** The reported values do not pass through the scorer. The artifact predates the externality fields, so it is a strict subset of what the current probe records rather than a contradiction of it.

**Disposition.** `historical-only`. The artifact validly records the run; the later changes add fields and tighten configuration validation without being shown to move the frozen values, and no re-verification is recorded.

**Owner.** SharpeArena paper/evidence owner

**Follow-up.** P13-R: re-verify the frozen F5 grid against the traded-bar validation rule, or record that the check was run.

**Status history.**

| Date | Disposition | Recorded by | Independent | Rationale |
|---|---|---|---|---|
| 2026-09-23 | historical-only | unknown | no | Initial entry: The artifact validly records the run; the later changes add fields and tighten configuration validation without being shown to move the frozen values, and no re-verification is recorded. |
| 2026-10-31 | historical-only | unknown | no | Rechecked against SharpeBench #118, #171 and #172, SharpeArena #118 and the foundation branch through 7769ef5: no evidence producer steps a vector environment, so SharpeArena #118 reaches no artifact here. The artifact digests were opened against the sharpearena tree and match. This row moved here from the SharpeBench register, which keeps a mirror of it. |

### `SA-f5-concave`

**Claim.** At the canonical calibration the concave-impact probe attributes negative P&L at both exponents, with seven of 23 stored sweep slots crossing zero at beta 0.5 and none at beta 0.7.

**Source.** `paper/sections/07-validation.tex:115`, `paper/sections/07-validation.tex:130`

**Labels.** `fig:f5-concave`

**Artifacts.**
- `paper/evidence/f5-manipulation.json`, sha256 `19b08f0cb184aeca94020336153c2bf9f2b8ec6d392ada88cde0da04e8ed29e9`, digest source: computed from the file on 2026-10-31 at sharpearena 7769ef5, equal to the digest paper/evidence/provenance.json records, key concave, one sub-key per exponent
- `paper/figures/f5-concave.pdf`, sha256 `d37f814664a5e5acddc09c01090100eb981b461bbeae1c25ae258d42832a4f6e`, digest source: computed from the file on 2026-10-31 at sharpearena 7769ef5, equal to the digest paper/evidence/provenance.json records

**Producer command.** python paper/src/make-f5-manipulation.py, which serializes the concave leg under concave (paper/sections/A-commands.tex)

**Producing commit.** Unknown. paper/sections/A-commands.tex:5 is the paper's single provenance statement: only f1-baselines.json serializes a runtime version (sharpearena 0.9.0), no evidence file serializes a SharpeBench version, and the paper declines to reconstruct one from creation dates. No artifact carries a producing commit; the manifest records only the commit it was generated at.

**Effective configuration.** As SA-tab-f5, exponents 0.5 and 0.7.

**Missing provenance.**
- As SA-tab-f5.

**Relevant later changes.**
- The Python reductions behind the realism, manipulation, adverse-selection, failure, ecology and predictability evidence are off the golden-hash path, so their floating-point reproducibility is the platform's rather than the paper's (paper/sections/10-limitations.tex:11).

**Present applicability.** Bounded by the manuscript's own stated limitation that the dimensionless flow sits two to three orders of magnitude below the unit crossover, so exponents below one amplify impact at the tested calibration (paper/sections/10-limitations.tex:19).

**Disposition.** `historical-only`. The artifact validly records the probe and the manuscript already scopes it as a stress setting rather than an estimate.

**Owner.** SharpeArena paper/evidence owner

**Follow-up.** none

**Status history.**

| Date | Disposition | Recorded by | Independent | Rationale |
|---|---|---|---|---|
| 2026-09-23 | historical-only | unknown | no | Initial entry: The artifact validly records the probe and the manuscript already scopes it as a stress setting rather than an estimate. |
| 2026-10-31 | historical-only | unknown | no | Rechecked against SharpeBench #118, #171 and #172, SharpeArena #118 and the foundation branch through 7769ef5: no evidence producer steps a vector environment, so SharpeArena #118 reaches no artifact here. The artifact digests were opened against the sharpearena tree and match. This row moved here from the SharpeBench register, which keeps a mirror of it. |

### `SA-f5-positive-control`

**Claim.** An asymmetric-schedule positive control over 135 cells finds four familywise-positive cells, the best at +21.8e-4 with familywise interval [10.0, 33.6]e-4, and the selected cell replicates on 32 fresh seeds at +26.3e-4 [23.7, 28.9]e-4.

**Source.** `paper/sections/07-validation.tex:135`, `paper/sections/07-validation.tex:151`

**Labels.** `fig:f5-positive-control`

**Artifacts.**
- `paper/evidence/f5-manipulation.json`, sha256 `19b08f0cb184aeca94020336153c2bf9f2b8ec6d392ada88cde0da04e8ed29e9`, digest source: computed from the file on 2026-10-31 at sharpearena 7769ef5, equal to the digest paper/evidence/provenance.json records, key positive_control, including the confirmation leg
- `paper/figures/f5-positive-control.pdf`, sha256 `983d1183c1776dee7ebc4e5aec5399a29cc12fc8a309e77eec62825ed75ca10c`, digest source: computed from the file on 2026-10-31 at sharpearena 7769ef5, equal to the digest paper/evidence/provenance.json records

**Producer command.** python paper/src/make-f5-manipulation.py, positive-control leg (paper/sections/A-commands.tex)

**Producing commit.** Unknown. paper/sections/A-commands.tex:5 is the paper's single provenance statement: only f1-baselines.json serializes a runtime version (sharpearena 0.9.0), no evidence file serializes a SharpeBench version, and the paper declines to reconstruct one from creation dates. No artifact carries a producing commit; the manifest records only the commit it was generated at.

**Effective configuration.** 135 cells as five duration ratios by three block fractions by three exponents by three arms, eight seeds each; fresh-band confirmation on 32 seeds at df 31.

**Missing provenance.**
- As SA-tab-f5.

**Relevant later changes.**
- The Python reductions behind the realism, manipulation, adverse-selection, failure, ecology and predictability evidence are off the golden-hash path, so their floating-point reproducibility is the platform's rather than the paper's (paper/sections/10-limitations.tex:11).

**Present applicability.** Independently recomputed during the August 2026 review round, which reproduced the headline cell, its intervals, the four familywise-positive cells, all 45 negative linear cells and the 32-seed confirmation (paper/review/rereview-2026-08-25.md).

**Disposition.** `still-applicable`. A documented recomputation exists in the review record, which is a comparison rather than an inference; the quantities do not pass through the scorer that later changed.

**Owner.** SharpeArena paper/evidence owner

**Follow-up.** none

**Status history.**

| Date | Disposition | Recorded by | Independent | Rationale |
|---|---|---|---|---|
| 2026-09-23 | still-applicable | unknown | no | Initial entry: A documented recomputation exists in the review record, which is a comparison rather than an inference; the quantities do not pass through the scorer that later changed. |
| 2026-10-31 | still-applicable | unknown | no | Rechecked against SharpeBench #118, #171 and #172, SharpeArena #118 and the foundation branch through 7769ef5: no evidence producer steps a vector environment, so SharpeArena #118 reaches no artifact here. The artifact digests were opened against the sharpearena tree and match. This row moved here from the SharpeBench register, which keeps a mirror of it. |

### `SA-f5-extended-sweeps`

**Claim.** Sixty extension cells give 23 global-familywise positive intervals, 31 negative and six crossing zero, the largest cell at +135.0e-4 with interval [122.1, 148.0]e-4, read as descriptive rather than as simultaneous coverage.

**Source.** `paper/sections/07-validation.tex:156`, `paper/sections/07-validation.tex:169`

**Labels.** `fig:f5-extended-sweeps`

**Artifacts.**
- `paper/evidence/f5-manipulation.json`, sha256 `19b08f0cb184aeca94020336153c2bf9f2b8ec6d392ada88cde0da04e8ed29e9`, digest source: computed from the file on 2026-10-31 at sharpearena 7769ef5, equal to the digest paper/evidence/provenance.json records, key extended_sweeps
- `paper/figures/f5-extended-sweeps.pdf`, sha256 `91d3b7113c9d54220167c130fb8ed13f093e261225a7ccdea981bd7febbaf260`, digest source: computed from the file on 2026-10-31 at sharpearena 7769ef5, equal to the digest paper/evidence/provenance.json records

**Producer command.** python paper/src/make-f5-manipulation.py, extended-sweep leg (paper/sections/A-commands.tex)

**Producing commit.** Unknown. paper/sections/A-commands.tex:5 is the paper's single provenance statement: only f1-baselines.json serializes a runtime version (sharpearena 0.9.0), no evidence file serializes a SharpeBench version, and the paper declines to reconstruct one from creation dates. No artifact carries a producing commit; the manifest records only the commit it was generated at.

**Effective configuration.** Six axes by five values by eight common seeds, corrected over the global 195-cell family.

**Missing provenance.**
- As SA-tab-f5.

**Relevant later changes.**
- The Python reductions behind the realism, manipulation, adverse-selection, failure, ecology and predictability evidence are off the golden-hash path, so their floating-point reproducibility is the platform's rather than the paper's (paper/sections/10-limitations.tex:11).

**Present applicability.** The manuscript states the familywise correction is descriptive rather than simultaneous, which bounds the claim independently of freshness.

**Disposition.** `historical-only`. The artifact validly records the sweep; no documented recomputation covers this leg, unlike the positive control.

**Owner.** SharpeArena paper/evidence owner

**Follow-up.** none

**Status history.**

| Date | Disposition | Recorded by | Independent | Rationale |
|---|---|---|---|---|
| 2026-09-23 | historical-only | unknown | no | Initial entry: The artifact validly records the sweep; no documented recomputation covers this leg, unlike the positive control. |
| 2026-10-31 | historical-only | unknown | no | Rechecked against SharpeBench #118, #171 and #172, SharpeArena #118 and the foundation branch through 7769ef5: no evidence producer steps a vector environment, so SharpeArena #118 reaches no artifact here. The artifact digests were opened against the sharpearena tree and match. This row moved here from the SharpeBench register, which keeps a mirror of it. |

### `SA-tab-f6`

**Claim.** Maker markout per filled unit over 24 paired episodes: the informed-uninformed gap widens from 0.072 at h = 1 to 0.436 at h = 20, with t-based 95 percent intervals at df 23.

**Source.** `paper/sections/07-validation.tex:179`, `paper/sections/07-validation.tex:204`, `paper/sections/07-validation.tex:172`

**Labels.** `tab:f6`, `fig:f6`

**Artifacts.**
- `paper/evidence/f6-adverse-selection.json`, sha256 `2e2e9199803693f6a15884e7461ed09d0a840ff36dc0405b5ac4e12833ee7e2b`, digest source: computed from the file on 2026-10-31 at sharpearena 7769ef5, equal to the digest paper/evidence/provenance.json records, keys comparison, detail_episode_makers, per_episode_markout_per_unit, gap_stats, ci_convention
- `paper/figures/f6-markouts.pdf`, sha256 `89d9fb5af7b4cd2e80cf0e87b4717d0b946a7879ba9837b5d7e270bd43751dfe`, digest source: computed from the file on 2026-10-31 at sharpearena 7769ef5, equal to the digest paper/evidence/provenance.json records

**Producer command.** python paper/src/make-f6-adverse-selection.py (paper/sections/A-commands.tex)

**Producing commit.** Unknown. paper/sections/A-commands.tex:5 is the paper's single provenance statement: only f1-baselines.json serializes a runtime version (sharpearena 0.9.0), no evidence file serializes a SharpeBench version, and the paper declines to reconstruct one from creation dates. No artifact carries a producing commit; the manifest records only the commit it was generated at.

**Effective configuration.** config = {n_episodes: 24, seed_base: 0, detail_seed: 0}. This is the thinnest config block of the eight findings: it does not record the maker roster, the alpha or the quote depths that the interpreting prose says drive the level result.

**Missing provenance.**
- As SA-tab-f4.
- The config block omits the maker roster, alpha and quote depths the level claim rests on.

**Relevant later changes.**
- Audit repair AP3 (docs/audits/2026-09-07/IMPLEMENTATION.md): F6's gate between two current code paths was reported as agreement with the committed vectors; the gate now carries its real name and a separate frozen-reference comparison reads the committed artifact before it is overwritten, reporting rather than raising.
- The endogenous arm and its key postdate the original run and share one artifact and one code path with the exogenous arm, pinned by a golden hash asserting the exogenous reports do not move when the endogenous arm runs (paper/sections/07-validation.tex:207).
- The Python reductions behind the realism, manipulation, adverse-selection, failure, ecology and predictability evidence are off the golden-hash path, so their floating-point reproducibility is the platform's rather than the paper's (paper/sections/10-limitations.tex:11).

**Present applicability.** The exogenous reports are pinned against movement when the endogenous arm runs, and AP3 replaced an assumed parity gate with a reported comparison. Neither establishes that the frozen values reproduce on the current tree.

**Disposition.** `historical-only`. The artifact validly records the run; the repairs named change how parity is reported rather than establishing that the frozen numbers still reproduce.

**Owner.** SharpeArena paper/evidence owner

**Follow-up.** none

**Status history.**

| Date | Disposition | Recorded by | Independent | Rationale |
|---|---|---|---|---|
| 2026-09-23 | historical-only | unknown | no | Initial entry: The artifact validly records the run; the repairs named change how parity is reported rather than establishing that the frozen numbers still reproduce. |
| 2026-10-31 | historical-only | unknown | no | Rechecked against SharpeBench #118, #171 and #172, SharpeArena #118 and the foundation branch through 7769ef5: no evidence producer steps a vector environment, so SharpeArena #118 reaches no artifact here. The artifact digests were opened against the sharpearena tree and match. This row moved here from the SharpeBench register, which keeps a mirror of it. |

### `SA-f6-endogenous`

**Claim.** The endogenous arm's calibration arithmetic predicts a 0.933 percent displacement against a measured 0.81 percent [0.77, 0.85], and the six-value lambda sweep crosses sign in the informed level between lambda 0.2 and 0.3.

**Source.** `paper/sections/07-validation.tex:207`, `paper/sections/07-validation.tex:222`, `paper/sections/07-validation.tex:245`, `paper/sections/07-validation.tex:270`

**Labels.** `tab:f6-endo`, `tab:f6-endo-sweep`, `fig:f6-endo`

**Artifacts.**
- `paper/evidence/f6-adverse-selection.json`, sha256 `2e2e9199803693f6a15884e7461ed09d0a840ff36dc0405b5ac4e12833ee7e2b`, digest source: computed from the file on 2026-10-31 at sharpearena 7769ef5, equal to the digest paper/evidence/provenance.json records, key endogenous, ten sub-keys
- `paper/figures/f6-endogenous.pdf`, sha256 `cce8ff8cb77299935d2df45f2967848ad8fe4fbb0e5ce81abb7e1ac1ff838cb7`, digest source: computed from the file on 2026-10-31 at sharpearena 7769ef5, equal to the digest paper/evidence/provenance.json records

**Producer command.** python paper/src/make-f6-adverse-selection.py, whose same invocation runs the endogenous arm and its six-value impact sweep (paper/sections/A-commands.tex)

**Producing commit.** Unknown. paper/sections/A-commands.tex:5 is the paper's single provenance statement: only f1-baselines.json serializes a runtime version (sharpearena 0.9.0), no evidence file serializes a SharpeBench version, and the paper declines to reconstruct one from creation dates. No artifact carries a producing commit; the manifest records only the commit it was generated at.

**Effective configuration.** As SA-tab-f6; the sweep is declared exploratory with no sweep-wide multiplicity claim.

**Missing provenance.**
- As SA-tab-f6.
- The native clearing-engine cross-check quoted at relative 1e-12 names no test, and no test reproducing it was located.

**Relevant later changes.**
- As SA-tab-f6.

**Present applicability.** Recorded in the same artifact as the exogenous arm and subject to the same unestablished reproducibility. The sweep is declared exploratory in the manuscript.

**Disposition.** `historical-only`. The artifact validly records the arm; the one cross-check that would strengthen it names no test that could be located.

**Owner.** SharpeArena paper/evidence owner

**Follow-up.** P15: name the test behind the 1e-12 native cross-check, or record that none exists.

**Status history.**

| Date | Disposition | Recorded by | Independent | Rationale |
|---|---|---|---|---|
| 2026-09-23 | historical-only | unknown | no | Initial entry: The artifact validly records the arm; the one cross-check that would strengthen it names no test that could be located. |
| 2026-10-31 | historical-only | unknown | no | Rechecked against SharpeBench #118, #171 and #172, SharpeArena #118 and the foundation branch through 7769ef5: no evidence producer steps a vector environment, so SharpeArena #118 reaches no artifact here. The artifact digests were opened against the sharpearena tree and match. This row moved here from the SharpeBench register, which keeps a mirror of it. |

### `SA-predictability`

**Claim.** An honest ridge AR(5) adversary reaches 55.4 percent directional accuracy on Calm against a 54.1 percent baseline, with a paired t over 16 Calm seeds of t(15) = 3.36, p = 0.004, and the known-seed oracle scores 100 percent accuracy and a deflated Sharpe of 1.00 on every tier.

**Source.** `paper/sections/07-validation.tex:273`, `paper/sections/07-validation.tex:285`

**Labels.** `fig:predictability`

**Artifacts.**
- `paper/evidence/predictability.json`, sha256 `bf4c2b3bb4c6311a0795a7b0b65a2fe00b76a06193f3051de0a1f63391782a8a`, digest source: computed from the file on 2026-10-31 at sharpearena 7769ef5, equal to the digest paper/evidence/provenance.json records, keys config, seed_search, tiers; every per-seed score and the measured search timing
- `paper/figures/predictability.pdf`, sha256 `75655375df8c1cd28c8af1f8dc5c205292a64aa8a5b4ccd24fb3063dc3786e9d`, digest source: computed from the file on 2026-10-31 at sharpearena 7769ef5, equal to the digest paper/evidence/provenance.json records

**Producer command.** python paper/src/make-predictability.py (paper/sections/A-commands.tex)

**Producing commit.** Unknown. paper/sections/A-commands.tex:5 is the paper's single provenance statement: only f1-baselines.json serializes a runtime version (sharpearena 0.9.0), no evidence file serializes a SharpeBench version, and the paper declines to reconstruct one from creation dates. No artifact carries a producing commit; the manifest records only the commit it was generated at.

**Effective configuration.** config = {tiers: calm, hard, extreme, n_symbols: 4, n_days: 120, seeds: 50000 to 50015, warmup_bars: 30, ar_order: 5, ridge_lambda: 1e-4, search_band_width: 65536, match_tolerance: 1e-9, policy: frictionless unit-gross long/short sign following scored via score_run deflated_sharpe}.

**Missing provenance.**
- The SharpeBench kernel version behind the deflated Sharpes is unserialized, and the limitations name this artifact among those carrying superseded moments.
- The artifact records per-seed scores rather than per-seed return series, so the deflated Sharpes cannot be recomputed from it under the corrected estimator.

**Relevant later changes.**
- Standardized-moment estimators mixed finite-sample normalizations, which moves deflated Sharpes and their intervals. SharpeBench corrected its estimators in 0.19.0 and SharpeArena its own confidence estimator in 0.25.0; the frozen F1, F3, witness and predictability artifacts predate both corrections and have not been rescored (paper/sections/10-limitations.tex:15).
- Audit repair AP1 (docs/audits/2026-09-07/IMPLEMENTATION.md): the oracle was scored on the full tape while the causal predictors started at the warmup bar; the oracle is now masked to the same window, and replaying the frozen tapes shows the published gap unaffected past the sixth decimal.
- The Python reductions behind the realism, manipulation, adverse-selection, failure, ecology and predictability evidence are off the golden-hash path, so their floating-point reproducibility is the platform's rather than the paper's (paper/sections/10-limitations.tex:11).

**Present applicability.** The accuracy and mean-squared-error figures do not pass through the scorer and are unaffected by the moment repair. The deflated Sharpes do, and are named in the limitations as carrying superseded moments and not rescored. AP1's effect is the one part with a documented replay comparison.

**Disposition.** `needs-new-experiment`. The deflated-Sharpe leg cannot be rescored from the artifact, which stores scores rather than return series, so recomputing it means re-running the producer. The producer is deterministic in its seeds, which makes the run cheap, but it is a new execution rather than a correction. P00-E maps evidence and authorizes no rerun. The follow-up names the ticket that would own one.

**Owner.** SharpeArena paper/evidence owner

**Follow-up.** P13-R/P14-R: re-run the predictability producer under the corrected pin and report it as a new experiment beside the frozen artifact.

**Status history.**

| Date | Disposition | Recorded by | Independent | Rationale |
|---|---|---|---|---|
| 2026-09-23 | needs-new-experiment | unknown | no | Initial entry: The deflated-Sharpe leg cannot be rescored from the artifact, which stores scores rather than return series, so recomputing it means re-running the producer. The producer is deterministic in its seeds, which makes the run cheap, but it is a new execution rather than a correction. P00-E maps evidence and authorizes no rerun. The follow-up names the ticket that would own one. |
| 2026-10-31 | needs-new-experiment | unknown | no | Rechecked against SharpeBench #118, #171 and #172, SharpeArena #118 and the foundation branch through 7769ef5: no evidence producer steps a vector environment, so SharpeArena #118 reaches no artifact here. The artifact digests were opened against the sharpearena tree and match. This row moved here from the SharpeBench register, which keeps a mirror of it. |

### `SA-seed-search`

**Claim.** A brute scan over a 2^16 band recovers the true seed in 16 of 16 trials with zero collisions from one observed bar at about 14.7 microseconds per candidate, extrapolating to roughly 8.6 million CPU-years at 2^64.

**Source.** `paper/sections/07-validation.tex:273`

**Artifacts.**
- `paper/evidence/predictability.json`, sha256 `bf4c2b3bb4c6311a0795a7b0b65a2fe00b76a06193f3051de0a1f63391782a8a`, digest source: computed from the file on 2026-10-31 at sharpearena 7769ef5, equal to the digest paper/evidence/provenance.json records, key seed_search

**Producer command.** python paper/src/make-predictability.py (paper/sections/A-commands.tex)

**Producing commit.** Unknown. paper/sections/A-commands.tex:5 is the paper's single provenance statement: only f1-baselines.json serializes a runtime version (sharpearena 0.9.0), no evidence file serializes a SharpeBench version, and the paper declines to reconstruct one from creation dates. No artifact carries a producing commit; the manifest records only the commit it was generated at.

**Effective configuration.** Scan band width 65536, match tolerance 1e-9, 16 trials.

**Missing provenance.**
- The artifact records no host block, unlike throughput.json, so the per-candidate timing and the extrapolation carry no machine identity.

**Relevant later changes.**
- The Python reductions behind the realism, manipulation, adverse-selection, failure, ecology and predictability evidence are off the golden-hash path, so their floating-point reproducibility is the platform's rather than the paper's (paper/sections/10-limitations.tex:11).

**Present applicability.** The recovery counts are deterministic properties of the recorded run. The timing and the CPU-year extrapolation are hardware-dependent with no recorded host, and the manuscript labels the extrapolation harness-specific.

**Disposition.** `historical-only`. The artifact validly records the scan; the timing leg is bound to an unrecorded machine, which the missing-provenance field states rather than the disposition hiding.

**Owner.** SharpeArena paper/evidence owner

**Follow-up.** none

**Status history.**

| Date | Disposition | Recorded by | Independent | Rationale |
|---|---|---|---|---|
| 2026-09-23 | historical-only | unknown | no | Initial entry: The artifact validly records the scan; the timing leg is bound to an unrecorded machine, which the missing-provenance field states rather than the disposition hiding. |
| 2026-10-31 | historical-only | unknown | no | Rechecked against SharpeBench #118, #171 and #172, SharpeArena #118 and the foundation branch through 7769ef5: no evidence producer steps a vector environment, so SharpeArena #118 reaches no artifact here. The artifact digests were opened against the sharpearena tree and match. This row moved here from the SharpeBench register, which keeps a mirror of it. |

### `SA-sealed-seeds`

**Claim.** On 16 tested slots the public-band scanner recovers 16 of 16 public seeds and 0 of 16 sealed seeds while the salt is withheld, and revealing the salt replays 16 of 16.

**Source.** `paper/sections/07-validation.tex:290`

**Artifacts.**
- `paper/evidence/sealed-seeds.json`, sha256 `a111ae0114f88f83a128970f60cebd00200d32540328137de927fa7ed6c8851f`, digest source: computed from the file on 2026-10-31 at sharpearena 7769ef5, equal to the digest paper/evidence/provenance.json records, keys config, table_build_s, public, sealed, note; every per-trial recovery and replay record

**Producer command.** python paper/src/make-sealed-seeds.py (paper/sections/A-commands.tex)

**Producing commit.** Unknown. paper/sections/A-commands.tex:5 is the paper's single provenance statement: only f1-baselines.json serializes a runtime version (sharpearena 0.9.0), no evidence file serializes a SharpeBench version, and the paper declines to reconstruct one from creation dates. No artifact carries a producing commit; the manifest records only the commit it was generated at.

**Effective configuration.** 16 slots, verification tier hard, 120 days, scan band [1000000, 1065536], match tolerance 1e-9.

**Missing provenance.**
- Runtime version unserialized.
- The separate salt-commitment artifact that audit repair AP4 introduced does not exist beside this file, so the frozen record predates the externally checkable commitment separation.
- The manuscript names the AP5 reveal-check repair but not the AP4 commitment-ordering repair, which is a disclosure gap rather than a resolved question.

**Relevant later changes.**
- Computation-changing repair AP5 (paper/sections/10-limitations.tex:15): the committed record's count of sixteen slots replayed was measured by a check that compared only each symbol's opening close in a two-day calm environment, while the declared evaluation is 120 days at the hard tier. The check now regenerates the complete tape and requires equality on every bar and symbol, so the frozen number is not evidence for the stronger property the reveal now tests.
- Audit repair AP4: the salt commitment is now written to its own artifact before the attacks run.

**Present applicability.** The manuscript states outright that the frozen count is not evidence for the property the current check tests. The replay claim as a reader would now read it is therefore unsupported by this artifact.

**Disposition.** `needs-new-experiment`. The measurement instrument changed, not just the scorer, and the artifact records the old check's outcome rather than inputs a new check could be run over. The producer is deterministic, so the new run is cheap, but it measures something the frozen record did not. P00-E maps evidence and authorizes no rerun. The follow-up names the ticket that would own one.

**Owner.** SharpeArena paper/evidence owner

**Follow-up.** P13-R/P14-R: re-run the sealed-seed reveal under the full-tape check and under the AP4 commitment ordering, reported as a new measurement.

**Status history.**

| Date | Disposition | Recorded by | Independent | Rationale |
|---|---|---|---|---|
| 2026-09-23 | needs-new-experiment | unknown | no | Initial entry: The measurement instrument changed, not just the scorer, and the artifact records the old check's outcome rather than inputs a new check could be run over. The producer is deterministic, so the new run is cheap, but it measures something the frozen record did not. P00-E maps evidence and authorizes no rerun. The follow-up names the ticket that would own one. |
| 2026-10-31 | needs-new-experiment | unknown | no | Rechecked against SharpeBench #118, #171 and #172, SharpeArena #118 and the foundation branch through 7769ef5: no evidence producer steps a vector environment, so SharpeArena #118 reaches no artifact here. The artifact digests were opened against the sharpearena tree and match. This row moved here from the SharpeBench register, which keeps a mirror of it. |

### `SA-tab-f1`

**Claim.** Eighteen baseline rows over six policies and three tiers: no policy is rank-eligible on any tier, because eligibility requires a pass^k rate of exactly 1.00 and the best rate is 0.75.

**Source.** `paper/sections/08-findings.tex:22`, `paper/sections/08-findings.tex:63`, `paper/sections/08-findings.tex:7`

**Labels.** `tab:f1`, `fig:f1`

**Artifacts.**
- `paper/evidence/f1-baselines.json`, sha256 `ec61a9a1f4340d58d497545ce401c159bd1e61201cb7c001773dd66b9e381f67`, digest source: computed from the file on 2026-10-31 at sharpearena 7769ef5, equal to the digest paper/evidence/provenance.json records, the only evidence file carrying a runtime version, package_version 0.9.0; rows carry deflated_sharpe, its bootstrap interval, passed_k_rate, mean_return and per_seed_returns
- `paper/figures/f1-baselines.pdf`, sha256 `07e253fa1d96cdef8b30aff1c1e4cebe7ef673c296ca2fa7b7e3cbad8406b624`, digest source: computed from the file on 2026-10-31 at sharpearena 7769ef5, equal to the digest paper/evidence/provenance.json records
- `paper/evidence/f1-baselines.rescore-v1-sb0.28.0.json`, sha256 `33c9a3cfacba0a8deaed94c32569e98859ad3907775d18b66afb7ed68d07882c`, digest source: computed from the file on 2026-10-31; paper/evidence/f1-baselines.rescore-v1-sb0.28.0.json is new in this round, rescore under SharpeBench 0.28.0, sharpearena 0.31.0, SPEC_HASH 0b426f58a65094c9

**Producer command.** python paper/src/make-f1-baselines.py (paper/sections/A-commands.tex) A current-method result is written by python paper/src/make-rescore-f1-f3.py, which reads the frozen file and steps no environment.

**Producing commit.** sharpearena 0.9.0 is serialized in the artifact. The SharpeBench version is not; paper/sections/A-commands.tex:5 records that the commits adding the artifact lock SharpeBench 0.5.0. The producing commit is unknown.

**Effective configuration.** config = {n_symbols: 4, n_days: 120, seeds: 0 to 15, tiers: calm, hard, extreme, bootstrap: {n_boot: 2000, resample_seed: 1537679398 (0x5BA72026), alpha: 0.05, seed-paired percentile bootstrap on the deflated Sharpe}}.

**Missing provenance.**
- Producing commit, SharpeBench kernel version, effective-configuration readback and host platform.
- paper/sections/08-findings.tex:55 records that the dispersion-source claim holds by the producer path rather than by readback, because the committed artifacts do not serialize which dispersion source the scorer took.

**Relevant later changes.**
- Standardized-moment estimators mixed finite-sample normalizations, which moves deflated Sharpes and their intervals. SharpeBench corrected its estimators in 0.19.0 and SharpeArena its own confidence estimator in 0.25.0; the frozen F1, F3, witness and predictability artifacts predate both corrections and have not been rescored (paper/sections/10-limitations.tex:15).
- A repair after release 0.29.0 withholds the flat policy's values at SharpeArena's own boundary, and the pinned scorer is now SharpeBench 0.28.0, which refuses a constant track itself. Under the repaired code the flat rows would report a deflation error instead of a deflated Sharpe, a pass^k rate and an interval; the frozen table still prints 0.0007 (paper/sections/08-findings.tex:55).
- Apart from F7 the committed F1 to F8 artifacts were not rerun after effective-configuration readback, SPEC_HASH, canonical pre-hash fixtures, typed FFI errors, strict input decoding, path-containment checks or the prospective forecast contract were added. The manuscript states those additions strengthen present executions without changing the observations of artifacts that were not rerun (paper/sections/10-limitations.tex:13).
- The Python reductions behind the realism, manipulation, adverse-selection, failure, ecology and predictability evidence are off the golden-hash path, so their floating-point reproducibility is the platform's rather than the paper's (paper/sections/10-limitations.tex:11).
- The rescore of 2026-10-31 recomputed every row from per_seed_returns. paper/sections/10-limitations.tex:15 still says these artifacts have not been rescored.

**Present applicability.** Rescored under the pinned 0.28.0 kernel. The largest movement in a scored deflated Sharpe is 7.9e-7, no pass^k rate moves, the kernel reproduces every interval it attaches, and no printed cell of the table changes outside the flat rows. The flat rows report the kernel's typed refusal on all three tiers, so the printed 0.00 is not what the current code emits. The claim holds: no policy is rank-eligible and the best rate is 0.75.

**Disposition.** `still-applicable`. A documented comparison now exists: the rescore stores each frozen value beside its current one. Sufficiency was checked rather than assumed, since every recorded series reproduces its cell's mean return exactly and the worked instance reads 0.1114 at six trials.

**Owner.** SharpeArena paper/evidence owner

**Follow-up.** Owner decision: cite the rescore in the manuscript and print the flat rows as unavailable, and update paper/sections/10-limitations.tex:15.

**Status history.**

| Date | Disposition | Recorded by | Independent | Rationale |
|---|---|---|---|---|
| 2026-09-23 | needs-rescore | unknown | no | Initial entry: The artifact serializes per_seed_returns, so the deflated Sharpes and their intervals are recomputable under the corrected estimator without re-running the environment. Sufficiency is evidenced by the manuscript's own worked instance, which rescores the committed F3 return series at six trials and reproduces 0.1114 exactly. Any rescore is separately versioned and does not overwrite the frozen artifact. P00-E maps evidence and authorizes no rerun. The follow-up names the ticket that would own one. |
| 2026-10-31 | still-applicable | unknown | no | Rescored from the committed per_seed_returns under SharpeBench 0.28.0. Moves to still-applicable: scored values change by at most 7.9e-7 and the flat rows are typed unavailable, which the manuscript already states they would be. |

### `SA-tab-witness`

**Claim.** The eligibility set is nonempty: observed crossings in nominal signal strength over five noise paths, and at every one of the 50 attained crossings the last failing gate is the per-run reliability gate.

**Source.** `paper/sections/08-findings.tex:78`, `paper/sections/08-findings.tex:99`, `paper/sections/08-findings.tex:66`

**Labels.** `tab:witness`, `fig:witness`

**Artifacts.**
- `paper/evidence/witness.json`, sha256 `bc80f9f93627cab952b2edf61d93d50a5de86c601aef5b17feab2eb18f16578b`, digest source: computed from the file on 2026-10-31 at sharpearena 7769ef5, equal to the digest paper/evidence/provenance.json records, keys finding, oracle_disclosure, config, results, boundaries, noise_replicates
- `paper/figures/witness.pdf`, sha256 `cb050f83fa5c7a5cf7dc1092dfe60567baaa42fd680cd26fd3ca3919d6d831e5`, digest source: computed from the file on 2026-10-31 at sharpearena 7769ef5, equal to the digest paper/evidence/provenance.json records

**Producer command.** python paper/src/make-witness.py (paper/sections/A-commands.tex)

**Producing commit.** Unknown. paper/sections/A-commands.tex:5 is the paper's single provenance statement: only f1-baselines.json serializes a runtime version (sharpearena 0.9.0), no evidence file serializes a SharpeBench version, and the paper declines to reconstruct one from creation dates. No artifact carries a producing commit; the manifest records only the commit it was generated at.

**Effective configuration.** config includes bands {held_out: 10016 to 10031, f1_table: 0 to 15}, seed_gap 10000, n_trials 6 on the F1 footprint, 13 strength points, bisect_resolution 0.005, kernel_gates {per_run_psr_bar: 0.9, dsr_bar: 0.95, alpha: 0.05}.

**Missing provenance.**
- SharpeBench version, which matters directly because the result is a kernel-gate crossing.
- The artifact records crossings and boundaries rather than per-seed return series, so the gates cannot be recomputed from it.

**Relevant later changes.**
- Standardized-moment estimators mixed finite-sample normalizations, which moves deflated Sharpes and their intervals. SharpeBench corrected its estimators in 0.19.0 and SharpeArena its own confidence estimator in 0.25.0; the frozen F1, F3, witness and predictability artifacts predate both corrections and have not been rescored (paper/sections/10-limitations.tex:15). The witness is named in that group.
- Apart from F7 the committed F1 to F8 artifacts were not rerun after effective-configuration readback, SPEC_HASH, canonical pre-hash fixtures, typed FFI errors, strict input decoding, path-containment checks or the prospective forecast contract were added. The manuscript states those additions strengthen present executions without changing the observations of artifacts that were not rerun (paper/sections/10-limitations.tex:13).
- The Python reductions behind the realism, manipulation, adverse-selection, failure, ecology and predictability evidence are off the golden-hash path, so their floating-point reproducibility is the platform's rather than the paper's (paper/sections/10-limitations.tex:11).

**Present applicability.** The crossings are gate outcomes computed by a scorer that has since been corrected, and the artifact does not carry the returns a corrected scorer would need. No record was found that this artifact was rerun.

**Disposition.** `needs-new-experiment`. Missing inputs prevent a rescore: the artifact stores crossings, not the trajectories behind them, so establishing the crossing under the corrected gates means re-running the bisection. The producer is deterministic in its seeds. P00-E maps evidence and authorizes no rerun. The follow-up names the ticket that would own one.

**Owner.** SharpeArena paper/evidence owner

**Follow-up.** P13-R/P14-R: re-run the rank-eligibility witness under the corrected pin as a declared new experiment.

**Status history.**

| Date | Disposition | Recorded by | Independent | Rationale |
|---|---|---|---|---|
| 2026-09-23 | needs-new-experiment | unknown | no | Initial entry: Missing inputs prevent a rescore: the artifact stores crossings, not the trajectories behind them, so establishing the crossing under the corrected gates means re-running the bisection. The producer is deterministic in its seeds. P00-E maps evidence and authorizes no rerun. The follow-up names the ticket that would own one. |
| 2026-10-31 | needs-new-experiment | unknown | no | Rechecked against SharpeBench #118, #171 and #172, SharpeArena #118 and the foundation branch through 7769ef5: no evidence producer steps a vector environment, so SharpeArena #118 reaches no artifact here. The artifact digests were opened against the sharpearena tree and match. This row moved here from the SharpeBench register, which keeps a mirror of it. |

### `SA-f2-regret`

**Claim.** Regret against the closed-form reference over 16 paired episodes is U-shaped in the half-spread, 84.6 [68.2, 101.1] at 0.05, minimal at 0.5 and 58.8 [55.3, 62.3] at 4.0, with 16 episodes detecting about 12 reward units at 80 percent power.

**Source.** `paper/sections/08-findings.tex:102`, `paper/sections/08-findings.tex:114`

**Labels.** `fig:f2`

**Artifacts.**
- `paper/evidence/f2-regret.json`, sha256 `7dcf2e656b4f73e8af7e4dba90721bab63f79b8de4a01c682b299b9f693c80ee`, digest source: computed from the file on 2026-10-31 at sharpearena 7769ef5, equal to the digest paper/evidence/provenance.json records, keys config, optimal_regret, fixed_spread_regret, per_episode_regret, regret_dispersion
- `paper/figures/f2-regret.pdf`, sha256 `634c38c28a152ebefbc5acbd8620c5820e89269e3486cceb912d8398dc346be6`, digest source: computed from the file on 2026-10-31 at sharpearena 7769ef5, equal to the digest paper/evidence/provenance.json records

**Producer command.** python paper/src/make-f2-regret.py (paper/sections/A-commands.tex)

**Producing commit.** Unknown. paper/sections/A-commands.tex:5 is the paper's single provenance statement: only f1-baselines.json serializes a runtime version (sharpearena 0.9.0), no evidence file serializes a SharpeBench version, and the paper declines to reconstruct one from creation dates. No artifact carries a producing commit; the manifest records only the commit it was generated at.

**Effective configuration.** config = {params: {sigma: 2.0, gamma: 0.1, kappa: 1.5, arrival_rate: 140.0, n_steps: 200, dt: 0.005, inventory_cap: 50, phi: 0.0015}, n_episodes: 16, seed_base: 0, half_spreads: seven points from 0.05 to 4.0, t-based 95 percent at df 15}.

**Missing provenance.**
- Producing commit, runtime version and host. The manuscript names sigma, gamma, kappa and the step count but not the arrival rate or phi, which the artifact does record.

**Relevant later changes.**
- Release 0.31.0 makes the market-making regret refuse two arms that drew different random streams, and the committed F2 regrets pass that check unchanged (paper/sections/10-limitations.tex:15).

**Present applicability.** Re-established against the current guard: a committed test runs every half-spread of this subsection through the checked function at the default parameters and reproduces the frozen regrets bit for bit (paper/sections/08-findings.tex:108).

**Disposition.** `still-applicable`. This is the one artifact in the manuscript with a recorded comparison against a later guard that reproduces the frozen numbers exactly, which is a check rather than an inference.

**Owner.** SharpeArena paper/evidence owner

**Follow-up.** none

**Status history.**

| Date | Disposition | Recorded by | Independent | Rationale |
|---|---|---|---|---|
| 2026-09-23 | still-applicable | unknown | no | Initial entry: This is the one artifact in the manuscript with a recorded comparison against a later guard that reproduces the frozen numbers exactly, which is a check rather than an inference. |
| 2026-10-31 | still-applicable | unknown | no | Rechecked against SharpeBench #118, #171 and #172, SharpeArena #118 and the foundation branch through 7769ef5: no evidence producer steps a vector environment, so SharpeArena #118 reaches no artifact here. The artifact digests were opened against the sharpearena tree and match. This row moved here from the SharpeBench register, which keeps a mirror of it. |

### `SA-tab-f3`

**Claim.** Within-tier train minus test deflated Sharpe gaps of +0.0012 Calm, -0.3388 Hard and -0.1012 Extreme, and a transfer matrix with Calm to Extreme +0.973.

**Source.** `paper/sections/08-findings.tex:124`, `paper/sections/08-findings.tex:117`

**Labels.** `tab:f3`

**Artifacts.**
- `paper/evidence/f3-generalization.json`, sha256 `3bff445e13dd4d094a45ebe20e713ec0bc63b06e017c30a399d3aa41a8cf50e1`, digest source: computed from the file on 2026-10-31 at sharpearena 7769ef5, equal to the digest paper/evidence/provenance.json records, keys config, generalization_gap, cross_regime_transfer, per_seed_returns, band_dsr_ci, transfer_gap_ci
- `paper/evidence/f3-generalization.rescore-v1-sb0.28.0.json`, sha256 `4cc288d900713ec81fec904793716d54c04672d974ae286da9d097ac23ce41ef`, digest source: computed from the file on 2026-10-31; paper/evidence/f3-generalization.rescore-v1-sb0.28.0.json is new in this round, rescore under SharpeBench 0.28.0, sharpearena 0.31.0, SPEC_HASH 0b426f58a65094c9

**Producer command.** python paper/src/make-f3-generalization.py (paper/sections/A-commands.tex). The appendix records that the transfer matrix is reported as a table only and that the paper carries no F3 figure. A current-method result is written by python paper/src/make-rescore-f1-f3.py, which reads the frozen file and steps no environment.

**Producing commit.** Unknown. paper/sections/A-commands.tex:5 is the paper's single provenance statement: only f1-baselines.json serializes a runtime version (sharpearena 0.9.0), no evidence file serializes a SharpeBench version, and the paper declines to reconstruct one from creation dates. No artifact carries a producing commit; the manifest records only the commit it was generated at.

**Effective configuration.** config = {n_symbols: 4, n_days: 120, n_train: 16, n_test: 16, seed_gap: 10000, transfer_seeds: 0 to 15, bootstrap: {n_boot: 2000, resample_seed: 0, alpha: 0.05}}. The resample seed differs from F1's.

**Missing provenance.**
- As SA-tab-f1, plus the SharpeBench version behind the deflated Sharpes.

**Relevant later changes.**
- Standardized-moment estimators mixed finite-sample normalizations, which moves deflated Sharpes and their intervals. SharpeBench corrected its estimators in 0.19.0 and SharpeArena its own confidence estimator in 0.25.0; the frozen F1, F3, witness and predictability artifacts predate both corrections and have not been rescored (paper/sections/10-limitations.tex:15). F3 is named in that group.
- Apart from F7 the committed F1 to F8 artifacts were not rerun after effective-configuration readback, SPEC_HASH, canonical pre-hash fixtures, typed FFI errors, strict input decoding, path-containment checks or the prospective forecast contract were added. The manuscript states those additions strengthen present executions without changing the observations of artifacts that were not rerun (paper/sections/10-limitations.tex:13).
- The Python reductions behind the realism, manipulation, adverse-selection, failure, ecology and predictability evidence are off the golden-hash path, so their floating-point reproducibility is the platform's rather than the paper's (paper/sections/10-limitations.tex:11).
- The rescore of 2026-10-31 recomputed the gaps, the transfer matrix and every interval from per_seed_returns, on the frozen producer's RNG streams.

**Present applicability.** Rescored under the pinned 0.28.0 kernel. The gaps read +0.0012, -0.3388 and -0.1012 and the transfer matrix +0.877, +0.973 and +0.096 as printed; the largest movement is 6.1e-7. The worked instance at six trials reads 0.111387, which prints as 0.1114 and differs from the frozen 0.111386 in the seventh decimal, so "exactly" holds at printed precision and not to the bit.

**Disposition.** `still-applicable`. A documented comparison now exists: the rescore stores each frozen value beside its current one, and no printed cell changes.

**Owner.** SharpeArena paper/evidence owner

**Follow-up.** none

**Status history.**

| Date | Disposition | Recorded by | Independent | Rationale |
|---|---|---|---|---|
| 2026-09-23 | needs-rescore | unknown | no | Initial entry: Recorded inputs suffice: per_seed_returns is committed and the manuscript records a worked rescore of that series, so a corrected-estimator result is a recomputation rather than a new run. P00-E maps evidence and authorizes no rerun. The follow-up names the ticket that would own one. |
| 2026-10-31 | still-applicable | unknown | no | Rescored from the committed per_seed_returns under SharpeBench 0.28.0. Moves to still-applicable: every printed cell is unchanged and the worked instance reproduces 0.1114. |

### `SA-tab-f7`

**Claim.** Failure-mode distributions over 384 episodes: clean rates of 0.40, 0.38 and 0.34 by tier, with 231 of 384 episodes ending in a process failure and 231 of 241 failures non-PnL.

**Source.** `paper/sections/08-findings.tex:158`, `paper/sections/08-findings.tex:180`, `paper/sections/08-findings.tex:151`

**Labels.** `tab:f7`, `fig:f7`

**Artifacts.**
- `paper/evidence/f7-failures.json`, sha256 `e30fc65eafc183df18bc5f7e0b6c80a4194889af2cbd3388786926b167f86bfc`, digest source: computed from the file on 2026-10-31 at sharpearena 7769ef5, equal to the digest paper/evidence/provenance.json records, keys config, effective_config, a 384-entry episodes list, rollup_by_tier and rollup_overall
- `paper/figures/f7-failures.pdf`, sha256 `a8848234748264a3d3dd489cb763d7238c1a5d6843af0800383e365f575f7a21`, digest source: computed from the file on 2026-10-31 at sharpearena 7769ef5, equal to the digest paper/evidence/provenance.json records

**Producer command.** python paper/src/make-f7-failures.py (paper/sections/A-commands.tex)

**Producing commit.** Regenerated for SharpeArena 0.29.0 by re-running its producer, which the manuscript states at paper/sections/08-findings.tex:153. The runtime version is not serialized in the artifact, so the attribution rests on the manuscript and the release history.

**Effective configuration.** config = {n_symbols: 4, n_days: 120, seeds: 0 to 15, max_steps: 512, max_drawdown: 0.5, three tiers, eight policies}; effective_config carries per-tier panel dimensions, window, verified true and 16 per-seed dataset fingerprints.

**Missing provenance.**
- Runtime version inside the artifact, SharpeBench version, producing commit and host.

**Relevant later changes.**
- The failure-taxonomy repair, under which an episode that failed, went missing or returned a nonfinite value can no longer default to a clean disposition, is reflected in this artifact because it was regenerated; the mandate draw change is too (paper/sections/10-limitations.tex:15).

**Present applicability.** The only committed artifact that absorbed the post-freeze repairs rather than lagging them, and the only one carrying an effective-configuration readback.

**Disposition.** `still-applicable`. The documented comparison is the recorded regeneration for 0.29.0 after the taxonomy repair and the mandate draw change, which the manuscript states and the artifact's readback block supports.

**Owner.** SharpeArena paper/evidence owner

**Follow-up.** P15: serialize a runtime version in the artifact so the 0.29.0 attribution does not rest on prose.

**Status history.**

| Date | Disposition | Recorded by | Independent | Rationale |
|---|---|---|---|---|
| 2026-09-23 | still-applicable | unknown | no | Initial entry: The documented comparison is the recorded regeneration for 0.29.0 after the taxonomy repair and the mandate draw change, which the manuscript states and the artifact's readback block supports. |
| 2026-10-31 | still-applicable | unknown | no | Rechecked against SharpeBench #118, #171 and #172, SharpeArena #118 and the foundation branch through 7769ef5: no evidence producer steps a vector environment, so SharpeArena #118 reaches no artifact here. The artifact digests were opened against the sharpearena tree and match. This row moved here from the SharpeBench register, which keeps a mirror of it. |

### `SA-tab-f8`

**Claim.** Ecology outcomes over eight replicator seeds per schedule: the root winner differs between the control and shocked schedules on five of eight seeds, and a single-seed reading does not survive replication.

**Source.** `paper/sections/08-findings.tex:192`, `paper/sections/08-findings.tex:224`, `paper/sections/08-findings.tex:183`

**Labels.** `tab:f8`, `fig:f8`

**Artifacts.**
- `paper/evidence/f8-ecology.json`, sha256 `bed6e9e3e5430e076685db9d3bd2b912ea531859f0134cff89aea4082990ff19`, digest source: computed from the file on 2026-10-31 at sharpearena 7769ef5, equal to the digest paper/evidence/provenance.json records, keys config, control, shocked, control_table, shocked_table, multi_seed
- `paper/figures/f8-ecology-control.pdf`, sha256 `299f46fb1d5032a85fac289948eb0ded8d1bd42234a71e61a8db91622cfa8ae8`, digest source: computed from the file on 2026-10-31 at sharpearena 7769ef5, equal to the digest paper/evidence/provenance.json records
- `paper/figures/f8-ecology-shocked.pdf`, sha256 `5ed376a14d2fe58951a032793e535921708fddb1aae9e94b8ac393020a403511`, digest source: computed from the file on 2026-10-31 at sharpearena 7769ef5, equal to the digest paper/evidence/provenance.json records

**Producer command.** python paper/src/make-f8-ecology.py (paper/sections/A-commands.tex)

**Producing commit.** Unknown. paper/sections/A-commands.tex:5 is the paper's single provenance statement: only f1-baselines.json serializes a runtime version (sharpearena 0.9.0), no evidence file serializes a SharpeBench version, and the paper declines to reconstruct one from creation dates. No artifact carries a producing commit; the manifest records only the commit it was generated at.

**Effective configuration.** config = {generations: 12, field_size: 8, seeds: 0 to 7, detail_seed: 0, n_symbols: 4, n_days: 120, max_steps: 256, innovate_every: 4, shock_period: 4}. The figures are seed 0 only, which the caption states.

**Missing provenance.**
- As the other non-F7 artifacts.

**Relevant later changes.**
- Pre-freeze regeneration during the August 2026 review round extended the producer to eight replicator seeds per schedule. No post-freeze change bearing on F8 was found.
- The Python reductions behind the realism, manipulation, adverse-selection, failure, ecology and predictability evidence are off the golden-hash path, so their floating-point reproducibility is the platform's rather than the paper's (paper/sections/10-limitations.tex:11).

**Present applicability.** Bounded by the manuscript's own statement that the replicator is an abstraction and that a comparable simulator runs 128 replications per configuration against this artifact's eight.

**Disposition.** `historical-only`. The artifact validly records the run and no later change reaches it, but nothing re-establishes it against the current tree.

**Owner.** SharpeArena paper/evidence owner

**Follow-up.** none

**Status history.**

| Date | Disposition | Recorded by | Independent | Rationale |
|---|---|---|---|---|
| 2026-09-23 | historical-only | unknown | no | Initial entry: The artifact validly records the run and no later change reaches it, but nothing re-establishes it against the current tree. |
| 2026-10-31 | historical-only | unknown | no | Rechecked against SharpeBench #118, #171 and #172, SharpeArena #118 and the foundation branch through 7769ef5: no evidence producer steps a vector environment, so SharpeArena #118 reaches no artifact here. The artifact digests were opened against the sharpearena tree and match. This row moved here from the SharpeBench register, which keeps a mirror of it. |

### `SA-tab-throughput`

**Claim.** Native Rust runs 747,468 steps a second, WebAssembly 508,127 and Python 14,240; the full six-policy field over the canonical held-out band takes 302 seconds.

**Source.** `paper/sections/06-protocol.tex:85`, `paper/sections/06-protocol.tex:78`

**Labels.** `tab:throughput`

**Artifacts.**
- `paper/evidence/throughput.json`, sha256 `dae509f677049e673bb9f3040b776e046fd7ee8210ee8f13e8a2922179a99d17`, digest source: computed from the file on 2026-10-31 at sharpearena 7769ef5, equal to the digest paper/evidence/provenance.json records, the only artifact with a host block: platform, processor and Python version, plus canonical_evaluation with the three measured runs

**Producer command.** python paper/src/make-throughput.py, which shells out to cargo run --release -p sharpearena --example bench-steps and node npm/sharpearena/bench/throughput.js (paper/sections/A-commands.tex)

**Producing commit.** Unknown. paper/sections/A-commands.tex:5 is the paper's single provenance statement: only f1-baselines.json serializes a runtime version (sharpearena 0.9.0), no evidence file serializes a SharpeBench version, and the paper declines to reconstruct one from creation dates. No artifact carries a producing commit; the manifest records only the commit it was generated at. The host block is recorded: Windows 11, an AMD64 family 25 processor and Python 3.12.6.

**Effective configuration.** episodes_per_repeat 500, repeats 3, steps_per_repeat 60000, wasm steps_per_episode 100.

**Missing provenance.**
- Runtime version, producing commit, and the Rust toolchain and Node versions behind the two shelled-out benchmarks.

**Relevant later changes.**
- Audit repair AP2: the WebAssembly throughput producer and the npm package manifest were outside the provenance source scope, so the producer could change without moving source_snapshot_sha256. Both are now in scope.

**Present applicability.** Hardware-dependent by nature, and the appendix names throughput as the one quantity a replicator should expect to differ. The host block makes the measurement identifiable, which no other artifact offers.

**Disposition.** `historical-only`. The artifact validly records a timed run on a named host; a current measurement would be a different machine's number rather than a correction of this one.

**Owner.** SharpeArena paper/evidence owner

**Follow-up.** none

**Status history.**

| Date | Disposition | Recorded by | Independent | Rationale |
|---|---|---|---|---|
| 2026-09-23 | historical-only | unknown | no | Initial entry: The artifact validly records a timed run on a named host; a current measurement would be a different machine's number rather than a correction of this one. |
| 2026-10-31 | historical-only | unknown | no | Rechecked against SharpeBench #118, #171 and #172, SharpeArena #118 and the foundation branch through 7769ef5: no evidence producer steps a vector environment, so SharpeArena #118 reaches no artifact here. The artifact digests were opened against the sharpearena tree and match. This row moved here from the SharpeBench register, which keeps a mirror of it. |

### `SA-orderbook-mark`

**Claim.** Over seeds 0 to 31 with no inventory penalty a lone quoting agent earns a mean episode reward of 881.4 against 897.1 for cash plus inventory at the final reference mid, where the frozen mark it replaced paid 2360.3.

**Source.** `paper/sections/03-environment.tex:112`

**Artifacts.**
- `crates/sharpearena-py/tests/test_lob_env.py`, no digest recorded, digest source: no digest: a test file that may be edited, test_a_lone_quote_walk_is_not_booked_as_reward asserts 881.4 and 897.1 over seeds 0 to 31; its docstring and sharpearena/lob_env.py record 2360.3 under the frozen mark, which no current code path produces

**Producer command.** pytest crates/sharpearena-py/tests/test_lob_env.py::test_a_lone_quote_walk_is_not_booked_as_reward, run for this row on 2026-10-31 and passing.

**Producing commit.** 06fb4c0 on 2026-09-17, which added the test and the mark it measures.

**Effective configuration.** Stated in prose only: seeds 0 to 31, no inventory penalty, one tick below the reference mid on the bid and twenty above on the ask, sixty steps of mid-walking. No seed serialization, no interval.

**Missing provenance.**
- No interval and no per-seed record: the test rounds two means to one decimal.
- The frozen-mark figure 2360.3 belongs to code since removed, so it stands as a recorded value and is not re-executed.

**Relevant later changes.**
- The claim is itself the record of a later change: the order-book environment's default inventory mark changed twice, so rewards from earlier runs reproduce only under one mark setting and rewards under the intervening fallback do not reproduce at all; no committed artifact uses the environment (paper/sections/10-limitations.tex:15).

**Present applicability.** Current for 881.4 and 897.1, which a committed test asserts on the present tree. The third number is the historical comparator the sentence presents it as.

**Disposition.** `still-applicable`. The artifact the 2026-09-23 entry could not locate is a committed test, which reproduces the two current numbers exactly at the printed precision.

**Owner.** SharpeArena paper/evidence owner

**Follow-up.** none

**Status history.**

| Date | Disposition | Recorded by | Independent | Rationale |
|---|---|---|---|---|
| 2026-09-23 | unresolved | unknown | no | Initial entry: The artifact cannot be located, so the ticket's rule applies: record unresolved rather than guess. |
| 2026-10-31 | still-applicable | unknown | no | Resolved as a provenance answer: test_a_lone_quote_walk_is_not_booked_as_reward asserts the numbers and passes on the current tree. The 2026-09-23 entry searched paper/evidence only. |

### `SA-seeded-shuffle`

**Claim.** Seat ordering under the default agent-index priority advantages the lower index, which the seeded shuffle removes.

**Source.** `paper/sections/03-environment.tex:110`

**Artifacts.**
- `crates/sharpearena-py/tests/test_lob_env.py`, no digest recorded, digest source: no digest: a test file that may be edited, test_default_priority_always_queues_seat_zero_first asserts 32 of 32 and test_seeded_shuffle_makes_identical_quoters_exchangeable asserts 15 of 32, both over seeds 0 to 31

**Producer command.** pytest on those two named tests in crates/sharpearena-py/tests/test_lob_env.py, run for this row on 2026-10-31 and passing.

**Producing commit.** 4473079 on 2026-09-17, which added the rule and its tests.

**Effective configuration.** Stated in the changelog only: seeds 0 to 31, two identical three-tick quoters.

**Missing provenance.**
- The manuscript states the property qualitatively and does not cite this measurement.

**Relevant later changes.**
- The seeded-shuffle priority is itself the later change; the measurement predates no repair.

**Present applicability.** Current: the two tests assert seat 0 filled more on 32 of 32 seeds under the default order, 14,222 units against 13,900, and on 15 of 32 under the seeded shuffle.

**Disposition.** `still-applicable`. The supporting measurement is held by committed tests that run on the current tree, so the qualitative claim has a recorded comparison behind it.

**Owner.** SharpeArena paper/evidence owner

**Follow-up.** none

**Status history.**

| Date | Disposition | Recorded by | Independent | Rationale |
|---|---|---|---|---|
| 2026-09-23 | unresolved | unknown | no | Initial entry: The supporting measurement has no artifact in the evidence scope, so no disposition can be chosen without guessing. |
| 2026-10-31 | still-applicable | unknown | no | Resolved as a provenance answer: two named tests assert the measurement and pass on the current tree. The 2026-09-23 entry searched paper/evidence only. |

### `SA-nonempirical-tables`

**Claim.** The observation and action wire schemas, the leakage-channel summary and the related-work comparison.

**Source.** `paper/sections/03-environment.tex:20`, `paper/sections/03-environment.tex:55`, `paper/sections/06-protocol.tex:36`, `paper/sections/09-related.tex:12`

**Labels.** `tab:obs-space`, `tab:act-space`, `tab:leakage`, `tab:related`

**Artifacts.**
- The leakage table is declared non-additive in its own section: every entry restates a property established in the section it cites, so the table adds no result. The two space tables describe the wire schema, and the related-work table's caption states that entries rest on the authors' reading where the cited authors do not claim the property.

**Producer command.** none; these tables are written from the interface, the protocol and the cited literature.

**Producing commit.** not applicable

**Effective configuration.** not applicable

**Missing provenance.**
- The related-work table's entries rest on the authors' reading, which the caption states, and there is no per-cell provenance sheet of the kind the companion product keeps.

**Relevant later changes.**
- A schema change to the observation or action space would move the two space tables; none was found in scope for this register.

**Present applicability.** Applicable as descriptions rather than measurements.

**Disposition.** `still-applicable`. No cell reports a measurement, so no artifact can go stale under these tables.

**Owner.** SharpeArena paper/evidence owner

**Follow-up.** P15: consider a per-cell provenance sheet for the related-work table, as the companion product keeps.

**Status history.**

| Date | Disposition | Recorded by | Independent | Rationale |
|---|---|---|---|---|
| 2026-09-23 | still-applicable | unknown | no | Initial entry: No cell reports a measurement, so no artifact can go stale under these tables. |
| 2026-10-31 | still-applicable | unknown | no | Rechecked against SharpeBench #118, #171 and #172, SharpeArena #118 and the foundation branch through 7769ef5: no evidence producer steps a vector environment, so SharpeArena #118 reaches no artifact here. The artifact digests were opened against the sharpearena tree and match. This row moved here from the SharpeBench register, which keeps a mirror of it. |

### `SA-sec-leakfree`

**Claim.** No data-layer operation returns a bar past the engine's cursor, on every tier; the guarantee is at the interface and does not cover in-band prediction of the public generator.

**Source.** `paper/sections/03-environment.tex:5`, `paper/sections/00-abstract.tex:2`

**Artifacts.**
- `crates/sharpearena/src/market.rs`, no digest recorded, digest source: no digest: a source or test file that may be edited, test every_tier_is_leak_free_never_surfaces_a_future_bar

**Producer command.** cargo test -p sharpearena every_tier_is_leak_free_never_surfaces_a_future_bar

**Producing commit.** not applicable: a property of the current tree, not a run

**Effective configuration.** not applicable

**Missing provenance.**
- none identified

**Relevant later changes.**
- none identified

**Present applicability.** Current. In-band prediction is the subject of SA-predictability.

**Disposition.** `still-applicable`. A test in this repository asserts the property and runs in continuous integration on the current tree.

**Owner.** SharpeArena paper/evidence owner

**Follow-up.** none

**Status history.**

| Date | Disposition | Recorded by | Independent | Rationale |
|---|---|---|---|---|
| 2026-10-31 | still-applicable | unknown | no | Initial entry: A test in this repository asserts the property and runs in continuous integration on the current tree. |

### `SA-sec-determinism`

**Claim.** The Rust core, its WebAssembly build and the scenario bytes returned through Python are pinned by committed goldens checked on native, WebAssembly and packaged surfaces.

**Source.** `paper/sections/03-environment.tex:82`, `paper/sections/04-contract.tex:29`

**Artifacts.**
- `crates/sharpearena/contract/attestation/scenario-goldens.json`, sha256 `39653c335e5d817e1c98ae29867707b8bbab6ae07250eb14efbf0f1cd7b518bd`, digest source: computed from the file on 2026-10-31 after integration; the martingale_v1_4x120_seed7 golden was appended and no existing golden moved, read by the native suite, wasm-pack and the npm suite
- `crates/sharpearena/contract/attestation/backtest-goldens.json`, sha256 `08734c88d5b360f2d7571ffa84c3dff7b9956b2ec054f40a590502780b6bdb46`, digest source: computed from the file on 2026-10-31; a golden that must not move, read by the native suite, wasm-pack and the npm suite
- `crates/sharpearena/contract/attestation/kernel-goldens.json`, sha256 `b5aa16776dc73ba2f64dfbcf4d96151dd978df8e1c2cfbb52506d0a6c2b3cb3b`, digest source: computed from the file on 2026-10-31; a golden that must not move, read by the native suite, wasm-pack and the npm suite

**Producer command.** the native, wasm-pack and npm suites, each reading the three committed golden files

**Producing commit.** not applicable: a property of the current tree, not a run

**Effective configuration.** not applicable

**Missing provenance.**
- none identified

**Relevant later changes.**
- none identified

**Present applicability.** Current: the three gates read these exact bytes.

**Disposition.** `still-applicable`. A test in this repository asserts the property and runs in continuous integration on the current tree.

**Owner.** SharpeArena paper/evidence owner

**Follow-up.** none

**Status history.**

| Date | Disposition | Recorded by | Independent | Rationale |
|---|---|---|---|---|
| 2026-10-31 | still-applicable | unknown | no | Initial entry: A test in this repository asserts the property and runs in continuous integration on the current tree. |

### `SA-sec-replay`

**Claim.** Replay recomputes performance from recorded decisions and frozen engine inputs, and a tamper test requires a changed decision to diverge; fields outside the replay input are not authenticated.

**Source.** `paper/sections/03-environment.tex:88`, `paper/sections/00-abstract.tex:2`

**Artifacts.**
- `npm/sharpearena/test/smoke.test.js`, no digest recorded, digest source: no digest: a source or test file that may be edited, the replay and tamper-detection suite paper/sections/A-commands.tex:22 names

**Producer command.** npm test --prefix npm/sharpearena

**Producing commit.** not applicable: a property of the current tree, not a run

**Effective configuration.** not applicable

**Missing provenance.**
- none identified

**Relevant later changes.**
- none identified

**Present applicability.** Current, with the stated boundary.

**Disposition.** `still-applicable`. A test in this repository asserts the property and runs in continuous integration on the current tree.

**Owner.** SharpeArena paper/evidence owner

**Follow-up.** none

**Status history.**

| Date | Disposition | Recorded by | Independent | Rationale |
|---|---|---|---|---|
| 2026-10-31 | still-applicable | unknown | no | Initial entry: A test in this repository asserts the property and runs in continuous integration on the current tree. |

### `SA-sec-wire-contract`

**Claim.** A governed contract at version 1.0 with JSON Schemas, a conformance kit and an additive-only policy, exercised by a committed conformance test.

**Source.** `paper/sections/04-contract.tex:5`

**Artifacts.**
- `crates/sharpearena/tests/conformance.rs`, no digest recorded, digest source: no digest: a source or test file that may be edited, the conformance test the section names
- `crates/sharpearena/tests/conformance_manifest.rs`, no digest recorded, digest source: no digest: a source or test file that may be edited, holds the kit to its manifest

**Producer command.** cargo test -p sharpearena --test conformance --test conformance_manifest

**Producing commit.** not applicable: a property of the current tree, not a run

**Effective configuration.** not applicable

**Missing provenance.**
- none identified

**Relevant later changes.**
- none identified

**Present applicability.** Current.

**Disposition.** `still-applicable`. A test in this repository asserts the property and runs in continuous integration on the current tree.

**Owner.** SharpeArena paper/evidence owner

**Follow-up.** none

**Status history.**

| Date | Disposition | Recorded by | Independent | Rationale |
|---|---|---|---|---|
| 2026-10-31 | still-applicable | unknown | no | Initial entry: A test in this repository asserts the property and runs in continuous integration on the current tree. |

### `SA-sec-transport-fault`

**Claim.** A transport fault is a typed failed cell on the checked paths and an empty-order hold on the low-level compatibility call, which a characterization test pins.

**Source.** `paper/sections/04-contract.tex:21`

**Artifacts.**
- `crates/sharpearena/src/transport_gate.rs`, no digest recorded, digest source: no digest: a source or test file that may be edited, tests a_wedged_agent_yields_a_scoreable_series_through_the_unguarded_engine and a_wedged_agent_is_a_failed_cell_not_a_return_series

**Producer command.** cargo test -p sharpearena transport_gate

**Producing commit.** not applicable: a property of the current tree, not a run

**Effective configuration.** not applicable

**Missing provenance.**
- none identified

**Relevant later changes.**
- none identified

**Present applicability.** Current.

**Disposition.** `still-applicable`. A test in this repository asserts the property and runs in continuous integration on the current tree.

**Owner.** SharpeArena paper/evidence owner

**Follow-up.** none

**Status history.**

| Date | Disposition | Recorded by | Independent | Rationale |
|---|---|---|---|---|
| 2026-10-31 | still-applicable | unknown | no | Initial entry: A test in this repository asserts the property and runs in continuous integration on the current tree. |

### `SA-sec-agentic-layer`

**Claim.** The model-agent field, the strategy-generation path, the edge manifest, the counterfactual ledger, the forward paper arm and trace promotion are architecture with stated guarantees, and no agent result is reported from any of them.

**Source.** `paper/sections/05-agentic.tex:5`, `paper/sections/05-agentic.tex:35`, `paper/sections/05-agentic.tex:45`, `paper/sections/05-agentic.tex:53`, `paper/sections/06-protocol.tex:66`

**Artifacts.**
- `crates/sharpearena-py/tests/test_strategy_generation.py`, no digest recorded, digest source: no digest: a source or test file that may be edited, strategy path
- `crates/sharpearena-py/tests/test_edge_manifest.py`, no digest recorded, digest source: no digest: a source or test file that may be edited, closed falsifiability manifest
- `crates/sharpearena-py/tests/test_counterfactual.py`, no digest recorded, digest source: no digest: a source or test file that may be edited, counterfactual ledger
- `crates/sharpearena-py/tests/test_paper_trading.py`, no digest recorded, digest source: no digest: a source or test file that may be edited, forward paper arm
- `crates/sharpearena-py/tests/test_trace_promotion.py`, no digest recorded, digest source: no digest: a source or test file that may be edited, trace promotion

**Producer command.** python -m pytest in crates/sharpearena-py

**Producing commit.** not applicable: a property of the current tree, not a run

**Effective configuration.** not applicable

**Missing provenance.**
- none identified

**Relevant later changes.**
- none identified

**Present applicability.** Current as architecture; the section reports no number from these paths.

**Disposition.** `still-applicable`. A test in this repository asserts the property and runs in continuous integration on the current tree.

**Owner.** SharpeArena paper/evidence owner

**Follow-up.** none

**Status history.**

| Date | Disposition | Recorded by | Independent | Rationale |
|---|---|---|---|---|
| 2026-10-31 | still-applicable | unknown | no | Initial entry: A test in this repository asserts the property and runs in continuous integration on the current tree. |

### `SA-sec-prospective-protocol`

**Claim.** Forecast contracts freeze settlement before submission, revisions are append-only, strict evidence is exported unscored for SharpeBench, a cross-product tutorial exercises the boundary, and a Lean model proves selected rules with no mechanical link to the code.

**Source.** `paper/sections/05-agentic.tex:19`, `paper/sections/05-agentic.tex:27`

**Artifacts.**
- `crates/sharpearena-py/tests/test_forecast.py`, no digest recorded, digest source: no digest: a source or test file that may be edited, ledger rules
- `crates/sharpearena-py/tests/test_forecast_evidence.py`, no digest recorded, digest source: no digest: a source or test file that may be edited, strict export
- `crates/sharpearena-py/tests/test_forecast_quality_tutorial.py`, no digest recorded, digest source: no digest: a source or test file that may be edited, cross-product tutorial
- `formal/SharpeArenaFormal.lean`, no digest recorded, digest source: no digest: a source or test file that may be edited, the Lean model

**Producer command.** python -m pytest in crates/sharpearena-py; lake build in formal/

**Producing commit.** not applicable: a property of the current tree, not a run

**Effective configuration.** not applicable

**Missing provenance.**
- none identified

**Relevant later changes.**
- none identified

**Present applicability.** Current for what the tests and the model each state. SharpeBench #171 changed the report the tutorial reaches, which SharpeBench records at SB-sec-forecast-report.

**Disposition.** `still-applicable`. A test in this repository asserts the property and runs in continuous integration on the current tree.

**Owner.** SharpeArena paper/evidence owner

**Follow-up.** none

**Status history.**

| Date | Disposition | Recorded by | Independent | Rationale |
|---|---|---|---|---|
| 2026-10-31 | still-applicable | unknown | no | Initial entry: A test in this repository asserts the property and runs in continuous integration on the current tree. |

### `SA-forecast-pilot`

**Claim.** A superseded convenience-sample protocol pilot remains committed for protocol auditability and produced no admitted model-validation result.

**Source.** `paper/sections/12-reproducibility.tex:9`, `paper/sections/01-introduction.tex:19`, `paper/sections/00-abstract.tex:2`

**Artifacts.**
- `paper/evidence/prospective-forecast-field/field-plan.json`, sha256 `61d225c357ec13dea63851afd17ca66866d841efd71dc1da18b74c85a3342a3c`, digest source: computed from the file on 2026-10-31 at sharpearena 7769ef5, equal to the digest paper/evidence/provenance.json records, the admitted attempt's plan
- `paper/evidence/prospective-forecast-field/resolution-manifest.json`, sha256 `c7f213512aa80558bd4c0ded3155798516b2a283911929a6e9c67012e2b78bbc`, digest source: computed from the file on 2026-10-31 at sharpearena 7769ef5, equal to the digest paper/evidence/provenance.json records, its resolution manifest
- `paper/evidence/prospective-forecast-field-attempt-1/attempt-status.json`, sha256 `099f7b75061aa02e93206cd2936c462e9603c2fcef85a9beecfe9274ba7c692b`, digest source: computed from the file on 2026-10-31 at sharpearena 7769ef5, equal to the digest paper/evidence/provenance.json records, an abandoned attempt, kept as such

**Producer command.** the sharpearena-prospective-field command; SharpeBench scores it and records the report at SB-forecast-pilot

**Producing commit.** recorded per artifact in the field's source manifest

**Effective configuration.** field-plan.json in each attempt directory

**Missing provenance.**
- none identified

**Relevant later changes.**
- none identified

**Present applicability.** Its checkpoints are obsolete and it is excluded from every result.

**Disposition.** `historical-only`. Retained so the protocol history is not rewritten, and bound to its own commit. It validates no current model.

**Owner.** SharpeArena paper/evidence owner

**Follow-up.** none

**Status history.**

| Date | Disposition | Recorded by | Independent | Rationale |
|---|---|---|---|---|
| 2026-10-31 | historical-only | unknown | no | Initial entry: Retained so the protocol history is not rewritten, and bound to its own commit. It validates no current model. |

### `SA-sec-protocol-seeds`

**Claim.** Resolution at 16 seeds: a pass rate near one half has a 95 percent half-width of about 0.24, a difference of two about 0.35, and 16 paired episodes detect a mean regret of about 12 reward units at 80 percent power.

**Source.** `paper/sections/06-protocol.tex:15`

**Artifacts.**
- `paper/evidence/f2-regret.json`, sha256 `7dcf2e656b4f73e8af7e4dba90721bab63f79b8de4a01c682b299b9f693c80ee`, digest source: computed from the file on 2026-10-31 at sharpearena 7769ef5, equal to the digest paper/evidence/provenance.json records, regret_dispersion supplies the standard deviation behind the regret figure

**Producer command.** none: arithmetic in the manuscript, over a committed dispersion

**Producing commit.** not applicable: arithmetic

**Effective configuration.** not applicable

**Missing provenance.**
- none identified

**Relevant later changes.**
- none identified

**Present applicability.** The two rates recompute as 0.245 and 0.346. The regret figure recomputes as 12.7 with the normal approximation at the half-spread 0.25 dispersion and 13.6 with the exact paired t, so "about 12" sits slightly low.

**Disposition.** `still-applicable`. The inputs are committed and the arithmetic is recorded here, and the conclusion the sentence draws does not depend on the difference between 12 and 13.6.

**Owner.** SharpeArena paper/evidence owner

**Follow-up.** P15: state the regret figure as about 13, or name the calculation.

**Status history.**

| Date | Disposition | Recorded by | Independent | Rationale |
|---|---|---|---|---|
| 2026-10-31 | still-applicable | unknown | no | Initial entry: The inputs are committed and the arithmetic is recorded here, and the conclusion the sentence draws does not depend on the difference between 12 and 13.6. |

### `SA-sec-throughput-derived`

**Claim.** The slowest surface runs ten million steps in about twelve minutes, the conformant field runs at roughly 1,830 steps a second, and a single agent on a single tier is one eighteenth of it.

**Source.** `paper/sections/00-abstract.tex:2`, `paper/sections/06-protocol.tex:99`, `paper/sections/06-protocol.tex:17`, `paper/sections/10-limitations.tex:7`

**Artifacts.**
- `paper/evidence/throughput.json`, sha256 `dae509f677049e673bb9f3040b776e046fd7ee8210ee8f13e8a2922179a99d17`, digest source: computed from the file on 2026-10-31 at sharpearena 7769ef5, equal to the digest paper/evidence/provenance.json records, python_steps_per_s and canonical_evaluation

**Producer command.** python paper/src/make-throughput.py; the derived figures are arithmetic on its output

**Producing commit.** not applicable: a property of the current tree, not a run

**Effective configuration.** not applicable

**Missing provenance.**
- none identified

**Relevant later changes.**
- none identified

**Present applicability.** 1e7 / 14,240 is 702 seconds, 11.7 minutes; 552,960 / 302 is 1,831; 256 / 4,608 is 1/18. Hardware-bound as SA-tab-throughput is.

**Disposition.** `historical-only`. The arithmetic recomputes from the artifact, and the artifact is bound to its recorded host.

**Owner.** SharpeArena paper/evidence owner

**Follow-up.** none

**Status history.**

| Date | Disposition | Recorded by | Independent | Rationale |
|---|---|---|---|---|
| 2026-10-31 | historical-only | unknown | no | Initial entry: The arithmetic recomputes from the artifact, and the artifact is bound to its recorded host. |

### `SA-sec-absences`

**Claim.** No agent is trained in this paper, no current-model field has been admitted, and no third party has submitted an agent.

**Source.** `paper/sections/01-introduction.tex:19`, `paper/sections/10-limitations.tex:25`

**Artifacts.**
- A statement of absence: there is nothing to measure.

**Producer command.** none

**Producing commit.** not applicable: a property of the current tree, not a run

**Effective configuration.** not applicable

**Missing provenance.**
- none identified

**Relevant later changes.**
- none identified

**Present applicability.** paper/evidence holds no trained-policy artifact and paper/evidence/model-artifacts holds no admitted model. The third-party statement cannot be read from the tree.

**Disposition.** `still-applicable`. A statement of the manuscript's own scope. The tree agrees with its first two parts, and the third is the owner's to keep current.

**Owner.** SharpeArena paper/evidence owner

**Follow-up.** none

**Status history.**

| Date | Disposition | Recorded by | Independent | Rationale |
|---|---|---|---|---|
| 2026-10-31 | still-applicable | unknown | no | Initial entry: A statement of the manuscript's own scope. The tree agrees with its first two parts, and the third is the owner's to keep current. |

### `SA-cited-figures`

**Claim.** Figures attributed to other works, among them the alpha-mining and market-making studies and the tool-call failure rate, are read from those works and are not results of this product.

**Source.** `paper/sections/01-introduction.tex:3`, `paper/sections/02-principles.tex:11`, `paper/sections/03-environment.tex:7`, `paper/sections/05-agentic.tex:33`, `paper/sections/06-protocol.tex:62`

**Artifacts.**
- Numbers quoted from cited works with page references.

**Producer command.** none

**Producing commit.** not applicable: a property of the current tree, not a run

**Effective configuration.** not applicable

**Missing provenance.**
- none identified

**Relevant later changes.**
- none identified

**Present applicability.** Applicable as citations. This register does not re-verify them against the cited pages.

**Disposition.** `still-applicable`. A citation carries no evidence artifact of this product, and freshness does not apply to it.

**Owner.** SharpeArena paper/evidence owner

**Follow-up.** none

**Status history.**

| Date | Disposition | Recorded by | Independent | Rationale |
|---|---|---|---|---|
| 2026-10-31 | still-applicable | unknown | no | Initial entry: A citation carries no evidence artifact of this product, and freshness does not apply to it. |
