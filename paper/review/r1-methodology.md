# Peer Review Report

## Manuscript Information
- **Title**: SharpeArena: The Point-in-Time (PIT) RL Environment for Trading Agents
- **Manuscript ID**: (none; local manuscript at `paper/main.tex`)
- **Review Date**: 2026-08-24
- **Review Round**: Round 1

---

## Reviewer Information

### Reviewer Role
Peer Reviewer 1 (Methodology)

### Reviewer Identity
RL evaluation methodology reviewer: generalization measurement (Procgen-style seed splits), statistical rigor of environment benchmarks, determinism and reproducibility claims for simulation infrastructure.

### Review Focus
Whether the eight findings' numbers trace to committed commands and evidence; whether the seed-band split methodology is sound; whether the golden-hash determinism claims are tested as claimed; adequacy of the trivial-policy baseline field; and whether effect sizes, uncertainty, and sample sizes (8 to 24 seeds/episodes, one ecology run) support the stated conclusions.

---

## Overall Assessment

### Recommendation
- [ ] Accept
- [ ] Minor Revision
- [x] **Major Revision**
- [ ] Reject

### Confidence Score
4. Mostly within my area of expertise (RL environment evaluation, benchmark statistics, reproducibility engineering); the market-microstructure modeling specifics are adjacent.

Confidence is an uncertainty/scope disclosure only; it never changes consensus counts, severity, decision bearing, or arbitration.

### Summary Assessment
The paper describes SharpeArena, a deterministic point-in-time RL trading environment, and reports eight scripted experiments as calibration findings. From the methodology seat the paper's strongest property is verified traceability: I independently recomputed every number in Tables 1 through 6 and the running text of Section 5 from the committed JSON under `paper/evidence/`, and every value matches (F1 board and CIs, F2 regret curve, F3 gap and transfer matrices, F4 per-fact pass counts including the clustered re-certification, F5 endpoints and size response, F6 markouts and the maker decomposition, F7 counts and per-policy claims, F8 outcome classifications and peaks). The design-property claims (golden hashes, wasm parity, npm tamper test, seed-band disjointness guards) exist in source where the paper says they do. The weaknesses are statistical, not engineering: outside F1, no finding carries any uncertainty quantification despite 8 to 24 unit samples; the F8 ecology headline, which reaches the abstract, rests on a single replicator seed; the paper's own noise calibration (0.34 DSR of 16-seed band luck) is never applied to the transfer matrix it sits beside; and the canonical seed-band protocol of Section 3.4 is defined but never exercised by the evidence run, which draws all evaluation seeds from the train band. These are repairable with reruns and added dispersion reporting, hence Major Revision rather than Reject.

---

## Strengths

### S1: Verified number-to-evidence traceability
Every empirical number in Section 5 that I checked (all six tables and the in-text values) recomputes exactly from the committed evidence JSON, and each evidence file is produced by a committed, seed-fixed script that imports only the published package API. This is materially better than the field norm for environment papers, and the `make-figures.py` re-render path lets a reader confirm figures contain no number the evidence does not.
**Evidence Anchor**: `dataset: paper/evidence/f1-baselines.json through f8-ecology.json, cross-checked against Tables 1-6 of sections/05-experiments.tex`

### S2: Determinism claims are tested as claimed
The committed FNV-1a golden hashes exist per scenario family at a fixed seed (`GOLDEN_CALM_4X120_SEED7_FNV1A` and siblings, including the clustered variant), the WebAssembly crate asserts byte-identity with the native engine for both stepping and scenario generation, and the npm test suite contains the replay-and-tamper check described in Section 3.3. The paper's scoping is also honest: the hash pins one canonical seed per family, and Section 2 plus the limitations correctly restrict the byte-identity guarantee to the Rust core.
**Evidence Anchor**: `dataset: crates/sharpearena/src/scenario_gen.rs golden-hash test constants; crates/sharpearena-wasm/src/lib.rs parity assertions; npm/sharpearena/test/smoke.test.js tamper test`

### S3: Paired-control experimental designs
F2 (candidate vs closed-form optimum on identical seeded episodes), F5 (manipulator vs zero-impact reference sharing seed and schedule), F6 (informed vs coin-flip-sided identical flow on a fixed price path), and F8 (shocked vs steady control schedule) all difference out episode-level luck by construction. This is the correct design family for small-sample simulator probes and substantially strengthens the directional claims.
**Evidence Anchor**: `text: §3.7 "the live and reference runs share the seed and the follower population, and differ only in that the reference sets both impact coefficients to zero"`

### S4: Honest negative results and calibrated instruments
F4 reports the realism certification failing its own conjunction (1 of 24 unclustered runs) and states it as a finding against the generator; F3 explicitly interprets the within-tier gaps of a zero-parameter policy as a noise calibration rather than as evidence of robustness; the limitations section pre-empts the main external-validity objections. This is the correct epistemic posture for a self-probing benchmark.
**Evidence Anchor**: `text: §5.4 "This is a finding against the generator, stated as such"`

### S5: F1 uncertainty done properly where it exists
The baseline board carries seed-paired bootstrap 95% CIs per row, and the text uses them correctly, flagging the min-variance Calm interval [0.0264, 1.0000] as near-vacuous rather than reading the 0.9849 point estimate at face value.
**Evidence Anchor**: `table: Table 1 context, CIs [0.9798, 1.0000], [0.8343, 1.0000], [0.0264, 1.0000] in §5.1 text`

### S6: Seed-split machinery is structurally sound
The Procgen-style integer-seed-interval model with a bounded train interval, an unsampled 10,000-seed gap, refusal of unbounded train intervals, a reserved evaluation namespace at seed 1e6, and a code-level disjointness guard is the right generalization-measurement design, and `cross_regime_transfer` holding the seed band fixed while swapping regimes is a genuinely stronger probe than a within-tier split.
**Evidence Anchor**: `dataset: crates/sharpearena-py/python/sharpearena/eval_seeds.py, EVAL_SEED_BASE = 1_000_000 with held_out_* seeds`

---

## Weaknesses

### W1: The F8 ecology finding rests on a single replicator run per schedule
**Problem**: The entire F8 experiment is one control trajectory and one shocked trajectory from a single seed (`"seed": 0` in the committed config). The abstract-level claim "a regime-shock schedule replaces the calm-regime ecology winner with the unlevered long book" is an N=1 result. Replicator dynamics with extinction thresholds, innovation events every 4 generations, and shared-book fitness coupling are exactly the kind of system where the identity of the dominant species can be sensitive to the seed, the founding shares, and the innovation timing.
**Evidence Anchor**: `dataset: paper/evidence/f8-ecology.json — config records generations 12, field_size 8, seed 0; one trajectory per schedule`
**Why it matters**: The paper generalizes from one run to a claim about which strategy classes survive regime transitions, and that claim is promoted to the abstract. Without replication there is no way to distinguish "volatility targeting is structurally eliminated by Hard transitions" from "it happened once at seed 0." The mechanistic explanation offered (leverage profile liquidated by the transition) is plausible but post hoc.
**Suggestion**: Rerun the ecology over a modest seed batch (8 to 16 replicator seeds per schedule, matching the paper's own convention elsewhere), and report the frequency with which the calm winner is replaced, plus the distribution of final dominant species. If the replacement is robust across seeds, the finding strengthens at trivial compute cost; if it is not, the current text overclaims.
**Severity**: Major
**Confidence**: 5 — core expertise: multi-run reliability of stochastic-population RL evaluations

### W2: No uncertainty quantification on any finding except F1
**Problem**: F2 reports mean regret over 16 episodes, F3 reports gap and transfer values over 16 seeds, F5 reports mean impact PnL over 8 seeds, and F6 reports mean markout gaps over 24 paired episodes, all as bare point estimates. No standard errors, no CIs, no per-unit dispersion anywhere outside F1. The committed evidence compounds this for F2 and F6: only aggregate means are serialized (`fixed_spread_regret` maps half-spread to a single float; the F6 `comparison` block holds only aggregate per-horizon means), so a reader cannot compute the dispersion from the artifacts either.
**Evidence Anchor**: `dataset: paper/evidence/f2-regret.json and f6-adverse-selection.json — aggregate means only, no per-episode values committed`
**Why it matters**: Several interpreted quantities are plausibly within seed noise. The clearest internal tension: Section 5.3 itself calibrates 16-seed band luck at up to 0.34 DSR, yet the adjacent transfer matrix reports Hard-to-Extreme transfer of 0.096, well inside that calibration, in the same table as the interpreted 0.877 and 0.973 entries, with no per-cell uncertainty and no caveat distinguishing signal cells from noise cells. Similarly, the F6 h=1 gap of 0.072 per unit is asserted as directional evidence ("non-vacuous in the required direction at every horizon") without a paired-difference interval showing it excludes zero. The paired designs make the fix cheap and powerful.
**Suggestion**: Report seed-paired bootstrap or t-based 95% CIs for the F2 regret means, the F3 gap and transfer entries, the F5 impact PnL points, and the F6 per-horizon paired gaps, and commit the per-episode/per-seed vectors in the evidence JSON. In Table 2 (right matrix), visually or textually separate cells whose magnitude exceeds the paper's own noise calibration from those that do not.
**Severity**: Major
**Confidence**: 5 — core expertise: statistical reporting for small-sample benchmark evaluations

### W3: Canonical seed-band protocol is defined but never exercised; F1's "held-out seed" language misdescribes the seeds actually used
**Problem**: Section 3.4 fixes canonical bands (train [0, 256), a 10,000-seed gap, test [10256, 10512), plus a reserved evaluation namespace at seed 1e6 with a committed held-out regression set). But every experiment in the evidence run draws seeds 0 through 15, inside the train band: F1, F4, F5, F7, and the F3 transfer seeds all use `seeds: [0..15]`, and F3's `generalization_gap` derives its test band from `n_train=16, gap=10000` rather than the canonical bands. The reserved 1e6 namespace is never touched by any reported number. F1's reading paragraph then states that the number an agent must beat is one that "survives every held-out seed," when the pass-per-seed gate in F1 was evaluated on train-band seeds.
**Evidence Anchor**: `text: §5.1 "a positive, deflated Sharpe that survives every held-out seed, and no reference policy produces one"`
**Why it matters**: For zero-parameter reference policies the train/held-out distinction carries no leakage risk, so no result is invalidated. But the paper's central generalization-measurement claim is that the protocol's bands are fixed and observed; the paper's own evidence run then models the opposite practice, and the "held-out" wording in F1 is not accurate for the seeds used. A reader implementing the protocol from the experiments rather than from Section 3.4 would evaluate on train seeds.
**Suggestion**: Either rerun the headline experiments on the canonical held-out band (or the reserved eval namespace) or state explicitly in Section 5's preamble which band the evidence seeds come from and why that is immaterial for parameter-free baselines; reword the F1 sentence ("survives every evaluation seed" or similar).
**Severity**: Minor
**Confidence**: 5 — core expertise: Procgen-style seed-split protocols

### W4: Bootstrap parameters claimed in text are not recorded in the committed evidence or the committed script
**Problem**: Section 5.1 states "seed-paired bootstrap 95% confidence interval on the deflated Sharpe (2,000 resamples)," but `make-f1-baselines.py` passes only `confidence=True` and the evidence config block records neither the resample count nor the bootstrap seed. The 2,000 figure lives in library defaults outside the paper's committed command surface, despite the paper's own standard that every number is recomputable from committed commands and evidence.
**Evidence Anchor**: `dataset: paper/evidence/f1-baselines.json — config records n_symbols, n_days, seeds, tiers; no bootstrap parameters`
**Why it matters**: If a future package version changes the default resample count or bootstrap RNG, the committed script will silently produce different CIs while still matching its own config block, weakening the paper's recompute guarantee exactly where it is advertised.
**Suggestion**: Pass the resample count and bootstrap seed explicitly in the script and serialize them into the evidence config.
**Severity**: Minor
**Confidence**: 4 — core expertise: reproducibility engineering; library internals not fully inspected

### W5: F5's "unprofitable everywhere" is a 6-point-per-axis, 8-seed statement, and the size-response table silently omits a sampled grid point
**Problem**: Each boundary sweep samples 6 coefficient values per axis over 8 seeds, and the conclusion "no profitability boundary exists" is a statement about those 18 grid points (plus 5 push weights). Table 3's right panel reports push weights 0.10, 0.25, 0.50, 1.00 but the committed sweep also sampled 0.75 (impact PnL -3.07e-4), which the table omits without note. Additionally no dispersion across the 8 seeds is reported anywhere in F5.
**Evidence Anchor**: `dataset: paper/evidence/f5-manipulation.json — size_response.push_weights [0.1, 0.25, 0.5, 0.75, 1.0] versus Table 3 rows 0.10/0.25/0.50/1.00`
**Why it matters**: The omitted point is directionally consistent, so nothing is hidden that would change the verdict, but a table that presents a subset of a committed grid without saying so invites exactly the selective-reporting suspicion the paper's evidence discipline is designed to preclude. The coarse grid also bounds how strongly "everywhere" should be read; the limitations section acknowledges the model is stylized but not the grid resolution.
**Suggestion**: Include the 0.75 row (or footnote the omission), state the per-axis grid explicitly in the F5 text, and add per-point seed dispersion.
**Severity**: Minor
**Confidence**: 5 — direct comparison of committed evidence against the table

### W6: The vol-clustering re-certification tunes and certifies on the same 8 seeds at a single undefended strength
**Problem**: The F4 remediation adds a `vol_clustering` knob, picks strength 0.5 with no reported sensitivity sweep, and re-certifies on the identical seeds 0 through 7 used to diagnose the failure. The reported improvement (Calm autocorrelation 5/8 to 8/8, Hard 1/8 to 8/8, Hard pass rate 0.000 to 0.250) is therefore an in-sample statement about the diagnosis panel.
**Evidence Anchor**: `dataset: paper/evidence/f4-realism.json — config vol_clustering 0.5, seeds [0..7] shared by tiers and clustered_tiers blocks`
**Why it matters**: Mild circularity: a knob introduced in response to a failed certification, evaluated at one strength on the certifying seeds, overstates how settled the fix is. The paper is careful to keep the leaderboard at strength 0 and to say the finding stands, which limits the damage, but the re-certification numbers would not survive a fresh-seed check if the improvement were seed-specific.
**Suggestion**: Re-certify the clustered generator on a disjoint seed set and report at 2 or 3 strengths, or label the current re-certification explicitly as in-sample on the diagnosis panel.
**Severity**: Minor
**Confidence**: 4 — core expertise: evaluation-set hygiene; the generator internals not audited

### W7: The F6 maker decomposition is a single episode reported at three-decimal precision
**Problem**: The subsidised-maker narrative (maker_1: 92 fills, toxic-fill rate 0.533, meta fills losing 0.197 per unit on 62% of quantity) comes from one detail episode at seed 0, while the aggregate comparison uses 24 episodes. The single-episode numbers are presented alongside the aggregate results with equal typographic authority.
**Evidence Anchor**: `dataset: paper/evidence/f6-adverse-selection.json — detail_episode_makers computed from detail_seed 0 only`
**Why it matters**: The decomposition identity (spread capture plus adverse drift equals markout) is exact and fine to illustrate on one episode, but the quantitative pattern claims (which maker is subsidised, the toxic-fill rates) are anecdotal at N=1 and could invert on another seed.
**Suggestion**: Either aggregate the maker decomposition over the same 24 episodes or label the paragraph explicitly as a single-episode illustration of the decomposition identity rather than a finding.
**Severity**: Minor
**Confidence**: 4 — core expertise: evaluation methodology; microstructure interpretation adjacent

---

## Detailed Comments

### Title & Abstract
The abstract's empirical claims all trace to evidence (checked). Two abstract-level phrasings inherit weaknesses above: the ecology-winner replacement (single seed, W1) and the implicit strength of "no baseline is rank-eligible" (train-band seeds, W3).

### Methodology / Research Design
- **Design type**: computational benchmark/environment paper with a descriptive calibration study. The eight experiments are correctly framed as instrument calibration, not as agent evaluation. The design matches the questions asked, with the sample-size caveats in W1 and W2.
- **Determinism claims**: verified as tested, with correctly disclosed scope (golden hashes and parity cover the Rust core and bindings; the Python probe layer that produced F4 through F8 is seeded-deterministic but outside the byte-identity guarantee, and the limitations say so). The transcendental-free claim is scoped to the scenario transform; the Avellaneda-Stoikov optimum uses ln by necessity, which is outside that scope and consistent with the paper's wording.
- **Seed-split methodology**: structurally sound (disjoint bounded intervals, gap, refusal of unbounded train intervals, code-level disjointness guard), but the paper's own evidence run does not model the protocol it fixes (W3).
- **Baselines**: trivial policies plus two behavioral counterparties, with no trained agents. For this paper's calibration purpose that is adequate and is candidly disclosed in the limitations ("they do not establish what a trained agent scores"; the F3 near-zero within-tier gaps are correctly labeled a control, not a robustness result). The consequence the authors should keep visible: none of F1, F3, F7 bounds trained-agent behavior, and the "honest zero" framing is the right one.

### Analysis Methods
- Only F1 has inferential statistics. See W2. Effect direction is consistently reported; effect magnitude is; uncertainty almost never is.
- The regret metric's zero-point check (optimum vs itself equals 0.0) is true by construction and validates plumbing, not the metric; the text slightly oversells it as validation.
- F7 is purely descriptive counts over a full factorial (8 policies x 16 seeds x 3 tiers) and is fine as reported; per-policy claims (flat 48/48 clean, overconfident 7/16 drawdown breaches on Calm, 4 of 5 stop-outs are max_sharpe) all verify against the episode-level evidence.

### Results Presentation
Tables match evidence exactly (verified for Tables 1 through 6). One silent grid-subset presentation (W5). Figures are regenerated from committed evidence by a committed script, which is best practice.

### Reproducibility
Strong: committed scripts, committed evidence, pinned package version recorded in the evidence, public-API-only scripts, no-LLM-in-the-loop statement. Two gaps: bootstrap parameters not committed (W4), and per-episode vectors missing from F2/F6 evidence (part of W2). I did not execute `cargo test` or the paper scripts end to end; my verification covers evidence-to-manuscript consistency and source-level existence of the claimed tests.

### Methodological Fallacies Checklist
- Selective reporting: one instance at the presentation layer (W5's omitted grid point), directionally immaterial.
- Overfitting/circularity: mild, in the F4 re-certification loop (W6).
- Small-N over-generalization: F8 (W1), F6 detail episode (W7).
- No p-hacking surface exists: the paper reports no significance tests at all, which is itself the W2 gap rather than a red flag of the usual kind.

### Statistical reporting adequacy (criterion-bound, Step 4a)
- Descriptive statistics: PARTLY_COMPLETE (means everywhere; dispersion only in F1).
- Effect sizes: MEETS in spirit (the reported quantities are themselves effect magnitudes in native units: DSR gaps, regret, per-unit markout); no standardized effect sizes are needed for this design.
- Confidence intervals: DOES_NOT_MEET outside F1 (W2).
- Power/precision justification: MISSING; no rationale is given for 8 vs 16 vs 24 units per experiment, and the one precision calibration the paper computes (0.34 DSR band luck at 16 seeds) is not fed back into design or interpretation.
- Assumption testing / missing data: NOT_APPLICABLE (simulated, complete data).
- Arithmetic recompute (bounded procedures): the manuscript reports no test statistics, p-values, degrees of freedom, or discrete-scale means with N, so none of `p_from_test_statistic`, `grim`, `grimmer`, `n_from_df` applies; checked all tables and running text of Section 5. In place of those procedures I performed direct evidence-to-manuscript recomputation of every reported value against `paper/evidence/*.json`; all values checked are consistent (representative checks: F1 Calm kelly_vol_target DSR 1.0000 / pass 0.75 and CI [0.9798, 1.0000]; F2 regret 84.6358 at 0.05 and 0.0374 at 0.5; F3 Hard gap -0.33879 -> -0.3388 and Calm-to-Extreme transfer 0.973; F4 Hard mean kurtosis 16.30 with 8/8; F5 lambda endpoint -31.7e-4; F6 h=20 gap 0.4362; F7 Extreme counts 50/5/31/32/10/0; F8 final shares 0.81 and 0.672).

---

## Questions for Authors

1. F8: How stable is the winner-replacement result across replicator seeds? Please report the replacement frequency over at least 8 seeds per schedule, or scope the claim to the single committed run.
2. F3: Given your own calibration that 16-seed band luck reaches 0.34 DSR, which cells of the transfer matrix do you regard as statistically distinguishable from band noise, and can you attach per-cell paired uncertainty to justify that partition?
3. Section 3.4 vs Section 5: Why does the evidence run evaluate on train-band seeds [0, 16) rather than the canonical held-out band or the reserved 1e6 evaluation namespace, and can the protocol text state the intended practice for external entrants unambiguously?
4. F6: Does the h=1 informed-vs-uninformed gap of 0.072 per unit exclude zero under a paired bootstrap over the 24 episodes?

---

## Minor Issues

### Figures and Tables
- Table 3 (right): add the push weight 0.75 row present in the committed sweep, or footnote its omission.
- Table 2 (right matrix): consider marking the diagonal "0 by construction" cells and the sub-noise cells typographically differently from the interpreted cells.

### Layout / Text
- Section 5.1: "survives every held-out seed" should be reworded to match the seeds actually used (see W3).
- Section 5.2: "validates the metric's zero point" overstates a by-construction identity; "confirms the plumbing" is closer.
- Appendix A: record the bootstrap resample count and seed in the F1 command surface (see W4).

---

## Criterion-Bound Judgements

Calibration status: `NOT_CALIBRATED`

| Dimension | Criterion source | Judgement | Evidence anchor(s) | Rationale | Uncertainty / scope limit | Decision bearing? |
|---|---|---|---|---|---|---|
| Methodological Rigor | Reviewer 1 remit; Step 2/4 of review protocol | PARTLY_MEETS | dataset: paper/evidence/f8-ecology.json seed 0; text: §5.3 noise calibration vs transfer matrix | Designs are sound (paired controls, disjoint bands) but N=1 ecology and absent uncertainty outside F1 leave several interpreted quantities unsupported | Did not execute the full test suites; source-level verification only | yes — W1 and W2 drive the Major Revision |
| Evidence Sufficiency | Step 5 (results integrity) | MEETS | dataset: paper/evidence/*.json vs Tables 1-6 | Every reported number traces to committed evidence; verified by independent recomputation | One presentation-layer grid omission (W5) | yes — supports repairability, argues against Reject |
| Reproducibility | Step 6 | MEETS | dataset: scenario_gen.rs golden hashes; npm smoke tamper test; §8 | Committed scripts, evidence, pinned versions, tested determinism | Bootstrap params uncommitted (W4); Python probe layer outside byte-identity, disclosed | yes — a core claim of the paper, and it holds |
| Statistical Reporting Adequacy | references/statistical_reporting_standards.md §1.3, §1.5 | PARTLY_MEETS | absence: §5.2-§5.8 — expected dispersion/CI reporting for F2, F3, F5, F6 point estimates; checked all Section 5 tables, running text, and committed evidence JSON | CIs exist only for F1; no precision justification for 8/16/24 unit counts | Native-unit effect magnitudes are reported throughout, which partially compensates | yes — W2 |
| Conclusions within data support | Step 5 | PARTLY_MEETS | text: abstract, "replaces the calm-regime ecology winner with the unlevered long book" | Most conclusions are carefully scoped and the limitations are candid; the F8 abstract claim and the F1 "held-out" wording exceed the committed evidence | none identified | yes — W1, W3 |
| Originality | outside methodology remit | NOT_ASSESSED | — | assigned to other reviewers | not my seat | no |
| Literature Integration | outside methodology remit | NOT_ASSESSED | — | assigned to Reviewer 2 | not my seat | no |
| Writing Quality | outside methodology remit | NOT_ASSESSED | — | assigned to other reviewers | not my seat | no |

Recommendation rationale: the unresolved decision-bearing criteria are Methodological Rigor and Statistical Reporting Adequacy, and both are repairable with reruns (F8 seed batch) and added dispersion reporting from data the authors already generate. The verified traceability and reproducibility make this a Major Revision, not a rejection: the infrastructure claims hold; the statistical layer over the eight findings does not yet match the rigor of the engineering layer beneath them.
