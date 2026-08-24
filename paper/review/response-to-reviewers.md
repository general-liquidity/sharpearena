# Response to Reviewers: SharpeArena (Round 1 Revision)

> **Historical response.** This letter records what changed in Round 1. Later
> reviewer-motivated experiments and corrections supersede statements here
> about features that did not yet exist; the current manuscript and integrity
> report are authoritative.

Manuscript: `paper/main.tex`, "SharpeArena: The Point-in-Time (PIT) RL Environment for Trading Agents"
Decision responded to: `editorial-decision.md` (Major Revision, 2026-08-24), roadmap `revision-roadmap.md` (R-01 through R-33).

We thank the panel. The revision follows the editorial arbitrations exactly where they were given. Summary of what changed: the abstract and introduction now frame all eight findings as instrument calibrations on tape the realism gate does not certify at the canonical configuration (C1); every leak-freedom claim is scoped to the interface boundary and a limitations paragraph names the generator-inversion question (C2); the trading-RL related-work lineage is added; "provable optimum" is renamed throughout; F5 is reframed as a theory-consistency check under linear impact; F8 was rerun over 8 replicator seeds per schedule and its single-seed abstract claim is retracted and replaced by the replicated distribution; F2, F3, F5 and F6 now carry committed per-episode/per-seed dispersion and CIs; and the companion citation carries a locator.

One honest headline from the new experiments deserves the panel's attention before the item-by-item list: the F8 replication overturned the manuscript's own single-seed narrative. Across 8 seeds per schedule, the shocked-schedule winner is a `kelly_vol_target` lineage on 7 of 8 seeds; the seed-0 dethronement by `equal_weight_long` that the previous abstract promoted occurs on exactly 1 of 8 seeds, and the control winner itself varies by seed (4/8 kelly, 2/8 equal-weight, 2/8 unresolved). Section 5.8 now reports the winner distributions and states the methodological conclusion: single-run ecology narratives do not survive replication. R1 and R3 were right to insist on this run.

Statuses used below: ADDRESSED (change made as required), ADDRESSED-VARIANT (item discharged through the branch the editorial decision or operator constraints selected), PARTIAL (part done, remainder stated), DEFERRED (not done, rationale given).

---

## Required revisions (must fix)

### R-01 / roadmap R-01: Learning experiment, or narrowed claims. ADDRESSED-VARIANT (rescoping branch)

We took the rescoping branch R0 offered. No learning run is added at this revision (compute is the stated constraint). Instead: the introduction's new "Scope and contribution boundary" paragraph states explicitly that no agent is trained, that the results calibrate instruments rather than demonstrate learned performance, and that the claims are environment-and-protocol; the limitations section's "Named future experiments" paragraph commits to a PPO baseline through the Gymnasium adapter as follow-up (1); and per DA M1 the same paragraph commits to the rank-eligibility existence run (4), with F1's text now stating outright that no policy of any kind has been shown rank-eligible and that the non-emptiness of the eligibility set is unestablished. The title is retained at the operator's explicit instruction; the "The" concern is addressed in text by scoping the claims (see S1).

### R-02 / roadmap R-02: Trading-RL related-work positioning. ADDRESSED

Section 6 gains a "Financial RL environments" paragraph positioning FinRL, FinRL-Meta, ABIDES-Gym, TradeMaster and mbt_gym on the paper's axes (leak-freedom mechanism, determinism guarantee, trajectory verifiability, external-agent contract, generalization protocol). Each system's properties were verified against its paper/repository before writing (FinRL arXiv 2011.09607, NeurIPS 2020 Deep RL workshop; FinRL-Meta NeurIPS 2022 D&B, arXiv 2211.03107; ABIDES-Gym ICAIF 2021, arXiv 2110.14771, Amrouni et al.; TradeMaster NeurIPS 2023 D&B, Sun et al.; mbt_gym ICAIF 2023, arXiv 2209.07823, Jerome et al.). Where a property is our reading rather than an authorial claim, the paragraph says so and stays descriptive ("to our knowledge"). mbt_gym is additionally acknowledged at the market-making task itself (Section 3.5), as R2 requested. All five bib entries added with verified metadata.

### R-03 / roadmap R-03: Companion contribution boundary and finding order. ADDRESSED

The introduction gains an explicit "Scope and contribution boundary" paragraph: this paper claims the environment and protocol; the scoring semantics and the unit bug are the companion's; F1 appears here only as validation that the producer-scorer separation catches scorer bugs. The abstract now leads with environment-owned results (F2 regret, F3 transfer, F5 consistency, F8 replication) and contains no kernel-owned number; "What the evidence shows" is reordered the same way, with F1 moved to last and introduced as pipeline validation. F1's section is retitled "Pipeline validation under the corrected kernel" and its text attributes the unit-bug discovery to the companion in so many words.

### R-04 / roadmap R-04: Stale Section 2 status paragraph. ADDRESSED

The final paragraph of Section 2 is rewritten in past tense: the eight findings were produced by the committed evidence run at sharpearena 0.8.0 over SharpeBench 0.5.0, and they calibrate instruments rather than certify market properties. No sentence in the revision leaves the execution status ambiguous.

### R-05 / roadmap R-08: F8 multi-seed replication. ADDRESSED (run executed)

`paper/src/make-f8-ecology.py` extended to 8 replicator seeds per schedule (seed 0 kept as the figures' detail run); `f8-ecology.json` regenerated with per-seed winners, outcome counts and winner distributions under a `multi_seed` key. Results: root-level winner differs between schedules on 5/8 seeds; shocked winner is a kelly_vol_target lineage on 7/8 seeds; the seed-0 replacement by equal_weight_long occurs on 1/8; control winners split 4/2/2 (kelly / equal-weight / unresolved-dominance). The F8 table (tab:f8) now reports the distribution instead of the two single-run rows; the abstract's ecology sentence is replaced by the replicated statement (which is a retraction of the old claim, stated as such). The ecology-literature paragraph R3 requested (Farmer 2002; Lux-Marchesi 1999) is added to Section 5.8 with the stated abstractions (no flow lags, no leverage constraints, single-innovator entry). Varied shock orderings and path-matched schedule pairs are named as mechanical extensions; note the current schedules chain episode seeds differently, which the text now discloses.

### R-06 / roadmap R-09: Uncertainty quantification for F2/F3/F5/F6. ADDRESSED (runs executed)

All four scripts extended to serialize per-episode/per-seed vectors and report 95% intervals, with each file recording its CI convention:
- F2: per-episode paired regret vectors per half-spread; t-based CIs in text and figure. The minimum (0.037 at half-spread 0.5) has CI [-8.7, 8.8]: statistically indistinguishable from the reference, which the text now says, replacing the "comes within 0.04" narration.
- F3: per-seed return series per band and per matrix cell; seed-resampled percentile bootstrap (B=2000), seed-paired across regimes for the transfer cells; band DSR CIs and per-cell gap CIs reported. Calm-to-Extreme (0.973) excludes zero decisively (CI [0.635, 0.999]); Calm-to-Hard (0.877) is positive throughout but wide ([0.054, 0.999]); the Hard-to-Extreme cell (0.096, CI [-0.218, 0.911]) is explicitly marked below the paper's own 0.34 noise calibration, straddles zero, and is not interpreted, in text, table caption and reading. The text also states the meta-result: intervals this wide are a calibration of what a 16-seed board can support.
- F5: per-seed impact PnL at every grid point of all three axes and the size response; t-based CIs; every CI excludes zero, which is now stated.
- F6: per-episode markout-per-unit vectors for both legs and paired gaps; t-based CIs. R1's Q4 is answered in the table and text: the h=1 gap of 0.072 has CI [0.052, 0.094] and excludes zero.
Where an API returned only aggregates, the scripts recompute the per-unit quantities through the same public entry points and assert agreement with the API aggregates before serializing.

### R-07 / roadmap R-15: "Provable optimum" reframing. ADDRESSED

"Provable optimum" appears nowhere in the revision. Section 3.5 (retitled "The task suite and the closed-form reference policy") states the closed form is the source model's asymptotic approximation, cites Gueant, Lehalle and Fernandez-Tapia (2013) as the exact treatment (bib entry verified: Mathematics and Financial Economics 7(4):477-507), and states that no optimality proof is offered for the implemented reward (which adds an inventory cap, running penalty and terminal charge). This answers R2's Q1: we do not claim exact optimality, we renamed instead. The zero-regret self-check is described as "confirms the plumbing" (R1's wording), and the model-boundedness caveat notes the no-adverse-selection fill assumption and its tension with F6 (R3 W6). Abstract, contributions and F2 all use "closed-form reference policy" / "closed-form Avellaneda-Stoikov reference".

### R-08 / roadmap R-16: Companion citation locator. ADDRESSED

`refs.bib` entry `toca2026sharpebench` now carries the repository locator (github.com/general-liquidity/sharpebench, version 0.5.0). Additionally, F1's text now contains the one-line unit-bug derivation (annualized prior applied per period; corrected by converting once at 252) so the claim is checkable standalone even before an arXiv identifier exists.

### R-09 / roadmap R-22: F5 theory engagement and ablation. PARTIAL (text complete; ablation named, not run)

Text leg: F5 opens with the theory paragraph, citing Huberman-Stanzl (2004; Econometrica 72(4):1247-1275, linearity rules out manipulation quasi-arbitrage) and Gatheral (2010; Quantitative Finance 10(7):749-759), and states that the clean sweep is the expected outcome under linear permanent impact, so F5 is a consistency check of the implementation against theory, not a certification. "Certifies" is removed from F5 and from the abstract's manipulation sentence. The probe's missing positive control is stated in the section itself.
Experiment leg: the market implements strictly linear impact; no concave or transient-impact parameter exists in `market.rs` (verified by inspection before deciding), so the ablation requires new engine code, out of scope for a text-window revision. It is committed to by name in the limitations ("Named future experiments" item 3, square-root per Almgren et al. 2005 or transient decay per Gatheral 2010), including its role as the DA's positive control. We accept that until it runs, F5 carries the reduced claim the revision now states.

### R-10 / roadmap R-23: F6 level versus gap. ADDRESSED (text branch)

Section 5.6 now separates the two certifications explicitly. The gap result is reported with its paired CIs and scoped to the differential cost of informed flow. The level result (positive informed markouts at every horizon) is reported against the environment with F4-grade wording ("goes against the environment... on the wrong side of the Section 2 standard"). R3's Q2 is answered: the cause is calibration and design, not accounting; fills struck at the pre-move mid book full quoted depth as spread capture, default depths are wide relative to the drift alpha=2.0 realizes, and the exogenous path means quotes never widen with toxicity as in Glosten-Milgrom (1985, now cited at the probe's introduction and in the discussion); it is not an intended weak-alpha certification. Recalibration and an endogenous-quote variant are the named follow-ups. The recalibration run itself is the optional branch and was not run.

### R-11 / roadmap R-27: C1 resolution (uncertified tape). ADDRESSED

- Abstract: "They are calibrations, not market findings: the suite's own realism gate fails the canonical configuration's tape on 23 of 24 certification runs, an opt-in volatility-clustering driver is the partial remediation, and certification is reported beside the results rather than required by them."
- Introduction ("What the evidence shows" first sentence): the findings are instrument calibrations on tape the gate declines to certify; "that framing governs every number below."
- Where the gate is introduced (Section 3.7): the gate's current role is stated plainly (reported, not a leaderboard precondition; canonical configuration currently fails), the intended endgame is stated (a passing generator revision or certification-conditioned leaderboard claims), and the verifiability-first sequencing is argued (a realistic but unverifiable tape cannot anchor a leaderboard; a verifiable, admittedly unrealistic one can still calibrate instruments).
- Section 5 preamble: first of three scoping statements repeats the certification status over the whole experimental section; Section 5.4 adds that every other experiment runs on the uncertified default tape.
The optional robustness rerun at vol_clustering=0.5 was not run (see Deferred).

### R-12 / roadmap R-28: C2 resolution (leak-freedom scope). ADDRESSED

- Abstract: "impossible by construction" is gone; the claim reads "excluded at the environment's own interface, and policed beyond it" (R2 W7's proposed scoping).
- Section 2, first principle (retitled "Leak-free at the interface"): explicitly an interface-shape guarantee, not an information guarantee, with a pointer to the limitations.
- Section 3.1: "it bounds what can be requested, not what can be inferred from the trailing window a request returns." Section 3.2: the generator's openness means future bars are in principle computable from past bars; held-out evaluation rests on seed bands, not secrecy.
- Limitations: new paragraph "The interface guarantee is not an information guarantee" states the inversion attack precisely (public four-operation transform, path unresponsive to actions on the canonical task, an agent that inverts the transform predicts future bars in-band without any forbidden call), states exactly what seed bands do and do not defend against (seed lookup yes, inversion no, because inversion needs no seed), and discloses that no bound or experiment is offered. The predictability probe (next-bar regressor vs persistence baseline, train band to held-out band) is named future experiment (2).
We did not overcorrect: the interface guarantee is still claimed as real and test-enforced, in those words.

---

## Suggested revisions (should fix / consider)

### S1: Abstract length, title article, register. PARTIAL

Abstract reduced from ~420 to ~300 words and restructured (three structural properties, contract and protocol, environment-owned findings); companion-keyed apparatus (deflated-Sharpe saturation narrative, "mis-united floor", eligibility conjunctions) removed; "mis-united" is gone; PIT expanded once. The title's "The" is retained at the operator's explicit instruction; the overclaim concern is addressed in text: the introduction's scope paragraph and the limitations state that adoption has not begun and claims are environment-and-protocol. We did not reach a strict halving because the required C1/C2 caveats (uncertified tape, interface scoping, coverage split) now occupy abstract space the roadmap itself mandates.

### S2: Significance register. ADDRESSED

"The strategic bet is interface ownership" is rewritten (the interface follows the Gym pattern; whether an ecosystem forms is an adoption question the paper does not answer). "Whoever defines the interface owns the ecosystem" is replaced with an implementer-facing stability rationale. The limitations paragraph on external entrants now closes with: the significance claim is the verifiability infrastructure checkable today; adoption is future work, not an assumption.

### S3: Seed provenance wording. ADDRESSED

Section 5 preamble (third scoping statement): all evidence seeds come from the train band [0,16), the canonical held-out band is never touched, and why that is immaterial for parameter-free policies (nothing is fitted, so nothing can leak). F1's "survives every held-out seed" reworded to "survives every evaluation seed".

### S4: Bootstrap parameters committed. ADDRESSED

`make-f1-baselines.py` now passes `n_boot=2000` and `resample_seed=0x5BA7_2026` explicitly and serializes them (with alpha and the convention) into the evidence config; F1 was rerun and the numbers are unchanged (the explicit values equal the library defaults). Section 5 preamble states B, alpha, the percentile convention and the resample seed; Appendix A names the defining source file (`sharpearena/confidence.py`).

### S5: F5 grid completeness. ADDRESSED

The 0.75 push-weight row is added to tab:f5 (impact PnL -3.07e-4, CI [-3.84, -2.30]e-4); the text states the grid explicitly (6 points per axis, 5 size points, 8 seeds); per-point seed dispersion added per R-06.

### S6: Vol-clustering re-certification hygiene. ADDRESSED (text branch)

Section 5.4 now labels the strength-0.5 re-certification as tuned and re-certified in-sample on the same eight diagnosis seeds, and names disjoint-seed multi-strength certification as the required follow-up before the driver can claim the fact generally.

### S7: F6 decomposition scope and units. ADDRESSED (labeling branch)

The decomposition paragraph is introduced as "illustrates the exact identity on one seeded episode (seed 0); it is an illustration, not an aggregate finding." Units defined: markout in the tape's absolute price units (mid starts at 100) per unit of filled quantity, stated in the section preamble and tab:f6 caption; tab:f5's notional base defined in its caption (fraction of capital per episode). The 24-episode aggregation of the decomposition was not run (optional branch).

### S8: Cont citation split. ADDRESSED

Section 3.7 and Section 5.4 now attribute the classical facts to Cont (2001), the timescale asymmetry to Zumbach (2009; Quantitative Finance 9(5):505-515, verified), and present the Fano factor explicitly as the authors' own diagnostic with its construction (variance-to-mean of large-move exceedance counts per window; 1 under Poisson) and bound rationale stated.

### S9: Probe foundational citations. ADDRESSED

Added at the probes' introductions in Section 3.7: Glosten-Milgrom (1985) for adverse selection (with the exogenous-path scoping), Allen-Gale (1992) for trade-based manipulation, Farmer (2002) and Lux-Marchesi (1999) for ecology (also cited in Section 5.8 and the contributions list). Huberman-Stanzl and Gatheral are at F5 per R-09; Budish et al. at the clearing mechanism per S11.

### S10: "Dominant failure mode" superlative. ADDRESSED

Now "a pervasive and usually invisible failure mode of backtests, alongside the selection bias the deflation literature centers", with the Bailey-Lopez de Prado citation carrying the selection-bias half. The ranking is dropped.

### S11: Batch-clearing scope and fees. ADDRESSED

New paragraph in Section 3.5: the single-price-per-step clearing and call-auction uncross are named as frequent-batch-auction-like (Budish, Cramton and Shim 2015, QJE 130(4):1547-1621, verified); the excluded margins are listed (latency races, queue-position value, sniping); and fees are answered (none modeled; every Sharpe, regret and markout is gross of fees), which answers R3's Q4. The FBA-laboratory framing is used.

### S12: Broader impact. ADDRESSED

New "Broader impact" paragraph in the limitations: an explicit norm that SharpeArena scores are claims about a synthetic environment and not performance marketing (with the persuasiveness-vs-validity distinction R3 drew), and the argued case for publishing the manipulation probe (theory-guaranteed losses under the shipped specification, diagnostic-only supported use, in-code disclaimer, legible schedules).

### S13: Golden-hash coverage scoping. ADDRESSED

Stated where the guarantee is advertised: Section 4.4 now bounds "remove the host from the trust equation" with the coverage split (Rust core, bindings and F1-F3 engine outputs to the byte; F4-F8 probe layer seeded-deterministic on one platform, outside the golden-hash path), and the Section 5 preamble repeats it as the second scoping statement. The abstract also carries the split in one clause.

### S14: F3 framing for untrained policies. ADDRESSED

"Training regime" language removed; the matrix uses "source regime" (rows relabeled in table caption and text). The section states that the policy is fixed and parameter-free, that the matrix reduces to pairwise differences of three per-tier DSRs, and that F3 measures the instrument, not transfer learning. Sub-noise cell marking per R-06.

### S15: F7 mechanism caveat. ADDRESSED

Section 5.7 now opens with the mechanism statement: mandates are sampled and assigned to policies that cannot observe them, structural breaches are guaranteed by the draw, the flat 36/36/31 counts are that mechanism's signature, and a mandate-aware field could invert the distribution. The headline is scoped to "this mandate-blind reference field", with mandate-aware agents named as the open measurement.

### S16: Calm narration consistency. ADDRESSED

Resolved by choosing the universal-bounds reading: Section 5.4 states the bounds encode what real-market tape looks like, so Calm is plainly not realistic tape under this battery, by design, and the gate does not grade it on a curve. Section 2's probe principle also now points at both current findings against the environment.

### S17: "Provably disjoint" phrasing. ADDRESSED

Section 3.4: "a disjoint test family (two separated integer intervals, with the disjointness asserted in tests)". The abstract says "disjoint" without the adverb.

### S18: Presentation minors. PARTIAL

Done: Table 1 caption states the eligibility rule (pass^k rate exactly 1.00); pass^k markup unified to one form; "subsidised" -> "subsidized"; Section 5.2 "validates the metric's zero point" -> "confirms the plumbing"; byrd2020abides converted to the published SIGSIM-PADS 2020 inproceedings entry.
Not done: Figure 3 / Table 2-right duplication retained (the figure is the visual aid, the table carries the CIs; we judge they now carry different content); minari version pin (the adapter deliberately does not pin a minari version, so the citation keeps the project URL and year); Gymnasium `check_env` citation at first functional mention (check_env is already credited in Section 6); the optional closing restatement paragraph.

---

## Answers to the reviewer question lists

- R0 Q1-Q4 (venue fit, learning experiment, companion boundary, status paragraph): answered by R-01 (rescoping branch, with the learning run committed by name), R-03 and R-04 above.
- R1 Q1 (F8 replication): run; distributions reported in Section 5.8 and above. R1 Q2 (dispersion): added per R-06. R1 Q3 (seed provenance): Section 5 preamble, S3. R1 Q4 (does the h=1 gap exclude zero): yes; CI [0.052, 0.094] over per-episode paired gaps.
- R2 Q1 (is the AS closed form optimal for the implemented reward): no claim is made; renamed to closed-form reference policy and the approximation status is stated (R-07). R2 Q2 (companion locator): added (R-08). R2 Q3 (related work): added (R-02). R2 Q4 (leak-freedom scope): scoped (R-12).
- R3 Q1 (does linear impact make F5 vacuous): substantially yes, and the section now says so; the concave ablation is the named discriminating experiment (R-09). R3 Q2 (why positive informed markouts): answered in Section 5.6, calibration and design, not intended (R-10). R3 Q3 (ecology single seed): replicated (R-05). R3 Q4 (fees and clearing scope): answered, gross of fees, FBA-like clearing (S11).

---

## Deferred items, with rationale

1. Learning baseline (R-01 experiment branch): compute-constrained; rescoping branch taken per R0's stated alternative; committed by name in limitations.
2. Concave/transient-impact F5 ablation (R-09 experiment leg): no nonlinear impact parameter exists in the engine (verified in `market.rs` before deciding); requires new core code, named future experiment with its positive-control role stated.
3. Rank-eligibility existence run (DA M1 / R-29): text branch taken as the task's arbitration allowed; F1 states the set's non-emptiness is unestablished; run committed by name.
4. Generator-predictability probe (R-12 optional leg): disclosed as open; probe design specified in limitations.
5. vol_clustering=0.5 robustness rerun of F1/F3/F7/F8 (R-11 optional leg): would strengthen the reframed claims but is not required for them; the reframing no longer rests on the certified/uncertified distinction being invisible.
6. F6 recalibration / endogenous-quote variant (R-10 optional leg) and 24-episode decomposition aggregation (S7 optional leg): named follow-ups; the level finding is reported against the environment regardless.

## Change locations (cross-reference)

- Abstract: `sections/00-abstract.tex` (rewritten).
- Introduction: `sections/01-introduction.tex` (rewritten; new scope-boundary paragraph; reordered evidence summary).
- Principles: `sections/02-principles.tex` (P1 retitled and scoped; P4 register; probe principle; final status paragraph rewritten).
- Environment: `sections/03-environment.tex` (3.1 interface scoping; 3.2 generator-openness note; 3.4 disjointness phrasing; 3.5 retitled, reference-policy paragraph, batch-clearing/fees paragraph, mbt_gym; 3.7 gate role, citation splits, probe foundations).
- Contract: `sections/04-contract.tex` (4.4 coverage split).
- Experiments: `sections/05-experiments.tex` (new preamble scoping statements; F1 retitle/attribution/eligibility caveat; F2 reference policy + CIs; F3 source-regime framing + CIs + sub-noise marking; F4 universal-bounds narration + in-sample label; F5 theory + full grid + CIs; F6 level-vs-gap + CIs + units + illustration label; F7 mechanism; F8 replicated rewrite).
- Related work: `sections/06-related.tex` (financial-RL paragraph).
- Limitations: `sections/07-limitations.tex` (inversion paragraph; named future experiments; broader impact; significance scoping).
- Appendix A: `sections/A-commands.tex` (bootstrap parameters and source; F8 multi-seed; dispersion layers).
- Evidence: `paper/evidence/f1-baselines.json`, `f2-regret.json`, `f3-generalization.json`, `f5-manipulation.json`, `f6-adverse-selection.json`, `f8-ecology.json` regenerated by the extended committed scripts in `paper/src/`.
- Bibliography: `refs.bib` (15 entries added/fixed, all metadata verified against sources).
