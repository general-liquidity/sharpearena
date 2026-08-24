# Editorial Decision

## Manuscript Information
- **Title**: SharpeArena: The Point-in-Time (PIT) RL Environment for Trading Agents
- **Manuscript ID**: local manuscript at `paper/main.tex` (sharpearena 0.8.0 / SharpeBench 0.5.0 evidence run)
- **Submission Date**: not recorded (preprint targeting NeurIPS 2026 Evaluations and Datasets track)
- **Decision Date**: 2026-08-24
- **Review Round**: Round 1

## Calibration Resolution

`calibration_status: NOT_CALIBRATED`

Current runtime boundary: this package is not upgraded from any candidate or prose-named profile. No `review-panel-provenance/1.0` artifact was supplied for this panel; all provenance axes are recorded as `unknown` below rather than inferred from persona or configuration.

## Review Panel Provenance (#540/#740)

- **Typed artifact**: NOT AVAILABLE (no provenance artifact was produced for this run)
- **Provenance axes**: role_separated `true` by construction of the five committed seat reports; fresh_context, blind_to_peer_outputs, model_family_distinct, provider_distinct, human_distinct all `unknown`
- **Binary independence claim**: Not computed. Role separation is not a claim of independent error processes; corroboration counts below are reported as corroboration, not as independent replication.
- **Correlated-error disclosure**: Model family per seat is unknown; correlated errors across seats cannot be excluded and no conclusion is implied for the other axes.

---

## Decision

### Major Revision

Re-review is required after revision.

---

## Devil's Advocate CRITICAL Adjudication (visible, per synthesis protocol)

`da_critical_adjudications: [C1=VALIDATED, C2=VALIDATED]`

### C1: The findings rest on tape the realism gate does not certify. VALIDATED.

**DA's argument** (r4-devils-advocate.md, C1): the realism gate fails the canonical configuration on 23 of 24 runs (all-facts pass 0/8 Calm, 0/8 Hard, 1/8 Extreme), the `vol_clustering` fix is explicitly opt-in and not the leaderboard default, yet every substantive finding (F1, F3, F7, F8) is computed on the uncertified tape and nothing downstream is conditioned on certification, while §3.7 states the gate exists so that a drifted generator "fails the certification instead of silently letting agents win on unrealistic markets."

**Editorial verification**: every factual link was checked against the manuscript and evidence. Table `tab:f4` in `sections/05-experiments.tex` shows the 0.000 / 0.000 / 0.125 pass rates; the same section states `vol_clustering = 0` is "the default and the canonical leaderboard configuration"; both R1 and the DA independently verified `paper/evidence/f4-realism.json` matches. §3.7 states the gate's purpose as quoted. The abstract presents the F1/F3/F8 numbers without the certification status attached.

**Corroboration**: R3 W4 raises the same substance from the economics side (headline numbers travel without the realism caveat that §7 attaches to them, severity Minor at that seat); R0 W6 raises the adjacent significance point. R1's verification confirms the numbers but does not dispute the framing critique.

**Adjudication**: VALIDATED as an internal-consistency and framing defect. The numbers themselves are not invalidated; the paper's limitations section states the situation candidly, five sections downstream of where the numbers are marketed. What is validated is the contradiction between the gate's stated purpose and its actual role: at this revision the gate is a reported diagnostic, not a gate, and the abstract and introduction present findings from the uncertified canonical tape at market-finding strength. Resolution is primarily textual: reframe the abstract and introduction so the eight findings are explicitly instrument calibrations on a generator that intentionally fails full realism certification at this revision (R3 W4's exact fix), state explicitly that certification is reported, not required, for the canonical configuration, and either condition leaderboard claims on certification or argue the verifiability-first sequencing the DA's "unexamined premise" section identifies. Roadmap items R-27, R-22. **This validated CRITICAL blocks Accept.**

### C2: Leak-free-by-construction is an interface claim presented at information-claim strength; generator inversion is unanalyzed. VALIDATED.

**DA's argument** (r4-devils-advocate.md, C2): scenarios are pure functions of a public 64-bit seed computed by an open generator restricted to multiply, add, divide and max; on the canonical task the price path does not respond to the agent's actions; an agent that learns or inverts the low-complexity transform from the trailing window predicts future bars exactly without calling any forbidden API. Held-out seed bands block lookup by seed identity, not inversion. The paper offers no analysis, bound, or experiment on this in-band attack, yet the abstract claims the failure classes are "impossible by construction."

**Editorial verification**: the textual basis is confirmed. Abstract: "impossible by construction rather than forbidden by rule." §2: "An agent cannot peek because there is nothing to call." §3.2: "A scenario is a pure function of one 64-bit seed" with the four-operation transform stated. The §7 side-channel paragraph covers only out-of-band channels (wall clock, filesystem, network); no passage in §3, §6, or §7 analyzes seed or state inference from the observed window. The absence claim is verified.

**Corroboration**: R2 W7 independently flags the abstract's "impossible by construction" as outrunning what §7 bounds (on different grounds: the deny-list and side channels); the ALE/memorization discussion in §6 (praised by R2 S2) addresses memorization via seed bands but not inversion.

**Adjudication**: VALIDATED as an overclaim at the stated strength. The DA has not demonstrated the attack (confidence 4, feasibility argued from generator structure), so the practical severity of inversion is open; but the burden of "impossible" sits with the paper, and the claim as written conflates an API-shape guarantee (real, test-enforced, verified by R1) with an information guarantee the paper never establishes. Resolution is textual plus one honest limitation: scope "leak-free" explicitly to the interface (no operation returns a post-cursor bar), add a limitations paragraph on deterministic-generator predictability and what seed bands do and do not defend against, and, as a should-fix, report a simple predictability probe. Roadmap items R-28, R-24. **This validated CRITICAL blocks Accept.**

Both CRITICALs are validated; per the decision standards a validated DA-CRITICAL blocks Accept. This is consistent with the unanimous seat recommendations below.

---

## Blocking Issues (3, immutable source order)

| Transport ref | Blocking issue | Source reviewer(s) | Evidence anchor | Resolving roadmap item |
|---|---|---|---|---|
| B1 | No learning-based experiment in a paper titled an RL environment; the environment's core instruments are calibrated only on fixed policies, and the venue's demonstrated-utility expectation is unmet unless claims and title are narrowed | R0 (W1); adjacent: DA M1 | `text: §7 "they do not establish what a trained agent scores"` | R-01 |
| B2 | Every substantive finding is computed on tape the paper's own realism gate refuses to certify (1/24), with nothing downstream conditioned on certification and the abstract presenting the numbers at market-finding strength | DA (C1); corroborated R3 (W4), R0 (W6) | `table: tab:f4; text: §5.4 "the default and the canonical leaderboard configuration"` | R-27 (with R-22) |
| B3 | "Leak-free / impossible by construction" is an interface-shape guarantee presented as an information guarantee; in-band prediction of the public deterministic generator is unanalyzed | DA (C2); corroborated R2 (W7) | `text: abstract "impossible by construction rather than forbidden by rule"; absence: §3/§7, no inversion analysis` | R-28 |

---

## Reviewer Summary

| Reviewer | Role | Recommendation | Confidence |
|---|---|---|---|
| Journal-Fit Reviewer (R0) | NeurIPS D&B senior area chair persona | Major Revision | 4 |
| Reviewer 1 | RL evaluation methodology | Major Revision | 4 |
| Reviewer 2 | RL environments and benchmarks domain | Major Revision | 5 |
| Reviewer 3 | Market-microstructure economics (perspective) | Major Revision | 4 |
| Devil's Advocate | Fixed adversarial seat | N/A, findings only | N/A, per-finding only |

Confidence is a self-reported scope disclosure only; it carried no weight in any consensus count or arbitration below.

---

## Consensus Analysis

Consensus is computed over the 4 non-DA seats per sub-claim; DA findings are tracked separately and adjudicated above. Sub-claim IDs (SC-n) are referenced by the Revision Roadmap.

### Points of Agreement

There is no CONSENSUS-4 or CONSENSUS-3 sub-claim: the seats' remits partition the manuscript, so most findings are single-seat or two-seat. The panel is nonetheless unanimous at the recommendation level (4/4 Major Revision), and unanimous in the strengths record: all five seats independently credit the artifact-backed reproducibility discipline, the paired-control experiment designs, and the candor of the F4 self-failure and the limitations section.

**Corroborated findings (2 of 4 seats agree, none disputes):**
- **SC-1** Missing trading-RL environment and market-simulator related work (FinRL/FinRL-Meta, ABIDES-Gym, TradeMaster, mbt_gym): raised by R0 (W2, Major) and R2 (W1, Major). The originality claim is credible to both seats but undemonstrated on the page. Silent: R1, R3 (out of remit).
- **SC-2** F8 ecology finding rests on a single replicator seed per schedule yet reaches the abstract: raised by R1 (W1, Major) and R3 (W3, Major); DA M7 makes the same point. Silent: R0, R2.
- **SC-3** "Provable optimum" mischaracterizes the Avellaneda-Stoikov closed form (approximate in the source model; optimality inside this environment unproven): raised by R2 (W2, Major) and R3 (W6, Minor). Same remedy proposed by both (rename, caveat, cite Gueant, Lehalle and Fernandez-Tapia 2013); the severity difference is recorded, both severities transported, and the roadmap item is driven by R2's Major finding since the phrase appears in headline positions (abstract, contributions, F2). Silent: R0, R1 (R1 notes the ln usage is consistent with the determinism scoping, which is a different sub-claim).
- **SC-4** F6's design and levels: R3 (W2, Major) argues positive informed-flow markouts at every horizon invert the adverse-selection economics the probe certifies, against the paper's own §2 standard; DA M3 corroborates. R1 verified the numbers without disputing the economic reading (silent on the interpretation; its remit was traceability). Silent: R0, R2.
- **SC-5** Abstract overclaims and mispositions: three seats converge on the abstract from different angles: R0 W5 (length, density, definite-article title), R2 W7 ("impossible by construction" outruns §7), R3 W4 (numbers travel without the instrument-calibration framing). Distinct sub-claims, one revision surface.

**Single-reviewer findings retained (no conflict, evaluated against their named criteria):**
- SC-6 No learning-based experiment (R0 W1, Major): decision-bearing under the venue-fit criterion; see Disagreement 1.
- SC-7 Companion-paper contribution boundary; F1-first ordering (R0 W3, Major; DA m2 adjacent).
- SC-8 Stale §2 status paragraph contradicting §5 (R0 W4, Major): verified editorially against `02-principles.tex` line 19 and `05-experiments.tex`; factual.
- SC-9 No uncertainty quantification outside F1; per-episode vectors not committed for F2/F6; transfer-matrix cells below the paper's own 0.34 DSR noise calibration narrated without caveat (R1 W2, Major; DA M4 corroborates the noise-floor arithmetic).
- SC-10 Canonical seed-band protocol defined but never exercised; F1 "held-out seed" wording inaccurate (R1 W3, Minor).
- SC-11 Companion citation has no locator despite carrying the F1 unit-bug claim and the scoring semantics (R2 W3, Major).
- SC-12 F5 clean verdict is near-tautological under linear permanent impact (Huberman-Stanzl 2004, Gatheral 2010); the probe has no failure mode as specified (R3 W1, Major; DA M3's no-positive-control point is the same defect stated as evaluation methodology).
- SC-13 through SC-22: the remaining Minor findings (Cont citation scope, probe-layer foundations, "dominant failure mode" superlative, batch-clearing scope, broader impact, bootstrap parameters, F5 omitted grid row, vol-clustering in-sample re-certification, F6 single-episode decomposition, presentation minors), each carried into the roadmap with its transported severity.

### Points of Disagreement

**Disagreement 1: Are trivial-policy baselines adequate?**
- **R0 view** (W1): an RL-environment paper with zero learning experiments misses the track's demonstrated-utility expectation; primary repairable blocker.
- **R1 view** (Detailed Comments, Baselines): for the paper's calibration purpose the fixed-policy field is adequate and candidly disclosed.
- **Disagreement type**: perspective difference (venue expectation vs internal validity), not an existence dispute.
- **Editor's Resolution**: both hold on their own criteria and they compose rather than conflict. R1's judgement is about what the experiments validly claim; R0's is about what the venue requires. The author must either add one modest learning experiment (satisfying R0 while leaving R1's assessment intact) or narrow the title and claims to environment-and-protocol, which is R0's own stated alternative. Roadmap R-01 carries both branches.
- **Resolution Rationale**: expertise-first; venue fit is R0's remit and calibration validity is R1's, and neither seat contradicts the other inside its remit.

**Disagreement 2: Severity of the "provable optimum" phrasing (SC-3).**
- **R2 view**: Major; a factual overclaim on the flagship calibration instrument, in headline positions.
- **R3 view**: Minor; legitimate inside the frozen-parameter model, needs a model-boundedness caveat.
- **Disagreement type**: severity disagreement, identical remedy.
- **Editor's Resolution**: the roadmap item is must-fix, driven by R2's finding; both transported severities appear on the row.
- **Resolution Rationale**: expertise-first is a tie (both seats are competent here); evidence-first favors R2 because the phrase appears in the abstract and contributions list, where R3's own W4 logic (headline placement raises the cost of an overclaim) applies.

No existence disagreement was found: no seat argued that another seat's finding is not a real problem.

---

## Decision Rationale

All four scoring seats independently recommend Major Revision, which the decision matrix maps directly to Major Revision, and both Devil's Advocate CRITICALs are validated, which independently blocks Accept. The panel's assessment is unusually convergent on both halves of the ledger. On the credit side: the verifiability triad (structural leak-freedom at the interface, byte-determinism with committed golden hashes, replay tamper evidence) is genuinely novel among trading environments (R0 S1, R2 originality MEETS), the number-to-evidence traceability was independently recomputed and holds throughout (R1 S1), and the self-incriminating candor of F4 and the limitations section is credited by every seat. This is why the decision is not Reject: the infrastructure claims hold and every identified defect is repairable.

On the debit side, the defects cluster into four repairable classes. First, framing outruns evidence at the headline layer: the uncertified-tape contradiction (C1), the leak-freedom overclaim (C2), "provable optimum" (SC-3), and the stale §2 status paragraph (SC-8) are all text-level repairs with outsized credibility effect. Second, positioning: the trading-RL environment literature is absent (SC-1) and the companion-paper boundary is blurred (SC-7, SC-11). Third, statistical support: F8 is N=1 (SC-2), and no finding outside F1 carries uncertainty (SC-9); both are cheap reruns of committed scripts. Fourth, probe validity: F5 needs the linear-impact theory engaged and an ablation that could fail (SC-12), and F6 needs the level-versus-gap distinction (SC-4). A less strict decision is unavailable because validated CRITICALs and four Major recommendations stand; a stricter one is unjustified because no defect is architectural. Re-review after revision is required.

---

## Required Revisions (Must Fix)

Transport refs follow immutable roadmap source order (seat order R0, R1, R2, R3, DA; within seat, report order) filtered to `obligation_class == must_fix`. R<n> is a transport reference, not a work rank. Full details, including the NEW-EXPERIMENT versus TEXT distinction and exact commands, are in `revision-roadmap.md`.

| Transport ref | Revision Item | Sub-Claim(s) | Severity | Evidence Anchor | Confidence | Source Reviewer | Obligation class | Cost scope | Bounded consequence |
|---|---|---|---|---|---|---|---|---|---|
| R1 | Add one learning experiment through the Gymnasium adapter, or narrow title and claims to environment-and-protocol | SC-6 | major | `text: §7 "they do not establish what a trained agent scores"` | 5, venue expertise | R0 | must_fix | re_analysis: §5 + title/abstract | unmet_venue_expectation: venue fit |
| R2 | Add a financial-RL-environments paragraph or table positioning against FinRL/FinRL-Meta, ABIDES-Gym, TradeMaster, mbt_gym on the paper's four property axes | SC-1 | major | `absence: §6, checked §6/§1/§7/refs.bib` | 5 (R2), 4 (R0) | R0 + R2 | must_fix | section: 06-related.tex | unverifiable_novelty_claim: originality |
| R3 | State the contribution boundary with the companion SharpeBench paper in one sentence; re-lead abstract and findings with environment-owned results (F2, F3, F5), repositioning F1 as pipeline validation | SC-7 | major | `text: §5.1 "the same unit bug the companion SharpeBench paper reports as its Finding 1"` | 4 | R0 (DA m2 adjacent) | must_fix | section: 00-abstract.tex, 01-introduction.tex, 05-experiments.tex | attribution_ambiguity: contribution accounting |
| R4 | Rewrite the stale §2 closing paragraph to past tense matching the executed evidence run | SC-8 | major | `text: §2 "their numbers arrive when the evidence run executes"` | 5 | R0 | must_fix | sentence: 02-principles.tex final paragraph | internal_contradiction: claim status |
| R5 | Replicate F8 over 8 to 16 replicator seeds per schedule and report winner-replacement frequency and dominant-species distribution, or scope the claim to the single committed run and remove it from the abstract | SC-2 | major | `dataset: paper/evidence/f8-ecology.json, seed 0, one trajectory per schedule` | 5 (R1), 4 (R3) | R1 + R3 (DA M7) | must_fix | re_analysis: make-f8-ecology.py + §5.8 + abstract | unsupported_generalization: abstract claim |
| R6 | Add seed-paired uncertainty (bootstrap or t-based CIs) to F2, F3, F5, F6; commit per-episode/per-seed vectors; separate transfer-matrix cells above and below the paper's own 0.34 DSR noise calibration | SC-9 | major | `dataset: paper/evidence/f2-regret.json, f6-adverse-selection.json, aggregate means only` | 5 | R1 (DA M4) | must_fix | re_analysis: make-f2/f3/f5/f6 scripts + §5 tables | uninterpretable_point_estimates: statistical adequacy |
| R7 | Replace "provable optimum" with "closed-form reference policy" (or prove exact optimality for the implemented reward); state the approximation status; cite Gueant, Lehalle and Fernandez-Tapia (2013) | SC-3 | major (R2), minor (R3) | `text: §3.5 "scored against a provable optimum"` | 4 (R2), 5 (R3) | R2 + R3 | must_fix | sentence: abstract, §1 contributions, §3.5, §5.2 | factual_overclaim: flagship instrument |
| R8 | Add a locator (arXiv ID/DOI/URL) to the SharpeBench companion citation, or make the F1 unit-bug derivation self-contained in an appendix | SC-11 | major | `dataset: refs.bib toca2026sharpebench, note "Preprint", no locator` | 5 | R2 | must_fix | sentence: refs.bib (+ optional appendix) | unverifiable_dependency: F1 evidence chain |
| R9 | Reframe F5: cite Huberman-Stanzl (2004) and Gatheral (2010), state that linear permanent impact makes the negative result expected, rephrase "certifies"; add one concave or transient-impact ablation so the probe has a failure mode | SC-12 | major | `text: §5.5 "the probe certifies that the Kyle-plus-Almgren-Chriss book prices this manipulation as a cost"` | 5 (R3), 4 (DA) | R3 (DA M3) | must_fix | re_analysis: make-f5-manipulation.py variant + §5.5, §6 | circular_certification: probe validity |
| R10 | F6: distinguish markout-gap sign from markout-level sign; report the positive informed-flow markout levels as a finding against the environment under the §2 standard, or recalibrate; cite Glosten-Milgrom (1985) and note endogenous quote response as future work | SC-4 | major | `table: tab:f6, informed markout positive at all horizons` | 5 | R3 (DA M3) | must_fix | section: §5.6 (+ optional recalibration) | inverted_economics: probe validity |
| R11 | Resolve C1: reframe abstract/introduction findings as instrument calibration on a generator that intentionally fails full realism certification at this revision; state explicitly that certification is reported, not required, for the canonical configuration, and defend the verifiability-first sequencing | SC (DA C1) | critical | `table: tab:f4; text: §5.4 "the default and the canonical leaderboard configuration"` | 5 | DA (R3 W4, R0 W6 corroborate) | must_fix | section: 00-abstract.tex, 01-introduction.tex, 02-principles.tex, 07-limitations.tex | validated_critical: blocks Accept |
| R12 | Resolve C2: scope "leak-free / impossible by construction" to the interface boundary throughout; add a limitations paragraph on in-band predictability of the public deterministic generator (what seed bands do and do not defend against) | SC (DA C2) | critical | `text: abstract "impossible by construction"; absence: §3/§7 inversion analysis` | 4 | DA (R2 W7 corroborates) | must_fix | section: 00-abstract.tex, 02-principles.tex, 07-limitations.tex | validated_critical: blocks Accept |

### Required Item Details

**R1: One learning experiment, or narrowed claims**
- **Problem**: all eight experiments use fixed policies; the shipped Gymnasium/PettingZoo/verifiers surfaces are never exercised by a learner (R0 W1).
- **Source**: r0-journal-fit.md W1; adjacent DA M1 (no existence proof that rank-eligibility is attainable).
- **Requirement**: train one standard RL baseline (e.g. PPO) on the train band per tier, evaluate on the held-out band and cross-regime, score through the kernel; it need not be strong. Alternatively, retitle and rescope to environment-and-protocol. Running the AS optimum through the kernel to show a rank-eligible entry exists (DA M1) should accompany either branch.
- **Acceptance criteria**: the revision contains either a trained-agent generalization-gap and transfer result scored through the kernel, or a title/abstract/claims set that no longer promises demonstrated RL utility, plus evidence that the eligibility set is non-empty or a stated acknowledgment that it is unestablished.

**R2: Trading-RL related-work positioning**
- **Problem**: §6 engages one simulator (ABIDES); the Gym-for-finance lineage is absent, so the originality claim cannot be assessed on the page.
- **Source**: r0-journal-fit.md W2; r2-domain.md W1 (with the recommended references and their verification status).
- **Requirement**: add a paragraph or table positioning SharpeArena against FinRL/FinRL-Meta, ABIDES-Gym, TradeMaster, mbt_gym on leak-freedom mechanism, determinism guarantee, trajectory verifiability, external-agent contract, generalization protocol; cite mbt_gym at §3.5/F2 specifically.
- **Acceptance criteria**: each named predecessor is cited and assessed against the four property axes; the §3.5 market-making task acknowledges mbt_gym.

**R3: Companion contribution boundary**
- **Problem**: the headline F1 finding is, on its face, the companion paper's Finding 1; attribution is ambiguous.
- **Source**: r0-journal-fit.md W3; r4-devils-advocate.md m2.
- **Requirement**: one explicit sentence in §1 stating what this paper claims versus the companion; lead the abstract and findings summary with environment-owned results; reposition F1 as validation that the producer-scorer split catches scorer bugs.
- **Acceptance criteria**: a reader can attribute every headline claim to exactly one paper; the abstract's first empirical sentence is environment-owned.

**R4: Stale status paragraph**
- **Problem**: §2's final paragraph says the findings' numbers "arrive when the evidence run executes," contradicting the executed results of §5 (verified editorially).
- **Source**: r0-journal-fit.md W4.
- **Requirement**: rewrite in past tense: the eight findings were produced by the committed evidence run at sharpearena 0.8.0 / SharpeBench 0.5.0.
- **Acceptance criteria**: no sentence in the revision leaves the execution status of the findings ambiguous.

**R5: F8 replication**
- **Problem**: the abstract-level ecology claim rests on one seed per schedule.
- **Source**: r1-methodology.md W1; r3-perspective.md W3; r4-devils-advocate.md M7.
- **Requirement**: extend `paper/src/make-f8-ecology.py` to 8 to 16 replicator seeds per schedule; report replacement frequency and dominant-species distribution; add one paragraph situating the replicator flow rule in the market-ecology literature (Farmer 2002; Lux-Marchesi 1999) and what it abstracts away (R3 W3.ii). If replication is not run, scope the text and remove the abstract claim.
- **Acceptance criteria**: the abstract's ecology sentence is supported by a reported multi-seed frequency, or absent.

**R6: Uncertainty quantification and dispersion**
- **Problem**: bare point estimates outside F1; per-episode vectors not committed; sub-noise transfer cells narrated.
- **Source**: r1-methodology.md W2; r4-devils-advocate.md M4.
- **Requirement**: seed-paired CIs for F2 regret means, F3 gap and transfer entries, F5 impact points, F6 per-horizon paired gaps; commit per-unit vectors in the evidence JSON; mark transfer-matrix cells below the 0.34 DSR noise calibration; answer whether the F6 h=1 gap of 0.072 excludes zero.
- **Acceptance criteria**: every interpreted quantity in §5 carries dispersion or an explicit sub-noise label, and the evidence JSONs allow recomputation of it.

**R7: "Provable optimum" reframing**
- **Problem**: the AS closed form is the source model's approximate solution; optimality inside this environment is unproven.
- **Source**: r2-domain.md W2; r3-perspective.md W6.
- **Requirement**: rename, caveat the model-boundedness and approximation status, cite Gueant, Lehalle and Fernandez-Tapia (2013), or prove/test exact optimality for the implemented reward and say so.
- **Acceptance criteria**: "provable optimum" appears nowhere unsupported; R2's Question 1 is answered in the text.

**R8: Companion citation locator**
- **Problem**: an unlocatable self-citation carries the scoring semantics and the F1 unit-bug claim.
- **Source**: r2-domain.md W3.
- **Requirement**: add arXiv ID/DOI/URL to `refs.bib` `toca2026sharpebench`, or summarize the unit-bug derivation in an appendix so the claim is checkable standalone.
- **Acceptance criteria**: the F1 unit-bug claim is verifiable from this paper's own artifacts or a retrievable citation.

**R9: F5 theory engagement and ablation**
- **Problem**: under linear permanent impact the negative manipulation result is close to a theorem (Huberman-Stanzl 2004; Gatheral 2010); the probe has no demonstrated failure mode.
- **Source**: r3-perspective.md W1; r4-devils-advocate.md M3 (positive-control framing).
- **Requirement**: cite the theory, downgrade "certifies" to consistency-check language, and add one ablation (square-root/concave per Almgren et al. 2005, or transient decay) showing whether the sweep then finds a profitable region; a spec engineered to permit manipulation that the probe flags would serve as the positive control DA M3 demands.
- **Acceptance criteria**: F5's text states why the linear result is expected, and the probe is shown able to fail on at least one specification.

**R10: F6 level versus gap**
- **Problem**: makers profit against informed flow at every horizon, violating the paper's own §2 standard, while the text reports only the gap direction as success.
- **Source**: r3-perspective.md W2; r4-devils-advocate.md M3.
- **Requirement**: report the level finding against the environment with F4-grade honesty; state whether it is intended calibration, a strike-price convention artifact, or a defect (R3's Question 2); distinguish the fixed-path markout design from equilibrium adverse selection with a Glosten-Milgrom citation.
- **Acceptance criteria**: §5.6 separately states the gap result and the level result, and the level result is explicitly reconciled with the §2 standard.

**R11: C1 resolution (uncertified tape)**
- **Problem**: see the C1 adjudication above.
- **Source**: r4-devils-advocate.md C1; r3-perspective.md W4; r0-journal-fit.md W6.
- **Requirement**: one clause in the abstract and one sentence in the introduction framing the eight findings as instrument calibration on intentionally uncertified tape; an explicit statement of the certification gate's current role (reported diagnostic, not leaderboard precondition); a stated argument for verifiability-first sequencing, or a change to the canonical configuration.
- **Acceptance criteria**: no reader who stops at the abstract can mistake the findings for certified-market results, and the gate's role is stated where the gate is introduced.

**R12: C2 resolution (leak-freedom scope)**
- **Problem**: see the C2 adjudication above.
- **Source**: r4-devils-advocate.md C2; r2-domain.md W7.
- **Requirement**: scope every "impossible by construction" and "leak-free" statement to the interface boundary; add a limitations paragraph on in-band predictability of a public low-complexity deterministic generator, stating that seed bands block seed lookup, not inversion, and that no bound or experiment on inversion is offered at this revision (or offer one).
- **Acceptance criteria**: the abstract's claim matches what §3 and §7 establish; the inversion question is either analyzed or explicitly disclosed as open.

---

## Suggested Revisions (Should Fix)

| Transport ref | Revision Item | Sub-Claim(s) | Severity | Evidence Anchor | Confidence | Source Reviewer | Obligation class | Cost scope | Bounded consequence |
|---|---|---|---|---|---|---|---|---|---|
| S1 | Halve the abstract; change title "The" to "A"; move interface-ownership strategy to a neutral discussion paragraph; expand PIT once | SC-5 | minor | `text: abstract "The corrected scoring kernel rewrites the baseline board"` | 4 | R0 (R2, DA M5 corroborate register) | should_fix | section: title, 00-abstract.tex | reviewer_misprioritization: venue-fit surface |
| S2 | Reframe significance as checkable-today infrastructure with adoption as future work; neutralize "Whoever defines the interface owns the ecosystem" | SC (R0 W6) | minor (R0), major (DA M5) | `text: §7 "no third party has yet submitted an agent"` | 4 | R0 + DA | should_fix | sentence: §1, §2, §7 | rhetoric_evidence_gap: significance |
| S3 | State evidence-seed provenance (train band) in §5 preamble and why immaterial for parameter-free baselines; reword F1 "survives every held-out seed" | SC-10 | minor | `text: §5.1 "survives every held-out seed"` | 5 | R1 | should_fix | sentence: §5 preamble, §5.1 | protocol_misdescription: seed bands |
| S4 | Commit bootstrap resample count and seed in `make-f1-baselines.py` and the evidence config | SC (R1 W4) | minor | `dataset: f1-baselines.json config, no bootstrap params` | 4 | R1 | should_fix | re_analysis: make-f1-baselines.py rerun | silent_reproducibility_gap: F1 CIs |
| S5 | Add the push-weight 0.75 row to tab:f5 (or footnote the omission); state the per-axis grid; add seed dispersion | SC (R1 W5) | minor | `dataset: f5-manipulation.json push_weights vs Table 3` | 5 | R1 | should_fix | sentence: tab:f5 + §5.5 | selective_presentation_appearance: F5 |
| S6 | Re-certify vol-clustering on disjoint seeds at 2 or 3 strengths, or label the re-certification as in-sample on the diagnosis panel | SC (R1 W6) | minor | `dataset: f4-realism.json, seeds [0..7] shared` | 4 | R1 | should_fix | re_analysis or sentence: §5.4 | circular_tuning_appearance: F4 remediation |
| S7 | Aggregate the F6 maker decomposition over 24 episodes or label it a single-episode illustration; define the markout unit and notional base | SC (R1 W7) | minor | `dataset: f6-adverse-selection.json detail_seed 0` | 4 (R1), 4 (DA) | R1 (DA m4, R3 minor) | should_fix | sentence: §5.6, tab:f5/f6 captions | anecdote_as_finding: F6 detail |
| S8 | Split the Cont (2001) citation: add Zumbach for timescale asymmetry; source or own the Fano factor | SC (R2 W4) | minor | `text: §3.7 stylized-facts list under \citep{cont2001empirical}` | 4 | R2 | should_fix | sentence: §3.7, §5.4, refs.bib | citation_scope_error: realism battery |
| S9 | Add foundational citations for the probes: Glosten-Milgrom (F6), Allen-Gale (F5), Farmer / Lux-Marchesi / Lo (F8) | SC (R2 W5) | minor | `absence: §3.7/§5 probe foundations` | 4 (R2), per-item (R3) | R2 + R3 | should_fix | sentence: §3.7, §5, §6, refs.bib | reinvention_appearance: probe ancestry |
| S10 | Soften or cite "look-ahead bugs are the dominant failure mode of backtests" | SC (R2 W6) | minor | `text: §1 "the dominant failure mode of backtests"` | 4 | R2 | should_fix | sentence: 01-introduction.tex | unsupported_superlative: motivation |
| S11 | Name the batch-auction clearing design (cite Budish et al. 2015); state excluded competition margins and whether fees are modeled (gross vs net Sharpes) | SC (R3 W5) | minor | `absence: §3.5/§7 clearing-mechanism scope` | 4 | R3 | should_fix | section: §3.5 or §7 | silent_scope_restriction: market design |
| S12 | Add a broader-impact paragraph: leaderboard-score marketing misuse norm; dual-use surface of the manipulation probe | SC (R3 W7) | minor | `absence: §7 broader-impact treatment` | 3 | R3 | should_fix | section: §7 or new paragraph | venue_expectation: broader impact |
| S13 | Scope "remove the host from the trust equation": state that F4-F8 sit outside the golden-hash guarantee where the guarantee is advertised, not only in §7 | SC (DA M2) | major | `text: §7 "float reproducibility there is the platform's, not the paper's"` | 5 | DA | should_fix | sentence: 00-abstract.tex, 04-contract.tex | guarantee_coverage_gap: trust chain |
| S14 | F3: remove or reframe "training regime" language for untrained policies; note the matrix reduces to per-tier DSR differences for fixed policies | SC (DA M4, text half) | major | `table: tab:f3` | 4 | DA | should_fix | sentence: §5.3 | implied_learning: F3 framing |
| S15 | F7: state the mandate-assignment mechanism as the driver of the breach counts; caveat that a mandate-aware field could invert the distribution | SC (DA M6) | major | `table: tab:f7; §5.7 mechanism sentence` | 4 | DA | should_fix | sentence: §5.7 | design_artifact_as_finding: F7 headline |
| S16 | Resolve the Calm-narration inconsistency: universal versus regime-conditional stylized-facts bounds | SC (DA m1) | minor | `text: §5.4 Calm paragraph` | 4 | DA | consider | sentence: §2, §5.4 | inconsistent_standard: F4 narration |
| S17 | Tone down "provably disjoint" for the seed-band intervals | SC (DA m3) | minor | `text: §3.4 "carves a provably disjoint test family"` | 5 | DA | consider | sentence: §3.4 | inflation: theorem-grade phrasing |
| S18 | Presentation minors: Table 1 eligibility-rule footnote; Fig 3 / Table 3 duplication; minari and byrd2020abides bib entries; pass^k rendering; "subsidised" spelling; "mis-united" phrasing; §5.2 "validates the zero point" wording | SC (minors) | minor | per source reports | per source | R0 + R1 + R2 + R3 | consider | sentence: various | polish: presentation |

---

## Revision Roadmap

### Source-traceability checklist

Kept in immutable source order; no work order is suggested. The author selects `will_address`, `wont_address`, or `not_on_point` later in the separate author-adjudication checkpoint. The full roadmap with TEXT / NEW-EXPERIMENT tags is `revision-roadmap.md`.

- [ ] R1: obligation `must_fix`: learning experiment or narrowed claims
- [ ] R2: obligation `must_fix`: trading-RL related-work positioning
- [ ] R3: obligation `must_fix`: companion contribution boundary
- [ ] R4: obligation `must_fix`: stale §2 status paragraph
- [ ] R5: obligation `must_fix`: F8 multi-seed replication or rescoping
- [ ] R6: obligation `must_fix`: uncertainty quantification F2/F3/F5/F6
- [ ] R7: obligation `must_fix`: "provable optimum" reframing
- [ ] R8: obligation `must_fix`: companion citation locator
- [ ] R9: obligation `must_fix`: F5 theory engagement + concave-impact ablation
- [ ] R10: obligation `must_fix`: F6 level-versus-gap
- [ ] R11: obligation `must_fix`: C1 resolution (uncertified tape framing)
- [ ] R12: obligation `must_fix`: C2 resolution (leak-freedom scoping)
- [ ] S1: obligation `should_fix`: abstract and title
- [ ] S2: obligation `should_fix`: significance register
- [ ] S3: obligation `should_fix`: seed provenance wording
- [ ] S4: obligation `should_fix`: bootstrap parameters committed
- [ ] S5: obligation `should_fix`: F5 grid completeness
- [ ] S6: obligation `should_fix`: vol-clustering re-certification hygiene
- [ ] S7: obligation `should_fix`: F6 decomposition scope and units
- [ ] S8: obligation `should_fix`: Cont citation split
- [ ] S9: obligation `should_fix`: probe foundational citations
- [ ] S10: obligation `should_fix`: "dominant failure mode" claim
- [ ] S11: obligation `should_fix`: batch-clearing and fee scope
- [ ] S12: obligation `should_fix`: broader-impact paragraph
- [ ] S13: obligation `should_fix`: golden-hash coverage scoping
- [ ] S14: obligation `should_fix`: F3 framing
- [ ] S15: obligation `should_fix`: F7 mechanism caveat
- [ ] S16: obligation `consider`: Calm narration consistency
- [ ] S17: obligation `consider`: "provably disjoint" phrasing
- [ ] S18: obligation `consider`: presentation minors

---

## Journal-Supplied Deadline (Optional Transport)

- **Exact deadline from source letter**: NOT PROVIDED

---

## Response Letter Instructions

Please use the format in `templates/revision_response_template.md` to respond to every reviewer comment item by item, including: a response and revision description for each Required Revision; a response for each Suggested Revision (adopted, or reason for not adopting); change markup; and a cross-reference table of new page numbers or paragraphs. The four reviewer questions lists (R0 Q1-Q4, R1 Q1-Q4, R2 Q1-Q4, R3 Q1-Q4) require explicit answers; several are load-bearing for must-fix items (R2's Q1 for R7; R3's Q2 for R10; R1's Q1 for R5).

---

## Closing

We encourage you to carefully consider the reviewers' comments and submit a substantially revised manuscript. Please note that the revised manuscript will undergo another round of review. The panel was unanimous that the underlying artifact is strong and the defects repairable: the verifiability engineering and the evidence discipline were verified and credited by every seat, and no finding requires new core machinery. The revision's center of gravity is bringing the paper's framing down to what its own artifacts already support, plus two inexpensive experiment batches (F8 seeds, dispersion reporting) and one probe ablation.

---

## Part 3: Reviewer Report Summary (Appendix)

### Journal-Fit Review (R0) Summary
- Recommendation: Major Revision | Confidence: 4
- Key Point: right artifact type with exemplary artifact backing, but an RL-environment paper without a learning experiment, without trading-RL related work, with a blurred companion boundary and a stale status paragraph, misses the track's bar; all four repairable.

### Reviewer 1 (Methodology) Summary
- Recommendation: Major Revision | Confidence: 4
- Key Point: every number independently recomputes from committed evidence and the determinism claims are tested as claimed, but the statistical layer (N=1 ecology, no uncertainty outside F1, unexercised seed-band protocol) does not yet match the engineering layer.

### Reviewer 2 (Domain) Summary
- Recommendation: Major Revision | Confidence: 5
- Key Point: general-RL scholarship is accurate and genuinely synthesized and the contribution is credibly novel, but the Gym-for-finance lineage is entirely absent, "provable optimum" is a factual overclaim, and the load-bearing companion citation has no locator.

### Reviewer 3 (Perspective) Summary
- Recommendation: Major Revision | Confidence: 4
- Key Point: the "distrust the simulator" stance is a real contribution, but the manipulation probe cannot fail under linear impact, the adverse-selection levels invert the economics being certified, and the ecology claim is one run; probes must be held to the paper's own §2 standard.

### Devil's Advocate Summary
- Recommendation: N/A: findings only
- Key Challenge: the findings rest on tape the paper's own realism gate refuses to certify (C1, validated), and "leak-free by construction" is an interface claim presented as an information claim with generator inversion unanalyzed (C2, validated).
