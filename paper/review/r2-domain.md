# Peer Review Report

## Manuscript Information
- **Title**: SharpeArena: The Point-in-Time (PIT) RL Environment for Trading Agents
- **Manuscript ID**: N/A (preprint; NeurIPS 2026 Evaluations and Datasets track style)
- **Review Date**: 2026-08-24
- **Review Round**: Round 1

---

## Reviewer Information

### Reviewer Role
Peer Reviewer 2 (Domain)

### Reviewer Identity
Senior researcher in reinforcement-learning environments and benchmarks (Gym/Gymnasium lineage, ALE evaluation protocols, Procgen and procedural generalization, PettingZoo, Minari, PLR curricula) with working expertise in agent-based market simulation (ABIDES lineage) and market microstructure models.

### Review Focus
Literature coverage and the correctness of every citation claim; positioning against prior RL environments and market simulators; theoretical framing (leak-freedom, determinism, contract governance); and whether the incremental contribution over Gym-for-finance predecessors is genuine. I do not assess experimental methodology internals (Reviewer 1) or cross-disciplinary reach (Reviewer 3).

---

## Overall Assessment

### Recommendation
- [x] **Major Revision** - Substantial revisions needed, re-review required after revision

### Confidence Score
5

Confidence is an uncertainty/scope disclosure only; it never changes consensus counts, severity, decision bearing, or arbitration.

### Summary Assessment
The paper presents SharpeArena, a procedurally generated, byte-deterministic, replay-verifiable RL environment for trading agents with a governed JSON wire contract, and reports eight scripted experiments calibrating its instruments. Within the general RL-environments literature the scholarship is strong: the Gym, Gymnasium, ALE, Procgen, PLR, PettingZoo, Minari and tau-bench citations are all real, their metadata is essentially correct, and each claim attributed to them is accurate, sometimes unusually precisely so (the Machado et al. determinism-versus-memorization lesson is correctly stated and correctly contrasted). The critical gap is the financial-RL environment literature: the Related Work section engages exactly one market simulator (ABIDES) and none of the Gym-for-finance predecessors (FinRL/FinRL-Meta, ABIDES-Gym, TradeMaster, mbt_gym), so the paper's positioning claim, and therefore its incremental contribution, cannot be assessed as written even though I believe the verifiability-and-governance axis is genuinely novel. Two accuracy problems compound this: the "provable optimum" framing of the Avellaneda-Stoikov closed form overstates what that model derives, and the stylized-facts battery attributes to Cont (2001) two statistics (Zumbach asymmetry, Fano factor) that do not appear there. The load-bearing companion citation (SharpeBench) has no locator. These are all repairable, hence Major Revision rather than Reject.

---

## Strengths

### S1: RL-environments citations are accurate in both metadata and attributed claim
Every entry in refs.bib that I can check corresponds to a real work with essentially correct metadata (arXiv IDs 1606.01540, 2407.17032, 1912.01588, 2010.03934, 2009.14471, 2406.12045 all match their titles and author lists; Machado et al. JAIR 61:523-562 is correct; Bailey and Lopez de Prado JPM 40(5):94-107 is correct; Kyle Econometrica 53(6):1315-1335, Almgren-Chriss J. Risk 3(2):5-39, Avellaneda-Stoikov Quant. Finance 8(3):217-224, Cont Quant. Finance 1(2):223-236 are all correct). The claims hung on them are what those papers actually say.
**Evidence Anchor**: text: §6 (Related work, "Evaluation rigor in RL") "determinism in the environment invites trajectory memorization, so evaluation must be designed against it"

### S2: The ALE lesson is applied with unusual precision, not name-dropped
The paper does not just cite Machado et al. (2018); it states the design fork correctly: ALE's remedy was injected stochasticity (sticky actions), and SharpeArena deliberately takes the other branch, keeping determinism for verifiability and defeating memorization with procedural distributions and held-out seed bands. This is a correct, non-superficial use of the evaluation-protocols literature and is exactly the right theoretical frame for a deterministic-by-design environment.
**Evidence Anchor**: text: §6 "SharpeArena keeps determinism for verifiability and defeats memorization with procedural scenario distributions and held-out seed bands instead of injected stochasticity"

### S3: The Avellaneda-Stoikov formulas are transcribed correctly
The reservation price r = s - q gamma sigma^2 tau and half-spread delta* = gamma sigma^2 tau / 2 + gamma^{-1} ln(1 + gamma/kappa) match the 2008 paper's closed-form quoting rule. Whatever my complaint in W2 about what that closed form is called, the mathematics attributed to the source is the source's mathematics.
**Evidence Anchor**: equation: §3.5 (sec:tasks) - the delta* expression

### S4: The positioning against ABIDES is honest and specific
The one simulator comparison the paper does make is done well: it concedes ABIDES's fidelity advantage and names the four properties ABIDES does not target (cross-runtime byte-determinism, structural leak-freedom, replay tamper evidence, governed wire contract). This is the right comparison template; W1 asks for it to be applied to the rest of the field.
**Evidence Anchor**: text: §6 "SharpeArena trades some of that fidelity for properties ABIDES does not target"

### S5: The limitations section does real epistemic work
Each limitation names the specific claim it bounds (F4's realism failure quantified as 1/24 conjunction passes; the golden-hash guarantee explicitly not covering the Python probe layer; the deny-list guard's side channels enumerated; no external entrants yet). This is the opposite of boilerplate and materially constrains overclaiming elsewhere.
**Evidence Anchor**: text: §7 "a deny-list is exactly as good as its patterns"

---

## Weaknesses

### W1: The Gym-for-finance predecessor literature is entirely absent, so the incremental contribution cannot be positioned
**Problem**: Related Work covers the general RL-environment lineage well but cites exactly one financial predecessor, ABIDES. The financial-RL environment literature that this paper is structurally an entry in is missing: FinRL and FinRL-Meta (the latter published at the NeurIPS Datasets and Benchmarks track, precisely this paper's stated venue, and precisely a "market environments and benchmarks" framework), ABIDES-Gym (the Gym wrapping of the one simulator the paper does cite), TradeMaster, and mbt_gym (a Gym environment family for model-based limit-order-book trading built around the same Avellaneda-Stoikov model this paper ships as its calibration task).
**Evidence Anchor**: absence: §6 Related work — expected engagement with FinRL/FinRL-Meta, ABIDES-Gym, TradeMaster, mbt_gym or any Gym-for-finance predecessor; checked §6, §1, §7, refs.bib
**Why it matters**: The contribution claim (a leak-free, deterministic, replay-verifiable, contract-governed trading environment) is plausibly novel on the verifiability-and-governance axis, but a reader cannot verify that novelty when the closest prior systems are never named. For a Datasets and Benchmarks submission, missing FinRL-Meta in particular will be noticed immediately. It also weakens the paper's own strongest argument: the predecessors mostly do not attempt byte-determinism, leak-freedom by construction, or governed wire contracts, so the comparison would favor the paper.
**Suggestion**: Add a "Financial RL environments" paragraph to §6 mirroring the ABIDES paragraph: state what each predecessor standardizes and which of the four properties (determinism, leak-freedom, replay verification, contract governance) each lacks. Recommended references: Liu et al., FinRL (arXiv 2011.09607) and FinRL-Meta (NeurIPS 2022 Datasets and Benchmarks, arXiv 2211.03107); Amrouni et al., ABIDES-Gym (ICAIF 2021, arXiv 2110.14771); Sun et al., TradeMaster (NeurIPS 2023 Datasets and Benchmarks) [UNVERIFIED exact venue year, search lead: TradeMaster RL-for-trading platform]; Jerome et al., mbt_gym (arXiv 2209.07823) [UNVERIFIED exact venue, search lead: ICAIF model-based LOB gym environments].
**Severity**: Major
**Confidence**: 5 - core expertise: RL environments and benchmarks

### W2: "Provable optimum" mischaracterizes the Avellaneda-Stoikov closed form
**Problem**: The paper repeatedly calls the shipped AS policy a "provable optimum" (abstract, contributions item 4, §3.5, F2) and says agents are "scored on regret against a provable optimum". The Avellaneda-Stoikov closed-form quotes are an approximate solution: the 2008 paper derives them via an asymptotic expansion of the HJB value function (their own text presents the spread rule as an approximation), and the exact solution of the AS market-making problem was only later characterized (Gueant, Lehalle and Fernandez-Tapia, "Dealing with the inventory risk", Mathematics and Financial Economics, 2013). Additionally, the observed regret of 0.0 for the optimum is by construction tautological (the candidate is the reference), which the paper concedes ("scores approximately zero regret against itself by construction") while still headlining "provable".
**Evidence Anchor**: text: §3.5 "it is scored against a provable optimum, and the optimum scores approximately zero regret against itself by construction"
**Why it matters**: Within the paper's own simulated dynamics the AS policy may or may not be the true optimum of the environment's reward; nothing in the paper proves it, and in the source model it is provably only approximate. "Provable optimum" is a factual overclaim on the environment's flagship calibration instrument, and F2's interpretation ("regret against a provable optimum ... removes that luck axis entirely") inherits it. A microstructure-literate reviewer will flag this immediately.
**Suggestion**: Rename to "closed-form reference policy" or "analytical baseline"; state that regret is measured against the AS closed-form quotes, which are the model's standard approximate-optimal solution; cite Gueant-Lehalle-Fernandez-Tapia 2013 for the exact treatment; or, if the environment's dynamics are constructed so that the closed form is exactly optimal for the implemented reward, prove or test that claim explicitly and say so.
**Severity**: Major
**Confidence**: 4 - core expertise adjacent: market-making models within simulator design

### W3: The load-bearing companion citation has no locator
**Problem**: toca2026sharpebench is cited eleven-plus times and carries the scoring semantics the whole paper delegates (deflated Sharpe gates, pass-k reliability, the unit-bug cross-claim in F1 "the same unit bug the companion SharpeBench paper reports as its Finding 1"). The bib entry is author, title, year, "Preprint": no arXiv ID, no DOI, no URL.
**Evidence Anchor**: dataset: refs.bib entry toca2026sharpebench - note field "Preprint" with no locator
**Why it matters**: F1's central narrative (the corrected deflation prior) and the entire "the kernel judges" architecture rest on a document the reader cannot retrieve. For the F1 unit-bug claim specifically, the citation is the only evidence offered that the pre-0.5.0 behavior was a bug in the stated direction. An unlocatable self-citation carrying this much weight is a verifiability failure in a paper whose thesis is verifiability.
**Suggestion**: Add the arXiv identifier or a persistent URL to the bib entry before submission; if the companion is not yet public, say so explicitly in the text and summarize the unit-bug derivation (annualized prior applied per period; conversion at 252) in an appendix so the claim is checkable standalone.
**Severity**: Major
**Confidence**: 5 - direct check of refs.bib against the manuscript's dependency on it

### W4: Citation-scope error: Zumbach asymmetry and the Fano factor are attributed to Cont (2001)
**Problem**: §3.7 and F4 describe the stylized-facts battery as computing "excess kurtosis, absolute-return autocorrelation, Zumbach timescale asymmetry, gain/loss skew, aggregational Gaussianity and a Fano burstiness factor" with the citation \citep{cont2001empirical} governing the list. Cont (2001) documents heavy tails, volatility clustering, aggregational Gaussianity, gain/loss asymmetry and slow decay of absolute-return autocorrelation; it contains neither the Zumbach time-reversal asymmetry (that is Zumbach's later work, circa 2009, on time-reversal invariance in finance) nor a Fano burstiness factor (a point-process statistic not in Cont's list).
**Evidence Anchor**: text: §3.7 "computes excess kurtosis, absolute-return autocorrelation, Zumbach timescale asymmetry, gain/loss skew, aggregational Gaussianity and a Fano burstiness factor"
**Why it matters**: A reader sent to Cont for the definition of two of the six battery statistics will not find them; the certification gate's authority is being borrowed from a source that does not cover it. This is exactly the class of citation-claim mismatch a citation audit exists to catch.
**Suggestion**: Split the citation: keep Cont (2001) for the classical facts, add Zumbach for the timescale asymmetry (search lead: Zumbach, time reversal invariance in finance, Quantitative Finance, circa 2009 [UNVERIFIED exact volume/pages]), and either cite a source for the Fano factor as a market stylized fact or present it as the authors' own added diagnostic with its bound justified in the text.
**Severity**: Minor
**Confidence**: 4 - core expertise: stylized-facts literature as used in simulator validation

### W5: The probe layer's microstructure concepts are used without their foundational citations
**Problem**: The adverse-selection probe operationalizes the Glosten-Milgrom mechanism (makers losing to informed flow, noise flow subsidizing informed flow) with no citation to Glosten and Milgrom (1985) or any adverse-selection source; the manipulation probe formalizes pump-and-dump profitability with no citation to the manipulation literature (e.g., Allen and Gale, 1992, on stock-price manipulation); the ecology probe runs replicator dynamics over strategy populations with no citation to the market-ecology literature (Farmer's market ecology work; Lo's Adaptive Markets Hypothesis).
**Evidence Anchor**: absence: §3.7 and §5 (F5, F6, F8) — expected citations grounding adverse selection, manipulation payoff, and strategy-ecology framings; checked §3.7, §5, §6, refs.bib
**Why it matters**: These probes are presented as contributions (contribution item 5). Uncited, they read as reinvented; cited, they read as principled operationalizations of known mechanisms, which is stronger. The markout-vs-informed-flow design in F6 is a direct simulation of the Glosten-Milgrom adverse-selection channel and should say so.
**Suggestion**: Add Glosten and Milgrom (1985), Journal of Financial Economics 14(1):71-100, for adverse selection; Allen and Gale (1992), Review of Financial Studies, for manipulation [UNVERIFIED exact volume/pages]; for ecology, literature on market ecology and evolutionary finance, e.g. work by J. D. Farmer and by A. W. Lo [UNVERIFIED exact citations, search leads].
**Severity**: Minor
**Confidence**: 4 - adjacent field: microstructure theory supporting simulator probes

### W6: Uncited empirical claim that look-ahead is "the dominant failure mode of backtests"
**Problem**: The introduction asserts "look-ahead bugs are the dominant failure mode of backtests, and they are usually invisible in the output" with no citation. This is a motivating empirical claim about field practice, not a design statement.
**Evidence Anchor**: text: §1 "look-ahead bugs are the dominant failure mode of backtests, and they are usually invisible in the output"
**Why it matters**: The claim is plausible and widely believed among practitioners, but I cannot ground a specific source establishing dominance over, say, selection bias (which Bailey and Lopez de Prado, already in the bibliography, argue is the central backtest pathology). As written it is an unsupported superlative in the paper's opening argument. [FIELD-NORM UNVERIFIED]
**Suggestion**: Either soften to "a pervasive and usually invisible failure mode" or cite the backtest-overfitting/backtesting-pitfalls literature (the existing bailey2014deflated entry, or the related Bailey et al. pseudo-mathematics line of work) and drop the "dominant" ranking.
**Severity**: Minor
**Confidence**: 4 - core expertise: benchmark-motivation claims

### W7: The abstract's "impossible by construction" overstates what the limitations section carefully bounds
**Problem**: The abstract claims the three failure classes are "impossible by construction rather than forbidden by rule". §7 then correctly concedes that the structural guarantee covers only the environment's own surface, that the harness-side guard is a pattern deny-list, and that side channels (wall clock, filesystem, network) defeat both layers. The body's framing (structural guarantee plus detection in depth) is defensible; the abstract's universal "impossible" is not the same claim.
**Evidence Anchor**: text: Abstract "built so that these failures are impossible by construction rather than forbidden by rule"
**Why it matters**: The abstract is the claim most readers and reviewers carry; it currently promises more than §7 delivers, and the gap is precisely on the property (leak-freedom) that names the paper.
**Suggestion**: Scope the abstract sentence to the environment boundary, e.g. "impossible through the environment's own interface, and policed beyond it", matching §3.1 and §7.
**Severity**: Minor
**Confidence**: 5 - direct textual comparison

---

## Detailed Comments

### Title & Abstract
- The "(PIT)" expansion in the title is helpful; "The Point-in-Time RL Environment" (definite article) reads as a uniqueness claim the missing related work (W1) cannot currently support. Consider "A Point-in-Time RL Environment".
- Abstract overclaim addressed in W7.

### Introduction
- The three-failure taxonomy (leakage, non-reproducibility, unverifiable trajectories) is a clean and, to my knowledge, original framing for trading-environment design. Well argued.
- The Gym "interface ownership" strategic framing is apt and correctly attributed.
- W6 applies to the opening paragraph.

### Literature Review / Theoretical Framework (primary remit)
- **Coverage**: Strong on the general RL-environments axis (Gym, Gymnasium, ALE revisitation, Procgen, PLR, PettingZoo, Minari, tau-bench: all present, all correctly used). Near-absent on the Gym-for-finance axis (W1) and thin on the microstructure foundations of the probe layer (W5).
- **Integration quality**: What is cited is genuinely synthesized, not enumerated; the ALE and Procgen lessons drive concrete design decisions and the paper says which. This is above the bar for the venue.
- **Research gap argument**: Currently made by construction ("no environment does X") without surveying candidates that might. The gap is probably real on the determinism/verifiability/governance axis; the paper must demonstrate it against named predecessors.
- **Theoretical framing**: The leak-freedom argument (capability absence rather than policy) and the determinism argument (arithmetic restriction plus golden hashes plus cross-runtime parity) are sound and clearly articulated. Two notes: (a) FNV-1a is a non-cryptographic hash; the paper's tamper-evidence story correctly rests on replay rather than the hash, but the reproducibility statement should avoid implying the hashes themselves resist an adversary. (b) The contract-governance framing (additive-only evolution, parallel namespaces for breaking changes) is imported from protocol engineering practice without citation; this is acceptable as engineering doctrine but a sentence acknowledging the lineage (semantic versioning, protocol evolution practice) would be honest.

### Results / Findings (domain-accuracy remarks only)
- F1's interpretation (eligibility as a conjunction, each leg catching what the other misses) is a correct and well-stated benchmark-design point; its evidentiary base depends on W3.
- F2 inherits W2's framing problem but the U-shaped regret result itself is the expected shape and is reported plainly.
- F3's distinction between within-tier gap (Procgen-style) and cross-regime transfer (held seeds, swapped regime) is a genuine and useful refinement of the Procgen protocol for market environments; the honest admission that the within-tier control is vacuous for parameter-free policies is to the paper's credit.
- F4 is the paper at its best: the certification gate fails its own generator and the paper reports it as a finding against itself. The vol_clustering follow-up paragraph reads as a post-hoc addition; make its status (shipped but off by default, finding stands at default) unmissable, which it currently nearly is.
- F5/F6/F8 are sensible probes; see W5 for their missing intellectual ancestry.

### Discussion / Limitations
- §7 is exemplary for this genre; each paragraph bounds a specific headline claim. No missing limitation within my remit except the positioning one implied by W1.

### References
- All fifteen entries are real works; no fabricated or phantom citations detected. Metadata spot-checks pass except the minor items listed below.

### Contribution to the Field
- **Incremental contribution**: Real, in my assessment, and located on an axis predecessors do not occupy: cross-runtime byte-determinism with committed hashes, structurally leak-free observation, decisions-only replay verification, and a governed language-agnostic wire contract. FinRL-Meta and ABIDES-Gym standardize access and tasks; neither attempts verifiability-by-construction. But this assessment is mine, made from field knowledge; the paper must earn it on the page by engaging those systems (W1).
- **Positioning**: The producer/judge split with SharpeBench is architecturally clean; its cost is dependence on an unlocatable companion (W3).
- **Overclaiming risk**: Concentrated in three places: "provable optimum" (W2), abstract "impossible by construction" (W7), and the definite-article title. Elsewhere the paper is notably careful.

#### Missing Key References
- Liu et al., FinRL (arXiv 2011.09607); Liu et al., FinRL-Meta (NeurIPS 2022 Datasets and Benchmarks, arXiv 2211.03107): the direct Gym-for-finance predecessors; required for positioning.
- Amrouni et al., ABIDES-Gym (ICAIF 2021, arXiv 2110.14771): the Gym interface to the one simulator the paper cites; required.
- Jerome et al., mbt_gym (arXiv 2209.07823) [UNVERIFIED exact venue]: Gym environments for Avellaneda-Stoikov-style LOB market making; directly relevant to §3.5.
- Sun et al., TradeMaster [UNVERIFIED exact metadata, search lead: NeurIPS Datasets and Benchmarks RL-for-trading platform]: relevant as a benchmark-suite predecessor.
- Gueant, Lehalle and Fernandez-Tapia (2013), Mathematics and Financial Economics: exact solution of the AS market-making problem; required if the "optimum" language is kept in any form.
- Glosten and Milgrom (1985), Journal of Financial Economics 14(1):71-100: adverse-selection foundation for F6.
- Zumbach (circa 2009), time reversal invariance in finance [UNVERIFIED exact volume/pages]: source for the Zumbach asymmetry statistic in the realism battery.
- Allen and Gale (1992), Review of Financial Studies [UNVERIFIED exact volume/pages]: manipulation-payoff foundation for F5.
- Market ecology and evolutionary finance, e.g. work by J. D. Farmer and by A. W. Lo [UNVERIFIED, search leads]: framing for F8.

---

## Questions for Authors

1. Is the Avellaneda-Stoikov closed-form policy exactly optimal for the reward implemented in your market-making environment (i.e., did you construct the dynamics so the approximation is exact), or is it the standard approximate solution? The answer determines whether "provable optimum" can be repaired or must be removed.
2. How does SharpeArena relate to FinRL-Meta and ABIDES-Gym concretely: which of your four headline properties (byte-determinism, structural leak-freedom, replay verification, contract governance) does each predecessor lack, and did you verify that by inspection of their code or documentation?
3. Will the SharpeBench companion paper have a public locator (arXiv ID or DOI) before submission, and if not, can the F1 unit-bug derivation be made self-contained in this paper's appendix?
4. For the realism battery, what is the intended source of authority for the Fano-factor bound: a literature stylized fact, or a diagnostic you introduce? The text currently implies Cont (2001) covers it.

---

## Minor Issues

### Citation Format
- refs.bib byrd2020abides: year given as 2020 but eprint 1904.12066 encodes an April 2019 arXiv posting; if citing the published SIGSIM-PADS 2020 version, cite it as an inproceedings; if citing the arXiv version, the year should match the posting or the entry should note the revision.
- refs.bib minari: authored as "{Farama Foundation}" with only a URL; if a citable Minari paper or versioned software DOI exists, prefer it, and pin the version used by the sharpearena-py adapter.
- refs.bib toca2026sharpebench: add locator (see W3).
- §6 cites cobbe2020procgen for "integer-seed-interval model"; Procgen's own terminology is level sets over seed ranges. The usage is faithful; consider "level-seed interval" phrasing to match the source.

### Language / Layout
- The abstract is a single very long paragraph carrying eight experimental results; consider moving the last three result sentences to the introduction.
- "pass\textsuperscript{$k$}" and "pass^k" render inconsistently across §1 and §5; unify.

---

## Criterion-Bound Judgements

Calibration status: `NOT_CALIBRATED`

Current seat reports cannot know the final actual panel topology and never self-upgrade from a candidate profile.

| Dimension | Criterion source | Judgement | Evidence anchor(s) | Rationale | Uncertainty / scope limit | Decision bearing? |
|---|---|---|---|---|---|---|
| Originality | reviewer skill quality_rubrics (domain remit) | MEETS | text: §6 "properties ABIDES does not target" | Verifiability-and-governance axis appears genuinely unoccupied among trading RL environments | Judgement rests on reviewer field knowledge; paper itself does not demonstrate it (W1) | yes: contribution is credible but must be positioned on the page |
| Methodological Rigor | reviewer skill quality_rubrics | NOT_ASSESSED | — | Reviewer 1's remit | none identified | no: outside remit |
| Evidence Sufficiency | reviewer skill quality_rubrics (citation-evidence slice only) | PARTLY_MEETS | dataset: refs.bib toca2026sharpebench "Preprint" | Key scoring semantics and the F1 unit-bug claim depend on an unlocatable companion (W3) | Only the citation-evidence slice assessed here | yes: W3 must be repaired |
| Argument Coherence | reviewer skill quality_rubrics (domain-argument slice) | PARTLY_MEETS | text: Abstract "impossible by construction" | Body argues structural-guarantee-plus-detection correctly; abstract and "provable optimum" outrun it (W2, W7) | Coherence of experimental logic left to Reviewer 1 | yes: W2 affects a headline contribution claim |
| Writing Quality | reviewer skill quality_rubrics | NOT_ASSESSED | — | Outside domain remit beyond terminology | none identified | no |
| Literature Integration | reviewer skill quality_rubrics | PARTLY_MEETS | absence: §6 Related work — expected Gym-for-finance predecessors; checked §6, §1, refs.bib | Excellent on general RL-environments axis; missing the financial-RL environment lineage entirely (W1) and probe-layer foundations (W5) | none identified | yes: W1 is the largest single revision item |
| Significance & Impact | reviewer skill quality_rubrics | NOT_ASSESSED | — | Reviewer 3's remit | none identified | no |

The recommendation follows from the unresolved decision-bearing criteria: Literature Integration (W1, repairable by adding and engaging four to six named predecessor systems), Argument Coherence (W2/W7, repairable by reframing "provable optimum" and scoping the abstract), and Evidence Sufficiency (W3, repairable by adding a locator or making the unit-bug claim self-contained). All three are repairable without new experiments, but each touches a headline claim, so re-review after revision is warranted. Strength on the general-RL citation axis does not offset the financial-predecessor gap.
