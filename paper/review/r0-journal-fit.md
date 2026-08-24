# Peer Review Report

## Manuscript Information
- **Title**: SharpeArena: The Point-in-Time (PIT) RL Environment for Trading Agents
- **Manuscript ID**: r0-journal-fit
- **Review Date**: 2026-08-24
- **Review Round**: Round 1

---

## Reviewer Information

### Reviewer Role
Journal-Fit Reviewer

### Reviewer Identity
Senior area chair persona for the NeurIPS Datasets & Benchmarks (Evaluations and Datasets) track, with background in RL environment design, benchmark governance, and ML for finance. This seat evaluates venue fit, originality against existing RL environments and market simulators, significance, contribution framing, and readership relevance. Statistics and RL methodology internals are owned by other seats and are not assessed here.

### Review Focus
Is this paper the kind of artifact the NeurIPS D&B track exists to publish, is its contribution genuinely new relative to Gym-family environments and market simulators, and is the contribution framed so the track's readership can evaluate and adopt it? I read all ten section files, the appendix, and inspected the repository (committed evidence JSONs under `paper/evidence/`, generator scripts under `paper/src/`, figure PDFs, README, EVALUATION.md) to check that the reproducibility claims are backed by artifacts.

---

## Overall Assessment

### Recommendation
- [ ] Accept
- [ ] Minor Revision
- [x] **Major Revision**
- [ ] Reject

### Confidence Score
4

Confidence is an uncertainty/scope disclosure only; it never changes consensus counts, severity, decision bearing, or arbitration.

### Calibration Status
`NOT_CALIBRATED`

### Summary Assessment
The paper presents SharpeArena, a procedurally generated point-in-time trading environment whose distinguishing properties are structural leak-freedom, cross-runtime byte-determinism pinned by committed golden hashes, replay-based tamper evidence over decision-only trajectories, and a governed language-agnostic JSON agent contract; eight seeded experiments, each backed by a committed script and evidence file, characterize baselines and probe the simulator itself. As a benchmark-infrastructure artifact this is squarely in the D&B track's scope, and the reproducibility discipline is well above track norms: I verified the evidence artifacts exist and the paper's candor about its own failures (the realism gate failing 23 of 24 certified runs) is rare and commendable. The decisive gaps for this venue are the absence of any learning-based experiment in a paper titled an RL environment, a related-work section that compares against only one market simulator and omits the existing trading-RL environment landscape, a blurred contribution boundary with the companion SharpeBench paper whose bug fix supplies the headline finding, and a stale claim-status paragraph that contradicts the experiments section. All four are repairable without new core machinery, hence Major Revision rather than Reject.

---

## Strengths

### S1: The verifiability triad is genuinely novel among trading environments
Leak-freedom by interface shape rather than by rule, byte-identical determinism across native, WASM, and Python surfaces enforced by committed FNV-1a golden hashes, and decisions-only trajectories whose results are recomputed at verification time form a coherent adversarial-verifiability design that no cited or uncited market simulator I know of targets as a package. The paper names the enforcing code for each property, which is exactly the register the D&B track asks for.
**Evidence Anchor**: `text: §2 "The first three principles are properties of committed code, checkable today by running the listed tests"`

### S2: Reproducibility discipline is exemplary and artifact-backed
Every reported number traces to a committed script with fixed seeds writing JSON evidence, an appendix lists the exact command per finding, and a figure regenerator lets readers confirm no figure contains a number the evidence lacks. I confirmed `paper/evidence/f1-baselines.json` through `f8-ecology.json`, the nine `make-f*.py` scripts, and all figure PDFs exist in the repository, with the package version recorded in the evidence.
**Evidence Anchor**: `dataset: paper/evidence/f1-baselines.json — package_version field 0.8.0, eight evidence JSONs matching Appendix A`

### S3: Honest self-incriminating findings
F4 reports the generator failing its own stylized-facts conjunction on 23 of 24 certified runs and states it as a finding against the generator, and the limitations section extends this candor to the baselines, the probe layer's exclusion from the golden-hash guarantee, and the deny-list guard's blind spots. This is the opposite of benchmark marketing and materially raises trust in the rest of the paper.
**Evidence Anchor**: `table: Table 4 (tab:f4) — all-facts pass rate 0.000 / 0.000 / 0.125 across tiers`

### S4: Generalization made a measured quantity via Procgen-style seed bands
Porting integer-seed-interval procedural generation with disjoint train, gap, and held-out bands to market scenarios, plus a cross-regime transfer matrix that holds seeds fixed while swapping the regime, gives the field a concrete overfitting instrument, and F3's 0.97 calm-to-extreme gap against a near-zero within-tier gap demonstrates the instrument detects what a within-tier split is structurally blind to.
**Evidence Anchor**: `text: §3.4 "Overfitting is then a measurement"`

### S5: The governed wire contract is a well-argued ecosystem contribution
A versioned, additive-only JSON contract with published schemas, conformance fixtures, and a written deprecation policy, explicitly decoupled from package versions, is an unusual and valuable degree of interface governance for a research environment, and the argument for why the host drops out of the trust equation is clearly made.
**Evidence Anchor**: `text: §4 "A breaking change never mutates the v1 types in place"`

---

## Weaknesses

### W1: No learning-based experiment in an RL environment paper
**Problem**: All eight experiments use fixed reference policies and closed-form allocators; no agent is trained on the environment anywhere in the paper. The limitations section concedes this, and concedes that the generalization-gap control in F3 says nothing about whether the environment resists overfitting by learners.
**Evidence Anchor**: `text: §7 "they do not establish what a trained agent scores"`
**Why it matters**: The D&B track's central acceptance question for an environment paper is whether the environment demonstrably supports the research it claims to enable. The paper ships Gymnasium, PettingZoo, and a `verifiers` training surface, yet never exercises any of them with a learner. Without even one trained agent (a PPO run through the Gymnasium adapter on the three tiers, reporting the generalization gap and transfer matrix for a policy that can actually overfit), the environment's core instruments are calibrated only on controls, and reviewers cannot judge whether the task suite is solvable, trivially exploitable, or degenerate for learning agents.
**Suggestion**: Add one modest learning experiment: a standard RL baseline trained on the train band per tier, evaluated on the held-out band and cross-regime, scored through the same kernel. This need not be strong; it needs to exist. If compute is the constraint, say so explicitly and reframe the paper's claims as environment-and-protocol only, with the title and abstract adjusted accordingly.
**Severity**: Major
**Confidence**: 5 — core expertise: benchmark and environment track expectations

### W2: Related work omits the existing trading-RL environment and market-simulator landscape
**Problem**: The market-simulation paragraph compares against exactly one system, ABIDES. The bibliography is roughly a dozen entries. Widely used or directly overlapping systems are not discussed: FinRL and TradeMaster (trading RL environment suites), ABIDES-Gym (the Gym wrapper of the one cited simulator), mbt_gym (an Avellaneda-Stoikov RL environment, directly overlapping the F2 task), gym-trading-env and similar open environments, and the learned/generative LOB simulator literature that also confronts realism certification.
**Evidence Anchor**: `text: §6 "ABIDES \citep{byrd2020abides} is the closest simulator in spirit"`
**Why it matters**: Originality is this paper's strongest card, and it is currently asserted rather than demonstrated. A D&B reviewer who knows mbt_gym exists will read the market-making task as re-implementing prior art without citation, even though the verifiability triad genuinely differentiates SharpeArena. The paper's own claims survive the comparison; the absence of the comparison is what will not survive review.
**Suggestion**: Expand the related-work section with a table or paragraph positioning SharpeArena against the trading-RL environment landscape on the axes the paper actually competes on: leak-freedom mechanism, determinism guarantee, trajectory verifiability, external-agent contract, generalization protocol. Cite mbt_gym at F2 specifically.
**Severity**: Major
**Confidence**: 4 — adjacent expertise: RL environments and finance-ML tooling literature

### W3: Blurred contribution boundary with the companion SharpeBench paper
**Problem**: The paper leads, in both abstract and introduction, with F1, whose substance is the discovery and correction of a unit bug in the companion paper's scoring kernel, explicitly identified as "the same unit bug the companion SharpeBench paper reports as its Finding 1." The scorer, the deflation logic, and the eligibility gates all live in the companion; this paper delegates ranking by design.
**Evidence Anchor**: `text: §5.1 "This is the same unit bug the companion SharpeBench paper reports as its Finding 1"`
**Why it matters**: Venue reviewers must be able to attribute each claim to exactly one paper. The strongest advertised empirical finding of this submission is, on its face, the other submission's finding, which invites overlap concerns and weakens the perceived independent contribution of the environment paper. It also makes the environment's headline dependent on a scorer this paper deliberately does not specify.
**Suggestion**: Lead the abstract and the findings summary with results the environment alone owns (F2 regret against the closed-form optimum, F3 transfer, F5 manipulation boundary), and reposition F1 as a validation that the producer-scorer separation lets the pipeline catch scorer bugs, with an explicit one-sentence statement of what is claimed here versus in the companion.
**Severity**: Major
**Confidence**: 4 — competence basis: track policy on overlapping companion submissions

### W4: Stale claim-status paragraph contradicts the experiments section
**Problem**: The final paragraph of Section 2 states that the empirical findings' "numbers arrive when the evidence run executes against the integrated kernel," implying the eight findings are not yet run. Section 5 then reports concrete executed numbers throughout, the evidence JSONs are committed, and the abstract quotes the results as facts.
**Evidence Anchor**: `text: §2 "their numbers arrive when the evidence run executes against the integrated kernel"`
**Why it matters**: For a paper whose entire thesis is that claims must be checkable against committed artifacts, a paragraph that leaves the reader unable to tell whether the results are real or pending is a self-inflicted credibility wound, and a reviewer who reads Section 2 carefully will flag the whole experiments section as ambiguous in status.
**Suggestion**: Rewrite the paragraph in the past tense to match reality: the design properties are checkable by running the listed tests, and the eight findings were produced by the committed evidence run at the pinned versions.
**Severity**: Major
**Confidence**: 5 — competence basis: internal consistency check across sections

### W5: Abstract is overlong, results-dense, and assumes the companion paper
**Problem**: The abstract runs well past track norms, packs in unexplained apparatus (deflated Sharpe saturation, per-run gates, "mis-united floor," eligibility conjunctions) that only makes sense to a reader who already knows SharpeBench, and the title's definite article ("The Point-in-Time (PIT) RL Environment") plus the README-grade positioning language in the introduction ("The strategic bet is interface ownership") read as product marketing rather than scholarly framing.
**Evidence Anchor**: `text: abstract "The corrected scoring kernel rewrites the baseline board"`
**Why it matters**: The abstract is the venue-fit surface. A track reviewer triaging submissions should come away with environment, properties, protocol, and one or two headline results; instead they get a compressed results dump keyed to another paper's scorer, which undersells the genuinely novel design contribution and oversells ecosystem ambition.
**Suggestion**: Halve the abstract: three sentences on the three structural properties, one on the contract and protocol, two on environment-owned findings. Change the title's "The" to "A". Move the interface-ownership strategy into a short discussion paragraph with a neutral register.
**Severity**: Minor
**Confidence**: 4 — competence basis: venue abstract conventions

### W6: Significance is conditional on adoption that has not started
**Problem**: The paper's significance case rests on the environment becoming shared infrastructure, yet no third party has submitted an agent, no hosted leaderboard exists, and the sole generator family fails its own realism conjunction, so demonstrated skill is skill against this generator's structure. The paper says all of this honestly in Section 7.
**Evidence Anchor**: `text: §7 "no third party has yet submitted an agent"`
**Why it matters**: The track accepts pre-adoption infrastructure, so this is not disqualifying, but the current framing (an "OpenAI-Gym moment," ecosystem ownership) writes checks the evidence cannot yet cash, and the gap between rhetoric and adoption status will draw reviewer skepticism that a plainer framing would avoid.
**Suggestion**: Frame significance as verifiability infrastructure whose properties are checkable today, with adoption as future work; consider adding any early external-usage signal available by revision time (downloads, conformance-kit users, a worked third-party agent example).
**Severity**: Minor
**Confidence**: 4 — competence basis: significance assessment for infrastructure papers

---

## Detailed Comments

### Journal Fit
The submission is the right artifact type for NeurIPS D&B / Evaluations and Datasets: an environment plus protocol plus governance paper with released multi-registry packages, CI-enforced properties, and per-number reproduction commands. Fit is specific, not generic: the reproducibility statement, the commands appendix, and the seed-band evaluation protocol map directly onto the track's checklist culture. The fit deficit is equally specific: the track expects an environment paper to demonstrate the environment doing its job for learning agents (W1), and expects positioning against the environments its readers already use (W2). With those repaired this is a natural fit. If the authors cannot add a learning experiment, an alternative venue shape would be a benchmark-infrastructure or ML-for-finance venue where protocol papers without trained baselines are more standard, but I would prefer the paper repaired for this track.

### Originality
The individual ingredients are borrowed and credited: Procgen's seed intervals, Gym's interface bet, ALE's determinism lesson, classical microstructure models. The original contribution is the composition: an adversarially verifiable trajectory producer where leak-freedom, byte-determinism, and tamper evidence are structural properties with named enforcing code, plus a protocol-governed external-agent contract. I am not aware of a market environment that offers this package, and the probe layer's "distrust the simulator" stance (realism certification, manipulation payoff boundaries, paired adverse-selection controls, ecology under shocks) is an original evaluative posture. The originality claim is currently under-defended against the uncited trading-environment landscape (W2), which is a presentation failure rather than a novelty failure.

### Significance
Look-ahead leakage and irreproducible backtests are real, chronic failure modes, and a producer that makes them structurally impossible addresses a genuine gap for anyone evaluating trading agents, a growing readership as LLM-agent trading work accelerates. Significance today is bounded by the synthetic-only generator (whose realism gate mostly fails), the absence of trained-agent evidence, and zero external adoption; the paper is candid on all three. Net: significant as infrastructure-in-earnest, with headroom that depends on revision W1 and on adoption the paper cannot yet demonstrate.

### Structural Coherence
Title through principles through environment through contract reads as one argument, and the seven-principles section is an effective organizing device: later sections genuinely read as consequences. Two coherence defects: the stale status paragraph (W4) directly contradicts Section 5, and the F1-first ordering makes the paper's empirical spine belong partly to the companion (W3). There is no over-promising in the limitations direction; if anything the paper under-claims relative to its artifacts.

### Title & Abstract
Title: "The" over-claims; "PIT" is expanded twice (title and abstract), once suffices. Abstract: see W5; it is accurate to the paper but too long, too dense, and keyed to the companion's scorer vocabulary. The strongest sentence in the abstract is the first ("A benchmark scores trajectories; something has to produce them"), and the abstract would be stronger if the rest stayed at that altitude.

### Conclusion
There is no conclusion section; the paper ends with limitations, reproducibility, and the appendix. For this track that is defensible, but a short closing paragraph restating what is claimed as checkable-today versus what awaits adoption would sharpen the contribution boundary and give the reader the intended takeaway rather than ending on licensing details.

### References
Formatting is fine (plainnat); coverage is the problem (W2). Roughly twelve entries is thin for a D&B submission staking an originality claim over two literatures (RL environments and market simulation). The self-citation to the companion is necessary but currently load-bearing for the headline finding.

---

## Questions for Authors

1. Can you add at least one trained-agent experiment (any standard RL baseline through the Gymnasium adapter) before revision, and if not, will you narrow the paper's claims and title to environment-and-protocol accordingly?
2. What precisely is claimed as a contribution of this paper versus the companion SharpeBench paper, and can you state that boundary in one sentence in the introduction? In particular, whose finding is the F1 unit bug?
3. How does the market-making task relate to existing Avellaneda-Stoikov RL environments such as mbt_gym, and how does SharpeArena position against FinRL, TradeMaster, and ABIDES-Gym on the axes the paper competes on?
4. Is the Section 2 closing paragraph ("their numbers arrive when the evidence run executes") a leftover from a pre-evidence draft, and do all reported numbers in Section 5 correspond to the committed evidence at sharpearena 0.8.0 / SharpeBench 0.5.0?

---

## Minor Issues

### Language / Grammar
- §2 final paragraph tense conflicts with §5 (also raised as W4; the fix is one paragraph).
- Abstract, "a mis-united floor": likely intended "mis-unitted" or plainer "a floor set by a unit error"; as written it parses oddly.

### Citation Format
- `minari` bibliography entry is cited without author-year style consistency in the text (\citep{minari}); confirm the entry renders with authors under plainnat.
- Consider citing Gymnasium's `check_env` and PettingZoo at first functional mention in §3 rather than only in §6.

### Figures and Tables
- Table 1 caption says "No policy is rank-eligible on any tier" while the table itself shows only DSR and pass^k; a footnote stating the eligibility rule (pass^k = 1.00 required) inside the table caption would make it self-contained.
- Figure 3 and the right panel of Table 3 present the same matrix; one could be dropped or the figure moved to an appendix for space.

### Layout
- The abstract exceeds typical NeurIPS abstract length; check the style file's expectations for the camera-ready.
- "PIT" expanded in both title and abstract; expand once.

---

## Criterion-Bound Judgements

Calibration status: `NOT_CALIBRATED`

| Dimension | Criterion source | Judgement | Evidence anchor(s) | Rationale | Uncertainty / scope limit | Decision bearing? |
|---|---|---|---|---|---|---|
| Venue fit (NeurIPS D&B) | Track call: environments/benchmarks with demonstrated utility and released artifacts | PARTLY_MEETS | `text: §7 "they do not establish what a trained agent scores"` | Right artifact type with exemplary artifact backing, but an RL-environment paper with zero learning experiments misses the track's demonstrated-utility expectation | Track norms vary by year and area chair | yes: primary repairable blocker (W1) |
| Originality | Step 2 of review protocol; comparison against Gym-family and market simulators | MEETS | `text: §6 "properties ABIDES does not target"` | The verifiability triad plus governed contract is a novel composition among trading environments | Under-defended against uncited environments (W2); a system I do not know could overlap | yes: the paper's strongest card, contingent on W2 repair |
| Significance | Step 3; impact if conclusions hold, breadth of readership | PARTLY_MEETS | `text: §7 "no third party has yet submitted an agent"` | Addresses real chronic failures; impact conditional on adoption not yet begun and on a generator failing its own realism gate | Adoption trajectory unknowable at review time | yes: caps recommendation below Accept, does not force Reject |
| Structural coherence | Step 4; title-abstract-body-conclusion consistency | PARTLY_MEETS | `text: §2 "their numbers arrive when the evidence run executes"` | One argument well carried, broken by a stale status paragraph and a companion-owned headline finding | none identified | yes: W3/W4 are cheap fixes with outsized credibility effect |
| Title & Abstract | Step 5; venue readership conventions | PARTLY_MEETS | `text: abstract "The corrected scoring kernel rewrites the baseline board"` | Accurate but overlong, companion-keyed, and over-claiming in the title's definite article | Register judgement, partly taste | no: quality issue, not decision-bearing alone |
| Readership relevance | Track readership: RL evaluation + ML-for-finance | MEETS | `text: §1 "An agent is a program in any language that reads a JSON Observation and writes a JSON Decision"` | Language-agnostic contract plus Gymnasium/PettingZoo/Minari surfaces address both communities | Finance-domain share of D&B readership is a minority | no |

Explain the recommendation by naming the unresolved decision-bearing criteria and their repairability: the recommendation is Major Revision because venue fit (W1), originality defense (W2), and structural coherence (W3, W4) are each decision-bearing and each repairable without new core machinery: one learning experiment, an expanded related-work section, a re-led abstract with an explicit companion boundary, and one rewritten paragraph. No strength offsets these numerically; equally, none of the failures is fatal, since the underlying artifacts already exist and the missing work is demonstrative and positional, not architectural.
