# SharpeArena manuscript: prose, structure and AI-slop review

Date: 2026-08-25. Scope: `paper/main.tex` and all 18 files in `paper/sections/`, including the eight `\input` fragments.
Mode: read-only. No manuscript file was modified.

Detectors applied: `anti-slop`, `humanizer`, `scientific-writing`, `research-paper-writing`, `every-style-editor`, `skimmable`, `paper-narrative`, `no-ai-slop` (detect-only), `humanink` (`--academic --light`, detect-only).
Detector output is treated as candidate evidence, not as a verdict. Section 6 lists every pattern the detectors flagged that is correct usage in a quantitative paper, marked as a false positive with the reason.

---

## 0. One-paragraph verdict

The manuscript has an unusual problem for an AI-slop review: it is not promotional, not padded with significance inflation, and not vague. It has been edited hard in the opposite direction, and the over-correction has become its own tic. The paper's dominant rhetorical move is the epistemic disclaimer appositive, "X, not Y" and "X rather than Y", which appears 104 times, and its dominant syntactic move is the assertion-then-gloss colon, which appears 76 times. Both started as honesty and have become rhythm. The second problem is architectural: the abstract and the experiments section are both organized as inventories rather than as arguments, and the eight fragments are visibly appended (they change person, they re-establish context the body already established, and three of them carry build comments into the typeset source directory). Fixing the four items in Section 7 would take the paper from "defensively over-qualified" to "confident and precise" without softening a single claim.

---

## 1. Abstract

### 1.1 Diagnosis

`sections/00-abstract.tex:2` is a single 374-word paragraph of 11 sentences. Structural problems, in order of severity:

**(a) No context, no gap, no implication.** The abstract is entirely approach plus results. A reader arrives at "SharpeArena is a point-in-time reinforcement-learning environment" with no statement of what is wrong with the current situation, and leaves after "does not survive replication" with no statement of what any of it means. Move structure is A-R, missing C, G and I. The material for the missing moves already exists in the paper: `01-introduction.tex:3` has the gap ("every one of its guarantees is conditional on the trajectory being honest") and `07-limitations.tex:19` has the implication ("The significance claim of this paper is therefore the verifiability infrastructure that is checkable today"). Neither reaches the abstract.

**(b) Three stacked relative clauses before the first verb of consequence.** Quoted:

> "SharpeArena is a point-in-time reinforcement-learning environment for trading agents whose interface excludes future-bar access, whose Rust core is pinned byte-for-byte by golden fingerprints its native, WebAssembly and Python surfaces each assert, and whose trajectories are recomputed from recorded decisions."

Sixty-one words, three coordinated `whose` clauses, the third one nested three levels deep ("golden fingerprints its native, WebAssembly and Python surfaces each assert" has a reduced relative inside a relative). The reader is asked to hold three architectural properties in working memory before being told why any of them is worth having. The rule-of-three shape here is also the humanizer #10 pattern, though in this case the triad is factual (the three are the paper's three actual guarantees), so the fix is not to break the triad but to give it a reason to exist first.

**(c) Nine of eleven sentences are result-listing, at uniform grain.** The 60-cell extension breakdown ("23 positive, 31 negative and 6 unresolved intervals") sits at the same emphasis level as the paper's headline realism failure. An abstract that gives equal space to everything gives emphasis to nothing.

**(d) Density of unglossed compound modifiers.** "finite-panel-calibrated realism diagnostic", "conditional-IID-calibrated Fano statistic", "normalized-flow concave-impact ablation", "predeclared diagnostic, confirmation and volatility-constrained calibration rule". Each is precise and each is defined in the body, but the abstract uses them before definition. Four such terms in a 374-word abstract is a wall.

**(e) The honest scope statement is buried mid-paragraph.** "The experiments calibrate these instruments rather than make market claims" is sentence 3 of 11, sandwiched between the contract sentence and the realism failure. It is the single most important framing sentence in the paper and it should be positioned where it governs everything after it, which it currently does by luck of ordering rather than by construction.

**(f) What is right and must be preserved.** The abstract states its worst result. "The canonical generator fails its own finite-panel-calibrated realism diagnostic on 23 of 24 seeded panels" is in sentence 4. The negative-result honesty ("no Calm setting survives", "which is not evidence of equivalence", "does not survive replication") is genuinely unusual and is the paper's strongest credibility signal. Any rewrite that softens it is worse, not better.

### 1.2 Proposed replacement abstract

Same facts, same scope, restructured as context, gap, approach, results, implication. 322 words.

> Reinforcement-learning results in trading are usually reported by the party that produced them, on environments that can leak future information without anyone noticing, that return different numbers on different machines, and whose runs no third party can recompute. A scoring rule cannot repair any of this. A deflation prior or a reliability gate is only as sound as the trajectory it consumes, so the guarantees have to be built into the producer rather than the scorer.
>
> SharpeArena is a point-in-time environment that makes those three failures unavailable rather than discouraged. Its data layer exposes no operation that returns a bar after the environment's own cursor. Its Rust core is restricted to arithmetic that is exact on every platform and pinned by golden fingerprints that the native, WebAssembly and Python surfaces each assert. Its runs record only the agent's decisions, so returns are recomputed at verification time and a doctored trajectory replays to something else. Agents connect through a governed JSON contract; scenarios are procedural functions of an integer seed drawn from disjoint train, gap and held-out bands; ranking is delegated to the separate SharpeBench kernel.
>
> The experiments calibrate these instruments with fixed reference policies. No agent is trained and no result is a claim about markets. The generator fails its own realism diagnostic on 23 of 24 seeded panels, and a predeclared sweep of the existing volatility-clustering knobs finds no Calm setting that survives it. Fixed-spread market making traces a U-shaped regret curve against the Avellaneda-Stoikov reference whose minimum 16 episodes cannot separate from the reference, which is not evidence of equivalence. Under normalized concave impact an asymmetric round trip pays 21.8e-4 with a Bonferroni familywise interval of [10.0, 33.6]e-4 and 26.3e-4 on a fresh 32-seed replication, while every sampled linear-impact cell loses money. Making the maker's own fills move the price drives informed-flow markout at 20 bars from +0.095 to -0.196 across a six-value impact sweep, while the paired informed-versus-uninformed gap stays positive. A public 2^16 seed band is inverted from one observed bar in 16 of 16 trials; an opt-in salted derivation defeats that scanner 0 of 16 while the salt is withheld. An out-of-band oracle witness shows the rank-eligibility set is non-empty on every tier and that the every-seed probabilistic-Sharpe gate is the last to open at every crossing. An eight-seed ecology replication reproduces its own single-seed narrative once.
>
> What the paper offers is a producer whose guarantees are properties of committed code and recomputable evidence rather than of the authors' good faith, together with the diagnostics that say plainly where its tape is not yet market-like.

Notes on the rewrite, so the author can accept or reject each choice deliberately:

- **Facts demoted to the body, not deleted:** the 60-cell extension breakdown (23/31/6 over the 195-cell family), the concave ablation's role in making the probe falsifiable, the exact pointwise interval [17.4, 26.2]e-4, the Fano statistic's exploratory status, the five-noise-path replication of the witness, and the phrase "all ten eligible cells". All are in Section 5 and the fragments. If the author prefers a strict superset, insert the pointwise interval and the Fano clause and accept about 360 words; the C-G-I structure survives either way.
- **"which is not evidence of equivalence" is retained verbatim.** It is the most reviewer-proof sentence in the abstract.
- **Three sentences replace the three `whose` clauses.** Same content, same order, one property per sentence, each with an active verb.
- **The scope sentence is promoted to the head of the results block** so it governs the eight results that follow instead of sitting fourth.
- **The final sentence is the implication move**, drawn from `07-limitations.tex:19`, and stops short of any adoption or realism claim.
- LaTeX note: the rewrite uses plain hyphens in "Avellaneda-Stoikov" for readability here; in the source keep `Avellaneda--Stoikov` and restore the math mode for the interval values.

---

## 2. Introduction (`sections/01-introduction.tex`)

### 2.1 Rhetorical job

Establish territory (RL environments for trading exist and are used), establish the niche (they cannot support a leaderboard because the producer is untrusted), occupy it (SharpeArena excludes three specific failures structurally), preview (contributions and what the evidence shows).

### 2.2 Does it do the job

Partly. Occupation and preview are strong. Territory and niche are outsourced.

**Finding I-1: the paper cedes its own footing in sentence two.** `01-introduction.tex:3`:

> "An eval is useless without an environment. The companion SharpeBench paper \citep{toca2026sharpebench} argues that ranking trading agents on raw return over short windows measures luck, and supplies a deterministic scoring kernel..."

The opening line is an aphorism (humanink #26, no-ai-slop "aphorism formulas") that asserts rather than establishes, and it is immediately cashed out by handing the argument to another paper. Forty-one of the first paragraph's opening words describe SharpeBench. A reviewer who has not read the companion has, at the end of sentence two, learned nothing about this paper's problem that this paper has established. The dependency is real and should be cited, but the paper must own its own gap first. Note also that the aphorism is not even the paper's actual claim: the paper's claim is the reverse dependency, that an environment is useless if its trajectories are not verifiable.

**Proposed replacement for `01-introduction.tex:3`, opening sentences:**

> Every published number in agent trading evaluation is produced by the party being evaluated, on an environment that party controls. Three failures upstream of scoring void any score, and none of them is visible in the score itself. An agent that observes one future bar has an edge no deflation can remove, and look-ahead bugs are a pervasive and usually invisible failure mode of backtests \citep{bailey2014deflated}. An environment that is not bit-reproducible cannot support a leaderboard, because the same submission scores differently on different machines and nobody can check anyone else's number. And a trajectory that cannot be recomputed from the environment that allegedly produced it is a self-reported claim, not evidence. The companion SharpeBench paper \citep{toca2026sharpebench} supplies the scorer that consumes these trajectories, deflating for search breadth \citep{bailey2014deflated}, gating on per-run reliability \citep{yao2024taubench} and zeroing entries that bypass risk controls; every one of its guarantees is conditional on the trajectory being honest, which is the problem this paper takes.

This keeps every fact and every citation, opens on the paper's own territory, and moves the companion from premise to consequence.

**Finding I-2: paragraph 2 is a rule-of-three built from three identically shaped sentences.** `01-introduction.tex:5`:

> "Look-ahead is prevented by the shape of the data layer... and additionally policed by a deny-list guard... Determinism is achieved by restricting the scenario generator's arithmetic... and additionally pinned by committed golden hashes... Trajectory honesty is achieved by recording only decisions... and additionally exercised by a test that doctors a trajectory..."

Three sentences, identical template `[property] is [achieved/prevented] by [structure], and additionally [verb] by [detector]`. The parallelism is defensible once, because the paper's actual design method is uniform, and the paragraph says so ("its design method is uniform"). But the sentence announces the parallelism and then performs it, which is one layer of redundancy. Cut the announcement or cut one of the three restatements. Recommended: keep the three sentences, delete "and its design method is uniform:" and let the shape carry it, since showing beats telling (no-ai-slop, interpretive metadiscourse).

**Finding I-3: `\paragraph{What the evidence shows.}` duplicates the abstract at roughly 90 percent overlap.** `01-introduction.tex:13` restates ten results the abstract has already given, in the same order, at the same grain, with the same qualifiers. Compare abstract "no Calm setting survives a predeclared diagnostic, confirmation and volatility-constrained calibration rule" against intro "also finds no setting that survives its diagnostic, confirmation and volatility rule". This is the humanink #28 mirror-conclusion pattern operating between two sections. The intro version should either be cut entirely (the contributions paragraph plus cross-references would carry it) or reduced to three sentences that name only what the reader needs before Section 2: the tape is uncertified, no baseline is rank-eligible, and the eligibility set is nevertheless non-empty. Everything else is `\cref` bait.

**Finding I-4: the contributions list is one 340-word paragraph with six inline numbered items.** `01-introduction.tex:11`. Item (5) alone is 92 words with three subordinate clauses. This is the `skimmable` failure mode: a numbered list rendered as prose is neither scannable nor readable. Recommend `\begin{enumerate}[leftmargin=*,itemsep=2pt]` and a hard cap of 35 words per item, with the qualifications moved to the sections they qualify.

**Finding I-5: structurally correct and worth preserving.** `\paragraph{Scope and contribution boundary.}` is genuinely good practice and unusual. Keep it verbatim.

---

## 3. Section-by-section structural findings

### 3.1 Design principles (`02-principles.tex`)

**Job:** give the reader a small set of commitments so the rest of the paper reads as consequences rather than as a feature list. **Does it:** yes. This is the best-structured section in the paper. The seven bolded principles are each one claim with one consequence, and `02-principles.tex:17` ("Distrust the simulator too") does the rare thing of naming the standard the paper's own results will later fail.

Two defects:

- `02-principles.tex:3`: "Every decision in the environment traces to one of seven principles, stated here so the rest of the paper reads as their consequences." This is a fragmented-header warm-up (humanizer #29) plus a promise the paper cannot audit. Cut to "Seven principles govern the design." or delete outright, since the section heading already says "Design principles".
- `02-principles.tex:19`: "One consequence is worth stating before the experiments." This is `it is worth noting` in costume (humanink #30, no-ai-slop "often-empty phrases"). The paragraph that follows is substantive. Open it with its own content: "The first three principles are properties of committed code, checkable by the listed tests."

### 3.2 The environment (`03-environment.tex`)

**Job:** describe the artifact precisely enough that the guarantees in Section 2 are believable. **Does it:** yes, and it is the most information-dense section. But it carries the section-length problem: `03-environment.tex:21` (procedural scenarios) is a single 320-word paragraph containing the band protocol, two Rust function descriptions, the F3 subset caveat, the Python namespace reservation, the sealed-seed pointer and the two generalization metrics. Six messages, one paragraph. `research-paper-writing` global principle 1 (one paragraph, one message) is violated more here than anywhere else in the manuscript. Split at "The Python layer additionally reserves" and again at "Overfitting is then a measurement".

`03-environment.tex:27` is the paper's best paragraph and should be a model for the rest: it states a restriction, names exactly what the restriction costs (latency races, queue-position value, sniping of stale quotes, no maker-taker fees), and tells the reader how to read the results inside it.

### 3.3 The agent contract (`04-contract.tex`)

**Job:** convince an implementer the interface is safe to build against. **Does it:** yes, and it is the shortest, cleanest section. The four subsections map to four questions an implementer actually asks. No structural defect.

One line to fix. `04-contract.tex:3`: "Everything else in this paper exists to make that loop trustworthy; this section is the loop's promise of stability." The second clause is a signpost that describes the section rather than doing it (humanizer #28). Cut after "trustworthy".

### 3.4 Experiments (`05-experiments.tex`) and the eight fragments

**Job:** report ten findings such that each one's scope, dispersion convention and provenance is checkable. **Does it:** the individual findings, yes, exceptionally well. The section as a navigable object, no.

**Finding E-1: the section explains its own layout for 280 words before reporting anything.** `05-experiments.tex:3` and `:5` are two paragraphs of pure meta: how many findings, how many subsections, what the placement rule is, which subsection follows which finding, three scoping statements, three dispersion conventions. The reader gets no result until line 9. This is the most severe over-signposting instance in the manuscript, and it is a symptom rather than a cause. The real problem is that fifteen flat `\subsection` peers genuinely cannot be navigated, so the author wrote a map. **Fix the structure and the map becomes unnecessary.**

**Finding E-2: proposed regrouping.** Fifteen flat subsections at one level, in order F1, witness, F2, F3, F4, calm-tails, F5, concave, positive-control, extended-sweeps, F6, endogenous, F7, F8, predictability, sealed-seeds, is not navigable from the table of contents. Three named `\subsection` groups with `\subsubsection` members give the reader the same content with a shape:

> **5.1 Does the pipeline measure anything? (scoring and generalization)**
> 5.1.1 F1 baseline board under the corrected kernel
> 5.1.2 Rank-eligibility witness
> 5.1.3 F2 regret against the closed-form reference
> 5.1.4 F3 generalization gap and cross-regime transfer
>
> **5.2 Do we believe the simulator? (probes against the environment)**
> 5.2.1 F4 stylized-facts realism
> 5.2.2 Calm calibration: a negative result
> 5.2.3 F5 manipulation boundary (linear)
> 5.2.4 Normalized-flow concavity
> 5.2.5 Asymmetric positive control
> 5.2.6 Extended sweeps
> 5.2.7 F6 adverse selection, exogenous and endogenous arms
> 5.2.8 F7 failure modes
> 5.2.9 F8 ecology under shocks
>
> **5.3 What can an adversary do with a public generator? (predictability)**
> 5.3.1 Predictability probe
> 5.3.2 Sealed evaluation seeds

Each group takes a two-sentence opener stating its question and its scoping convention. The 280-word map at `:3` and `:5` then reduces to about 80 words: the dispersion conventions (which are genuinely necessary and should arguably be a small table rather than a sentence) and the single provenance pointer. The placement-rule paragraph disappears entirely, because grouping makes placement self-evident.

**Finding E-3: do the fragments read as integrated prose or as appended blocks?** Eight fragments, assessed individually.

| Fragment | Reads as | Evidence |
|---|---|---|
| `concave-fragment` | **Integrated.** | Opens by naming what the preceding F5 sweep could not do ("cannot return a positive even in principle") and closes by handing off to the next fragment. Best of the eight. |
| `positive-control-fragment` | **Integrated.** | Picks up the concave fragment's handoff explicitly, and its closing paragraph reconciles both results. |
| `endogenous-adverse-fragment` | **Integrated in prose, appended in source.** | Paragraph 1 is a model of how to open a follow-up: it names the exact limitation of the parent finding and says what the arm adds. But lines 1 to 5 are build comments (`% Fragment: the endogenous arm...`, `% Producer: python paper/src/...`) sitting in the typeset source tree. |
| `arena-witness-fragment` | **Mostly integrated.** | Opens correctly from F1's empty board. But it shifts to "We answer by construction", the manuscript's first first-person plural, well after a body that has been strictly impersonal. |
| `predictability-fragment` | **Integrated.** | Opens from `\cref{sec:limitations}`, uses three `\paragraph` heads that make it the most skimmable unit in the paper. Its `\paragraph{What remains open.}` head is the model the other seven should copy. |
| `sealed-seeds-fragment` | **Integrated.** | Opens from the attack the previous fragment measured. Short, single-purpose. |
| `manipulation-sweeps-fragment` | **Appended.** | Opens "The positive control leaves push weight, $\lambda$, total leg length, hold length, short-side trips and follower gain outside its initial grid", a six-item list of omissions with no statement of what the sweep is for. Carries a build comment at line 1. Contains "Importantly," at line 11, the manuscript's only unearned emphasis adverb. |
| `calm-tails-fragment` | **Appended.** | Alone among the eight it is a `\paragraph`, not a `\subsection`, so the F4 finding and its follow-up sit at different structural levels for no stated reason. Carries two build comments. Its second sentence "This calibration matters:" is interpretive metadiscourse telling the reader the importance of what follows instead of showing it. |

Aggregate: five of eight read as integrated prose, three as appended blocks, and the three appended ones are exactly the three carrying `%` build comments into the source. The comments are a reliable tell and a mechanical fix.

**Finding E-4: uniform paragraph openings inside the experiments section.** Of 49 sentence-initial `The` occurrences in the manuscript, a large share cluster here as a fixed result-paragraph template: `The reading:` (`:37`), `The result` (`:173`), `The point of` (`:54`), `The within-tier gaps are` (`:92`), `The transfer matrix ... is where` (`:94`), `The level result` (`:219`), `The replication does not support` (`:287`), `The conclusion is methodological.` (`:289`). Eight result paragraphs open with a definite-article abstraction plus a copula. The content differs; the shape does not. Vary at least half by opening on the number or the mechanism instead: "Eligibility is a conjunction, and each leg catches what the other misses." reads better than "The reading: eligibility is a conjunction..." and loses nothing.

### 3.5 Related work (`06-related.tex`)

**Job:** position against the four property axes the paper competes on. **Does it:** yes, and the explicit axis list at `:7` ("leak-freedom mechanism, determinism guarantee, trajectory verifiability, and the external-agent contract, with the generalization protocol as a fifth") is good practice: it tells the reader what comparison is being made before making it.

One structural defect. `06-related.tex:7` is a single 430-word paragraph covering five systems. Each system gets two to four sentences and they run together. Either one paragraph per system, or a comparison table on the five named axes with the paragraph reduced to what a table cannot carry. Given that the axes are already enumerated, a table is the stronger option and would make the section's central claim ("none of these systems, to our knowledge, sets out to...") checkable at a glance instead of at the end of 430 words.

The epistemic hedging here is correct and should not be touched: "Where these descriptions rest on our reading of the systems' papers and repositories rather than on properties their authors claim, we have kept them descriptive". That is a false positive for vague attribution; see Section 6.

### 3.6 Limitations (`07-limitations.tex`)

**Job:** bound every claim the paper makes. **Does it:** yes, thoroughly, and this section is the paper's strongest asset. Eight `\paragraph` heads, each a full sentence naming the limitation, is exactly right for skimmability.

Two defects, both minor:

- The section is roughly 1,050 words of unbroken qualification and lands immediately after 5,100 words of experiments that are themselves heavily qualified. By this point the reader has absorbed roughly 100 scope disclaimers. Cutting the duplicated ones (see slop items 22 to 33) matters more here than anywhere, because the ones that remain need to land.
- `07-limitations.tex:3` closes with "the framing is a constraint on those claims, not a footnote to them". This is a rhetorical flourish about the paper's own rhetoric. It is the kind of line reviewers quote back skeptically. The constraint is demonstrated by the eight paragraphs that follow; it does not need announcing.

### 3.7 Reproducibility statement (`08-reproducibility.tex`)

**Job:** tell a replicator exactly what they need. **Does it:** yes, in three tight paragraphs. No structural defect. `:5` correctly states that a package version is not a complete environment specification, which is more precise than most reproducibility statements. `:7` ("Language models were used to draft and edit prose; every bibliography entry was checked against its source") is the right disclosure at the right length, though see Section 5 on its voice.

### 3.8 Appendix A (`A-commands.tex`)

**Job:** make every number traceable to a command. **Does it:** yes. The structure (one `\paragraph` per finding, verbatim command, output paths, then a note on what else the command does) is uniform and correct, and uniformity is a virtue in an appendix in a way it is not in prose.

One defect: `A-commands.tex:3` mixes the provenance statement with three unrelated notes about F1's version serialization, the F4/F5 producers, and the witness artifact. Split the version-provenance caveat into its own `\paragraph{Version provenance.}` since it is the one thing a replicator most needs and it is currently buried mid-sentence.

---

## 4. Numbered AI-slop list

Format: number, category, `file:line`, quoted text, replacement. Categories use the twelve requested plus detector cross-references. This list is intended as complete for the categories requested; where a category has more than 15 instances of an identical form, the entries give every distinct form plus the full line list for the remainder.

### 4.1 Category A: "not X but Y" used for rhythm, and disclaimer parallelism (dominant defect, 104 instances)

The construction is `assertion, not counter-assertion` or `assertion rather than counter-assertion`. Roughly 40 of these carry real disambiguating work. The rest are rhythmic and, worse, are load-bearing in the reader's fatigue budget: after the twentieth, the reader stops registering the twenty-first, which means the ones that matter stop working.

1. **Rhythmic disclaimer** `01-introduction.tex:9` "so the results calibrate the instruments rather than demonstrate learned performance" -> The same sentence already says "No agent is trained in this paper: every experiment uses fixed reference policies". Delete the clause; it restates its own premise.
2. **Earned disclaimer, keep** `02-principles.tex:5` "An agent cannot peek through the interface because there is nothing to call, not because a rule forbids it." -> Keep. This one earns it; it is the section's whole point. Listed to mark the contrast with the rest.
3. **Rhythmic disclaimer** `03-environment.tex:7` "the property is the shape of the interface, not a rule." -> Delete. Verbatim restatement of item 2, two pages later.
4. **Rhythmic disclaimer** `03-environment.tex:13` "The commitment is pinned, not assumed:" -> "Committed FNV-1a fingerprints of canonical generated scenarios and of a scripted order sequence's fill tape pin the outputs." The colon-gloss that follows already says this.
5. **Rhythmic disclaimer** `05-experiments.tex:52` "the metric's zero point is the reference, not a claim of unbeatability" -> Delete. `03-environment.tex:29` already said "without claiming the zero point is unbeatable" about the same metric.
6. **Rhythmic disclaimer** `05-experiments.tex:52` "The resolution is quantified rather than asserted:" -> Cut the clause, keep the colon's content: "At the observed per-episode dispersion, sixteen paired episodes detect a mean regret of about 12 reward units at 80 percent power..."
7. **Rhythmic disclaimer** `05-experiments.tex:92` "One cross-table difference is a convention, not a discrepancy:" -> "Hard's train-band DSR of 0.1231 here and Table 1's 0.1114 for the same policy on the same sixteen seeds differ only in the declared trial count the deflation prior consumes." The explanation is the disclaimer.
8. **Rhythmic disclaimer** `05-experiments.tex:173` "The module's disclaimer remains explicit: it diagnoses the simulator, not a trading strategy." -> Delete from the results paragraph. It appears again at `07-limitations.tex:21`, which is where it belongs.
9. **Earned disclaimer, keep** `05-experiments.tex:253` "are the signature of that assignment mechanism, not a behavioral finding about agents" -> Keep. Genuinely disambiguating; a reviewer would otherwise misread the counts.
10. **Rhythmic disclaimer** `arena-witness-fragment.tex:28` "The result is an existence proof and a calibration, not evidence about an agent." -> Keep the first half, delete "not evidence about an agent": the paragraph's own later sentences, the fragment's opening, and `07-limitations.tex:7` all say it.
11. **Rhythmic disclaimer, quadruplicated** `manipulation-sweeps-fragment.tex:20` "this is an observed finite-grid maximum, not an optimum" -> The same fragment says "remain a finite grid" at `:16`, "The result expands the diagnostic's search coverage, not its economic scope" at `:21`, and the figure caption at `:31` says "The displayed maxima are sampled cells, not optimized values." Four sayings of one thing in a 33-line fragment. Keep the caption, delete the other three.

**Remaining instances of the same form, with recommended action "keep the first per finding, delete subsequent restatements within the same finding":** `00-abstract.tex:2` (2 instances), `01-introduction.tex:11,13`, `02-principles.tex:7,13,15,17,19`, `03-environment.tex:3,9,17,21,25,27,29,35,41,43,45`, `04-contract.tex:11,19`, `05-experiments.tex:3,5,11,15,35,37,65,69,94,105,109,126,128,143,197,215,217,219,221,255,266,287,289`, `06-related.tex:3,7,9`, `07-limitations.tex:3,5,7,9,11,13,15,17,19,21`, `08-reproducibility.tex:3,5`, `A-commands.tex:3,10`, `arena-witness-fragment.tex:3,8,12`, `calm-tails-fragment.tex:12,24`, `concave-fragment.tex:8,10,19`, `endogenous-adverse-fragment.tex:17,62,64`, `positive-control-fragment.tex:7,9,11,20`, `predictability-fragment.tex:9,11,20`, `sealed-seeds-fragment.tex:8,12`.

**Quantified target:** 104 down to about 45. Rule to apply mechanically: each distinct scope caveat is stated once in its finding and once in `07-limitations.tex`, never three times, and never in a figure caption plus the body plus the abstract.

### 4.2 Category B: assertion-then-gloss colon (76 instances)

Roughly half introduce genuine lists and are correct. The rest are the no-ai-slop "colon reveals" pattern: a claim, a colon, then the claim restated concretely, used for drama.

12. `05-experiments.tex:37` "The reading: eligibility is a conjunction, and each leg catches what the other misses." -> "Eligibility is a conjunction, and each leg catches what the other misses."
13. `05-experiments.tex:37` "The corrected board is the honest zero: the number a real agent has to beat is a positive, deflated Sharpe that survives every evaluation seed, and no reference policy produces one." -> "The number a real agent has to beat is a positive deflated Sharpe that survives every evaluation seed, and no reference policy produces one." ("the honest zero" is a coined aphorism doing no work the rest of the sentence does not do.)
14. `05-experiments.tex:65` "One framing note before the numbers: the policy is fixed and parameter-free..." -> "The policy is fixed and parameter-free, so nothing is trained." Announcing a note before making it is pure signposting.
15. `05-experiments.tex:94` "The wide intervals are themselves a calibration result: a bounded, heavily deflated score bootstrapped over 16 seeds is a coarse instrument" -> "A bounded, heavily deflated score bootstrapped over 16 seeds is a coarse instrument, which is itself a calibration result."
16. `05-experiments.tex:215` "Two quantities must be separated here: the sign of the paired estimated gap and the sign of the levels, and they come out on opposite sides." -> **Keep.** This genuinely sets up the next two paragraphs and the separation is the finding.
17. `05-experiments.tex:217` "That is the signature shape of adverse selection: the loss is in the drift after the fill, not in the fill itself." -> "The loss is in the drift after the fill rather than in the fill itself, which is the signature shape of adverse selection." (Also removes a Category A instance.)
18. `05-experiments.tex:219` "The cause is calibration and design, not accounting:" -> "Three calibration choices produce it:" then the existing list. Names the count, drops the disclaimer, keeps the list colon.
19. `endogenous-adverse-fragment.tex:64` "The scope is stated plainly." -> Delete. The four clauses that follow state it; announcing plainness is the opposite of plainness.
20. `calm-tails-fragment.tex:12` "This calibration matters: no cell satisfies all three predeclared steps, so there is no certified Calm preset and no final/report-band estimate to promote." -> "No cell satisfies all three predeclared steps, so there is no certified Calm preset and no report-band estimate to promote."
21. **Correct usage, keep** `03-environment.tex:13` "The generator is deliberately public and deterministic; that makes the hashes checkable and future bars computable from a recovered seed." -> Semicolon, not colon, and the second clause adds a consequence rather than restating.

**Remaining colon-gloss instances recommended for conversion to plain sentences:** `01-introduction.tex:5,9`; `02-principles.tex:17` (2); `03-environment.tex:7` (2), `:9` (2), `:17` (2), `:29` (2), `:41`, `:43`, `:45` (2); `05-experiments.tex:3` (2), `:11` (2), `:35`, `:52`, `:92`, `:94` (2), `:173`, `:221`; `06-related.tex:3`; `07-limitations.tex:3`, `:21`; `predictability-fragment.tex:7,11`.

### 4.3 Category C: restatement sentences (deletable without loss)

22. `01-introduction.tex:13` the entire `\paragraph{What the evidence shows.}` (297 words) -> Reduce to three sentences or delete. Ninety percent overlaps the abstract. See Finding I-3.
23. `05-experiments.tex:3` "This section reports ten findings across fifteen subsections." plus the placement-rule listing (about 170 words) -> Delete under the regrouping in Finding E-2.
24. `05-experiments.tex:5` "Where the diagnostic applies it is reported beside the results, never required by them." -> Duplicate of `03-environment.tex:41` "Certification is reported beside the results rather than required for the leaderboard". Delete one.
25. `02-principles.tex:3` "stated here so the rest of the paper reads as their consequences" -> Delete. The heading says "Design principles".
26. `04-contract.tex:3` "this section is the loop's promise of stability" -> Delete after "trustworthy".
27. `05-experiments.tex:173` "This finite sweep is not a proof over schedules or parameters outside the grid." -> The sentence before says "consistent with the linear model's no-manipulation conditions on the sampled grid" and the sentence after names the two extensions that cover what it misses. Delete.
28. `concave-fragment.tex:10` "These are finite-grid, pointwise summaries of this symmetric schedule, not a universal no-manipulation result" -> The same paragraph already says "No sampled mean is positive", and the fragment's last line says "The null identifies the schedule's limitation rather than closing the economic question". Delete.
29. `07-limitations.tex:3` "This is why the abstract and \cref{sec:intro} frame every finding that runs on this generator as instrument calibration on uncertified tape; the framing is a constraint on those claims, not a footnote to them." -> Delete both clauses. The paragraph's first four sentences establish it.
30. `07-limitations.tex:5` "\"Leak-free\" therefore means exactly the interface property and no more." -> The paragraph opens "The leak-freedom of \cref{sec:leakfree} is a statement about the interface's shape: no operation returns a post-cursor bar. It is not a claim that public deterministic scenarios are unpredictable." Delete the closing restatement.
31. `05-experiments.tex:126` "The informational Fano ratios are reported as observations only. They do not identify why variance concentrates and do not decide certification." -> The table caption at `:109` says "Fano is a conditional-IID-calibrated exploratory ratio and has no pass column" and the figure caption at `:133` says "Fano is exploratory". Reduce the body to one clause or delete.
32. `arena-witness-fragment.tex:8` "not a proof that eligibility is globally monotone between unsampled strengths" -> `:28` says "any single-path threshold is one draw from that range" and the caption says "drawn only after monotonicity is verified". Keep those two, delete this one.
33. `05-experiments.tex:255` "Whether the same holds for agents that can read their mandate is an open measurement, not a claim of this paper." -> `:253` already said "a mandate-aware field could invert the distribution entirely, and measuring one is the interesting follow-up". Delete.

### 4.4 Category D: over-signposting and interpretive metadiscourse

34. `05-experiments.tex:3` "one placement rule governs the whole section: every subsection sits directly after the finding it extends" plus the six-clause enumeration of which follows which -> Delete; make it structurally true instead (Finding E-2).
35. `05-experiments.tex:5` "Three scoping statements govern the whole section. First... Second... Third..." -> Keep the content, drop the announcement: open directly with "The findings that run on the canonical position-trading generator...". The First/Second/Third scaffolding is fine once the announcement is gone.
36. `02-principles.tex:19` "One consequence is worth stating before the experiments." -> Delete.
37. `05-experiments.tex:253` "One mechanism drives the largest counts and must be stated first:" -> "Mandates are sampled per scenario and assigned to fixed policies that cannot observe or adapt to them, so a structural breach is guaranteed whenever the draw pairs a shorting policy with a long-only mandate."
38. `05-experiments.tex:255` "The headline, scoped to what was measured:" -> Delete the label, keep the sentence.
39. `05-experiments.tex:289` "The conclusion is methodological." -> "One replicator trajectory is one draw from a heavy-tailed outcome distribution, and single-run ecology narratives do not survive replication."
40. `05-experiments.tex:37` "One caveat the board cannot answer on its own:" -> "The board alone does not show that the eligibility set is non-empty for this task family."
41. `03-environment.tex:39` "A synthetic market can flatter agents in ways real markets do not, so the suite probes itself." -> A one-line paragraph restating the subsection title "The probe layer: distrusting the simulator" (humanizer #29, fragmented header). Delete or merge into the Realism paragraph.
42. `endogenous-adverse-fragment.tex:40` "Two things happen when the flow moves the price, and they are the two the design predicts" -> "Both effects the design predicts appear." then First/Second as written. The self-congratulatory framing is unnecessary; the numbers show it.

### 4.5 Category E: unearned emphasis and rubber-stamp qualifiers

43. `manipulation-sweeps-fragment.tex:11` "Importantly, unequal ``uniform'' legs use separate $1/n_{\rm up}$ and $1/n_{\rm down}$ increments" -> Delete "Importantly,". The only instance of this adverb in the manuscript, and it tells rather than shows.
44. `05-experiments.tex:94` "Calm to Extreme excludes zero decisively ($[0.635, 0.999]$)" -> Delete "decisively". The interval is the evidence; the adverb adds nothing an interval does not.
45. `01-introduction.tex:13` "Cross-regime transfer costs the Calm-selected reference nearly its entire DSR on Extreme, decisively so" -> Same word, same claim, second location. Delete "decisively so".
46. `05-experiments.tex:94` "so at 16 seeds only the Calm-to-Extreme collapse is resolved beyond doubt" -> "resolved" alone. "Beyond doubt" is an epistemic overclaim inside a sentence about a wide bootstrap interval, which is self-undermining.
47. `05-experiments.tex:94` "and Calm to Extreme costs $0.973$, essentially the whole score" -> "costs 0.973 of a score bounded in [0,1]". The number is more forceful than the intensifier.
48. `endogenous-adverse-fragment.tex:40` "the gap survives essentially unchanged" -> "the gap changes by at most 0.011 per unit". The clause that follows already gives the six numbers; name the delta.
49. `03-environment.tex:21` "a strictly stronger robustness probe than a within-tier split" -> "strictly stronger" is an unproven comparative. Either "a different and generally harder probe" or supply the argument.
50. `05-experiments.tex:255` "The Calm-to-Hard step (0.56 versus 0.53) sits well inside it" -> "sits inside it". The half-width already does the emphasis work.
51. `03-environment.tex:21` "a genuinely mean-reverting spread" and `03-environment.tex:25` "two market models with genuine microstructure" -> Rubber-stamp qualifiers (humanink #32). Delete both; the descriptions that follow establish it.
52. `05-experiments.tex:11` "That was not strictness." -> Dramatic fragment (no-ai-slop "dramatic fragmentation"). Merge: "The all-zero board reflected a unit error rather than strictness: the kernel applied its annualized 0.5 deflation prior per period without conversion..."

### 4.6 Category F: rule-of-three overuse

Most triads in this paper are factual enumerations of things that number three, and are false positives (Section 6). The genuine instances:

53. `01-introduction.tex:5` three sentences on one template with "and additionally" three times -> See Finding I-2. Delete the announcing clause, keep the three sentences.
54. `02-principles.tex:17` "A synthetic market that pays manipulation without bound, or that lets makers profit from informed flow, or that emits Gaussian tape, is an answer key for the wrong exam." -> Triple "or that" plus an aphorism kicker. Replace: "A synthetic market that pays manipulation without bound, that lets makers profit from informed flow, or that emits Gaussian tape will reward the wrong behavior." ("Answer key for the wrong exam" is humanink #26, dead metaphor. The same metaphor recurs at `03-environment.tex:9` as "the answer key", where it names the raw dataset concretely and is doing real work; keep it there, drop it here.)
55. `07-limitations.tex:17` "a novel leak path through a side channel the environment does not own, wall-clock time, filesystem state, a network call, is out of scope for both layers" -> Three appositives interrupt the subject and verb across 17 words. Recast: "a novel leak path through a side channel the environment does not own (wall-clock time, filesystem state, a network call) is out of scope for both layers."
56. `05-experiments.tex:35` "On Calm, drift is free deflated Sharpe: \texttt{kelly\_vol\_target} and \texttt{equal\_weight\_long} saturate the DSR at 1.0000 (bootstrap CIs ...) and \texttt{min\_variance} reaches 0.9849, though with a near-vacuous interval of ..." -> Not a triad problem but a 92-word-sentence problem. Split after the parenthetical intervals.

### 4.7 Category G: vague attribution

The paper is unusually good here: nearly every claim is anchored to a `\citep` or a committed artifact. Two entries:

57. **False positive, no change** `06-related.tex:7` "to our knowledge it does not target cross-runtime byte-determinism or tamper-evident trajectories" and "none of these systems, to our knowledge, sets out to..." -> Correct scholarly hedges, and the same paragraph explicitly discloses the basis ("Where these descriptions rest on our reading of the systems' papers and repositories..."). Logged because both `humanizer` #5 and `no-ai-slop` "weasel attribution" flag them.
58. `03-environment.tex:25` "two market models with genuine microstructure" -> An abstraction placed immediately before a list that specifies it exactly. Delete the abstraction, keep the list.

### 4.8 Category H: inflated abstraction and aphorism

59. `01-introduction.tex:3` "An eval is useless without an environment." -> See Finding I-1. Replace with the paper's own gap statement.
60. `02-principles.tex:11` "The contract is the product." -> `X is the Y of Z` aphorism form (humanizer #32) used as a principle heading, and the only one of the seven headings that is a slogan rather than a property. Replace with "The contract is governed, not just published." or "The interface is stable under governance."
61. `05-experiments.tex:37` "The corrected board is the honest zero" -> "honest zero" is a coined term used at `05-experiments.tex:37` and `07-limitations.tex:7` and implied in the abstract's framing, without ever being defined. Either define it once at first use or drop it and state the number.
62. `02-principles.tex:17` "is an answer key for the wrong exam" -> See item 54.
63. `05-experiments.tex:255` "half of everything that goes wrong is process, not PnL, which is exactly the half a return-ranked board never sees" -> Kicker construction (no-ai-slop "fake-profound kickers") plus a Category A disclaimer plus "exactly". The claim is real and belongs; the shape is a mic-drop. Replace: "On this mandate-blind reference field, roughly half of the failures are process failures that a return-ranked board does not record."

### 4.9 Category I: hedge stacking

The paper's hedging is mostly calibrated and correct (Section 6). Genuine stacking:

64. `predictability-fragment.tex:9` "so these summaries support ``near the unconditional baseline,'' not statistical indistinguishability from 50 percent and not proof that no edge exists" -> Three negations in one clause, after a sentence that already says "No hypothesis test or equivalence margin was preregistered". Reduce to: "so these summaries support only the descriptive statement that the adversary lands near the unconditional baseline."
65. `endogenous-adverse-fragment.tex:62` "Thus the data bracket an observed sign change between sampled values; they do not identify a crossover at $\lambda\approx0.27$ or attach uncertainty to an interpolated value." -> The preceding sentence already says "an exploratory six-value calibration, not a multiplicity-adjusted crossing estimate", and the figure caption says "no interpolated crossover estimate is claimed". Keep this one, delete the earlier clause and the caption clause.
66. `positive-control-fragment.tex:20` "It does not establish a universal region, an optimum, or a theorem-level correspondence: the search fixes push size, $\lambda$, $V$, total leg length and most follower gains, and omits short-side, overshooting and longer-hold trips." -> A triple negation plus a five-item fix list plus a three-item omission list, 45 words. Split into two sentences and drop "theorem-level correspondence", which no reader would have assumed.
67. **False positive, keep verbatim** `05-experiments.tex:52` "Both intervals include zero, so sixteen episodes do not resolve either point from the analytical reference; this failure to reject zero is not evidence of equivalence." -> The single most valuable sentence in the results, and the hedge is the finding. Flagged by `humanink` #23.

### 4.10 Category J: uniform paragraph openings

68. Eight result paragraphs in `05-experiments.tex` open `The [abstract noun] is/are ...`: lines 37, 52, 54, 92, 94, 173, 219, 287, 289. -> Vary at least four. Suggested rewrites are in items 12, 13, 15 and 39.
69. **False positive with a caveat** Six subsections open with a `\texttt{function\_name}` as grammatical subject: `05-experiments.tex:50` (`mm\_regret`), `:65` (`generalization\_gap`), `:141` (`impact\_pnl` context), `:197` (`compare\_informed\_vs\_uninformed`), `:234` (`classify\_episode\_failure`), `:266` (`run\_ecology`). The convention is deliberate, consistent and useful: it tells the reader which committed script produced the numbers. Keep it. But six identical openings in one section is part of why the section reads as a catalogue; the regrouping in E-2 breaks the run naturally with group openers.

### 4.11 Category K: decorative parallelism

70. `01-introduction.tex:5` See items 53 and Finding I-2.
71. **False positive, keep** `02-principles.tex:9` "Runs record the agent's decisions and nothing derived; replay against the frozen scenario reproduces returns byte-for-byte, and a tampered trajectory reproduces something else." -> Three-clause balanced period, effective and load-bearing.
72. **False positive, keep** `03-environment.tex:41,43,45,47` The four probe paragraphs each open with a bolded one-word label then a definition sentence. Uniform by design and correct in a taxonomy.
73. `05-experiments.tex:5` "F1 and F3 use 2,000 seed bootstrap resamples, F2, F5 and F6 use t-based 95\% intervals over committed vectors, and F4, F7 and F8 are count-based, reporting pass counts, episode counts and seed-level winner distributions rather than interval estimates." -> The parallelism is correct but the sentence is a table wearing prose. Convert to a three-row table; readers will want to look this up rather than re-read it.

### 4.12 Category L: build artifacts and formatting slop in the typeset source

74. `calm-tails-fragment.tex:1-2` `% F4 follow-up: Calm-tier calibration under the finite-panel-corrected realism gate.` and `% Evidence: paper/evidence/f4-realism.json, key "calm_calibration".` -> Delete or move to a build manifest. Diff-anchored writing (humanizer #30).
75. `endogenous-adverse-fragment.tex:1-5` five-line comment block including `% Producer: python paper/src/make-f6-adverse-selection.py` -> Same. The producer is already in `A-commands.tex:74`.
76. `manipulation-sweeps-fragment.tex:1` `% Extended F5 sweep. Evidence: paper/evidence/f5-manipulation.json, extended_sweeps.` -> Same.
77. `predictability-fragment.tex:1-3` three-line comment block -> Same.
78. **Correct, use as template** `arena-witness-fragment.tex:1` carries no comment block.
79. `calm-tails-fragment.tex:4` `\paragraph{...}\label{sec:calm-tails}` -> A `sec:`-prefixed label on a `\paragraph` while every sibling follow-up is a `\subsection`. Promote to `\subsection` for structural consistency, or demote the other seven; either is defensible, mixing is not.
80. `main.tex:3-6` The comment "Every empirical number in this paper is produced by a command listed in Appendix A and recomputable from the committed evidence." is a substantive claim living only in a LaTeX comment. It is also made in `08-reproducibility.tex:5`, so no information is lost, but a claim that important should not exist in two places with one of them invisible.

---

## 5. Voice and stance

**Current state.** The manuscript is about 95 percent impersonal ("The engine owns the time cursor", "A scenario is a pure function of one 64-bit seed"), with 15 first-person-plural instances clustered almost entirely in the fragments:

| File:line | Instance |
|---|---|
| `03-environment.tex:29` | "We therefore call it the closed-form reference policy." |
| `06-related.tex:7` | "our reading of the systems' papers", "we have kept them descriptive", "to our knowledge" (2 uses) |
| `07-limitations.tex:21` | "We judge publishing it net-positive" |
| `arena-witness-fragment.tex:3` | "We answer by construction with an out-of-band oracle" |
| `arena-witness-fragment.tex:28` | "with five paths per cell we do not test them" |
| `calm-tails-fragment.tex:5` | "We swept the existing opt-in families" |
| `concave-fragment.tex:10` | "We rerun the symmetric push-and-unwind sweep" |
| `manipulation-sweeps-fragment.tex:6` | "We sweep each axis one at a time" |
| `positive-control-fragment.tex:20` | "our finite schedules" |
| `predictability-fragment.tex:7,11,20` | "we report the kernel's", "We implemented the strongest cheap version", "We implemented and tested the narrow algebraic step" |

The distribution is diagnostic rather than random: the body is impersonal, the fragments say "we". This is the same fault line as the appended-block finding in E-3, and it is visible to a reader as a change of register mid-section.

**Recommended convention: authorial "we" throughout, restricted to three uses.** For a single-author NeurIPS submission this is the field norm and it is more honest than the passive alternatives, because it names who made each judgment call. Restrict it to:

1. **Decisions the authors made that a reader could reasonably have made differently.** "We therefore call it the closed-form reference policy" (`03-environment.tex:29`) and "We judge publishing it net-positive" (`07-limitations.tex:21`) are exactly right and should be the model.
2. **Actions taken in this work.** "We swept", "We rerun", "We implemented". Correct in the fragments; extend to the body, where the same actions currently hide in the passive ("price panels from eight seeded episodes are graded by...", `05-experiments.tex:105`; "Each policy's pooled return series is scored by...", `:9`).
3. **Explicit epistemic limits.** "with five paths per cell we do not test them", "to our knowledge". Correct as written.

Everything else stays impersonal, because the environment's properties are properties of the artifact and not of the authors. "The engine owns the time cursor" must never become "we make the engine own the time cursor".

**What to change, concretely:** convert the most agentive passives in `05-experiments.tex` to first-person plural so the fragments stop standing out. Candidates: `:9` "is rolled over sixteen scenario seeds" -> "We roll ... over sixteen scenario seeds"; `:105` "price panels from eight seeded episodes are graded" -> "We grade price panels from eight seeded episodes"; `:141` "is scored against its zero-impact paired reference" -> "We score ... against"; `:197`, `:234` and `:266` similarly. Do not touch the tool-as-subject convention noted in item 69; `\texttt{run\_ecology} seeds the replicator` is fine and useful.

**One stance inconsistency to resolve.** `08-reproducibility.tex:7` says "Language models were used to draft and edit prose". Passive and agentless, in a disclosure statement where the agent is the point. Recommend: "We used language models to draft and edit prose, and checked every bibliography entry against its source."

---

## 6. Detector findings marked as false positives

Recorded explicitly per instruction. Each was flagged by at least one detector and is correct usage in a quantitative paper.

1. **Repeated technical terms** (`humanizer` #11 synonym cycling, inverted). "deflated Sharpe", "seed band", "markout", "familywise interval" repeat dozens of times without variation. Correct: terminology stability is `research-paper-writing` execution rule 4. Do not vary.
2. **Measured hedging** (`humanink` #23, `no-ai-slop` "empty qualifiers"). "not statistically distinguishable", "this failure to reject zero is not evidence of equivalence", "exploratory", "pointwise". These are the paper's epistemic content. Detectors flag them as excessive; they are the opposite. Only the stacked cases in Category I are genuine.
3. **Parallel structure in results reporting** (`humanink` #29, `humanizer` #10). The four probe paragraphs (`03-environment.tex:41-47`), the tier-by-tier reporting in F1, F3 and F4, and the per-finding "entry point, seeds, quantity measured" structure are parallel because the underlying objects are parallel. Correct.
4. **Passive voice in methods** (`humanizer` #13). "price panels are graded by `certify_realism`" is standard methods register. The Section 5 recommendation converts some of these for voice consistency, not because passive is wrong.
5. **Rule of three where three things exist** (`humanizer` #10). "train, gap and held-out bands", "Calm, Hard and Extreme", "multiply, add, divide and max" (a quartet), "native, WebAssembly and Python", "deflation, reliability and process gates". All factual enumerations. Not slop.
6. **"to our knowledge" in related work** (`humanizer` #5, `no-ai-slop` weasel attribution). Required scholarly hedge, and its basis is disclosed in the same paragraph. See item 57.
7. **Long sentences with heavy subordination in method descriptions.** Flagged by `every-style-editor` readability. Some are genuinely too long (item 56), but the compound-modifier density in e.g. `endogenous-adverse-fragment.tex:9` carries real precision about the impact law. Split only where the subordination is sequential rather than definitional.
8. **Colons introducing genuine lists.** Roughly 38 of the 76 colon instances introduce enumerations (`03-environment.tex:9` "reading the raw dataset, slicing past the current index, peeking the next bar, or feeding the scenario seed"). Correct.
9. **`\texttt{}` function names as sentence subjects** (`no-ai-slop` "never let inanimate things do human verbs"). Deliberate provenance convention. See item 69.
10. **Negative-result framing throughout** (`humanink` #6 formulaic challenges sections). The limitations section and the many "this is a negative result" statements are substance, not formulaic hedging. Do not compress `07-limitations.tex` on slop grounds; compress it only by removing the duplications in Category C.
11. **`humanink` numeric AI score.** Not computed and not reported. A score would be a guess about provenance; the named patterns above are checkable evidence, which is what detect-only mode requires.
12. **Em dashes.** Zero prose em dashes found. Every `--` in the source is a LaTeX en-dash in a compound name (Avellaneda--Stoikov, Almgren--Chriss, commit--reveal, informed--uninformed, native--WebAssembly, diagnostic--confirmation). Fully clean; the `humanizer` #14 rule is already satisfied.
13. **Curly quotes.** The ``` `` ``` and `''` forms are correct LaTeX quoting, not Unicode curly quotes. Clean.
14. **Bold and emoji.** No `\textbf{}` used for mid-sentence emphasis; the only bold is the principle labels in `02-principles.tex` and the probe labels in `03-environment.tex`, both structural. No emoji. Clean.
15. **Title case in headings.** All section and subsection titles are sentence case. Clean.

---

## 7. Prioritized fix list

**P0, structural, do these first, they subsume many line edits**

1. **Replace the abstract** with the Section 1.2 draft. Cost: one file. Effect: fixes the most-read 374 words in the paper, and the C-G-I structure gives every reviewer a frame for the honest scope.
2. **Regroup the experiments section** into the three named groups of Finding E-2. This deletes the 280-word meta-preamble (items 23, 34, 35), breaks the uniform-opener run (items 68, 69), and makes the fragments' placement self-evident.
3. **Rewrite the introduction's first paragraph** per Finding I-1 so the paper establishes its own gap before citing the companion.

**P1, the two dominant tics, mechanical and high-yield**

4. **Apply the once-per-finding rule to scope disclaimers.** Category A, 104 instances down to about 45. Rule: each caveat appears once in its finding and once in limitations, never in the caption too.
5. **Convert the roughly 38 rhythmic colon-glosses to plain sentences.** Category B, items 12 to 21 plus the listed remainder.
6. **Delete the restatement paragraphs.** Category C, items 22 to 33, including cutting `\paragraph{What the evidence shows.}` to three sentences.

**P2, integration and voice**

7. **Delete the four `%` build-comment blocks** in the fragments (items 74 to 77) and promote `calm-tails-fragment` to `\subsection` (item 79).
8. **Settle the voice convention** per Section 5: authorial "we" for decisions, actions and epistemic limits; impersonal for artifact properties. Convert the agentive passives in `05-experiments.tex` and the disclosure line in `08-reproducibility.tex:7`.
9. **Rewrite the two appended fragment openings** (`manipulation-sweeps-fragment.tex:5`, `calm-tails-fragment.tex:12`) so each states its question before its omissions.

**P3, line-level**

10. Emphasis words and rubber-stamp qualifiers: items 43 to 52.
11. Aphorisms and kickers: items 59 to 63.
12. Hedge stacking: items 64 to 66.
13. Convert the dispersion-conventions sentence (`05-experiments.tex:5`) and the related-work comparison (`06-related.tex:7`) to tables.
14. Convert the contributions paragraph to a real `enumerate` with a 35-word cap per item.

**Do not do**

- Do not soften the 23-of-24 realism failure, the unresolved F2 minimum, the 1-of-8 ecology replication, or any "this is not evidence of equivalence" statement. These are the paper's credibility.
- Do not vary terminology for style.
- Do not compress `07-limitations.tex` beyond removing the specific duplications listed.
