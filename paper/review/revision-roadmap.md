# Revision Roadmap: SharpeArena (Round 1)

Companion to `editorial-decision.md` (decision: Major Revision). Items are in immutable source order: seat order R0 (journal fit), R1 (methodology), R2 (domain), R3 (perspective), DA (devil's advocate); within a seat, report order. A deduplicated item sits at its first source's position and lists every seat that raised it. **Order is traceability, not ranking or work order.**

Each item is tagged:
- **TEXT**: fixable by reframing, caveats, citations, or wording; no new computation.
- **NEW-EXPERIMENT**: requires running or extending a committed script; the exact command or extension is named.
- **TEXT or NEW-EXPERIMENT**: the reviewers offered both a rescoping branch and an experiment branch; either discharges the item.

Severities are transported from the source reports, never re-derived. Confidence values are the seats' self-reported scope disclosures.

---

## R0: Journal-Fit Reviewer (r0-journal-fit.md)

### R-01 | Learning experiment, or narrowed claims: **NEW-EXPERIMENT (or TEXT rescoping branch)**
- **Source**: R0 W1 (Major, conf 5); adjacent DA M1.
- **Location**: §5 (new subsection), §7, title, abstract.
- **Change**: no agent is ever trained on the environment; the Gymnasium/PettingZoo/verifiers surfaces are shipped but unexercised.
- **Fix (experiment branch)**: one standard RL baseline (e.g. PPO through the `sharpearena-py` Gymnasium adapter) trained on the train band per tier, evaluated on the held-out band and cross-regime, scored through the kernel; report the generalization gap and transfer matrix for a policy that can actually overfit. New script `paper/src/make-f9-learner.py` following the existing make-f*.py evidence conventions (fixed seeds, committed JSON, public API only).
- **Fix (rescoping branch)**: retitle and rewrite abstract/claims to environment-and-protocol only, stating compute as the constraint explicitly.

### R-02 | Trading-RL related-work positioning: **TEXT**
- **Source**: R0 W2 (Major, conf 4) + R2 W1 (Major, conf 5). Corroborated.
- **Location**: `sections/06-related.tex`; also §3.5 (mbt_gym at the market-making task), `refs.bib`.
- **Change**: §6 compares against exactly one simulator (ABIDES); FinRL, FinRL-Meta, ABIDES-Gym, TradeMaster, mbt_gym, gym-trading-env and the learned-LOB-simulator literature are absent.
- **Fix**: add a "Financial RL environments" paragraph or comparison table mirroring the ABIDES paragraph, positioning each predecessor on the five axes the paper competes on: leak-freedom mechanism, determinism guarantee, trajectory verifiability, external-agent contract, generalization protocol. References per R2 W1: FinRL (arXiv 2011.09607), FinRL-Meta (NeurIPS 2022 D&B, arXiv 2211.03107), ABIDES-Gym (ICAIF 2021, arXiv 2110.14771), mbt_gym (arXiv 2209.07823, venue UNVERIFIED), TradeMaster (metadata UNVERIFIED, verify before citing). Cite mbt_gym at §3.5/F2 specifically.

### R-03 | Companion contribution boundary and finding order: **TEXT**
- **Source**: R0 W3 (Major, conf 4); DA m2 (Minor, conf 4).
- **Location**: `sections/00-abstract.tex`, `sections/01-introduction.tex`, §5.1, §5 findings summary.
- **Change**: the paper leads with F1, whose substance is the companion SharpeBench paper's Finding 1; attribution between the two papers is ambiguous.
- **Fix**: one explicit sentence in §1 stating what this paper claims versus the companion (whose finding is the unit bug); lead the abstract and findings summary with environment-owned results (F2 regret, F3 transfer, F5 boundary); reposition F1 as validation that the producer-scorer separation catches scorer bugs.

### R-04 | Stale §2 claim-status paragraph: **TEXT**
- **Source**: R0 W4 (Major, conf 5). Verified editorially against `02-principles.tex` line 19 and §5.
- **Location**: `sections/02-principles.tex`, final paragraph ("their numbers arrive when the evidence run executes against the integrated kernel").
- **Change**: the paragraph implies the eight findings are pending while §5 reports executed numbers and the evidence is committed.
- **Fix**: rewrite in past tense: design properties are checkable by running the listed tests, and the eight findings were produced by the committed evidence run at sharpearena 0.8.0 / SharpeBench 0.5.0.

### R-05 | Abstract length, density, and title article: **TEXT**
- **Source**: R0 W5 (Minor, conf 4); R2 detailed comments and minor issues; R3 abstract comments.
- **Location**: title; `sections/00-abstract.tex`.
- **Change**: abstract exceeds track norms, packs companion-keyed apparatus (deflated Sharpe saturation, "mis-united floor", eligibility conjunctions); title's "The" over-claims; PIT expanded twice.
- **Fix**: halve the abstract (three sentences on the three structural properties, one on contract and protocol, two on environment-owned findings); change "The" to "A"; expand PIT once; move the last result sentences to the introduction (R2's suggestion); fix "mis-united" phrasing.

### R-06 | Significance register and adoption rhetoric: **TEXT**
- **Source**: R0 W6 (Minor, conf 4); DA M5 (Major, conf 4). Severity transported from both.
- **Location**: §1 ("The strategic bet is interface ownership"), §2 ("Whoever defines the interface owns the ecosystem"), §7.
- **Change**: significance case rests on adoption that has not started; "OpenAI-Gym moment" and ecosystem-ownership claims are commercial-strategy register presented as design principles, unfalsified by the paper's own limitations.
- **Fix**: frame significance as verifiability infrastructure checkable today, adoption as future work; move interface-ownership strategy to a short, neutral discussion paragraph; add any external-usage signal available by revision time.

### R-07 | R0 presentation minors: **TEXT**
- **Source**: R0 Minor Issues (Minor).
- **Location**: Table 1 caption; Figure 3 vs Table 3 (right panel); `refs.bib` (`minari` rendering under plainnat); §3 first functional mention of Gymnasium/PettingZoo; closing paragraph.
- **Change/Fix**: footnote the eligibility rule (pass^k = 1.00 required) in Table 1's caption so it is self-contained; drop or appendix one of the duplicated transfer-matrix presentations; verify the minari entry renders with authors; cite Gymnasium `check_env` and PettingZoo at first functional mention; consider a short closing paragraph restating checkable-today versus awaits-adoption claims.

---

## R1: Methodology (r1-methodology.md)

### R-08 | F8 multi-seed replication: **NEW-EXPERIMENT**
- **Source**: R1 W1 (Major, conf 5) + R3 W3 (Major, conf 4) + DA M7 (Major, conf 4). Strongest-corroborated item on the roadmap.
- **Location**: `paper/src/make-f8-ecology.py`, `paper/evidence/f8-ecology.json`, §5.8, abstract.
- **Change**: the abstract-level claim "a regime-shock schedule replaces the calm-regime ecology winner with the unlevered long book" is N=1 (`"seed": 0`, one trajectory per schedule).
- **Fix**: extend `paper/src/make-f8-ecology.py` to run 8 to 16 replicator seeds per schedule (matching the paper's own seed conventions elsewhere) and, per R3, varied shock orderings; report winner-replacement frequency, extinction counts, and the distribution of final dominant species; regenerate the evidence JSON and figures via `paper/src/make-figures.py`. If not run, scope the claim to the single committed run and remove it from the abstract (TEXT fallback). Also add one paragraph situating the replicator flow rule in the market-ecology literature (Farmer 2002; Lux and Marchesi 1999; SFI artificial stock market) and stating what it abstracts away (flow lags, leverage constraints, entry), per R3 W3.ii.

### R-09 | Uncertainty quantification for F2/F3/F5/F6 + F3 dispersion reporting: **NEW-EXPERIMENT**
- **Source**: R1 W2 (Major, conf 5); DA M4 (noise-floor arithmetic).
- **Location**: `paper/src/make-f2-regret.py`, `make-f3-generalization.py`, `make-f5-manipulation.py`, `make-f6-adverse-selection.py`; their evidence JSONs; §5.2, §5.3, §5.5, §5.6; Table 2 (right matrix).
- **Change**: bare point estimates everywhere outside F1; F2/F6 evidence serializes aggregate means only, so dispersion cannot be recomputed from the artifacts; the transfer matrix narrates the Hard-to-Extreme entry (0.096) that sits inside the paper's own 0.34 DSR noise calibration.
- **Fix**: extend the four scripts to serialize per-episode/per-seed vectors and report seed-paired bootstrap or t-based 95% CIs for: F2 regret means per half-spread, F3 gap and per-cell transfer entries, F5 impact PnL points, F6 per-horizon paired gaps (answering R1 Q4: does the h=1 gap of 0.072 exclude zero?). In Table 2's right matrix, typographically or textually separate cells above the 0.34 DSR noise calibration from those below it, and mark the by-construction diagonal.

### R-10 | Seed-band protocol wording and provenance: **TEXT (rerun branch optional)**
- **Source**: R1 W3 (Minor, conf 5).
- **Location**: §5 preamble; §5.1 ("a positive, deflated Sharpe that survives every held-out seed"); §3.4.
- **Change**: all evidence seeds are drawn from the train band [0, 16); the canonical held-out band and the reserved 1e6 namespace are never touched; F1's "held-out" wording misdescribes the seeds used.
- **Fix**: state in §5's preamble which band the evidence seeds come from and why that is immaterial for parameter-free baselines; reword §5.1 to "survives every evaluation seed". Optional NEW-EXPERIMENT branch: rerun headline experiments on the canonical held-out band or the reserved eval namespace (same scripts, seed arguments changed).

### R-11 | Bootstrap parameters committed: **NEW-EXPERIMENT (small)**
- **Source**: R1 W4 (Minor, conf 4).
- **Location**: `paper/src/make-f1-baselines.py`; `paper/evidence/f1-baselines.json` config block; §5.1; Appendix A.
- **Change**: the text claims "2,000 resamples" but the script passes only `confidence=True`; resample count and bootstrap seed live in library defaults outside the committed command surface.
- **Fix**: pass the resample count and bootstrap RNG seed explicitly in `make-f1-baselines.py`, serialize both into the evidence config, rerun, and record them in the Appendix A command.

### R-12 | F5 grid completeness: **TEXT**
- **Source**: R1 W5 (Minor, conf 5).
- **Location**: Table 3 (tab:f5, right panel); §5.5 text.
- **Change**: the committed sweep sampled push weight 0.75 (impact PnL -3.07e-4) but the table reports only 0.10/0.25/0.50/1.00 with no note; the per-axis grid (6 points per axis, 8 seeds) is not stated in the text.
- **Fix**: add the 0.75 row (or footnote the omission), state the grid explicitly, and note how coarsely "everywhere" should be read; per-point seed dispersion folds into R-09.

### R-13 | Vol-clustering re-certification hygiene: **TEXT or NEW-EXPERIMENT**
- **Source**: R1 W6 (Minor, conf 4).
- **Location**: §5.4 remediation paragraph; `paper/src/make-f4-realism.py`; `paper/evidence/f4-realism.json`.
- **Change**: the `vol_clustering = 0.5` fix is tuned and re-certified on the same seeds 0..7 used to diagnose the failure, at one undefended strength.
- **Fix (text)**: label the re-certification explicitly as in-sample on the diagnosis panel. **Fix (experiment)**: extend `make-f4-realism.py` to re-certify on a disjoint seed set at 2 or 3 strengths.

### R-14 | F6 maker decomposition scope: **TEXT (aggregation branch optional)**
- **Source**: R1 W7 (Minor, conf 4); DA m4 (Minor, conf 4); R3 minor (units).
- **Location**: §5.6 decomposition paragraph; tab:f6 and tab:f5 captions.
- **Change**: the subsidised-maker narrative comes from one detail episode (detail_seed 0) presented with the same authority as the 24-episode aggregate; markout units ("per filled unit") and the tab:f5 notional base are undefined.
- **Fix**: label the paragraph a single-episode illustration of the exact decomposition identity (or extend `make-f6-adverse-selection.py` to aggregate over the 24 episodes); define the price/quantity normalization in both table captions.

---

## R2: Domain (r2-domain.md)

*(R2 W1 is merged into R-02 above.)*

### R-15 | "Provable optimum" reframing: **TEXT**
- **Source**: R2 W2 (Major, conf 4) + R3 W6 (Minor, conf 5). Severity divergence recorded and arbitrated in the decision letter; item is must-fix.
- **Location**: abstract, §1 contributions item 4, §3.5 (sec:tasks), §5.2 (F2 interpretation); `refs.bib`.
- **Change**: the AS closed form is the source model's asymptotic approximation (exact treatment: Gueant, Lehalle and Fernandez-Tapia 2013); nothing proves it optimal for the implemented reward; the zero-regret self-check is by construction.
- **Fix**: rename to "closed-form reference policy" or "analytical baseline"; state approximation status and model-boundedness (arithmetic Brownian mid, exponential fills, CARA, no adverse selection in the fill process, noting the tension with F6 per R3 W6); cite Gueant-Lehalle-Fernandez-Tapia 2013; alternatively, prove or test exact optimality for the implemented reward and say so (answers R2 Q1). Also soften §5.2 "validates the metric's zero point" to "confirms the plumbing" (R1 minor).

### R-16 | Companion citation locator: **TEXT**
- **Source**: R2 W3 (Major, conf 5).
- **Location**: `refs.bib` entry `toca2026sharpebench`; optionally a new appendix.
- **Change**: the citation carrying the scoring semantics and the F1 unit-bug claim has no arXiv ID, DOI, or URL.
- **Fix**: add a persistent locator before submission; if the companion is not yet public, say so in the text and summarize the unit-bug derivation (annualized prior applied per period; conversion at 252) in an appendix so the claim is checkable standalone.

### R-17 | Cont (2001) citation scope: **TEXT**
- **Source**: R2 W4 (Minor, conf 4).
- **Location**: §3.7 stylized-facts list; §5.4 (F4); `refs.bib`.
- **Change**: Zumbach timescale asymmetry and the Fano burstiness factor are attributed to Cont (2001), which contains neither.
- **Fix**: keep Cont for the classical facts; add Zumbach (time reversal invariance in finance, circa 2009, verify volume/pages) for the asymmetry; cite a source for the Fano factor as a market stylized fact or present it explicitly as the authors' own diagnostic with its bound justified.

### R-18 | Probe-layer foundational citations: **TEXT**
- **Source**: R2 W5 (Minor, conf 4); R3 reading list and W1/W2/W3 citation requests.
- **Location**: §3.7, §5.5, §5.6, §5.8, §6; `refs.bib`.
- **Change**: the probes operationalize known mechanisms without their foundations: adverse selection (Glosten-Milgrom 1985, JFE 14(1):71-100), manipulation payoff (Allen-Gale 1992, RFS, verify pages), strategy ecology (Farmer 2002; Lux-Marchesi 1999; Lo's Adaptive Markets, verify exact citations).
- **Fix**: add the citations at the probes' introductions so they read as principled operationalizations rather than reinventions. (Huberman-Stanzl and Gatheral go with R-22; Budish et al. with R-25.)

### R-19 | "Dominant failure mode" superlative: **TEXT**
- **Source**: R2 W6 (Minor, conf 4).
- **Location**: `sections/01-introduction.tex` ("look-ahead bugs are the dominant failure mode of backtests").
- **Change**: unsupported empirical superlative; Bailey and Lopez de Prado (already in the bibliography) argue selection bias is the central backtest pathology.
- **Fix**: soften to "a pervasive and usually invisible failure mode", or cite the backtest-overfitting literature and drop the ranking.

### R-20 | Abstract "impossible by construction" scoping: **TEXT**
- **Source**: R2 W7 (Minor, conf 5). Feeds DA C2 (see R-24).
- **Location**: `sections/00-abstract.tex`.
- **Change**: the abstract's universal "impossible by construction" exceeds what §7 carefully bounds (deny-list guard, side channels).
- **Fix**: scope to the environment boundary, e.g. "impossible through the environment's own interface, and policed beyond it", matching §3.1 and §7.

### R-21 | R2 bibliographic and rendering minors: **TEXT**
- **Source**: R2 Minor Issues (Minor).
- **Location**: `refs.bib` (`byrd2020abides` year vs eprint 1904.12066; `minari` versioning), §6 ("integer-seed-interval" vs Procgen's own "level sets over seed ranges" phrasing), §1/§5 (pass^k rendering consistency).
- **Fix**: cite the published SIGSIM-PADS 2020 ABIDES or match the arXiv year; pin the Minari version used by the adapter or cite a versioned DOI; align the Procgen phrasing; unify pass^k markup.

---

## R3: Perspective (r3-perspective.md)

### R-22 | F5 manipulation-probe theory and ablation: **TEXT + NEW-EXPERIMENT**
- **Source**: R3 W1 (Major, conf 5); DA M3 (Major, conf 4: no positive control); DA Ignored-Alternatives 2.
- **Location**: §5.5 (F5), §6, §7 impact-model paragraph; `paper/src/make-f5-manipulation.py`; `refs.bib`.
- **Change**: under linear permanent impact, unprofitable round-trip manipulation is close to a theorem (Huberman-Stanzl 2004, Econometrica; Gatheral 2010, Quantitative Finance), so the probe's clean verdict has near-zero diagnostic power and "certifies" is circular; the probe has never been shown able to detect the defect it hunts.
- **Fix (text)**: cite Huberman-Stanzl and Gatheral; state explicitly that linear permanent impact makes the negative result expected, so F5 is a consistency check, not a realism certification; rephrase "certifies" throughout F5 and in the abstract's manipulation sentence.
- **Fix (experiment)**: concave-impact F5 ablation: extend `paper/src/make-f5-manipulation.py` with a square-root (per Almgren et al. 2005) or transient-decay impact variant and rerun `impact_boundary_sweep` / `size_response`, reporting whether a profitable region appears. A specification engineered to permit profitable manipulation that the probe then flags doubles as DA M3's positive control, turning F5 from a tautology into an instrument demonstration.

### R-23 | F6 level-versus-gap and adverse-selection economics: **TEXT (recalibration branch optional)**
- **Source**: R3 W2 (Major, conf 5); DA M3 (second half).
- **Location**: §5.6 (F6), tab:f6 interpretation, §2 cross-reference; `refs.bib`.
- **Change**: makers earn positive markouts against informed flow at every horizon (0.689/0.489/0.386 per unit at h = 1/5/20), which violates the paper's own §2 standard ("lets makers profit from informed flow"); the design holds the price path exogenous, so this is markout accounting against drift, not Glosten-Milgrom adverse selection; the paper reports only the gap direction as success.
- **Fix (text)**: separate the two certifications (sign of the gap vs sign of the level); report the level finding against the environment with F4-grade honesty and answer R3 Q2 (intended weak alpha, strike-at-pre-move-mid artifact, or defect); distinguish the fixed-path markout decomposition from equilibrium adverse selection with a Glosten-Milgrom (1985) citation; note endogenous spread response as future work.
- **Fix (experiment, optional)**: recalibrate informed alpha or strike fills at post-impact prices in `make-f6-adverse-selection.py` and rerun; or add a probe variant where quotes widen with realized toxicity.

*(R3 W3 is merged into R-08.)*

### R-24 | Instrument-calibration framing at the headline layer: **TEXT**
- **Source**: R3 W4 (Minor, conf 4); resolves DA C1 jointly with R-27; R0 W6 adjacent.
- **Location**: `sections/00-abstract.tex` (one clause), `sections/01-introduction.tex` (one sentence).
- **Change**: headline numbers (Calm DSR saturation at 1.0, the 0.97 transfer gap, the markout range) are marketed as findings without the framing that they were produced on a generator failing its own realism conjunction 23/24, with Calm actually platykurtic.
- **Fix**: one abstract clause and one introduction sentence framing the eight findings as instrument calibration on a generator that intentionally fails full realism certification at this revision; the §7 content moves upstream, it does not change.

### R-25 | Batch-clearing scope and fees: **TEXT**
- **Source**: R3 W5 (Minor, conf 4).
- **Location**: §3.5 or §7 (new short paragraph); `refs.bib`.
- **Change**: the single-price-per-step clearing and call-auction uncross are frequent-batch-auction designs (Budish, Cramton and Shim 2015, QJE); latency competition, queue-position value, sniping, and maker-taker fees do not exist in the environment and no fee structure is mentioned; the scope restriction is silent.
- **Fix**: name the clearing mechanism as batch-auction-like with the citation; state the excluded competition margins; state whether fees/rebates/spread costs are modeled and whether the F1/F3 Sharpes and F2/F6 numbers are gross or net (answers R3 Q4). Framed positively, this is an FBA-laboratory selling point.

### R-26 | Broader-impact paragraph: **TEXT**
- **Source**: R3 W7 (Minor, conf 3).
- **Location**: §7 or a dedicated broader-impact statement.
- **Change**: no treatment of leaderboard-score misuse toward retail audiences (verifiability makes a synthetic score more persuasive, not more externally valid) or of the manipulation probe's dual-use surface as a payoff-search harness.
- **Fix**: short paragraph: an explicit norm (or license text) that SharpeArena scores are claims about a synthetic environment and are not performance marketing; one sentence on why publishing the manipulation probe is net-positive once R-22's reframing is in place.

*(R3 W6 is merged into R-15; R3's units minors into R-14; "subsidised" spelling into R-33.)*

---

## DA: Devil's Advocate (r4-devils-advocate.md)

### R-27 | C1: uncertified-tape framing (VALIDATED CRITICAL): **TEXT (robustness rerun optional)**
- **Source**: DA C1 (Critical, conf 5); corroborated R3 W4, R0 W6.
- **Location**: `sections/00-abstract.tex`, `01-introduction.tex`, `02-principles.tex` (§3.7 gate-purpose sentence), `05-experiments.tex` (F4), `07-limitations.tex`.
- **Change**: the realism gate fails the canonical configuration 23/24; the fix is opt-in and not the leaderboard default; every substantive finding is computed on the uncertified tape; nothing downstream is conditioned on certification; the gate's stated purpose ("fails the certification instead of silently letting agents win on unrealistic markets") is exactly the shipped situation. Either the gate gates, or it is a reported diagnostic; the paper currently implies both.
- **Fix (text, required)**: apply R-24's upstream framing; where the gate is introduced (§3.7), state its current role explicitly: at this revision certification is reported alongside results and is not a leaderboard precondition; state the intended endgame (a future generator revision passes, or leaderboard claims become conditioned on certification); answer the DA's unexamined premise by arguing why verifiability-first, realism-later is the right sequencing.
- **Fix (experiment, optional)**: rerun F1/F3/F7/F8 at `vol_clustering = 0.5` (same make-f*.py scripts, spec flag changed) and report whether the findings are robust to the partially remediated tape; this would materially strengthen the reframed claims.

### R-28 | C2: leak-freedom scope and generator inversion (VALIDATED CRITICAL): **TEXT (predictability probe optional)**
- **Source**: DA C2 (Critical, conf 4); corroborated R2 W7 (see R-20); DA Ignored-Alternatives 1.
- **Location**: `sections/00-abstract.tex`, `02-principles.tex` ("An agent cannot peek because there is nothing to call"), `03-environment.tex` (§3.1, §3.2, §3.4), `07-limitations.tex` (new paragraph).
- **Change**: scenarios are pure functions of a public seed under a four-operation transform, and on the canonical task the path ignores the agent's actions; an agent that learns or inverts the transform from the trailing window predicts future bars in-band; seed bands block seed lookup, not inversion; no analysis, bound, or experiment addresses this, yet the abstract says "impossible by construction".
- **Fix (text, required)**: scope every leak-freedom claim to the interface boundary (with R-20's abstract wording); add a limitations paragraph naming in-band predictability of a public deterministic generator as an open question distinct from the side channels already listed, and stating precisely what the seed bands defend against.
- **Fix (experiment, optional but recommended)**: a simple predictability probe: fit a next-bar predictor (even ridge regression or a small MLP) on trailing windows from train-band seeds and report its held-out-band accuracy against a persistence baseline; either result is informative and would convert the disclosure into a measurement.

### R-29 | M1: rank-eligibility attainability: **NEW-EXPERIMENT**
- **Source**: DA M1 (Major, conf 4); adjacent to R-01.
- **Location**: §5.1 (F1), tab:f1 discussion.
- **Change**: best pass-per-seed is 0.75; even `flat` fails the per-run gate on all 16 seeds, suggesting an unstated activity/profitability requirement; no entry is ever shown to be rank-eligible, so "the number a real agent has to beat" may be unbeatable.
- **Fix**: run the AS closed-form policy (or any policy expected to pass) through the kernel's full eligibility conjunction and report whether a rank-eligible entry exists (extend `make-f1-baselines.py` or a small companion script); document what the per-run gate requires; if the eligibility set is empty for all tested policies, say so and discuss discriminative power.

### R-30 | M2: golden-hash coverage versus trust-chain rhetoric: **TEXT**
- **Source**: DA M2 (Major, conf 5).
- **Location**: `sections/00-abstract.tex`, `04-contract.tex` ("remove the host from the trust equation"), §5 preamble.
- **Change**: F4 through F8 are produced by the Python probe layer outside the byte-identity guarantee (the paper admits this in §7); the abstract and §4 advertise host-independent byte-verifiable numbers without that scoping, so the headline guarantee covers the minority of the headline evidence.
- **Fix**: state the coverage split where the guarantee is advertised: byte-identity covers the Rust core, bindings, and F1-F3's engine outputs; F4-F8 are seeded-deterministic on one platform. One sentence in the abstract or §4 plus one in §5's preamble.

*(DA M3 is merged into R-22 and R-23; DA M5 into R-06; DA M7 into R-08.)*

### R-31 | M4: F3 framing for untrained policies: **TEXT**
- **Source**: DA M4 (Major, conf 4); dispersion half merged into R-09.
- **Location**: §5.3 (F3), tab:f3 and its caption.
- **Change**: "training regime Calm" language implies learning where none occurs; for a fixed policy the 3x3 matrix reduces to pairwise differences of three per-tier DSRs already visible in Table 1; at least one narrated entry (0.096) sits below the paper's own noise floor.
- **Fix**: replace "trained/training regime" with "reference/source regime" wording for the fixed-policy matrix; state that for parameter-free policies the matrix is a calibration of the instrument, not a transfer result; apply R-09's sub-noise cell marking.

### R-32 | M6: F7 mandate-assignment mechanism: **TEXT**
- **Source**: DA M6 (Major, conf 4).
- **Location**: §5.7 (F7), the "half of everything that goes wrong is process, not PnL" headline.
- **Change**: mandates are sampled per scenario and assigned to fixed policies that cannot observe or adapt to them, so structural breaches are guaranteed by the draw; the flat-across-tiers breach counts (36/36/31) are the signature of that mechanism, and a mandate-aware field could invert the distribution.
- **Fix**: state the mechanism as the driver of the counts and scope the headline to a property of the mandate-blind reference field, not of "agents"; note that mandate-aware agents are the interesting future measurement.

### R-33 | DA minors and residual polish: **TEXT**
- **Source**: DA m1 (Minor, conf 4), m3 (Minor, conf 5); R3 spelling minor.
- **Location**: §2/§5.4 (Calm narration), §3.4 ("carves a provably disjoint test family"), §5.6 ("subsidised").
- **Change/Fix**: resolve the Calm inconsistency by choosing one reading (stylized facts as universal bounds, in which case Calm is stated plainly as unrealistic tape, or regime-conditional bounds, in which case the certification defines them per regime); tone down "provably disjoint" (two separated integer intervals) so it does not sit at theorem grade beside the genuinely nontrivial guarantees; harmonize American spelling.

---

## Counts

- Items: 33 (R-01 through R-33), deduplicated across 5 seats.
- By driving severity (transported): 2 critical (R-27, R-28); 14 major (R-01, R-02, R-03, R-04, R-08, R-09, R-15, R-16, R-22, R-23, R-29, R-30, R-31, R-32); 1 mixed minor/major from its two sources (R-06); 16 minor (R-05, R-07, R-10, R-11, R-12, R-13, R-14, R-17, R-18, R-19, R-20, R-21, R-24, R-25, R-26, R-33).
- By fix type: TEXT-only 21; NEW-EXPERIMENT required 4 (R-08, R-09, R-11, R-29); TEXT plus a required experiment leg 1 (R-22 concave-impact ablation); either-branch 1 (R-01: learning run or rescoping); TEXT with optional experiment branch 6 (R-10, R-13, R-14, R-23, R-27, R-28).
- DA CRITICAL adjudications: C1 VALIDATED (R-27), C2 VALIDATED (R-28). Both block Accept; decision is Major Revision.
