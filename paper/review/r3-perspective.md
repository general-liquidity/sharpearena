# Peer Review Report

## Manuscript Information
- **Title**: SharpeArena: The Point-in-Time (PIT) RL Environment for Trading Agents
- **Manuscript ID**: (none; preprint, NeurIPS 2026 Evaluations and Datasets track target)
- **Review Date**: 2026-08-24
- **Review Round**: Round 1

---

## Reviewer Information

### Reviewer Role
Peer Reviewer 3 (Perspective)

### Reviewer Identity
Market-microstructure economist. Home literature: Kyle (1985) and its manipulation corollaries, Glosten-Milgrom adverse selection, Almgren-Chriss optimal execution, Avellaneda-Stoikov market making, Budish-Cramton-Shim market design, and the agent-based market ecology tradition (Farmer, Lux-Marchesi, the SFI artificial stock market). I am reviewing an RL-environment systems paper from the economics side; I do not judge the software engineering, the determinism machinery, or the RL evaluation methodology per se, and I acknowledge that some of my concerns may reflect conventions that differ between financial economics and ML benchmarks.

### Review Focus
Economic realism of the impact and clearing models; economic validity of the manipulation and adverse-selection probes; whether the ecology/replicator layer maps to real market dynamics; whether conclusions drawn on the synthetic generator transfer to real markets; cross-disciplinary framing opportunities; and stakeholder and deployment implications, including what an agent trained on this environment actually learns.

---

## Overall Assessment

### Recommendation
- [ ] Accept
- [ ] Minor Revision
- [x] **Major Revision**
- [ ] Reject

### Confidence Score
4

Mostly within my area of expertise (market microstructure, impact modeling, market ecology); the RL-benchmark and systems components are adjacent for me.

Confidence is an uncertainty/scope disclosure only; it never changes consensus counts, severity, decision bearing, or arbitration.

### Summary Assessment

The paper describes a point-in-time RL environment for trading agents with three engineering guarantees (leak-freedom by construction, cross-runtime byte determinism, replay-based tamper evidence), a governed JSON wire contract, procedural scenarios with disjoint seed bands, and a probe layer that turns diagnostics back on the simulator itself. From an economics standpoint the design stance is unusually good: the "distrust the simulator" principle, the paired zero-impact control in the manipulation probe, regret against a closed-form optimum, and the candid F4 realism failure are all things I wish more market simulators did. The core problems are economic, not procedural. The manipulation probe's clean verdict is close to a foregone conclusion under linear permanent impact, a known theoretical result the paper does not engage; the adverse-selection probe holds the price path exogenous and its makers remain profitable against informed flow at every horizon, which inverts the economics it claims to certify; and the ecology findings rest on one matched trajectory pair under a replicator mapping whose correspondence to capital reallocation in real markets is asserted only implicitly. Each is repairable by reframing, added theory citations, and modest additional sweeps, but as written the probe layer certifies less than the paper implies, and agents trained here would learn some economically wrong lessons. I recommend major revision.

---

## Strengths

### S1: The "distrust the simulator" stance is a genuine methodological contribution
Most market simulators used for RL are validated once, informally, and then trusted. Elevating simulator self-probing to a design principle, and shipping the probes as first-class artifacts with committed evidence, is a practice the agent-based finance community has argued for since the 1990s and rarely gets. The willingness to report the F4 realism certification as a failure of the paper's own generator (1 of 24 runs passing) rather than burying it is scientifically honest and rare in benchmark papers.
**Evidence Anchor**: `text: sections/02-principles.tex "A synthetic market that pays manipulation without bound, or that lets makers profit from informed flow, or that emits Gaussian tape, is an answer key for the wrong exam."`

### S2: Regret against a closed-form optimum is the right economic scoring idea
Scoring the market-making task on regret against the Avellaneda-Stoikov closed form, rather than on raw PnL against other agents, removes the flow-luck axis and imports a decision-theoretic notion of skill into a leaderboard context. The U-shaped fixed-spread regret curve (84.6 at 0.05 half-spread, 0.037 at 0.5, 58.8 at 4.0) behaves exactly as the theory predicts, which validates the metric's zero point.
**Evidence Anchor**: `text: sections/05-experiments.tex "regret is 84.6 at a half-spread of 0.05 (quoting too tight, filled constantly, run over by inventory), falls to a minimum of 0.037 at 0.5"`

### S3: The paired zero-impact control is clean attribution design
Scoring the manipulator against a reference run that shares the seed and follower population and differs only in zeroed impact coefficients isolates exactly the profit attributable to moving the price. This is good experimental economics regardless of my concerns about the impact specification itself, and the same paired-control discipline recurs in the adverse-selection probe (alpha-sided versus coin-sided legs on a fixed schedule).
**Evidence Anchor**: `text: sections/03-environment.tex "the reference sets both impact coefficients to zero, so the difference in the manipulator's profit is exactly what moving the price was worth"`

### S4: The limitations section names the right limitations
The paper concedes the single synthetic generator family, the stylized impact models, the trivial baselines, and the untested governance promise, and explicitly warns that "a stylized model can fail realistically for the wrong reason." Several of my weaknesses below are extensions of caveats the authors already opened; the disagreement is about how far upstream those caveats need to travel.
**Evidence Anchor**: `text: sections/07-limitations.tex "F5's clean verdict, manipulation unprofitable everywhere, is a statement about this specification at the tested coefficients"`

---

## Weaknesses

### W1: The manipulation probe's clean verdict is close to a theorem of the chosen impact model, not an empirical finding about the simulator
**Problem**: F5 sweeps Kyle lambda, temporary impact eta, follower gain, and push size, finds pumping unprofitable everywhere, and presents this as the probe "certifying" the environment. But it is a classical result that linear, permanent, time-independent price impact rules out profitable round-trip manipulation essentially by construction: Huberman and Stanzl (2004, Econometrica, "Price manipulation and quasi-arbitrage") show linear permanent impact is what excludes it, and Gatheral (2010, Quantitative Finance, "No-dynamic-arbitrage and market impact") extends the argument to transient impact with decay. In a linear Kyle plus Almgren-Chriss book, a push-and-unwind pays its own impact twice and can only recover through the follower response; unless followers are calibrated to front-load enough momentum flow, the loss is guaranteed and monotone in the impact coefficients, which is exactly the table the paper reports. The probe is therefore incapable of failing on the interesting axis: the manipulation channels that matter in real markets (concave/square-root impact where pushing gets cheaper at the margin, transient impact with slow decay, stop-loss and liquidation-cascade triggers, spoofing and layering in the visible book, closing-auction pressure, cross-venue and settlement manipulation) are all outside the specification, and several are outside what the environment can represent at all.
**Evidence Anchor**: `text: sections/05-experiments.tex "instead the probe certifies that the Kyle-plus-Almgren-Chriss book prices this manipulation as a cost at every tested point"`
**Why it matters**: The abstract and F5 present "manipulation never pays" as evidence the simulator is well specified. Under the linearity theorem, the same result would obtain for nearly any parameterization, so the probe's diagnostic power over this environment is close to nil, and a reader from finance will see the claim as circular. Worse, an agent population trained and red-teamed only in this environment learns that manipulation is structurally unprofitable, which is false in real books precisely where impact is concave or trigger-laden.
**Suggestion**: (i) Cite Huberman-Stanzl and Gatheral and state explicitly that linear permanent impact makes the negative result expected, so F5 is a consistency check, not a certification of realism. (ii) Add one ablation that gives the probe a chance to fail: a concave (e.g. square-root, per Almgren et al. 2005, "Direct estimation of equity market impact") or transient-decay impact variant, and show whether the boundary sweep then finds a profitable region; that would turn F5 from a tautology into a genuine instrument demonstration. (iii) Rephrase "certifies" throughout F5.
**Severity**: Major
**Confidence**: 5 — core expertise: market impact and manipulation theory

### W2: The adverse-selection probe fixes the price path exogenously and its makers profit against informed flow, inverting the economics it certifies
**Problem**: In the F6 design the informed and uninformed legs hold the price path fixed and differ only in trade-sign correlation with subsequent drift, and fills are struck at the pre-move mid. Two economic consequences follow. First, informed trading does not move prices here, so this is not adverse selection in the Glosten-Milgrom or Kyle sense (where the information content of order flow is impounded into quotes and the maker's defense is the spread); it is a markout accounting exercise against exogenous drift, and makers have no endogenous quote response to defend with or be measured against. Second, and more troubling for what the environment teaches: Table F6 shows maker markouts that are positive at every horizon even against informed flow (0.689, 0.489, 0.386 per filled unit at h = 1, 5, 20). The informed-versus-uninformed gap has the right sign and shape, which the paper correctly highlights, but the level says makers in this environment earn money trading against informed meta-orders. In real markets, markouts against informed flow are negative; that is the whole reason adverse selection is a cost that disciplines spreads. An environment where trading against information is merely "less profitable" prices liquidity provision as a free lunch.
**Evidence Anchor**: `table: Table (tab:f6) — Informed markout 0.386 at horizon 20, positive at all horizons`
**Why it matters**: The paper's own principle says a simulator "that lets makers profit from informed flow" is "an answer key for the wrong exam" (Section 2). By that stated standard, F6's levels are a failed certification, yet F6 is reported as the machinery being "non-vacuous in the required direction." Sign of the gap and sign of the level are different certifications; the paper conflates them. An RL market maker trained here will quote tighter than any real venue would tolerate, because informed flow never actually costs it money.
**Suggestion**: (i) Report the level finding against the environment with the same honesty F4 receives: state that positive informed-flow markouts indicate the informed alpha or the strike-at-pre-move-mid convention underprices information, and either recalibrate the alpha strength or strike fills at post-impact prices. (ii) Add a sentence distinguishing this fixed-path markout decomposition from equilibrium adverse selection (Glosten and Milgrom 1985, Journal of Financial Economics), and note that endogenous spread response is future work. (iii) Consider a Glosten-Milgrom-style probe variant where quotes can widen with realized toxicity, which would also connect to the F6 maker decomposition already in the paper.
**Severity**: Major
**Confidence**: 5 — core expertise: adverse selection and markout analysis

### W3: The ecology findings rest on a single matched trajectory pair, and the replicator mapping to real capital dynamics is never argued
**Problem**: F8 runs one control schedule and one shock schedule (12 generations, 8 founders, innovation every 4 generations) and concludes that regime rotation replaces the calm-regime winner with the unlevered long book. Two issues. First, sample size: with one seed schedule pair, no error bars, and extinction thresholds and innovation cadence chosen without sensitivity analysis, the specific succession result (kelly_vol_target extinct by generation 5, equal_weight_long dominant at 0.67) is an anecdote, however plausible. Second, mapping: replicator dynamics reallocate population share in proportion to realized fitness each generation, which is a stylized stand-in for how capital actually moves across strategies (fund flows chase trailing returns with lags and convexities, leverage constraints and margin spirals force exits discontinuously, and new entry is not parameter perturbation of the leader). The economics literature that does this seriously (Farmer 2002, "Market force, ecology and evolution", Industrial and Corporate Change; Lux and Marchesi 1999, Nature; the SFI artificial stock market of Arthur et al. 1997) treats the flow-of-capital rule as a modeling choice requiring justification, because the ecology's qualitative conclusions are known to be sensitive to it. The paper adopts the replicator silently.
**Evidence Anchor**: `text: sections/05-experiments.tex "runs matched trajectories under a steady all-Calm control schedule and a \texttt{regime\_shocks} schedule"`
**Why it matters**: The direction of the F8 finding, leverage-intensive volatility exploitation dies at a regime break while unlevered exposure survives, reproduces a well-known fragility (short-vol and vol-targeting drawdowns at volatility transitions), so it reads as validation. But because it is one run under one flow rule, the paper cannot distinguish "the ecology probe measures survival dynamics" from "the ecology probe produced one plausible story." Practitioners and market-design readers will discount the layer accordingly.
**Suggestion**: (i) Replicate F8 over multiple ecology seeds and shock orderings and report outcome distributions (which species dominates, extinction counts) rather than one trajectory pair. (ii) Add one paragraph situating the replicator choice in the market-ecology literature and stating what it abstracts away (flow lags, leverage constraints, entry). (iii) Soften "That is the finding the ecology probe exists to produce" to a claim about one illustrative run unless the replication is added.
**Severity**: Major
**Confidence**: 4 — core expertise: market ecology models; the specific replicator implementation details are the authors'

### W4: Headline empirical numbers travel without the realism caveat that F4 attaches to them
**Problem**: The abstract and introduction lead with quantitative findings (drift saturates deflated Sharpe at 1.0 on Calm; a 0.97 calm-to-extreme transfer gap; adverse selection priced at 0.07 to 0.44 per filled unit) produced on a generator that the paper's own certification fails on 23 of 24 runs, with the Calm tier actually platykurtic (mean excess kurtosis -0.97), i.e. thinner-tailed than Gaussian. The limitations section says this clearly, but the abstract presents the numbers as market findings without the instrument-calibration framing.
**Evidence Anchor**: `text: sections/07-limitations.tex "An agent that excels here has demonstrated skill against this generator's structure, not against a market, and the realism gate says so rather than hiding it."`
**Why it matters**: Transferability is the first question an economist or practitioner asks of any synthetic-market result. The paper has the right answer (these are calibration measurements of instruments, not statements about markets) but stores it five sections downstream of where the numbers are marketed. Readers who stop at the abstract will over-read every number, especially the saturated Calm-tier Sharpe, which describes an easier-than-Gaussian world.
**Suggestion**: One clause in the abstract and one sentence in the introduction framing the eight findings as instrument calibration on a generator that intentionally fails full realism certification at this revision. This costs nothing and pre-empts the standard referee objection.
**Severity**: Minor
**Confidence**: 4 — core expertise: stylized facts and simulator validation

### W5: The discrete-time batch clearing is a frequent-batch-auction world, and the absence of fees, latency, and queue competition is not stated as an economic scope condition
**Problem**: The endogenous shared-book market clears aggregate flow at a single price per step, and the LOB market ships a call-auction uncross; both are, economically, batch-auction market designs in the Budish, Cramton, and Shim (2015, QJE, "The high-frequency trading arms race") sense. This is a defensible and arguably attractive design choice, but it means whole classes of real-market behavior (latency competition, queue-position value, sniping, maker-taker fee games) do not exist in the environment, and no fee or rebate structure is mentioned anywhere. The paper never says so; the limitations section discusses leak channels and stylized impact but not the clearing mechanism's scope.
**Evidence Anchor**: `absence: Sections 3.5 and 7 — expected a statement that batch clearing removes latency/queue/fee competition from the strategy space; checked 03-environment.tex, 07-limitations.tex, 06-related.tex`
**Why it matters**: For "who trains on this," it matters that an agent trained here never encounters the dominant cost axes of real intraday trading (fees and queue priority) and never learns speed matters. Framed positively, the environment is a clean FBA laboratory, which is itself a cross-disciplinary selling point to the market-design community; unframed, it is a silent scope restriction.
**Suggestion**: Add a short paragraph naming the clearing mechanism as batch-auction-like, citing Budish et al. (2015), stating the excluded competition margins, and noting whether fees are modeled (and if not, why the regret and markout numbers are gross rather than net).
**Severity**: Minor
**Confidence**: 4 — core expertise: market design and clearing mechanisms

### W6: "Provable optimum" for Avellaneda-Stoikov should carry its model-boundedness caveat
**Problem**: The AS closed form is optimal only within the AS model (arithmetic Brownian mid, exponential fill intensities, CARA utility, finite horizon), and the half-spread expression is itself an asymptotic approximation in the original paper. Since the environment IS the AS model with a frozen parameter set, zero regret for the closed form is legitimate, but the phrase "provable optimum" invites over-reading: agents optimized against this task are being optimized toward AS-world assumptions, notably the absence of adverse selection in the fill process, which the paper's own F6 module treats as first-order elsewhere.
**Evidence Anchor**: `text: sections/03-environment.tex "optimal half-spread $\delta^* = \gamma\sigma^2\tau/2 + \gamma^{-1}\ln(1+\gamma/\kappa)$, skewed by inventory"`
**Why it matters**: A leaderboard framed around regret to a model-internal optimum is a benchmark of model-fitting, not of market making. Fine, and useful as a calibration instrument, but the reader should not carry "provable" outside the model boundary; and the disconnect between the MM task (no adverse selection) and F6 (adverse selection is the maker's central cost) deserves one sentence.
**Suggestion**: Add the model-boundedness caveat where the optimum is introduced, note the approximation status of the closed form, and cite Gueant, Lehalle, and Fernandez-Tapia (2013, Mathematics and Financial Economics, "Dealing with the inventory risk") as the rigorous treatment of the same problem.
**Severity**: Minor
**Confidence**: 5 — core expertise: optimal market-making models

### W7: Deployment and stakeholder implications of a verifiable trading-agent leaderboard are not discussed
**Problem**: The paper is explicitly building toward third-party agent submission and a leaderboard (Section 7 notes the intake does not exist yet). Two stakeholder effects go unexamined. First, a byte-verifiable score on a stylized market is exactly the kind of number that migrates into marketing ("our agent achieves deflated Sharpe X on SharpeArena") aimed at retail audiences who cannot parse the synthetic-generator caveat; the environment's cryptographic-grade verifiability makes the number more persuasive, not more externally valid. Second, the probe layer is dual-use by construction: `impact_boundary_sweep` and `size_response` are, functionally, tools for searching where manipulation becomes profitable under a given impact specification, and while the shipped module carries a disclaimer, the paper does not discuss the dual-use surface of publishing a calibrated manipulation-payoff search harness.
**Evidence Anchor**: `absence: Section 7 or a broader-impact statement — expected discussion of leaderboard-score misuse toward retail audiences and of the manipulation probe's dual-use surface; checked 07-limitations.tex, 08-reproducibility.tex, 01-introduction.tex`
**Why it matters**: NeurIPS datasets-and-benchmarks reviewing expects a broader-impact treatment, and in the financial domain the specific failure modes are known: benchmark scores repurposed as performance advertising, and red-team tooling repurposed as strategy search. Both are cheap to address and expensive to ignore.
**Suggestion**: Add a short broader-impact paragraph: (i) an explicit statement that SharpeArena scores are claims about a synthetic environment and a norm (or license text) discouraging their use in performance marketing; (ii) one sentence on why the manipulation probe's publication is net-positive (it only ever certifies stylized specifications, per W1) once W1's reframing is in place.
**Severity**: Minor
**Confidence**: 3 — adjacent field: applying general benchmark-governance standards to finance

---

## Detailed Comments

### Title & Abstract
The abstract is dense but accurate about the engineering claims. Per W4, the empirical sentences should carry the instrument-calibration framing; per W1, "the manipulation probe finds pumping unprofitable at every tested impact coefficient and size" should not be presented as an open empirical question that happened to resolve reassuringly.

### Introduction
The three upstream failures (look-ahead, non-reproducibility, unverifiable trajectories) are real and well argued, and the Gym interface-ownership analogy is apt. From the economics side, the introduction would benefit from one sentence acknowledging that the environment's economic content (linear impact, AS quoting, synthetic tape) is deliberately stylized, so the reader calibrates expectations before Section 5.

### Design principles (Section 2)
"Distrust the simulator" is the paper's best idea. My W1 and W2 are best read as holding the probe layer to the principle's own standard: a probe that cannot fail (W1) or whose failure is misread as success (W2, the positive informed markout levels) does not yet discharge it.

### Environment (Section 3)
The Kyle-plus-Almgren-Chriss composition and the call-auction LOB are clearly described. See W5 on the unstated market-design consequences and W6 on the AS optimum. The DeferredDesk construction (no code path from claim to resolving datum) is elegant and, to my knowledge, novel in this setting as a structural treatment of post-episode information.

### Contract and governance (Section 4)
Outside my remit except to note that "governed like a protocol rather than a library API" is the right frame, and that the governance promise being "social" while the replay properties are "technical" is an honest and useful distinction.

### Experiments (Section 5)
F1-F3 are competent instrument calibrations and the F1 unit-bug correction is refreshingly candid. F4 is the model of how to report a self-test. F5 needs the Huberman-Stanzl/Gatheral reframing (W1). F6 needs the level-versus-gap distinction (W2). F7 is a nice result and its headline ("half of everything that goes wrong is process, not PnL") is the paper's most transferable practitioner lesson. F8 needs replication or softer claims (W3).

### Related work
ABIDES is the right comparison and the trade-off statement is fair. The market-simulation paragraph should add the manipulation-theory citations (W1) and, ideally, the agent-based-market-ecology lineage (W3); the market-design connection (W5) is optional but would broaden the audience.

### Limitations
Strong section; the best in its genre I have reviewed recently. W4, W5, and W7 identify caveats that belong here (or upstream) but are currently absent or under-placed.

### Conclusion
No separate conclusion section; the limitations section effectively serves the role. Acceptable for the track.

---

## Questions for Authors

1. Given Huberman and Stanzl (2004), what parameterization of your current linear-impact market could the manipulation boundary sweep ever have flagged as profitable? If none exists, what does F5 certify beyond internal consistency, and would you consider a concave- or transient-impact ablation so the probe has a failure mode?
2. In F6, makers earn positive markouts against informed flow at all horizons. Do you interpret this as (a) intended (informed alpha calibrated weak), (b) a consequence of striking fills at the pre-move mid, or (c) a defect the probe should report against the environment under your Section 2 standard? What changes if fills are struck post-impact?
3. Is the F8 result stable across ecology seeds and shock orderings? If you have run more than the one matched pair, please report the distribution; if not, please say so in the text.
4. Are trading fees, rebates, or a bid-ask spread cost modeled anywhere in the canonical position-trading environment, and are the F1/F3 deflated Sharpes gross or net? If gross, how should entrants interpret the "honest zero"?

---

## Minor Issues

### Language / Grammar
- Section 5.6 uses "subsidised" (British spelling) while the rest of the paper uses American spelling ("standardized", "generalization"); harmonize.

### Citation Format
- The impact-model citations (Kyle 1985, Almgren-Chriss 2001, Avellaneda-Stoikov 2008) are present; the manipulation-theory and adverse-selection-theory anchors (Huberman-Stanzl 2004, Gatheral 2010, Glosten-Milgrom 1985) are absent and needed for W1/W2.

### Figures and Tables
- Table tab:f5 reports impact PnL "in units of $10^{-4}$, per-bar"; state the notional base (per unit of wealth? per push notional?) so the magnitudes are interpretable.
- Table tab:f6 units ("per filled unit") should specify the price/quantity normalization.

### Layout
- None noted.

---

## Criterion-Bound Judgements

Calibration status: `NOT_CALIBRATED`

Current seat reports cannot know the final actual panel topology and never self-upgrade from a candidate profile.

| Dimension | Criterion source | Judgement | Evidence anchor(s) | Rationale | Uncertainty / scope limit | Decision bearing? |
|---|---|---|---|---|---|---|
| Originality | Reviewer remit (perspective: economic framing) | MEETS | `text: sections/02-principles.tex "Distrust the simulator too."` | Probe-the-simulator stance and DeferredDesk are genuinely fresh from the economics side; the microstructure content itself is textbook-classical by design | I do not audit novelty against the full RL-benchmark literature (R2's remit) | yes: supports publishability |
| Methodological Rigor | Reviewer remit (economic validity of probes) | PARTLY_MEETS | `table: Table (tab:f6) — positive informed markouts at all horizons`; `text: sections/05-experiments.tex "the probe certifies that the Kyle-plus-Almgren-Chriss book prices this manipulation as a cost"` | W1 (near-tautological manipulation probe) and W2 (misread markout levels) mean two of five probes certify less than claimed; F8 is n=1 | Engineering/determinism rigor is outside my remit and looks strong | yes: drives Major Revision |
| Evidence Sufficiency | Reviewer remit (transfer of synthetic conclusions) | PARTLY_MEETS | `text: sections/07-limitations.tex "the full stylized-facts conjunction passes on only 1 of 24 certified runs"` | Evidence is sufficient for instrument-calibration claims, insufficient for the market-level readings the abstract's phrasing invites (W4); F8 needs replication (W3) | Judged only on economic-transfer grounds | yes: drives reframing requests |
| Argument Coherence | Reviewer remit | PARTLY_MEETS | `text: sections/02-principles.tex "or that lets makers profit from informed flow"` | Section 2 states a standard that F6's own levels violate without the paper noticing the tension (W2) | Coherence of the systems argument is others' remit | yes: W2 |
| Writing Quality | General standards | MEETS | `text: sections/05-experiments.tex "The corrected board is the honest zero"` | Clear, direct, unusually candid prose; dense abstract | Style conventions differ across fields | no |
| Literature Integration | Reviewer remit (economics literature only) | PARTLY_MEETS | `absence: Related work — expected manipulation-theory and market-ecology citations; checked 06-related.tex, refs usage in 03/05` | Classical microstructure cited; manipulation theory (Huberman-Stanzl, Gatheral), equilibrium adverse selection (Glosten-Milgrom), and market-ecology lineage absent | Systematic coverage audit is R2's remit; this is a perspective-level gap list | yes: required additions for W1/W2/W3 |
| Significance & Impact | Reviewer remit (stakeholders/deployment) | MEETS | `text: sections/04-contract.tex "A leaderboard entry produced on this environment can be recomputed by anyone from the frozen seed and the submitted decisions"` | Verifiable trading-agent evaluation is a real gap; significance survives my weaknesses because they are repairable framing/calibration issues | Impact conditional on the not-yet-existing intake (paper concedes this); broader-impact statement missing (W7) | yes: supports Major Revision over Reject |

Recommendation rationale: the decision-bearing unresolved criteria are Methodological Rigor and Argument Coherence as they apply to the probe layer (W1, W2, W3), all repairable through reframing, added theory citations, one or two additional sweeps, and F8 replication, without touching the environment's core engineering claims. Strength on significance does not offset these; it is why the repair is worth requiring rather than rejecting.

---

## Cross-Disciplinary Reading Recommendations

- Huberman, G. and Stanzl, W. (2004), "Price manipulation and quasi-arbitrage", Econometrica. Directly governs W1: linear permanent impact excludes profitable round-trip manipulation, making F5's negative result expected rather than informative.
- Gatheral, J. (2010), "No-dynamic-arbitrage and market impact", Quantitative Finance. Extends the manipulation-exclusion logic to transient impact with decay; the natural theory anchor for a probe ablation that could actually fail.
- Glosten, L. and Milgrom, P. (1985), "Bid, ask and transaction prices in a specialist market with heterogeneously informed traders", Journal of Financial Economics. The equilibrium adverse-selection benchmark against which F6's fixed-path markout design should be positioned (W2).
- Budish, E., Cramton, P., and Shim, J. (2015), "The high-frequency trading arms race: frequent batch auctions as a market design response", Quarterly Journal of Economics. Frames the environment's batch clearing as an FBA design and names the competition margins it removes (W5).
- Farmer, J. D. (2002), "Market force, ecology and evolution", Industrial and Corporate Change; and Lux, T. and Marchesi, M. (1999), "Scaling and criticality in a stochastic multi-agent model of a financial market", Nature. The market-ecology lineage the replicator layer should be situated in (W3).
- Almgren, R., Thum, C., Hauptmann, E., and Li, H. (2005), "Direct estimation of equity market impact", Risk. Empirical concave-impact evidence motivating the W1 ablation.
- Gueant, O., Lehalle, C.-A., and Fernandez-Tapia, J. (2013), "Dealing with the inventory risk: a solution to the market making problem", Mathematics and Financial Economics. Rigorous treatment of the AS problem; supports the "provable optimum" caveat (W6).
- [UNVERIFIED] Franke, R. and Westerhoff, F., work on structural stochastic volatility and method-of-simulated-moments calibration of agent-based market models (Journal of Economic Dynamics and Control, circa 2012). Search lead for calibrating the generator to real-market moments, which would address the F4 failures more systematically than the vol_clustering knob.

---

Reviewer note on scope: I have not evaluated statistical methodology (R1), literature completeness (R2), or internal-consistency stress testing (DA), and my recommendation weighs only the economic-perspective findings above.
