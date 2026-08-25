# Rigor, Structure and Coherence Audit, 2026-08-25

Subject: `paper/main.tex` and every `sections/*.tex` fragment as of Git `v0.13.0-1-gfa49438`, after the five-seat review and the Codex correction pass. The parallel math audit is out of scope here; this report covers structure, claim-support alignment, scientific depth, coherence, writing quality and citations.

Method: full read of every input fragment; grep sweeps for dashes, ditto cells, labels, references, citation keys and revision-relative language; spot-checks of reported numbers against `paper/evidence/*.json` and `paper/evidence/provenance.json`; inspection of `paper/src/make-*.py`, `crates/sharpearena-py/python/sharpearena/generalization.py`, `docs/training.md` and the Python and Rust test trees for the claims that name enforcing code. Nothing was modified.

Severity scale: CRITICAL blocks submission; MAJOR must be fixed before re-review; MINOR should be fixed; OK records a check that passed.

Counts: CRITICAL 0, MAJOR 9, MINOR 17, OK 12.

---

## 1. Structure

**S1. MAJOR. The counting scheme "ten findings plus three follow-ups" is never made legible on the page.** `sections/05-experiments.tex:3` says "Ten findings, eight core calibrations plus two reviewer-motivated probes, and three follow-ups", but only F1 to F8 carry numbers. The reader must infer that the two unnumbered probes are the concave ablation (`concave-fragment.tex:1`) and the predictability probe (`predictability-fragment.tex:5`), and that the follow-ups are the witness, the positive control and the sealed seeds. The abstract (`00-abstract.tex:2`) and the contributions list (`01-introduction.tex:11`, item 6) repeat "ten" and "three" without naming them. Fix: give every subsection a stable tag (F5a concave, F5b positive control, F9 predictability, F9a sealed seeds, F1a witness, or number the follow-ups W1 to W3) and list the mapping once in the Section 5 preamble.

**S2. MAJOR. The follow-ups are interleaved with their parent findings but the preamble describes them as answering "questions the ten left open".** The witness sits between F1 and F2 (`05-experiments.tex:46`), before nine of the ten findings exist; the sealed-seeds fragment sits after the predictability probe (`05-experiments.tex:302`); the positive control after the concave ablation (`05-experiments.tex:190`). The placement is defensible (each follow-up is adjacent to its parent) but the text never says so, so the section reads as spliced. Fix: one sentence in the preamble, "each follow-up is placed directly after the finding it answers", and retitle the follow-up subsections with the parent tag so the interleaving is visible in the table of contents.

**S3. MAJOR. Revision-relative language makes the paper read as a diff against an earlier draft rather than a standalone document.** Instances: "is now reported only as exploratory" (`00-abstract.tex:2`); "overturns its original single-seed headline" (`00-abstract.tex:2`); "in this revision" and "were regenerated after the estimator and dimensional corrections" (`05-experiments.tex:3`); "including the one this paper's earlier draft promoted to the abstract" (`05-experiments.tex:284`); "The newer opt-in jump-burst knobs" (`05-experiments.tex:129`); "the old public $2^{16}$ band" (`sealed-seeds-fragment.tex:12`); "can now bind that hash" (`sealed-seeds-fragment.tex:10`); "Two named earlier are done" (`07-limitations.tex:9`); "The current F4, F5 and witness artifacts were regenerated after their estimator corrections" (`02-principles.tex:19`); "byte-identical to their previous values" (`A-commands.tex:54`). A first-time reader has no "before". Fix: delete every reference to a prior draft or prior artifact state; where the point is scientific (the single-seed F8 narrative does not survive replication), state it as a result of the replication, not as a retraction of an earlier version.

**S4. MAJOR. The provenance statement is triplicated nearly verbatim and still omits the one fact it needs.** `02-principles.tex:19`, `05-experiments.tex:3` and `A-commands.tex:3` each say F1 serializes 0.9.0, legacy files record SharpeBench 0.5.0 where stated, and other artifacts carry no runtime version. None of the three states the Git revision that `provenance.json` records (`repository_head` = `9f16e9fb`), even though all three say provenance "is the containing Git revision". Fix: keep one statement (Section 8 or Appendix A), print the short hash, and state which artifacts were produced at that head (per `provenance.json`, all 25) and which were produced by an older package (F1, 0.9.0).

**S5. MINOR. Duplicated scoping sentences across abstract, introduction, Section 3.7, Section 5 preamble, F4 and limitations.** The 2-of-24 / 22-of-24 certification figure appears six times (`00-abstract.tex:2`, `01-introduction.tex:13`, `03-environment.tex:41`, `05-experiments.tex:5`, `05-experiments.tex:127`, `07-limitations.tex:3`) with three different nouns ("seeded panels", "runs", "canonical runs"). "Certification is reported beside the results, not required" appears three times. Fix: abstract and F4 keep the number; the rest cross-reference F4; pick one noun (panel).

**S6. MINOR. The positive-control "not searched" list is stated three times in two limitation paragraphs.** `07-limitations.tex:9` ("push size, $\lambda$, legs longer than ten bars, holds longer than one bar or short-side trips") and `07-limitations.tex:13` ("push size, $\lambda$, longer legs, longer holds or short-side trips") repeat `positive-control-fragment.tex:20`. Fix: state once in the fragment, cross-reference from limitations.

**S7. MINOR. The intro's "What the evidence shows" paragraph over-compresses F3.** `01-introduction.tex:13` says "Cross-regime transfer costs the Calm-selected reference nearly its entire DSR", but F3 (`05-experiments.tex:95`) resolves only Calm to Extreme beyond doubt; Calm to Hard's interval reaches 0.054. Fix: "costs the Calm-selected reference 0.97 of DSR on Extreme (resolved) and 0.88 on Hard (interval reaches 0.05)".

**S8. OK.** No conflicting numbers were found between abstract, introduction, Section 5 preamble, limitations and the evidence JSONs for F2, F3, F4, F8, predictability, sealed seeds and witness (spot-checked: F2 regret dispersion, F3 transfer CIs, F4 pass rates, F8 `multi_seed`, predictability `seed_search`, witness `boundaries` and `config`, sealed-seeds `config`).

**S9. OK.** The D&B arc is coherent: failure classes (Section 1), principles (2), enforcing code (3), contract (4), calibrations (5), related work (6), limitations (7), reproducibility (8), commands (A). Every section maps to a contribution item.

---

## 2. Claim-support alignment

Claim map (abstract and contributions, with the section that carries the evidence):

| Claim | Where | Evidence | Status |
|---|---|---|---|
| Interface excludes future-bar access | abstract; C1 | 3.1 property test | supported |
| Rust core pinned byte-for-byte across native, WebAssembly and Python bindings | abstract; C1; 4.4; 7 | Rust golden FNV-1a (`lob_market.rs:497`), WASM parity tests | see A1 |
| Trajectories recomputed from decisions | abstract; C1 | 3.3, npm tamper test | supported |
| Governed contract v1.0 | C2 | 4 | supported |
| Disjoint train / gap / held-out bands | C3; 3.4 | `train_test_seeds`, `docs/training.md` | see A2 |
| Regret against AS reference | C4; F2 | `f2-regret.json` | supported |
| Concave path leaves linear golden path byte-identical | C5; concave fragment | `test_market_concavity.py` | supported |
| Sealed derivation keeps "the disjointness proof public" | C5 | sealed fragment | see A3 |
| 22/24 fail realism | abstract | F4 | supported |
| Volatility-persistence driver improves clustering | abstract | F4 (in-sample) | supported with caveat stated |
| U-shaped regret, minimum unresolved | abstract | F2 CIs | supported |
| Best no-follower cell +21.8, familywise positive | abstract | positive-control fragment | supported |
| All 45 linear cells negative | abstract | positive-control fragment | supported |
| Honest AR near baseline; band recovered 16/16 | abstract | predictability | supported |
| Sealed 0/16 with salt withheld | abstract | sealed seeds | supported |
| Witness: eligibility attainable on every tier, PSR conjunction last to open | abstract | witness | supported |
| F8: replacement in 1 of 8 | abstract | `f8-ecology.json multi_seed` | supported |

**A1. MAJOR. "Pinned byte-for-byte across ... Python bindings" is not backed by a Python-side golden test.** The abstract (`00-abstract.tex:2`), 4.4 (`04-contract.tex:19`) and limitations (`07-limitations.tex:11`: "where golden hashes and parity tests enforce it") name the Python binding inside the byte-identity guarantee. The Rust crate pins `GOLDEN_TAPE_FNV1A` and the WASM crate asserts native parity, but the Python test tree contains no fingerprint assertion: `test_market_concavity.py:4` only states in a docstring that the default remains the golden path, `test_equivalence.py` proves batched equals scalar within Python, and `test_market.py:134` proves same-seed determinism within Python. That is determinism of the binding against itself, not byte-identity against the native golden value. Fix: either add a Python test that recomputes the committed FNV-1a fingerprints through the binding, or narrow the wording to "the Rust core and its WebAssembly build are pinned by golden hashes; the Python binding calls the same compiled engine and is tested for self-determinism".

**A2. MAJOR. The "held-out" band exercised by F3 and the witness lies inside the gap the protocol says is never sampled.** Section 3.4 (`03-environment.tex:21`) fixes canonical bands: train $[0,256)$, a gap of ten thousand seeds "that is never sampled", held-out $[10256,10512)$. `make-f3-generalization.py:134` computes `test_seeds = range(16 + 10000, 16 + 10000 + 16)`, i.e. $[10016,10032)$; `make-witness.py:97` calls `train_test_seeds(16, 16, 0, 10000)` and lands on the same band (`witness.json config.bands.held_out`). Both bands fall inside $[256,10256)$. The construction is internally sound (16-seed train, 10,000 gap, 16-seed test), but it is not the canonical band the paper advertises, and the text calls it "held-out" without qualification (`05-experiments.tex:65`, `arena-witness-fragment.tex:19`). This is the residue of the Round 1 finding SC-10 ("protocol defined but never exercised"). Fix: state in 3.4 that the canonical bands are the 256-seed instance of `train_test_seeds(n, n, 0, 10000)` and that the paper's 16-seed evidence uses the 16-seed instance $[10016,10032)$; rename the witness table's band column to "16-seed split" or give the seed range.

**A3. MAJOR. Three different things are called "held-out".** (i) The canonical band $[10256,10512)$ (3.4); (ii) the 16-seed split band $[10016,10032)$ (F3, witness); (iii) the `EVAL_SEED_BASE = 10^6` namespace with its "named held-out regression set" (3.4, `03-environment.tex:21`), which is the band the predictability scanner attacks (`predictability.json config`: scan band $[1000000, 1065536]$) and the sealed derivation replaces (`sealed-seeds-fragment.tex:3`: "The public held-out set is disjoint from training but enumerable"). The sealed-seeds fragment's opening sentence therefore refers to (iii) while the reader has just left F8 and F3 thinking of (ii). Fix: introduce three names in 3.4 (train band, split band, evaluation namespace) and use them consistently; C5's "disjointness proof" should read "disjointness by construction" (the fragment's own wording) since the paper elsewhere toned down "provably disjoint".

**A4. MAJOR. The "instrument calibration on uncertified tape" framing is applied to experiments that do not run on the certified generator at all.** The Section 5 preamble (`05-experiments.tex:5`) says "the findings are instrument calibrations on a generator whose canonical tape the realism gate of F4 declines to certify". F4 grades the tier generator, which F1, F3, F7, F8, the witness and the predictability probe consume. F2 runs the Avellaneda-Stoikov market-making environment (`make-f2-regret.py` docstring: `MarketMakingEnv`, its own $\sigma, \gamma, \kappa$ arrival model), F5 and its two follow-ups run the endogenous shared-book market, and F6 runs an exogenous drift path inside the adverse-selection module. None of those price processes is graded by `certify_realism`, so for them the honest statement is stronger than "uncertified": their tape was never submitted to the gate. Fix: split the first scoping statement into (a) experiments on the tier generator, where certification fails 22/24, and (b) experiments on the market-model price processes, which the realism gate does not evaluate at all; add the same one-line qualifier to F2, F5 and F6.

**A5. MINOR. Abstract "opt-in volatility-persistence driver" versus body "vol_clustering knob adds deterministic EMA persistence" versus limitations "weak volatility clustering".** Three names for one mechanism. Fix: "volatility-clustering driver" everywhere, with the knob name in parentheses once.

**A6. MINOR. "Dispersion is reported throughout" (`05-experiments.tex:5`) overstates.** F4, F7 and F8 report counts without intervals; the predictability probe reports across-seed SDs and explicitly no test. Fix: "Dispersion is reported wherever a mean is interpreted: ..." and list F4, F7, F8 as count-based.

**A7. MINOR. The introduction's "held-out seed bands" sentence cites Procgen for measured generalization, then C3 claims "making generalization a measured quantity" while limitations (`07-limitations.tex:7`) correctly says F3's near-zero gaps are the expected control for parameter-free policies.** The contribution is the instrument, not a measurement of generalization. Fix: C3 reads "making generalization gap a reportable quantity; no learner has yet been measured (Section 7)".

**A8. OK.** The companion contribution boundary is stated once in one paragraph (`01-introduction.tex:9`), F1 is positioned as pipeline validation and attributes the unit bug to the companion, and the abstract contains no kernel-owned number.

**A9. OK.** Every numeric claim in the abstract maps to a subsection and to a committed JSON; none was found to exceed its section's stated scope except S7.

---

## 3. Scientific depth

**D1. OK. Scope statements demanded in Round 1 are all present.** F5 linear tautology: `05-experiments.tex:142` ("model-consistency check, not a universal certificate"). F6 exogenous path: `03-environment.tex:45` and `05-experiments.tex:216`. F8 replication: `05-experiments.tex:261` to `284`. Sealed seeds not a MAC: `sealed-seeds-fragment.tex:8` and `07-limitations.tex:5`. Witness oracle not an entrant: `arena-witness-fragment.tex:3`, `07-limitations.tex:7`, and `witness.json oracle_disclosure`.

**D2. OK. Alternative explanations are engaged where they matter.** F3 band luck (0.34 noise calibration); F4 episode length as the cause of weak autocorrelation; F6 strike-price convention and wide default depths; F7 mandate-assignment mechanism; F8 heavy-tailed single-draw; positive control selection (Bonferroni over 135) and the compounding caveat (`positive-control-fragment.tex:7`).

**D3. MAJOR. Broader impact omits the disclosure that this paper publishes a one-second scanner against the seed band the companion's currently open window uses.** Limitations state that "the currently open companion window does not claim to use" sealed seeds (`07-limitations.tex:5`), and the predictability fragment states that any bounded public deployment "should treat its scenarios as fully known to a motivated adversary" (`predictability-fragment.tex:11`). Together those sentences say the live SharpeBench window is enumerable with the published `make-predictability.py`. The broader-impact paragraph (`07-limitations.tex:19`) discusses leaderboard marketing and the manipulation probe but not this. Fix: add two sentences: the scanner is published because the vulnerability is structural and already inferable from the public generator; the operator's mitigation (sealed derivation plus salt commitment in the forward window) is shipped and the open window's status is stated; entries to that window should be read with the attack in mind.

**D4. MINOR. F4's third gated check (aggregational Gaussianity) is absent from `tab:f4` although it is part of the conjunction that decides the pass rate.** The caption says "(not shown)" (`05-experiments.tex:110`). The text gives one Calm value ($-0.58$) but no per-tier pass count. Fix: add a mean and pass column, or state per-tier pass counts in the text (the JSON has them).

**D5. MINOR. The Fano statistic's construction is described only by name.** `03-environment.tex:41` calls it "an exploratory Fano exceedance-count ratio"; `05-experiments.tex:106` gives the 121-bar / 12-window arithmetic. Neither defines the ratio (variance-to-mean of exceedance counts, 1 under Poisson). The Round 1 response letter (`response-to-reviewers.md:117`) says the construction is stated; it is not in the current text. Fix: one clause defining it at first mention.

**D6. MINOR. PSR is never expanded or cited in this paper.** `arena-witness-fragment.tex:12` ("marginal PSR reach 0.90"), `:27`, and `01-introduction.tex:13`. The probabilistic Sharpe ratio is Bailey and López de Prado (2012, Journal of Risk 15(2)); the paper cites only the 2014 deflated-Sharpe paper. Fix: expand at first use and either cite the 2012 paper or say the gate is specified in the companion.

**D7. MINOR. The positive control's economic reading is thin.** The best cells are the two arms with no follower flow, one of them with zero temporary impact; the fragment notes this but does not say what a reader should conclude: the profit appears only when the probe removes the two friction channels that the canonical arm keeps. Fix: one sentence stating that the positive is a property of the frictionless concave arm, and that the canonical arm has no pointwise-positive cell, before the "instrument can detect" conclusion.

**D8. MINOR. F1's saturated DSR of 1.0000 with a bootstrap CI upper bound of 1.0000 is a boundary artifact of a probability-valued score; the text says "saturate" but never says the ceiling is the statistic's range.** Fix: one clause, "DSR is a probability and is capped at 1".

---

## 4. Coherence

**C1. MAJOR. Notation clash: $\kappa$ denotes the Avellaneda-Stoikov order-arrival decay in 3.5 and F2 ($\kappa = 1.5$, `03-environment.tex:29`, `05-experiments.tex:50`) and the permanent-impact exponent in the concave fragment, the positive control, F5, the abstract, the introduction and the limitations ($\kappa \in \{0.5, 0.7\}$).** Both appear in Section 5 within two pages. Fix: rename the exponent (e.g. $\beta$; $\alpha$ is taken by the F6 informed alpha, $\eta$ and $\lambda$ by impact) and update `A-commands.tex:54`, the figure legends (`f5-concave.pdf`, `f5-positive-control.pdf` are generated from the JSON key `impact_exponent`) and `test_market_concavity.py` docstrings if they name $\kappa$.

**C2. MINOR. $s$ denotes mid price in the AS formula (`03-environment.tex:29`, $r = s - q\gamma\sigma^2\tau$) and oracle strength in the witness (`arena-witness-fragment.tex:5`).** Different sections, but the witness appears before F2 in reading order. Fix: use $m$ for mid or $\rho$ for signal-truth correlation (the table caption already calls $s$ "signal-truth correlation").

**C3. MAJOR. Four figures are never referenced in the text: `fig:witness`, `fig:f5-concave`, `fig:f5-positive-control`, `fig:predictability`.** Each fragment includes the float but no `\cref`. All four are the figures of the new material, so the reviewer-motivated results are the ones whose figures float unanchored. Fix: reference each where its numbers are discussed (`arena-witness-fragment.tex:27`, `concave-fragment.tex:10`, `positive-control-fragment.tex:11`, `predictability-fragment.tex:9`).

**C4. MINOR. "Acceptance set" (witness, three uses) versus "eligibility set" (F1, limitations, two uses).** Fix: "eligibility set" throughout; the witness title can keep "acceptance" only if defined as a synonym once.

**C5. MINOR. Hyphenation of name pairs is inconsistent.** "Avellaneda--Stoikov" four times, "Avellaneda-Stoikov" three; "Almgren--Chriss" once, "Almgren-Chriss" twice; "commit--reveal" twice, "commit-reveal" once. Fix: en-dash (`--`) for all author pairs and the compound "commit-reveal" with a hyphen, applied uniformly.

**C6. MINOR. Tier and verdict vocabularies are stable, with one gap.** Tiers Calm / Hard / Extreme are consistent. Failure dispositions are listed as five in 3.6 (cascade-wiped, bankrupt, stopped-out, mandate breach, clean) but `tab:f7` has no cascade column although 3.6 says the schema "always carries every mode". Fix: add the zero column or note the omission in the caption.

**C7. MINOR. Version statements.** The bib note says the companion is "Version 0.8.0" (`refs.bib:137`) while the text says the corrected kernel is 0.5.0 and F1 was produced under "the corrected kernel" (`05-experiments.tex:7`). Both are true (the companion paper version versus the kernel version that fixed the bug) but the reader is not told so. Fix: in F1 or Section 8, one clause: "kernel fix shipped in SharpeBench 0.5.0; the companion paper is cited at 0.8.0".

**C8. OK.** Appendix A commands match the text: witness coarse grid (13 points, `witness.json`), bisection resolution 0.005 (bracket widths 0.0031 are consistent with stopping below 0.005), predictability seeds and band width, sealed-seeds 16 slots, F8 eight seeds, positive-control 135 cells, F1 bootstrap parameters ($B$ = 2000, `resample_seed` matches `f1-baselines.json config.bootstrap`).

**C9. OK.** Every `\label` is defined once and every `\cref` resolves; `main.log` reports no undefined references or citations, 29 pages, no overfull boxes.

**C10. OK.** "Leak-free at the interface" phrasing is applied consistently: abstract, C1, principle 1, 3.1, related work ("structurally unavailable at the interface"), limitations. No residual "impossible by construction".

**C11. MINOR. The Section 3.2 sentence "Held-out evaluation therefore requires high-entropy unpublished seeds, not merely disjoint public bands" (`03-environment.tex:13`) is not reflected in 3.4, which presents the public canonical bands as the protocol without a pointer to the sealed option.** Fix: one sentence at the end of 3.4 cross-referencing the sealed derivation as the recommended held-out mode for adversarial settings.

**C12. MINOR. Limitations say sealed seeds "remove the bounded-band attack" (`07-limitations.tex:9`) whereas the fragment says they defeat "this particular bounded-band enumeration budget" (`sealed-seeds-fragment.tex:12`).** Fix: use the fragment's wording in limitations.

**C13. MINOR. Positive-cell count drifts: "two selected $\kappa=0.5$ cells" (`01-introduction.tex:13`, `05-experiments.tex:172`, figure caption) versus "a profitable round trip" and "finds one that pays" (`07-limitations.tex:9`, `:13`).** Fix: "two frictionless-arm cells" in limitations.

---

## 5. Writing quality

**W1. OK. House rules hold.** No em-dash (U+2014), no en-dash outside `--` in name compounds, no `---` sequence; no ditto cells or `"` placeholders in any table.

**W2. MINOR. "Honest" appears twelve times across seven files, including "honest zero", "honest conclusion", "honest reading", "honest adversary", "honest statistical adversary", "honest regression".** In the predictability fragment the word is a technical label (the adversary without generator knowledge) and in F1 and limitations it is self-description. The self-descriptive uses read as reassurance. Fix: keep the technical label, replace "honest zero" with "corrected zero" or "baseline zero", "The honest conclusion is methodological" with "The conclusion is methodological", "The honest reading of the probe" with "The probe therefore shows".

**W3. MINOR. Rhetorical flourishes that the anti-slop pass should trim:** "The lesson is sharp" (`predictability-fragment.tex:11`), "worst citizen" and "regardless of the weather" (`05-experiments.tex:250`), "we report that plainly" (`:282`), "with the same candor as F4" (`:216`), "The minimum deserves its interval" (`:52`), "is worth stating" (`02-principles.tex:19`, `04-contract.tex:19`). None is wrong; each adds a sentence of voice where the number already speaks. Fix: delete or flatten.

**W4. MINOR. Abstract is 281 words and one paragraph of eleven sentences with seven numeric results.** NeurIPS has no hard cap, but the Round 1 should-fix S1 asked for halving and the current text reads as a results list. Fix: cut the F2, concave and witness sentences to clauses; keep 22/24, +21.8 familywise, 16/16 versus 0/16, 1 of 8.

**W5. MINOR. Paragraph-level "one message" check fails in three places.** `05-experiments.tex:37` (F1 reading) carries three messages: conjunction logic, the honest zero, and the forward reference to the witness. `05-experiments.tex:95` (F3 matrix) carries the noise calibration, the two Calm cells, the Hard-to-Extreme cell, the meta-result about 16-seed boards and the antisymmetry remark. `predictability-fragment.tex:20` carries the open problem, the finalizer inversion result and the bounded-band conclusion. Fix: split each at the message boundary.

**W6. MINOR. Title retains the definite article at the operator's instruction (Round 1 S1).** Recorded, not relitigated: "The Point-in-Time (PIT) RL Environment" is a superlative the limitations do not support ("no third party has yet submitted an agent"). If the article stays, the abstract's first sentence should not also lead with "SharpeArena is a point-in-time" (indefinite), which is the paper contradicting its own title within two lines.

---

## 6. Citations

Eight spot-checks of use against source (metadata verified against the bibliographic record from memory; no network was used, so DOIs were not resolved):

| Key | Metadata | Use in text | Verdict |
|---|---|---|---|
| `cont2001empirical` | QF 1(2):223-236, correct | kurtosis, absolute-return autocorrelation, gain/loss asymmetry, aggregational Gaussianity (`03-environment.tex:41`, `05-experiments.tex:106`) | OK; all four are in Cont's catalogue; Zumbach and Fano are no longer attributed to Cont |
| `zumbach2009time` | QF 9(5):505-515, correct | timescale asymmetry "in the sense of Zumbach" | OK |
| Fano ratio | uncited, presented as the authors' exploratory diagnostic | `03-environment.tex:41` | OK as ownership; see D5 for the missing definition |
| `gueant2013dealing` | Math. Financ. Econ. 7(4):477-507, correct | exact treatment of the AS problem (`03-environment.tex:29`); co-cited with AS in C4 | OK; in C4 the pairing implies Guéant et al. supply the shipped policy, which they do not; cite there only AS |
| `huberman2004price` | Econometrica 72(4):1247-1275, correct | linear time-independent permanent impact rules out round-trip manipulation | OK; the necessity direction (linearity is required for no quasi-arbitrage) is the paper's main theorem and the text's "under the precise linear ... assumptions ... ruled out" is a fair sufficiency reading |
| `gatheral2010no` | QF 10(7):749-759, correct | no-dynamic-arbitrage conditions; nonlinear permanent impact admits profitable asymmetric constructions (`concave-fragment.tex:19`) | OK |
| `budish2015high` | QJE 130(4):1547-1621, correct | single-price batch clearing is closer to a frequent batch auction than to a continuous book (`03-environment.tex:27`) | OK |
| `avellaneda2008high` | QF 8(3):217-224, correct | reservation price and half-spread $\gamma\sigma^2\tau/2 + \gamma^{-1}\ln(1+\gamma/\kappa)$ | OK; the half-spread is the source's spread $\gamma\sigma^2(T-t) + (2/\gamma)\ln(1+\gamma/k)$ halved |

Additional checks: `glosten1985bid`, `allen1992stock`, `kyle1985continuous`, `almgren2001optimal`, `farmer2002market`, `lux1999scaling`, `bailey2014deflated`, `machado2018revisiting` carry correct journal, volume and page metadata and are used for the claims their abstracts support.

**R1. MINOR. Bib hygiene: `almgren2005direct` is in `refs.bib:214` but never cited.** The 2026-08-25 integrity report states "every bibliography key is cited"; that is false for this key. With `plainnat` the entry is silently dropped, so the PDF is unaffected. Fix: cite it in the concave fragment as the empirical motivation for $\kappa<1$ (square-root impact), which strengthens the ablation, or delete it.

**R2. MINOR. Missing standard citations.** (i) SplitMix64: Steele, Lea and Flood, "Fast splittable pseudorandom number generators", OOPSLA 2014, for `predictability-fragment.tex:20` and `sealed-seeds-fragment.tex:8`. (ii) Probabilistic Sharpe ratio: Bailey and López de Prado 2012 (see D6). (iii) FNV-1a: the Fowler-Noll-Vo IETF draft, optional. (iv) The introduction's "look-ahead bugs are a pervasive ... failure mode of backtests" (`01-introduction.tex:3`) leans on `bailey2014deflated`, which is about selection bias; look-ahead is treated in López de Prado, Advances in Financial Machine Learning (2018), chapter 11. Fix: add (i) and (ii); either add (iv) or drop the look-ahead clause from that citation.

**R3. MINOR. `toca2026sharpebench` still has no persistent locator** (`refs.bib:132`: GitHub URL, "Preprint", note "Version 0.8.0"). The Round 1 must-fix R8 was discharged by adding the URL and the in-text derivation; a repository URL is not a stable citation. Fix: add an arXiv identifier or a Zenodo DOI for the 0.8.0 tag.

**R4. MINOR. Title casing and entry details.** `jerome2023mbtgym` title "Mbt-gym" should be "{mbt-gym}" (the package name is lowercase). `minari` is cited as a website by "Farama Foundation" with year 2024; if the Minari paper is used elsewhere in the project, prefer it (UNVERIFIED: an arXiv paper by Younis et al. may exist; check before citing). `cobbe2020procgen` and `jiang2021prioritized` carry `eprint` fields in `@inproceedings` entries, which `plainnat` prints as "arXiv" beside the proceedings; harmless but noisy.

**R5. OK.** All 29 cited keys resolve; no key is cited that is not in the bib; author names with diacritics are correctly braced.

---

## Fix order suggested (not ranked by severity)

1. C3 (reference the four figures) and C1 (rename the exponent) are mechanical and touch every fragment; do them first so later edits do not re-collide.
2. A2 and A3 (seed-band naming) require one paragraph in 3.4 and column relabels in the witness table.
3. A4 (which experiments run on gated tape) is one split sentence in the Section 5 preamble plus three one-line qualifiers.
4. S3 (strip revision-relative language) and S4 (one provenance statement with the hash) are text-only.
5. A1 (Python golden test or narrowed claim) is the only item that may need code.
6. D3 (broader-impact disclosure of the scanner) is two sentences.
7. S1 and S2 (finding tags and the placement sentence) close the structure items.
8. Remaining MINOR items are single-line edits.

---

## Summary

1. Scope: full read of `main.tex`, ten section files, five fragments, `refs.bib`; cross-checked against evidence JSONs, producer scripts, test trees and `docs/training.md`; no file modified.
2. Counts: CRITICAL 0, MAJOR 9, MINOR 17, OK 12.
3. House rules pass: no em-dashes, no ditto cells, no undefined references, 29 pages clean.
4. MAJOR A2/A3: F3 and the witness draw their "held-out" seeds from $[10016,10032)$, inside the gap that Section 3.4 says is never sampled; "held-out" names three different bands.
5. MAJOR A4: the "uncertified tape" framing is applied to F2, F5 and F6, whose price processes the realism gate never evaluates at all.
6. MAJOR A1: byte-pinning "across Python bindings" has no Python-side golden-hash test; only Rust and WASM are pinned.
7. MAJOR C1: $\kappa$ is both the AS arrival decay (F2) and the impact exponent (F5 follow-ups) within the same section.
8. MAJOR C3: the four new-material figures (witness, concave, positive control, predictability) are never referenced.
9. MAJOR S1/S2/S3/S4: the ten-plus-three count is never spelled out, follow-up placement is unexplained, revision-relative language ("now", "earlier draft", "regenerated", "previous values") persists in abstract and body, and the provenance statement is triplicated without the Git hash.
10. MAJOR D3: broader impact does not disclose that the published band scanner applies to the companion's currently open public window.
11. Reviewer-demanded scope statements (F5 tautology, F6 exogenous path, F8 replication, sealed seeds not a MAC, witness oracle not an entrant) are all present and correctly placed.
12. Citations: eight spot-checks pass on metadata and use; `almgren2005direct` is uncited (integrity report claim is wrong), SplitMix64 and PSR lack sources, and the companion citation still has no persistent locator.
