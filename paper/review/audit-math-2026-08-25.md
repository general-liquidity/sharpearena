# Mathematical and Statistical Audit, SharpeArena paper

Date: 2026-08-25. Read-only audit of `paper/main.tex` + `sections/*.tex` (including input fragments), `paper/evidence/*.json`, `paper/src/make-*.py`, and the cited code (`crates/sharpearena/src/{scenario_gen,market,lob_market,leaderboard_ci}.rs`, `crates/sharpearena-py/python/sharpearena/{market_making,adverse_selection,manipulation,realism,generalization,confidence,eval_seeds}.py`, plus `sharpebench/crates/sharpebench-sim/src/data.rs` for the base generator). Every quantitative claim below was recomputed from the committed evidence with numpy/scipy (Python 3.12.6, scipy 1.16.2); code paths were read, and where the environment could not be imported the RNG semantics were replicated and tested directly.

Severity: CRITICAL = a reported number or headline conclusion is wrong. MAJOR = a conclusion or its stated strength is not supported as written, or an inconsistency a reader will trip over. MINOR = correct but improvable, or a small mismatch. OK = verified.

Summary counts: 0 CRITICAL, 6 MAJOR, 14 MINOR, 21 OK.

---

## 1. Models as stated vs as implemented

### 1.1 Kyle + Almgren-Chriss clearing. OK

`market.rs:9-40` and `clear_bar_concave` (`market.rs:867-1090`). As documented: cleared reference mid = exogenous mid times the running permanent multiplier `M_t` (prior flow only); fill = `mid * (1 + f_t * (lambda*Q + eta*q_i)/V)`; `M_{t+1} = M_t * (1 + lambda*Q/V)`. The permanent piece is charged once (execution at the post-impact price, which then becomes the next reference), so the composition genuinely is Kyle permanent + AC temporary, and the paper's description (03-environment.tex:25) matches the code. The mark-to-market at the pre-impact cleared mid books the impact cost immediately and recovers the permanent piece next bar; internally consistent.

### 1.2 Nonlinear exponent on dimensionless flow. OK

`signed_pow` (`market.rs:311-328`) and the branch at `market.rs:991-996` and `market.rs:1055-1059`: exponent applied to `net_flow/V` in both the temporary crowd term and the permanent update; own-size term stays `eta*q_i/V`; `exponent == 1.0` is special-cased to return the identical bits, so the linear golden path is preserved exactly as claimed (concave-fragment.tex:8). The dimensional argument in the fragment is correct: `n * I_k(Q/n; V) = n^(1-k) * I_k(Q; V)` follows from `lambda * (Q/(nV))^k * n = n^(1-k) * lambda * (Q/V)^k`. Verified symbolically.

- MINOR (M-01). The concave ablation at fixed `lambda` conflates impact shape with scale: at the tested calibration `|Q/V| ~ 1e-3`, so `|x|^0.5 / x ~ 30`, i.e. kappa=0.5 is also a roughly 30x impact amplification. The limitations paragraph (07-limitations.tex:13) discloses this ("two to three orders of magnitude below the unit crossover... amplify impact"), so it is not hidden, but the linear-vs-concave rows of Fig. f5-concave are not a controlled shape comparison. Fix: add a scale-matched arm with `lambda_k = lambda * x0^(1-k)` at the canonical flow `x0`, or state in Sec. F5-concave (not only limitations) that the concave arm changes both shape and magnitude.

### 1.3 Avellaneda-Stoikov reference policy. OK, with one code-doc mismatch

`market_making.py:72-80,215-231`: `r = s - q*gamma*sigma^2*tau`, `delta* = gamma*sigma^2*tau/2 + (1/gamma)*ln(1+gamma/kappa)`, `bid_depth = delta* + q*gamma*sigma^2*tau`, `ask_depth = delta* - q*gamma*sigma^2*tau`. This matches A-S (2008): reservation price `r = s - q*gamma*sigma^2*(T-t)` and total spread `gamma*sigma^2*(T-t) + (2/gamma)*ln(1+gamma/k)` quoted symmetrically around `r`. The fill mechanism (Poisson arrivals at rate `A*dt` thinned with prob `exp(-kappa*delta)`) is an exact thinning realization of the A-S intensity `A*exp(-k*delta)`. The paper's framing (03-environment.tex:29) is accurate and properly hedged: it calls the closed form "the source model's asymptotic approximation, not an exact optimum", cites Gueant et al. for the exact treatment, and notes the implemented reward adds an inventory cap, running penalty `phi*q^2` and terminal liquidation the source model lacks. Note: the current text does not contain the phrase "approximate HJB solution" anywhere; the "asymptotic approximation" wording is the correct characterization of A-S 2008.

- MINOR (M-02). The module docstrings still oversell what the paper renamed: `market_making.py:7-8` says "analytically optimal quoting policy... a provable optimum", and `analytically_optimal_policy`'s docstring says "the ground-truth optimum". The paper (r1 revision) correctly downgraded this to "closed-form reference policy"; the code documentation was not updated. Fix: align the docstrings with the paper's language (the function name can stay for API stability with a note).

### 1.4 Regret definition and the paired-episode claim. OK (verified nontrivially)

`mm_regret` (`market_making.py:257-278`): regret = mean over episodes of (reference reward minus candidate reward), both rolled on the same seed; reference regret is exactly 0 by construction (same rollouts subtracted). The subtle risk audited: both policies share one `np.random.default_rng(seed)`, and numpy's `binomial` can consume variable RNG state, which would make "same seed" not imply "same mid path" and void the pairing. Tested directly: per step the env draws poisson, poisson, binomial, binomial, normal; numpy `Generator.binomial(n, p)` consumes fixed state for fixed `n` regardless of `p` (verified for p in {0.1, 0.5, 0.9, 1.0}; only `n = 0` differs, and `n` comes from the policy-independent poisson draws; `p = 0` cannot occur since `exp(-1.5*5) > 0`). A full replica of the env confirmed byte-identical mid paths across the A-S policy and fixed spreads 0.5 and 4.0 on seeds 0-3. The paired t-CI in F2 is therefore a genuine common-random-numbers pairing.

### 1.5 Stylized-facts statistics as implemented. Mixed

`realism.py`. Excess kurtosis: `mean((x-m)^4)/s^4 - 3` with population std; standard. `abs_return_autocorr`: mean of lag-1..10 autocorrelations of `|r|` with full-sample denominator; standard estimator. `zumbach_asymmetry`: coarse = trailing 5-bar MA of `|r|`, lag = window so the two measures never share a return; the claim is correct (corr(coarse_t, fine_{t+5}) uses fine at t-4..t vs t+5). `aggregational_gaussianity`: `kurt(r) - kurt(sum over 4-bar blocks)`. `fano_factor`: exceedance counts (threshold mean + 2 std of `|r|`, estimated in-panel) per 10-bar window, variance with `ddof=1` over the 12 windows, as the paper says (05-experiments.tex:106). Gate = conjunction of three sign checks, applied per seed panel; the "2 of 24" arithmetic is correct per seed (see 3.4).

Two validity problems with the gated statistics themselves:

- MAJOR (M-03). The |r|-autocorrelation gate at threshold 0 has a negatively biased finite-sample null, so "fails the gate" is largely a property of the test, not the tape. The sample ACF of an iid series has expectation about -1/T per lag; simulation of the exact pipeline (T=120 returns, 4 symbols, panel-averaged, lags 1..10) gives mean -0.0079, sd 0.0128 for iid Gaussian, and P(statistic > 0) = 0.25. Hard's per-seed values (mean -0.0107) and Extreme's (-0.0118) sit within one sd of the null mean, and the observed 1/8 pass rates are consistent with a no-clustering null (expected about 2/8). Moreover an iid Student-t(3) panel (fat tails, zero clustering) passes the full three-check conjunction about 23% of the time, i.e. the conjunction cannot separate the Hard tape from fat-tailed iid noise; the only discriminating leg is the biased ACF. Consequences: (a) calling the three gated facts "calibrated directional facts" (03-environment.tex:41; 05-experiments.tex:5,110) is unjustified; they are uncalibrated sign checks exactly like the Fano ratio the paper excludes for that very reason; (b) "absolute-return autocorrelation is slightly negative at this episode length" (05-experiments.tex:127) should say the values are consistent with zero clustering plus the -1/T small-sample bias. Concrete fix: bias-correct the ACF (add (T-k)/(T(T-1)) per lag or use the circular permutation null), or gate at the null's 95th percentile estimated by permuting each series in time (destroys clustering, preserves marginals); report the resulting null pass rate next to the 2/24. The headline "generator fails its own gate" survives (the clustered variant demonstrates the gate can be passed), but its strength as stated does not.

- MAJOR (M-04). The aggregational-Gaussianity statistic has the wrong sign convention for platykurtic tape, so Calm's 0/8 on that fact is a definitional artifact. The statistic is kurt(h=1) - kurt(h=4) > 0, which encodes "kurtosis decays toward Gaussian from above". Calm's innovations are uniform (`sharpebench-sim/src/data.rs:220`: `shock = (next()-0.5)*0.02`; excess kurtosis of a uniform is -1.2), so Calm's returns are platykurtic by construction (observed -0.97) and aggregation moves kurtosis up toward 0, which is Gaussian convergence, yet the statistic is negative (-0.58) and fails 8/8. Simulation: iid uniform panels fail the check 100% of the time while literally obeying the CLT; even iid Gaussian passes only 68% (pure noise around 0). Fix: define the fact as |kurt(1)| - |kurt(h)| > 0 or |kurt(h)| < |kurt(1)|, and separately note that Calm's platykurtosis is a design consequence of uniform innovations (the code comment "Gaussian-ish shock" at data.rs:185 is wrong; uniform is not Gaussian-ish in the fourth moment). Calm still fails the kurtosis check either way, so the 22/24 headline moves at most to a differently composed failure, but the current table (05-experiments.tex:127) misattributes the Calm failure.

- MINOR (M-05). Zumbach coarse series uses `np.convolve(..., 'full')[:len]`, so the first window-1 entries are partial sums over an implicit zero pad; a small edge bias on a 120-bar series. Drop the first window-1 points. Informational-only statistic, so no conclusion is affected.

### 1.6 Sealed-seed derivation. OK, one off-by-one in the displayed formula

`scenario_gen.rs:524-540`: FNV-1a/64 absorbs salt bytes plus length, three chained SplitMix64 finalizer rounds with the salt digest re-mixed, then `EVAL_SEED_BASE + (s % span)` with `span = u64::MAX - EVAL_SEED_BASE`. Band membership: `s % span` is in `[0, span-1]`, so seeds lie in `[10^6, 2^64 - 2]`; always at or above `EVAL_SEED_BASE`, never overflowing. Disjointness from the train band `[0, 10^6)` holds by construction, salt or no salt, exactly as claimed. The paper's honesty about PRF-style-not-MAC and no injectivity claim matches the code comment.

- MINOR (M-06). sealed-seeds-fragment.tex:6 displays the modulus as `2^64 - EVAL_SEED_BASE`, but the code reduces modulo `u64::MAX - EVAL_SEED_BASE = 2^64 - 1 - EVAL_SEED_BASE`. Off by one; harmless (bias ~5e-20) but the displayed derivation does not reproduce the committed seeds. Fix the displayed formula to `2^64 - 1 - EVAL_SEED_BASE`.

### 1.7 Deflated-Sharpe CI (`leaderboard_ci.rs`) vs the kernel units fix. OK

The estimator chain matches Bailey and Lopez de Prado: PSR with `z = (SR - SR*) * sqrt(n-1) / sqrt(1 - g3*SR + (g4-1)/4 * SR^2)` (g4 raw kurtosis, correct form); `E[max SR] = sr_std * ((1-gamma)*Z^-1(1-1/N) + gamma*Z^-1(1-1/(Ne)))` with Euler-Mascheroni gamma, correct. The units fix is implemented as documented: the annualized prior `TRIALS_SR_STD_DEFAULT = 0.5` is divided by `sqrt(252)` exactly once, in each public entry point (`deflated_sharpe` line 213, `bootstrap_dsr_ci` line 327, `paired_dsr_diff` line 391); the internal `deflated_sharpe_per_period` never converts, so the bootstrap loop cannot double-convert. The kernel base footprint is folded in the Python binding (`sharpearena-py/src/lib.rs:556,584`: `effective = KERNEL_BASE_TRIALS(50) + declared`), and the strongest end-to-end check passes: in `f1-baselines.json` all 18 rows have `deflated_sharpe_ci.point == score_run deflated_sharpe` to < 1e-9, so the CI brackets exactly the leaderboard statistic. Consistency spot-check of the pre-fix bug magnitude: an unconverted 0.5 annualized prior with N = 56 gives `E[max SR] = 0.5*(0.423*2.10 + 0.577*2.48) = 1.16` per period = 18.4 annualized; the paper's "near an annualized Sharpe of 18" (05-experiments.tex:11) is right.

Erf is Abramowitz-Stegun 7.1.26 (abs err ~1.5e-7) and Acklam's inverse normal; both adequate for the quantiles used. Percentile quantile uses linear interpolation of order statistics; standard.

---

## 2. Recomputations (paper value vs recomputed)

### 2.1 Bonferroni critical value. OK

`t_{1-.05/(2*135), 7}`: recomputed `scipy.stats.t.ppf(0.9998148148, 7) = 6.391202695754376`. Identical to the constant in `make-f5-manipulation.py:88` and to the value stored in every `familywise_inference` block. The family size 135 = 3 exponents x 5 legs x 3 splits x 3 arms is correct, and the per-cell df = 7 (8 seeds) is correct.

### 2.2 F2 t-CIs from per-episode vectors. OK

Recomputed from `f2-regret.json/per_episode_regret` with both the script's t = 2.131 and exact 2.1314 (difference < 0.004 in the bounds):

| half-spread | paper | recomputed |
|---|---|---|
| 0.05 | 84.6 [68.2, 101.1] | 84.636 [68.214, 101.057] |
| 0.5 | 0.037 [-8.7, 8.8] | 0.037 [-8.730, 8.804] |
| 1.0 | [-5.4, 12.4] | 3.475 [-5.447, 12.396] |
| 2.0 | 36.6 [33.0, 40.2] | 36.633 [33.025, 40.241] |
| 4.0 | 58.8 [55.3, 62.3] | 58.804 [55.342, 62.267] |

All match. Wilcoxon signed-rank agrees with every t verdict (p < 1e-4 where t rejects; p = 0.86 at 0.5), so the mild non-normality (Shapiro p = 0.014-0.034 at three grid points) does not change any conclusion.

- MINOR (M-07). "Sixteen episodes do not resolve either point" would be stronger as a sensitivity statement: at hs = 0.5 the per-episode sd is 16.45, so the 80%-power minimum detectable regret at n=16 is about 12.3 reward units (about 87 episodes would be needed to resolve a regret of 5). One sentence in F2 would quantify what "not resolved" means.

### 2.3 F5 cells. OK (all three requested cells and more)

Linear table (tab:f5) vs recomputation from `dispersion` per-seed vectors (units 1e-4, t = 2.365, df 7): Kyle lambda high end -31.75 [-44.32, -19.18] (paper -31.7 [-44.3, -19.2]); eta high end -22.44 [-23.49, -21.38] (paper -22.4 [-23.5, -21.4]); follower high end -6.71 [-10.14, -3.29] (paper -6.7 [-10.1, -3.3]); center lambda=0.1 -3.44 [-4.27, -2.62] (paper -3.4 [-4.3, -2.6]); full size-response column matches to rounding. Every per-point CI excludes zero on the linear grid; "no boundary" and "unprofitable everywhere" flags verified in the JSON.

Concave: kappa=0.5 base -8.91 [-19.58, +1.76] (paper -8.9 [-19.6, 1.8]); kappa=0.7 base -13.53 [-21.18, -5.87] (paper -13.5 [-21.2, -5.9]). Crossing counts recomputed: kappa=0.5 exactly 7 of 23 stored slots cross zero, and the 7 collapse to 5 distinct configurations (eta=0, eta=0.1, push=0.75, push=1.0, and the canonical point stored three times under kyle_lambda=0.1 / eta=0.05 / follower_gain=30); kappa=0.7: 0 of 23. Matches concave-fragment.tex:10 exactly. No positive mean anywhere on the symmetric grid; `profitable_anywhere` false for both exponents.

Positive control: 135 cells confirmed. Best cell = theory arm, kappa=0.5, 9:1 uniform: mean +21.81e-4, pointwise [17.4, 26.2]e-4, familywise [9.99, 33.62]e-4 (paper: +21.8, [17.4, 26.2], [10.0, 33.6]). Temporary-impact twin: +18.37e-4, familywise [6.4, 30.4]e-4; matches. Exactly 2 of 135 familywise intervals stay positive; 45 linear cells all have negative means (max -1.29e-4); kappa=0.7 has no positive mean (max -4.28e-4); the canonical follower arm has no pointwise-positive cell and in fact no positive mean at all. The 1:9 vs 9:1 theory-arm linear values are -3.5790e-4 vs -3.5799e-4: "numerically near-symmetric, not bit-identical", exactly as written.

- MAJOR (M-08). The familywise positivity of the two selected cells rests entirely on the t/normality assumption at n = 8. A distribution-free check cannot reach the Bonferroni level at this sample size: the best possible two-sided sign-test p at n=8 is 2^-7 = 0.0078, which is 21x larger than the Bonferroni alpha 0.05/135 = 0.00037. So if a reader does not grant normality of per-seed impact PnL, the selection-adjusted claim is unverifiable from 8 seeds in principle. Mitigation observed in the data: both selected cells are 8/8 positive per seed (theory: 14.7 to 31.6, temporary: 11.3 to 28.2, all e-4), Shapiro p = 0.64/0.57, and the effect is about 12 within-cell sds from zero, so the claim is very probably right. Concrete fix (cheap and decisive): rerun only the two selected schedules on fresh seeds 8..39 as a confirmatory test with no selection and no Bonferroni; 32 fresh seeds make even the sign test conclusive (p ~ 4.7e-10). State in the fragment that the familywise interval is normality-dependent at n=8 until that replication exists.

### 2.4 F3 bootstrap CIs. OK

Construction (make-f3-generalization.py): B = 2000, percentile, seeds resampled with replacement, per-band pooled-DSR CIs, and seed-paired resampling for the transfer cells (identical index vector applied to both regimes); RNG streams `default_rng((0, job_id))`, serialized. The script asserts the bootstrap point equals the public API's pooled point to 1e-9 for every band and cell (lines 176-180). All table values match the JSON: gaps +0.0012 / -0.3388 / -0.1012; matrix +-0.877, +-0.973, +-0.096; CIs [0.635, 0.999] (Calm-Extreme), [0.054, 0.999] (Calm-Hard), [-0.218, 0.911] (Hard-Extreme); Hard band CIs [0.000, 0.944] and [0.037, 0.948]. Antisymmetry and zero diagonal hold by construction, as stated.

### 2.5 Seed-band disjointness arithmetic. OK, with a scope inconsistency (see 5.2)

`train_test_seeds(n_train, n_test, 0, 10000)` (`generalization.py:27-44`): train `[0, n_train)`, test `[n_train + 10000, n_train + 10000 + n_test)`, disjointness asserted. For the canonical statement train `[0,256)`, gap 10000, test `[10256, 10512)`: consistent and disjoint; also documented in EVALUATION.md:41. `EVAL_SEED_BASE = 10^6` (`scenario_gen.rs:480`) with all eight `EVAL_SEEDS` offsets in `[0, 4100)` above it; the module asserts every seed >= base and uniqueness (`eval_seeds.py:84-95`) and cross-checks the native constant. All membership arithmetic verified.

### 2.6 Scan timing extrapolation. OK

`predictability.json`: total 15.389 s over 16 scans of 65536 candidates = 14.6757 us per candidate (paper 14.7), 0.962 s per scan (paper ~0.96). `2^64 * 14.6757e-6 s / (86400*365.25) = 8.5786e6` CPU-years; the paper's "~8.6 million CPU-years" matches the stored 8578588.9.

- MINOR (M-09). The per-candidate cost is that of constructing a 4-symbol 2-day Python `TradingEnv` per candidate. A native scan of the same fingerprint (a handful of SplitMix64 draws and multiplies) would be 10^2 to 10^4 times cheaper, so the 8.6M figure is a property of this harness, not of the transform. The conclusion (enumeration of 2^64 infeasible) survives a 10^4 speedup (still ~860 CPU-years per inversion times cores), but the fragment should label the number "with this Python harness; a native scan is orders of magnitude cheaper and does not change the conclusion".

### 2.7 The 2^11 candidate-state claim. OK

`next_unit` (`scenario_gen.rs:123-129` and data.rs:206-213) returns `(z >> 11) / 2^53`: exactly the top 53 bits of the finalizer output as a float with no additional rounding (53-bit values are exactly representable), discarding 11 bits. The finalizer `mix64` is a bijection (each step is an invertible xor-shift or odd multiply), so one exact unit observation leaves precisely `2^11 = 2048` candidate post-increment states. Claim verified. (Unstated but true: two consecutive exact units would cut the 2^22 candidate pairs to ~1 via the fixed golden-ratio increment; the paper correctly notes exact units are never exposed.)

### 2.8 Sealed-seed run arithmetic. OK

Coverage `65536 / (2^64 - 10^6) = 3.5527e-15` (paper "about 3.6e-15"); expected recoveries at this budget 5.7e-14 (stored). Public 16/16 recovered and prefix-verified; sealed 0/16; reveal replays 16/16; salt commitment SHA-256 matches the revealed salt. All 16 public seeds lie in the scanned band; no sealed seed does.

- MINOR (M-10). Given the coverage number, 0/16 is arithmetically guaranteed rather than an experimental outcome; the probability of even one hit was 5.7e-14. The fragment already calls the result "narrow", but it would be more honest to say the sealed leg is a coverage identity (the informative content is that the derivation left the band, plus the commit-reveal replay), not a measured defeat.

---

## 3. Evidence consistency (three or more cells per table)

### 3.1 Table 1 (F1). OK

All 18 DSR and pass^k cells match `f1-baselines.json` (kelly 1.0000/0.75, 0.0277/0.31(25), 0.0029/0.125 shown as 0.12; equal-weight 1.0000/0.625, 0.1114/0.3125, 0.0234/0.0625; min-var 0.9849/0.4375, 0.0024/0.1875, 0.0243/0.0625; flat 0.0007/0; momentum row includes the 0.0625 Extreme oddity, present in JSON as pass rate 0.0625 with DSR 1e-5). Text CIs [0.9798,1.0000], [0.8343,1.0000], [0.0264,1.0000] match. "Mean per-bar returns down to -1.6e-3 on Extreme": max_sharpe Extreme mean return -1.578e-3. Matches.

### 3.2 Table F4. OK as arithmetic (see M-03/M-04 for validity)

Means and pass counts match the JSON exactly: Calm -0.97 kurt 0/8, +0.0002 acf 5/8, Fano 1.30, rate 0.000; Hard +16.30 8/8, -0.0107 1/8, 0.92, 0.125; Extreme +11.38 8/8, -0.0118 1/8, 0.86, 0.125. Per-seed conjunction re-derived from the stored per-seed checks: 0+1+1 = 2 of 24; the gate is applied per seed, and `passed` equals the conjunction of exactly the three gated checks for every one of the 48 stored reports (canonical and clustered). Clustered variant: acf passes 8/8, 8/8, 5/8 and three-check rates 0.0, 1.0, 0.625, matching "0/8, 8/8 and 5/8". Aggregational Gaussianity Calm -0.577 (paper -0.58).

### 3.3 Table F5 / concave / positive control. OK

See 2.3; every quoted cell reproduced from per-seed vectors, including both selected familywise intervals, the crossing counts, and the near-symmetry statement.

### 3.4 Table F6. OK, one estimator-mixing note

Levels and gaps match `comparison` (pooled, quantity-weighted): 0.689/0.489/0.386, 0.761/0.791/0.823, gaps 0.072/0.302/0.436. CIs match `gap_stats` (per-episode, unweighted): [0.052, 0.094], [0.228, 0.383], [0.303, 0.579]; recomputed identically with t(23) = 2.069. Detail episode: maker_0 135 fills, 0.526 per unit at h=20, toxic 0.296; maker_1 92 fills, aggregate 0.197, meta -0.197 on 62% of quantity, toxic 0.533; spread capture 82.8, adverse drift -64.7, identity 82.8 - 64.7 = 18.1 = markout at h=20 holds exactly. Paired design verified in code: the price path lives on dedicated streams (`rng_price`, and `rng_meta` draws alphas in both legs), so the two legs share the mid path exactly.

- MINOR (M-11). The table's point column is the pooled quantity-weighted gap (0.072) while its CI column is built on the per-episode unweighted gap (mean 0.073); at h=20 the two are 0.436 vs 0.441. Both are stored and the pooled point lies inside every CI, and the producer's docstring discloses the convention, but the caption should say the CI is centered on the per-episode estimator, or the per-episode mean should be printed beside the pooled one. Also, the level claims ("makers earn a positive mean markout... 0.689, 0.489, 0.386") carry no interval; the per-episode level CIs are tight (informed h=20: [0.34, 0.43]) and worth printing since the level finding is called out as a finding against the environment.

### 3.5 Table F7. OK

All 21 count cells match `rollup_by_tier` exactly (72/0/36/11/9, 68/0/36/14/10, 50/5/31/32/10; clean rates 0.5625, 0.53125, 0.390625). Episode-level claims verified against the 384 stored episodes: stopped_out = 5, all on Extreme, 4 of them max_sharpe (the fifth is disposition_effect); overconfident on Calm: 7 of 16 mandate_drawdown; max_sharpe on Extreme: 3 clean of 16; flat clean 48/48.

### 3.6 Table F8. OK

Winner distributions match: control kelly 4, equal-weight 2, flat-no-dominant 2; shocked kelly 7, equal-weight 1; root winner differs on 5 of 8; seed-0 handoff occurs once. Seed-0 details: control kelly peak share 1.00, final 0.81; shocked equal-weight final 0.672; shocked extinctions per run 4-7 (4,6,6,6,7,5,6,7); the two no-dominant control seeds end with a flat variant at share 0.10 and all eleven persistent. All match the text.

- MINOR (M-12). Two counting-level quibbles. (a) "The root winner differs between the two schedules on 5 of 8 seeds, so shocks do reorder outcomes more often than not" (05-experiments.tex:282): 2 of those 5 are the seeds where the control produced no resolved dominant at all (largest share 0.10), so only 3 of the 6 resolved pairs are genuine winner replacements, and 5/8 vs 4/8 is far inside binomial noise at n=8 (P(>=5 | p=0.5) = 0.36); "more often than not" is not supported. Say "on 5 of 8 seeds the root label differs (2 of these because the control resolved no dominant); at n=8 this is not distinguishable from no effect". (b) "overconfident by generation 1": the stored extinction events are at generation 0 on both schedules; if generations are zero-indexed the text should say generation 0 (or "in the first generation").

### 3.7 Witness table. OK

All 12 cells of tab:witness match `witness.json` boundaries: sign_follow held-out none / [0.7250, 0.7281] / [0.3000, 0.3031]; f1 band none / [0.7875, 0.7906] / [0.4438, 0.4469]; deadband held-out [0.9000, 0.9031] / [0.5031, 0.5063] / [0.3813, 0.3844]; f1 band [0.6563, 0.6594] / [0.6188, 0.6219] / [0.3688, 0.3719]. (Stored values 0.90312, 0.50313-0.50625 etc. round to the printed 4-decimal brackets.) Bracket widths ~0.003 are consistent with bisection from a 0.05 coarse step to the 0.005 resolution.

### 3.8 Predictability fragment. OK

55.4 +- 2.8 vs 54.1 +- 2.4 (stored 0.5540/0.0277, 0.5406/0.0243); Hard 51.2 +- 3.3, Extreme 50.6 +- 2.1; MSE Hard 1.242e-4 vs 1.217e-4 (paper 1.24 vs 1.22), Extreme 1.273e-3 vs 1.167e-3 (paper 1.27 vs 1.17); oracle 100%/0 MSE/DSR 1.00 on all tiers; 16/16 recovered, zero collisions, all 16 prefix-verified per tier. "Centered on the published start_level": band = [50000 - 2^15, 50000 + 2^15), confirmed. The tier-invariance premise of the first-bar fingerprint is true in code: `amplify` and `burst_jumps` rewrite `series[1..]` only, `vol_cluster` likewise (scenario_gen.rs:147-236), so bar 0 is tier-invariant; also confirmed empirically by the 48 successful prefix verifications.

---

## 4. Statistical validity

- MAJOR (M-13). Witness thresholds carry no Monte Carlo uncertainty over the injected noise path. Each (scenario seed, variant) uses exactly one epsilon realization (`_noise_seed`, make-witness.py:123-127), reused across strengths (good: CRN in s), but the threshold itself is a functional of that one epsilon draw. The brackets in tab:witness are bisection resolution, not statistical intervals, and the caption says so, but reporting four decimals invites over-reading, and the cross-band differences discussed in the text (held-out 0.725 vs f1 0.7875 on Hard) are differences between single noise realizations. The structural conclusions are robust (the acceptance set is non-empty; the every-seed PSR gate binds last at all 10 thresholds, verified: at every lo-row the sole failing gate is per_run_gate with min seed PSR 0.863-0.899, all other gates passing). Fix: rerun with 3-5 noise seeds and report the threshold range, or round the table to two decimals and state that the bracket is numerical resolution under one fixed noise path.

- MAJOR (M-14). The "0.34 seed-noise calibration" (tab:f3 caption, 05-experiments.tex:69,93,95) is a single realized gap (Hard within-tier, -0.3388) promoted to a decision threshold for classifying transfer cells. One draw has no sampling distribution attached; the Extreme gap in the same table is -0.10 and the Calm gap +0.001, so "how much band luck a 16-seed evaluation carries" varies by a factor of 300 across the three available draws. The bootstrap CIs already do this job correctly (the Hard-Extreme cell straddles zero; the Calm cells do not). Fix: drop the 0.34 threshold and classify cells solely by whether the seed-paired CI excludes zero; or, if a noise scale is wanted, bootstrap the within-tier gap's distribution and quote its 95th percentile instead of the point draw.

- MINOR (M-15). Cross-table deflation inconsistency: F3 (and its band CIs) score with `n_trials = 0` (kernel base 50 only), while F1 declares `n_trials = 6`. Same policy, same seeds, same tier: Hard equal-weight is 0.1231 in Table 2 and 0.1114 in Table 1 with no explanation anywhere in the paper. Not an error (both conventions are internally consistent and serialized), but a reader who compares adjacent tables will conclude one of them is wrong. Add a footnote, or score F3 with the F1 footprint.

- MINOR (M-16). Bounded-statistic bootstrap: the pooled DSR is in [0,1] and saturates (Calm points sit at 1.0000), so percentile bootstrap intervals like [0.9251, 1.0000] and [0.000, 0.944] are asymmetric truncations with known undercoverage near the bounds at n=16. The paper already flags coarseness; naming the mechanism (bounded, strongly nonlinear functional; percentile CIs near the boundary undercover) and considering BCa would be better. No conclusion changes.

- MINOR (M-17). Pass^k rates and monotonicity claims at n=16 have binomial noise of about +-0.24 at rate 0.5 (95%); "degrades monotonically with stress (0.62, 0.31, 0.06)" (05-experiments.tex:35) and the F7 clean-rate ordering are point comparisons without intervals. The orderings are plausible and consistent, but one clause noting the +-0.2 binomial width at n=16 would keep the table honest.

- MINOR (M-18). Predictability sd convention: the stored accuracy_std is ddof=0 (0.0277) not ddof=1 (0.0286); the fragment only calls them "across-seed standard deviations", so no error, but state the convention. Conversely the paper undersells one result: the Calm honest-vs-baseline accuracy difference is significant under a paired t across the 16 shared seeds (t = 3.36, p = 0.004), which supports "detects the tier's designed momentum autocorrelation" more strongly than the prose implies; the momentum design is real (data.rs:221: `momentum = 0.9*momentum + 0.1*shock` feeding returns).

- F8 replication statement: correctly reported and self-correcting (single-seed narrative retracted; 1/8 replacement stated in abstract, table and text; counts verified). The residual counting quibble is M-12. The statement "shock-driven winner replacement occurs in one seed, not seven" is fine given the seed-0 narrative it retracts, though "not seven" reads ambiguously against the 7/8 kelly-dominance count in the same table; consider "in one seed of eight".

- Witness monotonicity/bisection logic: correct as implemented. The producer checks monotonicity on the coarse grid before bisecting (`locate_boundary`, make-witness.py:224-281: any eligible-then-ineligible adjacent pair aborts threshold identification and reports the observed set), refines only a monotone first crossing, and every stored bisection sequence is internally consistent (no bisection point contradicts the bracket ordering; verified for all 10). Caveat: bisection cannot detect non-monotonicity strictly inside the bracket, and the coarse check cannot either; with CRN and threshold-crossing gates this is a reasonable assumption, and the observed data never contradicts it.

- Realism-gate conjunction per seed: correctly applied per seed (section 3.2 above); the "2 of 24" is a per-seed conjunction, not a conjunction of per-fact rates. The validity issue is M-03/M-04, not the arithmetic.

- Comparisons of point estimates without uncertainty: the remaining instances are the F6 levels (M-11), pass^k orderings (M-17), F8 5-of-8 (M-12), and the F4 clustered-vs-canonical pass-count deltas (5/8 to 8/8 etc.), which are in-sample by admission and not tested; each is either labeled or minor.

---

## 5. Logic and scope

### 5.1 Do the conclusions follow?

- F1: "eligibility is a conjunction, each leg catches what the other misses" is supported by the board (deflation saturates on Calm drift; pass^k fails everything) and the witness closes the non-emptiness gap it names. Sound.
- F2: U-shape resolved, minimum unresolved, explicitly not equivalence. Sound (add MDE, M-07).
- F3: instrument framing (not transfer learning) is correct and stated; the antisymmetry/3-number reduction is stated. The 0.34 device is the weak link (M-14).
- F4: the arithmetic conclusion follows, but the strength of "fails its own gate on 22 of 24" leans on an uncalibrated, biased gate (M-03) and a sign-inverted fact for Calm (M-04). The vol_clustering re-certification is openly labeled in-sample on the diagnosis seeds (05-experiments.tex:129), and no remediation claim is made for the un-run knobs; that circularity is disclosed, not hidden.
- F5: the linear null is correctly framed as model consistency (Huberman-Stanzl / Gatheral), the concave arm as the falsifiability channel, and the positive control as exploratory with selection-adjusted inference. The chain "symmetric null is a measurement, not a corollary" is right. Residual: normality dependence at n=8 (M-08) and shape-vs-scale confound (M-01).
- F6: gap vs level separation is exactly the right decomposition, and the level finding is reported against the environment. Sound (M-11).
- F7: the mandate-assignment mechanism is stated before the counts and the headline is scoped to the mandate-blind field. Sound.
- F8: replication overturning the single-seed story is reported plainly; residual counting nuance M-12.
- Predictability/sealed/witness: conclusions follow from the measurements; the open cryptanalytic step is stated rather than claimed closed.

### 5.2 Scope changes and inconsistencies

- MAJOR (M-19). Canonical-band statement vs bands actually used. Section 3.4 (03-environment.tex:21) fixes "the canonical bands: train seeds [0, 256), a gap of ten thousand seeds that is never sampled, and a held-out test band [10256, 10512)". No experiment uses these bands: F3 and the witness use `train_test_seeds(16, 16, 0, 10000)`, i.e. train [0,16) and "held-out" [10016, 10032), which lies inside the canonical protocol's never-sampled gap [256, 10256); the reserved 10^6 namespace is used only by the sealed-seed follow-up. The Section 5 preamble discloses that core evidence uses train-band seeds "except F3, which exercises the split", but never says the F3/witness held-out band is not the canonical held-out band, and the witness fragment and figure caption call it simply "the held-out band". Disjointness is unaffected ([0,16) vs [10016,10032) is disjoint), so no result is invalidated, but the protocol text and the evidence describe different bands. Fix: one sentence in 3.4 or the F3 preamble: "the evidence run instantiates the split at n=16 (train [0,16), test [10016,10032)); the canonical 256-seed bands are the protocol default, not the bands behind these tables", and rename "held-out band" to "held-out band (n=16 instantiation)" in the witness fragment.

- The corrections trail itself is clean: the claimed regenerations (F4, F5, witness) are consistent with file mtimes and content; the concave counts, familywise constants and CRN/monotonicity mechanics in the final integrity report were independently re-verified here and hold. The "linear and concave keys byte-identical to their previous values" claim (A-commands.tex:54) was not re-verifiable from a single snapshot (no prior artifact in the tree) and is taken on the manifest's word; flag for the provenance file rather than this audit.

- No silent scope changes were found in the correction from Fano-gated to three-fact-gated certification: the abstract, 03-environment, F4 and limitations all consistently describe the three-check conjunction and the ungated Fano, and the stored `RealismReport.checks` contain exactly the three gated facts.

---

## 6. Verified-OK register (compact)

1. Bonferroni t(7) quantile at 1-.05/270 = 6.391202695754376, exact.
2. F2 CIs (7 grid points) reproduce; Wilcoxon concurs; pairing is real CRN (RNG-consumption analysis + env replica).
3. A-S reservation/half-spread formulas match A-S 2008; thinning matches the exponential intensity; skewed depths correct.
4. Kyle/AC clearing as documented; concave exponent on Q/V only; kappa=1 bit-identical; slicing identity correct.
5. Regret zero-point exact by construction.
6. F3: all table cells and 4 CIs reproduce; seed-paired transfer bootstrap as described; diagonal exactly 0; antisymmetry exact.
7. F4: all means/counts reproduce; conjunction per seed correct; 2/24 and clustered 0-8/8-8/5-8 correct; Calm agg -0.577.
8. F5: all linear/concave/size cells and CIs reproduce; 7/23 and 0/23 crossings, 5 distinct configs; 135 cells; 2 familywise-positive cells; 45 linear means all negative; canonical arm none positive; 1:9 vs 9:1 near-symmetry.
9. F6: pooled and per-episode values reproduce; markout identity exact; paired price path verified in code.
10. F7: all 21 counts + 4 per-policy claims verified from 384 episodes.
11. F8: winner distributions, 5/8, 1/8, extinction ranges, seed-0 shares verified.
12. Witness: all 12 brackets, binding-gate claim (min PSR < 0.90 sole failure at every lo row), 10/10 coarse monotonicity, CRN reuse verified.
13. Predictability: accuracies, MSEs, 16/16, zero collisions, per-candidate 14.68 us, 0.962 s/scan, 8.58e6 years.
14. 2^11 = 2^(64-53) candidate states; finalizer bijective; floats exact for 53-bit integers.
15. Sealed: band membership arithmetic, coverage 3.55e-15, commit-reveal replay, disjointness-without-salt.
16. DSR units fix: annualized prior divided by sqrt(252) exactly once per public entry point; bootstrap loops use the per-period core; base-50 footprint folded in the binding; ci.point == score_run point on all 18 F1 rows.
17. Pre-fix bug magnitude "annualized ~18" reproduces (1.16 per period at N=56).
18. Seed bands: train/test and eval-namespace disjointness assertions all correct.
19. First-bar tier invariance (amplify/vol_cluster/burst_jumps rewrite bars 1..) confirmed in code and by 48 prefix verifications.
20. PSR/DSR estimator algebra matches Bailey and Lopez de Prado (raw-kurtosis form).
21. "2 of 24" language consistent across abstract, preamble, F4 and limitations.

---

## 7. Findings index

| ID | Severity | Location | Issue | Fix |
|---|---|---|---|---|
| M-03 | MAJOR | realism.py:90-101; 03-environment.tex:41; 05-experiments.tex:5,106,110,127 | The |r|-ACF gate at 0 has a biased, uncalibrated null (iid passes 25%; iid t3 passes the full conjunction ~23%); "calibrated directional facts" unjustified; Hard/Extreme failures consistent with -1/T bias | Bias-correct or permutation-null the ACF gate; report null pass rate beside 2/24; drop the word "calibrated" |
| M-04 | MAJOR | realism.py:136-145; 05-experiments.tex:127 | Aggregational-Gaussianity sign convention fails platykurtic tape that is converging to Gaussian (iid uniform fails 100%); Calm's kurt -0.97 is the uniform-innovation design (data.rs:220) | Redefine as abs-kurtosis shrinkage; attribute Calm's platykurtosis to uniform innovations; fix "Gaussian-ish" comment |
| M-08 | MAJOR | positive-control-fragment.tex:9-11; make-f5-manipulation.py:86-130 | Familywise positivity of the 2 selected cells rests on t/normality at n=8; no distribution-free test can reach alpha/135 at n=8 | Confirmatory rerun of the two selected schedules on 32 fresh seeds (no selection, no Bonferroni); meanwhile state the normality dependence |
| M-13 | MAJOR | arena-witness-fragment.tex:12-27; make-witness.py:123-127 | Thresholds conditional on one noise path per (seed,variant); brackets are numerical resolution, printed to 4 decimals | Replicate over 3-5 noise seeds and report ranges, or round and label single-path |
| M-14 | MAJOR | 05-experiments.tex:69,93,95 | "0.34 seed-noise calibration" is one realized gap used as a decision threshold | Classify cells by CI-excludes-zero only, or bootstrap the noise scale |
| M-19 | MAJOR | 03-environment.tex:21 vs make-f3/make-witness (test band [10016,10032)) | Canonical bands [0,256)/[10256,10512) never used; F3/witness "held-out band" sits inside the canonical gap; not stated | One sentence naming the n=16 instantiation and its bands |
| M-01 | MINOR | concave-fragment.tex; 07-limitations.tex:13 | Concave arm confounds shape with ~30x scale at |Q/V|~1e-3 (acknowledged only in limitations) | Scale-matched lambda arm or in-section statement |
| M-02 | MINOR | market_making.py:7,215-222 | Code docstrings still say "analytically optimal / provable optimum" | Align with "closed-form reference policy" |
| M-05 | MINOR | realism.py:118 | Zumbach coarse MA has zero-padded partial windows at series start | Drop first window-1 points |
| M-06 | MINOR | sealed-seeds-fragment.tex:6 vs scenario_gen.rs:538 | Displayed modulus 2^64-BASE vs code 2^64-1-BASE | Fix displayed formula |
| M-07 | MINOR | 05-experiments.tex:52 | "Not resolved at 16 episodes" lacks a sensitivity number (MDE ~12 reward units) | Add one MDE sentence |
| M-09 | MINOR | predictability-fragment.tex:11 | 8.6M CPU-years is Python-harness-specific; native scan 10^2-10^4 cheaper | Label the harness dependence |
| M-10 | MINOR | sealed-seeds-fragment.tex:12 | 0/16 sealed is a coverage identity (P(hit) 5.7e-14) | Call it a coverage identity plus replay check |
| M-11 | MINOR | tab:f6 + text | Point column pooled, CI column per-episode (0.436 vs 0.441 at h=20); level claims lack CIs | State the convention; print per-episode means or level CIs |
| M-12 | MINOR | 05-experiments.tex:267,282 | 5/8 includes 2 unresolved-control seeds; n=8 cannot support "more often than not"; "generation 1" vs stored generation-0 event | Recount as 3/6 resolved; soften; fix generation index |
| M-15 | MINOR | tab:f1 vs tab:f3 | Hard equal-weight 0.1114 (n_trials=6) vs 0.1231 (n_trials=0) across adjacent tables, unexplained | Footnote or align footprints |
| M-16 | MINOR | F3/F1 band CIs | Percentile bootstrap on a bounded saturating DSR undercovers near 0/1 | Name mechanism; consider BCa |
| M-17 | MINOR | 05-experiments.tex:35; F7 | Pass^k / clean-rate orderings without binomial uncertainty (+-0.2 at n=16) | One clause on binomial width |
| M-18 | MINOR | predictability-fragment.tex:9 | sd convention (ddof=0) unstated; Calm honest edge is actually significant (paired t p=0.004) | State convention; optionally report the paired test |

No CRITICAL findings: every number checked in every table and fragment reproduces from the committed evidence, and the three structural mechanisms audited in depth (paired regret episodes, the DSR units fix and footprint folding, and the sealed-seed band arithmetic) are implemented correctly.
