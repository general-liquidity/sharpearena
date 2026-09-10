//! Statistical-confidence layer for the leaderboard ranking.
//!
//! The benchmark ranks on the deflated Sharpe (which discounts overfit-luck) plus
//! pass^k (per-run reliability), but neither answers the A/B question a leaderboard
//! actually has to defend: *is A's number better than B's beyond seed noise, or did A
//! just draw a kinder held-out band?* This module closes that leg (Advances in
//! Financial Machine Learning, Ch. 19, A/B testing under sampling uncertainty) with
//! two self-contained, deterministic tools:
//!
//! 1. a **seed-paired bootstrap CI** on the deflated Sharpe. The held-out seeds are the
//!    independent sampling units; resampling them with replacement and recomputing the
//!    deflated Sharpe on each resample gives a percentile interval that says how much of
//!    the headline number is stable versus a lucky seed draw.
//! 2. a **paired-difference significance test** across the *shared* held-out seed band.
//!    For every bootstrap draw the same resampled seed indices feed both entries, so the
//!    shared-path covariance is retained. Pairing does not remove all luck or isolate
//!    skill. A difference CI containing zero does not establish a difference, nor prove
//!    equivalence. Multiple pairwise comparisons here are not multiplicity-adjusted.
//!
//! The deflated-Sharpe math is the Bailey & López de Prado estimator, ported here
//! self-contained (no `sharpebench-stats` dependency). Its empirical moments use the
//! n-normalized second moment the exact-pinned SharpeBench 0.20.0 scoring kernel also
//! uses; the Python `run_baselines` still attaches an interval to a kernel row only
//! where the kernel reproduces this estimator's point bit-for-bit. The resample RNG is
//! a fixed-seed SplitMix64, so a confidence report replays bit-for-bit from its
//! resample seed.

use serde::{Deserialize, Serialize};

/// **Annualized** cross-trial Sharpe dispersion the deflation assumes, mirroring
/// `sharpebench_core::ScoreConfig::default().trials_sr_std`. It is a free modelling
/// prior, not a value taken from Bailey and López de Prado (2014): their worked example
/// gives 1/2 as the *variance* of annualized Sharpe ratios, a standard deviation of about
/// 0.707, so 0.5 is less demanding than that example by a factor of `sqrt(2)`. Matching
/// configuration alone does not establish estimator parity; the per-row bit-for-bit
/// witness does. Like the kernel, every public entry point here takes this in annualized
/// units and converts it per period exactly once, dividing by `sqrt(periods_per_year)`.
pub const TRIALS_SR_STD_DEFAULT: f64 = 0.5;

/// Bars per year on SharpeArena's daily scenarios, mirroring
/// `sharpebench_core::ScoreConfig::default().periods_per_year`. Nothing here applies it
/// implicitly: every public entry point takes `periods_per_year` from its caller, and the
/// Python binding passes this value by default, as the daily-default `score_run` does.
/// The deflation prior is stated annualized; applied per period unconverted it set the
/// expected-maximum bar to an annualized Sharpe of ~18 on daily bars, which nothing
/// clears. Converting between the two by `sqrt(periods_per_year)` assumes that a Sharpe
/// ratio scales with the square root of the number of periods, which holds only for
/// independent, identically distributed returns.
pub const PERIODS_PER_YEAR: f64 = 252.0;

/// The scoring kernel's own baseline multiple-testing footprint, mirroring
/// `sharpebench_core::ScoreConfig::default().n_trials`. The effective deflation count is
/// this plus the agent's *declared* in-sample trials, so a CI that wants to bracket the
/// leaderboard number must deflate against `KERNEL_BASE_TRIALS + declared`.
pub const KERNEL_BASE_TRIALS: u32 = 50;

// --- self-contained statistics (n-normalized moments; see the module note above) --------

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn std_dev(xs: &[f64]) -> f64 {
    let n = xs.len();
    if n < 2 {
        return 0.0;
    }
    let m = mean(xs);
    let ss: f64 = xs.iter().map(|x| (x - m) * (x - m)).sum();
    (ss / (n as f64 - 1.0)).sqrt()
}

// Empirical standardized moments use the same n-normalized second moment as
// their numerator. Sharpe volatility above deliberately uses n-1 instead.
fn population_std_dev(xs: &[f64], center: f64) -> f64 {
    (xs.iter().map(|x| (x - center).powi(2)).sum::<f64>() / xs.len() as f64).sqrt()
}

fn skewness(xs: &[f64]) -> f64 {
    let n = xs.len();
    if n < 2 {
        return 0.0;
    }
    let m = mean(xs);
    let s = population_std_dev(xs, m);
    if s == 0.0 {
        return 0.0;
    }
    let sum: f64 = xs.iter().map(|x| ((x - m) / s).powi(3)).sum();
    sum / n as f64
}

fn kurtosis(xs: &[f64]) -> f64 {
    let n = xs.len();
    if n < 2 {
        return 3.0;
    }
    let m = mean(xs);
    let s = population_std_dev(xs, m);
    if s == 0.0 {
        return 3.0;
    }
    let sum: f64 = xs.iter().map(|x| ((x - m) / s).powi(4)).sum();
    sum / n as f64
}

/// Error function (Abramowitz & Stegun 7.1.26).
fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t
            * (-x * x).exp();
    sign * y
}

fn norm_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Inverse standard normal CDF (Acklam's rational approximation).
fn norm_ppf(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239e0,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838e0,
        -2.549_732_539_343_734e0,
        4.374_664_141_464_968e0,
        2.938_163_982_698_783e0,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996e0,
        3.754_408_661_907_416e0,
    ];
    const P_LOW: f64 = 0.02425;
    const P_HIGH: f64 = 1.0 - P_LOW;

    if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= P_HIGH {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

/// Per-period Sharpe ratio. 0.0 if volatility is 0.
pub fn sharpe_ratio(returns: &[f64]) -> f64 {
    let s = std_dev(returns);
    if s == 0.0 {
        return 0.0;
    }
    mean(returns) / s
}

/// Probabilistic Sharpe Ratio against `sr_benchmark`, correcting for track length,
/// skewness and kurtosis. In `[0, 1]`. It is one minus the p-value of the one-sided test
/// of `H0: SR <= sr_benchmark`, not the probability that the true Sharpe exceeds the
/// benchmark; López de Prado, Lipton and Zoonekynd (2026) warn against that reading.
fn probabilistic_sharpe_ratio(returns: &[f64], sr_benchmark: f64) -> f64 {
    let n = returns.len();
    if n < 2 {
        return 0.0;
    }
    let sr = sharpe_ratio(returns);
    let g3 = skewness(returns);
    let g4 = kurtosis(returns);
    let denom = (1.0 - g3 * sr + ((g4 - 1.0) / 4.0) * sr * sr)
        .max(1e-12)
        .sqrt();
    let z = (sr - sr_benchmark) * (n as f64 - 1.0).sqrt() / denom;
    norm_cdf(z)
}

const DISPERSION_REFUSAL: &str = "trials_sr_std must be finite and nonnegative";
const TRIALS_REFUSAL: &str = "n_trials must be at least one";
const PERIODS_REFUSAL: &str = "periods_per_year must be finite and positive";

/// Expected maximum Sharpe under `n_trials` independent trials given cross-trial Sharpe
/// dispersion `trials_sr_std` (Bailey & López de Prado, E[max SR_N]).
///
/// A zero dispersion or a single trial leaves nothing to deflate for, so either gives a
/// zero bar. A negative, NaN or infinite dispersion, or zero trials, is refused instead:
/// coercing it to the same zero would hand a malformed footprint the most favourable bar
/// and inflate the deflated Sharpe. SharpeBench refuses the same dispersions (R02).
fn expected_max_sharpe(trials_sr_std: f64, n_trials: u32) -> Result<f64, ConfidenceError> {
    if !trials_sr_std.is_finite() || trials_sr_std < 0.0 {
        return Err(ConfidenceError(DISPERSION_REFUSAL));
    }
    if n_trials == 0 {
        return Err(ConfidenceError(TRIALS_REFUSAL));
    }
    let n = n_trials as f64;
    if n <= 1.0 || trials_sr_std == 0.0 {
        return Ok(0.0);
    }
    const GAMMA: f64 = 0.577_215_664_901_532_9; // Euler–Mascheroni
    let e = std::f64::consts::E;
    let z1 = norm_ppf(1.0 - 1.0 / n);
    let z2 = norm_ppf(1.0 - 1.0 / (n * e));
    Ok(trials_sr_std * ((1.0 - GAMMA) * z1 + GAMMA * z2))
}

/// Convert the annualized dispersion prior to per period, exactly once.
fn per_period_dispersion(
    trials_sr_std: f64,
    periods_per_year: f64,
) -> Result<f64, ConfidenceError> {
    if !periods_per_year.is_finite() || periods_per_year <= 0.0 {
        return Err(ConfidenceError(PERIODS_REFUSAL));
    }
    Ok(trials_sr_std / periods_per_year.sqrt())
}

/// Deflated Sharpe Ratio: the PSR against the expected-maximum Sharpe seen by chance
/// across `n_trials`. Near 1.0 ⇒ the edge is very unlikely to be selection luck; near
/// 0.0 ⇒ indistinguishable from luck. `trials_sr_std` is **annualized** (like
/// `ScoreConfig::trials_sr_std`) and converted per period here at the caller's
/// `periods_per_year`, so this matches the scoring kernel given the same `n_trials`,
/// `trials_sr_std` and `periods_per_year`, and a bootstrap over it brackets the
/// leaderboard point. Refuses a negative or non-finite dispersion, zero trials and a
/// non-finite or non-positive `periods_per_year`.
pub fn deflated_sharpe(
    returns: &[f64],
    n_trials: u32,
    trials_sr_std: f64,
    periods_per_year: f64,
) -> Result<f64, ConfidenceError> {
    let per_period_sr_std = per_period_dispersion(trials_sr_std, periods_per_year)?;
    let sr_star = expected_max_sharpe(per_period_sr_std, n_trials)?;
    Ok(probabilistic_sharpe_ratio(returns, sr_star))
}

// --- deterministic resampling ----------------------------------------------------------

/// SplitMix64: a tiny, fully deterministic PRNG. A fixed resample seed replays the exact
/// same bootstrap, so a confidence report is reproducible byte-for-byte.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A uniform `usize` in `[0, n)` via Lemire's multiply-shift (unbiased enough for
    /// resampling and branch-free).
    fn below(&mut self, n: usize) -> usize {
        debug_assert!(n > 0);
        ((self.next_u64() as u128 * n as u128) >> 64) as usize
    }
}

/// `q`-quantile of an unsorted slice by linear interpolation between order statistics
/// (`q` in `[0, 1]`). Empty ⇒ 0.0.
fn quantile(values: &[f64], q: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if v.len() == 1 {
        return v[0];
    }
    let pos = q.clamp(0.0, 1.0) * (v.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - lo as f64;
    v[lo] + (v[hi] - v[lo]) * frac
}

/// Pool the return series of the seeds selected by `idx` into one flat track.
fn pool_selected(per_seed: &[Vec<f64>], idx: &[usize]) -> Vec<f64> {
    let mut pooled = Vec::new();
    for &i in idx {
        pooled.extend_from_slice(&per_seed[i]);
    }
    pooled
}

// --- public results --------------------------------------------------------------------

/// A seed-bootstrap request has no supported confidence estimate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConfidenceError(pub &'static str);

impl std::fmt::Display for ConfidenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}

impl std::error::Error for ConfidenceError {}

fn validate_seed_bootstrap(
    per_seed: &[Vec<f64>],
    trials_sr_std: f64,
    n_boot: usize,
    alpha: f64,
) -> Result<(), ConfidenceError> {
    if per_seed.len() < 2 {
        return Err(ConfidenceError(
            "seed bootstrap requires at least two independent seed units",
        ));
    }
    if n_boot < 2 {
        return Err(ConfidenceError("n_boot must be at least two"));
    }
    if !alpha.is_finite() || alpha <= 0.0 || alpha >= 1.0 {
        return Err(ConfidenceError(
            "alpha must be finite and strictly between zero and one",
        ));
    }
    if !trials_sr_std.is_finite() || trials_sr_std < 0.0 {
        return Err(ConfidenceError(DISPERSION_REFUSAL));
    }
    for row in per_seed {
        if row.len() < 2 || row.iter().any(|r| !r.is_finite()) {
            return Err(ConfidenceError(
                "each seed requires at least two finite returns",
            ));
        }
    }
    Ok(())
}

fn checked_dsr(
    returns: &[f64],
    n_trials: u32,
    per_period_sr_std: f64,
) -> Result<f64, ConfidenceError> {
    // Check before floors or CDF saturation can conceal failed intermediates.
    let sr = sharpe_ratio(returns);
    let g3 = skewness(returns);
    let g4 = kurtosis(returns);
    let denom = 1.0 - g3 * sr + ((g4 - 1.0) / 4.0) * sr * sr;
    let sr_star = expected_max_sharpe(per_period_sr_std, n_trials)?;
    if [mean(returns), std_dev(returns), sr, g3, g4, denom, sr_star]
        .iter()
        .any(|v| !v.is_finite())
    {
        return Err(ConfidenceError("seed-bootstrap statistic is not finite"));
    }
    let value = probabilistic_sharpe_ratio(returns, sr_star);
    if !value.is_finite() {
        return Err(ConfidenceError("seed-bootstrap estimate is not finite"));
    }
    Ok(value)
}

/// A percentile bootstrap confidence interval on the deflated Sharpe.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DsrCi {
    /// The point deflated Sharpe on the full (unresampled) seed band (the leaderboard number).
    pub point: f64,
    /// Lower confidence bound (the `alpha/2` quantile of the bootstrap distribution).
    pub lo: f64,
    /// Upper confidence bound (the `1 - alpha/2` quantile).
    pub hi: f64,
    /// CI width `hi - lo`. A wider interval means the number rests on fewer / noisier seeds.
    pub width: f64,
    /// Confidence level actually reported, `1 - alpha`.
    pub confidence: f64,
    /// Number of bootstrap resamples used.
    pub n_boot: usize,
}

/// The outcome of a paired-difference significance test between two entries.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PairedDiff {
    /// Point deflated-Sharpe difference `A - B` on the full shared seed band.
    pub point_diff: f64,
    /// Lower / upper bounds of the bootstrap CI on the paired difference.
    pub lo: f64,
    pub hi: f64,
    /// Two-sided bootstrap p-value for `H0: A == B`.
    pub p_value: f64,
    /// Confidence level, `1 - alpha`.
    pub confidence: f64,
    /// `true` when the difference CI excludes zero under this resampling procedure.
    pub significant: bool,
    /// `"a_better"`, `"b_better"`, or legacy `"tied"`. The latter means difference not
    /// established, not equivalence. Retained as a wire label for existing consumers.
    pub verdict: String,
    /// Number of bootstrap resamples used.
    pub n_boot: usize,
}

/// Seed-paired percentile bootstrap CI on the deflated Sharpe.
///
/// `per_seed` is one return series per held-out seed (the independent sampling units).
/// Each of `n_boot` resamples draws `per_seed.len()` seeds with replacement, pools their
/// returns, and recomputes the deflated Sharpe; the CI is the `[alpha/2, 1 - alpha/2]`
/// percentile interval. `resample_seed` fixes the RNG so the report is reproducible.
/// Requires at least two seed units, two bootstrap draws, finite observations and
/// a confidence level strictly between zero and one, and it refuses the deflation
/// inputs [`deflated_sharpe`] refuses. `trials_sr_std` is annualized and converted at
/// the caller's `periods_per_year`. These are necessary input conditions, not a
/// guarantee of independence or finite-sample coverage.
pub fn bootstrap_dsr_ci(
    per_seed: &[Vec<f64>],
    n_trials: u32,
    trials_sr_std: f64,
    periods_per_year: f64,
    n_boot: usize,
    resample_seed: u64,
    alpha: f64,
) -> Result<DsrCi, ConfidenceError> {
    validate_seed_bootstrap(per_seed, trials_sr_std, n_boot, alpha)?;
    let per_period_sr_std = per_period_dispersion(trials_sr_std, periods_per_year)?;
    let n = per_seed.len();
    let full: Vec<f64> = per_seed.iter().flatten().copied().collect();
    let point = checked_dsr(&full, n_trials, per_period_sr_std)?;
    let confidence = 1.0 - alpha;

    let mut rng = SplitMix64(resample_seed);
    let mut samples = Vec::with_capacity(n_boot);
    let mut idx = vec![0usize; n];
    for _ in 0..n_boot {
        for slot in idx.iter_mut() {
            *slot = rng.below(n);
        }
        let pooled = pool_selected(per_seed, &idx);
        samples.push(checked_dsr(&pooled, n_trials, per_period_sr_std)?);
    }

    let lo = quantile(&samples, alpha / 2.0);
    let hi = quantile(&samples, 1.0 - alpha / 2.0);
    Ok(DsrCi {
        point,
        lo,
        hi,
        width: hi - lo,
        confidence,
        n_boot,
    })
}

/// Paired-difference significance test between two entries scored on the **same** held-out
/// seed band.
///
/// `a_per_seed[i]` and `b_per_seed[i]` must be the two entries' return series on the *same*
/// seed `i` (pairing retains shared-path covariance). Each bootstrap draw
/// picks one resampled set of seed indices and applies it to both entries, forming the
/// deflated-Sharpe difference `DSR(A) - DSR(B)`. The CI is the percentile interval of that
/// paired difference; when it excludes zero the difference is established under this
/// resampling for this seed band, which is not a statement that either entry has skill.
///
/// Both bands must contain the same number of seed units and equal per-pair return
/// lengths. Identity alignment and independence remain caller assumptions; no
/// implicit shared-prefix truncation is performed. Other input requirements match
/// [`bootstrap_dsr_ci`].
#[allow(clippy::too_many_arguments)]
pub fn paired_dsr_diff(
    a_per_seed: &[Vec<f64>],
    b_per_seed: &[Vec<f64>],
    n_trials: u32,
    trials_sr_std: f64,
    periods_per_year: f64,
    n_boot: usize,
    resample_seed: u64,
    alpha: f64,
) -> Result<PairedDiff, ConfidenceError> {
    validate_seed_bootstrap(a_per_seed, trials_sr_std, n_boot, alpha)?;
    validate_seed_bootstrap(b_per_seed, trials_sr_std, n_boot, alpha)?;
    if a_per_seed.len() != b_per_seed.len()
        || a_per_seed
            .iter()
            .zip(b_per_seed)
            .any(|(a, b)| a.len() != b.len())
    {
        return Err(ConfidenceError(
            "paired seed bands and per-pair return lengths must match",
        ));
    }
    let per_period_sr_std = per_period_dispersion(trials_sr_std, periods_per_year)?;
    let n = a_per_seed.len();
    let a_full: Vec<f64> = a_per_seed[..n].iter().flatten().copied().collect();
    let b_full: Vec<f64> = b_per_seed[..n].iter().flatten().copied().collect();
    let point_diff = checked_dsr(&a_full, n_trials, per_period_sr_std)?
        - checked_dsr(&b_full, n_trials, per_period_sr_std)?;
    let confidence = 1.0 - alpha;

    let mut rng = SplitMix64(resample_seed);
    let mut diffs = Vec::with_capacity(n_boot);
    let mut idx = vec![0usize; n];
    let mut n_le = 0usize; // resamples with diff <= 0
    let mut n_ge = 0usize; // resamples with diff >= 0
    for _ in 0..n_boot {
        for slot in idx.iter_mut() {
            *slot = rng.below(n);
        }
        let a_pool = pool_selected(&a_per_seed[..n], &idx);
        let b_pool = pool_selected(&b_per_seed[..n], &idx);
        let d = checked_dsr(&a_pool, n_trials, per_period_sr_std)?
            - checked_dsr(&b_pool, n_trials, per_period_sr_std)?;
        if d <= 0.0 {
            n_le += 1;
        }
        if d >= 0.0 {
            n_ge += 1;
        }
        diffs.push(d);
    }

    let lo = quantile(&diffs, alpha / 2.0);
    let hi = quantile(&diffs, 1.0 - alpha / 2.0);
    // Two-sided bootstrap p-value: twice the smaller tail mass, capped at 1.
    let tail = n_le.min(n_ge) as f64 / n_boot as f64;
    let p_value = (2.0 * tail).min(1.0);
    let significant = lo > 0.0 || hi < 0.0;
    Ok(PairedDiff {
        point_diff,
        lo,
        hi,
        p_value,
        confidence,
        significant,
        verdict: verdict_for(point_diff, significant),
        n_boot,
    })
}

fn verdict_for(point_diff: f64, significant: bool) -> String {
    if !significant {
        "tied".to_string()
    } else if point_diff > 0.0 {
        "a_better".to_string()
    } else {
        "b_better".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standardized_moments_use_one_population_normalization() {
        // Independent exact central moments also pinned by sharpebench-stats:
        // m2=3/16, m3=3/32, m4=21/256 for [0,0,0,1].
        let asymmetric = [0.0, 0.0, 0.0, 1.0];
        assert!((skewness(&asymmetric) - 2.0 / 3.0_f64.sqrt()).abs() < 1e-12);
        assert!((kurtosis(&asymmetric) - 7.0 / 3.0).abs() < 1e-12);
        assert!((kurtosis(&[1.0, 2.0, 3.0, 4.0]) - 41.0 / 25.0).abs() < 1e-12);
        assert!(skewness(&[1.0, 2.0, 3.0, 4.0]).abs() < 1e-12);
    }

    #[test]
    fn standardized_moments_are_defined_for_small_nonconstant_samples() {
        assert!(skewness(&[1.0, 2.0]).abs() < 1e-12);
        assert!((kurtosis(&[1.0, 2.0]) - 1.0).abs() < 1e-12);
        assert!((kurtosis(&[1.0, 2.0, 3.0]) - 1.5).abs() < 1e-12);
        assert!((skewness(&[0.0, 0.0, 1.0]) - 1.0 / 2.0_f64.sqrt()).abs() < 1e-12);
        assert!((kurtosis(&[10.0, 10.0, 10.0, 8.0]) - 7.0 / 3.0).abs() < 1e-12);
        assert!((skewness(&[10.0, 10.0, 10.0, 8.0]) + 2.0 / 3.0_f64.sqrt()).abs() < 1e-12);
    }

    // A long, low-vol, positive-drift track: a steady deterministic wobble around a
    // small positive mean. Many such seeds ⇒ a stable, high deflated Sharpe.
    fn steady_seed(offset: usize, len: usize) -> Vec<f64> {
        (0..len)
            .map(|i| 0.001 + 0.0002 * (((i + offset) % 7) as f64 - 3.0))
            .collect()
    }

    // A deterministic track with a prescribed per-period Sharpe: a standardized sine
    // pattern (mean 0, unit std) rescaled so mean/std == `target`. Lets a test place a
    // track squarely in the DSR's sensitive band instead of on its 0/1 rails.
    fn seed_with_sharpe(target: f64, len: usize, phase: f64) -> Vec<f64> {
        let base: Vec<f64> = (0..len).map(|i| (i as f64 + phase).sin()).collect();
        let m = base.iter().sum::<f64>() / len as f64;
        let var = base.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (len as f64 - 1.0);
        let sd = var.sqrt();
        let scale = 0.01; // per-period std of the emitted series
        base.iter()
            .map(|x| target * scale + ((x - m) / sd) * scale)
            .collect()
    }

    fn ci_default(per_seed: &[Vec<f64>]) -> DsrCi {
        bootstrap_dsr_ci(
            per_seed,
            56,
            TRIALS_SR_STD_DEFAULT,
            PERIODS_PER_YEAR,
            2000,
            0x00C1,
            0.05,
        )
        .unwrap()
    }

    #[test]
    fn confidence_boundaries_refuse_missing_support_and_computed_overflow() {
        let valid = vec![vec![0.01, 0.02], vec![-0.01, 0.01]];
        for n_boot in [0, 1] {
            assert!(bootstrap_dsr_ci(&valid, 56, 0.5, PERIODS_PER_YEAR, n_boot, 1, 0.05).is_err());
            assert!(
                paired_dsr_diff(&valid, &valid, 56, 0.5, PERIODS_PER_YEAR, n_boot, 1, 0.05)
                    .is_err()
            );
        }
        for rows in [
            vec![],
            vec![vec![0.01, 0.02]],
            vec![vec![], vec![0.01, 0.02]],
            vec![vec![f64::NAN, 0.01], vec![0.01, 0.02]],
            vec![vec![1e308, 1e308], vec![0.01, 0.02]],
        ] {
            assert!(bootstrap_dsr_ci(&rows, 56, 0.5, PERIODS_PER_YEAR, 10, 1, 0.05).is_err());
            assert!(paired_dsr_diff(&rows, &rows, 56, 0.5, PERIODS_PER_YEAR, 10, 1, 0.05).is_err());
        }
        for alpha in [f64::NAN, 0.0, 1.0] {
            assert!(bootstrap_dsr_ci(&valid, 56, 0.5, PERIODS_PER_YEAR, 10, 1, alpha).is_err());
        }
        let mut extra = valid.clone();
        extra.push(vec![0.01, 0.02]);
        assert!(paired_dsr_diff(&valid, &extra, 56, 0.5, PERIODS_PER_YEAR, 10, 1, 0.05).is_err());
        extra = valid.clone();
        extra[0].push(0.03);
        assert!(paired_dsr_diff(&valid, &extra, 56, 0.5, PERIODS_PER_YEAR, 10, 1, 0.05).is_err());
        assert!(bootstrap_dsr_ci(&valid, 56, 0.5, PERIODS_PER_YEAR, 2, 1, 0.05).is_ok());
    }

    #[test]
    fn ci_brackets_the_point_dsr() {
        let seeds: Vec<Vec<f64>> = (0..10).map(|s| steady_seed(s, 120)).collect();
        let ci = ci_default(&seeds);
        assert!(
            ci.lo <= ci.point + 1e-12 && ci.point <= ci.hi + 1e-12,
            "point {} must lie within [{}, {}]",
            ci.point,
            ci.lo,
            ci.hi
        );
        assert!(ci.width >= 0.0);
    }

    #[test]
    fn ci_is_wider_for_a_noisier_shorter_track() {
        // Both tracks sit in the DSR's sensitive band. With the annualized prior
        // converted at 252 periods/year, n_trials=3 puts sr* ≈ 0.027 per period. The
        // stable entry is many long seeds whose per-seed Sharpe is tightly clustered near
        // sr*, so any resample lands the same place; the noisy entry is a few short seeds
        // with widely dispersed Sharpe, so the resample composition swings the number.
        let stable: Vec<Vec<f64>> = (0..12)
            .map(|s| seed_with_sharpe(0.026 + 0.001 * (s as f64 % 3.0), 40, s as f64))
            .collect();
        let noisy: Vec<Vec<f64>> = [-0.3, 0.03, 0.35]
            .iter()
            .enumerate()
            .map(|(s, &t)| seed_with_sharpe(t, 16, s as f64))
            .collect();
        let stable_ci = bootstrap_dsr_ci(
            &stable,
            3,
            TRIALS_SR_STD_DEFAULT,
            PERIODS_PER_YEAR,
            2000,
            0x00C1,
            0.05,
        )
        .unwrap();
        let noisy_ci = bootstrap_dsr_ci(
            &noisy,
            3,
            TRIALS_SR_STD_DEFAULT,
            PERIODS_PER_YEAR,
            2000,
            0x00C1,
            0.05,
        )
        .unwrap();
        assert!(
            noisy_ci.width > stable_ci.width,
            "noisy/short width {} should exceed stable/long width {}",
            noisy_ci.width,
            stable_ci.width
        );
    }

    #[test]
    fn paired_flags_close_entries_as_tied() {
        // A and B share the seed band; per seed they differ by a small, sign-alternating
        // margin, so the pooled edge is a wash and resamples straddle zero.
        let a: Vec<Vec<f64>> = (0..8).map(|s| steady_seed(s, 120)).collect();
        let b: Vec<Vec<f64>> = (0..8)
            .map(|s| {
                let sign = if s % 2 == 0 { 1.0 } else { -1.0 };
                steady_seed(s, 120)
                    .iter()
                    .map(|r| r + sign * 0.00003)
                    .collect()
            })
            .collect();
        let d = paired_dsr_diff(
            &a,
            &b,
            56,
            TRIALS_SR_STD_DEFAULT,
            PERIODS_PER_YEAR,
            2000,
            0x5EED,
            0.05,
        )
        .unwrap();
        assert!(!d.significant, "close entries should be tied, got {d:?}");
        assert_eq!(d.verdict, "tied");
        assert!(d.lo <= 0.0 && d.hi >= 0.0, "tied CI must straddle 0: {d:?}");
    }

    #[test]
    fn paired_separates_clearly_different_skill() {
        // A has a steady positive edge on every seed; B loses on every seed. Every
        // resample keeps A above B, so the difference CI clears zero.
        let a: Vec<Vec<f64>> = (0..8).map(|s| steady_seed(s, 120)).collect();
        let b: Vec<Vec<f64>> = (0..8)
            .map(|s| steady_seed(s, 120).iter().map(|r| -r).collect())
            .collect();
        let d = paired_dsr_diff(
            &a,
            &b,
            56,
            TRIALS_SR_STD_DEFAULT,
            PERIODS_PER_YEAR,
            2000,
            0x5EED,
            0.05,
        )
        .unwrap();
        assert!(
            d.significant,
            "clear skill gap should be significant: {d:?}"
        );
        assert_eq!(d.verdict, "a_better");
        assert!(
            d.lo > 0.0,
            "CI must exclude zero on the positive side: {d:?}"
        );
        assert!(d.p_value < 0.05, "p-value should be small: {d:?}");
    }

    #[test]
    fn identical_entries_are_tied_with_zero_diff() {
        let a: Vec<Vec<f64>> = (0..6).map(|s| steady_seed(s, 90)).collect();
        let d = paired_dsr_diff(
            &a,
            &a,
            56,
            TRIALS_SR_STD_DEFAULT,
            PERIODS_PER_YEAR,
            500,
            0x1234,
            0.05,
        )
        .unwrap();
        assert_eq!(d.point_diff, 0.0);
        assert!(!d.significant);
        assert_eq!(d.verdict, "tied");
        assert_eq!(d.p_value, 1.0);
    }

    #[test]
    fn bootstrap_is_deterministic_per_resample_seed() {
        let seeds: Vec<Vec<f64>> = (0..8).map(|s| steady_seed(s, 100)).collect();
        let a = bootstrap_dsr_ci(
            &seeds,
            56,
            TRIALS_SR_STD_DEFAULT,
            PERIODS_PER_YEAR,
            1000,
            42,
            0.05,
        )
        .unwrap();
        let b = bootstrap_dsr_ci(
            &seeds,
            56,
            TRIALS_SR_STD_DEFAULT,
            PERIODS_PER_YEAR,
            1000,
            42,
            0.05,
        )
        .unwrap();
        assert_eq!(a.lo, b.lo);
        assert_eq!(a.hi, b.hi);
    }

    // A drifting sine track whose deflated Sharpe sits inside (0, 1) rather than on a rail.
    fn wobble(k: usize, len: usize) -> Vec<f64> {
        (0..len)
            .map(|i| 0.0004 * (k as f64 + 1.0) + 0.01 * ((i * 7 + k * 3) as f64).sin())
            .collect()
    }

    fn refused(message: &'static str) -> Result<f64, ConfidenceError> {
        Err(ConfidenceError(message))
    }

    #[test]
    fn expected_max_sharpe_refuses_a_malformed_dispersion_or_trial_count() {
        for bad in [
            -0.03,
            -f64::MIN_POSITIVE,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ] {
            for n_trials in [0, 1, 2, 56] {
                assert_eq!(
                    expected_max_sharpe(bad, n_trials),
                    refused(DISPERSION_REFUSAL),
                    "dispersion {bad} with {n_trials} trials"
                );
            }
        }
        for dispersion in [0.0, 0.03] {
            assert_eq!(expected_max_sharpe(dispersion, 0), refused(TRIALS_REFUSAL));
        }
    }

    #[test]
    fn zero_dispersion_or_a_single_trial_means_no_deflation() {
        assert_eq!(expected_max_sharpe(0.0, 1), Ok(0.0));
        assert_eq!(expected_max_sharpe(0.0, 56), Ok(0.0));
        assert_eq!(expected_max_sharpe(0.03, 1), Ok(0.0));
        assert!(expected_max_sharpe(0.03, 56).unwrap() > 0.0);
        let r = wobble(0, 250);
        assert_eq!(
            deflated_sharpe(&r, 1, TRIALS_SR_STD_DEFAULT, PERIODS_PER_YEAR),
            Ok(probabilistic_sharpe_ratio(&r, 0.0))
        );
        assert_eq!(
            deflated_sharpe(&r, 56, 0.0, PERIODS_PER_YEAR),
            Ok(probabilistic_sharpe_ratio(&r, 0.0))
        );
    }

    #[test]
    fn every_entry_point_refuses_the_malformed_deflation_inputs() {
        let r = wobble(1, 250);
        let seeds: Vec<Vec<f64>> = (0..4).map(|k| wobble(k, 30)).collect();
        for (n_trials, dispersion, periods, message) in [
            (56, -0.5, PERIODS_PER_YEAR, DISPERSION_REFUSAL),
            (56, f64::NAN, PERIODS_PER_YEAR, DISPERSION_REFUSAL),
            (56, f64::INFINITY, PERIODS_PER_YEAR, DISPERSION_REFUSAL),
            (0, 0.5, PERIODS_PER_YEAR, TRIALS_REFUSAL),
            (56, 0.5, 0.0, PERIODS_REFUSAL),
            (56, 0.5, -252.0, PERIODS_REFUSAL),
            (56, 0.5, f64::NAN, PERIODS_REFUSAL),
            (56, 0.5, f64::INFINITY, PERIODS_REFUSAL),
        ] {
            let expected = ConfidenceError(message);
            assert_eq!(
                deflated_sharpe(&r, n_trials, dispersion, periods),
                Err(expected)
            );
            assert_eq!(
                bootstrap_dsr_ci(&seeds, n_trials, dispersion, periods, 10, 1, 0.05).unwrap_err(),
                expected
            );
            assert_eq!(
                paired_dsr_diff(&seeds, &seeds, n_trials, dispersion, periods, 10, 1, 0.05)
                    .unwrap_err(),
                expected
            );
        }
    }

    // Recorded from the pre-repair estimator (merge 1380156) on these inputs,
    // on Windows. Platform libm results differ in the last few bits (macOS was
    // observed 2 ULPs away), so recorded constants are compared within
    // `RECORDED_ULPS`. The comparison against the pinned kernel runs on the
    // same platform as the code under test and stays exact.
    const RECORDED_ULPS: u64 = 16;

    fn assert_near_bits(actual: f64, recorded: u64, what: &str) {
        let distance = actual.to_bits().abs_diff(recorded);
        assert!(
            distance <= RECORDED_ULPS,
            "{what}: {actual:e} is {distance} ULPs from the recorded {:e}",
            f64::from_bits(recorded)
        );
    }

    const PRE_REPAIR_DSR_BITS: [[u64; 4]; 4] = [
        [
            0x3fe9e4475ee60d5f,
            0x3fe57d7e2a2dec58,
            0x3fdf85ae4dd7371d,
            0x3fd420559819191a,
        ],
        [
            0x3fecdf79ecd74410,
            0x3fe9cb5657a1f5b0,
            0x3fe4f86e0d74c779,
            0x3fde53c5688b7087,
        ],
        [
            0x3fefdbb56242dee7,
            0x3fef8a9d89ca36b2,
            0x3feea6d23b921f0f,
            0x3fecb20a9c6c8210,
        ],
        [
            0x3feff85d1285151e,
            0x3fefe1e5e4e2d40a,
            0x3fef931ceb3880a4,
            0x3feeb9f6c51aefd6,
        ],
    ];

    #[test]
    fn valid_inputs_keep_their_bits_and_match_the_pinned_kernel() {
        use sharpebench_core::deflated_sharpe::deflated_sharpe_ratio;
        let per_period = TRIALS_SR_STD_DEFAULT / PERIODS_PER_YEAR.sqrt();
        for (k, row) in PRE_REPAIR_DSR_BITS.iter().enumerate() {
            let r = wobble(k, 250);
            for (&n_trials, &bits) in [2u32, 7, 56, 1000].iter().zip(row) {
                let arena =
                    deflated_sharpe(&r, n_trials, TRIALS_SR_STD_DEFAULT, PERIODS_PER_YEAR).unwrap();
                let kernel = deflated_sharpe_ratio(&r, n_trials, per_period).unwrap();
                assert_near_bits(arena, bits, &format!("k={k} n_trials={n_trials}"));
                assert_eq!(
                    arena.to_bits(),
                    kernel.to_bits(),
                    "k={k} n_trials={n_trials}"
                );
            }
        }
        let seeds: Vec<Vec<f64>> = (0..6).map(|k| wobble(k, 60)).collect();
        let other: Vec<Vec<f64>> = (0..6).map(|k| wobble(k + 2, 60)).collect();
        let ci = bootstrap_dsr_ci(&seeds, 56, 0.5, PERIODS_PER_YEAR, 500, 0x00C1, 0.05).unwrap();
        for (value, recorded, what) in [
            (ci.point, 0x3fefb35f82144746, "ci.point"),
            (ci.lo, 0x3fea7314cad5fa33, "ci.lo"),
            (ci.hi, 0x3fefff6f0be1f7f0, "ci.hi"),
        ] {
            assert_near_bits(value, recorded, what);
        }
        let d =
            paired_dsr_diff(&seeds, &other, 56, 0.5, PERIODS_PER_YEAR, 500, 0x5EED, 0.05).unwrap();
        for (value, recorded, what) in [
            (d.point_diff, 0xbf832657bd83c680, "d.point_diff"),
            (d.lo, 0xbfc79f0a93ff4cfe, "d.lo"),
            (d.hi, 0xbf12c5492bcdaccd, "d.hi"),
        ] {
            assert_near_bits(value, recorded, what);
        }
        assert_eq!(d.p_value, 0.0);
    }

    #[test]
    fn periods_per_year_sets_the_per_period_deflation_bar() {
        use sharpebench_core::deflated_sharpe::deflated_sharpe_ratio;
        let r = wobble(1, 250);
        let daily = deflated_sharpe(&r, 56, 0.5, PERIODS_PER_YEAR).unwrap();
        for periods in [12.0, 52.0, 252.0 * 6.5] {
            let per_period = 0.5 / f64::sqrt(periods);
            let value = deflated_sharpe(&r, 56, 0.5, periods).unwrap();
            let expected =
                probabilistic_sharpe_ratio(&r, expected_max_sharpe(per_period, 56).unwrap());
            assert_eq!(value.to_bits(), expected.to_bits(), "{periods}");
            let kernel = deflated_sharpe_ratio(&r, 56, per_period).unwrap();
            assert_eq!(value.to_bits(), kernel.to_bits(), "{periods}");
            // Fewer periods a year carry more of the annualized dispersion into each
            // period, raising the bar the same per-period track has to clear.
            if periods < PERIODS_PER_YEAR {
                assert!(value < daily, "{periods}: {value} vs daily {daily}");
            } else {
                assert!(value > daily, "{periods}: {value} vs daily {daily}");
            }
        }
        let seeds: Vec<Vec<f64>> = (0..4).map(|k| wobble(k, 60)).collect();
        let pooled: Vec<f64> = seeds.iter().flatten().copied().collect();
        let weekly = bootstrap_dsr_ci(&seeds, 56, 0.5, 52.0, 10, 1, 0.05).unwrap();
        assert_eq!(
            weekly.point,
            deflated_sharpe(&pooled, 56, 0.5, 52.0).unwrap()
        );
        assert_ne!(
            weekly.point,
            deflated_sharpe(&pooled, 56, 0.5, PERIODS_PER_YEAR).unwrap()
        );
    }
}
