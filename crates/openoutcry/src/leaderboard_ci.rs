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
//!    price-path luck common to both cancels and the difference isolates skill. If the
//!    difference CI straddles zero the two entries are statistically tied; otherwise one
//!    ranks above the other beyond seed noise.
//!
//! The deflated-Sharpe math is the Bailey & López de Prado estimator, ported here
//! self-contained (no `sharpebench-stats` dependency) so the CI brackets the very
//! statistic the leaderboard reports. The resample RNG is a fixed-seed SplitMix64, so a
//! confidence report replays bit-for-bit from its resample seed.

use serde::{Deserialize, Serialize};

/// Cross-trial Sharpe dispersion the deflation assumes, mirroring
/// `sharpebench_core::ScoreConfig::default().trials_sr_std`. Kept in sync so a CI built
/// here brackets the point deflated Sharpe the scoring kernel reports.
pub const TRIALS_SR_STD_DEFAULT: f64 = 0.5;

/// The scoring kernel's own baseline multiple-testing footprint, mirroring
/// `sharpebench_core::ScoreConfig::default().n_trials`. The effective deflation count is
/// this plus the agent's *declared* in-sample trials, so a CI that wants to bracket the
/// leaderboard number must deflate against `KERNEL_BASE_TRIALS + declared`.
pub const KERNEL_BASE_TRIALS: u32 = 50;

// --- self-contained statistics (ported to match the scoring kernel bit-for-bit) -------

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

fn skewness(xs: &[f64]) -> f64 {
    let n = xs.len();
    if n < 3 {
        return 0.0;
    }
    let m = mean(xs);
    let s = std_dev(xs);
    if s == 0.0 {
        return 0.0;
    }
    let sum: f64 = xs.iter().map(|x| ((x - m) / s).powi(3)).sum();
    sum / n as f64
}

fn kurtosis(xs: &[f64]) -> f64 {
    let n = xs.len();
    if n < 4 {
        return 3.0;
    }
    let m = mean(xs);
    let s = std_dev(xs);
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

/// Probabilistic Sharpe Ratio: probability the true Sharpe exceeds `sr_benchmark`,
/// correcting for track length, skewness and kurtosis. In `[0, 1]`.
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

/// Expected maximum Sharpe under `n_trials` independent trials given cross-trial Sharpe
/// dispersion `trials_sr_std` (Bailey & López de Prado, E[max SR_N]).
fn expected_max_sharpe(trials_sr_std: f64, n_trials: u32) -> f64 {
    let n = n_trials.max(1) as f64;
    if n <= 1.0 || trials_sr_std <= 0.0 {
        return 0.0;
    }
    const GAMMA: f64 = 0.577_215_664_901_532_9; // Euler–Mascheroni
    let e = std::f64::consts::E;
    let z1 = norm_ppf(1.0 - 1.0 / n);
    let z2 = norm_ppf(1.0 - 1.0 / (n * e));
    trials_sr_std * ((1.0 - GAMMA) * z1 + GAMMA * z2)
}

/// Deflated Sharpe Ratio: the PSR against the expected-maximum Sharpe seen by chance
/// across `n_trials`. Near 1.0 ⇒ the edge is very unlikely to be selection luck; near
/// 0.0 ⇒ indistinguishable from luck. Matches the scoring kernel given the same
/// `n_trials` / `trials_sr_std`, so a bootstrap over this brackets the leaderboard point.
pub fn deflated_sharpe(returns: &[f64], n_trials: u32, trials_sr_std: f64) -> f64 {
    let sr_star = expected_max_sharpe(trials_sr_std, n_trials);
    probabilistic_sharpe_ratio(returns, sr_star)
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
    /// `true` when the difference CI excludes zero (A and B separate beyond seed noise).
    pub significant: bool,
    /// `"a_better"`, `"b_better"`, or `"tied"`: the leaderboard-facing verdict.
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
pub fn bootstrap_dsr_ci(
    per_seed: &[Vec<f64>],
    n_trials: u32,
    trials_sr_std: f64,
    n_boot: usize,
    resample_seed: u64,
    alpha: f64,
) -> DsrCi {
    let n = per_seed.len();
    let full: Vec<f64> = per_seed.iter().flatten().copied().collect();
    let point = deflated_sharpe(&full, n_trials, trials_sr_std);
    let confidence = 1.0 - alpha;

    if n == 0 || n_boot == 0 {
        return DsrCi {
            point,
            lo: point,
            hi: point,
            width: 0.0,
            confidence,
            n_boot,
        };
    }

    let mut rng = SplitMix64(resample_seed);
    let mut samples = Vec::with_capacity(n_boot);
    let mut idx = vec![0usize; n];
    for _ in 0..n_boot {
        for slot in idx.iter_mut() {
            *slot = rng.below(n);
        }
        let pooled = pool_selected(per_seed, &idx);
        samples.push(deflated_sharpe(&pooled, n_trials, trials_sr_std));
    }

    let lo = quantile(&samples, alpha / 2.0);
    let hi = quantile(&samples, 1.0 - alpha / 2.0);
    DsrCi {
        point,
        lo,
        hi,
        width: hi - lo,
        confidence,
        n_boot,
    }
}

/// Paired-difference significance test between two entries scored on the **same** held-out
/// seed band.
///
/// `a_per_seed[i]` and `b_per_seed[i]` must be the two entries' return series on the *same*
/// seed `i` (the pairing is what cancels the shared price-path luck). Each bootstrap draw
/// picks one resampled set of seed indices and applies it to both entries, forming the
/// deflated-Sharpe difference `DSR(A) - DSR(B)`. The CI is the percentile interval of that
/// paired difference; when it excludes zero the entries separate beyond seed noise.
///
/// Only the shared prefix `min(len_a, len_b)` is used, so a caller that accidentally passes
/// mismatched bands still gets a paired (never a mismatched-index) comparison.
pub fn paired_dsr_diff(
    a_per_seed: &[Vec<f64>],
    b_per_seed: &[Vec<f64>],
    n_trials: u32,
    trials_sr_std: f64,
    n_boot: usize,
    resample_seed: u64,
    alpha: f64,
) -> PairedDiff {
    let n = a_per_seed.len().min(b_per_seed.len());
    let a_full: Vec<f64> = a_per_seed[..n].iter().flatten().copied().collect();
    let b_full: Vec<f64> = b_per_seed[..n].iter().flatten().copied().collect();
    let point_diff = deflated_sharpe(&a_full, n_trials, trials_sr_std)
        - deflated_sharpe(&b_full, n_trials, trials_sr_std);
    let confidence = 1.0 - alpha;

    if n == 0 || n_boot == 0 {
        let significant = point_diff != 0.0;
        return PairedDiff {
            point_diff,
            lo: point_diff,
            hi: point_diff,
            p_value: if significant { 0.0 } else { 1.0 },
            confidence,
            significant,
            verdict: verdict_for(point_diff, significant),
            n_boot,
        };
    }

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
        let d = deflated_sharpe(&a_pool, n_trials, trials_sr_std)
            - deflated_sharpe(&b_pool, n_trials, trials_sr_std);
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
    PairedDiff {
        point_diff,
        lo,
        hi,
        p_value,
        confidence,
        significant,
        verdict: verdict_for(point_diff, significant),
        n_boot,
    }
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
        bootstrap_dsr_ci(per_seed, 56, TRIALS_SR_STD_DEFAULT, 2000, 0x00C1, 0.05)
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
        // Both tracks sit in the DSR's sensitive band (n_trials=3 ⇒ sr* ≈ 0.43). The
        // stable entry is many long seeds whose per-seed Sharpe is tightly clustered near
        // sr*, so any resample lands the same place; the noisy entry is a few short seeds
        // with widely dispersed Sharpe, so the resample composition swings the number.
        let stable: Vec<Vec<f64>> = (0..12)
            .map(|s| seed_with_sharpe(0.42 + 0.01 * (s as f64 % 3.0), 40, s as f64))
            .collect();
        let noisy: Vec<Vec<f64>> = [0.05, 0.45, 0.9]
            .iter()
            .enumerate()
            .map(|(s, &t)| seed_with_sharpe(t, 16, s as f64))
            .collect();
        let stable_ci = bootstrap_dsr_ci(&stable, 3, TRIALS_SR_STD_DEFAULT, 2000, 0x00C1, 0.05);
        let noisy_ci = bootstrap_dsr_ci(&noisy, 3, TRIALS_SR_STD_DEFAULT, 2000, 0x00C1, 0.05);
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
        let d = paired_dsr_diff(&a, &b, 56, TRIALS_SR_STD_DEFAULT, 2000, 0x5EED, 0.05);
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
        let d = paired_dsr_diff(&a, &b, 56, TRIALS_SR_STD_DEFAULT, 2000, 0x5EED, 0.05);
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
        let d = paired_dsr_diff(&a, &a, 56, TRIALS_SR_STD_DEFAULT, 500, 0x1234, 0.05);
        assert_eq!(d.point_diff, 0.0);
        assert!(!d.significant);
        assert_eq!(d.verdict, "tied");
        assert_eq!(d.p_value, 1.0);
    }

    #[test]
    fn bootstrap_is_deterministic_per_resample_seed() {
        let seeds: Vec<Vec<f64>> = (0..8).map(|s| steady_seed(s, 100)).collect();
        let a = bootstrap_dsr_ci(&seeds, 56, TRIALS_SR_STD_DEFAULT, 1000, 42, 0.05);
        let b = bootstrap_dsr_ci(&seeds, 56, TRIALS_SR_STD_DEFAULT, 1000, 42, 0.05);
        assert_eq!(a.lo, b.lo);
        assert_eq!(a.hi, b.hi);
    }
}
