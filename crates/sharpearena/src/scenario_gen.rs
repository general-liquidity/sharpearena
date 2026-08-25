//! Seeded procedural scenario generation — Procgen's `(start_level, num_levels)`
//! integer-seed-interval model, ported to the trading environment.
//!
//! A scenario is a **pure deterministic function of one `u64` seed**: the same
//! `(ScenarioSpec, seed)` always yields a byte-identical [`Dataset`]. Train/test
//! generalization is governed exactly the way Procgen governs it — by splitting the
//! seed *interval*, not the data — so an agent provably never trains on a test seed.
//!
//! `Calm` is the mild [`Dataset::synthetic`] panel; `Hard` / `Extreme` post-process
//! that same seeded panel — amplifying each bar's volatility around the symbol's mean
//! return and injecting seeded jumps. The transform uses only mul/add/div/`max` (no
//! `ln`/`exp`, which differ across libm implementations), so a generated panel is
//! byte-identical across Rust, WASM, and Python, and `n_symbols` / `n_days` are
//! honored by every tier.

use serde::{Deserialize, Serialize};

use crate::richness::ObservationRichness;
use crate::Dataset;

/// How adversarial a generated scenario is. The discrete difficulty tier maps each
/// seed onto a different family of the existing leak-free generators.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DistributionMode {
    /// Mild, momentum-autocorrelated synthetic panel ([`Dataset::synthetic`]).
    #[default]
    Calm,
    /// The seeded panel with amplified volatility and occasional jumps.
    Hard,
    /// The seeded panel with high volatility and frequent, larger jumps.
    Extreme,
    /// Consecutive symbol pairs share a common-trend random walk, so `y - beta*x`
    /// is a genuinely mean-reverting (stationary AR(1)) spread — a real
    /// cointegrated structure for a market-neutral mandate to exploit. The plain
    /// synthetic panel generates each symbol independently, so it has none.
    #[serde(rename = "cointegrated_pairs")]
    CointegratedPairs,
    /// A momentum→reversion regime change: a trending segment (high drift,
    /// persistent momentum) spliced to a mean-reverting/whipsaw segment, with the
    /// changepoint location drawn from the seed.
    #[serde(rename = "regime_shift")]
    RegimeShift,
}

/// A reproducible scenario family: a seed interval `[start_level, start_level +
/// num_levels)` (`num_levels == 0` ⇒ unbounded `[start_level, u64::MAX)`), the panel
/// dimensions, the difficulty tier, and the observation-richness disclosure. `Default` is
/// the mild 4×120 Calm family over the unbounded interval at the default (`Standard`)
/// disclosure (matching the synthetic façade defaults).
///
/// [`distribution_mode`](Self::distribution_mode) and [`obs_richness`](Self::obs_richness)
/// are **orthogonal** difficulty axes: the former sets how adversarial the price path is,
/// the latter how much of it the agent is shown. Together they span a `(regime × richness)`
/// grid (see [`ObservationRichness`]).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScenarioSpec {
    pub start_level: u64,
    /// Size of the legal seed interval; `0` means unbounded.
    pub num_levels: u64,
    pub n_symbols: usize,
    pub n_days: usize,
    pub distribution_mode: DistributionMode,
    /// How much of the market the agent is shown (trailing lookback + optional
    /// fundamentals/news). Additive and `#[serde(default)]`, so a spec serialized before
    /// this field parses back to the historical disclosure and is byte-identical.
    #[serde(default)]
    pub obs_richness: ObservationRichness,
    /// Opt-in volatility-clustering strength (`0.0` = off, the default). When positive,
    /// a `vol_cluster` post-pass modulates each bar's return deviation by a persistence
    /// state driven by realized absolute returns (an EMA recursion), so `|return|`
    /// autocorrelation turns positive — the Cont volatility-clustering stylized fact.
    /// Additive and `#[serde(default)]`: at `0.0` the generated panel is byte-identical
    /// to the historical output, and a spec serialized before this field parses back to
    /// the unclustered generator.
    #[serde(default)]
    pub vol_clustering: f64,
    /// Opt-in probability of beginning a deterministic jump burst on a bar.
    /// Zero preserves the historical generator byte for byte. Together with
    /// [`jump_burst_persistence`](Self::jump_burst_persistence), this is the
    /// calibration knob for fat tails in Calm and super-Poisson large-move
    /// arrivals in Hard, rather than an undocumented change to either tier.
    #[serde(default)]
    pub jump_burst_probability: f64,
    /// Conditional probability that a jump burst continues for one more bar.
    /// `0.0` makes jump starts isolated; values toward one make large moves
    /// arrive in clusters and increase the Fano intermittency statistic.
    #[serde(default)]
    pub jump_burst_persistence: f64,
    /// Absolute simple-return size of each extra burst jump. A value of zero
    /// disables the post-pass even if a probability was serialized by mistake.
    #[serde(default)]
    pub jump_burst_size: f64,
}

impl Default for ScenarioSpec {
    fn default() -> Self {
        Self {
            start_level: 0,
            num_levels: 0,
            n_symbols: 4,
            n_days: 120,
            distribution_mode: DistributionMode::Calm,
            obs_richness: ObservationRichness::default(),
            vol_clustering: 0.0,
            jump_burst_probability: 0.0,
            jump_burst_persistence: 0.0,
            jump_burst_size: 0.0,
        }
    }
}

impl ScenarioSpec {
    /// A **Calm calibration candidate**: the default Calm family with the two existing
    /// opt-in knob families set to the nearest low-volatility cell found by the paper's
    /// calibration grid. Under the repaired finite-panel gate it passes 7/8 diagnostic
    /// seeds but only 5/8 disjoint confirmation seeds (realized-volatility ratio 1.1497),
    /// so it does *not* meet the pre-declared selection rule. The alternative that reaches
    /// 8/8 diagnostic and 7/8 confirmation seeds violates the 25% volatility cap. The
    /// values remain useful as a reproducible negative-result candidate in
    /// `paper/src/make-f4-realism.py` (`calm_calibration`).
    ///
    /// This is a *declared configuration*, not a change to Calm: `Default` and the
    /// golden path are untouched, and a serialized default spec still parses to the
    /// uncertified canonical tape.  It is deliberately named a candidate: the paper's
    /// finite-panel-calibrated ACF gate rejects it on its held-out confirmation band.
    /// The preset is pinned by its own golden fingerprint for reproducible follow-up.
    pub fn calm_calibration_candidate() -> Self {
        Self {
            vol_clustering: 0.5,
            jump_burst_probability: 0.02,
            jump_burst_persistence: 0.0,
            jump_burst_size: 0.015,
            ..Self::default()
        }
    }
}

/// SplitMix64 — the same dependency-free PRNG family [`Dataset::synthetic`] uses, so
/// jump injection stays cross-runtime deterministic (no transcendental calls).
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        SplitMix64(seed ^ 0x1234_5678_9ABC_DEF0)
    }

    /// Next draw in `[0, 1)`.
    fn next_unit(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Difficulty knobs for [`amplify`]: a `mode_salt` (so each tier draws a distinct jump
/// stream), the volatility multiplier on each bar's deviation from the symbol's mean
/// return, and the per-bar jump probability and magnitude.
struct AmplifyParams {
    mode_salt: u64,
    vol_mult: f64,
    jump_prob: f64,
    jump_size: f64,
}

/// Post-process a base panel: amplify each symbol's per-bar volatility around its mean
/// return and inject seeded jumps. Returns are recomputed simple (mul/add/div/`max`
/// only — no `ln`/`exp`), so the result is byte-identical across runtimes and prices
/// stay strictly positive (the `max(-0.95)` floor keeps `1 + r > 0`).
fn amplify(mut base: Dataset, seed: u64, p: AmplifyParams) -> Dataset {
    let mut rng = SplitMix64::new(seed ^ p.mode_salt);
    for series in base.closes.values_mut() {
        if series.len() < 2 {
            continue;
        }
        let rets: Vec<f64> = (1..series.len())
            .map(|t| series[t] / series[t - 1] - 1.0)
            .collect();
        let mean = rets.iter().sum::<f64>() / rets.len() as f64;
        let mut price = series[0];
        for (i, r) in rets.iter().enumerate() {
            let jump = if rng.next_unit() < p.jump_prob {
                if rng.next_unit() < 0.5 {
                    p.jump_size
                } else {
                    -p.jump_size
                }
            } else {
                0.0
            };
            let adjusted = (mean + p.vol_mult * (r - mean) + jump).max(-0.95);
            price *= 1.0 + adjusted;
            series[i + 1] = price;
        }
    }
    base
}

/// EMA persistence of the volatility-clustering state: `state ← PERSISTENCE * state +
/// (1 - PERSISTENCE) * |realized deviation|`. High enough that a burst of large moves
/// keeps the local scale elevated for many bars (slow-decaying `|r|` autocorrelation).
const CLUSTER_PERSISTENCE: f64 = 0.85;

/// Floor on the per-bar clustering scale so a long quiet stretch can never collapse
/// the tape to zero volatility (mirrors the `max(-0.95)` return floor).
const MIN_CLUSTER_SCALE: f64 = 0.05;

/// Opt-in volatility-clustering post-pass (`strength > 0.0`). Each symbol's per-bar
/// deviation from its mean return is re-scaled by `1 + strength * (state / baseline - 1)`,
/// where `state` is an EMA of the *realized* absolute output deviations and `baseline`
/// is the series' mean absolute deviation. A large move raises `state`, which amplifies
/// the next bars' deviations, which feeds back into `state` — persistent volatility
/// episodes, i.e. positive `|return|` autocorrelation (the fixed point of the recursion
/// is `state = baseline`, so the tape's overall scale is preserved). No RNG: the state
/// is a pure function of the prior generated bars, and the arithmetic is mul/add/div/`max`
/// only (`|x|` is `max(x, -x)`), so the result stays byte-identical across runtimes.
fn vol_cluster(mut base: Dataset, strength: f64) -> Dataset {
    for series in base.closes.values_mut() {
        if series.len() < 2 {
            continue;
        }
        let rets: Vec<f64> = (1..series.len())
            .map(|t| series[t] / series[t - 1] - 1.0)
            .collect();
        let mean = rets.iter().sum::<f64>() / rets.len() as f64;
        let baseline = (rets.iter().map(|r| (r - mean).max(mean - r)).sum::<f64>()
            / rets.len() as f64)
            .max(1e-9);
        let mut state = baseline;
        let mut price = series[0];
        for (i, r) in rets.iter().enumerate() {
            let scale = (1.0 + strength * (state / baseline - 1.0)).max(MIN_CLUSTER_SCALE);
            let adjusted = (mean + scale * (r - mean)).max(-0.95);
            price *= 1.0 + adjusted;
            series[i + 1] = price;
            let dev = (adjusted - mean).max(mean - adjusted);
            state = CLUSTER_PERSISTENCE * state + (1.0 - CLUSTER_PERSISTENCE) * dev;
        }
    }
    base
}

/// Opt-in clustered jump post-pass. An inactive bar begins a burst with
/// `start_probability`; once active, each following bar stays in the burst with
/// `persistence`. The first formulation calibrates *arrival* and *clustering*
/// independently: a user can add rare isolated Calm jumps, or retain the same
/// marginal jump scale while making the Hard tier's exceedances intermittent.
///
/// It deliberately runs after the tier and volatility-clustering transforms.
/// The generated jump is a simple-return addition, prices remain positive by
/// the same `-0.95` floor as [`amplify`], and the only randomness is this
/// deterministic SplitMix64 stream. With all three knobs at zero the function
/// is never called, preserving existing cross-runtime golden bytes.
fn burst_jumps(
    mut base: Dataset,
    seed: u64,
    start_probability: f64,
    persistence: f64,
    size: f64,
) -> Dataset {
    let mut rng = SplitMix64::new(seed ^ 0x4A55_4D50_4255_5253);
    for series in base.closes.values_mut() {
        if series.len() < 2 {
            continue;
        }
        let rets: Vec<f64> = (1..series.len())
            .map(|t| series[t] / series[t - 1] - 1.0)
            .collect();
        let mut price = series[0];
        let mut active = false;
        for (i, r) in rets.iter().enumerate() {
            let p = if active {
                persistence
            } else {
                start_probability
            };
            active = rng.next_unit() < p;
            let jump = if active {
                if rng.next_unit() < 0.5 {
                    size
                } else {
                    -size
                }
            } else {
                0.0
            };
            price *= 1.0 + (*r + jump).max(-0.95);
            series[i + 1] = price;
        }
    }
    base
}

/// A strictly-positive price floor — the level analog of the `max(-0.95)` return
/// floor, keeping every generated close above zero with mul/add/`max` only.
const MIN_PRICE: f64 = 0.01;

/// Draw a symmetric shock in `[-amp, amp)` from one `next_unit` draw (mul/add only).
fn signed(rng: &mut SplitMix64, amp: f64) -> f64 {
    (rng.next_unit() - 0.5) * 2.0 * amp
}

/// Overwrite the panel with **cointegrated pairs**. Consecutive symbols (sorted
/// order) form a pair sharing a per-pair common-trend random walk `f_t` (an I(1)
/// integrated level): `x_t = a_x*f_t + idio_x`, `y_t = a_y*f_t + idio_y + spread_t`,
/// where `idio_*` are stationary white noise and `spread_t = phi*spread_{t-1} +
/// noise` is a stationary AR(1) (`phi = 0.85`). With `beta = a_y/a_x` the common
/// trend cancels in `y - beta*x`, leaving the stationary spread — so the pair is
/// genuinely cointegrated. Pure mul/add/div/`max`; an odd trailing symbol gets a
/// standalone positive random walk. Prices stay `>= MIN_PRICE`.
#[allow(clippy::needless_range_loop)] // indices pair two distinct series at bar `t`
fn cointegrated_pairs(mut base: Dataset, seed: u64) -> Dataset {
    let mut rng = SplitMix64::new(seed ^ 0x436F_5061_6972_735F);
    let mut series: Vec<&mut Vec<f64>> = base.closes.values_mut().collect();
    let n = series.len();
    let mut i = 0;
    while i + 1 < n {
        let len = series[i].len().min(series[i + 1].len());
        let a_x = 0.8 + 0.4 * rng.next_unit();
        let a_y = 0.8 + 0.4 * rng.next_unit();
        let mut f = 100.0_f64;
        let mut spread = 0.0_f64;
        for t in 0..len {
            if t > 0 {
                f = (f + signed(&mut rng, 1.5)).max(MIN_PRICE);
            }
            let idio_x = signed(&mut rng, 0.6);
            let idio_y = signed(&mut rng, 0.6);
            spread = 0.85 * spread + signed(&mut rng, 1.0);
            series[i][t] = (a_x * f + idio_x).max(MIN_PRICE);
            series[i + 1][t] = (a_y * f + idio_y + spread).max(MIN_PRICE);
        }
        i += 2;
    }
    if i < n {
        let len = series[i].len();
        let mut f = 100.0_f64;
        for t in 0..len {
            if t > 0 {
                f = (f + signed(&mut rng, 1.5)).max(MIN_PRICE);
            }
            series[i][t] = f;
        }
    }
    base
}

/// Overwrite each symbol with a **momentum→reversion regime shift**: a trending
/// segment `[0, cp)` (positive drift + persistent momentum AR(1) on returns) spliced
/// to a whipsaw segment `[cp, len)` (negative return autocorrelation, no drift,
/// higher per-bar volatility). The changepoint `cp` is drawn per symbol from the seed
/// inside `[len/3, 2*len/3]`, so the first and last thirds are always in their
/// respective regimes. Returns are compounded with the `max(-0.95)` floor (mul/add
/// only — no `ln`/`exp`), keeping prices positive and cross-runtime identical.
#[allow(clippy::needless_range_loop)] // `t` and `t-1` lookback drive the recurrence
fn regime_shift(mut base: Dataset, seed: u64) -> Dataset {
    let mut rng = SplitMix64::new(seed ^ 0x5265_676D_5368_6674);
    for s in base.closes.values_mut() {
        let len = s.len();
        if len < 2 {
            continue;
        }
        let lo = len / 3;
        let span = ((2 * len) / 3 - lo).max(1);
        let cp = lo + (rng.next_unit() * span as f64) as usize;
        let mut price = 100.0_f64;
        s[0] = price;
        let mut prev_r = 0.0_f64;
        for t in 1..len {
            let r = if t < cp {
                0.004 + 0.7 * prev_r + signed(&mut rng, 0.01)
            } else {
                -0.6 * prev_r + signed(&mut rng, 0.03)
            };
            let r = r.max(-0.95);
            price *= 1.0 + r;
            s[t] = price;
            prev_r = r;
        }
    }
    base
}

/// Generate the [`Dataset`] for `spec` under `seed`. Deterministic: identical
/// `(spec, seed)` ⇒ identical `Dataset`. `n_symbols` / `n_days` are honored by every
/// tier; `Hard` / `Extreme` amplify the same seeded panel (see `amplify`), while
/// `CointegratedPairs` / `RegimeShift` overwrite it with a bespoke structured panel.
pub fn generate_scenario(spec: &ScenarioSpec, seed: u64) -> Dataset {
    let base = Dataset::synthetic(spec.n_symbols, spec.n_days, seed);
    let tiered = match spec.distribution_mode {
        DistributionMode::Calm => base,
        DistributionMode::Hard => amplify(
            base,
            seed,
            AmplifyParams {
                mode_salt: 0x4861_7264_5f5f_5f5f,
                vol_mult: 1.8,
                jump_prob: 0.02,
                jump_size: 0.06,
            },
        ),
        DistributionMode::Extreme => amplify(
            base,
            seed,
            AmplifyParams {
                mode_salt: 0x4578_7472_5f5f_5f5f,
                vol_mult: 3.0,
                jump_prob: 0.06,
                jump_size: 0.13,
            },
        ),
        DistributionMode::CointegratedPairs => cointegrated_pairs(base, seed),
        DistributionMode::RegimeShift => regime_shift(base, seed),
    };
    let clustered = if spec.vol_clustering > 0.0 {
        vol_cluster(tiered, spec.vol_clustering)
    } else {
        tiered
    };
    if spec.jump_burst_probability > 0.0 && spec.jump_burst_size > 0.0 {
        burst_jumps(
            clustered,
            seed,
            spec.jump_burst_probability.clamp(0.0, 1.0),
            spec.jump_burst_persistence.clamp(0.0, 1.0),
            spec.jump_burst_size,
        )
    } else {
        clustered
    }
}

/// The concrete seed for the `index`-th level of `spec`'s interval, mirroring
/// Procgen: `start_level + (index % effective_num_levels)`, where the effective span
/// is `num_levels` (bounded) or the full `[start_level, u64::MAX)` width (unbounded).
pub fn level_seed(spec: &ScenarioSpec, index: u64) -> u64 {
    let span = if spec.num_levels == 0 {
        u64::MAX - spec.start_level
    } else {
        spec.num_levels
    };
    spec.start_level + (index % span)
}

/// Carve a **provably disjoint** test family from a (necessarily bounded) `train`
/// family: the test interval starts at `train.start_level + train.num_levels + gap`,
/// so no seed is shared. Panel dimensions and difficulty are inherited from `train`.
pub fn train_test_split(
    train: ScenarioSpec,
    n_test: u64,
    gap: u64,
) -> (ScenarioSpec, ScenarioSpec) {
    debug_assert!(
        train.num_levels > 0,
        "an unbounded train interval admits no disjoint test split"
    );
    let test_start = train.start_level + train.num_levels + gap;
    let test = ScenarioSpec {
        start_level: test_start,
        num_levels: n_test,
        ..train.clone()
    };
    debug_assert!(
        test.start_level >= train.start_level + train.num_levels,
        "test interval [{}, …) overlaps train [{}, {})",
        test.start_level,
        train.start_level,
        train.start_level + train.num_levels
    );
    (train, test)
}

/// Carve a **cross-regime (cross-distribution) transfer** protocol from an in-sample
/// `train` family. The returned out-of-distribution family shares `train`'s *seed band*
/// and panel dimensions but swaps the [`DistributionMode`] to `test_mode`. Scoring a
/// policy that was selected on the in-distribution family against the out-of-distribution
/// family is a **zero-shot regime-transfer** test: the seeds are held fixed so the only
/// thing that varies is the market regime, which isolates distribution shift from seed
/// luck.
///
/// This is a strictly stronger robustness probe than [`train_test_split`]'s within-tier
/// seed gap. That split holds the *regime* fixed and varies the *seeds* (catching seed
/// overfit); this one holds the *seeds* fixed and varies the *regime* (catching the
/// regime-specific overfit a within-tier gap is blind to). When `test_mode` equals
/// `train.distribution_mode` the two families are identical, so the resulting transfer
/// gap is exactly zero by construction.
pub fn cross_regime_split(
    train: ScenarioSpec,
    test_mode: DistributionMode,
) -> (ScenarioSpec, ScenarioSpec) {
    let test = ScenarioSpec {
        distribution_mode: test_mode,
        ..train.clone()
    };
    (train, test)
}

// --- Sealed evaluation seeds ---------------------------------------------------------------

/// Start of the held-out evaluation band `[EVAL_SEED_BASE, u64::MAX)`. Train seeds live
/// in `[0, EVAL_SEED_BASE)`, so any seed at or above this value is disjoint from the
/// train band by construction. Mirrors `sharpearena.dataset.EVAL_SEED_BASE`.
pub const EVAL_SEED_BASE: u64 = 1_000_000;

/// Minimum salt length [`sealed_seed`]'s callers should enforce at the operator
/// boundary. The derivation is only as unguessable as the salt: 16 random bytes put the
/// salt itself outside any enumeration budget, whereas a short passphrase re-creates the
/// bounded-band table-scan the predictability probe measured.
pub const MIN_SEALED_SALT_BYTES: usize = 16;

/// SplitMix64's output finalizer (the `next_unit` mixing step without the state
/// increment): an invertible 64-bit bijection with full avalanche.
fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Derive the seed for the named eval slot `slot` under a secret `salt`, landing in the
/// held-out band `[EVAL_SEED_BASE, u64::MAX)`.
///
/// The predictability probe showed that a *public* bounded seed band is invertible by a
/// table scan (one observed bar, about a second), so the environment's unpredictability
/// rests entirely on seed secrecy and entropy. Sealed seeds keep the public band
/// *structure* (every sealed seed is `>= EVAL_SEED_BASE`, so disjointness from the train
/// band holds by construction and needs no salt to verify) while making the concrete
/// seeds a function of a salt the evaluator keeps outside the repository.
///
/// Construction: the salt bytes are absorbed by FNV-1a/64 (with the byte length folded
/// in), then the digest, a fixed domain tag, and the slot index are driven through three
/// rounds of the SplitMix64 finalizer, chaining the salt digest back in between rounds
/// so no single round's inverse exposes the salt. The mixed word is reduced modulo the
/// band width and offset by `EVAL_SEED_BASE`. Deterministic in `(salt, slot)`; distinct
/// salts or slots give (with overwhelming probability) distinct seeds.
///
/// **What this is and is not.** It is a keyed PRF-*style* derivation built only from
/// primitives the crate already carries (no crypto dependency), and it is *not* a
/// certified cryptographic MAC: FNV-1a is not collision-resistant and the SplitMix64
/// finalizer is a public bijection, so an adversary holding several `(slot, seed)` pairs
/// may be able to recover the absorbed salt digest algebraically. What it buys: with a
/// high-entropy salt no adversary who knows only the band structure, the slot names, and
/// the generator can enumerate the seeds, which is precisely the hole the probe measured.
/// Any seed that becomes public (via the observed tape) is spent; the salt is revealed
/// after the evaluation so the run replays, and never reused. Operators who need
/// resistance to salt recovery from disclosed seeds should derive the salt per evaluation
/// from a real KDF and treat this function as the final band-mapping step only.
pub fn sealed_seed(salt: &[u8], slot: u64) -> u64 {
    const DOMAIN: u64 = 0x5365_616C_6564_4576; // "SealedEv"
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in salt {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h ^= salt.len() as u64;
    h = h.wrapping_mul(0x0000_0100_0000_01b3);

    let mut s = mix64(h ^ DOMAIN);
    s = mix64(s ^ slot.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    s = mix64(s ^ h.rotate_left(32));

    let span = u64::MAX - EVAL_SEED_BASE;
    EVAL_SEED_BASE + (s % span)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Dependency-free FNV-1a/64 over bytes — the canonical-JSON fingerprint used to
    /// pin cross-runtime serialization determinism without adding a hash crate.
    fn fnv1a(bytes: &[u8]) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }

    /// Golden fingerprint of `generate_scenario(&Calm{4×120}, seed=7)` serialized to
    /// JSON. A published generalization number must reproduce on any runtime, so this
    /// pins the FP/serialization determinism; the wasm crate asserts the same value.
    const GOLDEN_CALM_4X120_SEED7_FNV1A: u64 = 0xb7cf_976c_7121_9c52;
    const GOLDEN_HARD_4X120_SEED7_FNV1A: u64 = 0x2ef5_aff1_a716_05e6;
    const GOLDEN_EXTREME_4X120_SEED7_FNV1A: u64 = 0xb082_0c4d_2c73_7f88;
    const GOLDEN_COINTEGRATED_4X120_SEED7_FNV1A: u64 = 0xa3d2_2742_4ef0_5868;
    const GOLDEN_REGIME_4X120_SEED7_FNV1A: u64 = 0x8b82_2cf3_c9d3_038f;
    /// Golden fingerprint of the clustered generator: `Hard` 4×120, seed 7,
    /// `vol_clustering = 0.5`. Pins the opt-in [`vol_cluster`] pass the same way the
    /// unclustered tiers are pinned; the wasm crate asserts the same value.
    const GOLDEN_HARD_CLUSTERED_4X120_SEED7_FNV1A: u64 = 0xa1d2_31f7_e114_a381;
    /// Golden fingerprint of [`ScenarioSpec::calm_calibration_candidate`] at seed 7. Pins the
    /// candidate's knob values and generated bytes, not a certification claim.
    const GOLDEN_CALM_CALIBRATION_CANDIDATE_4X120_SEED7_FNV1A: u64 = 0x32a6_3f8e_5743_ec93;

    /// Mean per-symbol stdev of simple returns — a realized-volatility proxy.
    fn realized_vol(d: &Dataset) -> f64 {
        let mut acc = 0.0;
        for series in d.closes.values() {
            let rets: Vec<f64> = (1..series.len())
                .map(|t| series[t] / series[t - 1] - 1.0)
                .collect();
            let mean = rets.iter().sum::<f64>() / rets.len() as f64;
            let var = rets.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / rets.len() as f64;
            acc += var.sqrt();
        }
        acc / d.closes.len() as f64
    }

    fn golden_spec() -> ScenarioSpec {
        ScenarioSpec {
            distribution_mode: DistributionMode::Calm,
            n_symbols: 4,
            n_days: 120,
            ..ScenarioSpec::default()
        }
    }

    #[test]
    fn generate_is_deterministic() {
        let spec = ScenarioSpec {
            distribution_mode: DistributionMode::Hard,
            ..ScenarioSpec::default()
        };
        let a = serde_json::to_string(&generate_scenario(&spec, 42)).unwrap();
        let b = serde_json::to_string(&generate_scenario(&spec, 42)).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn distribution_modes_diverge() {
        let calm = ScenarioSpec::default();
        let hard = ScenarioSpec {
            distribution_mode: DistributionMode::Hard,
            ..ScenarioSpec::default()
        };
        let extreme = ScenarioSpec {
            distribution_mode: DistributionMode::Extreme,
            ..ScenarioSpec::default()
        };
        let cj = serde_json::to_string(&generate_scenario(&calm, 1)).unwrap();
        let hj = serde_json::to_string(&generate_scenario(&hard, 1)).unwrap();
        let ej = serde_json::to_string(&generate_scenario(&extreme, 1)).unwrap();
        assert_ne!(cj, hj);
        assert_ne!(hj, ej);
    }

    #[test]
    fn distribution_mode_serializes_lowercase() {
        assert_eq!(
            serde_json::to_string(&DistributionMode::Extreme).unwrap(),
            "\"extreme\""
        );
    }

    #[test]
    fn level_seed_bounded_wraps_within_interval() {
        let spec = ScenarioSpec {
            start_level: 100,
            num_levels: 8,
            ..ScenarioSpec::default()
        };
        for index in 0..32 {
            let s = level_seed(&spec, index);
            assert!((100..108).contains(&s));
        }
        assert_eq!(level_seed(&spec, 0), 100);
        assert_eq!(level_seed(&spec, 8), 100);
        assert_eq!(level_seed(&spec, 9), 101);
    }

    #[test]
    fn level_seed_unbounded_is_offset() {
        let spec = ScenarioSpec {
            start_level: 5,
            num_levels: 0,
            ..ScenarioSpec::default()
        };
        assert_eq!(level_seed(&spec, 0), 5);
        assert_eq!(level_seed(&spec, 17), 22);
    }

    #[test]
    fn train_test_split_is_disjoint() {
        let train = ScenarioSpec {
            start_level: 0,
            num_levels: 1000,
            ..ScenarioSpec::default()
        };
        let (train, test) = train_test_split(train, 200, 50);
        let train_end = train.start_level + train.num_levels;
        assert!(test.start_level >= train_end);
        // No legal train seed equals any legal test seed.
        for ti in [0u64, 1, 999] {
            let train_seed = level_seed(&train, ti);
            for xi in [0u64, 1, 199] {
                assert_ne!(train_seed, level_seed(&test, xi));
            }
        }
        assert_eq!(test.start_level, 1050);
        assert_eq!(test.num_levels, 200);
    }

    #[test]
    fn golden_hash_is_stable() {
        let json = serde_json::to_string(&generate_scenario(&golden_spec(), 7)).unwrap();
        assert_eq!(fnv1a(json.as_bytes()), GOLDEN_CALM_4X120_SEED7_FNV1A);
    }

    #[test]
    fn golden_hash_hard_extreme_stable() {
        let hard = ScenarioSpec {
            distribution_mode: DistributionMode::Hard,
            ..golden_spec()
        };
        let extreme = ScenarioSpec {
            distribution_mode: DistributionMode::Extreme,
            ..golden_spec()
        };
        let hj = serde_json::to_string(&generate_scenario(&hard, 7)).unwrap();
        let ej = serde_json::to_string(&generate_scenario(&extreme, 7)).unwrap();
        assert_eq!(fnv1a(hj.as_bytes()), GOLDEN_HARD_4X120_SEED7_FNV1A);
        assert_eq!(fnv1a(ej.as_bytes()), GOLDEN_EXTREME_4X120_SEED7_FNV1A);
    }

    fn coint_spec() -> ScenarioSpec {
        ScenarioSpec {
            distribution_mode: DistributionMode::CointegratedPairs,
            ..golden_spec()
        }
    }

    fn regime_spec() -> ScenarioSpec {
        ScenarioSpec {
            distribution_mode: DistributionMode::RegimeShift,
            ..golden_spec()
        }
    }

    #[test]
    fn structured_modes_serialize_snake_case() {
        assert_eq!(
            serde_json::to_string(&DistributionMode::CointegratedPairs).unwrap(),
            "\"cointegrated_pairs\""
        );
        assert_eq!(
            serde_json::to_string(&DistributionMode::RegimeShift).unwrap(),
            "\"regime_shift\""
        );
    }

    #[test]
    fn structured_modes_are_deterministic() {
        for spec in [coint_spec(), regime_spec()] {
            let a = serde_json::to_string(&generate_scenario(&spec, 7)).unwrap();
            let b = serde_json::to_string(&generate_scenario(&spec, 7)).unwrap();
            assert_eq!(a, b);
        }
    }

    #[test]
    fn structured_modes_diverge_from_calm() {
        let cj = serde_json::to_string(&generate_scenario(&golden_spec(), 7)).unwrap();
        let pj = serde_json::to_string(&generate_scenario(&coint_spec(), 7)).unwrap();
        let rj = serde_json::to_string(&generate_scenario(&regime_spec(), 7)).unwrap();
        assert_ne!(cj, pj);
        assert_ne!(cj, rj);
        assert_ne!(pj, rj);
    }

    #[test]
    fn golden_hash_structured_modes_stable() {
        let pj = serde_json::to_string(&generate_scenario(&coint_spec(), 7)).unwrap();
        let rj = serde_json::to_string(&generate_scenario(&regime_spec(), 7)).unwrap();
        assert_eq!(fnv1a(pj.as_bytes()), GOLDEN_COINTEGRATED_4X120_SEED7_FNV1A);
        assert_eq!(fnv1a(rj.as_bytes()), GOLDEN_REGIME_4X120_SEED7_FNV1A);
    }

    fn clustered_hard_spec() -> ScenarioSpec {
        ScenarioSpec {
            distribution_mode: DistributionMode::Hard,
            vol_clustering: 0.5,
            ..golden_spec()
        }
    }

    /// Mean autocorrelation of `|r - mean|` over lags 1..=lags, averaged across symbols —
    /// the volatility-clustering stylized fact the realism gate checks.
    fn abs_return_autocorr(d: &Dataset, lags: usize) -> f64 {
        let mut acc = 0.0;
        let mut n = 0usize;
        for series in d.closes.values() {
            let rets: Vec<f64> = (1..series.len())
                .map(|t| series[t] / series[t - 1] - 1.0)
                .collect();
            let mean = rets.iter().sum::<f64>() / rets.len() as f64;
            let a: Vec<f64> = rets.iter().map(|r| (r - mean).abs()).collect();
            let am = a.iter().sum::<f64>() / a.len() as f64;
            let c: Vec<f64> = a.iter().map(|v| v - am).collect();
            let denom: f64 = c.iter().map(|v| v * v).sum();
            if denom == 0.0 {
                continue;
            }
            for k in 1..=lags {
                let num: f64 = (k..c.len()).map(|t| c[t] * c[t - k]).sum();
                acc += num / denom;
                n += 1;
            }
        }
        acc / n as f64
    }

    #[test]
    fn vol_clustering_zero_is_byte_identical() {
        for mode in [
            DistributionMode::Calm,
            DistributionMode::Hard,
            DistributionMode::Extreme,
            DistributionMode::CointegratedPairs,
            DistributionMode::RegimeShift,
        ] {
            let plain = ScenarioSpec {
                distribution_mode: mode,
                ..golden_spec()
            };
            let zeroed = ScenarioSpec {
                vol_clustering: 0.0,
                ..plain.clone()
            };
            let a = serde_json::to_string(&generate_scenario(&plain, 7)).unwrap();
            let b = serde_json::to_string(&generate_scenario(&zeroed, 7)).unwrap();
            assert_eq!(a, b, "vol_clustering = 0.0 must be a no-op for {mode:?}");
        }
    }

    #[test]
    fn vol_clustering_raises_abs_return_autocorr() {
        let plain = ScenarioSpec {
            distribution_mode: DistributionMode::Hard,
            ..golden_spec()
        };
        let mut raised = 0;
        for seed in 0..8u64 {
            let base = abs_return_autocorr(&generate_scenario(&plain, seed), 10);
            let clustered =
                abs_return_autocorr(&generate_scenario(&clustered_hard_spec(), seed), 10);
            if clustered > base {
                raised += 1;
            }
        }
        assert!(
            raised >= 7,
            "clustered |r| autocorrelation should exceed unclustered on ≥7/8 seeds, got {raised}"
        );
    }

    #[test]
    fn vol_clustering_prices_positive_and_deterministic() {
        let a = generate_scenario(&clustered_hard_spec(), 7);
        let b = generate_scenario(&clustered_hard_spec(), 7);
        assert_eq!(
            serde_json::to_string(&a).unwrap(),
            serde_json::to_string(&b).unwrap()
        );
        for series in a.closes.values() {
            for &p in series {
                assert!(p.is_finite() && p > 0.0, "price {p} must be finite and > 0");
            }
        }
    }

    #[test]
    fn jump_burst_zero_is_byte_identical() {
        let plain = ScenarioSpec {
            distribution_mode: DistributionMode::Calm,
            ..golden_spec()
        };
        // A persisted setting with a zero start probability is still off.
        let disabled = ScenarioSpec {
            jump_burst_probability: 0.0,
            jump_burst_persistence: 0.95,
            jump_burst_size: 0.08,
            ..plain.clone()
        };
        assert_eq!(
            serde_json::to_string(&generate_scenario(&plain, 19)).unwrap(),
            serde_json::to_string(&generate_scenario(&disabled, 19)).unwrap()
        );
    }

    #[test]
    fn jump_bursts_are_deterministic_and_keep_prices_positive() {
        let spec = ScenarioSpec {
            distribution_mode: DistributionMode::Calm,
            jump_burst_probability: 0.03,
            jump_burst_persistence: 0.80,
            jump_burst_size: 0.08,
            ..golden_spec()
        };
        let a = generate_scenario(&spec, 19);
        let b = generate_scenario(&spec, 19);
        assert_eq!(
            serde_json::to_string(&a).unwrap(),
            serde_json::to_string(&b).unwrap()
        );
        for series in a.closes.values() {
            assert!(series.iter().all(|p| p.is_finite() && *p > 0.0));
        }
    }

    #[test]
    fn golden_hash_clustered_hard_stable() {
        let json = serde_json::to_string(&generate_scenario(&clustered_hard_spec(), 7)).unwrap();
        assert_eq!(
            fnv1a(json.as_bytes()),
            GOLDEN_HARD_CLUSTERED_4X120_SEED7_FNV1A
        );
    }

    #[test]
    fn calm_calibration_candidate_leaves_default_untouched() {
        // The preset must be a pure superset of the default: same family, same panel,
        // and the default itself still generates the canonical golden bytes.
        let preset = ScenarioSpec::calm_calibration_candidate();
        assert_eq!(preset.distribution_mode, DistributionMode::Calm);
        assert_eq!(preset.n_symbols, ScenarioSpec::default().n_symbols);
        assert_eq!(preset.n_days, ScenarioSpec::default().n_days);
        assert_ne!(preset, ScenarioSpec::default());
        let json = serde_json::to_string(&generate_scenario(&ScenarioSpec::default(), 7)).unwrap();
        assert_eq!(fnv1a(json.as_bytes()), GOLDEN_CALM_4X120_SEED7_FNV1A);
    }

    #[test]
    fn golden_hash_calm_calibration_candidate_stable() {
        let json = serde_json::to_string(&generate_scenario(
            &ScenarioSpec::calm_calibration_candidate(),
            7,
        ))
        .unwrap();
        assert_eq!(
            fnv1a(json.as_bytes()),
            GOLDEN_CALM_CALIBRATION_CANDIDATE_4X120_SEED7_FNV1A
        );
    }

    #[test]
    fn calm_calibration_candidate_stays_calm() {
        // The calibration bound: realized volatility within 25% of default Calm on
        // the diagnostic seeds, and every price finite and positive.
        for seed in 0..8u64 {
            let base = realized_vol(&generate_scenario(&ScenarioSpec::default(), seed));
            let cert = generate_scenario(&ScenarioSpec::calm_calibration_candidate(), seed);
            let ratio = realized_vol(&cert) / base;
            assert!(
                ratio <= 1.25,
                "Calm calibration candidate vol ratio {ratio} exceeds 1.25 at seed {seed}"
            );
            for series in cert.closes.values() {
                assert!(series.iter().all(|p| p.is_finite() && *p > 0.0));
            }
        }
    }

    /// Ordinary-least-squares slope of `y` on `x` (with intercept), mul/add/div only.
    fn ols_beta(x: &[f64], y: &[f64]) -> f64 {
        let n = x.len() as f64;
        let mx = x.iter().sum::<f64>() / n;
        let my = y.iter().sum::<f64>() / n;
        let mut cov = 0.0;
        let mut var = 0.0;
        for (xi, yi) in x.iter().zip(y) {
            cov += (xi - mx) * (yi - my);
            var += (xi - mx) * (xi - mx);
        }
        cov / var
    }

    /// Lo–MacKinlay variance ratio at horizon `q`: `Var(s_t - s_{t-q}) / (q *
    /// Var(s_t - s_{t-1}))`. `< 1` ⇒ mean-reverting, `≈ 1` ⇒ random walk, `> 1` ⇒
    /// trending. Pure mul/add/div.
    fn variance_ratio(s: &[f64], q: usize) -> f64 {
        let diff_var = |k: usize| {
            let d: Vec<f64> = (k..s.len()).map(|t| s[t] - s[t - k]).collect();
            let m = d.iter().sum::<f64>() / d.len() as f64;
            d.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / d.len() as f64
        };
        diff_var(q) / (q as f64 * diff_var(1))
    }

    #[test]
    fn cointegrated_spread_is_mean_reverting() {
        let d = generate_scenario(&coint_spec(), 7);
        let cols: Vec<&Vec<f64>> = d.closes.values().collect();
        // 4 symbols → 2 pairs; every pair's residual spread is bounded (VR < 1),
        // whereas each leg alone is an integrated random walk (VR ≈ 1, not < 1).
        let mut pairs = 0;
        let mut i = 0;
        while i + 1 < cols.len() {
            let x = cols[i];
            let y = cols[i + 1];
            let beta = ols_beta(x, y);
            let spread: Vec<f64> = x.iter().zip(y).map(|(xi, yi)| yi - beta * xi).collect();
            let vr = variance_ratio(&spread, 8);
            assert!(
                vr < 1.0,
                "pair {i} spread VR {vr} should be < 1 (mean-reverting)"
            );
            let leg_vr = variance_ratio(x, 8);
            assert!(
                leg_vr > vr,
                "integrated leg VR {leg_vr} should exceed spread VR {vr}"
            );
            pairs += 1;
            i += 2;
        }
        assert_eq!(pairs, 2);
    }

    #[test]
    fn regime_shift_halves_differ() {
        let d = generate_scenario(&regime_spec(), 7);
        // First third is always trending, last third always whipsaw (cp ∈
        // [len/3, 2*len/3]). Compare realized drift + vol over those two windows.
        let seg_stats = |series: &[f64], a: usize, b: usize| {
            let rets: Vec<f64> = (a + 1..b)
                .map(|t| series[t] / series[t - 1] - 1.0)
                .collect();
            let mean = rets.iter().sum::<f64>() / rets.len() as f64;
            let var = rets.iter().map(|r| (r - mean) * (r - mean)).sum::<f64>() / rets.len() as f64;
            (mean, var.sqrt())
        };
        let mut trend_drift = 0.0;
        let mut whip_drift = 0.0;
        let mut trend_vol = 0.0;
        let mut whip_vol = 0.0;
        let n = d.closes.len() as f64;
        for series in d.closes.values() {
            let len = series.len();
            let (td, tv) = seg_stats(series, 0, len / 3);
            let (wd, wv) = seg_stats(series, (2 * len) / 3, len);
            trend_drift += td;
            whip_drift += wd;
            trend_vol += tv;
            whip_vol += wv;
        }
        trend_drift /= n;
        whip_drift /= n;
        trend_vol /= n;
        whip_vol /= n;
        assert!(
            trend_drift > whip_drift,
            "trending drift {trend_drift} should exceed whipsaw drift {whip_drift}"
        );
        assert!(
            whip_vol > trend_vol,
            "whipsaw vol {whip_vol} should exceed trending vol {trend_vol}"
        );
    }

    #[test]
    fn structured_mode_prices_are_positive_and_finite() {
        for spec in [coint_spec(), regime_spec()] {
            let d = generate_scenario(&spec, 3);
            for series in d.closes.values() {
                for &p in series {
                    assert!(p.is_finite() && p > 0.0, "price {p} must be finite and > 0");
                }
            }
        }
    }

    #[test]
    fn realized_vol_increases_with_difficulty() {
        let spec = |m| ScenarioSpec {
            distribution_mode: m,
            ..ScenarioSpec::default()
        };
        let calm = realized_vol(&generate_scenario(&spec(DistributionMode::Calm), 7));
        let hard = realized_vol(&generate_scenario(&spec(DistributionMode::Hard), 7));
        let extreme = realized_vol(&generate_scenario(&spec(DistributionMode::Extreme), 7));
        assert!(calm < hard, "calm {calm} should be < hard {hard}");
        assert!(hard < extreme, "hard {hard} should be < extreme {extreme}");
    }

    #[test]
    fn cross_regime_split_preserves_band_swaps_mode() {
        let train = ScenarioSpec {
            start_level: 100,
            num_levels: 8,
            distribution_mode: DistributionMode::Calm,
            ..ScenarioSpec::default()
        };
        let (in_dist, out_dist) = cross_regime_split(train.clone(), DistributionMode::Extreme);
        // In-distribution family is returned unchanged; the out-of-distribution family
        // shares the seed band and panel dims but flips the regime.
        assert_eq!(in_dist, train);
        assert_eq!(out_dist.start_level, train.start_level);
        assert_eq!(out_dist.num_levels, train.num_levels);
        assert_eq!(out_dist.n_symbols, train.n_symbols);
        assert_eq!(out_dist.n_days, train.n_days);
        assert_eq!(out_dist.distribution_mode, DistributionMode::Extreme);
    }

    #[test]
    fn cross_regime_identical_mode_is_byte_identical() {
        // test_mode == train mode => the transfer gap is zero by construction: the two
        // families generate byte-identical panels across the whole seed band.
        let train = ScenarioSpec {
            start_level: 0,
            num_levels: 6,
            distribution_mode: DistributionMode::Hard,
            ..ScenarioSpec::default()
        };
        let (in_dist, out_dist) = cross_regime_split(train.clone(), DistributionMode::Hard);
        for index in 0..in_dist.num_levels {
            let seed = level_seed(&in_dist, index);
            let a = serde_json::to_string(&generate_scenario(&in_dist, seed)).unwrap();
            let b = serde_json::to_string(&generate_scenario(&out_dist, seed)).unwrap();
            assert_eq!(a, b, "identical-mode panels must match at seed {seed}");
        }
    }

    #[test]
    fn cross_regime_different_mode_diverges() {
        // A real zero-shot shift: holding the seed fixed, calm and extreme panels differ
        // at every level in the band.
        let train = ScenarioSpec {
            start_level: 0,
            num_levels: 6,
            distribution_mode: DistributionMode::Calm,
            ..ScenarioSpec::default()
        };
        let (in_dist, out_dist) = cross_regime_split(train, DistributionMode::Extreme);
        for index in 0..in_dist.num_levels {
            let seed = level_seed(&in_dist, index);
            let a = serde_json::to_string(&generate_scenario(&in_dist, seed)).unwrap();
            let b = serde_json::to_string(&generate_scenario(&out_dist, seed)).unwrap();
            assert_ne!(a, b, "cross-regime panels must differ at seed {seed}");
        }
    }

    // -- sealed evaluation seeds ---------------------------------------------------------

    const SALT_A: &[u8] = b"operator-secret-salt-A-0123456789";
    const SALT_B: &[u8] = b"operator-secret-salt-B-0123456789";

    #[test]
    fn sealed_seed_lands_in_held_out_band() {
        for slot in 0..256u64 {
            for salt in [SALT_A, SALT_B, b"x".as_slice(), b"".as_slice()] {
                assert!(sealed_seed(salt, slot) >= EVAL_SEED_BASE);
            }
        }
    }

    #[test]
    fn sealed_seed_is_deterministic() {
        for slot in 0..64u64 {
            assert_eq!(sealed_seed(SALT_A, slot), sealed_seed(SALT_A, slot));
        }
        // Pinned value: the Python tests assert the same number, so a change to the
        // derivation is a contract change (a sealed evaluation would stop replaying).
        assert_eq!(sealed_seed(SALT_A, 0), 0x040a_380d_f918_05c2);
    }

    #[test]
    fn sealed_seed_is_salt_and_slot_sensitive() {
        let mut seen = std::collections::HashSet::new();
        for slot in 0..64u64 {
            let a = sealed_seed(SALT_A, slot);
            let b = sealed_seed(SALT_B, slot);
            assert_ne!(a, b, "salts must not collide at slot {slot}");
            assert!(seen.insert(a), "slot collision under salt A at {slot}");
            assert!(seen.insert(b), "slot collision under salt B at {slot}");
        }
        // A single flipped salt byte moves every slot.
        let mut flipped = SALT_A.to_vec();
        flipped[0] ^= 1;
        for slot in 0..16u64 {
            assert_ne!(sealed_seed(SALT_A, slot), sealed_seed(&flipped, slot));
        }
        // The length fold distinguishes a salt from its zero-extended form.
        let mut extended = SALT_A.to_vec();
        extended.push(0);
        assert_ne!(sealed_seed(SALT_A, 0), sealed_seed(&extended, 0));
    }

    #[test]
    fn sealed_seed_is_disjoint_from_train_band() {
        let train = ScenarioSpec {
            start_level: 0,
            num_levels: EVAL_SEED_BASE,
            ..ScenarioSpec::default()
        };
        for slot in 0..64u64 {
            let s = sealed_seed(SALT_A, slot);
            assert!(s >= train.start_level + train.num_levels);
        }
    }

    #[test]
    fn sealed_seed_is_not_near_the_public_band_start() {
        // The probe's table scan covers a 2^16 window; sealed seeds are spread over the
        // whole 2^64 band, so none of these should fall inside any such window at the
        // base (probability 2^-48 per seed).
        for slot in 0..64u64 {
            let s = sealed_seed(SALT_A, slot);
            assert!(s - EVAL_SEED_BASE >= 1 << 16);
        }
    }

    #[test]
    fn sealed_seed_generates_a_valid_scenario() {
        let spec = ScenarioSpec::default();
        let seed = sealed_seed(SALT_A, 3);
        let a = serde_json::to_string(&generate_scenario(&spec, seed)).unwrap();
        let b = serde_json::to_string(&generate_scenario(&spec, seed)).unwrap();
        assert_eq!(a, b);
    }
}
