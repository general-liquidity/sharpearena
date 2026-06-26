//! Seeded procedural scenario generation — Procgen's `(start_level, num_levels)`
//! integer-seed-interval model, ported to the trading environment.
//!
//! A scenario is a **pure deterministic function of one `u64` seed**: the same
//! `(ScenarioSpec, seed)` always yields a byte-identical [`Dataset`]. Train/test
//! generalization is governed exactly the way Procgen governs it — by splitting the
//! seed *interval*, not the data — so an agent provably never trains on a test seed.
//!
//! The generator composes ONLY existing leak-free primitives ([`Dataset::synthetic`],
//! [`Dataset::stress_suite`], [`Dataset::masked`]); deeper per-scenario vol/jump
//! parameterization (continuous difficulty, not the discrete Calm/Hard/Extreme tiers)
//! would require new constructors in `sharpebench-sim` and is deliberately deferred.

use serde::{Deserialize, Serialize};

use crate::Dataset;

/// How adversarial a generated scenario is. The discrete difficulty tier maps each
/// seed onto a different family of the existing leak-free generators.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DistributionMode {
    /// Mild, momentum-autocorrelated synthetic panel ([`Dataset::synthetic`]).
    #[default]
    Calm,
    /// A crisis-suite panel (flash-crash / whipsaw), selected by the seed.
    Hard,
    /// The same crisis panel, contamination-masked ([`Dataset::masked`]) so the
    /// agent cannot pattern-match a memorized ticker or calendar window on top of
    /// the adversarial price path.
    Extreme,
}

/// A reproducible scenario family: a seed interval `[start_level, start_level +
/// num_levels)` (`num_levels == 0` ⇒ unbounded `[start_level, u64::MAX)`), the panel
/// dimensions, and the difficulty tier. `Default` is the mild 4×120 Calm family over
/// the unbounded interval (matching the synthetic façade defaults).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScenarioSpec {
    pub start_level: u64,
    /// Size of the legal seed interval; `0` means unbounded.
    pub num_levels: u64,
    pub n_symbols: usize,
    pub n_days: usize,
    pub distribution_mode: DistributionMode,
}

impl Default for ScenarioSpec {
    fn default() -> Self {
        Self {
            start_level: 0,
            num_levels: 0,
            n_symbols: 4,
            n_days: 120,
            distribution_mode: DistributionMode::Calm,
        }
    }
}

/// Pick one crisis-suite panel deterministically by seed. The suite is non-empty by
/// construction, so the index is always valid.
fn stress_pick(seed: u64) -> Dataset {
    let suite = Dataset::stress_suite(seed);
    let idx = (seed % suite.len() as u64) as usize;
    suite
        .into_iter()
        .nth(idx)
        .expect("stress suite is non-empty")
        .1
}

/// Generate the [`Dataset`] for `spec` under `seed`. Deterministic: identical
/// `(spec, seed)` ⇒ identical `Dataset`.
///
/// `n_symbols` / `n_days` are honored by the `Calm` tier; `Hard` / `Extreme` draw
/// from the crisis suite, which fixes its own panel dimensions (parameterizing those
/// is the deferred `sharpebench-sim` follow-up noted in the module docs).
pub fn generate_scenario(spec: &ScenarioSpec, seed: u64) -> Dataset {
    match spec.distribution_mode {
        DistributionMode::Calm => Dataset::synthetic(spec.n_symbols, spec.n_days, seed),
        DistributionMode::Hard => stress_pick(seed),
        DistributionMode::Extreme => stress_pick(seed).masked(),
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
}
