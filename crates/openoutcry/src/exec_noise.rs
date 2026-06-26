//! Execution-noise perturbation — the determinism-sensitive core of the trading
//! analog of ALE sticky actions, moved down from the Python `ExecutionNoiseWrapper`.
//!
//! The perturbation is a **pure deterministic function** of `(seed, step_index)`: the
//! same inputs always yield a byte-identical realized action, so a published
//! benchmark-integrity number (a non-zero `delay_prob` / `slippage_bps`) reproduces
//! from any surface, not just the Python wrapper. Both knobs default to zero
//! (pass-through, no draws).
//!
//! Two knobs, mirroring the wrapper:
//! - **delay / "sticky"** — with probability `delay_prob` the *previous* realized
//!   action is applied instead of the requested one (the order lands one bar late),
//!   breaking open-loop trajectory replay.
//! - **slippage** — bounded multiplicative jitter on the target weights, scaled by
//!   `slippage_bps` basis points.
//!
//! **Deliberate change from the Python original:** the wrapper drew Gaussian jitter
//! (`rng.normal`). Gaussian sampling routes through `ln`/`sqrt`-based transforms whose
//! last bits differ across libm implementations, so a Gaussian stream is *not*
//! byte-identical across Rust / WASM / Python. This core uses a **bounded uniform**
//! draw in `[-1, 1)` instead — same intent (zero-mean multiplicative jitter), but it
//! uses only mul/add (no transcendentals), so the perturbation is cross-runtime
//! reproducible. The trade-off is a bounded (not heavy-tailed) jitter; the jitter
//! magnitude is capped at `|requested[i]| * slippage_bps / 10_000`. The caller clips
//! the result back into the action space.

/// SplitMix64 — the same dependency-free PRNG family `scenario_gen` uses, so the
/// per-step draw stays cross-runtime deterministic (no transcendental calls).
struct SplitMix64(u64);

impl SplitMix64 {
    /// Seed a per-step stream deterministically from `(seed, step_index)`. The
    /// step index is mixed in with the golden-ratio odd constant so consecutive
    /// steps draw distinct, non-overlapping streams from the same base seed.
    fn derive(seed: u64, step_index: u64) -> Self {
        SplitMix64(seed ^ step_index.wrapping_mul(0x9E37_79B9_7F4A_7C15))
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

/// The two execution-noise knobs. Both default to `0.0` ⇒ exact pass-through.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ExecNoise {
    /// Probability in `[0, 1]` that a step replays the previous realized action.
    pub delay_prob: f64,
    /// Multiplicative jitter scale in basis points (`bps / 10_000` of the weight).
    pub slippage_bps: f64,
}

/// Perturb the `requested` action into a realized one, deterministically from
/// `(state_rng_seed, step_index)`.
///
/// With probability `cfg.delay_prob` the `previous` action is returned (the
/// delay/sticky case). Otherwise each requested weight gets bounded multiplicative
/// jitter `requested[i] * (1 + slippage_bps/10_000 * u)`, where `u` is a uniform draw
/// in `[-1, 1)`. With both knobs at `0.0` the requested action is returned unchanged
/// and no draws are taken. The caller is responsible for clipping the result back
/// into the action space.
pub fn perturb(
    state_rng_seed: u64,
    step_index: u64,
    requested: &[f64],
    previous: &[f64],
    cfg: &ExecNoise,
) -> Vec<f64> {
    // Default-off fast path: no knobs ⇒ exact pass-through, no draws.
    if cfg.delay_prob <= 0.0 && cfg.slippage_bps <= 0.0 {
        return requested.to_vec();
    }

    let mut rng = SplitMix64::derive(state_rng_seed, step_index);

    // Sticky / delay: with prob `delay_prob` the previous realized action lands this bar.
    if cfg.delay_prob > 0.0 && rng.next_unit() < cfg.delay_prob {
        return previous.to_vec();
    }

    if cfg.slippage_bps <= 0.0 {
        return requested.to_vec();
    }

    let scale = cfg.slippage_bps / 10_000.0;
    requested
        .iter()
        .map(|&x| {
            let u = 2.0 * rng.next_unit() - 1.0; // bounded uniform in [-1, 1)
            x * (1.0 + scale * u)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(delay_prob: f64, slippage_bps: f64) -> ExecNoise {
        ExecNoise {
            delay_prob,
            slippage_bps,
        }
    }

    #[test]
    fn same_seed_and_step_is_identical() {
        let req = [0.2, -0.5, 0.7];
        let prev = [0.0, 0.0, 0.0];
        let c = cfg(0.1, 25.0);
        let a = perturb(42, 9, &req, &prev, &c);
        let b = perturb(42, 9, &req, &prev, &c);
        assert_eq!(a, b);
    }

    #[test]
    fn different_step_diverges() {
        let req = [0.2, -0.5, 0.7];
        let prev = [0.0, 0.0, 0.0];
        let c = cfg(0.0, 25.0);
        let a = perturb(42, 9, &req, &prev, &c);
        let b = perturb(42, 10, &req, &prev, &c);
        assert_ne!(a, b);
    }

    #[test]
    fn delay_prob_one_returns_previous() {
        let req = [0.2, -0.5, 0.7];
        let prev = [-0.9, 0.1, 0.4];
        // Any seed/step: delay_prob = 1.0 ⇒ `u < 1.0` always holds, so previous is applied.
        for step in 0..16u64 {
            let out = perturb(7, step, &req, &prev, &cfg(1.0, 50.0));
            assert_eq!(out, prev.to_vec());
        }
    }

    #[test]
    fn no_knobs_is_exact_passthrough() {
        let req = [0.2, -0.5, 0.7];
        let prev = [-0.9, 0.1, 0.4];
        let out = perturb(123, 3, &req, &prev, &cfg(0.0, 0.0));
        assert_eq!(out, req.to_vec());
    }

    #[test]
    fn jitter_stays_within_slippage_bound() {
        let req = [0.2, -0.5, 0.7, 1.0, -1.0];
        let prev = [0.0; 5];
        let slippage_bps = 30.0;
        let scale = slippage_bps / 10_000.0;
        // delay_prob = 0 ⇒ no sticky branch; every element is jittered.
        for step in 0..64u64 {
            let out = perturb(99, step, &req, &prev, &cfg(0.0, slippage_bps));
            for (o, r) in out.iter().zip(req.iter()) {
                assert!(
                    (o - r).abs() <= r.abs() * scale + 1e-12,
                    "jitter {} exceeded bound {} for requested {r}",
                    (o - r).abs(),
                    r.abs() * scale
                );
            }
        }
    }

    #[test]
    fn slippage_only_perturbs_and_ignores_previous() {
        let req = [0.2, -0.5, 0.7];
        let prev = [9.9, 9.9, 9.9];
        let out = perturb(5, 1, &req, &prev, &cfg(0.0, 100.0));
        // No element collapsed onto `previous`; each stayed near its requested weight.
        assert_ne!(out, prev.to_vec());
        for (o, r) in out.iter().zip(req.iter()) {
            assert!((o - r).abs() <= r.abs() * 0.01 + 1e-12);
        }
    }
}
