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

/// An execution-noise setting outside the range its own documentation declares.
///
/// These are **reportable benchmark-integrity knobs**: a run that discloses them is
/// making a claim about the difficulty it was scored under. Left unvalidated the three
/// out-of-range shapes all fail quietly rather than loudly: a negative knob takes the
/// "no knobs configured" fast path, so a run disclosing `delay_prob = -0.1` was in fact
/// noise-free; a probability above one makes `rng.next_unit() < delay_prob` always true,
/// so the agent's own decisions never reach the market; and a NaN passes every `>` and
/// `<=` guard, so the realized action is NaN. The `[INVALID_ARGUMENT]` code is the one
/// the pyo3 boundary re-raises under, since this type crosses it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ExecNoiseError {
    /// `delay_prob` is not a finite probability in `[0, 1]`.
    DelayProb {
        /// The value the caller supplied.
        delay_prob: f64,
    },
    /// `slippage_bps` is not a finite non-negative basis-point scale.
    SlippageBps {
        /// The value the caller supplied.
        slippage_bps: f64,
    },
}

impl std::fmt::Display for ExecNoiseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExecNoiseError::DelayProb { delay_prob } => write!(
                f,
                "[INVALID_ARGUMENT] delay_prob {delay_prob} is outside the finite range [0, 1]"
            ),
            ExecNoiseError::SlippageBps { slippage_bps } => write!(
                f,
                "[INVALID_ARGUMENT] slippage_bps {slippage_bps} is negative or not finite"
            ),
        }
    }
}

impl std::error::Error for ExecNoiseError {}

impl ExecNoise {
    /// Check both knobs against the ranges their documentation declares, so a caller that
    /// holds a config before it has an action to perturb can refuse at the boundary where
    /// the operator set it rather than at the first step.
    ///
    /// # Errors
    ///
    /// [`ExecNoiseError`] naming the offending knob and its value.
    pub fn validate(&self) -> Result<(), ExecNoiseError> {
        if !(self.delay_prob.is_finite() && (0.0..=1.0).contains(&self.delay_prob)) {
            return Err(ExecNoiseError::DelayProb {
                delay_prob: self.delay_prob,
            });
        }
        if !(self.slippage_bps.is_finite() && self.slippage_bps >= 0.0) {
            return Err(ExecNoiseError::SlippageBps {
                slippage_bps: self.slippage_bps,
            });
        }
        Ok(())
    }
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
///
/// # Errors
///
/// [`ExecNoiseError`] when either knob is outside its declared range, before any draw is
/// taken. See [`ExecNoiseError`] for why each shape has to be refused rather than run.
pub fn perturb(
    state_rng_seed: u64,
    step_index: u64,
    requested: &[f64],
    previous: &[f64],
    cfg: &ExecNoise,
) -> Result<Vec<f64>, ExecNoiseError> {
    cfg.validate()?;

    // Default-off fast path: no knobs ⇒ exact pass-through, no draws.
    if cfg.delay_prob <= 0.0 && cfg.slippage_bps <= 0.0 {
        return Ok(requested.to_vec());
    }

    let mut rng = SplitMix64::derive(state_rng_seed, step_index);

    // Sticky / delay: with prob `delay_prob` the previous realized action lands this bar.
    if cfg.delay_prob > 0.0 && rng.next_unit() < cfg.delay_prob {
        return Ok(previous.to_vec());
    }

    if cfg.slippage_bps <= 0.0 {
        return Ok(requested.to_vec());
    }

    let scale = cfg.slippage_bps / 10_000.0;
    Ok(requested
        .iter()
        .map(|&x| {
            let u = 2.0 * rng.next_unit() - 1.0; // bounded uniform in [-1, 1)
            x * (1.0 + scale * u)
        })
        .collect())
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

    /// The perturbation fixtures below all carry in-range knobs; the refusal path has its
    /// own tests, one per knob.
    fn perturbed(
        state_rng_seed: u64,
        step_index: u64,
        requested: &[f64],
        previous: &[f64],
        cfg: &ExecNoise,
    ) -> Vec<f64> {
        perturb(state_rng_seed, step_index, requested, previous, cfg)
            .expect("fixture knobs are in range")
    }

    #[test]
    fn same_seed_and_step_is_identical() {
        let req = [0.2, -0.5, 0.7];
        let prev = [0.0, 0.0, 0.0];
        let c = cfg(0.1, 25.0);
        let a = perturbed(42, 9, &req, &prev, &c);
        let b = perturbed(42, 9, &req, &prev, &c);
        assert_eq!(a, b);
    }

    #[test]
    fn different_step_diverges() {
        let req = [0.2, -0.5, 0.7];
        let prev = [0.0, 0.0, 0.0];
        let c = cfg(0.0, 25.0);
        let a = perturbed(42, 9, &req, &prev, &c);
        let b = perturbed(42, 10, &req, &prev, &c);
        assert_ne!(a, b);
    }

    #[test]
    fn delay_prob_one_returns_previous() {
        let req = [0.2, -0.5, 0.7];
        let prev = [-0.9, 0.1, 0.4];
        // Any seed/step: delay_prob = 1.0 ⇒ `u < 1.0` always holds, so previous is applied.
        for step in 0..16u64 {
            let out = perturbed(7, step, &req, &prev, &cfg(1.0, 50.0));
            assert_eq!(out, prev.to_vec());
        }
    }

    #[test]
    fn no_knobs_is_exact_passthrough() {
        let req = [0.2, -0.5, 0.7];
        let prev = [-0.9, 0.1, 0.4];
        let out = perturbed(123, 3, &req, &prev, &cfg(0.0, 0.0));
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
            let out = perturbed(99, step, &req, &prev, &cfg(0.0, slippage_bps));
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
        let out = perturbed(5, 1, &req, &prev, &cfg(0.0, 100.0));
        // No element collapsed onto `previous`; each stayed near its requested weight.
        assert_ne!(out, prev.to_vec());
        for (o, r) in out.iter().zip(req.iter()) {
            assert!((o - r).abs() <= r.abs() * 0.01 + 1e-12);
        }
    }

    /// Isolation: `slippage_bps` is held at `0.0`, which is in range, so the only knob
    /// that can be refused is `delay_prob`. Each rejected value previously produced a
    /// different wrong answer rather than an error: `-1.0` and `-0.1` took the
    /// "no knobs configured" pass-through, `4.0` made every step sticky, `NaN` poisoned
    /// the action.
    #[test]
    fn an_out_of_range_delay_prob_is_refused() {
        let req = [0.2, -0.5, 0.7];
        let prev = [9.9, 9.9, 9.9];
        for bad in [-1.0, -0.1, 1.000_001, 4.0, f64::NAN, f64::INFINITY] {
            let err = perturb(5, 1, &req, &prev, &cfg(bad, 0.0)).unwrap_err();
            assert!(
                matches!(err, ExecNoiseError::DelayProb { .. }),
                "delay_prob = {bad} must be refused, got {err:?}"
            );
            assert!(err.to_string().contains(&bad.to_string()), "{err}");
        }
        // Both ends of the declared range still run.
        assert_eq!(perturbed(5, 1, &req, &prev, &cfg(0.0, 0.0)), req.to_vec());
        assert_eq!(perturbed(5, 1, &req, &prev, &cfg(1.0, 0.0)), prev.to_vec());
    }

    /// Isolation: `delay_prob` is held at `0.0`, which is in range, so the only knob that
    /// can be refused is `slippage_bps`. There is no upper bound on a basis-point scale,
    /// so a large finite value is deliberately still accepted.
    #[test]
    fn an_out_of_range_slippage_bps_is_refused() {
        let req = [0.2, -0.5, 0.7];
        let prev = [9.9, 9.9, 9.9];
        for bad in [-100.0, -1e-9, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let err = perturb(5, 1, &req, &prev, &cfg(0.0, bad)).unwrap_err();
            assert!(
                matches!(err, ExecNoiseError::SlippageBps { .. }),
                "slippage_bps = {bad} must be refused, got {err:?}"
            );
            assert!(err.to_string().contains(&bad.to_string()), "{err}");
        }
        assert_eq!(perturbed(5, 1, &req, &prev, &cfg(0.0, 0.0)), req.to_vec());
        assert_ne!(
            perturbed(5, 1, &req, &prev, &cfg(0.0, 10_000.0)),
            req.to_vec()
        );
    }

    /// The refusal happens before any draw, so a bad config cannot consume the step's
    /// stream and shift the perturbation a later, valid call would produce.
    #[test]
    fn a_refused_config_takes_no_draw() {
        let req = [0.2, -0.5, 0.7];
        let prev = [0.0, 0.0, 0.0];
        let good = cfg(0.0, 25.0);
        let before = perturbed(42, 9, &req, &prev, &good);
        assert!(perturb(42, 9, &req, &prev, &cfg(f64::NAN, 25.0)).is_err());
        assert_eq!(perturbed(42, 9, &req, &prev, &good), before);
    }
}
