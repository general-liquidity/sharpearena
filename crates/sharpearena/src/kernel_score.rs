//! One return track scored by the pinned SharpeBench kernel, withheld where it has no
//! Sharpe ratio.
//!
//! [`score_returns`] is the score behind the Python `score_run` binding: the pinned
//! kernel's `CompositeScore` for a one-run submission. A constant track has a sample
//! variance of zero, so it has no Sharpe ratio and none of the statistics read off one
//! exists for it.
//!
//! SharpeBench `=0.27.0` refuses such a track itself: `deflation_error` names the
//! refusal, the deflated Sharpe, PSR and deflation bar read the 0.0 floor, the
//! interval, the per-cost and percentile figures and the rolling-Sharpe summary are
//! absent, and the run clears no per-run PSR bar, so it is neither pass^k nor
//! rank-eligible. Arena's own refusal below was written against `=0.26.0`, whose
//! scorer substituted a Sharpe of zero for an all-zero track and a Sharpe near 1e15
//! for a constant nonzero one, and it predates that release. Under the pin it is a
//! second line rather than the only one: on every input [`check_sharpe_defined`]
//! covers, the kernel has already recorded a `deflation_error` with the same wording
//! and the same fields, so the withholding branch does not fire and `score_returns`
//! returns the kernel's score verbatim. It is kept because it is the boundary that
//! makes the behaviour Arena's own rather than a property of whichever kernel is
//! pinned, and because it would fire again on a pin that lost the refusal.
//!
//! The Python `sharpearena.kernel_score` helpers read any `*_error` key as
//! unavailability, so every consumer that reads a ranked number through them records
//! the reason instead, whichever of the two boundaries wrote it.

use sharpebench_core::{score_agent, AgentSubmission, CompositeScore, Run, ScoreConfig, Trace};

use crate::leaderboard_ci::check_sharpe_defined;

/// Score one per-period return track with the pinned kernel, as `score_run` reports it.
///
/// The submission is one run of `returns` with no trace, a neutral 0.5 confidence per
/// bar, outcomes `return > 0`, zero cost and `n_trials` declared in-sample trials, scored
/// under the default configuration at `periods_per_year`. With one run and one execution
/// seed per window, the pooled track and the only per-run track are both `returns`.
///
/// When the kernel recorded no `deflation_error` and [`check_sharpe_defined`] refuses
/// `returns`, the score is withheld: `deflation_error` carries the refusal (for a constant
/// track, [`crate::leaderboard_ci::CONSTANT_TRACK_REFUSAL`]); `deflated_sharpe`, `psr`,
/// `deflation_bar_per_period`, `deflation_bar_annualized_equivalent` and `composite` read
/// the 0.0 floor; `dsr_ci_low`, `dsr_ci_high`, `dsr_se`, `dsr_per_cost`, `dsr_percentile`,
/// `rolling_min_sharpe` and `rolling_frac_positive` are absent; and `passed_k` and
/// `rank_eligible` are false, because a run with no Sharpe ratio clears no per-run PSR
/// bar. A `deflation_error` the kernel recorded itself is kept, since it names the input
/// the kernel refused first. Every other field is the kernel's, and a track with a Sharpe
/// ratio is returned exactly as the kernel scored it.
///
/// Under the `=0.27.0` pin the first condition never holds: the kernel refuses every
/// track [`check_sharpe_defined`] refuses, before this function is asked, and writes the
/// same reason into the same fields, so what this function returns is the kernel's own
/// score for every input. The tests below still assert the withheld shape rather than
/// that identity, because the shape is what consumers read and what a pin that lost the
/// refusal would have to keep. The branch is the boundary, not the behaviour; see the
/// module note.
pub fn score_returns(returns: Vec<f64>, n_trials: u32, periods_per_year: f64) -> CompositeScore {
    let refusal = check_sharpe_defined(&returns).err();
    let outcomes: Vec<bool> = returns.iter().map(|r| *r > 0.0).collect();
    let confidences = vec![0.5_f64; returns.len()];
    let run = Run {
        returns,
        trace: Trace::default(),
        confidences,
        outcomes,
        cost: 0.0,
    };
    let submission = AgentSubmission {
        agent_id: "verifiers-rollout".to_string(),
        runs: vec![run],
        in_sample_trials: n_trials,
        candidates: Vec::new(),
    };
    let mut score = score_agent(
        &submission,
        &ScoreConfig::for_periods_per_year(periods_per_year),
    );
    if let (Some(refusal), None) = (refusal, &score.deflation_error) {
        score.deflation_error = Some(refusal.to_string());
        score.deflated_sharpe = 0.0;
        score.psr = 0.0;
        score.deflation_bar_per_period = 0.0;
        score.deflation_bar_annualized_equivalent = 0.0;
        score.composite = 0.0;
        score.dsr_ci_low = None;
        score.dsr_ci_high = None;
        score.dsr_se = None;
        score.dsr_per_cost = None;
        score.dsr_percentile = None;
        score.rolling_min_sharpe = None;
        score.rolling_frac_positive = None;
        score.passed_k = false;
        score.rank_eligible = false;
    }
    score
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::leaderboard_ci::{CONSTANT_TRACK_REFUSAL, NON_FINITE_SHARPE_REFUSAL};

    /// The pinned kernel's own score of the submission `score_returns` builds.
    fn kernel(returns: &[f64], n_trials: u32) -> CompositeScore {
        let run = Run {
            returns: returns.to_vec(),
            trace: Trace::default(),
            confidences: vec![0.5; returns.len()],
            outcomes: returns.iter().map(|r| *r > 0.0).collect(),
            cost: 0.0,
        };
        score_agent(
            &AgentSubmission {
                agent_id: "verifiers-rollout".to_string(),
                runs: vec![run],
                in_sample_trials: n_trials,
                candidates: Vec::new(),
            },
            &ScoreConfig::for_periods_per_year(252.0),
        )
    }

    /// `score` with every field the refusal withholds set to what the refusal writes, so
    /// a comparison against it isolates those fields from everything the kernel reports.
    fn withheld(mut score: CompositeScore, reason: &str) -> CompositeScore {
        score.deflation_error = Some(reason.to_string());
        score.deflated_sharpe = 0.0;
        score.psr = 0.0;
        score.deflation_bar_per_period = 0.0;
        score.deflation_bar_annualized_equivalent = 0.0;
        score.composite = 0.0;
        score.dsr_ci_low = None;
        score.dsr_ci_high = None;
        score.dsr_se = None;
        score.dsr_per_cost = None;
        score.dsr_percentile = None;
        score.rolling_min_sharpe = None;
        score.rolling_frac_positive = None;
        score.passed_k = false;
        score.rank_eligible = false;
        score
    }

    /// Paper audit 2026-09-14: `flat` never trades, so its track is identically zero and
    /// its Sharpe ratio is 0/0, and a constant nonzero track's is c/0. `score_run` now
    /// withholds both with the constant-track reason. The serialized score carries that
    /// reason, carries no interval or rolling Sharpe, and reads no PSR or deflated Sharpe
    /// above the floor, whatever the sign, length or declared trials. Everything else is
    /// the kernel's own report.
    #[test]
    fn a_constant_track_is_withheld_with_the_constant_track_reason() {
        for value in [0.0, 0.001, -0.002] {
            for (len, n_trials) in [(2, 0), (120, 6), (408, 6)] {
                let track = vec![value; len];
                let score = score_returns(track.clone(), n_trials, 252.0);
                let label = format!("value {value} len {len}");
                assert_eq!(
                    score.deflation_error.as_deref(),
                    Some(CONSTANT_TRACK_REFUSAL),
                    "{label}"
                );
                assert_eq!(
                    score,
                    withheld(kernel(&track, n_trials), CONSTANT_TRACK_REFUSAL),
                    "{label}"
                );
                let json = serde_json::to_value(&score).unwrap();
                assert_eq!(json["deflation_error"], CONSTANT_TRACK_REFUSAL, "{label}");
                for absent in ["dsr_ci_low", "dsr_ci_high", "dsr_se"] {
                    assert!(json.get(absent).is_none(), "{label} {absent}");
                }
                assert!(json["rolling_min_sharpe"].is_null(), "{label}");
                assert_eq!(
                    (json["psr"].as_f64(), json["deflated_sharpe"].as_f64()),
                    (Some(0.0), Some(0.0))
                );
                assert_eq!(
                    (json["passed_k"].as_bool(), json["rank_eligible"].as_bool()),
                    (Some(false), Some(false))
                );
            }
        }
    }

    /// A track that is not constant but whose computed standard deviation underflows to
    /// zero has no Sharpe ratio either, and is withheld under that name.
    #[test]
    fn an_underflowing_track_is_withheld_as_a_non_finite_sharpe() {
        let track: Vec<f64> = (0..60).map(|i| (i % 2) as f64 * 1e-170).collect();
        let score = score_returns(track.clone(), 6, 252.0);
        assert_eq!(
            score.deflation_error.as_deref(),
            Some(NON_FINITE_SHARPE_REFUSAL)
        );
        assert_eq!(
            score,
            withheld(kernel(&track, 6), NON_FINITE_SHARPE_REFUSAL)
        );
    }

    /// A refusal the kernel makes itself keeps the kernel's reason: a non-finite
    /// observation is refused by the kernel before constancy is considered, including a
    /// constant infinite track.
    #[test]
    fn the_kernel_refusal_takes_precedence() {
        for track in [vec![0.01, f64::NAN, 0.02], vec![f64::INFINITY; 4]] {
            let own = kernel(&track, 6);
            assert!(own.deflation_error.is_some(), "{track:?}");
            let score = score_returns(track.clone(), 6, 252.0);
            assert_eq!(score.deflation_error, own.deflation_error, "{track:?}");
            assert_ne!(
                score.deflation_error.as_deref(),
                Some(CONSTANT_TRACK_REFUSAL)
            );
        }
    }

    /// The controls: a dispersed track, a low-volatility one, a sparse one and a
    /// one-bar one are returned exactly as the kernel scored them, bit for bit in their
    /// serialized form.
    #[test]
    fn a_track_with_a_sharpe_ratio_is_the_kernels_score_unchanged() {
        let dispersed: Vec<f64> = (0..120)
            .map(|i| 0.001 + 0.02 * (i as f64 * 0.9 + 1.0).sin())
            .collect();
        let low_volatility: Vec<f64> = dispersed.iter().map(|r| r * 1e-9).collect();
        let tiny: Vec<f64> = (0..250)
            .map(|i| 0.001 + 1e-12 * ((i % 5) as f64 - 2.0))
            .collect();
        let mut sparse = vec![0.0; 120];
        sparse[119] = 1e-9;
        for track in [dispersed, low_volatility, tiny, sparse, vec![0.001]] {
            let score = score_returns(track.clone(), 6, 252.0);
            let own = kernel(&track, 6);
            assert_eq!(
                serde_json::to_string(&score).unwrap(),
                serde_json::to_string(&own).unwrap(),
                "{track:?}"
            );
        }
        let low = score_returns(
            (0..120)
                .map(|i| 1e-9 * (0.001 + 0.02 * (i as f64 * 0.9 + 1.0).sin()))
                .collect(),
            6,
            252.0,
        );
        assert!(
            low.deflation_error.is_none() && low.deflated_sharpe > 0.0 && low.deflated_sharpe < 1.0
        );
    }
}
