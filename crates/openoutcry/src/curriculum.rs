//! Adaptive difficulty-targeting curriculum: Prioritized Level Replay (PLR), ported.
//!
//! A fixed-schedule curriculum (rotate the tier, walk a seed list) never reacts to how
//! the agent is actually doing: it replays trivially-solved levels and slams the agent
//! into hopeless ones in equal measure. Prioritized Level Replay instead spends the next
//! episode on a level in the agent's **zone of proximal development**: one it solves
//! *sometimes* (empirically the 30-70%-solve band), where the learning signal is richest.
//!
//! [`AdaptiveCurriculum`] tracks a per-level solve rate from recorded episode outcomes and
//! scores each candidate level by the ZPD weight `p * (1 - p)` (the Bernoulli variance):
//! maximal at `p = 0.5`, decaying to zero as a level becomes trivially easy (`p -> 1`) or
//! hopeless (`p -> 0`), so the mid-difficulty band is up-weighted and both tails are
//! down-weighted. Selection is a **pure deterministic function of the recorded outcome
//! history**: `argmax` weight with ties broken by lowest level index, no RNG, so a replay
//! is reproducible from the outcome log alone. Unseen levels carry a `prior` pseudo-rate
//! (default `0.5`, the peak) so the curriculum explores each once before it settles into
//! replaying the mid band.

use crate::scenario_gen::{level_seed, ScenarioSpec};

/// Default pseudo success rate for a level with no recorded attempts. `0.5` sits at the
/// peak of the ZPD weight, so an unseen level is treated as maximally informative and is
/// explored before the curriculum starts replaying the observed mid-difficulty band.
const DEFAULT_PRIOR: f64 = 0.5;

/// A Prioritized-Level-Replay difficulty-targeting selector over a fixed candidate set of
/// scenario levels (seeds). Deterministic given the recorded success history.
#[derive(Clone, Debug)]
pub struct AdaptiveCurriculum {
    levels: Vec<u64>,
    solves: Vec<u32>,
    attempts: Vec<u32>,
    prior: f64,
}

impl AdaptiveCurriculum {
    /// Build a curriculum over an explicit, de-duplicated candidate level set (order
    /// preserved). Panics if `levels` is empty (a curriculum needs something to schedule).
    pub fn new(levels: impl IntoIterator<Item = u64>) -> Self {
        Self::with_prior(levels, DEFAULT_PRIOR)
    }

    /// [`AdaptiveCurriculum::new`] with an explicit unseen-level `prior` in `[0, 1]`.
    pub fn with_prior(levels: impl IntoIterator<Item = u64>, prior: f64) -> Self {
        let mut deduped: Vec<u64> = Vec::new();
        for lv in levels {
            if !deduped.contains(&lv) {
                deduped.push(lv);
            }
        }
        assert!(
            !deduped.is_empty(),
            "curriculum needs a non-empty level set"
        );
        let n = deduped.len();
        Self {
            levels: deduped,
            solves: vec![0; n],
            attempts: vec![0; n],
            prior,
        }
    }

    /// Materialize the first `n_levels` seeds of a [`ScenarioSpec`]'s interval (via
    /// [`level_seed`]) as the candidate set. Panics if `n_levels == 0`.
    pub fn from_spec(spec: &ScenarioSpec, n_levels: u64) -> Self {
        assert!(n_levels > 0, "curriculum needs at least one level");
        Self::new((0..n_levels).map(|i| level_seed(spec, i)))
    }

    /// The scheduled candidate levels (seeds), in selection-tie-break order.
    pub fn levels(&self) -> &[u64] {
        &self.levels
    }

    fn index_of(&self, level: u64) -> Option<usize> {
        self.levels.iter().position(|&l| l == level)
    }

    /// Observed solve rate of `level`: the recorded `solves / attempts`, or the `prior`
    /// pseudo-rate when the level has not been attempted. Returns `None` for a level
    /// outside the candidate set.
    pub fn success_rate(&self, level: u64) -> Option<f64> {
        let i = self.index_of(level)?;
        let a = self.attempts[i];
        Some(if a == 0 {
            self.prior
        } else {
            self.solves[i] as f64 / a as f64
        })
    }

    /// ZPD replay weight of `level`: `p * (1 - p)`, peaking at `p = 0.5` and falling to
    /// zero at both the trivial and hopeless extremes. `0.0` for an unknown level.
    pub fn weight(&self, level: u64) -> f64 {
        match self.success_rate(level) {
            Some(p) => p * (1.0 - p),
            None => 0.0,
        }
    }

    /// Record one episode outcome for `level` (`solved` = the agent met the success
    /// criterion). Unknown levels are ignored (guarded by a debug assertion).
    pub fn record(&mut self, level: u64, solved: bool) {
        match self.index_of(level) {
            Some(i) => {
                self.attempts[i] += 1;
                self.solves[i] += u32::from(solved);
            }
            None => debug_assert!(false, "recorded outcome for off-schedule level {level}"),
        }
    }

    /// The next level to replay: the highest-weight (mid-difficulty) candidate, ties
    /// broken by lowest index. A pure deterministic function of the recorded history.
    pub fn select_next(&self) -> u64 {
        let mut best = self.levels[0];
        let mut best_w = self.weight(best);
        for &lv in &self.levels[1..] {
            let w = self.weight(lv);
            if w > best_w {
                best = lv;
                best_w = w;
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zpd_weighting_favors_mid_difficulty() {
        // easy (always solved) and hard (never solved) collapse to weight 0; the
        // half-solved level carries the maximal replay weight and is selected.
        let mut c = AdaptiveCurriculum::new([10u64, 20, 30]);
        for _ in 0..4 {
            c.record(10, true); // p -> 1.0
            c.record(30, false); // p -> 0.0
        }
        c.record(20, true);
        c.record(20, false);
        c.record(20, true);
        c.record(20, false); // p -> 0.5

        assert!((c.weight(10) - 0.0).abs() < 1e-12);
        assert!((c.weight(30) - 0.0).abs() < 1e-12);
        assert!(c.weight(20) > c.weight(10));
        assert!(c.weight(20) > c.weight(30));
        assert_eq!(c.select_next(), 20);
    }

    #[test]
    fn deterministic_given_success_history() {
        let history = [(10u64, true), (30, false), (20, true), (20, false)];
        let build = || {
            let mut c = AdaptiveCurriculum::new([10u64, 20, 30]);
            for &(lv, ok) in &history {
                c.record(lv, ok);
            }
            c
        };
        let a = build();
        let b = build();
        assert_eq!(a.select_next(), b.select_next());
        for lv in [10u64, 20, 30] {
            assert_eq!(a.weight(lv), b.weight(lv));
        }
    }

    #[test]
    fn explores_unseen_levels_before_replaying() {
        // Fresh curriculum: every level sits at the prior peak, so ties break to the
        // lowest index; solving a level to mastery drops it out and the next unseen one
        // is surfaced.
        let mut c = AdaptiveCurriculum::new([5u64, 6, 7]);
        assert_eq!(c.select_next(), 5);
        c.record(5, true); // p(5) -> 1.0, weight 0
        assert_eq!(c.select_next(), 6);
        c.record(6, true);
        assert_eq!(c.select_next(), 7);
    }

    #[test]
    fn success_rate_reports_prior_for_unseen_and_none_off_schedule() {
        let mut c = AdaptiveCurriculum::with_prior([1u64, 2], 0.5);
        assert_eq!(c.success_rate(1), Some(0.5));
        assert_eq!(c.success_rate(99), None);
        c.record(1, true);
        c.record(1, false);
        assert_eq!(c.success_rate(1), Some(0.5));
    }

    #[test]
    fn from_spec_materializes_level_seeds() {
        let spec = ScenarioSpec {
            start_level: 100,
            num_levels: 4,
            ..ScenarioSpec::default()
        };
        let c = AdaptiveCurriculum::from_spec(&spec, 4);
        assert_eq!(c.levels(), &[100, 101, 102, 103]);
    }

    #[test]
    fn deduplicates_candidate_levels() {
        let c = AdaptiveCurriculum::new([7u64, 7, 8, 7, 8]);
        assert_eq!(c.levels(), &[7, 8]);
    }
}
