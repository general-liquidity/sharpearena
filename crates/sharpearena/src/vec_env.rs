//! Vectorized, batched environment — gym3's "vectorized-first" design ported to the
//! trading env. A [`VecTradingEnv`] holds `B` independent [`TradingEnv`] lanes (each a
//! distinct seed/scenario/window) and steps them as a batch.
//!
//! Auto-reset is selectable via [`AutoresetMode`]: the default `NextStep` returns a
//! finished lane's terminal step verbatim and resets it on the *following* step
//! (Gymnasium 1.x default); `SameStep` resets in place and surfaces the terminal
//! obs/info through `final_obs`/`final_info` (gym3's `first` flag); `Disabled` never
//! auto-resets. The other lanes keep stepping regardless. There is exactly one engine —
//! each lane is a real [`TradingEnv`], so the batch math is the scalar math (B = 1 is
//! just a single-lane batch).
//!
//! Determinism is structural: every lane owns its own seeded book/RNG, dataset, and
//! cursor, with no cross-lane shared mutable state, so the rayon-parallel step is
//! byte-identical to a serial step (asserted by `parallel_matches_serial` and
//! `every_dispatch_produces_the_same_trajectory`). On `wasm32`, where threads do not
//! build, the same loop runs serially.
//!
//! Scheduling is therefore a pure cost decision. One transition is microseconds of work,
//! and a Rayon fork-join costs more the larger the pool it wakes, so a batch only goes
//! parallel when its estimated work pays for a fork on the current pool (see
//! [`Dispatch`]). The estimate reads lane, symbol and thread counts, never a clock.

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use crate::{
    generate_scenario, CostModel, Dataset, Decision, DistributionMode, MarketObservation,
    ScenarioSpec, StepInfo, TradingEnv, Window,
};
use std::num::NonZeroUsize;

/// Per-lane construction config: a panel of `n_symbols` × `n_days` under
/// `distribution_mode`, seeded by `seed`, over an optional `window` (full series when
/// `None`) under `costs`. `seed` drives the *scenario* (the price path); `exec_seed`
/// (when set) independently seeds *execution noise*, so the two streams can be split.
#[derive(Clone, Debug)]
pub struct LaneConfig {
    pub n_symbols: usize,
    pub n_days: usize,
    pub seed: u64,
    pub exec_seed: Option<u64>,
    pub distribution_mode: DistributionMode,
    /// Opt-in volatility-clustering strength (`0.0` = off; see
    /// [`ScenarioSpec::vol_clustering`]).
    pub vol_clustering: f64,
    /// Opt-in extra jump-burst start probability (zero keeps historical bytes).
    pub jump_burst_probability: f64,
    /// Conditional continuation probability of an extra jump burst.
    pub jump_burst_persistence: f64,
    /// Absolute simple-return size of an extra burst jump.
    pub jump_burst_size: f64,
    pub window: Option<Window>,
    pub costs: CostModel,
}

impl LaneConfig {
    /// A `Calm` lane over a synthetic `n_symbols` × `n_days` panel seeded by `seed`, with
    /// the full window, default costs, and execution noise seeded by `seed` (matching the
    /// scalar `TradingEnv` defaults).
    pub fn new(n_symbols: usize, n_days: usize, seed: u64) -> Self {
        Self {
            n_symbols,
            n_days,
            seed,
            exec_seed: None,
            distribution_mode: DistributionMode::Calm,
            vol_clustering: 0.0,
            jump_burst_probability: 0.0,
            jump_burst_persistence: 0.0,
            jump_burst_size: 0.0,
            window: None,
            costs: CostModel::default(),
        }
    }

    fn build(&self) -> TradingEnv {
        let data = match self.distribution_mode {
            // The unclustered Calm fast path stays byte-identical to the historical build.
            DistributionMode::Calm
                if self.vol_clustering == 0.0
                    && self.jump_burst_probability == 0.0
                    && self.jump_burst_persistence == 0.0
                    && self.jump_burst_size == 0.0 =>
            {
                Dataset::synthetic(self.n_symbols, self.n_days, self.seed)
            }
            mode => generate_scenario(
                &ScenarioSpec {
                    n_symbols: self.n_symbols,
                    n_days: self.n_days,
                    distribution_mode: mode,
                    vol_clustering: self.vol_clustering,
                    jump_burst_probability: self.jump_burst_probability,
                    jump_burst_persistence: self.jump_burst_persistence,
                    jump_burst_size: self.jump_burst_size,
                    ..ScenarioSpec::default()
                },
                self.seed,
            ),
        };
        let window = self.window.unwrap_or(Window {
            start: 0,
            end: data.len(),
        });
        TradingEnv::new(
            data,
            window,
            self.costs,
            self.exec_seed.unwrap_or(self.seed),
        )
    }
}

/// How a finished lane is recycled, mirroring `gymnasium.vector.AutoresetMode`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AutoresetMode {
    /// Return the terminal step verbatim; reset on the *following* step with reward 0,
    /// `terminated`/`truncated` false, and `first = true` (Gymnasium 1.x default).
    #[default]
    NextStep,
    /// Reset in place this step; surface the terminal obs/info via `final_obs`/`final_info`
    /// and flag `first = true` (gym3's same-step convention).
    SameStep,
    /// Never auto-reset; a finished lane stays at its terminal bar.
    Disabled,
}

impl AutoresetMode {
    /// Parse the wire label (`"next_step" | "same_step" | "disabled"`).
    pub fn from_label(label: &str) -> Option<Self> {
        match label {
            "next_step" => Some(Self::NextStep),
            "same_step" => Some(Self::SameStep),
            "disabled" => Some(Self::Disabled),
            _ => None,
        }
    }

    /// The wire label for this mode.
    pub fn label(self) -> &'static str {
        match self {
            Self::NextStep => "next_step",
            Self::SameStep => "same_step",
            Self::Disabled => "disabled",
        }
    }
}

/// The result of one batched step (structure-of-arrays, length `B`).
///
/// `terminated[i]` means lane `i` blew up (NAV ≤ 0); `truncated[i]` means it ran out of
/// bars or reached an explicit episode limit; `first[i]` means a new episode just started this step (lane `i` was
/// auto-reset, so `observations[i]` is the new episode's t0). `rewards`/`infos` always
/// describe the step that just executed. Under [`AutoresetMode::SameStep`], a lane that
/// finished this step carries its terminal obs/info in `final_obs[i]`/`final_info[i]`
/// (both `None` otherwise).
pub struct BatchStep {
    pub observations: Vec<MarketObservation>,
    pub rewards: Vec<f64>,
    pub terminated: Vec<bool>,
    pub truncated: Vec<bool>,
    pub first: Vec<bool>,
    pub infos: Vec<StepInfo>,
    pub final_obs: Vec<Option<MarketObservation>>,
    pub final_info: Vec<Option<StepInfo>>,
}

/// One lane's per-step output before it is transposed into [`BatchStep`]'s SoA layout.
struct LaneOutcome {
    observation: MarketObservation,
    reward: f64,
    terminated: bool,
    truncated: bool,
    first: bool,
    info: StepInfo,
    final_obs: Option<MarketObservation>,
    final_info: Option<StepInfo>,
}

/// Step a single lane under `mode`, threading the lane's `pending_reset` flag (only
/// `NextStep` uses it). When a lane finishes:
/// - `NextStep` returns its terminal step verbatim and arms `pending_reset`; the *next*
///   call resets it (reward 0, flags false, `first = true`), ignoring that step's decision.
/// - `SameStep` resets in place and stashes the terminal obs/info in `final_obs`/`final_info`
///   (`first = true`), so no tail of the prior episode leaks into the next observation.
/// - `Disabled` returns the terminal step verbatim and never resets.
fn step_lane(
    env: &mut TradingEnv,
    decision: &Decision,
    pending_reset: &mut bool,
    mode: AutoresetMode,
    elapsed: &mut usize,
    capped_final: &mut Option<(MarketObservation, f64, bool)>,
    limit: Option<NonZeroUsize>,
) -> LaneOutcome {
    if let Some((observation, nav, terminated)) = capped_final {
        return LaneOutcome {
            observation: observation.clone(),
            reward: 0.0,
            terminated: *terminated,
            truncated: true,
            first: false,
            info: StepInfo {
                nav: *nav,
                carry: 0.0,
                events: Vec::new(),
            },
            final_obs: None,
            final_info: None,
        };
    }
    if mode == AutoresetMode::NextStep && *pending_reset {
        *pending_reset = false;
        *elapsed = 0;
        let observation = env.reset();
        let info = StepInfo {
            nav: observation.cash,
            carry: 0.0,
            events: Vec::new(),
        };
        return LaneOutcome {
            observation,
            reward: 0.0,
            terminated: false,
            truncated: false,
            first: true,
            info,
            final_obs: None,
            final_info: None,
        };
    }

    let res = env.step(decision.clone());
    *elapsed = elapsed.saturating_add(1);
    let terminated = res.info.nav <= 0.0;
    let capped = limit.is_some_and(|limit| *elapsed >= limit.get());
    let truncated = res.done || capped;
    let ended = terminated || truncated;

    match mode {
        AutoresetMode::SameStep if ended => {
            *elapsed = 0;
            let final_info = StepInfo {
                nav: res.info.nav,
                carry: res.info.carry,
                events: res.info.events.clone(),
            };
            let observation = env.reset();
            LaneOutcome {
                observation,
                reward: res.reward,
                terminated,
                truncated,
                first: true,
                info: res.info,
                final_obs: Some(res.observation),
                final_info: Some(final_info),
            }
        }
        _ => {
            if capped && mode == AutoresetMode::Disabled {
                *capped_final = Some((res.observation.clone(), res.info.nav, terminated));
            }
            if ended && mode == AutoresetMode::NextStep {
                *pending_reset = true;
            }
            LaneOutcome {
                observation: res.observation,
                reward: res.reward,
                terminated,
                truncated,
                first: false,
                info: res.info,
                final_obs: None,
                final_info: None,
            }
        }
    }
}

impl BatchStep {
    fn from_outcomes(outcomes: Vec<LaneOutcome>) -> Self {
        let mut step = BatchStep {
            observations: Vec::with_capacity(outcomes.len()),
            rewards: Vec::with_capacity(outcomes.len()),
            terminated: Vec::with_capacity(outcomes.len()),
            truncated: Vec::with_capacity(outcomes.len()),
            first: Vec::with_capacity(outcomes.len()),
            infos: Vec::with_capacity(outcomes.len()),
            final_obs: Vec::with_capacity(outcomes.len()),
            final_info: Vec::with_capacity(outcomes.len()),
        };
        for o in outcomes {
            step.observations.push(o.observation);
            step.rewards.push(o.reward);
            step.terminated.push(o.terminated);
            step.truncated.push(o.truncated);
            step.first.push(o.first);
            step.infos.push(o.info);
            step.final_obs.push(o.final_obs);
            step.final_info.push(o.final_info);
        }
        step
    }
}

/// Fork-join cost per squared pool thread. A batch on a `T`-thread pool goes parallel only
/// when its estimated work is at least `T * T` times this. A native-only tuning value: it
/// picks a schedule and cannot reach a result, because every schedule yields the same
/// bytes.
///
/// Measured on a contended 64-core host over 1 to 64 threads and 8 to 256 lanes, the
/// break-even work grows faster than linearly in the pool size: one-lane tasks beat the
/// serial loop at 24 us of work on 4 threads, need well over 100 us on 8, and lose at
/// 768 us on 32 and 64. Splitting lanes into pool-sized chunks lost to one-lane tasks at
/// every small pool measured, so a parallel batch keeps Rayon's own splitting.
#[cfg(not(target_arch = "wasm32"))]
const FORK_NS_PER_THREAD_SQ: u64 = 2_000;

/// Estimated nanoseconds to step one lane of `n_symbols`. A line fitted to measured
/// release-build transitions from 1 to 64 symbols, floored at the small-panel cost. It
/// only has to be right about orders of magnitude.
#[cfg(not(target_arch = "wasm32"))]
fn lane_cost_ns(n_symbols: usize) -> u64 {
    (n_symbols as u64)
        .saturating_mul(2_000)
        .saturating_sub(5_000)
        .max(1_000)
}

/// How one `step_batch` call schedules its lanes. Every variant produces the same bytes;
/// the choice only moves wall time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Dispatch {
    /// Loop on the calling thread, with no fork-join.
    Serial,
    /// Rayon with at least `min_lanes` lanes per task.
    #[cfg(not(target_arch = "wasm32"))]
    Parallel { min_lanes: NonZeroUsize },
}

/// Pick the schedule for `lanes` lanes costing `work_ns` in total on a `threads`-sized
/// pool. A batch that cannot pay for a fork-join on this pool stays serial; one that can
/// runs one lane per Rayon task, as before this planner existed.
#[cfg(not(target_arch = "wasm32"))]
fn plan_dispatch(lanes: usize, work_ns: u64, threads: usize) -> Dispatch {
    if threads <= 1 || lanes <= 1 || work_ns < fork_break_even_ns(threads) {
        return Dispatch::Serial;
    }
    Dispatch::Parallel {
        min_lanes: NonZeroUsize::MIN,
    }
}

/// The estimated work at which a fork-join on a `threads`-sized pool pays for itself.
#[cfg(not(target_arch = "wasm32"))]
fn fork_break_even_ns(threads: usize) -> u64 {
    let threads = u64::try_from(threads).unwrap_or(u64::MAX);
    threads
        .saturating_mul(threads)
        .saturating_mul(FORK_NS_PER_THREAD_SQ)
}

/// A batch of `B` independent [`TradingEnv`] lanes stepped together.
pub struct VecTradingEnv {
    envs: Vec<TradingEnv>,
    seeds: Vec<u64>,
    /// Symbols per lane, read only by the scheduling estimate. `from_envs` cannot see a
    /// built lane's panel, so its entries stay zero until `reset_batch` observes them.
    lane_symbols: Vec<usize>,
    mode: AutoresetMode,
    pending_reset: Vec<bool>,
    elapsed_steps: Vec<usize>,
    capped_final: Vec<Option<(MarketObservation, f64, bool)>>,
    episode_limit: Option<NonZeroUsize>,
}

impl VecTradingEnv {
    /// Build a batch from per-lane configs. Each lane is an independent synthetic env.
    pub fn from_configs(configs: &[LaneConfig]) -> Self {
        let envs: Vec<TradingEnv> = configs.iter().map(LaneConfig::build).collect();
        let seeds = configs.iter().map(|c| c.seed).collect();
        let lane_symbols = configs.iter().map(|c| c.n_symbols).collect();
        let pending_reset = vec![false; envs.len()];
        let elapsed_steps = vec![0; envs.len()];
        let capped_final = vec![None; envs.len()];
        VecTradingEnv {
            envs,
            seeds,
            lane_symbols,
            mode: AutoresetMode::default(),
            pending_reset,
            elapsed_steps,
            capped_final,
            episode_limit: None,
        }
    }

    /// Build a batch from pre-constructed lanes (the most general entry point). `seeds`
    /// is out-of-band provenance threaded into `info`; it must match `envs` in length.
    pub fn from_envs(envs: Vec<TradingEnv>, seeds: Vec<u64>) -> Self {
        assert_eq!(
            envs.len(),
            seeds.len(),
            "envs and seeds must have equal length"
        );
        let lane_symbols = vec![0; envs.len()];
        let pending_reset = vec![false; envs.len()];
        let elapsed_steps = vec![0; envs.len()];
        let capped_final = vec![None; envs.len()];
        VecTradingEnv {
            envs,
            seeds,
            lane_symbols,
            mode: AutoresetMode::default(),
            pending_reset,
            elapsed_steps,
            capped_final,
            episode_limit: None,
        }
    }

    /// Select the auto-reset mode (builder form). Defaults to [`AutoresetMode::NextStep`].
    pub fn with_autoreset_mode(mut self, mode: AutoresetMode) -> Self {
        self.mode = mode;
        self
    }

    /// Cap executed steps per lane without shortening the market-data window.
    /// The positive limit is a truncation, not bankruptcy. Autoreset uses the
    /// selected mode; a disabled capped lane freezes with zero subsequent reward.
    /// Without a limit the historical native horizon is unchanged.
    pub fn with_episode_limit(mut self, limit: NonZeroUsize) -> Self {
        self.episode_limit = Some(limit);
        self
    }

    /// The auto-reset mode this batch applies to finished lanes.
    pub fn autoreset_mode(&self) -> AutoresetMode {
        self.mode
    }

    /// The number of lanes (`B`).
    pub fn len(&self) -> usize {
        self.envs.len()
    }

    /// Whether the batch has zero lanes.
    pub fn is_empty(&self) -> bool {
        self.envs.is_empty()
    }

    /// The per-lane generating seeds (out-of-band provenance, never a feature).
    pub fn seeds(&self) -> &[u64] {
        &self.seeds
    }

    /// Reset every lane to the start of its window; return each lane's first
    /// observation (in lane order). Clears any pending deferred reset.
    pub fn reset_batch(&mut self) -> Vec<MarketObservation> {
        self.pending_reset.iter_mut().for_each(|p| *p = false);
        self.elapsed_steps.fill(0);
        self.capped_final.fill(None);
        let observations: Vec<MarketObservation> =
            self.envs.iter_mut().map(TradingEnv::reset).collect();
        self.lane_symbols = observations.iter().map(|o| o.symbols.len()).collect();
        observations
    }

    /// Step every lane with its decision (`decisions[i]` drives lane `i`) and apply
    /// the selected auto-reset mode. `decisions.len()` must equal the lane count.
    pub fn step_batch(&mut self, decisions: &[Decision]) -> BatchStep {
        self.step_batch_with(decisions, self.dispatch())
    }

    /// The schedule the next `step_batch` will use on the ambient Rayon pool.
    #[cfg(not(target_arch = "wasm32"))]
    fn dispatch(&self) -> Dispatch {
        let work_ns = self
            .lane_symbols
            .iter()
            .fold(0u64, |sum, &s| sum.saturating_add(lane_cost_ns(s)));
        plan_dispatch(self.envs.len(), work_ns, rayon::current_num_threads())
    }

    /// The schedule on `wasm32`, where there are no threads to dispatch to.
    #[cfg(target_arch = "wasm32")]
    fn dispatch(&self) -> Dispatch {
        Dispatch::Serial
    }

    fn step_batch_with(&mut self, decisions: &[Decision], dispatch: Dispatch) -> BatchStep {
        assert_eq!(
            decisions.len(),
            self.envs.len(),
            "decisions length must equal the lane count"
        );

        let mode = self.mode;
        let limit = self.episode_limit;

        let outcomes: Vec<LaneOutcome> = match dispatch {
            Dispatch::Serial => self
                .envs
                .iter_mut()
                .zip(decisions.iter())
                .zip(self.pending_reset.iter_mut())
                .zip(self.elapsed_steps.iter_mut())
                .zip(self.capped_final.iter_mut())
                .map(|((((env, decision), pending), elapsed), final_state)| {
                    step_lane(env, decision, pending, mode, elapsed, final_state, limit)
                })
                .collect(),
            #[cfg(not(target_arch = "wasm32"))]
            Dispatch::Parallel { min_lanes } => self
                .envs
                .par_iter_mut()
                .zip(decisions.par_iter())
                .zip(self.pending_reset.par_iter_mut())
                .zip(self.elapsed_steps.par_iter_mut())
                .zip(self.capped_final.par_iter_mut())
                .with_min_len(min_lanes.get())
                .map(|((((env, decision), pending), elapsed), final_state)| {
                    step_lane(env, decision, pending, mode, elapsed, final_state, limit)
                })
                .collect(),
        };

        BatchStep::from_outcomes(outcomes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Action, Order, TradingEnv as ScalarEnv};

    fn configs(seeds: &[u64]) -> Vec<LaneConfig> {
        seeds.iter().map(|&s| LaneConfig::new(4, 60, s)).collect()
    }

    /// A flat target-weight decision over a known symbol axis — deterministic so the
    /// engine math (not the agent) is what the tests compare.
    fn decision_for(obs: &MarketObservation, weight: f64) -> Decision {
        let orders = obs
            .symbols
            .iter()
            .map(|s| Order {
                symbol: s.symbol.clone(),
                action: if weight > 0.0 {
                    Action::Buy
                } else {
                    Action::Hold
                },
                target_weight: weight,
                confidence: Some(0.5),
                rationale: String::new(),
            })
            .collect();
        Decision {
            orders,
            reasoning: String::new(),
            cost: None,
        }
    }

    fn decisions_for(obs: &[MarketObservation], weight: f64) -> Vec<Decision> {
        obs.iter().map(|o| decision_for(o, weight)).collect()
    }

    #[test]
    fn b1_is_byte_identical_to_scalar_engine() {
        let seed = 11;
        let mut scalar = ScalarEnv::new(
            Dataset::synthetic(4, 60, seed),
            Window { start: 0, end: 60 },
            CostModel::default(),
            seed,
        );
        let mut batch = VecTradingEnv::from_configs(&configs(&[seed]))
            .with_autoreset_mode(AutoresetMode::SameStep);

        let mut s_obs = scalar.reset();
        let b_obs = batch.reset_batch();
        assert_eq!(
            serde_json::to_string(&s_obs).unwrap(),
            serde_json::to_string(&b_obs[0]).unwrap(),
            "reset observation must match the scalar env byte-for-byte"
        );

        // Step in lockstep over the whole episode; the engine math (reward, NAV, events,
        // and every non-terminal observation) must be byte-identical to the scalar env.
        loop {
            let s_dec = decision_for(&s_obs, 0.25);
            let b_dec = decisions_for(std::slice::from_ref(&s_obs), 0.25);
            let s_res = scalar.step(s_dec);
            let b_res = batch.step_batch(&b_dec);

            assert_eq!(s_res.reward, b_res.rewards[0], "reward divergence");
            assert_eq!(s_res.info.nav, b_res.infos[0].nav, "nav divergence");
            assert_eq!(
                s_res.info.events, b_res.infos[0].events,
                "per-step events divergence"
            );
            assert_eq!(s_res.done, b_res.truncated[0], "truncation divergence");

            if s_res.done {
                // The scalar env clamps to the terminal bar; the batch auto-resets and
                // surfaces a fresh t0 — the one intentional, documented difference.
                assert!(b_res.first[0], "finished lane must flag first=true");
                break;
            }
            assert!(!b_res.first[0], "mid-episode lane must not flag first");
            assert_eq!(
                serde_json::to_string(&s_res.observation).unwrap(),
                serde_json::to_string(&b_res.observations[0]).unwrap(),
                "non-terminal observation must match the scalar env byte-for-byte"
            );
            s_obs = s_res.observation;
        }
    }

    #[test]
    fn parallel_matches_serial() {
        let cfgs = configs(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let mut par = VecTradingEnv::from_configs(&cfgs);
        let mut ser = VecTradingEnv::from_configs(&cfgs);

        let mut par_obs = par.reset_batch();
        let _ = ser.reset_batch();

        for _ in 0..120 {
            let decs = decisions_for(&par_obs, 0.5);
            let par_step = par.step_batch(&decs);

            // The serial reference: the same per-lane work, looped, no rayon. Both
            // batches use the default mode, so the serial loop mirrors it exactly.
            let ser_step = BatchStep::from_outcomes(
                ser.envs
                    .iter_mut()
                    .zip(decs.iter())
                    .zip(ser.pending_reset.iter_mut())
                    .zip(ser.elapsed_steps.iter_mut())
                    .zip(ser.capped_final.iter_mut())
                    .map(|((((env, decision), pending), elapsed), final_state)| {
                        step_lane(
                            env,
                            decision,
                            pending,
                            AutoresetMode::NextStep,
                            elapsed,
                            final_state,
                            None,
                        )
                    })
                    .collect(),
            );

            assert_eq!(par_step.rewards, ser_step.rewards);
            assert_eq!(par_step.terminated, ser_step.terminated);
            assert_eq!(par_step.truncated, ser_step.truncated);
            assert_eq!(par_step.first, ser_step.first);
            for (a, b) in par_step.infos.iter().zip(ser_step.infos.iter()) {
                assert_eq!(a.nav, b.nav);
                assert_eq!(a.events, b.events);
            }
            for (a, b) in par_step
                .observations
                .iter()
                .zip(ser_step.observations.iter())
            {
                assert_eq!(
                    serde_json::to_string(a).unwrap(),
                    serde_json::to_string(b).unwrap()
                );
            }
            par_obs = par_step.observations;
        }
    }

    /// Every observable byte of one batched step, floats by bit pattern.
    fn fingerprint(step: &BatchStep) -> String {
        let info = |i: &StepInfo| {
            format!(
                "{:x}/{:x}/{}",
                i.nav.to_bits(),
                i.carry.to_bits(),
                serde_json::to_string(&i.events).unwrap()
            )
        };
        format!(
            "{}|{:x?}|{:?}|{:?}|{:?}|{:?}|{}|{:?}",
            serde_json::to_string(&step.observations).unwrap(),
            step.rewards.iter().map(|r| r.to_bits()).collect::<Vec<_>>(),
            step.terminated,
            step.truncated,
            step.first,
            step.infos.iter().map(info).collect::<Vec<_>>(),
            serde_json::to_string(&step.final_obs).unwrap(),
            step.final_info
                .iter()
                .map(|i| i.as_ref().map(info))
                .collect::<Vec<_>>(),
        )
    }

    /// Lanes that differ in panel width, horizon, distribution and execution seed, so a
    /// schedule that swaps or reorders lanes cannot go unnoticed.
    fn mixed_configs() -> Vec<LaneConfig> {
        (0..9u64)
            .map(|i| LaneConfig {
                exec_seed: (i % 2 == 0).then_some(1_000 + i),
                distribution_mode: if i % 3 == 0 {
                    DistributionMode::Hard
                } else {
                    DistributionMode::Calm
                },
                ..LaneConfig::new(2 + (i as usize % 4), 22 + 3 * i as usize, 40 + i)
            })
            .collect()
    }

    /// Drive `cfgs` under `dispatch` (`None` is the public `step_batch` and its plan),
    /// across a mid-run `reset_batch`, and record each step's fingerprint.
    fn trajectory(
        cfgs: &[LaneConfig],
        mode: AutoresetMode,
        limit: Option<NonZeroUsize>,
        dispatch: Option<Dispatch>,
    ) -> Vec<String> {
        let mut env = VecTradingEnv::from_configs(cfgs).with_autoreset_mode(mode);
        if let Some(limit) = limit {
            env = env.with_episode_limit(limit);
        }
        let mut prints = Vec::new();
        for steps in [90, 30] {
            let mut obs = env.reset_batch();
            for t in 0..steps {
                let weight = [0.0, 0.3, 0.9, 0.5][t % 4];
                let decs = decisions_for(&obs, weight);
                let step = match dispatch {
                    Some(dispatch) => env.step_batch_with(&decs, dispatch),
                    None => env.step_batch(&decs),
                };
                prints.push(fingerprint(&step));
                obs = step.observations;
            }
        }
        prints
    }

    #[test]
    fn every_dispatch_produces_the_same_trajectory() {
        // A real multi-thread pool, so the parallel arms split even on a small runner.
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let cfgs = mixed_configs();
        for mode in [
            AutoresetMode::NextStep,
            AutoresetMode::SameStep,
            AutoresetMode::Disabled,
        ] {
            for limit in [None, NonZeroUsize::new(7)] {
                let serial = trajectory(&cfgs, mode, limit, Some(Dispatch::Serial));
                for min_lanes in [1, 2, 4, 9, 100] {
                    let dispatch = Dispatch::Parallel {
                        min_lanes: NonZeroUsize::new(min_lanes).unwrap(),
                    };
                    let parallel = pool.install(|| trajectory(&cfgs, mode, limit, Some(dispatch)));
                    assert_eq!(serial, parallel, "{mode:?} {limit:?} {dispatch:?}");
                }
                let planned = pool.install(|| trajectory(&cfgs, mode, limit, None));
                assert_eq!(serial, planned, "{mode:?} {limit:?} planned");
            }
        }
    }

    #[test]
    fn dispatch_plan_stays_serial_until_the_work_pays_for_a_fork() {
        let one = Dispatch::Parallel {
            min_lanes: NonZeroUsize::MIN,
        };
        for threads in [2, 4, 8, 16, 64] {
            let even = fork_break_even_ns(threads);
            assert_eq!(plan_dispatch(64, even - 1, threads), Dispatch::Serial);
            assert_eq!(plan_dispatch(64, even, threads), one, "{threads}");
        }
        assert_eq!(plan_dispatch(1, u64::MAX, 64), Dispatch::Serial);
        assert_eq!(plan_dispatch(0, 0, 64), Dispatch::Serial);
        assert_eq!(plan_dispatch(512, u64::MAX, 1), Dispatch::Serial);
        assert_eq!(plan_dispatch(usize::MAX, u64::MAX, usize::MAX), one);
    }

    #[test]
    fn break_even_grows_with_the_square_of_the_pool() {
        assert_eq!(fork_break_even_ns(1), 2_000);
        assert_eq!(fork_break_even_ns(4), 32_000);
        assert_eq!(fork_break_even_ns(8), 128_000);
        assert_eq!(fork_break_even_ns(64), 8_192_000);
        assert_eq!(fork_break_even_ns(usize::MAX), u64::MAX);
    }

    #[test]
    fn lane_cost_grows_with_the_panel_and_keeps_a_floor() {
        assert_eq!(lane_cost_ns(0), 1_000);
        assert_eq!(lane_cost_ns(1), 1_000);
        assert_eq!(lane_cost_ns(8), 11_000);
        assert_eq!(lane_cost_ns(64), 123_000);
        assert_eq!(lane_cost_ns(usize::MAX), u64::MAX - 5_000);
    }

    #[test]
    fn dispatch_reads_the_pool_and_the_lane_panels() {
        let pool = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
        };
        let (small, large) = (pool(4), pool(64));
        // The INT-13 workload at 256 lanes: worth a fork on 4 threads, not on 64.
        let narrow = VecTradingEnv::from_configs(&vec![LaneConfig::new(4, 30, 1); 256]);
        assert!(matches!(
            small.install(|| narrow.dispatch()),
            Dispatch::Parallel { .. }
        ));
        assert_eq!(large.install(|| narrow.dispatch()), Dispatch::Serial);
        // Same lane count, wide panels: the work pays for a fork on 64 threads too.
        let wide = VecTradingEnv::from_configs(&vec![LaneConfig::new(64, 30, 1); 256]);
        assert!(matches!(
            large.install(|| wide.dispatch()),
            Dispatch::Parallel { .. }
        ));
        // `from_envs` cannot see a panel until the first reset reports it.
        let envs: Vec<TradingEnv> = (0..4).map(|_| LaneConfig::new(64, 30, 1).build()).collect();
        let mut built = VecTradingEnv::from_envs(envs, vec![1; 4]);
        assert_eq!(small.install(|| built.dispatch()), Dispatch::Serial);
        built.reset_batch();
        assert!(matches!(
            small.install(|| built.dispatch()),
            Dispatch::Parallel { .. }
        ));
    }

    #[test]
    fn episode_limit_counts_each_lane_and_resets_counters() {
        for mode in [
            AutoresetMode::NextStep,
            AutoresetMode::SameStep,
            AutoresetMode::Disabled,
        ] {
            let mut env = VecTradingEnv::from_configs(&[
                LaneConfig::new(2, 1, 11),
                LaneConfig::new(2, 10, 29),
            ])
            .with_autoreset_mode(mode)
            .with_episode_limit(NonZeroUsize::new(2).unwrap());
            let initial = env.reset_batch();
            let decisions = decisions_for(&initial, 0.0);
            let first = env.step_batch(&decisions);
            assert_eq!(first.truncated, [true, false]);
            let second = env.step_batch(&decisions);
            assert!(second.truncated[1]);
            assert!(!second.terminated[1]);
            if mode == AutoresetMode::SameStep {
                assert!(second.first[1]);
                assert!(second.final_obs[1].is_some());
                assert_eq!(env.elapsed_steps, [0, 0]);
            } else if mode == AutoresetMode::NextStep {
                assert!(!second.first[1]);
                let reset = env.step_batch(&decisions);
                assert!(reset.first[1]);
                assert!(!reset.truncated[1]);
                assert_eq!(reset.rewards[1], 0.0);
                assert_eq!(env.elapsed_steps[1], 0);
            } else {
                let frozen = env.step_batch(&decisions);
                assert!(frozen.truncated[1]);
                assert_eq!(frozen.rewards[1], 0.0);
                assert_eq!(
                    serde_json::to_string(&second.observations[1]).unwrap(),
                    serde_json::to_string(&frozen.observations[1]).unwrap()
                );
            }
            env.reset_batch();
            assert_eq!(env.elapsed_steps, [0, 0]);
            assert!(env.capped_final.iter().all(Option::is_none));
            assert!(!env.step_batch(&decisions).truncated[1]);
        }
    }

    #[test]
    fn auto_reset_keeps_the_batch_running() {
        // Two short lanes that exhaust their windows quickly; after each finishes the
        // batch must keep producing observations and flag `first` on the reset step.
        let cfgs = vec![LaneConfig::new(3, 25, 1), LaneConfig::new(3, 25, 2)];
        let mut env = VecTradingEnv::from_configs(&cfgs);
        let mut obs = env.reset_batch();

        let mut resets = [0usize; 2];
        for _ in 0..120 {
            let decs = decisions_for(&obs, 0.1);
            let step = env.step_batch(&decs);
            assert_eq!(step.observations.len(), 2, "batch never stalls");
            for (lane, &first) in step.first.iter().enumerate() {
                if first {
                    resets[lane] += 1;
                    // A reset lane's observation is a fresh t0 — its NAV is back to the
                    // starting capital (never the blown-up / terminal value).
                    assert!(step.infos[lane].nav.is_finite());
                }
            }
            obs = step.observations;
        }
        assert!(
            resets[0] > 1 && resets[1] > 1,
            "each lane should auto-reset multiple times over 120 steps, got {resets:?}"
        );
    }

    #[test]
    fn auto_reset_observation_is_a_fresh_t0() {
        // A finished lane's reset observation must equal a brand-new env's reset obs
        // for the same seed (no leak of the prior episode tail).
        let seed = 7;
        let mut env = VecTradingEnv::from_configs(&[LaneConfig::new(3, 22, seed)]);
        let mut obs = env.reset_batch();
        let reference = {
            let mut fresh = VecTradingEnv::from_configs(&[LaneConfig::new(3, 22, seed)]);
            serde_json::to_string(&fresh.reset_batch()[0]).unwrap()
        };

        loop {
            let decs = decisions_for(&obs, 0.0);
            let step = env.step_batch(&decs);
            if step.first[0] {
                assert_eq!(
                    serde_json::to_string(&step.observations[0]).unwrap(),
                    reference,
                    "reset observation must equal a fresh env's t0"
                );
                break;
            }
            obs = step.observations;
        }
    }

    fn fresh_t0(seed: u64, n_symbols: usize, n_days: usize) -> String {
        let mut env = VecTradingEnv::from_configs(&[LaneConfig::new(n_symbols, n_days, seed)]);
        serde_json::to_string(&env.reset_batch()[0]).unwrap()
    }

    #[test]
    fn next_step_defers_reset_to_following_step() {
        // The default mode: a finished lane returns its terminal step verbatim, and the
        // reset surfaces on the *next* step (reward 0, flags false, first=true).
        let t0 = fresh_t0(7, 3, 22);
        let mut env = VecTradingEnv::from_configs(&[LaneConfig::new(3, 22, 7)]);
        assert_eq!(env.autoreset_mode(), AutoresetMode::NextStep);
        let mut obs = env.reset_batch();
        loop {
            let step = env.step_batch(&decisions_for(&obs, 0.0));
            if step.truncated[0] || step.terminated[0] {
                assert!(
                    !step.first[0],
                    "next_step must not reset on the ending step"
                );
                assert!(step.final_obs[0].is_none());
                assert_ne!(
                    serde_json::to_string(&step.observations[0]).unwrap(),
                    t0,
                    "ending step returns the terminal obs, not a fresh t0"
                );
                let next = env.step_batch(&decisions_for(&step.observations, 0.0));
                assert!(next.first[0], "next_step resets on the following step");
                assert_eq!(next.rewards[0], 0.0);
                assert!(!next.terminated[0] && !next.truncated[0]);
                assert_eq!(
                    serde_json::to_string(&next.observations[0]).unwrap(),
                    t0,
                    "the deferred-reset obs must equal a fresh t0"
                );
                break;
            }
            obs = step.observations;
        }
    }

    #[test]
    fn same_step_surfaces_final_obs_and_info() {
        // same_step resets in place; the terminal obs/info ride along in final_obs/info.
        let t0 = fresh_t0(7, 3, 22);
        let mut env = VecTradingEnv::from_configs(&[LaneConfig::new(3, 22, 7)])
            .with_autoreset_mode(AutoresetMode::SameStep);
        let mut obs = env.reset_batch();
        loop {
            let step = env.step_batch(&decisions_for(&obs, 0.0));
            if step.first[0] {
                assert_eq!(
                    serde_json::to_string(&step.observations[0]).unwrap(),
                    t0,
                    "same_step primary obs is the fresh t0"
                );
                let final_obs = step.final_obs[0].as_ref().expect("final_obs present");
                assert_ne!(
                    serde_json::to_string(final_obs).unwrap(),
                    t0,
                    "final_obs is the terminal obs, not t0"
                );
                assert!(step.final_info[0].is_some(), "final_info present");
                break;
            }
            assert!(step.final_obs[0].is_none());
            obs = step.observations;
        }
    }

    #[test]
    fn disabled_never_resets() {
        let mut env = VecTradingEnv::from_configs(&[LaneConfig::new(3, 22, 7)])
            .with_autoreset_mode(AutoresetMode::Disabled);
        let mut obs = env.reset_batch();
        let mut ended = false;
        for _ in 0..80 {
            let step = env.step_batch(&decisions_for(&obs, 0.0));
            ended |= step.truncated[0];
            assert!(!step.first[0], "disabled never flags first");
            assert!(step.final_obs[0].is_none());
            obs = step.observations;
        }
        assert!(ended, "the lane should exhaust its window within 80 steps");
    }

    #[test]
    fn split_exec_seed_keeps_dataset_changes_noise() {
        // Same scenario seed ⇒ identical price path; distinct exec_seed ⇒ distinct
        // execution noise, so realized rewards diverge once the lane trades.
        let base = LaneConfig::new(3, 60, 5);
        let cfg_a = LaneConfig {
            exec_seed: Some(1000),
            ..base.clone()
        };
        let cfg_b = LaneConfig {
            exec_seed: Some(2000),
            ..base
        };
        let mut a = VecTradingEnv::from_configs(std::slice::from_ref(&cfg_a));
        let mut b = VecTradingEnv::from_configs(std::slice::from_ref(&cfg_b));
        let mut obs_a = a.reset_batch();
        let mut obs_b = b.reset_batch();
        assert_eq!(
            serde_json::to_string(&obs_a[0]).unwrap(),
            serde_json::to_string(&obs_b[0]).unwrap(),
            "identical scenario seed ⇒ identical t0 price path"
        );
        let mut rewards_a = Vec::new();
        let mut rewards_b = Vec::new();
        for _ in 0..30 {
            let sa = a.step_batch(&decisions_for(&obs_a, 0.5));
            let sb = b.step_batch(&decisions_for(&obs_b, 0.5));
            rewards_a.push(sa.rewards[0]);
            rewards_b.push(sb.rewards[0]);
            obs_a = sa.observations;
            obs_b = sb.observations;
        }
        assert_ne!(
            rewards_a, rewards_b,
            "distinct exec seeds must perturb execution noise"
        );
    }

    #[test]
    fn hard_distribution_diverges_from_calm() {
        // The amplify transform preserves the first bar, so t0 matches; divergence shows
        // up along the path. Compare reward trajectories over the same scenario seed.
        let calm = LaneConfig::new(3, 60, 9);
        let hard = LaneConfig {
            distribution_mode: DistributionMode::Hard,
            ..calm.clone()
        };
        let mut c = VecTradingEnv::from_configs(std::slice::from_ref(&calm));
        let mut h = VecTradingEnv::from_configs(std::slice::from_ref(&hard));
        let mut obs_c = c.reset_batch();
        let mut obs_h = h.reset_batch();
        let mut rewards_c = Vec::new();
        let mut rewards_h = Vec::new();
        for _ in 0..40 {
            let sc = c.step_batch(&decisions_for(&obs_c, 0.5));
            let sh = h.step_batch(&decisions_for(&obs_h, 0.5));
            rewards_c.push(sc.rewards[0]);
            rewards_h.push(sh.rewards[0]);
            obs_c = sc.observations;
            obs_h = sh.observations;
        }
        assert_ne!(
            rewards_c, rewards_h,
            "hard tier must post-process the panel away from calm"
        );
    }
}
