//! pyo3 bindings for SharpeArena — the leak-free, point-in-time trading-agent
//! environment. The binding exchanges the wire-contract JSON at the boundary
//! (observations and decisions are JSON strings), which keeps the surface robust
//! and identical to the language-agnostic protocol any external agent speaks.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyType;
use pyo3::wrap_pyfunction;
use sharpearena::exec_noise::{perturb as core_perturb_action, ExecNoise};
use sharpearena::leaderboard_ci::{
    bootstrap_dsr_ci as core_bootstrap_dsr_ci, paired_dsr_diff as core_paired_dsr_diff,
    KERNEL_BASE_TRIALS, TRIALS_SR_STD_DEFAULT,
};
use sharpearena::lob_market::{OrderBook, OrderKind, Side};
use sharpearena::market::{EllipticUncertaintySet, MarketClearing, MarketParams};
use sharpearena::vec_env::AutoresetMode;
use sharpearena::{
    generate_scenario, CostModel, Dataset, Decision, DistributionMode, LaneConfig, Mandate,
    RichnessTier, ScenarioSpec, TradingEnv as CoreEnv, VecTradingEnv as CoreVecEnv, Window,
};
use sharpebench_core::{score_agent, AgentSubmission, Run, ScoreConfig, Trace};

/// Parse the wire `distribution_mode` label, rejecting unknown tiers with a `ValueError`.
fn parse_distribution_mode(mode: &str) -> PyResult<DistributionMode> {
    match mode {
        "calm" => Ok(DistributionMode::Calm),
        "hard" => Ok(DistributionMode::Hard),
        "extreme" => Ok(DistributionMode::Extreme),
        "cointegrated_pairs" => Ok(DistributionMode::CointegratedPairs),
        "regime_shift" => Ok(DistributionMode::RegimeShift),
        other => Err(PyValueError::new_err(format!(
            "unknown distribution_mode {other:?} (expected calm | hard | extreme | \
             cointegrated_pairs | regime_shift)"
        ))),
    }
}

/// Parse the wire `richness` label into a [`RichnessTier`], the information-disclosure
/// difficulty axis orthogonal to `distribution_mode`. `standard` is the historical default
/// disclosure, so an unset richness reproduces the prior observations byte-for-byte.
fn parse_richness_tier(richness: &str) -> PyResult<RichnessTier> {
    match richness {
        "data_poor" => Ok(RichnessTier::DataPoor),
        "standard" => Ok(RichnessTier::Standard),
        "data_rich" => Ok(RichnessTier::DataRich),
        other => Err(PyValueError::new_err(format!(
            "unknown richness {other:?} (expected data_poor | standard | data_rich)"
        ))),
    }
}

/// Build the synthetic dataset for a tier: `Calm` is the mild panel; `Hard`/`Extreme`
/// post-process that same seeded panel (see `sharpearena::generate_scenario`).
/// `vol_clustering > 0.0` opts into the volatility-clustering post-pass; at `0.0`
/// (the default) the output is byte-identical to the historical build.
#[allow(clippy::too_many_arguments)]
fn build_dataset(
    n_symbols: usize,
    n_days: usize,
    seed: u64,
    mode: DistributionMode,
    vol_clustering: f64,
    jump_burst_probability: f64,
    jump_burst_persistence: f64,
    jump_burst_size: f64,
) -> Dataset {
    match mode {
        DistributionMode::Calm
            if vol_clustering == 0.0
                && jump_burst_probability == 0.0
                && jump_burst_persistence == 0.0
                && jump_burst_size == 0.0 =>
        {
            Dataset::synthetic(n_symbols, n_days, seed)
        }
        m => generate_scenario(
            &ScenarioSpec {
                n_symbols,
                n_days,
                distribution_mode: m,
                vol_clustering,
                jump_burst_probability,
                jump_burst_persistence,
                jump_burst_size,
                ..ScenarioSpec::default()
            },
            seed,
        ),
    }
}

/// Dependency-free FNV-1a/64 over bytes — the fingerprint the crate's cross-runtime
/// scenario goldens already use, reused here so a readback and a golden pin name the
/// same number for the same panel.
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// The configuration an environment is **actually** running, read out of what the
/// constructor arguments produced rather than echoed from those arguments: the panel
/// dimensions come from the built [`Dataset`], the window from the clamped [`Window`],
/// the execution seed from the value handed to the engine after default resolution, and
/// `dataset_fnv1a64` fingerprints the generated tape itself.
///
/// The scenario-generation knobs (seed, tier, clustering, jump bursts) are deliberately
/// not repeated here: echoing them back would prove nothing. They are *bound* by the
/// fingerprint instead. A caller regenerates the panel from the configuration it wrote
/// into its own results and compares the two numbers, which is what catches a harness
/// bug that pointed the environment at a different tape than the label claims — an
/// inversion every other guarantee (fixed seeds, goldens, provenance digests) survives,
/// because the wrong tape is still produced reproducibly.
fn effective_config_json(data: &Dataset, window: &Window, exec_seed: u64) -> PyResult<String> {
    let bytes = serde_json::to_string(data).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(serde_json::json!({
        "n_symbols": data.symbols().len(),
        "n_bars": data.len(),
        "window_start": window.start,
        "window_end": window.end,
        "exec_seed": exec_seed,
        "dataset_fnv1a64": format!("{:016x}", fnv1a64(bytes.as_bytes())),
    })
    .to_string())
}

/// Parse the wire `autoreset_mode` label, rejecting unknown modes with a `ValueError`.
fn parse_autoreset_mode(mode: &str) -> PyResult<AutoresetMode> {
    AutoresetMode::from_label(mode).ok_or_else(|| {
        PyValueError::new_err(format!(
            "unknown autoreset_mode {mode:?} (expected next_step | same_step | disabled)"
        ))
    })
}

/// A Gym-style, steppable, leak-free trading environment.
///
/// Construct over a deterministic synthetic dataset (default) or, via the
/// [`from_csv`](Self::from_csv) classmethod, over a frozen long-format CSV.
/// `reset` returns the first observation as a JSON string; `step` takes a
/// decision JSON string and returns `(observation_json, reward, done, info_json)`.
#[pyclass(name = "TradingEnv")]
pub struct PyTradingEnv {
    inner: CoreEnv,
    seed: u64,
    effective: String,
}

fn build_window(start: Option<usize>, end: Option<usize>, len: usize) -> Window {
    Window {
        start: start.unwrap_or(0),
        end: end.unwrap_or(len),
    }
}

fn build_costs(
    fee_bps: Option<f64>,
    slippage_bps: Option<f64>,
    impact_bps: Option<f64>,
    financing_bps: Option<f64>,
    max_participation: Option<f64>,
) -> CostModel {
    let d = CostModel::default();
    CostModel {
        fee_bps: fee_bps.unwrap_or(d.fee_bps),
        slippage_bps: slippage_bps.unwrap_or(d.slippage_bps),
        impact_bps: impact_bps.unwrap_or(d.impact_bps),
        financing_bps: financing_bps.unwrap_or(d.financing_bps),
        max_participation: max_participation.unwrap_or(d.max_participation),
        trf_cost: d.trf_cost,
        noise: d.noise,
    }
}

#[pymethods]
impl PyTradingEnv {
    /// Build an environment over a synthetic dataset of `n_symbols` × `n_days`,
    /// seeded by `seed`. The window defaults to the full series; costs default to
    /// [`CostModel::default`] unless overridden.
    #[new]
    #[pyo3(signature = (
        n_symbols = 4,
        n_days = 120,
        seed = 0,
        window_start = None,
        window_end = None,
        fee_bps = None,
        slippage_bps = None,
        impact_bps = None,
        financing_bps = None,
        max_participation = None,
        distribution_mode = "calm",
        exec_seed = None,
        vol_clustering = 0.0,
        jump_burst_probability = 0.0,
        jump_burst_persistence = 0.0,
        jump_burst_size = 0.0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_symbols: usize,
        n_days: usize,
        seed: u64,
        window_start: Option<usize>,
        window_end: Option<usize>,
        fee_bps: Option<f64>,
        slippage_bps: Option<f64>,
        impact_bps: Option<f64>,
        financing_bps: Option<f64>,
        max_participation: Option<f64>,
        distribution_mode: &str,
        exec_seed: Option<u64>,
        vol_clustering: f64,
        jump_burst_probability: f64,
        jump_burst_persistence: f64,
        jump_burst_size: f64,
    ) -> PyResult<Self> {
        let mode = parse_distribution_mode(distribution_mode)?;
        let data = build_dataset(
            n_symbols,
            n_days,
            seed,
            mode,
            vol_clustering,
            jump_burst_probability,
            jump_burst_persistence,
            jump_burst_size,
        );
        let window = build_window(window_start, window_end, data.len());
        if window.start >= window.end || window.end > data.len() {
            return Err(PyValueError::new_err(format!(
                "invalid window [{}, {}) over {} bars",
                window.start,
                window.end,
                data.len()
            )));
        }
        let costs = build_costs(
            fee_bps,
            slippage_bps,
            impact_bps,
            financing_bps,
            max_participation,
        );
        let resolved_exec_seed = exec_seed.unwrap_or(seed);
        let effective = effective_config_json(&data, &window, resolved_exec_seed)?;
        Ok(PyTradingEnv {
            inner: CoreEnv::new(data, window, costs, resolved_exec_seed),
            seed,
            effective,
        })
    }

    /// Build an environment over a frozen long-format CSV (`date,symbol,close[,dividend]`).
    #[classmethod]
    #[pyo3(signature = (
        csv_text,
        seed = 0,
        window_start = None,
        window_end = None,
        fee_bps = None,
        slippage_bps = None,
        impact_bps = None,
        financing_bps = None,
        max_participation = None,
        exec_seed = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn from_csv(
        _cls: &Bound<'_, PyType>,
        csv_text: &str,
        seed: u64,
        window_start: Option<usize>,
        window_end: Option<usize>,
        fee_bps: Option<f64>,
        slippage_bps: Option<f64>,
        impact_bps: Option<f64>,
        financing_bps: Option<f64>,
        max_participation: Option<f64>,
        exec_seed: Option<u64>,
    ) -> PyResult<Self> {
        let data = Dataset::from_csv(csv_text).map_err(PyValueError::new_err)?;
        let window = build_window(window_start, window_end, data.len());
        if window.start >= window.end || window.end > data.len() {
            return Err(PyValueError::new_err(format!(
                "invalid window [{}, {}) over {} bars",
                window.start,
                window.end,
                data.len()
            )));
        }
        let costs = build_costs(
            fee_bps,
            slippage_bps,
            impact_bps,
            financing_bps,
            max_participation,
        );
        let resolved_exec_seed = exec_seed.unwrap_or(seed);
        let effective = effective_config_json(&data, &window, resolved_exec_seed)?;
        Ok(PyTradingEnv {
            inner: CoreEnv::new(data, window, costs, resolved_exec_seed),
            seed,
            effective,
        })
    }

    /// The seed that generated this environment's scenario — out-of-band provenance,
    /// never a feature in the observation. The Python wrapper threads this into the
    /// `info` dict so a trajectory can be tied back to its generating seed.
    #[getter]
    fn scenario_seed(&self) -> u64 {
        self.seed
    }

    /// The effective configuration as JSON, read back from what the constructor actually
    /// built (see [`effective_config_json`]). A runner compares this against what it
    /// requested and fails the arm when the two disagree.
    #[getter]
    fn effective_config(&self) -> &str {
        &self.effective
    }

    /// Reset to the start of the window; return the first point-in-time
    /// observation as a wire-format JSON string.
    fn reset(&mut self) -> PyResult<String> {
        let obs = self.inner.reset();
        serde_json::to_string(&obs).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Apply `decision_json` (a wire-format `Decision`) at the current bar and
    /// advance one step. Returns `(observation_json, reward, done, info_json)`,
    /// where `info_json` carries the post-step NAV and this step's process events.
    fn step(&mut self, decision_json: &str) -> PyResult<(String, f64, bool, String)> {
        let decision: Decision = serde_json::from_str(decision_json)
            .map_err(|e| PyValueError::new_err(format!("invalid decision JSON: {e}")))?;
        let res = self.inner.step(decision);
        let observation = serde_json::to_string(&res.observation)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let info = serde_json::json!({
            "nav": res.info.nav,
            "events": res.info.events,
        });
        Ok((observation, res.reward, res.done, info.to_string()))
    }

    /// An O(1) snapshot of the mutable sim state (cursor + book) as JSON — the native
    /// checkpoint that replaces replay-from-decisions. Pair with [`restore_state`].
    fn clone_state(&self) -> PyResult<String> {
        let state = self.inner.clone_state();
        serde_json::to_string(&state).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Restore the env to a snapshot produced by [`clone_state`] in O(1) (no replay).
    fn restore_state(&mut self, state_json: &str) -> PyResult<()> {
        let state: sharpearena::EnvState = serde_json::from_str(state_json)
            .map_err(|e| PyValueError::new_err(format!("invalid env state: {e}")))?;
        self.inner.restore_state(state);
        Ok(())
    }
}

/// A vectorized, batched bank of `B` independent [`PyTradingEnv`]-equivalent lanes —
/// gym3's "vectorized-first" design. Each lane is a distinct synthetic scenario (one
/// seed per lane); the batch steps them together and recycles finished lanes per the
/// selected `autoreset_mode` (`next_step` default | `same_step` | `disabled`), so the
/// batch never stalls. Returns **structure-of-arrays batched JSON** (one call per batch,
/// not `B` separate calls).
#[pyclass(name = "VecTradingEnv")]
pub struct PyVecTradingEnv {
    inner: CoreVecEnv,
}

fn observations_to_json(
    observations: &[sharpearena::MarketObservation],
) -> PyResult<Vec<serde_json::Value>> {
    observations
        .iter()
        .map(|o| serde_json::to_value(o).map_err(|e| PyValueError::new_err(e.to_string())))
        .collect()
}

#[pymethods]
impl PyVecTradingEnv {
    /// Build a batch over `seeds` (one synthetic `n_symbols` × `n_days` lane per seed).
    /// All lanes share the window and cost overrides; window defaults to the full
    /// series and costs to [`CostModel::default`].
    #[new]
    #[pyo3(signature = (
        seeds,
        n_symbols = 4,
        n_days = 120,
        window_start = None,
        window_end = None,
        fee_bps = None,
        slippage_bps = None,
        impact_bps = None,
        financing_bps = None,
        max_participation = None,
        distribution_mode = "calm",
        exec_seed = None,
        autoreset_mode = "next_step",
        vol_clustering = 0.0,
        jump_burst_probability = 0.0,
        jump_burst_persistence = 0.0,
        jump_burst_size = 0.0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        seeds: Vec<u64>,
        n_symbols: usize,
        n_days: usize,
        window_start: Option<usize>,
        window_end: Option<usize>,
        fee_bps: Option<f64>,
        slippage_bps: Option<f64>,
        impact_bps: Option<f64>,
        financing_bps: Option<f64>,
        max_participation: Option<f64>,
        distribution_mode: &str,
        exec_seed: Option<u64>,
        autoreset_mode: &str,
        vol_clustering: f64,
        jump_burst_probability: f64,
        jump_burst_persistence: f64,
        jump_burst_size: f64,
    ) -> PyResult<Self> {
        if seeds.is_empty() {
            return Err(PyValueError::new_err("seeds must be non-empty"));
        }
        let mode = parse_distribution_mode(distribution_mode)?;
        let reset_mode = parse_autoreset_mode(autoreset_mode)?;
        let window = match (window_start, window_end) {
            (None, None) => None,
            _ => {
                let w = Window {
                    start: window_start.unwrap_or(0),
                    end: window_end.unwrap_or(n_days),
                };
                if w.start >= w.end || w.end > n_days {
                    return Err(PyValueError::new_err(format!(
                        "invalid window [{}, {}) over {} bars",
                        w.start, w.end, n_days
                    )));
                }
                Some(w)
            }
        };
        let costs = build_costs(
            fee_bps,
            slippage_bps,
            impact_bps,
            financing_bps,
            max_participation,
        );
        let configs: Vec<LaneConfig> = seeds
            .iter()
            .map(|&seed| LaneConfig {
                n_symbols,
                n_days,
                seed,
                exec_seed,
                distribution_mode: mode,
                vol_clustering,
                jump_burst_probability,
                jump_burst_persistence,
                jump_burst_size,
                window,
                costs,
            })
            .collect();
        Ok(PyVecTradingEnv {
            inner: CoreVecEnv::from_configs(&configs).with_autoreset_mode(reset_mode),
        })
    }

    /// Build a batch over one frozen long-format CSV. Every lane sees identical
    /// point-in-time market data while `seeds` select independent execution-noise
    /// streams. This is the historical-data counterpart to the synthetic constructor.
    #[classmethod]
    #[pyo3(signature = (
        csv_text,
        seeds,
        window_start = None,
        window_end = None,
        fee_bps = None,
        slippage_bps = None,
        impact_bps = None,
        financing_bps = None,
        max_participation = None,
        autoreset_mode = "next_step",
    ))]
    #[allow(clippy::too_many_arguments)]
    fn from_csv(
        _cls: &Bound<'_, PyType>,
        csv_text: &str,
        seeds: Vec<u64>,
        window_start: Option<usize>,
        window_end: Option<usize>,
        fee_bps: Option<f64>,
        slippage_bps: Option<f64>,
        impact_bps: Option<f64>,
        financing_bps: Option<f64>,
        max_participation: Option<f64>,
        autoreset_mode: &str,
    ) -> PyResult<Self> {
        if seeds.is_empty() {
            return Err(PyValueError::new_err("seeds must be non-empty"));
        }
        let data = Dataset::from_csv(csv_text).map_err(PyValueError::new_err)?;
        let window = build_window(window_start, window_end, data.len());
        if window.start >= window.end || window.end > data.len() {
            return Err(PyValueError::new_err(format!(
                "invalid window [{}, {}) over {} bars",
                window.start,
                window.end,
                data.len()
            )));
        }
        let costs = build_costs(
            fee_bps,
            slippage_bps,
            impact_bps,
            financing_bps,
            max_participation,
        );
        let reset_mode = parse_autoreset_mode(autoreset_mode)?;
        let envs = seeds
            .iter()
            .map(|&seed| CoreEnv::new(data.clone(), window, costs, seed))
            .collect();
        Ok(PyVecTradingEnv {
            inner: CoreVecEnv::from_envs(envs, seeds).with_autoreset_mode(reset_mode),
        })
    }

    /// The number of lanes (`B`).
    #[getter]
    fn num_envs(&self) -> usize {
        self.inner.len()
    }

    /// The per-lane generating seeds (out-of-band provenance, never a feature).
    #[getter]
    fn scenario_seeds(&self) -> Vec<u64> {
        self.inner.seeds().to_vec()
    }

    /// The auto-reset mode applied to finished lanes (`next_step | same_step | disabled`).
    #[getter]
    fn autoreset_mode(&self) -> &'static str {
        self.inner.autoreset_mode().label()
    }

    /// Reset every lane; return `{ "n": B, "observations": [..] }` as a JSON string.
    fn reset_batch(&mut self) -> PyResult<String> {
        let observations = observations_to_json(&self.inner.reset_batch())?;
        let out = serde_json::json!({ "n": observations.len(), "observations": observations });
        Ok(out.to_string())
    }

    /// Step every lane. `decisions_json` is a JSON array of exactly `B` wire-format
    /// `Decision`s (`decisions[i]` drives lane `i`). Returns the structure-of-arrays
    /// batch as a JSON string:
    /// `{ "n", "observations", "rewards", "terminated", "truncated", "first", "infos",
    /// "final_obs", "final_info" }`. Recycling follows `autoreset_mode`: under `same_step`
    /// a finished lane's `observations[i]` is the reset t0 (`first[i]` true) and its terminal
    /// obs/info ride in `final_obs[i]`/`final_info[i]`; under `next_step` the terminal step is
    /// returned verbatim and the reset surfaces on the following step (`final_obs`/`final_info`
    /// are `null`).
    fn step_batch(&mut self, decisions_json: &str) -> PyResult<String> {
        let decisions: Vec<Decision> = serde_json::from_str(decisions_json)
            .map_err(|e| PyValueError::new_err(format!("invalid decisions JSON: {e}")))?;
        if decisions.len() != self.inner.len() {
            return Err(PyValueError::new_err(format!(
                "expected {} decisions, got {}",
                self.inner.len(),
                decisions.len()
            )));
        }
        let step = self.inner.step_batch(&decisions);
        let observations = observations_to_json(&step.observations)?;
        let infos: Vec<serde_json::Value> = step
            .infos
            .iter()
            .map(|i| serde_json::json!({ "nav": i.nav, "events": i.events }))
            .collect();
        let mut final_obs = Vec::with_capacity(step.final_obs.len());
        for o in &step.final_obs {
            final_obs.push(match o {
                Some(obs) => {
                    serde_json::to_value(obs).map_err(|e| PyValueError::new_err(e.to_string()))?
                }
                None => serde_json::Value::Null,
            });
        }
        let final_info: Vec<serde_json::Value> = step
            .final_info
            .iter()
            .map(|i| match i {
                Some(info) => serde_json::json!({ "nav": info.nav, "events": info.events }),
                None => serde_json::Value::Null,
            })
            .collect();
        let out = serde_json::json!({
            "n": self.inner.len(),
            "observations": observations,
            "rewards": step.rewards,
            "terminated": step.terminated,
            "truncated": step.truncated,
            "first": step.first,
            "infos": infos,
            "final_obs": final_obs,
            "final_info": final_info,
        });
        Ok(out.to_string())
    }
}

/// Score a sequence of per-period returns with the **same SharpeBench kernel** the
/// benchmark uses — the real deflated Sharpe / PSR / pass^k / process verdict, not a
/// Python reimplementation. `n_trials` folds in the agent's declared in-sample search
/// budget (more search ⇒ more deflation). `periods_per_year` is explicit so a
/// historical hourly, daily, or weekly field cannot silently inherit the daily-equity
/// default. Returns the `CompositeScore` as a JSON string. This is what lets the
/// `verifiers` rubric reward be *calibrated* rather than approximate.
#[pyfunction]
#[pyo3(signature = (returns, n_trials = 0, periods_per_year = 252.0))]
fn score_run(returns: Vec<f64>, n_trials: u32, periods_per_year: f64) -> PyResult<String> {
    if !periods_per_year.is_finite() || periods_per_year <= 0.0 {
        return Err(PyValueError::new_err(
            "periods_per_year must be finite and positive",
        ));
    }
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
    let score = score_agent(
        &submission,
        &ScoreConfig::for_periods_per_year(periods_per_year),
    );
    serde_json::to_string(&score).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Seed-paired bootstrap confidence interval on the **deflated Sharpe** the leaderboard
/// ranks on. `per_seed_returns` is one per-bar return series per held-out seed (the
/// independent sampling units). `n_trials` is the agent's *declared* in-sample search
/// budget; it is folded onto the scoring kernel's baseline footprint (`KERNEL_BASE_TRIALS`)
/// so the interval brackets the same point deflated Sharpe `score_run` reports. Returns the
/// `DsrCi` (`{point, lo, hi, width, confidence, n_boot}`) as a JSON string. Deterministic in
/// `resample_seed`, so the report replays bit-for-bit.
#[pyfunction]
#[pyo3(signature = (per_seed_returns, n_trials = 0, n_boot = 2000, resample_seed = 0x5BA7_2026, alpha = 0.05))]
fn bootstrap_dsr_ci(
    per_seed_returns: Vec<Vec<f64>>,
    n_trials: u32,
    n_boot: usize,
    resample_seed: u64,
    alpha: f64,
) -> PyResult<String> {
    let effective = KERNEL_BASE_TRIALS.saturating_add(n_trials);
    let ci = core_bootstrap_dsr_ci(
        &per_seed_returns,
        effective,
        TRIALS_SR_STD_DEFAULT,
        n_boot,
        resample_seed,
        alpha,
    );
    serde_json::to_string(&ci).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Paired-difference significance test between two leaderboard entries scored on the **same**
/// held-out seed band. `a_per_seed_returns[i]` and `b_per_seed_returns[i]` are the two
/// entries' return series on the *same* seed `i`; the pairing cancels the shared price-path
/// luck so the bootstrap difference isolates skill. Returns the `PairedDiff`
/// (`{point_diff, lo, hi, p_value, confidence, significant, verdict, n_boot}`) as JSON;
/// `verdict` is `"a_better"`, `"b_better"`, or `"tied"`. Deterministic in `resample_seed`.
#[pyfunction]
#[pyo3(signature = (a_per_seed_returns, b_per_seed_returns, n_trials = 0, n_boot = 2000, resample_seed = 0x5BA7_2026, alpha = 0.05))]
fn paired_dsr_diff(
    a_per_seed_returns: Vec<Vec<f64>>,
    b_per_seed_returns: Vec<Vec<f64>>,
    n_trials: u32,
    n_boot: usize,
    resample_seed: u64,
    alpha: f64,
) -> PyResult<String> {
    let effective = KERNEL_BASE_TRIALS.saturating_add(n_trials);
    let diff = core_paired_dsr_diff(
        &a_per_seed_returns,
        &b_per_seed_returns,
        effective,
        TRIALS_SR_STD_DEFAULT,
        n_boot,
        resample_seed,
        alpha,
    );
    serde_json::to_string(&diff).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Whether `decision_json` deserializes to the wire-contract [`Decision`] type — the
/// boundary `contains()` for actions, so a caller can validate an agent's output
/// against the action space without stepping the environment.
#[pyfunction]
fn validate_decision_json(decision_json: &str) -> bool {
    serde_json::from_str::<Decision>(decision_json).is_ok()
}

/// The exact published JSON Schema embedded in the wheel. Model adapters use this
/// for constrained decoding, so a packaged evaluator cannot drift from the repository
/// contract or depend on a source-tree-relative file at runtime.
#[pyfunction]
fn decision_schema_json() -> &'static str {
    include_str!("../../sharpearena/contract/decision.schema.json")
}

/// Deterministically sample the scenario mandate from `seed`, returned as wire JSON
/// (`{"style","max_drawdown","benchmark","text"}`). Byte-identical to the Rust/WASM core.
#[pyfunction]
#[pyo3(signature = (seed, n_symbols = 4, allow_short = true))]
fn sample_mandate_json(seed: u64, n_symbols: usize, allow_short: bool) -> PyResult<String> {
    let m = sharpearena::mandate::sample_mandate(seed, n_symbols, allow_short);
    serde_json::to_string(&m).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// The bounded breach penalty in `[0, 1]` for a mandate (wire JSON) over the recorded
/// per-bar `returns` and per-bar target-`weights` vectors. `0` = clean, `1` = fully breached.
#[pyfunction]
fn mandate_breach(mandate_json: &str, returns: Vec<f64>, weights: Vec<Vec<f64>>) -> PyResult<f64> {
    let m: Mandate = serde_json::from_str(mandate_json)
        .map_err(|e| PyValueError::new_err(format!("invalid mandate JSON: {e}")))?;
    Ok(sharpearena::mandate::mandate_breach(&m, &returns, &weights))
}

/// Derive the held-out seed for eval slot `slot` under a secret `salt` (see
/// `sharpearena::sealed_seed`). Always `>= EVAL_SEED_BASE`, so disjointness from the
/// train band holds without knowing the salt. Rejects salts shorter than
/// `MIN_SEALED_SALT_BYTES`: the derivation is only as unguessable as the salt.
#[pyfunction]
fn sealed_seed(salt: &[u8], slot: u64) -> PyResult<u64> {
    if salt.len() < sharpearena::MIN_SEALED_SALT_BYTES {
        return Err(PyValueError::new_err(format!(
            "sealed-seed salt must be at least {} bytes (got {})",
            sharpearena::MIN_SEALED_SALT_BYTES,
            salt.len()
        )));
    }
    Ok(sharpearena::sealed_seed(salt, slot))
}

/// Generate a procedural scenario and return the native engine's own serde-JSON
/// serialization of it, the exact bytes the Rust core (and, via the identical kernel,
/// the WebAssembly build) fingerprints with its committed FNV-1a goldens. Exposing the
/// serialization rather than a Python object lets a Python-side golden test assert the
/// same cross-runtime fingerprints the native and wasm test suites pin.
#[pyfunction]
#[pyo3(signature = (
    seed,
    n_symbols = 4,
    n_days = 120,
    distribution_mode = "calm",
    vol_clustering = 0.0,
    jump_burst_probability = 0.0,
    jump_burst_persistence = 0.0,
    jump_burst_size = 0.0,
))]
#[allow(clippy::too_many_arguments)]
fn generate_scenario_json(
    seed: u64,
    n_symbols: usize,
    n_days: usize,
    distribution_mode: &str,
    vol_clustering: f64,
    jump_burst_probability: f64,
    jump_burst_persistence: f64,
    jump_burst_size: f64,
) -> PyResult<String> {
    let mode = parse_distribution_mode(distribution_mode)?;
    let spec = ScenarioSpec {
        n_symbols,
        n_days,
        distribution_mode: mode,
        vol_clustering,
        jump_burst_probability,
        jump_burst_persistence,
        jump_burst_size,
        ..ScenarioSpec::default()
    };
    serde_json::to_string(&generate_scenario(&spec, seed))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// The `ScenarioSpec::calm_calibration_candidate` preset generated at `seed`, as the
/// native serde-JSON serialization, the surface behind the candidate's own committed
/// golden fingerprint (a reproducible negative-result configuration, not a certified
/// preset).
#[pyfunction]
fn calm_calibration_candidate_scenario_json(seed: u64) -> PyResult<String> {
    serde_json::to_string(&generate_scenario(
        &ScenarioSpec::calm_calibration_candidate(),
        seed,
    ))
    .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Perturb a `requested` action vector into a realized one via the deterministic Rust
/// `exec_noise` core — the trading analog of ALE sticky actions. With probability
/// `delay_prob` the `previous` action is returned (the order lands a bar late); otherwise
/// each weight gets bounded multiplicative uniform jitter scaled by `slippage_bps`. The
/// draw is keyed on `(seed, step_index)`, so it is byte-reproducible across runtimes.
#[pyfunction]
#[pyo3(signature = (seed, step_index, requested, previous, delay_prob = 0.0, slippage_bps = 0.0))]
fn perturb_action(
    seed: u64,
    step_index: u64,
    requested: Vec<f64>,
    previous: Vec<f64>,
    delay_prob: f64,
    slippage_bps: f64,
) -> Vec<f64> {
    core_perturb_action(
        seed,
        step_index,
        &requested,
        &previous,
        &ExecNoise {
            delay_prob,
            slippage_bps,
        },
    )
}

/// An endogenous price-impact **shared-book market** (M2): `N` agents trade one book per
/// symbol and their aggregate flow moves the cleared price (Kyle permanent + Almgren-
/// Chriss temporary impact). Distinct from the competition surface — here one agent's
/// orders move the price the others see. JSON at the boundary: `reset_market()` returns
/// the initial per-agent observations + market metadata; `step_market(orders_json)`
/// clears one bar and returns the per-agent fills/rewards/observations.
#[pyclass(name = "PyMarketClearing")]
pub struct PyMarketClearing {
    inner: MarketClearing,
    params: MarketParams,
    seed: u64,
    uncertainty: Option<EllipticUncertaintySet>,
    /// Permanent-impact exponent (`1.0` = the frozen linear golden-hash path; `< 1.0` =
    /// concave, the Huberman-Stanzl manipulation-admitting regime). Gated exactly like the
    /// uncertainty set: at `1.0` the native call runs byte-identical linear arithmetic.
    impact_exponent: f64,
}

/// Validate the permanent-impact exponent, surfacing a bad input as a Python `ValueError`
/// instead of the Rust panic inside `clear_bar_concave`.
fn validate_impact_exponent(impact_exponent: f64) -> PyResult<f64> {
    if !(impact_exponent > 0.0 && impact_exponent.is_finite()) {
        return Err(PyValueError::new_err(
            "impact_exponent must be a positive finite number (1.0 = linear)",
        ));
    }
    Ok(impact_exponent)
}

/// Validate-and-build the elliptic uncertainty set from optional ctor/setter inputs.
/// Both radii absent means the point-estimate path; a lone radius pairs with 0.0.
/// Validated here (not via `EllipticUncertaintySet::new`) so a bad input surfaces as a
/// Python `ValueError` instead of a Rust panic.
fn build_uncertainty(
    lambda_radius: Option<f64>,
    eta_radius: Option<f64>,
    correlation: f64,
) -> PyResult<Option<EllipticUncertaintySet>> {
    if lambda_radius.is_none() && eta_radius.is_none() {
        return Ok(None);
    }
    let a = lambda_radius.unwrap_or(0.0);
    let b = eta_radius.unwrap_or(0.0);
    if a < 0.0 || b < 0.0 {
        return Err(PyValueError::new_err(
            "uncertainty radii must be non-negative",
        ));
    }
    if !(-1.0..=1.0).contains(&correlation) {
        return Err(PyValueError::new_err(
            "uncertainty_correlation must lie in [-1, 1]",
        ));
    }
    Ok(Some(EllipticUncertaintySet {
        lambda_radius: a,
        eta_radius: b,
        correlation,
    }))
}

#[pymethods]
impl PyMarketClearing {
    /// Synthetic `n_symbols` × `n_days` panel seeded by `seed`, for `n_agents` agents each
    /// starting with `capital` cash, under Kyle/Almgren-Chriss coefficients.
    #[new]
    #[pyo3(signature = (
        n_symbols = 4,
        n_days = 120,
        seed = 0,
        n_agents = 2,
        capital = 1.0,
        kyle_lambda = 0.1,
        eta = 0.05,
        volume_scale = 1.0,
        vol_scale = 0.0,
        distribution_mode = "calm",
        richness = "standard",
        lambda_radius = None,
        eta_radius = None,
        uncertainty_correlation = 0.0,
        impact_exponent = 1.0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_symbols: usize,
        n_days: usize,
        seed: u64,
        n_agents: usize,
        capital: f64,
        kyle_lambda: f64,
        eta: f64,
        volume_scale: f64,
        vol_scale: f64,
        distribution_mode: &str,
        richness: &str,
        lambda_radius: Option<f64>,
        eta_radius: Option<f64>,
        uncertainty_correlation: f64,
        impact_exponent: f64,
    ) -> PyResult<Self> {
        if n_agents < 1 {
            return Err(PyValueError::new_err("n_agents must be >= 1"));
        }
        let mode = parse_distribution_mode(distribution_mode)?;
        let tier = parse_richness_tier(richness)?;
        let data = build_dataset(n_symbols, n_days, seed, mode, 0.0, 0.0, 0.0, 0.0);
        let inner =
            MarketClearing::from_dataset_with_richness(&data, n_agents, capital, tier.richness());
        let params = MarketParams {
            lambda: kyle_lambda,
            eta,
            volume_scale,
            vol_scale,
        };
        let uncertainty = build_uncertainty(lambda_radius, eta_radius, uncertainty_correlation)?;
        let impact_exponent = validate_impact_exponent(impact_exponent)?;
        Ok(PyMarketClearing {
            inner,
            params,
            seed,
            uncertainty,
            impact_exponent,
        })
    }

    #[getter]
    fn scenario_seed(&self) -> u64 {
        self.seed
    }

    /// The active elliptic uncertainty set as JSON
    /// (`{lambda_radius, eta_radius, correlation}`), or `None` on the point-estimate path.
    #[getter]
    fn uncertainty(&self) -> Option<String> {
        self.uncertainty.as_ref().map(|u| {
            serde_json::json!({
                "lambda_radius": u.lambda_radius,
                "eta_radius": u.eta_radius,
                "correlation": u.correlation,
            })
            .to_string()
        })
    }

    /// Install (or, with both radii `None`, remove) the elliptic uncertainty set used by
    /// every subsequent [`step_market`](Self::step_market) call.
    #[pyo3(signature = (lambda_radius = None, eta_radius = None, correlation = 0.0))]
    fn set_uncertainty(
        &mut self,
        lambda_radius: Option<f64>,
        eta_radius: Option<f64>,
        correlation: f64,
    ) -> PyResult<()> {
        self.uncertainty = build_uncertainty(lambda_radius, eta_radius, correlation)?;
        Ok(())
    }

    /// The permanent-impact exponent every [`step_market`](Self::step_market) call clears
    /// at (`1.0` = the linear golden-hash path, `< 1.0` = concave permanent impact).
    #[getter]
    fn impact_exponent(&self) -> f64 {
        self.impact_exponent
    }

    /// Set the permanent-impact exponent used by every subsequent
    /// [`step_market`](Self::step_market) call. `1.0` restores the linear path.
    fn set_impact_exponent(&mut self, impact_exponent: f64) -> PyResult<()> {
        self.impact_exponent = validate_impact_exponent(impact_exponent)?;
        Ok(())
    }

    /// The active observation-richness disclosure as a JSON object
    /// `{lookback, fundamentals, news}`, the information-poverty difficulty axis.
    #[getter]
    fn richness(&self) -> String {
        let r = self.inner.richness();
        serde_json::json!({
            "lookback": r.lookback,
            "fundamentals": r.fundamentals,
            "news": r.news,
        })
        .to_string()
    }

    #[getter]
    fn symbols(&self) -> Vec<String> {
        self.inner.symbols().to_vec()
    }

    #[getter]
    fn num_agents(&self) -> usize {
        self.inner.n_agents()
    }

    #[getter]
    fn done(&self) -> bool {
        self.inner.is_done()
    }

    /// Pre-trade per-agent observations + market metadata as a JSON string.
    fn reset_market(&self) -> PyResult<String> {
        let observations = observations_to_json(&self.inner.initial_observations())?;
        let out = serde_json::json!({
            "symbols": self.inner.symbols(),
            "n_agents": self.inner.n_agents(),
            "n_bars": self.inner.n_bars(),
            "start_bar": self.inner.start_bar(),
            "cursor": self.inner.cursor(),
            "capital": self.inner.capital(),
            "observations": observations,
        });
        Ok(out.to_string())
    }

    /// Clear one bar. `orders_json` is a JSON array of exactly `n_agents` target-weight
    /// vectors, each length `n_symbols` (canonical agent order, sorted symbol order).
    fn step_market(&mut self, orders_json: &str) -> PyResult<String> {
        let agent_orders: Vec<Vec<f64>> = serde_json::from_str(orders_json)
            .map_err(|e| PyValueError::new_err(format!("invalid orders JSON: {e}")))?;
        if agent_orders.len() != self.inner.n_agents() {
            return Err(PyValueError::new_err(format!(
                "expected {} agent order vectors, got {}",
                self.inner.n_agents(),
                agent_orders.len()
            )));
        }
        let n_sym = self.inner.symbols().len();
        if let Some(bad) = agent_orders.iter().position(|o| o.len() != n_sym) {
            return Err(PyValueError::new_err(format!(
                "agent {bad} order vector has {} weights, expected {n_sym}",
                agent_orders[bad].len()
            )));
        }
        // `step_concave(.., None, 1.0)` runs the identical arithmetic as `step` (the
        // exponent at 1.0 returns the flow term unchanged, no `powf` evaluated), so the
        // point-estimate wire output stays byte-identical when no set and no non-unit
        // exponent are installed.
        let result = self.inner.step_concave(
            &agent_orders,
            &self.params,
            self.uncertainty.as_ref(),
            self.impact_exponent,
        );
        let mut value =
            serde_json::to_value(&result).map_err(|e| PyValueError::new_err(e.to_string()))?;
        if let serde_json::Value::Object(ref mut map) = value {
            map.insert("cursor".to_string(), serde_json::json!(self.inner.cursor()));
        }
        Ok(value.to_string())
    }
}

fn parse_side(s: &str) -> PyResult<Side> {
    match s {
        "buy" => Ok(Side::Buy),
        "sell" => Ok(Side::Sell),
        other => Err(PyValueError::new_err(format!(
            "unknown side {other:?} (expected buy | sell)"
        ))),
    }
}

/// Parse one flat order JSON object into an `(agent, OrderKind)` tuple. Shape:
/// `{agent, kind: "limit"|"market"|"cancel"|"modify", side?, price_tick?, qty?, id?, new_qty?}`.
fn parse_order(v: &serde_json::Value) -> PyResult<(usize, OrderKind)> {
    let bad = |m: &str| PyValueError::new_err(format!("invalid order: {m}"));
    let agent = v["agent"].as_u64().ok_or_else(|| bad("agent"))? as usize;
    let kind = v["kind"].as_str().ok_or_else(|| bad("kind"))?;
    let order = match kind {
        "limit" => OrderKind::Limit {
            side: parse_side(v["side"].as_str().ok_or_else(|| bad("side"))?)?,
            price_tick: v["price_tick"].as_i64().ok_or_else(|| bad("price_tick"))?,
            qty: v["qty"].as_u64().ok_or_else(|| bad("qty"))?,
        },
        "market" => OrderKind::Market {
            side: parse_side(v["side"].as_str().ok_or_else(|| bad("side"))?)?,
            qty: v["qty"].as_u64().ok_or_else(|| bad("qty"))?,
        },
        "cancel" => OrderKind::Cancel {
            id: v["id"].as_u64().ok_or_else(|| bad("id"))?,
        },
        "modify" => OrderKind::Modify {
            id: v["id"].as_u64().ok_or_else(|| bad("id"))?,
            new_qty: v["new_qty"].as_u64().ok_or_else(|| bad("new_qty"))?,
        },
        other => return Err(bad(&format!("unknown kind {other:?}"))),
    };
    Ok((agent, order))
}

fn ladder_json(book: &OrderBook, levels: usize) -> serde_json::Value {
    serde_json::to_value(book.depth_ladder(levels)).unwrap_or(serde_json::Value::Null)
}

/// A deterministic integer-tick **limit-order-book** matching engine (M3): price-time
/// priority, market/limit/cancel/modify, partial fills, and a depth-ladder observation
/// (`mid` / `microprice` / `queue_imbalance`). JSON at the boundary so the tape is
/// byte-identical across runtimes. `step_book` applies a batch of agent orders in
/// canonical order and returns the resulting fills + post-step ladder.
#[pyclass(name = "PyOrderBook")]
pub struct PyOrderBook {
    inner: OrderBook,
    levels: usize,
}

#[pymethods]
impl PyOrderBook {
    #[new]
    #[pyo3(signature = (tick_size = 0.01, levels = 10))]
    fn new(tick_size: f64, levels: usize) -> Self {
        PyOrderBook {
            inner: OrderBook::new(tick_size),
            levels,
        }
    }

    /// Clear the book and return the (empty) depth-ladder snapshot as JSON.
    fn reset_book(&mut self) -> String {
        self.inner = OrderBook::new(self.inner.tick_size());
        serde_json::json!({ "ladder": ladder_json(&self.inner, self.levels) }).to_string()
    }

    /// Apply a JSON array of flat agent orders (canonical order, price-time priority) and
    /// return `{ "fills": [Fill, ...], "ladder": LadderSnapshot }` as JSON.
    fn step_book(&mut self, orders_json: &str) -> PyResult<String> {
        let arr: Vec<serde_json::Value> = serde_json::from_str(orders_json)
            .map_err(|e| PyValueError::new_err(format!("invalid orders JSON: {e}")))?;
        let orders: Vec<(usize, OrderKind)> =
            arr.iter().map(parse_order).collect::<PyResult<Vec<_>>>()?;
        let fills = self.inner.step(&orders);
        let out = serde_json::json!({
            "fills": serde_json::to_value(&fills).map_err(|e| PyValueError::new_err(e.to_string()))?,
            "ladder": ladder_json(&self.inner, self.levels),
        });
        Ok(out.to_string())
    }

    /// The current depth-ladder snapshot as JSON (without stepping).
    fn ladder(&self) -> String {
        ladder_json(&self.inner, self.levels).to_string()
    }

    /// Single-price call-auction uncross over the current book (read-only): the batch
    /// open/close clearing the continuous book lacks. Returns
    /// `{ "clearing_tick": i64, "matched_qty": u64 }` at the volume-maximizing price, or
    /// `null` when nothing crosses. Does not mutate the book or the fill tape.
    fn uncross(&self) -> String {
        match self.inner.uncross() {
            Some((clearing_tick, matched_qty)) => {
                serde_json::json!({ "clearing_tick": clearing_tick, "matched_qty": matched_qty })
                    .to_string()
            }
            None => "null".to_string(),
        }
    }

    /// Read-only walk-the-book cost of filling `qty` on `side` (`"buy"` sweeps asks, `"sell"`
    /// sweeps bids) against current depth. Returns the `SweepCost` as JSON
    /// (`{ "avg_px_tick", "slippage_ticks", "filled_qty" }`) without mutating the book.
    fn sweep_cost(&self, side: &str, qty: u64) -> PyResult<String> {
        let cost = self.inner.sweep_cost(parse_side(side)?, qty);
        serde_json::to_string(&cost).map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

/// The `sharpearena_py` native module (imported as `sharpearena.sharpearena_py`).
#[pymodule]
fn sharpearena_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTradingEnv>()?;
    m.add_class::<PyVecTradingEnv>()?;
    m.add_class::<PyMarketClearing>()?;
    m.add_class::<PyOrderBook>()?;
    m.add_function(wrap_pyfunction!(score_run, m)?)?;
    m.add_function(wrap_pyfunction!(bootstrap_dsr_ci, m)?)?;
    m.add_function(wrap_pyfunction!(paired_dsr_diff, m)?)?;
    m.add_function(wrap_pyfunction!(validate_decision_json, m)?)?;
    m.add_function(wrap_pyfunction!(decision_schema_json, m)?)?;
    m.add_function(wrap_pyfunction!(sample_mandate_json, m)?)?;
    m.add_function(wrap_pyfunction!(mandate_breach, m)?)?;
    m.add_function(wrap_pyfunction!(perturb_action, m)?)?;
    m.add_function(wrap_pyfunction!(sealed_seed, m)?)?;
    m.add_function(wrap_pyfunction!(generate_scenario_json, m)?)?;
    m.add_function(wrap_pyfunction!(
        calm_calibration_candidate_scenario_json,
        m
    )?)?;
    m.add("EVAL_SEED_BASE", sharpearena::EVAL_SEED_BASE)?;
    m.add("MIN_SEALED_SALT_BYTES", sharpearena::MIN_SEALED_SALT_BYTES)?;
    m.add(
        "__doc__",
        "Native pyo3 bindings for the SharpeArena trading-agent environment.",
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn calm(seed: u64) -> Dataset {
        build_dataset(4, 120, seed, DistributionMode::Calm, 0.0, 0.0, 0.0, 0.0)
    }

    /// The readback reports the built panel's own dimensions and the clamped window,
    /// not the numbers a caller asked for.
    #[test]
    fn effective_config_reports_the_built_panel() {
        let data = calm(7);
        let window = Window { start: 0, end: 40 };
        let v: serde_json::Value =
            serde_json::from_str(&effective_config_json(&data, &window, 11).unwrap()).unwrap();
        assert_eq!(v["n_symbols"], 4);
        assert_eq!(v["n_bars"], 120);
        assert_eq!(v["window_end"], 40);
        assert_eq!(v["exec_seed"], 11);
    }

    /// The fingerprint is the crate's committed cross-runtime golden for the Calm
    /// 4x120 panel at seed 7, so a readback and a golden name the same number.
    #[test]
    fn dataset_fingerprint_matches_the_committed_scenario_golden() {
        let data = calm(7);
        let window = Window { start: 0, end: 120 };
        let v: serde_json::Value =
            serde_json::from_str(&effective_config_json(&data, &window, 0).unwrap()).unwrap();
        assert_eq!(v["dataset_fnv1a64"], "b7cf976c71219c52");
    }

    /// An inverted arm is exactly what the fingerprint has to catch: the same seed and
    /// dimensions under a different tier must not fingerprint alike.
    #[test]
    fn dataset_fingerprint_separates_tiers_at_one_seed() {
        let window = Window { start: 0, end: 120 };
        let hard = build_dataset(4, 120, 7, DistributionMode::Hard, 0.0, 0.0, 0.0, 0.0);
        let a: serde_json::Value =
            serde_json::from_str(&effective_config_json(&calm(7), &window, 0).unwrap()).unwrap();
        let b: serde_json::Value =
            serde_json::from_str(&effective_config_json(&hard, &window, 0).unwrap()).unwrap();
        assert_ne!(a["dataset_fnv1a64"], b["dataset_fnv1a64"]);
    }
}
