//! # SharpeArena — the trading-agent environment
//!
//! A leak-free, point-in-time market environment with a dead-simple, language-agnostic
//! agent contract: **the harness sends an [`MarketObservation`], the agent returns a
//! [`Decision`], repeat.** Look-ahead is structurally impossible — [`Dataset`] never hands
//! out a future bar — and trajectories are recompute-from-raw-decisions, so an agent cannot
//! lie about its returns.
//!
//! ## One engine, several surfaces
//!
//! The deterministic engine is shared with the published SharpeBench simulator rather than
//! copied. This crate adds the Gym-style [`reset`]/[`step`] lifecycle, procedural scenarios,
//! vector and multi-agent market primitives, the governed wire contract, sealed evaluation
//! seeds, and a cross-surface [`SPEC_HASH`] handshake. The same engine is exposed through
//! Rust, Python/Gymnasium, WASM/npm, and the language-neutral observation/decision protocol.
//!
//! [`reset`]: https://gymnasium.farama.org
//! [`step`]: https://gymnasium.farama.org
#![forbid(unsafe_code)]

// --- The frozen wire-contract version (the standard SharpeArena governs) --------------------

pub mod contract;
pub use contract::CONTRACT_VERSION;

// --- Cross-surface tape-semantics fingerprint (refuse-on-mismatch handshake) ---------------

pub mod spec_hash;
pub use spec_hash::{SPEC_HASH, SPEC_HASH_HEX};

// --- Seeded procedural scenario generation (Procgen-style seed intervals) ------------------

pub mod scenario_gen;
pub use scenario_gen::{
    cross_regime_split, generate_scenario, level_seed, sealed_seed, train_test_split,
    DistributionMode, ScenarioSpec, SealedSalt, SealedSaltError, EVAL_SEED_BASE,
    MIN_SEALED_SALT_BYTES,
};

// --- Information-disclosure difficulty (the axis orthogonal to the regime tiers) -----------

pub mod richness;
pub use richness::{ObservationRichness, RichnessTier, DEFAULT_LOOKBACK};

// --- Adaptive difficulty-targeting curriculum (Prioritized Level Replay) -------------------

pub mod curriculum;
pub use curriculum::AdaptiveCurriculum;

// --- Statistical-confidence layer for the leaderboard (bootstrap CI + paired A/B test) -----

pub mod leaderboard_ci;
pub use leaderboard_ci::{bootstrap_dsr_ci, deflated_sharpe, paired_dsr_diff, DsrCi, PairedDiff};

// --- Vectorized, batched environment (gym3's "vectorized-first" design) -------------------

pub mod vec_env;
pub use vec_env::{BatchStep, LaneConfig, VecTradingEnv};

// --- Per-scenario trading mandates (MiniGrid Fetch-style per-episode objective) -----------

pub mod mandate;
pub use mandate::{mandate_breach, sample_mandate, Mandate, MandateStyle};

// --- Execution-noise perturbation (seeded sticky-actions / slippage; ALE-style) -----------

pub mod exec_noise;
pub use exec_noise::{perturb as perturb_action, ExecNoise};

// --- Limit-order-book matching engine (M3) -------------------------------------------------

pub mod lob_market;
pub use lob_market::{Fill, LadderSnapshot, OrderBook, OrderKind, RestingOrder, Side};

// --- Endogenous price-impact shared-book market (M2) ---------------------------------------

pub mod market;
pub use market::{
    clear_bar, clear_bar_concave, clear_bar_robust, AgentFill, ClearResult, EllipticUncertaintySet,
    ImpactCoefficients, MarketClearing, MarketParams,
};

// --- Transport-fault gate: a masked hold is a failed cell, never a return series ----------

pub mod transport_gate;
pub use sharpebench_sim::transport::{DecideError, TransportDiagnostics, TransportHealth};
pub use transport_gate::{run_backtest_checked, transport_fault, CellOutcome, TransportFault};

// --- Point-in-time simulator surface (extraction from `sharpebench-sim`) ------------------

pub use sharpebench_sim::{
    // Trajectory capture + replay-recompute (the tamper-evidence path).
    replay_run,
    replay_submission,
    // The single-backtest engine + its window type.
    run_backtest,
    run_backtest_capture,
    // Walk-forward out-of-sample windows + regime tagging.
    tag_regime,
    walk_forward,
    // In-process reference/baseline agents + the trait they implement.
    Agent,
    BuyAndHold,
    // Leak-free data model + execution cost model.
    CostModel,
    CostProfile,
    Dataset,
    // O(1) environment snapshot (clone_state / restore_state).
    EnvState,
    // External transports — a conforming agent is just a program that reads observations
    // (stdin / `POST /decide`) and writes decisions. The diagnostics types travel with
    // them: a wire fault returns an empty-orders hold, so the health record is the only
    // thing that distinguishes a masked fault from a deliberate hold, and a consumer that
    // cannot name the type cannot read it. See `transport_gate`.
    ExternalAgent,
    HoldAgent,
    HttpAgent,
    Momentum,
    RandomAgent,
    Regime,
    // The Gym-style open-loop environment: `reset()` / `step()` over the same engine,
    // plus the named crisis-suite/scenario bundle.
    Scenario,
    StepInfo,
    StepResult,
    TeamAgent,
    TradingEnv,
    Window,
};

// Vol-targeted, drawdown-braked reference agent — new in sharpebench-sim 0.5.0, which
// exposes it at module level only (hence the `agent::` path).
pub use sharpebench_sim::agent::RiskManaged;

// --- The language-agnostic wire contract (the standard SharpeArena governs) -----------------

pub use sharpebench_protocol::{
    Action, AgentTrajectory, Decision, DecisionCost, DecisionStep, MarketObservation, Order,
    PositionState, RunTrajectory, SymbolSnapshot,
};

// --- The scored output (so callers read returns/trace without a second dependency) --------

pub use sharpebench_core::Run;
