// Typed views of the SharpeArena engine's JSON shapes. The wire contract (what an
// agent sees / returns) is typed precisely; engine outputs (Run/trace) carry the
// headline fields plus an index signature so they stay forward-compatible as the
// engine adds reported axes.

// --- The agent ⇄ harness wire contract (SharpeArena Agent Interface v1.0) ---------

/** Discrete action label; sizing is carried by `target_weight`. */
export type Action = "buy" | "sell" | "hold" | "close";

/** A single per-instrument instruction. */
export interface Order {
  symbol: string;
  action: Action;
  /** Signed target portfolio weight; negative values represent shorts. */
  target_weight: number;
  /** Stated conviction in [0, 1]; scored for calibration. Defaults to 0.5. */
  confidence?: number;
  /** Optional one-line rationale for this order, captured into the run trace. */
  rationale?: string;
}

/**
 * Self-reported compute spend for one decision (`DecisionCost` in
 * `crates/sharpearena/contract/decision.schema.json`). Every field defaults to zero,
 * so a partial report is valid. The scoring kernel accumulates it into the run cost
 * behind the cost-normalized leaderboard columns.
 */
export interface DecisionCost {
  /** Dollar cost of the compute spent on this decision. The preferred unit. */
  cost_usd?: number;
  /** Prompt/input tokens consumed. */
  tokens_in?: number;
  /** Completion/output tokens produced. */
  tokens_out?: number;
  /**
   * Reasoning tokens, surfaced separately. Providers typically bill these inside
   * `tokens_out`, so they are not re-added to the token total.
   */
  reasoning_tokens?: number;
}

/** What the agent returns at one decision point. */
export interface Decision {
  orders: Order[];
  /** Free-text rationale, captured into the trajectory for auditability. */
  reasoning?: string;
  /** Optional self-reported spend; omit entirely when not reported. */
  cost?: DecisionCost;
}

/** Point-in-time data for one instrument (only data at/before `date`). */
export interface SymbolSnapshot {
  symbol: string;
  /** Trailing closes up to and including `date` (oldest first). */
  close_history: number[];
  fundamentals?: Record<string, number>;
  news?: string[];
}

/** The agent's current holding in one instrument. */
export interface PositionState {
  symbol: string;
  shares: number;
  avg_price: number;
}

/** What the agent sees at one decision point. */
export interface MarketObservation {
  date: string;
  cash: number;
  symbols: SymbolSnapshot[];
  portfolio: PositionState[];
}

// --- The scored engine output ----------------------------------------------------

/** One observable event in a decision trace (tagged union on `event`). */
export interface TraceEvent {
  event: string;
  [k: string]: unknown;
}
export interface Trace {
  events: TraceEvent[];
}

/**
 * One backtest run's output: per-period returns + the decision trace + per-step
 * confidences/outcomes. Recomputed from raw decisions, never self-reported.
 */
export interface Run {
  returns: number[];
  trace: Trace;
  confidences: number[];
  outcomes: boolean[];
  /** Compute/token cost (any consistent unit); 0 = not reported. */
  cost: number;
  [k: string]: unknown;
}

// --- Trajectory (the recompute-to-verify artifact) -------------------------------

/** One captured decision step: the agent's raw output at one observation. */
export interface DecisionStep {
  step: number;
  observation_id: string;
  decision: Decision;
}

/** One captured backtest run (window × seed): the raw decisions + replay coords. */
export interface RunTrajectory {
  window_start: number;
  window_end: number;
  seed: number;
  steps: DecisionStep[];
}

// --- Dataset + engine inputs -----------------------------------------------------

/** A leak-free point-in-time price panel: a shared date axis + per-symbol closes. */
export interface Dataset {
  dates: string[];
  /** symbol → closes, each array aligned to `dates`. */
  closes: Record<string, number[]>;
  /** symbol → per-step cash dividend, aligned to `dates`. May be omitted. */
  dividends?: Record<string, number[]>;
}

/** A simulation window over the date axis: half-open `[start, end)`. */
export interface Window {
  start: number;
  end: number;
}

/** Basis-point transaction-cost model; every field falls back to the engine default. */
export interface CostModel {
  fee_bps?: number;
  slippage_bps?: number;
  impact_bps?: number;
  financing_bps?: number;
  /** Max fraction of NAV traded per step; omit for unlimited liquidity. */
  max_participation?: number;
}

/** Deterministic synthetic-panel parameters. */
export interface SyntheticParams {
  n_symbols?: number;
  n_days?: number;
  seed?: number;
}

/** Where a baseline run's prices come from: synthetic params OR raw CSV text. */
export interface DatasetSource {
  synthetic?: SyntheticParams;
  /** `date,symbol,close[,dividend]` long-format CSV (header required). */
  csv?: string;
}

/** The named baseline agents shipped in-process by the engine. */
export type BaselineAgent = "buy_and_hold" | "hold" | "momentum" | "random";

/** Config for {@link runBaseline}. */
export interface BaselineConfig {
  agent: BaselineAgent;
  /** Defaults to a 4×120 synthetic panel (seed 0). */
  dataset?: DatasetSource;
  /** Defaults to `{ start: 20, end: dataset.length }` (20-bar warm-up). */
  window?: Window;
  /** Execution seed (slippage noise; also seeds the `random` agent). */
  seed?: number;
  costs?: CostModel;
  /** Trailing window for the `momentum` baseline (default 10). */
  momentum_lookback?: number;
}

/** Walk-forward window-generation parameters. */
export interface WalkForwardParams {
  n_days: number;
  warmup: number;
  test: number;
  step: number;
}

/** A coarse market-regime label over a window. */
export type Regime = "bull" | "bear" | "chop";

/** One named adversarial stress scenario. */
export interface StressScenario {
  name: string;
  dataset: Dataset;
}

// --- Procedural scenario generation ----------------------------------------------

/** How adversarial a generated scenario is (`DistributionMode` in the kernel). */
export type DistributionMode =
  | "calm"
  | "hard"
  | "extreme"
  | "cointegrated_pairs"
  | "regime_shift";

/**
 * How much of the market the agent is shown at each step. The kernel deserializes this
 * with `deny_unknown_fields` and no per-field defaults, so supplying it at all means
 * supplying all three.
 */
export interface ObservationRichness {
  /** Trailing closes surfaced per symbol. */
  lookback: number;
  /** Populate each snapshot's point-in-time `fundamentals` map. */
  fundamentals: boolean;
  /** Populate each snapshot's point-in-time `news` headlines. */
  news: boolean;
}

/**
 * A scenario family: a seed interval plus the panel shape and difficulty tier.
 *
 * The five base fields are **required whenever a `spec` is supplied at all**: the kernel
 * struct carries `deny_unknown_fields` and no `#[serde(default)]` on them, so a partial
 * spec is refused by name (`missing field 'start_level'`) rather than defaulted. Omit
 * `spec` entirely for the default 4x120 Calm family. Fields added after the original
 * contract are `#[serde(default)]` in the kernel and optional here.
 */
export interface ScenarioSpec {
  start_level: number;
  /** Size of the legal seed interval; `0` means unbounded. */
  num_levels: number;
  n_symbols: number;
  n_days: number;
  distribution_mode: DistributionMode;
  obs_richness?: ObservationRichness;
  /** Opt-in volatility-clustering strength (`0` = off). */
  vol_clustering?: number;
  /** Opt-in probability of beginning a deterministic jump burst on a bar. */
  jump_burst_probability?: number;
  /** Conditional probability that a jump burst continues one more bar. */
  jump_burst_persistence?: number;
  /** Absolute simple-return size of each extra burst jump. */
  jump_burst_size?: number;
}

/** Input to {@link generateScenario}: the family plus the level seed to draw. */
export interface ScenarioInput {
  spec?: ScenarioSpec;
  seed?: number;
}
