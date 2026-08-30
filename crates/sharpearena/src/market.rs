//! Endogenous price-impact shared-book market (M2) — the flagship multi-agent engine.
//!
//! SharpeArena's existing multi-agent surface ([`crate::vec_env`], and the PettingZoo
//! competition env) is *competition*: `N` agents each trade their **own** copy of the
//! same frozen path, so one agent's flow never moves the price another sees. This module
//! makes the market **endogenous** — `N` agents share one book per symbol and their
//! aggregate order flow moves the cleared price.
//!
//! ## The model (Kyle 1985 linear impact + an Almgren-Chriss permanent/temporary split)
//!
//! The frozen synthetic panel ([`Dataset::synthetic`]) is the **exogenous**
//! (fundamental / noise-trader) component. Each bar `t`, for each symbol `s`:
//!
//! - The **cleared reference mid** the agents transact around is the exogenous mid
//!   scaled by the *running permanent-impact multiplier* `M`, which accumulates the
//!   price pressure of every **prior** bar's flow:
//!   `cleared_mid_t = exo_mid_t * M_t`. Crucially `M_t` depends only on flow strictly
//!   before `t`, so the reference price an agent decides against is not moved by any
//!   bar-`t` order (its own or a peer's).
//! - Each agent submits a target weight; it converts to a signed **order size**
//!   `q_i = capital * (w_i − prev_w_i) / cleared_mid_t` (the change in desired notional
//!   divided by price). Aggregate **net flow** `Q_t = Σ_i q_i`, summed in **sorted agent
//!   order** for float determinism.
//! - **Permanent impact** (Kyle) updates the multiplier for the *next* bar:
//!   `M_{t+1} = M_t * (1 + lambda * Q_t / V)`, where `lambda` is Kyle's impact coefficient
//!   and `V` an ADV-like volume normalizer. The bump carries forward forever (a permanent
//!   move of the reference price), exactly the accumulating-multiplier the spec asks for.
//! - **Temporary impact** (Almgren-Chriss) is what agent `i` actually pays this bar:
//!   `fill_i = cleared_mid_t * (1 + f_t * (lambda * Q_t + eta * q_i) / V)` — it pays for the
//!   crowd's flow (`lambda * Q_t`) plus its own size (`eta * q_i`). Temporary impact does
//!   not persist; it is a per-fill execution cost. `f_t` is an optional **volatility-scaling
//!   factor** (`MarketParams::vol_scale`): with `vol_scale = 0` (default) `f_t = 1` and the
//!   term is exactly the static cost above; with `vol_scale > 0`,
//!   `f_t = min(1 + vol_scale * vol_t, 3)` where `vol_t` is the trailing mean-squared
//!   cleared return (a `sqrt`-free variance proxy over the last `VOL_WINDOW` *past* bars),
//!   so spreads/slippage widen in high-vol regimes and the cap bounds divergence.
//! - **Per-agent reward** is that agent's own realized portfolio return over the bar,
//!   marked at the cleared mids and using its **own** fill prices.
//!
//! ## Robust impact: an optional elliptic uncertainty set
//!
//! `lambda` and `eta` above are **point estimates**, and a point estimate of impact is a
//! strong claim: both are fitted from the same trades on the same tape, nobody observes
//! them directly, and an agent tuned against one exact `lambda` is tuned against a number
//! nobody knows. Following "Robust Reinforcement Learning in Finance: Modeling Market
//! Impact with Elliptic Uncertainty Sets", [`EllipticUncertaintySet`] replaces the pair
//! with a set and clears each bar against the **worst case inside it**.
//!
//! The set is entirely opt-in: [`clear_bar`] and [`MarketClearing::step`] are unchanged,
//! and the robust entry points ([`clear_bar_robust`], [`MarketClearing::step_robust`])
//! given `None` execute the identical float operations on the identical values. Published
//! results and the cross-runtime golden hashes are pinned to the point-estimate dynamics,
//! so the default path is never allowed to move.
//!
//! See [`EllipticUncertaintySet`] for the geometry, the closed-form worst case, and why an
//! ellipse rather than a box is the right shape for two correlated impact coefficients.
//!
//! ## Concave permanent impact: an optional impact exponent
//!
//! Under **linear** permanent impact, round-trip manipulation is unprofitable by
//! construction (Huberman-Stanzl 2004: linear permanent impact is the unique
//! quasi-arbitrage-free specification), so a manipulation probe run against the linear
//! model can only ever confirm theory: it cannot fail. To make that probe falsifiable,
//! the concave entry points ([`clear_bar_concave`], [`MarketClearing::step_concave`])
//! accept an `impact_exponent` applied to the **permanent (Kyle) normalized-flow term**:
//! `Q/V` is replaced by `sign(Q/V) * |Q/V|^exponent` in the temporary fill's crowd term
//! and the permanent multiplier update, while the Almgren-Chriss own-size term `eta * q_i`
//! stays linear and divided by `V`. `exponent = 1.0` uses the exact legacy expression
//! `(lambda * Q + eta * q_i) / V`, so the arithmetic is byte-identical to
//! [`clear_bar`] / [`clear_bar_robust`]. Exponents below one make permanent impact concave
//! in flow, the square-root-law regime under which Huberman-Stanzl predict manipulation
//! can become profitable.
//!
//! Like the uncertainty set, the exponent is **opt-in and gated outside the golden-hash
//! path**: [`clear_bar`] and [`MarketClearing::step`] never touch it, and a general power
//! requires `powf`, a libm transcendental excluded from the mul/add/div-only cross-runtime
//! determinism guarantee below. Published results and the cross-runtime golden hashes are
//! pinned to the linear point-estimate dynamics, which the concave path is never allowed
//! to move.
//!
//! ## Determinism
//!
//! Every step is a pure function of `(exogenous path, lambda, eta, V, vol_scale, capital,
//! agents' actions in sorted order)`. Only `mul / add / div` are used — no `ln` / `exp` /
//! `sqrt` or other transcendentals that differ across libm builds — so a cleared path is
//! byte-identical across Rust, WASM, and Python. The volatility proxy is a trailing
//! *variance* (mean of squared returns), chosen precisely to avoid `sqrt`; its ring buffer
//! holds only returns from past cleared bars, and `vol_scale = 0` multiplies the impact
//! term by an exact `1.0`, so the default path is bit-for-bit the pre-vol-scaling fill.
//! Aggregation folds the per-agent sizes in canonical (sorted) agent order, so the parallel
//! collection of actions cannot perturb `Q_t`. The one square root in the crate, in
//! [`EllipticUncertaintySet::worst_case`], is admitted deliberately: IEEE 754 mandates that
//! `sqrt` be **correctly rounded**, so unlike `ln` / `exp` it is not an implementation-
//! defined libm approximation and reproduces bit-for-bit on every target. There is still no
//! RNG, no clock, and no I/O anywhere on the clearing path.
//!
//! ## Leak-free invariant
//!
//! An agent's observation at `t` reflects only **cleared prices ≤ t** (public, post-clear
//! market data) and **its own fills** — never another agent's *pending* order for `t`.
//! Two structural facts enforce this: (1) the cleared reference mid `cleared_mid_t` and
//! every agent's order *size* `q_i` are computed from `M_t`, which embeds only flow before
//! `t`; (2) the Parallel API collects **all** bar-`t` actions before producing any bar
//! `t+1` observation, so no agent's bar-`t` decision can see a peer's bar-`t` intent. The
//! realized fill *price* does reflect the aggregate cleared flow `Q_t` — that is the
//! price-impact channel of a real market, not an information leak.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::richness::ObservationRichness;
use crate::{Dataset, MarketObservation, PositionState, SymbolSnapshot};

/// Bars of trailing exogenous history burned in before the first decision, mirroring the
/// open-loop env's warm-up so an agent's first observation already has trailing closes.
const WARMUP: usize = 20;
/// Below this absolute size a position/NAV is treated as flat (avoids divide-by-zero in
/// the return and average-price bookkeeping). Comparisons only — never an additive fudge.
const EPS: f64 = 1e-12;
/// Trailing-return magnitude above which a derived news headline reads as a trend rather
/// than range-bound (2%). Only consulted when [`ObservationRichness::news`] is set.
const NEWS_THRESHOLD: f64 = 0.02;
/// Trailing window (in past cleared bars) of the volatility proxy used by `vol_scale`.
const VOL_WINDOW: usize = 20;
/// Upper bound on the volatility-scaling factor, so an extreme `vol_scale` (or a violent
/// vol regime) cannot blow the temporary impact up without limit.
const VOL_FACTOR_CAP: f64 = 3.0;

/// The impact coefficients: Kyle's permanent `lambda`, Almgren-Chriss temporary `eta`,
/// and the ADV-like `volume_scale` (`V`) that defines dimensionless flow. The linear
/// path is `(lambda * Q + eta * q_i) / V`. On the opt-in nonlinear path, permanent
/// impact is `lambda * sign(Q/V) * |Q/V|^beta`; this normalization keeps `lambda`
/// dimensionless (for the declared `beta`) and comparable when the share/notional unit is
/// rescaled.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MarketParams {
    /// Kyle's permanent price-impact coefficient (per unit normalized net flow).
    pub lambda: f64,
    /// Almgren-Chriss temporary impact coefficient (per unit normalized **own** size).
    pub eta: f64,
    /// Volume / ADV normalizer `V` that `lambda * Q` and `eta * q_i` are divided by.
    pub volume_scale: f64,
    /// Optional volatility scaling of the **temporary impact**. `0.0` (default) is off —
    /// the fill is the static Kyle/Almgren-Chriss cost. When `> 0`, the temporary-impact
    /// term is multiplied by `min(1 + vol_scale * trailing_vol, VOL_FACTOR_CAP)`, where
    /// `trailing_vol` is the mean of squared cleared returns over the last `VOL_WINDOW`
    /// *past* bars — so execution costs widen as realized volatility rises.
    pub vol_scale: f64,
}

impl Default for MarketParams {
    fn default() -> Self {
        Self {
            lambda: 0.1,
            eta: 0.05,
            volume_scale: 1.0,
            vol_scale: 0.0,
        }
    }
}

/// The impact coefficients actually applied to one symbol on one bar: Kyle's permanent
/// `lambda` and Almgren-Chriss's temporary `eta`. On the point-estimate path these are
/// just [`MarketParams::lambda`] / [`MarketParams::eta`]; under an
/// [`EllipticUncertaintySet`] they are that bar's worst case inside the set.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ImpactCoefficients {
    /// Kyle's permanent price-impact coefficient applied this bar.
    pub lambda: f64,
    /// Almgren-Chriss's temporary impact coefficient applied this bar.
    pub eta: f64,
}

/// An elliptic uncertainty set around the impact point estimate `(lambda, eta)`, after
/// "Robust Reinforcement Learning in Finance: Modeling Market Impact with Elliptic
/// Uncertainty Sets".
///
/// # The geometry, and why an ellipse rather than a box
///
/// The set is the unit level set of a covariance `S` centred on the point estimate:
///
/// ```text
/// U = { theta : (theta - theta_hat)^T S^-1 (theta - theta_hat) <= 1 }
/// S = [[ a^2,      rho*a*b ],
///      [ rho*a*b,  b^2     ]]
/// ```
///
/// with `a = lambda_radius`, `b = eta_radius`, `rho = correlation`. `a` and `b` are the
/// half-widths of the plausible range of each coefficient on its own; `rho` says how their
/// errors move together.
///
/// A box (an independent interval per coefficient) would be the wrong shape here, for two
/// reasons. The first is statistical: `lambda` and `eta` are not measured separately. They
/// are two coefficients of one fit to one set of executed trades, so their estimation
/// errors are coupled, and a fit that attributes more of the observed slippage to the
/// permanent component necessarily attributes less to the temporary one. A box asserts the
/// two vary independently and therefore admits its corners, `(lambda_max, eta_max)` in
/// particular, which under any nonzero correlation is a combination the data never
/// supports. Robustness bought against a corner nobody can occupy is paid for in
/// conservatism and returned as nothing. An ellipse is the level set of the estimator's own
/// covariance, so it encodes the coupling directly and excludes exactly those corners.
///
/// The second reason is that it makes the worst case a closed form. Impact cost is linear
/// in `theta` (see [`worst_case`](Self::worst_case)), and the maximum of a linear function
/// over an ellipse is the classical second-order-cone expression: one matrix-vector product
/// and one square root, no search, no iteration, no sampling. The uncertainty set therefore
/// costs a handful of flops per symbol per bar and introduces no RNG, which is what lets it
/// sit on a determinism-critical clearing path at all.
///
/// # Sign convention
///
/// Worst case means **most expensive for the traders**, so the set is resolved in the
/// direction that maximises the bar's aggregate impact cost. A negative `correlation` can
/// therefore push one coefficient below its point estimate while the other rises: that is
/// the ellipse doing its job, and it is precisely the behaviour a box cannot express. Both
/// resolved coefficients are floored at zero, since a negative impact coefficient would
/// turn execution into a rebate, which is not a market this model describes.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EllipticUncertaintySet {
    /// Half-width `a` of the plausible range of Kyle's `lambda`, in `lambda`'s own units.
    pub lambda_radius: f64,
    /// Half-width `b` of the plausible range of Almgren-Chriss's `eta`, in `eta`'s units.
    pub eta_radius: f64,
    /// Correlation `rho` in `[-1, 1]` between the two coefficients' estimation errors.
    /// `0` gives axis-aligned axes; `-1` or `1` degenerates the ellipse to a line segment.
    pub correlation: f64,
}

impl EllipticUncertaintySet {
    /// Build a set from the two half-widths and their correlation.
    ///
    /// Panics on a negative radius or a `correlation` outside `[-1, 1]`, either of which
    /// would describe a matrix that is not a covariance.
    pub fn new(lambda_radius: f64, eta_radius: f64, correlation: f64) -> Self {
        assert!(
            lambda_radius >= 0.0 && eta_radius >= 0.0,
            "uncertainty radii must be non-negative"
        );
        assert!(
            (-1.0..=1.0).contains(&correlation),
            "correlation must lie in [-1, 1]"
        );
        Self {
            lambda_radius,
            eta_radius,
            correlation,
        }
    }

    /// A circular set: the same half-width on both coefficients, uncorrelated. The weakest
    /// honest statement of impact uncertainty, and the natural default when no estimator
    /// covariance is available.
    pub fn isotropic(radius: f64) -> Self {
        Self::new(radius, radius, 0.0)
    }

    /// The point of the set that maximises this bar's impact cost, in closed form.
    ///
    /// One bar's aggregate execution cost above the cleared mid is linear in
    /// `theta = (lambda, eta)`:
    ///
    /// ```text
    /// cost(theta)  ~  lambda * cost_lambda + eta * cost_eta  =  c^T theta
    /// ```
    ///
    /// so [`clear_bar_robust`] passes `cost_lambda = Q^2` and `cost_eta = sum_i q_i^2` for
    /// the symbol (both non-negative, both in the units of the cost the agents actually
    /// pay). Maximising `c^T theta` over `U` has the standard solution
    ///
    /// ```text
    /// theta_wc = theta_hat + S c / sqrt(c^T S c)
    /// ```
    ///
    /// whose attained cost is `c^T theta_hat + sqrt(c^T S c)`: the point estimate plus the
    /// support function of the ellipse in the cost direction. `c^T S c >= 0` for any `c`
    /// because `|correlation| <= 1` makes `S` positive semidefinite, and the answer depends
    /// on `c` only through its direction, so the units chosen for the cost do not matter.
    ///
    /// A degenerate direction (`c^T S c == 0`: no flow at all, a zero-radius set, or a
    /// fully degenerate ellipse orthogonal to the cost) has no interior maximiser and falls
    /// back to the point estimate, which is also what keeps a no-flow bar identical to the
    /// non-robust path.
    pub fn worst_case(
        &self,
        params: &MarketParams,
        cost_lambda: f64,
        cost_eta: f64,
    ) -> ImpactCoefficients {
        let point = ImpactCoefficients {
            lambda: params.lambda,
            eta: params.eta,
        };
        let a = self.lambda_radius;
        let b = self.eta_radius;
        let cross = self.correlation * a * b;
        // S c
        let sc_lambda = a * a * cost_lambda + cross * cost_eta;
        let sc_eta = cross * cost_lambda + b * b * cost_eta;
        // c^T S c, non-negative by positive semidefiniteness of S.
        let quad = cost_lambda * sc_lambda + cost_eta * sc_eta;
        if quad <= 0.0 {
            return point;
        }
        let norm = quad.sqrt();
        ImpactCoefficients {
            lambda: floor_at_zero(point.lambda + sc_lambda / norm),
            eta: floor_at_zero(point.eta + sc_eta / norm),
        }
    }
}

/// The signed power `sign(q) * |q|^exponent` applied to the permanent-impact flow term on
/// the concave path.
///
/// `exponent == 1.0` is special-cased to return `q` itself (the identical bits, with no
/// `powf` evaluated), which is what makes [`clear_bar_concave`] at exponent one
/// byte-identical to the frozen linear path. Any other exponent goes through `powf`, a
/// libm transcendental deliberately excluded from the golden-hash determinism guarantee;
/// that is why the exponent lives on the gated concave entry points and never on
/// [`clear_bar`].
fn signed_pow(q: f64, exponent: f64) -> f64 {
    if exponent == 1.0 {
        q
    } else if q >= 0.0 {
        q.powf(exponent)
    } else {
        -((-q).powf(exponent))
    }
}

/// Clamp a resolved impact coefficient at zero. A negative coefficient would pay traders to
/// execute, which is outside the Kyle / Almgren-Chriss model, so the floor is a modelling
/// statement rather than numerical hygiene.
fn floor_at_zero(x: f64) -> f64 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

/// One agent's fill in one symbol this bar: the signed size traded and the
/// temporary-impact price it paid.
#[derive(Clone, Debug, Serialize)]
pub struct AgentFill {
    pub symbol: String,
    /// Signed shares traded this bar (`q_i`); positive = buy, negative = sell.
    pub size: f64,
    /// The temporary-impact execution price the agent paid (Almgren-Chriss).
    pub fill_price: f64,
}

/// The result of clearing one bar. All per-agent vectors are in canonical (sorted) agent
/// order; all per-symbol vectors are in sorted symbol order.
#[derive(Clone, Debug, Serialize)]
pub struct ClearResult {
    /// The cleared reference mid per symbol (`exo_mid * M`) — the public post-clear tape.
    pub cleared_mids: Vec<f64>,
    /// Aggregate signed net flow per symbol (`Q_t`).
    pub net_flow: Vec<f64>,
    /// Per-agent realized portfolio return over the bar (the reward).
    pub rewards: Vec<f64>,
    /// Per-agent post-bar NAV (cash + positions marked at the cleared mids).
    pub navs: Vec<f64>,
    /// Per-agent, per-symbol fills (size + temporary-impact fill price).
    pub fills: Vec<Vec<AgentFill>>,
    /// Per-agent next observation (cleared price history ≤ t + own portfolio/cash).
    pub observations: Vec<MarketObservation>,
    /// Whether the path is exhausted after this bar (no more bars to clear).
    pub done: bool,
    /// The impact coefficients this bar was actually cleared at, per symbol, present only
    /// when an [`EllipticUncertaintySet`] was supplied. `None` on the point-estimate path,
    /// where the coefficients are by definition [`MarketParams::lambda`] /
    /// [`MarketParams::eta`]; the field is then omitted from the serialized result
    /// entirely, so the default wire shape is unchanged.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub robust_impact: Option<Vec<ImpactCoefficients>>,
}

/// One agent's running book: cash, per-symbol holdings, per-symbol accumulated cost
/// (for the displayed average price), and its prior target weights (for sizing the
/// change-in-desired-notional order).
#[derive(Clone, Debug)]
struct AgentBook {
    cash: f64,
    shares: Vec<f64>,
    cost_basis: Vec<f64>,
    prev_weight: Vec<f64>,
}

/// A fixed-window, running mean-of-squared-returns volatility proxy for one symbol. It is
/// deliberately a trailing **variance** (squared returns), not a standard deviation, so it
/// needs no `sqrt` and stays byte-identical across runtimes — only mul/add/div. Point-in-
/// time by construction: it only ever holds returns from cleared bars strictly in the past.
#[derive(Clone, Debug)]
struct VolTracker {
    /// Ring of the last [`VOL_WINDOW`] squared returns.
    ring: [f64; VOL_WINDOW],
    /// Next write slot (also the oldest entry once the ring is full).
    head: usize,
    /// Entries written so far (saturates at [`VOL_WINDOW`]).
    count: usize,
    /// Running sum of the squared returns currently in the ring.
    sum_sq: f64,
}

impl VolTracker {
    fn new() -> Self {
        Self {
            ring: [0.0; VOL_WINDOW],
            head: 0,
            count: 0,
            sum_sq: 0.0,
        }
    }

    /// Mean squared return over the buffered window — the vol proxy (`0.0` until the first
    /// push, so a cold buffer applies no scaling).
    fn proxy(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum_sq / self.count as f64
        }
    }

    /// Record one realized cleared return, evicting the oldest entry once the ring is full.
    fn push(&mut self, ret: f64) {
        let sq = ret * ret;
        if self.count == VOL_WINDOW {
            self.sum_sq -= self.ring[self.head];
        } else {
            self.count += 1;
        }
        self.ring[self.head] = sq;
        self.sum_sq += sq;
        self.head = (self.head + 1) % VOL_WINDOW;
    }
}

/// The shared-book market state: the running permanent-impact multiplier, the realized
/// cleared tape, and every agent's book. Built from a [`Dataset`]; driven one bar at a
/// time by [`clear_bar`] (or the [`MarketClearing::step`] convenience that feeds it the
/// current bar's exogenous mids).
pub struct MarketClearing {
    symbols: Vec<String>,
    dates: Vec<String>,
    /// Exogenous (fundamental) closes per symbol, full path: `exo[s][bar]`.
    exo: Vec<Vec<f64>>,
    capital: f64,
    /// Running permanent-impact multiplier `M` per symbol (1.0 = untouched).
    impact_mult: Vec<f64>,
    /// The previous bar's cleared mid per symbol — the mark for holding-PnL.
    prev_mid: Vec<f64>,
    /// The realized cleared tape per symbol (grows one entry per cleared bar).
    cleared_history: Vec<Vec<f64>>,
    /// Trailing realized-volatility proxy per symbol, fed by past cleared returns and read
    /// by [`MarketParams::vol_scale`] to widen temporary impact in high-vol regimes.
    vol: Vec<VolTracker>,
    agents: Vec<AgentBook>,
    /// The next bar index to clear.
    cursor: usize,
    /// The first bar that is cleared *with* trading (after the warm-up burn-in).
    start_bar: usize,
    n_bars: usize,
    /// How much of the market each observation discloses (trailing lookback + optional
    /// fundamentals/news). Default is the historical [`DEFAULT_LOOKBACK`]-bar, no-fields
    /// disclosure (see [`ObservationRichness`]).
    ///
    /// [`DEFAULT_LOOKBACK`]: crate::DEFAULT_LOOKBACK
    richness: ObservationRichness,
}

impl MarketClearing {
    /// Build an endogenous market over `data` for `n_agents`, each starting with
    /// `capital` cash, at the historical (default) observation disclosure. The exogenous
    /// path is taken from `data`'s closes; the first `WARMUP` bars are an untraded burn-in
    /// (cleared == exogenous) so the first observation has trailing history.
    pub fn from_dataset(data: &Dataset, n_agents: usize, capital: f64) -> Self {
        Self::from_dataset_with_richness(data, n_agents, capital, ObservationRichness::default())
    }

    /// [`from_dataset`](Self::from_dataset) with an explicit observation-richness
    /// disclosure, the information-poverty difficulty axis. `richness` controls only how
    /// much *past / contextual* information each observation surfaces (trailing lookback,
    /// optional fundamentals/news); it never reveals a future bar, so the leak-free
    /// invariant is untouched. Passing [`ObservationRichness::default`] reproduces
    /// [`from_dataset`](Self::from_dataset) byte-for-byte.
    pub fn from_dataset_with_richness(
        data: &Dataset,
        n_agents: usize,
        capital: f64,
        richness: ObservationRichness,
    ) -> Self {
        assert!(n_agents >= 1, "a market needs at least one agent");
        let symbols = data.symbols();
        let n_sym = symbols.len();
        let n_bars = data.len();
        assert!(
            n_sym >= 1 && n_bars >= 2,
            "need at least one symbol and two bars"
        );
        let exo: Vec<Vec<f64>> = symbols
            .iter()
            .map(|s| data.closes.get(s).cloned().unwrap_or_default())
            .collect();
        let start_bar = WARMUP.min(n_bars.saturating_sub(1)).max(1);
        // Burn-in tape: the untraded exogenous closes strictly before the first traded bar.
        let cleared_history: Vec<Vec<f64>> = exo
            .iter()
            .map(|series| series[..start_bar.min(series.len())].to_vec())
            .collect();
        let prev_mid: Vec<f64> = exo.iter().map(|s| s[start_bar.min(s.len() - 1)]).collect();
        // Seed each symbol's vol proxy from its untraded burn-in tape (cleared == exogenous,
        // all strictly before the first traded bar), so vol scaling is live from bar one.
        let vol: Vec<VolTracker> = cleared_history
            .iter()
            .map(|series| {
                let mut tracker = VolTracker::new();
                for w in series.windows(2) {
                    if w[0].abs() > EPS {
                        tracker.push((w[1] - w[0]) / w[0]);
                    }
                }
                tracker
            })
            .collect();
        let agents = (0..n_agents)
            .map(|_| AgentBook {
                cash: capital,
                shares: vec![0.0; n_sym],
                cost_basis: vec![0.0; n_sym],
                prev_weight: vec![0.0; n_sym],
            })
            .collect();
        MarketClearing {
            symbols,
            dates: data.dates.clone(),
            exo,
            capital,
            impact_mult: vec![1.0; n_sym],
            prev_mid,
            cleared_history,
            vol,
            agents,
            cursor: start_bar,
            start_bar,
            n_bars,
            richness,
        }
    }

    /// The observation-richness disclosure this market surfaces (the information-poverty
    /// difficulty axis).
    pub fn richness(&self) -> ObservationRichness {
        self.richness
    }

    /// The sorted symbol axis (canonical order for every per-symbol vector).
    pub fn symbols(&self) -> &[String] {
        &self.symbols
    }

    /// The date axis (full path).
    pub fn dates(&self) -> &[String] {
        &self.dates
    }

    /// The number of agents sharing the book.
    pub fn n_agents(&self) -> usize {
        self.agents.len()
    }

    /// The total number of bars on the path.
    pub fn n_bars(&self) -> usize {
        self.n_bars
    }

    /// The next bar index to clear.
    pub fn cursor(&self) -> usize {
        self.cursor
    }

    /// The first bar cleared with trading (after the warm-up burn-in).
    pub fn start_bar(&self) -> usize {
        self.start_bar
    }

    /// The per-agent starting capital.
    pub fn capital(&self) -> f64 {
        self.capital
    }

    /// Whether the path is exhausted (no more bars to clear).
    pub fn is_done(&self) -> bool {
        self.cursor >= self.n_bars
    }

    /// The exogenous mids at the current cursor bar (clamped to the last bar).
    pub fn exo_mid_at_cursor(&self) -> Vec<f64> {
        let bar = self.cursor.min(self.n_bars - 1);
        self.exo.iter().map(|s| s[bar.min(s.len() - 1)]).collect()
    }

    /// The pre-trade observations for each agent's **first** decision: the burn-in cleared
    /// tape (untraded, so cleared == exogenous) terminated by the first traded bar's
    /// exogenous reference mid. Agents in canonical order.
    pub fn initial_observations(&self) -> Vec<MarketObservation> {
        let date = self.dates.get(self.start_bar).cloned().unwrap_or_default();
        (0..self.agents.len())
            .map(|agent| {
                let symbols = self
                    .symbols
                    .iter()
                    .enumerate()
                    .map(|(s, sym)| {
                        let mut hist = self.cleared_history[s].clone();
                        hist.push(self.exo[s][self.start_bar.min(self.exo[s].len() - 1)]);
                        self.snapshot(sym, &hist)
                    })
                    .collect();
                self.observation(agent, &date, symbols)
            })
            .collect()
    }

    /// Clear the current cursor bar, feeding [`clear_bar`] this bar's exogenous mids.
    /// Convenience over the free function for callers driving the stored path directly.
    pub fn step(&mut self, agent_orders: &[Vec<f64>], params: &MarketParams) -> ClearResult {
        let exo_mid = self.exo_mid_at_cursor();
        clear_bar(&exo_mid, agent_orders, params, self)
    }

    /// [`step`](Self::step) against an optional [`EllipticUncertaintySet`]. `None` is the
    /// point estimate and runs the identical arithmetic as [`step`](Self::step).
    pub fn step_robust(
        &mut self,
        agent_orders: &[Vec<f64>],
        params: &MarketParams,
        uncertainty: Option<&EllipticUncertaintySet>,
    ) -> ClearResult {
        let exo_mid = self.exo_mid_at_cursor();
        clear_bar_robust(&exo_mid, agent_orders, params, uncertainty, self)
    }

    /// [`step_robust`](Self::step_robust) with an explicit permanent-impact exponent.
    /// `impact_exponent = 1.0` (with `uncertainty = None`) is the point-estimate linear
    /// path and runs the identical arithmetic as [`step`](Self::step); see
    /// [`clear_bar_concave`].
    pub fn step_concave(
        &mut self,
        agent_orders: &[Vec<f64>],
        params: &MarketParams,
        uncertainty: Option<&EllipticUncertaintySet>,
        impact_exponent: f64,
    ) -> ClearResult {
        let exo_mid = self.exo_mid_at_cursor();
        clear_bar_concave(
            &exo_mid,
            agent_orders,
            params,
            uncertainty,
            impact_exponent,
            self,
        )
    }

    /// Assemble one agent's observation from a prepared symbol-snapshot list: its own
    /// cash and per-symbol holdings (with a displayed average entry price). Holdings are
    /// the agent's own — never any peer's pending state.
    fn observation(
        &self,
        agent: usize,
        date: &str,
        symbols: Vec<SymbolSnapshot>,
    ) -> MarketObservation {
        let book = &self.agents[agent];
        let portfolio = self
            .symbols
            .iter()
            .enumerate()
            .map(|(s, sym)| {
                let shares = book.shares[s];
                let avg_price = if shares.abs() > EPS {
                    (book.cost_basis[s] / shares).abs()
                } else {
                    0.0
                };
                PositionState {
                    symbol: sym.clone(),
                    shares,
                    avg_price,
                }
            })
            .collect();
        MarketObservation {
            date: date.to_string(),
            cash: book.cash,
            symbols,
            portfolio,
        }
    }

    /// Assemble one symbol's snapshot from its full cleared/exogenous history, honoring the
    /// market's [`ObservationRichness`]: surface the last `richness.lookback` closes, and
    /// populate `fundamentals` / `news` from point-in-time derived context only when the
    /// corresponding flag is set. Every input close is `<= t` (it is drawn from the cleared
    /// tape or the current reference mid), so the snapshot is leak-free at any richness.
    fn snapshot(&self, symbol: &str, full_history: &[f64]) -> SymbolSnapshot {
        let close_history = trailing(full_history, self.richness.lookback);
        let fundamentals = if self.richness.fundamentals {
            derive_fundamentals(&close_history)
        } else {
            BTreeMap::new()
        };
        let news = if self.richness.news {
            derive_news(symbol, &close_history)
        } else {
            Vec::new()
        };
        SymbolSnapshot {
            symbol: symbol.to_string(),
            close_history,
            fundamentals,
            news,
        }
    }
}

/// Trailing closes (≤ `lookback`) ending at the last entry of `series`. `lookback` is the
/// [`ObservationRichness::lookback`] disclosure; the historical default is
/// [`DEFAULT_LOOKBACK`](crate::DEFAULT_LOOKBACK).
fn trailing(series: &[f64], lookback: usize) -> Vec<f64> {
    let start = series.len().saturating_sub(lookback);
    series[start..].to_vec()
}

/// Point-in-time fundamentals derived purely from the surfaced trailing closes (all `<= t`),
/// so the map is leak-free by construction. Only mul/add/div/min/max are used (no
/// transcendentals), so it stays byte-identical across runtimes. Empty when the window is
/// too short to define a trailing return.
fn derive_fundamentals(closes: &[f64]) -> BTreeMap<String, f64> {
    let mut map = BTreeMap::new();
    if closes.len() < 2 {
        return map;
    }
    let first = closes[0];
    let last = closes[closes.len() - 1];
    let trailing_return = if first.abs() > EPS {
        last / first - 1.0
    } else {
        0.0
    };
    let mut high = closes[0];
    let mut low = closes[0];
    for &c in &closes[1..] {
        if c > high {
            high = c;
        }
        if c < low {
            low = c;
        }
    }
    map.insert("trailing_return".to_string(), trailing_return);
    map.insert("window_high".to_string(), high);
    map.insert("window_low".to_string(), low);
    map
}

/// A single point-in-time news headline derived purely from the surfaced trailing closes
/// (all `<= t`), so it is leak-free. The wording is a deterministic function of the trailing
/// return's sign and magnitude; `f64` formatting is deterministic across runtimes. Empty
/// when the window is too short to define a trailing return.
fn derive_news(symbol: &str, closes: &[f64]) -> Vec<String> {
    if closes.len() < 2 {
        return Vec::new();
    }
    let first = closes[0];
    let last = closes[closes.len() - 1];
    let ret = if first.abs() > EPS {
        last / first - 1.0
    } else {
        0.0
    };
    let pct = ret * 100.0;
    let bars = closes.len();
    let headline = if ret > NEWS_THRESHOLD {
        format!("{symbol}: uptrend, {pct:+.2}% over {bars} bars")
    } else if ret < -NEWS_THRESHOLD {
        format!("{symbol}: downtrend, {pct:+.2}% over {bars} bars")
    } else {
        format!("{symbol}: range-bound, {pct:+.2}% over {bars} bars")
    };
    vec![headline]
}

/// Clear one bar of the endogenous market.
///
/// `exo_mid[s]` is the exogenous (fundamental) mid for symbol `s` at `state.cursor`;
/// `agent_orders[i][s]` is agent `i`'s target weight for symbol `s` (canonical order). The
/// function (1) forms the cleared reference mid from the *prior* accumulated impact, (2)
/// converts each agent's weight change to a signed size and aggregates net flow in sorted
/// agent order, (3) fills each agent at its Almgren-Chriss temporary-impact price and
/// books its bar return, (4) extends the cleared tape and builds each agent's next
/// observation, and (5) folds this bar's flow into the permanent-impact multiplier for the
/// next bar. See the module docs for the equations and the leak-free argument.
///
/// This is the point-estimate path and is exactly [`clear_bar_robust`] with no uncertainty
/// set. Its dynamics are frozen: published results and the cross-runtime golden hashes are
/// pinned to them.
pub fn clear_bar(
    exo_mid: &[f64],
    agent_orders: &[Vec<f64>],
    params: &MarketParams,
    state: &mut MarketClearing,
) -> ClearResult {
    clear_bar_robust(exo_mid, agent_orders, params, None, state)
}

/// [`clear_bar`] against an optional [`EllipticUncertaintySet`] over the impact
/// coefficients.
///
/// With `uncertainty = None` this *is* [`clear_bar`]: every symbol resolves to
/// `(params.lambda, params.eta)` and the fill and permanent-impact expressions evaluate the
/// same float operations on the same values, so the cleared path is bit-for-bit the
/// point-estimate path and [`ClearResult::robust_impact`] is `None`.
///
/// With a set supplied, each symbol is cleared at that bar's worst case inside the set. The
/// cost direction is the bar's own realised flow: `cost_lambda = Q_s^2` (what the crowd's
/// net flow costs, `Q_s` shares moved at `lambda * Q_s / V` per share) and
/// `cost_eta = sum_i q_i^2` (what each agent's own size costs itself). Both are aggregate
/// quantities of the whole book rather than any one agent's, so the resolved coefficients
/// do not depend on which agent is being evaluated, and both are non-negative, so the
/// direction is well defined whenever anything traded. The same resolved `lambda` drives
/// both the temporary fill and the permanent multiplier for the next bar, so the robust
/// market stays internally consistent rather than robustifying the execution cost while
/// leaving the price it feeds back into unrobustified.
///
/// The set adds no state: the coefficients are recomputed from each bar's flow, so the
/// function remains a pure function of `(exogenous path, params, set, actions in sorted
/// order)`, with no RNG, no clock, and no I/O.
pub fn clear_bar_robust(
    exo_mid: &[f64],
    agent_orders: &[Vec<f64>],
    params: &MarketParams,
    uncertainty: Option<&EllipticUncertaintySet>,
    state: &mut MarketClearing,
) -> ClearResult {
    clear_bar_concave(exo_mid, agent_orders, params, uncertainty, 1.0, state)
}

/// [`clear_bar_robust`] with an explicit **permanent-impact exponent**: the concavity
/// ablation that makes the manipulation probe falsifiable (see the module docs).
///
/// With `impact_exponent = 1.0` this *is* [`clear_bar_robust`]: `signed_pow` then
/// returns the flow unchanged (the same bits, no `powf` evaluated), so the cleared path is
/// bit-for-bit the frozen linear path the golden hashes are pinned to. Any other exponent
/// replaces the normalized flow `Q/V` with `sign(Q/V) * |Q/V|^exponent` in the permanent
/// (Kyle) term of both the temporary fill and the permanent multiplier update; the
/// Almgren-Chriss own-size term stays linear. A general power requires `powf`, a libm
/// transcendental outside the
/// mul/add/div-only cross-runtime guarantee, which is why the exponent is gated onto these
/// opt-in entry points exactly as the [`EllipticUncertaintySet`] is and never touches
/// [`clear_bar`].
///
/// When both an uncertainty set and a non-unit exponent are supplied, the set's cost
/// direction is still formed from the raw flow (`Q^2`, `sum q_i^2`): the set stresses the
/// coefficients, the exponent reshapes the flow term, and the two compose.
pub fn clear_bar_concave(
    exo_mid: &[f64],
    agent_orders: &[Vec<f64>],
    params: &MarketParams,
    uncertainty: Option<&EllipticUncertaintySet>,
    impact_exponent: f64,
    state: &mut MarketClearing,
) -> ClearResult {
    assert!(
        impact_exponent > 0.0,
        "impact_exponent must be positive (1.0 = linear)"
    );
    let n_sym = state.symbols.len();
    let n_agents = state.agents.len();
    assert_eq!(exo_mid.len(), n_sym, "exo_mid must cover every symbol");
    assert_eq!(agent_orders.len(), n_agents, "one order vector per agent");
    for orders in agent_orders {
        assert_eq!(
            orders.len(),
            n_sym,
            "each order vector must cover every symbol"
        );
    }

    let v = params.volume_scale;

    // (1) cleared reference mid = exogenous mid * accumulated permanent impact (prior bars).
    let cleared_mid: Vec<f64> = exo_mid
        .iter()
        .zip(&state.impact_mult)
        .map(|(m, mult)| m * mult)
        .collect();

    // Volatility-scaling factor for temporary impact (one per symbol). Read the trailing
    // vol proxy — which holds only *past* cleared returns — and form a capped multiplier.
    // `vol_scale == 0` yields an exact `1.0`, so the fill below is byte-identical to the
    // pre-vol-scaling formula.
    let vol_factor: Vec<f64> = (0..n_sym)
        .map(|s| {
            if params.vol_scale > 0.0 {
                let f = 1.0 + params.vol_scale * state.vol[s].proxy();
                if f > VOL_FACTOR_CAP {
                    VOL_FACTOR_CAP
                } else {
                    f
                }
            } else {
                1.0
            }
        })
        .collect();

    // (2) per-agent signed order size q = Δ(desired notional) / price, then aggregate net
    //     flow per symbol by folding the agents in canonical (sorted) order.
    let q: Vec<Vec<f64>> = agent_orders
        .iter()
        .enumerate()
        .map(|(i, orders)| {
            let prev = &state.agents[i].prev_weight;
            orders
                .iter()
                .zip(prev)
                .zip(&cleared_mid)
                .map(|((w, pw), mid)| state.capital * (w - pw) / mid)
                .collect()
        })
        .collect();
    let mut net_flow = vec![0.0_f64; n_sym];
    for agent_q in &q {
        for (s, qis) in agent_q.iter().enumerate() {
            net_flow[s] += qis;
        }
    }

    // (2b) resolve the impact coefficients this bar clears at. Absent an uncertainty set
    //      that is the point estimate for every symbol and the arithmetic below is
    //      unchanged; with one, it is the worst case in the direction of the bar's own
    //      aggregate cost. Folded in sorted agent order, like the flow itself.
    let robust_impact: Option<Vec<ImpactCoefficients>> = uncertainty.map(|set| {
        (0..n_sym)
            .map(|s| {
                let mut own_cost = 0.0_f64;
                for agent_q in &q {
                    own_cost += agent_q[s] * agent_q[s];
                }
                set.worst_case(params, net_flow[s] * net_flow[s], own_cost)
            })
            .collect()
    });
    let impact: Vec<ImpactCoefficients> = match &robust_impact {
        Some(resolved) => resolved.clone(),
        None => vec![
            ImpactCoefficients {
                lambda: params.lambda,
                eta: params.eta,
            };
            n_sym
        ],
    };

    // (3) fill each agent at its temporary-impact price, advance its book, and book the
    //     bar's realized return (marked at the cleared mids, paid at its own fills).
    let mut fills: Vec<Vec<AgentFill>> = Vec::with_capacity(n_agents);
    let mut rewards = vec![0.0_f64; n_agents];
    let mut navs = vec![0.0_f64; n_agents];
    for i in 0..n_agents {
        // NAV before this bar's price move, marked at the prior cleared mid.
        let nav_prev = {
            let book = &state.agents[i];
            book.cash
                + book
                    .shares
                    .iter()
                    .zip(&state.prev_mid)
                    .map(|(sh, m)| sh * m)
                    .sum::<f64>()
        };
        let mut agent_fills = Vec::with_capacity(n_sym);
        for s in 0..n_sym {
            let qi = q[i][s];
            let mid = cleared_mid[s];
            // Keep the published linear arithmetic byte-identical. A nonlinear
            // exponent acts on dimensionless crowd flow Q/V; the own-size term remains
            // linear and normalized by V.
            let impact_term = if impact_exponent == 1.0 {
                (impact[s].lambda * net_flow[s] + impact[s].eta * qi) / v
            } else {
                impact[s].lambda * signed_pow(net_flow[s] / v, impact_exponent)
                    + impact[s].eta * qi / v
            };
            let fill = mid * (1.0 + vol_factor[s] * impact_term);
            let sym = state.symbols[s].clone();
            let book = &mut state.agents[i];
            book.cash -= qi * fill;
            let new_shares = book.shares[s] + qi;
            if new_shares.abs() < EPS {
                book.cost_basis[s] = 0.0;
            } else {
                book.cost_basis[s] += qi * fill;
            }
            book.shares[s] = new_shares;
            book.prev_weight[s] = agent_orders[i][s];
            agent_fills.push(AgentFill {
                symbol: sym,
                size: qi,
                fill_price: fill,
            });
        }
        // NAV after trading, marked at the new cleared mid.
        let nav_post = {
            let book = &state.agents[i];
            book.cash
                + book
                    .shares
                    .iter()
                    .zip(&cleared_mid)
                    .map(|(sh, m)| sh * m)
                    .sum::<f64>()
        };
        navs[i] = nav_post;
        rewards[i] = if nav_prev.abs() > EPS {
            (nav_post - nav_prev) / nav_prev
        } else {
            0.0
        };
        fills.push(agent_fills);
    }

    // (4) extend the realized cleared tape, then build each agent's next observation.
    for (hist, mid) in state.cleared_history.iter_mut().zip(&cleared_mid) {
        hist.push(*mid);
    }
    let date = state.dates.get(state.cursor).cloned().unwrap_or_default();
    let observations: Vec<MarketObservation> = (0..n_agents)
        .map(|agent| {
            let symbols = state
                .symbols
                .iter()
                .enumerate()
                .map(|(s, sym)| state.snapshot(sym, &state.cleared_history[s]))
                .collect();
            state.observation(agent, &date, symbols)
        })
        .collect();

    // (5) permanent impact accumulates into the running multiplier for the next bar; the
    //     cleared mid becomes the mark for the next bar's holding PnL.
    for (s, (mult, flow)) in state.impact_mult.iter_mut().zip(&net_flow).enumerate() {
        let permanent = if impact_exponent == 1.0 {
            impact[s].lambda * *flow / v
        } else {
            impact[s].lambda * signed_pow(*flow / v, impact_exponent)
        };
        *mult *= 1.0 + permanent;
    }
    // Fold this bar's realized cleared return into each symbol's trailing vol proxy, for the
    // *next* bar's scaling. `prev_mid` still holds the previous bar's cleared mid here.
    for (s, mid) in cleared_mid.iter().enumerate() {
        let prev = state.prev_mid[s];
        if prev.abs() > EPS {
            state.vol[s].push((mid - prev) / prev);
        }
    }
    state.prev_mid.copy_from_slice(&cleared_mid);
    state.cursor += 1;
    let done = state.cursor >= state.n_bars;

    ClearResult {
        cleared_mids: cleared_mid,
        net_flow,
        rewards,
        navs,
        fills,
        observations,
        done,
        robust_impact,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A flat-everywhere order block (`n_agents × n_sym`, all weight `w`).
    fn block(n_agents: usize, n_sym: usize, w: f64) -> Vec<Vec<f64>> {
        vec![vec![w; n_sym]; n_agents]
    }

    #[test]
    fn zero_flow_reproduces_the_exogenous_path() {
        // N agents all flat == the frozen path: cleared price equals the exogenous close
        // at every bar, and every reward is exactly zero.
        let data = Dataset::synthetic(3, 50, 4);
        let params = MarketParams::default();
        let mut m = MarketClearing::from_dataset(&data, 3, 1.0);
        let flat = block(3, 3, 0.0);
        loop {
            let bar = m.cursor();
            let r = m.step(&flat, &params);
            for (s, mid) in r.cleared_mids.iter().enumerate() {
                let exo = data.close_at(&m.symbols()[s], bar).unwrap();
                assert_eq!(
                    *mid, exo,
                    "flat flow must leave the cleared price == exogenous"
                );
            }
            assert!(r.net_flow.iter().all(|f| *f == 0.0));
            assert!(r.rewards.iter().all(|x| *x == 0.0));
            if r.done {
                break;
            }
        }
    }

    #[test]
    fn a_coordinated_buy_lifts_the_cleared_price_above_exogenous() {
        // A large coordinated buy raises the cleared price: positive flow on the entry
        // bar, and the *permanent* component keeps the cleared mid above the exogenous
        // path on subsequent bars even after the target is reached (no new flow).
        let data = Dataset::synthetic(2, 40, 6);
        let params = MarketParams {
            lambda: 0.5,
            eta: 0.0,
            volume_scale: 1.0,
            vol_scale: 0.0,
        };
        let mut m = MarketClearing::from_dataset(&data, 4, 1.0);
        let buy = block(4, 2, 0.8);

        // Entry bar: net buy flow is strictly positive (impact applies to the NEXT bar).
        let entry = m.step(&buy, &params);
        assert!(
            entry.net_flow.iter().all(|f| *f > 0.0),
            "the entry bar must show positive net buy flow"
        );

        // Holding the same target: no fresh flow, but the permanent bump persists, so the
        // cleared price now sits strictly above the exogenous close.
        let bar = m.cursor();
        let hold = m.step(&buy, &params);
        assert!(
            hold.net_flow.iter().all(|f| f.abs() < EPS),
            "no new flow once the target weight is reached"
        );
        for (s, mid) in hold.cleared_mids.iter().enumerate() {
            let exo = data.close_at(&m.symbols()[s], bar).unwrap();
            assert!(
                *mid > exo,
                "permanent impact must keep cleared {mid} above exogenous {exo}"
            );
        }
    }

    #[test]
    fn permanent_impact_accumulates_under_sustained_flow() {
        // Ramp the target weight each bar to sustain positive flow; the cleared/exogenous
        // ratio (the observed accumulated multiplier) is non-decreasing and ends higher.
        let data = Dataset::synthetic(1, 60, 5);
        let params = MarketParams {
            lambda: 0.2,
            eta: 0.0,
            volume_scale: 5.0,
            vol_scale: 0.0,
        };
        let mut m = MarketClearing::from_dataset(&data, 2, 1.0);
        let mut w = 0.0;
        let mut ratios = Vec::new();
        loop {
            w += 0.2;
            let bar = m.cursor();
            let r = m.step(&vec![vec![w]; 2], &params);
            let exo = data.close_at(&m.symbols()[0], bar).unwrap();
            ratios.push(r.cleared_mids[0] / exo);
            if r.done || ratios.len() >= 20 {
                break;
            }
        }
        for win in ratios.windows(2) {
            assert!(
                win[1] >= win[0] - EPS,
                "the impact multiplier must not shrink under sustained buying: {ratios:?}"
            );
        }
        assert!(
            *ratios.last().unwrap() > ratios[0] + 1e-9,
            "sustained buying must lift the multiplier: {ratios:?}"
        );
    }

    #[test]
    fn identical_inputs_yield_identical_results() {
        // Determinism: the same path + params + actions reproduce byte-identical
        // observations, rewards, and cleared mids; different actions diverge.
        let data = Dataset::synthetic(3, 45, 12);
        let params = MarketParams {
            lambda: 0.3,
            eta: 0.15,
            volume_scale: 2.0,
            vol_scale: 0.0,
        };
        let run = |weight: f64| {
            let mut m = MarketClearing::from_dataset(&data, 3, 1.0);
            let orders = block(3, 3, weight);
            let mut log: Vec<(String, Vec<f64>, Vec<f64>)> = Vec::new();
            loop {
                let r = m.step(&orders, &params);
                log.push((
                    serde_json::to_string(&r.observations).unwrap(),
                    r.rewards.clone(),
                    r.cleared_mids.clone(),
                ));
                if r.done {
                    break;
                }
            }
            log
        };
        assert_eq!(run(0.5), run(0.5), "identical inputs must be identical");
        assert_ne!(run(0.5), run(0.2), "different actions must diverge");
    }

    #[test]
    fn aggregation_is_canonical_order_independent() {
        // The net flow is defined as the fold of per-agent sizes in sorted agent order, so
        // however a parallel collector assembles the actions, materializing them in
        // canonical order yields the identical cleared result.
        let data = Dataset::synthetic(2, 40, 3);
        let params = MarketParams {
            lambda: 0.4,
            eta: 0.1,
            volume_scale: 1.0,
            vol_scale: 0.0,
        };
        // Distinct per-agent orders so the ordering would matter if it weren't canonical.
        let agent_orders: Vec<Vec<f64>> = (0..4)
            .map(|i| vec![0.1 * (i as f64 + 1.0), -0.05 * (i as f64)])
            .collect();

        let mut direct = MarketClearing::from_dataset(&data, 4, 1.0);
        let rd = direct.step(&agent_orders, &params);

        // Simulate parallel completion: insert into a map in reverse order, then read it
        // back in sorted-key (canonical) order.
        let mut map: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
        for i in (0..4).rev() {
            map.insert(i, agent_orders[i].clone());
        }
        let reassembled: Vec<Vec<f64>> = map.into_values().collect();
        let mut shuffled = MarketClearing::from_dataset(&data, 4, 1.0);
        let rs = shuffled.step(&reassembled, &params);

        assert_eq!(rd.net_flow, rs.net_flow);
        assert_eq!(rd.cleared_mids, rs.cleared_mids);
        assert_eq!(
            serde_json::to_string(&rd.observations).unwrap(),
            serde_json::to_string(&rs.observations).unwrap()
        );
    }

    #[test]
    fn peer_order_does_not_leak_into_own_sizing_or_cleared_price() {
        // Leak-free: vary agent 1's bar-t order; agent 0's traded size and the bar's
        // cleared reference mid are invariant (both depend only on flow strictly before t).
        // Agent 0's fill *price* does move with the realized aggregate flow — that is the
        // price-impact channel of a shared market, not a peer-intent leak.
        let data = Dataset::synthetic(2, 40, 8);
        let params = MarketParams {
            lambda: 0.5,
            eta: 0.2,
            volume_scale: 1.0,
            vol_scale: 0.0,
        };
        let mut m1 = MarketClearing::from_dataset(&data, 2, 1.0);
        let mut m2 = MarketClearing::from_dataset(&data, 2, 1.0);
        let with_flat_peer = vec![vec![0.3, 0.0], vec![0.0, 0.0]];
        let with_buying_peer = vec![vec![0.3, 0.0], vec![0.9, 0.5]];
        let r1 = m1.step(&with_flat_peer, &params);
        let r2 = m2.step(&with_buying_peer, &params);

        assert_eq!(
            r1.cleared_mids, r2.cleared_mids,
            "the cleared mid at t embeds only prior-bar flow, so a peer's t-order can't move it"
        );
        let sizes1: Vec<f64> = r1.fills[0].iter().map(|f| f.size).collect();
        let sizes2: Vec<f64> = r2.fills[0].iter().map(|f| f.size).collect();
        assert_eq!(
            sizes1, sizes2,
            "agent 0's traded size depends only on its own weights and the cleared mid"
        );
        let px1: Vec<f64> = r1.fills[0].iter().map(|f| f.fill_price).collect();
        let px2: Vec<f64> = r2.fills[0].iter().map(|f| f.fill_price).collect();
        assert_ne!(
            px1, px2,
            "the realized fill price reflects aggregate flow — impact, not a leak"
        );
    }

    #[test]
    fn initial_observation_has_warmup_history_and_no_positions() {
        let data = Dataset::synthetic(3, 60, 1);
        let m = MarketClearing::from_dataset(&data, 2, 1.0);
        let obs = m.initial_observations();
        assert_eq!(obs.len(), 2);
        for o in &obs {
            assert_eq!(o.cash, 1.0);
            assert!(o.portfolio.iter().all(|p| p.shares == 0.0));
            for snap in &o.symbols {
                assert!(
                    !snap.close_history.is_empty(),
                    "warm-up history must be present"
                );
                // The burn-in tape is untraded, so it equals the exogenous closes; its last
                // entry is the first traded bar's exogenous reference mid.
                let last = *snap.close_history.last().unwrap();
                assert_eq!(last, data.close_at(&snap.symbol, m.start_bar()).unwrap());
            }
        }
    }

    #[test]
    fn done_flips_on_the_final_bar() {
        let data = Dataset::synthetic(2, 24, 2);
        let mut m = MarketClearing::from_dataset(&data, 2, 1.0);
        let flat = block(2, 2, 0.0);
        let mut steps = 0;
        loop {
            let r = m.step(&flat, &params_default());
            steps += 1;
            assert_eq!(r.observations.len(), 2);
            if r.done {
                break;
            }
        }
        assert_eq!(steps, m.n_bars() - m.start_bar());
        assert!(m.is_done());
    }

    fn params_default() -> MarketParams {
        MarketParams::default()
    }

    /// A single-symbol dataset over a handcrafted close path (for calm-vs-volatile tests).
    fn dataset_from_closes(closes: Vec<f64>) -> Dataset {
        let dates = (0..closes.len()).map(|i| format!("t{i}")).collect();
        let mut map = BTreeMap::new();
        map.insert("AAA".to_string(), closes);
        Dataset {
            dates,
            closes: map,
            dividends: BTreeMap::new(),
        }
    }

    #[test]
    fn vol_scale_zero_matches_the_legacy_fill_formula() {
        // vol_scale = 0 leaves the temporary-impact term exactly as before: every fill is
        // mid * (1 + (lambda*Q + eta*q_i)/V), recomputed here from the public result fields.
        // This pins the default path byte-for-byte against the pre-change formula.
        let data = Dataset::synthetic(2, 40, 3);
        let params = MarketParams {
            lambda: 0.5,
            eta: 0.25,
            volume_scale: 2.0,
            vol_scale: 0.0,
        };
        let mut m = MarketClearing::from_dataset(&data, 3, 1.0);
        let orders = block(3, 2, 0.6);
        loop {
            let r = m.step(&orders, &params);
            for fills in &r.fills {
                for (s, f) in fills.iter().enumerate() {
                    let expected = r.cleared_mids[s]
                        * (1.0
                            + (params.lambda * r.net_flow[s] + params.eta * f.size)
                                / params.volume_scale);
                    assert_eq!(
                        f.fill_price, expected,
                        "vol_scale=0 must be the legacy fill"
                    );
                }
            }
            if r.done {
                break;
            }
        }
    }

    #[test]
    fn vol_scaling_widens_fills_more_in_a_high_vol_stretch() {
        // Calm path (tiny returns) vs volatile path (large alternating returns), each long
        // enough to seed the trailing-vol buffer from the burn-in tape. With the SAME orders
        // and path, vol scaling moves only the fill PRICE (sizing + cleared mid are
        // untouched), so the realized widening factor equals the vol multiplier — which must
        // be larger on the volatile path.
        let calm = dataset_from_closes((0..30).map(|i| 100.0 + i as f64 * 0.01).collect());
        let volatile = dataset_from_closes(
            (0..30)
                .map(|i| if i % 2 == 0 { 100.0 } else { 125.0 })
                .collect(),
        );
        let base = MarketParams {
            lambda: 0.4,
            eta: 0.2,
            volume_scale: 1.0,
            vol_scale: 0.0,
        };
        let scaled = MarketParams {
            vol_scale: 10.0,
            ..base
        };
        let buy = block(2, 1, 0.8);

        let widening = |data: &Dataset| {
            let mut m0 = MarketClearing::from_dataset(data, 2, 1.0);
            let mut mv = MarketClearing::from_dataset(data, 2, 1.0);
            let r0 = m0.step(&buy, &base);
            let rv = mv.step(&buy, &scaled);
            let mid = r0.cleared_mids[0];
            assert_eq!(
                mid, rv.cleared_mids[0],
                "vol scaling must not move the cleared mid"
            );
            let base_impact = r0.fills[0][0].fill_price - mid;
            assert!(base_impact.abs() > EPS, "the entry bar must actually trade");
            (rv.fills[0][0].fill_price - mid) / base_impact
        };

        let calm_factor = widening(&calm);
        let vol_factor = widening(&volatile);
        assert!(
            calm_factor >= 1.0 - EPS,
            "the factor never shrinks impact: {calm_factor}"
        );
        assert!(
            vol_factor > calm_factor + 1e-6,
            "a high-vol stretch must widen fills more than a calm one: \
             vol={vol_factor} calm={calm_factor}"
        );
    }

    #[test]
    fn the_vol_factor_is_capped() {
        // An extreme vol_scale on a volatile path saturates the cap: the realized widening
        // factor cannot exceed VOL_FACTOR_CAP however large vol_scale (or the vol) grows.
        let volatile = dataset_from_closes(
            (0..30)
                .map(|i| if i % 2 == 0 { 100.0 } else { 140.0 })
                .collect(),
        );
        let base = MarketParams {
            lambda: 0.4,
            eta: 0.2,
            volume_scale: 1.0,
            vol_scale: 0.0,
        };
        let huge = MarketParams {
            vol_scale: 1.0e9,
            ..base
        };
        let buy = block(2, 1, 0.8);

        let mut m0 = MarketClearing::from_dataset(&volatile, 2, 1.0);
        let mut mh = MarketClearing::from_dataset(&volatile, 2, 1.0);
        let r0 = m0.step(&buy, &base);
        let rh = mh.step(&buy, &huge);
        let mid = r0.cleared_mids[0];
        let factor = (rh.fills[0][0].fill_price - mid) / (r0.fills[0][0].fill_price - mid);
        assert!(
            factor <= VOL_FACTOR_CAP + 1e-9,
            "the widening factor must be capped at {VOL_FACTOR_CAP}: {factor}"
        );
        assert!(
            (factor - VOL_FACTOR_CAP).abs() < 1e-6,
            "an extreme vol_scale must saturate the cap: {factor}"
        );
    }

    #[test]
    fn vol_scaled_clearing_is_deterministic() {
        // Determinism with vol_scale active: same path + params + actions reproduce
        // byte-identical fills and observations.
        let data = Dataset::synthetic(2, 45, 9);
        let params = MarketParams {
            lambda: 0.3,
            eta: 0.15,
            volume_scale: 2.0,
            vol_scale: 4.0,
        };
        let run = || {
            let mut m = MarketClearing::from_dataset(&data, 3, 1.0);
            let orders = block(3, 2, 0.4);
            let mut log: Vec<(Vec<f64>, String)> = Vec::new();
            loop {
                let r = m.step(&orders, &params);
                let px: Vec<f64> = r.fills.iter().flatten().map(|f| f.fill_price).collect();
                log.push((px, serde_json::to_string(&r.observations).unwrap()));
                if r.done {
                    break;
                }
            }
            log
        };
        assert_eq!(run(), run(), "vol-scaled clearing must be deterministic");
    }

    // --- Elliptic impact-uncertainty axis --------------------------------------------------

    /// Serialize a whole `ClearResult` (every public field, including the observations and
    /// the fills), so a comparison is over the complete cleared bar and not a chosen subset.
    fn result_blob(r: &ClearResult) -> String {
        serde_json::to_string(r).unwrap()
    }

    /// Roll `orders` to exhaustion under an optional uncertainty set, collecting the full
    /// serialized result of every bar.
    fn robust_rollout(
        data: &Dataset,
        n_agents: usize,
        params: &MarketParams,
        uncertainty: Option<&EllipticUncertaintySet>,
        orders: &[Vec<f64>],
    ) -> Vec<String> {
        let mut m = MarketClearing::from_dataset(data, n_agents, 1.0);
        let mut log = Vec::new();
        loop {
            let r = m.step_robust(orders, params, uncertainty);
            log.push(result_blob(&r));
            if r.done {
                break;
            }
        }
        log
    }

    #[test]
    fn absent_uncertainty_set_is_byte_identical_to_the_point_estimate() {
        // The load-bearing guarantee: an uncertainty set is opt-in, so the robust entry
        // point given `None` must reproduce `step` bit-for-bit over a full path, across
        // every result field (cleared mids, net flow, rewards, NAVs, fills, observations).
        // Published results and the cross-runtime golden hashes are pinned to this path.
        let data = Dataset::synthetic(3, 60, 17);
        let params = MarketParams {
            lambda: 0.35,
            eta: 0.18,
            volume_scale: 2.0,
            vol_scale: 3.0,
        };
        let orders: Vec<Vec<f64>> = (0..3)
            .map(|i| vec![0.2 * (i as f64 + 1.0), -0.1 * (i as f64), 0.05])
            .collect();

        let mut legacy_market = MarketClearing::from_dataset(&data, 3, 1.0);
        let mut legacy = Vec::new();
        loop {
            let r = legacy_market.step(&orders, &params);
            assert!(
                r.robust_impact.is_none(),
                "the point-estimate path reports no resolved coefficients"
            );
            legacy.push(result_blob(&r));
            if r.done {
                break;
            }
        }

        let robust_none = robust_rollout(&data, 3, &params, None, &orders);
        assert_eq!(
            legacy, robust_none,
            "clear_bar_robust(None) must be byte-identical to clear_bar"
        );
    }

    #[test]
    fn the_point_estimate_wire_shape_omits_the_robust_field() {
        // The serialized result is the Python/WASM wire shape. On the default path the
        // new field is skipped entirely, so no downstream JSON consumer sees a new key.
        let data = Dataset::synthetic(2, 30, 4);
        let mut m = MarketClearing::from_dataset(&data, 2, 1.0);
        let r = m.step(&block(2, 2, 0.4), &MarketParams::default());
        assert!(!result_blob(&r).contains("robust_impact"));
    }

    #[test]
    fn a_zero_radius_set_reproduces_the_point_estimate_path() {
        // A set of zero width is the point estimate expressed as a (degenerate) set, so it
        // must clear identically: c^T S c is 0, the worst case falls back, and every
        // resolved coefficient equals the point estimate.
        let data = Dataset::synthetic(2, 40, 11);
        let params = MarketParams {
            lambda: 0.3,
            eta: 0.12,
            volume_scale: 1.5,
            vol_scale: 0.0,
        };
        let orders = block(3, 2, 0.5);
        let set = EllipticUncertaintySet::isotropic(0.0);

        let point = robust_rollout(&data, 3, &params, None, &orders);
        let mut m = MarketClearing::from_dataset(&data, 3, 1.0);
        let mut bar = 0;
        loop {
            let r = m.step_robust(&orders, &params, Some(&set));
            for coefficients in r.robust_impact.as_ref().unwrap() {
                assert_eq!(coefficients.lambda, params.lambda);
                assert_eq!(coefficients.eta, params.eta);
            }
            // Everything but the (identical) reported coefficients must match the
            // point-estimate bar. Compare as parsed values so field ordering is irrelevant.
            // Both sides go through the same to_string + from_str round trip: without the
            // `float_roundtrip` feature serde_json's parse can land 1 ulp off, so a parsed
            // tree only compares equal to another parsed tree, not to a `to_value` one.
            let mut value: serde_json::Value = serde_json::from_str(&result_blob(&r)).unwrap();
            value.as_object_mut().unwrap().remove("robust_impact");
            let expected: serde_json::Value = serde_json::from_str(&point[bar]).unwrap();
            assert_eq!(
                value, expected,
                "a zero-radius set must clear the point-estimate path"
            );
            bar += 1;
            if r.done {
                break;
            }
        }
    }

    #[test]
    fn a_no_flow_bar_falls_back_to_the_point_estimate() {
        // With nothing traded the cost direction is the zero vector, which has no worst
        // case. The fallback keeps a flat market on the exogenous path even under a set.
        let data = Dataset::synthetic(2, 30, 6);
        let params = MarketParams::default();
        let set = EllipticUncertaintySet::new(0.5, 0.25, 0.4);
        let mut m = MarketClearing::from_dataset(&data, 2, 1.0);
        let flat = block(2, 2, 0.0);
        loop {
            let bar = m.cursor();
            let r = m.step_robust(&flat, &params, Some(&set));
            for coefficients in r.robust_impact.as_ref().unwrap() {
                assert_eq!(coefficients.lambda, params.lambda);
                assert_eq!(coefficients.eta, params.eta);
            }
            for (s, mid) in r.cleared_mids.iter().enumerate() {
                assert_eq!(*mid, data.close_at(&m.symbols()[s], bar).unwrap());
            }
            if r.done {
                break;
            }
        }
    }

    #[test]
    fn the_worst_case_attains_the_support_function_of_the_ellipse() {
        // The closed form is only correct if it lands on the boundary at the maximiser.
        // Check the attained value identity c^T theta_wc = c^T theta_hat + sqrt(c^T S c),
        // exactly the support function of the ellipse in the cost direction.
        let params = MarketParams {
            lambda: 0.4,
            eta: 0.2,
            volume_scale: 1.0,
            vol_scale: 0.0,
        };
        let set = EllipticUncertaintySet::new(0.1, 0.06, 0.3);
        let (cl, ce) = (4.0, 0.75);
        let wc = set.worst_case(&params, cl, ce);
        let (a, b, rho) = (set.lambda_radius, set.eta_radius, set.correlation);
        let quad = a * a * cl * cl + 2.0 * rho * a * b * cl * ce + b * b * ce * ce;
        let attained = cl * wc.lambda + ce * wc.eta;
        let expected = cl * params.lambda + ce * params.eta + quad.sqrt();
        assert!(
            (attained - expected).abs() < 1e-12,
            "attained {attained} != support {expected}"
        );
    }

    #[test]
    fn no_point_in_the_ellipse_costs_more_than_the_worst_case() {
        // Optimality, checked directly: sweep the boundary of the ellipse (a Cholesky map
        // of a rational parameterization of the unit circle, so the sweep needs no trig and
        // lands exactly on the circle) and confirm nothing beats the closed form.
        let params = MarketParams {
            lambda: 0.5,
            eta: 0.25,
            volume_scale: 1.0,
            vol_scale: 0.0,
        };
        for &rho in &[-0.8_f64, -0.25, 0.0, 0.25, 0.8] {
            let set = EllipticUncertaintySet::new(0.12, 0.05, rho);
            let (cl, ce) = (2.5_f64, 0.4_f64);
            let wc = set.worst_case(&params, cl, ce);
            let best = cl * wc.lambda + ce * wc.eta;
            // Cholesky of S: L = [[a, 0], [rho*b, b*sqrt(1 - rho^2)]].
            let (a, b) = (set.lambda_radius, set.eta_radius);
            let l10 = rho * b;
            let l11 = b * (1.0 - rho * rho).sqrt();
            for k in -200..=200 {
                let t = k as f64 / 50.0;
                // (u, v) is exactly on the unit circle for any t.
                let denom = 1.0 + t * t;
                let u = (1.0 - t * t) / denom;
                let v = 2.0 * t / denom;
                let lambda = params.lambda + a * u;
                let eta = params.eta + l10 * u + l11 * v;
                let cost = cl * lambda + ce * eta;
                assert!(
                    cost <= best + 1e-9,
                    "rho={rho}: boundary point costs {cost} > worst case {best}"
                );
            }
        }
    }

    #[test]
    fn a_positively_correlated_set_raises_both_coefficients() {
        // With non-negative correlation and a non-negative cost direction (which the bar's
        // own flow always is: Q^2 and sum q_i^2), S c has non-negative entries, so the
        // worst case is weakly above the point estimate on both axes and strictly above on
        // at least one.
        let params = MarketParams {
            lambda: 0.3,
            eta: 0.15,
            volume_scale: 1.0,
            vol_scale: 0.0,
        };
        let set = EllipticUncertaintySet::new(0.08, 0.04, 0.6);
        let wc = set.worst_case(&params, 9.0, 2.0);
        assert!(wc.lambda > params.lambda, "lambda must widen: {wc:?}");
        assert!(wc.eta > params.eta, "eta must widen: {wc:?}");
    }

    #[test]
    fn a_negative_correlation_is_not_a_box_corner() {
        // The whole reason for an ellipse: with negatively correlated estimation errors,
        // charging more permanent impact means charging less temporary impact. A box would
        // take its corner and raise both; the ellipse pushes one down.
        let params = MarketParams {
            lambda: 0.3,
            eta: 0.15,
            volume_scale: 1.0,
            vol_scale: 0.0,
        };
        let set = EllipticUncertaintySet::new(0.08, 0.04, -0.9);
        // A cost direction dominated by the permanent leg.
        let wc = set.worst_case(&params, 25.0, 0.5);
        assert!(wc.lambda > params.lambda, "lambda still widens: {wc:?}");
        assert!(
            wc.eta < params.eta,
            "a negatively correlated set must trade eta off against lambda: {wc:?}"
        );
    }

    #[test]
    fn resolved_coefficients_never_go_negative() {
        // A wide set on a small point estimate would otherwise turn impact into a rebate.
        let params = MarketParams {
            lambda: 0.01,
            eta: 0.01,
            volume_scale: 1.0,
            vol_scale: 0.0,
        };
        let set = EllipticUncertaintySet::new(5.0, 5.0, -1.0);
        let wc = set.worst_case(&params, 1.0, 4.0);
        assert!(wc.lambda >= 0.0 && wc.eta >= 0.0, "no rebates: {wc:?}");
    }

    #[test]
    fn a_set_makes_the_market_strictly_more_expensive_to_trade() {
        // The evaluation claim: an agent facing an uncertainty set is charged the worst
        // case, so its aggregate execution cost on a trading bar is strictly higher than
        // under the point estimate, with the sizing and the cleared mid untouched.
        let data = Dataset::synthetic(2, 40, 21);
        let params = MarketParams {
            lambda: 0.3,
            eta: 0.15,
            volume_scale: 1.0,
            vol_scale: 0.0,
        };
        let set = EllipticUncertaintySet::isotropic(0.05);
        let buy = block(3, 2, 0.7);

        let mut point_market = MarketClearing::from_dataset(&data, 3, 1.0);
        let mut robust_market = MarketClearing::from_dataset(&data, 3, 1.0);
        let point = point_market.step(&buy, &params);
        let robust = robust_market.step_robust(&buy, &params, Some(&set));

        assert_eq!(
            point.cleared_mids, robust.cleared_mids,
            "the set must not move the reference mid (it embeds only prior-bar flow)"
        );
        for (pf, rf) in point.fills.iter().zip(&robust.fills) {
            for (p, r) in pf.iter().zip(rf) {
                assert_eq!(p.size, r.size, "sizing is unchanged by the set");
                assert!(
                    r.fill_price > p.fill_price,
                    "a buyer must pay strictly more under the worst case: {} vs {}",
                    r.fill_price,
                    p.fill_price
                );
            }
        }
        for (p, r) in point.rewards.iter().zip(&robust.rewards) {
            assert!(r < p, "the robust bar return must be worse: {r} vs {p}");
        }
    }

    #[test]
    fn the_resolved_lambda_also_drives_the_permanent_multiplier() {
        // Internal consistency: the bar's worst-case lambda feeds the next bar's reference
        // price, not just the fill. A sustained buy therefore leaves the cleared mid above
        // where the point estimate would have left it.
        let data = Dataset::synthetic(1, 40, 13);
        let params = MarketParams {
            lambda: 0.2,
            eta: 0.0,
            volume_scale: 4.0,
            vol_scale: 0.0,
        };
        let set = EllipticUncertaintySet::new(0.1, 0.0, 0.0);
        let buy = block(2, 1, 0.9);

        let mut point_market = MarketClearing::from_dataset(&data, 2, 1.0);
        let mut robust_market = MarketClearing::from_dataset(&data, 2, 1.0);
        point_market.step(&buy, &params);
        robust_market.step_robust(&buy, &params, Some(&set));
        // Second bar: no fresh flow, so the difference is purely the accumulated permanent
        // impact of bar one.
        let point = point_market.step(&buy, &params);
        let robust = robust_market.step_robust(&buy, &params, Some(&set));
        assert!(
            robust.cleared_mids[0] > point.cleared_mids[0],
            "the worst-case lambda must carry into the reference price: {} vs {}",
            robust.cleared_mids[0],
            point.cleared_mids[0]
        );
    }

    #[test]
    fn robust_impact_is_reported_per_symbol_and_varies_with_flow() {
        // The reported coefficients are per symbol, because the cost direction is a
        // per-symbol quantity: two symbols traded at different sizes resolve differently.
        let data = Dataset::synthetic(2, 30, 5);
        let params = MarketParams {
            lambda: 0.3,
            eta: 0.15,
            volume_scale: 1.0,
            vol_scale: 0.0,
        };
        let set = EllipticUncertaintySet::new(0.09, 0.02, 0.0);
        let mut m = MarketClearing::from_dataset(&data, 2, 1.0);
        // Symbol 0 is crossed (the two agents offset, so net flow is zero but both still
        // pay their own size); symbol 1 is one-sided. Two genuinely different directions.
        let orders = vec![vec![0.9, 0.5], vec![-0.9, 0.5]];
        let r = m.step_robust(&orders, &params, Some(&set));
        let resolved = r.robust_impact.as_ref().unwrap();
        assert_eq!(resolved.len(), 2, "one coefficient pair per symbol");
        assert_eq!(r.net_flow[0], 0.0, "the crossed symbol has no net flow");
        assert_eq!(
            resolved[0].lambda, params.lambda,
            "with no net flow the cost direction is pure eta, so lambda is not stressed"
        );
        assert!(
            resolved[0].eta > params.eta,
            "the crossed symbol still stresses eta: {resolved:?}"
        );
        assert!(
            resolved[1].lambda > params.lambda,
            "the one-sided symbol stresses lambda too: {resolved:?}"
        );
        assert_ne!(
            resolved[0], resolved[1],
            "different per-symbol flow must resolve differently: {resolved:?}"
        );
    }

    #[test]
    fn robust_clearing_is_deterministic() {
        // No RNG, no clock, no I/O: the same path, params, set, and actions reproduce a
        // byte-identical stream of cleared bars.
        let data = Dataset::synthetic(3, 50, 8);
        let params = MarketParams {
            lambda: 0.25,
            eta: 0.1,
            volume_scale: 2.0,
            vol_scale: 2.0,
        };
        let set = EllipticUncertaintySet::new(0.07, 0.03, -0.4);
        let orders = block(3, 3, 0.45);
        let run = || robust_rollout(&data, 3, &params, Some(&set), &orders);
        assert_eq!(run(), run(), "robust clearing must be deterministic");
    }

    #[test]
    #[should_panic(expected = "correlation must lie in")]
    fn an_out_of_range_correlation_is_rejected() {
        EllipticUncertaintySet::new(0.1, 0.1, 1.5);
    }

    #[test]
    #[should_panic(expected = "radii must be non-negative")]
    fn a_negative_radius_is_rejected() {
        EllipticUncertaintySet::new(-0.1, 0.1, 0.0);
    }

    // --- Concave permanent-impact axis ------------------------------------------------------

    #[test]
    fn a_unit_exponent_is_byte_identical_to_the_linear_path() {
        // The load-bearing guarantee for the concavity gate: exponent 1.0 must reproduce
        // the frozen linear path bit-for-bit over a full rollout, across every result
        // field, with and without an uncertainty set. The golden hashes are pinned to the
        // linear path; this pins the gate to it.
        let data = Dataset::synthetic(3, 60, 23);
        let params = MarketParams {
            lambda: 0.35,
            eta: 0.18,
            volume_scale: 2.0,
            vol_scale: 3.0,
        };
        let orders: Vec<Vec<f64>> = (0..3)
            .map(|i| vec![0.2 * (i as f64 + 1.0), -0.1 * (i as f64), 0.05])
            .collect();

        let legacy = robust_rollout(&data, 3, &params, None, &orders);
        let mut m = MarketClearing::from_dataset(&data, 3, 1.0);
        let mut unit = Vec::new();
        loop {
            let r = m.step_concave(&orders, &params, None, 1.0);
            unit.push(result_blob(&r));
            if r.done {
                break;
            }
        }
        assert_eq!(
            legacy, unit,
            "step_concave(.., 1.0) must be byte-identical to step"
        );

        let set = EllipticUncertaintySet::new(0.07, 0.03, -0.4);
        let robust = robust_rollout(&data, 3, &params, Some(&set), &orders);
        let mut m = MarketClearing::from_dataset(&data, 3, 1.0);
        let mut unit_robust = Vec::new();
        loop {
            let r = m.step_concave(&orders, &params, Some(&set), 1.0);
            unit_robust.push(result_blob(&r));
            if r.done {
                break;
            }
        }
        assert_eq!(
            robust, unit_robust,
            "exponent 1.0 must compose with an uncertainty set unchanged"
        );
    }

    /// First-bar fill impact (fill price minus cleared mid, agent 0, symbol 0) under a
    /// given exponent, on a flat 100.0 path so the cleared mid and the sizing are shared
    /// with the linear run. `eta = 0` so the entire impact term is the permanent leg.
    fn first_bar_crowd_impact(capital: f64, exponent: f64) -> (f64, f64) {
        let data = dataset_from_closes(vec![100.0; 30]);
        let params = MarketParams {
            lambda: 0.5,
            eta: 0.0,
            volume_scale: 1.0,
            vol_scale: 0.0,
        };
        let buy = block(2, 1, 0.8);
        let mut m = MarketClearing::from_dataset(&data, 2, capital);
        let r = m.step_concave(&buy, &params, None, exponent);
        let mid = r.cleared_mids[0];
        (r.fills[0][0].fill_price - mid, r.net_flow[0])
    }

    #[test]
    fn nonlinear_impact_is_invariant_to_a_common_flow_unit_rescaling() {
        let run = |capital: f64, volume_scale: f64| {
            let data = dataset_from_closes(vec![100.0; 30]);
            let params = MarketParams {
                lambda: 0.5,
                eta: 0.0,
                volume_scale,
                vol_scale: 0.0,
            };
            let mut market = MarketClearing::from_dataset(&data, 2, capital);
            market.step_concave(&block(2, 1, 0.8), &params, None, 0.5)
        };
        let base = run(100.0, 1.0);
        let rescaled = run(1_000.0, 10.0);
        let base_fraction = base.fills[0][0].fill_price / base.cleared_mids[0] - 1.0;
        let rescaled_fraction = rescaled.fills[0][0].fill_price / rescaled.cleared_mids[0] - 1.0;
        assert!((base_fraction - rescaled_fraction).abs() < 1e-12);
        assert!((base.net_flow[0] / 1.0 - rescaled.net_flow[0] / 10.0).abs() < 1e-12);
    }

    #[test]
    fn a_concave_exponent_charges_large_flow_less_and_small_flow_more() {
        // The concavity statement itself, in flow units (|Q| relative to 1): sqrt impact
        // undercharges the linear model for large trades and overcharges it for small
        // ones, with the crossover at |Q| = 1. capital = 100 puts the two-agent 0.8-weight
        // buy at Q = 1.6 > 1; capital = 1 puts it at Q = 0.016 < 1.
        let (linear_large, q_large) = first_bar_crowd_impact(100.0, 1.0);
        let (concave_large, _) = first_bar_crowd_impact(100.0, 0.5);
        assert!(
            q_large > 1.0,
            "the large-trade arm must clear |Q| > 1: {q_large}"
        );
        assert!(
            concave_large < linear_large,
            "concave impact must charge a large trade less: {concave_large} vs {linear_large}"
        );

        let (linear_small, q_small) = first_bar_crowd_impact(1.0, 1.0);
        let (concave_small, _) = first_bar_crowd_impact(1.0, 0.5);
        assert!(
            q_small > 0.0 && q_small < 1.0,
            "the small-trade arm must clear 0 < |Q| < 1: {q_small}"
        );
        assert!(
            concave_small > linear_small,
            "concave impact must charge a small trade more: {concave_small} vs {linear_small}"
        );
    }

    #[test]
    fn the_concave_exponent_also_bends_the_permanent_multiplier() {
        // Internal consistency, mirroring the robust-lambda test: the exponent reshapes
        // the flow term that feeds the next bar's reference price, not just the fill. A
        // large (Q > 1) buy therefore leaves the concave cleared mid below the linear one
        // on the following bar.
        let data = dataset_from_closes(vec![100.0; 30]);
        let params = MarketParams {
            lambda: 0.5,
            eta: 0.0,
            volume_scale: 1.0,
            vol_scale: 0.0,
        };
        let buy = block(2, 1, 0.8);
        let mut linear = MarketClearing::from_dataset(&data, 2, 100.0);
        let mut concave = MarketClearing::from_dataset(&data, 2, 100.0);
        linear.step_concave(&buy, &params, None, 1.0);
        concave.step_concave(&buy, &params, None, 0.5);
        let rl = linear.step_concave(&buy, &params, None, 1.0);
        let rc = concave.step_concave(&buy, &params, None, 0.5);
        assert!(
            rc.cleared_mids[0] < rl.cleared_mids[0],
            "a concave exponent must carry into the reference price: {} vs {}",
            rc.cleared_mids[0],
            rl.cleared_mids[0]
        );
        assert!(
            rc.cleared_mids[0] > 100.0,
            "the concave permanent bump is still a bump: {}",
            rc.cleared_mids[0]
        );
    }

    #[test]
    fn concave_clearing_is_deterministic() {
        // powf is admitted on the gated path only; within one runtime it is still a pure
        // function, so the same inputs reproduce a byte-identical stream.
        let data = Dataset::synthetic(2, 45, 19);
        let params = MarketParams {
            lambda: 0.3,
            eta: 0.15,
            volume_scale: 1.0,
            vol_scale: 2.0,
        };
        let orders = block(3, 2, 0.5);
        let run = || {
            let mut m = MarketClearing::from_dataset(&data, 3, 1.0);
            let mut log = Vec::new();
            loop {
                let r = m.step_concave(&orders, &params, None, 0.5);
                log.push(result_blob(&r));
                if r.done {
                    break;
                }
            }
            log
        };
        assert_eq!(run(), run(), "concave clearing must be deterministic");
    }

    #[test]
    #[should_panic(expected = "impact_exponent must be positive")]
    fn a_non_positive_exponent_is_rejected() {
        let data = Dataset::synthetic(1, 30, 1);
        let mut m = MarketClearing::from_dataset(&data, 1, 1.0);
        m.step_concave(&block(1, 1, 0.1), &MarketParams::default(), None, 0.0);
    }

    // --- Observation-richness (information-disclosure) axis --------------------------------

    use crate::richness::{ObservationRichness, RichnessTier};

    /// Roll `orders` over a market to exhaustion, collecting the serialized observations of
    /// every step (plus the initial pre-trade observations).
    fn rollout_observations(mut m: MarketClearing, orders: &[Vec<f64>]) -> Vec<String> {
        let params = MarketParams::default();
        let mut log = vec![serde_json::to_string(&m.initial_observations()).unwrap()];
        loop {
            let r = m.step(orders, &params);
            log.push(serde_json::to_string(&r.observations).unwrap());
            if r.done {
                break;
            }
        }
        log
    }

    #[test]
    fn default_richness_is_byte_identical_to_standard_tier() {
        // The whole additive-only guarantee: a market built with no richness setting and one
        // built with the explicit Standard tier emit a byte-identical observation stream.
        let data = Dataset::synthetic(3, 60, 4);
        let orders = block(2, 3, 0.3);
        let default_log =
            rollout_observations(MarketClearing::from_dataset(&data, 2, 1.0), &orders);
        let standard_log = rollout_observations(
            MarketClearing::from_dataset_with_richness(
                &data,
                2,
                1.0,
                RichnessTier::Standard.richness(),
            ),
            &orders,
        );
        assert_eq!(
            default_log, standard_log,
            "Standard richness must reproduce the default observation stream byte-for-byte"
        );
    }

    #[test]
    fn data_poor_shows_fewer_bars_and_withholds_optional_fields() {
        let data = Dataset::synthetic(2, 60, 7);
        let m = MarketClearing::from_dataset_with_richness(
            &data,
            2,
            1.0,
            RichnessTier::DataPoor.richness(),
        );
        for obs in m.initial_observations() {
            for snap in &obs.symbols {
                assert!(
                    snap.close_history.len() <= 3,
                    "DataPoor caps the trailing history at 3 bars, got {}",
                    snap.close_history.len()
                );
                assert!(
                    snap.fundamentals.is_empty(),
                    "DataPoor withholds fundamentals"
                );
                assert!(snap.news.is_empty(), "DataPoor withholds news");
            }
        }
    }

    #[test]
    fn data_rich_shows_more_bars_and_populates_optional_fields() {
        let data = Dataset::synthetic(2, 120, 9);
        let rich = MarketClearing::from_dataset_with_richness(
            &data,
            2,
            1.0,
            RichnessTier::DataRich.richness(),
        );
        let standard = MarketClearing::from_dataset(&data, 2, 1.0);

        let rich_obs = rich.initial_observations();
        let std_obs = standard.initial_observations();
        for (ro, so) in rich_obs.iter().zip(&std_obs) {
            for (rs, ss) in ro.symbols.iter().zip(&so.symbols) {
                assert!(
                    rs.close_history.len() > ss.close_history.len(),
                    "DataRich must surface strictly more history than Standard: {} vs {}",
                    rs.close_history.len(),
                    ss.close_history.len()
                );
                assert!(
                    rs.close_history.len() <= 50,
                    "DataRich caps the trailing history at 50 bars"
                );
                // Fundamentals: the three derived point-in-time context fields.
                assert!(rs.fundamentals.contains_key("trailing_return"));
                assert!(rs.fundamentals.contains_key("window_high"));
                assert!(rs.fundamentals.contains_key("window_low"));
                // News: exactly one deterministic headline mentioning the symbol.
                assert_eq!(rs.news.len(), 1);
                assert!(rs.news[0].contains(&rs.symbol));
            }
        }
    }

    #[test]
    fn every_tier_is_leak_free_never_surfaces_a_future_bar() {
        // Under all three tiers the last close surfaced for a symbol is exactly this bar's
        // cleared mid (never a future bar), and the surfaced window is a suffix of the
        // realized cleared tape whose length cannot exceed the number of cleared bars.
        let data = Dataset::synthetic(3, 80, 2);
        let orders = block(2, 3, 0.25);
        for tier in RichnessTier::all() {
            let mut m = MarketClearing::from_dataset_with_richness(&data, 2, 1.0, tier.richness());
            let params = MarketParams::default();
            let mut cleared_bars = m.start_bar(); // burn-in bars already on the tape
            loop {
                let r = m.step(&orders, &params);
                cleared_bars += 1;
                for obs in &r.observations {
                    for (s, snap) in obs.symbols.iter().enumerate() {
                        assert_eq!(
                            *snap.close_history.last().unwrap(),
                            r.cleared_mids[s],
                            "{tier:?}: the last surfaced close must be this bar's cleared mid"
                        );
                        assert!(
                            snap.close_history.len() <= cleared_bars,
                            "{tier:?}: cannot surface more closes than have been cleared"
                        );
                    }
                }
                if r.done {
                    break;
                }
            }
        }
    }

    #[test]
    fn richness_clearing_stays_deterministic() {
        // Populating fundamentals/news must not perturb determinism: same data + orders
        // reproduce a byte-identical DataRich observation stream.
        let data = Dataset::synthetic(2, 50, 5);
        let orders = block(2, 2, 0.4);
        let run = || {
            rollout_observations(
                MarketClearing::from_dataset_with_richness(
                    &data,
                    2,
                    1.0,
                    RichnessTier::DataRich.richness(),
                ),
                &orders,
            )
        };
        assert_eq!(run(), run(), "DataRich clearing must be deterministic");
    }

    #[test]
    fn custom_richness_overrides_lookback_independently_of_fields() {
        // The config is not just the three presets: a bespoke ObservationRichness honors an
        // arbitrary lookback with the optional fields still gated by their own flags.
        let data = Dataset::synthetic(1, 60, 3);
        let m = MarketClearing::from_dataset_with_richness(
            &data,
            1,
            1.0,
            ObservationRichness {
                lookback: 7,
                fundamentals: true,
                news: false,
            },
        );
        for snap in &m.initial_observations()[0].symbols {
            assert!(snap.close_history.len() <= 7);
            assert!(!snap.fundamentals.is_empty(), "fundamentals flag honored");
            assert!(snap.news.is_empty(), "news flag honored independently");
        }
    }
}
