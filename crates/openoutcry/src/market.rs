//! Endogenous price-impact shared-book market (M2) — the flagship multi-agent engine.
//!
//! OpenOutcry's existing multi-agent surface ([`crate::vec_env`], and the PettingZoo
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
//!   `fill_i = cleared_mid_t * (1 + (lambda * Q_t + eta * q_i) / V)` — it pays for the
//!   crowd's flow (`lambda * Q_t`) plus its own size (`eta * q_i`). Temporary impact does
//!   not persist; it is a per-fill execution cost.
//! - **Per-agent reward** is that agent's own realized portfolio return over the bar,
//!   marked at the cleared mids and using its **own** fill prices.
//!
//! ## Determinism
//!
//! Every step is a pure function of `(exogenous path, lambda, eta, V, capital, agents'
//! actions in sorted order)`. Only `mul / add / div` are used — no `ln` / `exp` or other
//! transcendentals that differ across libm builds — so a cleared path is byte-identical
//! across Rust, WASM, and Python. Aggregation folds the per-agent sizes in canonical
//! (sorted) agent order, so the parallel collection of actions cannot perturb `Q_t`.
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

use crate::{Dataset, MarketObservation, PositionState, SymbolSnapshot};

/// Bars of trailing exogenous history burned in before the first decision, mirroring the
/// open-loop env's warm-up so an agent's first observation already has trailing closes.
const WARMUP: usize = 20;
/// Trailing closes surfaced in each observation (mirrors the sim engine's `LOOKBACK`).
const LOOKBACK: usize = 20;
/// Below this absolute size a position/NAV is treated as flat (avoids divide-by-zero in
/// the return and average-price bookkeeping). Comparisons only — never an additive fudge.
const EPS: f64 = 1e-12;

/// The impact coefficients: Kyle's permanent `lambda`, Almgren-Chriss temporary `eta`,
/// and the ADV-like `volume_scale` (`V`) that both are normalized by. All are in the
/// natural units of `net_flow` (signed shares); pick them for your notional scale.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct MarketParams {
    /// Kyle's permanent price-impact coefficient (per unit normalized net flow).
    pub lambda: f64,
    /// Almgren-Chriss temporary impact coefficient (per unit normalized **own** size).
    pub eta: f64,
    /// Volume / ADV normalizer `V` that `lambda * Q` and `eta * q_i` are divided by.
    pub volume_scale: f64,
}

impl Default for MarketParams {
    fn default() -> Self {
        Self {
            lambda: 0.1,
            eta: 0.05,
            volume_scale: 1.0,
        }
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
    agents: Vec<AgentBook>,
    /// The next bar index to clear.
    cursor: usize,
    /// The first bar that is cleared *with* trading (after the warm-up burn-in).
    start_bar: usize,
    n_bars: usize,
}

impl MarketClearing {
    /// Build an endogenous market over `data` for `n_agents`, each starting with
    /// `capital` cash. The exogenous path is taken from `data`'s closes; the first
    /// `WARMUP` bars are an untraded burn-in (cleared == exogenous) so the first
    /// observation has trailing history.
    pub fn from_dataset(data: &Dataset, n_agents: usize, capital: f64) -> Self {
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
            agents,
            cursor: start_bar,
            start_bar,
            n_bars,
        }
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
                        SymbolSnapshot {
                            symbol: sym.clone(),
                            close_history: trailing(&hist),
                            fundamentals: BTreeMap::new(),
                            news: Vec::new(),
                        }
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
}

/// Trailing closes (≤ [`LOOKBACK`]) ending at the last entry of `series`.
fn trailing(series: &[f64]) -> Vec<f64> {
    let start = series.len().saturating_sub(LOOKBACK);
    series[start..].to_vec()
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
pub fn clear_bar(
    exo_mid: &[f64],
    agent_orders: &[Vec<f64>],
    params: &MarketParams,
    state: &mut MarketClearing,
) -> ClearResult {
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
            let fill = mid * (1.0 + (params.lambda * net_flow[s] + params.eta * qi) / v);
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
                .map(|(s, sym)| SymbolSnapshot {
                    symbol: sym.clone(),
                    close_history: trailing(&state.cleared_history[s]),
                    fundamentals: BTreeMap::new(),
                    news: Vec::new(),
                })
                .collect();
            state.observation(agent, &date, symbols)
        })
        .collect();

    // (5) permanent impact accumulates into the running multiplier for the next bar; the
    //     cleared mid becomes the mark for the next bar's holding PnL.
    for (mult, flow) in state.impact_mult.iter_mut().zip(&net_flow) {
        *mult *= 1.0 + params.lambda * flow / v;
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
}
