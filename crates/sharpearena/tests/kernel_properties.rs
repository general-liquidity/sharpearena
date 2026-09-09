//! Production-linked state-transition properties for the shipped kernel.
//!
//! Every property here executes the production functions in `crates/sharpearena/src/`
//! (`market.rs`, `lob_market.rs`, `vec_env.rs`) over seeded, hand-rolled random cases; no
//! arithmetic is restated on labels disconnected from the implementation. The doc comment
//! on each test names the kernel function it binds, with file and line, so a failure
//! points straight back at the transition it contradicts.
//!
//! The generator is a local SplitMix64 (the pattern `exec_noise.rs:29-40` uses), fixed base
//! seed per test, `CASES` cases each. There is no property-testing dependency in the
//! workspace and none is added: the cases are deterministic and replayable by seed.
//!
//! Example-based neighbours these properties generalize rather than duplicate:
//! `market.rs` `reset_replays_book_impact_volatility_and_terminal_state` and
//! `done_flips_on_the_final_bar`, the `lob_market.rs` priority tests, and the `vec_env.rs`
//! scalar-vs-batch identity and autoreset tests. The bare-reset regression the audit
//! recorded (`docs/audits/2026-09-07/arena-reviewer.md` item 4) was repaired under batch G
//! (`VERIFICATION-LOG.md`, "Native market reset"); the reset properties below pin that
//! repair for every seeded shape rather than reopening it.

use std::collections::BTreeMap;

use serde::Serialize;
use sharpearena::vec_env::AutoresetMode;
use sharpearena::{
    Action, Dataset, Decision, Fill, LaneConfig, MarketClearing, MarketObservation, MarketParams,
    ObservationRichness, Order, OrderBook, OrderKind, Side, VecTradingEnv,
};

/// Seeded cases per property. Sized so the whole file runs in a few seconds.
const CASES: usize = 64;
const CAPITAL: f64 = 1_000_000.0;
const EPS: f64 = 1e-12;

/// Dependency-free SplitMix64, the same family `exec_noise.rs` and `scenario_gen` use.
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        SplitMix64(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    fn next_unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform integer in `[lo, hi]`.
    fn range(&mut self, lo: usize, hi: usize) -> usize {
        lo + (self.next_u64() % (hi - lo + 1) as u64) as usize
    }

    /// Uniform in `[-mag, mag)`.
    fn signed(&mut self, mag: f64) -> f64 {
        (2.0 * self.next_unit() - 1.0) * mag
    }

    fn coin(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }
}

fn json<T: Serialize>(value: &T) -> String {
    serde_json::to_string(value).expect("serializable")
}

/// A random market shape: `(seed, n_symbols, n_days, n_agents)`. `n_days` starts above the
/// 20-bar warm-up so every case has at least two traded bars.
fn shape(rng: &mut SplitMix64) -> (u64, usize, usize, usize) {
    (
        rng.next_u64(),
        rng.range(1, 4),
        rng.range(22, 60),
        rng.range(1, 3),
    )
}

/// Random target weights in `[-0.3, 0.3)`, one vector per agent.
fn weights(rng: &mut SplitMix64, n_agents: usize, n_sym: usize) -> Vec<Vec<f64>> {
    (0..n_agents)
        .map(|_| (0..n_sym).map(|_| rng.signed(0.3)).collect())
        .collect()
}

/// A random long-only decision over the observation's symbol axis.
fn decision(rng: &mut SplitMix64, obs: &MarketObservation) -> Decision {
    let orders = obs
        .symbols
        .iter()
        .map(|s| Order {
            symbol: s.symbol.clone(),
            action: Action::Buy,
            target_weight: rng.next_unit() * 0.5,
            confidence: 0.5,
            rationale: String::new(),
        })
        .collect();
    Decision {
        orders,
        reasoning: String::new(),
        cost: None,
    }
}

fn decisions(rng: &mut SplitMix64, obs: &[MarketObservation]) -> Vec<Decision> {
    obs.iter().map(|o| decision(rng, o)).collect()
}

fn lanes(rng: &mut SplitMix64) -> Vec<LaneConfig> {
    let n_lanes = rng.range(1, 3);
    let n_sym = rng.range(1, 4);
    let n_days = rng.range(22, 60);
    let base = rng.next_u64();
    (0..n_lanes)
        .map(|i| LaneConfig::new(n_sym, n_days, base.wrapping_add(i as u64)))
        .collect()
}

// --- Reset ------------------------------------------------------------------------------

/// Binds `MarketClearing::reset` (`market.rs:573`). After `k` steps, including past the
/// terminal bar, `reset` restores the cursor to `start_bar`, clears `is_done`, and yields
/// `initial_observations()` byte-equal to a market freshly built by `from_dataset`. A
/// second `reset` is a no-op, and the next cleared bar equals the fresh market's.
#[test]
fn market_reset_after_k_steps_matches_a_fresh_market() {
    let mut rng = SplitMix64::new(0x5EED_0001);
    let params = MarketParams::default();
    for _ in 0..CASES {
        let (seed, n_sym, n_days, n_agents) = shape(&mut rng);
        let data = Dataset::synthetic(n_sym, n_days, seed);
        let mut fresh = MarketClearing::from_dataset(&data, n_agents, CAPITAL);
        let mut walked = MarketClearing::from_dataset(&data, n_agents, CAPITAL);
        let horizon = n_days - walked.start_bar();
        let k = rng.range(0, horizon + 2);
        for _ in 0..k {
            let w = weights(&mut rng, n_agents, n_sym);
            walked.step(&w, &params);
        }
        walked.reset();
        assert!(!walked.is_done(), "reset must clear the terminal state");
        assert_eq!(walked.cursor(), fresh.start_bar());
        let expected = json(&fresh.initial_observations());
        assert_eq!(json(&walked.initial_observations()), expected);
        walked.reset();
        assert_eq!(
            json(&walked.initial_observations()),
            expected,
            "reset; reset must equal reset"
        );
        let w = weights(&mut rng, n_agents, n_sym);
        assert_eq!(
            json(&walked.step(&w, &params)),
            json(&fresh.step(&w, &params)),
            "the first bar after reset must clear identically to a fresh market"
        );
    }
}

/// Binds `VecTradingEnv::reset_batch` (`vec_env.rs:340`). After `k` batched steps under
/// the default `NextStep` mode, `reset_batch` returns each lane's fresh first observation
/// byte-equal to a freshly built batch, and the following `step_batch` agrees on reward,
/// NAV and observation lane by lane.
#[test]
fn vec_reset_batch_after_k_steps_matches_fresh_lanes() {
    let mut rng = SplitMix64::new(0x5EED_0002);
    for _ in 0..CASES {
        let cfgs = lanes(&mut rng);
        let mut fresh = VecTradingEnv::from_configs(&cfgs);
        let mut walked = VecTradingEnv::from_configs(&cfgs);
        let k = rng.range(0, cfgs[0].n_days + 2);
        let mut obs = walked.reset_batch();
        for _ in 0..k {
            let decs = decisions(&mut rng, &obs);
            obs = walked.step_batch(&decs).observations;
        }
        let fresh_obs = fresh.reset_batch();
        let walked_obs = walked.reset_batch();
        assert_eq!(json(&walked_obs), json(&fresh_obs));
        let decs = decisions(&mut rng, &fresh_obs);
        let a = fresh.step_batch(&decs);
        let b = walked.step_batch(&decs);
        assert_eq!(a.rewards, b.rewards);
        assert_eq!(json(&a.observations), json(&b.observations));
        for (ia, ib) in a.infos.iter().zip(&b.infos) {
            assert_eq!(ia.nav, ib.nav);
        }
    }
}

// --- Step -------------------------------------------------------------------------------

/// Binds `MarketClearing::step` (`market.rs:683`) and `clear_bar` (`market.rs:865`). Two
/// markets over the same dataset fed the same order sequence produce byte-equal
/// `ClearResult`s; each step advances `cursor` by exactly one; the observation date is
/// `dates[cursor_before]`; with an untruncated lookback the surfaced close history has
/// exactly `cursor_after` entries (burn-in plus every cleared bar); and `done` equals
/// `is_done()`.
#[test]
fn market_step_is_deterministic_and_advances_one_bar() {
    let mut rng = SplitMix64::new(0x5EED_0003);
    let params = MarketParams::default();
    for _ in 0..CASES {
        let (seed, n_sym, n_days, n_agents) = shape(&mut rng);
        let data = Dataset::synthetic(n_sym, n_days, seed);
        let richness = ObservationRichness {
            lookback: n_days,
            fundamentals: false,
            news: false,
        };
        let mut a = MarketClearing::from_dataset_with_richness(&data, n_agents, CAPITAL, richness);
        let mut b = MarketClearing::from_dataset_with_richness(&data, n_agents, CAPITAL, richness);
        while !a.is_done() {
            let before = a.cursor();
            let w = weights(&mut rng, n_agents, n_sym);
            let ra = a.step(&w, &params);
            let rb = b.step(&w, &params);
            assert_eq!(json(&ra), json(&rb), "same inputs must clear identically");
            assert_eq!(a.cursor(), before + 1);
            assert_eq!(b.cursor(), before + 1);
            assert_eq!(ra.done, a.is_done());
            for obs in &ra.observations {
                assert_eq!(obs.date, a.dates()[before]);
                for snapshot in &obs.symbols {
                    assert_eq!(snapshot.close_history.len(), a.cursor());
                }
            }
        }
    }
}

/// Binds `OrderBook::step` (`lob_market.rs:319`). The step canonicalizes a bar's batch by
/// `(agent, submission index)`, so the invariance it promises is over how the agents'
/// streams were interleaved when collected, each agent's own sequence kept. A random
/// interleaving of one bar's `(agent, OrderKind)` batch yields the identical fill tape
/// and the identical resting book, and a second book fed the original batch agrees.
#[test]
fn lob_step_is_batch_order_invariant() {
    let mut rng = SplitMix64::new(0x5EED_0004);
    for _ in 0..CASES {
        let mut a = OrderBook::new(0.01);
        let mut b = OrderBook::new(0.01);
        let mut c = OrderBook::new(0.01);
        let n_agents = rng.range(1, 4);
        for _ in 0..8 {
            let batch = random_batch(&mut rng, n_agents, a.next_order_id());
            let shuffled = interleave(&mut rng, &batch);
            assert_eq!(shuffled.len(), batch.len());
            let fa = a.step(&batch);
            let fb = b.step(&shuffled);
            let fc = c.step(&batch);
            assert_eq!(fa, fb, "a permuted batch must produce the same tape");
            assert_eq!(fa, fc, "the same batch must produce the same tape");
            assert_eq!(a.depth_ladder(usize::MAX), b.depth_ladder(usize::MAX));
            assert_eq!(a.depth_ladder(usize::MAX), c.depth_ladder(usize::MAX));
            assert_eq!(a.next_order_id(), b.next_order_id());
        }
    }
}

fn random_batch(rng: &mut SplitMix64, n_agents: usize, next_id: u64) -> Vec<(usize, OrderKind)> {
    let n = rng.range(1, 12);
    (0..n)
        .map(|_| {
            let agent = rng.range(0, n_agents - 1);
            let side = if rng.coin() { Side::Buy } else { Side::Sell };
            let kind = match rng.range(0, 3) {
                0 | 1 => OrderKind::Limit {
                    side,
                    price_tick: rng.range(990, 1010) as i64,
                    qty: rng.range(1, 20) as u64,
                },
                2 => OrderKind::Market {
                    side,
                    qty: rng.range(1, 15) as u64,
                },
                _ => {
                    let id = rng.range(0, next_id as usize + 2) as u64;
                    if rng.coin() {
                        OrderKind::Cancel { id }
                    } else {
                        OrderKind::Modify {
                            id,
                            new_qty: rng.range(0, 25) as u64,
                        }
                    }
                }
            };
            (agent, kind)
        })
        .collect()
}

/// A random interleaving of the per-agent streams in `batch`: each agent's orders keep
/// their relative order, the agents' turns are drawn at random.
fn interleave(rng: &mut SplitMix64, batch: &[(usize, OrderKind)]) -> Vec<(usize, OrderKind)> {
    let mut streams: BTreeMap<usize, std::collections::VecDeque<(usize, OrderKind)>> =
        BTreeMap::new();
    for &order in batch {
        streams.entry(order.0).or_default().push_back(order);
    }
    let mut out = Vec::with_capacity(batch.len());
    while !streams.is_empty() {
        let agents: Vec<usize> = streams.keys().copied().collect();
        let agent = agents[rng.range(0, agents.len() - 1)];
        let stream = streams.get_mut(&agent).expect("live stream");
        out.push(stream.pop_front().expect("non-empty stream"));
        if stream.is_empty() {
            streams.remove(&agent);
        }
    }
    out
}

// --- Terminal state ---------------------------------------------------------------------

/// Binds `MarketClearing::is_done` (`market.rs:649`) and the cursor clamp in
/// `exo_mid_at_cursor` (`market.rs:654`). Once the path is exhausted, `m` further steps
/// keep `is_done` true and `done` set, keep advancing the cursor by one, and clear at the
/// last bar's exogenous mid (the clamp is on the mid; the observation date past the end
/// is empty, because `dates.get(cursor)` has no bar to name). Absorption is on the
/// terminal predicate, not on state equality, because the cursor keeps counting.
#[test]
fn market_terminal_state_absorbs() {
    let mut rng = SplitMix64::new(0x5EED_0005);
    let params = MarketParams::default();
    for _ in 0..CASES {
        let (seed, n_sym, n_days, n_agents) = shape(&mut rng);
        let data = Dataset::synthetic(n_sym, n_days, seed);
        let mut market = MarketClearing::from_dataset(&data, n_agents, CAPITAL);
        while !market.is_done() {
            let w = weights(&mut rng, n_agents, n_sym);
            market.step(&w, &params);
        }
        let last_exo: Vec<f64> = market
            .symbols()
            .iter()
            .map(|s| data.closes[s][market.n_bars() - 1])
            .collect();
        for _ in 0..rng.range(1, 5) {
            let before = market.cursor();
            let w = weights(&mut rng, n_agents, n_sym);
            assert_eq!(
                market.exo_mid_at_cursor(),
                last_exo,
                "the mid clamps to the last bar"
            );
            let result = market.step(&w, &params);
            assert!(market.is_done(), "done must stay done");
            assert!(result.done);
            assert_eq!(market.cursor(), before + 1);
            for obs in &result.observations {
                assert!(obs.date.is_empty(), "no bar exists past the end");
            }
        }
    }
}

/// Binds `VecTradingEnv::step_batch` under `AutoresetMode::Disabled` and `NextStep`
/// (`vec_env.rs:118`, `step_lane`). Disabled: once a lane is truncated it stays truncated
/// for `m` further steps and never flags `first`. NextStep: the step after the terminal
/// step has `first == true`, reward `0.0`, neither `terminated` nor `truncated`, and an
/// observation byte-equal to a fresh `reset_batch`.
#[test]
fn vec_terminal_state_absorbs_or_resets_per_mode() {
    let mut rng = SplitMix64::new(0x5EED_0006);
    for _ in 0..CASES {
        let cfgs = lanes(&mut rng);
        let n_lanes = cfgs.len();

        let mut disabled =
            VecTradingEnv::from_configs(&cfgs).with_autoreset_mode(AutoresetMode::Disabled);
        let mut obs = disabled.reset_batch();
        let mut step = disabled.step_batch(&decisions(&mut rng, &obs));
        while !step.truncated.iter().all(|&t| t) {
            obs = step.observations;
            step = disabled.step_batch(&decisions(&mut rng, &obs));
        }
        for _ in 0..rng.range(1, 5) {
            obs = step.observations;
            step = disabled.step_batch(&decisions(&mut rng, &obs));
            assert_eq!(
                step.truncated,
                vec![true; n_lanes],
                "disabled stays truncated"
            );
            assert_eq!(step.first, vec![false; n_lanes]);
        }

        let mut next = VecTradingEnv::from_configs(&cfgs);
        let t0 = json(&VecTradingEnv::from_configs(&cfgs).reset_batch());
        let mut obs = next.reset_batch();
        let mut step = next.step_batch(&decisions(&mut rng, &obs));
        while !step.truncated.iter().all(|&t| t) {
            obs = step.observations;
            step = next.step_batch(&decisions(&mut rng, &obs));
        }
        let after = next.step_batch(&decisions(&mut rng, &step.observations));
        assert_eq!(after.first, vec![true; n_lanes]);
        assert_eq!(after.rewards, vec![0.0; n_lanes]);
        assert_eq!(after.terminated, vec![false; n_lanes]);
        assert_eq!(after.truncated, vec![false; n_lanes]);
        assert_eq!(json(&after.observations), t0);
    }
}

// --- Fills ------------------------------------------------------------------------------

fn depth(book: &OrderBook, side: Side) -> BTreeMap<i64, u64> {
    let ladder = book.depth_ladder(usize::MAX);
    let levels = match side {
        Side::Buy => ladder.bids,
        Side::Sell => ladder.asks,
    };
    levels.into_iter().map(|[p, q]| (p, q as u64)).collect()
}

fn fills_at(fills: &[Fill]) -> BTreeMap<i64, u64> {
    let mut at = BTreeMap::new();
    for f in fills {
        *at.entry(f.price_tick).or_insert(0) += f.qty;
    }
    at
}

/// Binds `OrderBook::match_against` through `process_limit` / `process_market`
/// (`lob_market.rs:168`, `:194`, `:201`) and `uncross` (`lob_market.rs:394`). For every
/// aggressive order against a seeded resting book: on the crossed side, fills at a level
/// plus depth after equal depth before; on the order's own side, depth after at the limit
/// price plus filled quantity equals depth before plus the submitted quantity, and every
/// other level is unchanged; each fill price lies between the touched best and the limit;
/// fills at one level arrive in ascending `maker_id` (FIFO); a buy sweep's prices are
/// non-decreasing and a sell sweep's non-increasing; and `uncross` leaves the book and
/// the id counter untouched.
#[test]
fn lob_fills_conserve_depth_and_respect_price_time_priority() {
    let mut rng = SplitMix64::new(0x5EED_0007);
    for _ in 0..CASES {
        let mut book = OrderBook::new(0.01);
        for _ in 0..rng.range(5, 30) {
            let side = if rng.coin() { Side::Buy } else { Side::Sell };
            book.process_limit(
                side,
                rng.range(980, 1020) as i64,
                rng.range(1, 20) as u64,
                rng.range(0, 3),
            );
        }
        for _ in 0..rng.range(1, 6) {
            let side = if rng.coin() { Side::Buy } else { Side::Sell };
            let opposite = match side {
                Side::Buy => Side::Sell,
                Side::Sell => Side::Buy,
            };
            let qty = rng.range(1, 40) as u64;
            let agent = rng.range(0, 3);
            let limit = if rng.coin() {
                Some(rng.range(975, 1025) as i64)
            } else {
                None
            };
            let own_before = depth(&book, side);
            let crossed_before = depth(&book, opposite);
            let touch = match side {
                Side::Buy => book.best_ask(),
                Side::Sell => book.best_bid(),
            };
            let id_before = book.next_order_id();

            let ladder_before = book.depth_ladder(usize::MAX);
            let _ = book.uncross();
            assert_eq!(book.depth_ladder(usize::MAX), ladder_before);
            assert_eq!(book.next_order_id(), id_before, "uncross is read-only");

            let fills = match limit {
                Some(price) => book.process_limit(side, price, qty, agent),
                None => book.process_market(side, qty, agent),
            };
            let filled: u64 = fills.iter().map(|f| f.qty).sum();
            assert!(filled <= qty);

            let crossed_after = depth(&book, opposite);
            let at = fills_at(&fills);
            let levels: Vec<i64> = crossed_before
                .keys()
                .chain(crossed_after.keys())
                .chain(at.keys())
                .copied()
                .collect();
            for p in levels {
                let before = crossed_before.get(&p).copied().unwrap_or(0);
                let after = crossed_after.get(&p).copied().unwrap_or(0);
                let taken = at.get(&p).copied().unwrap_or(0);
                assert_eq!(taken + after, before, "crossed-side depth at {p}");
            }

            let own_after = depth(&book, side);
            let mut expected = own_before.clone();
            if let Some(price) = limit {
                let rest = qty - filled;
                if rest > 0 {
                    *expected.entry(price).or_insert(0) += rest;
                }
            }
            assert_eq!(own_after, expected, "own-side depth");

            let mut last: Option<(i64, u64)> = None;
            for f in &fills {
                assert_eq!(f.taker_agent, agent);
                assert_eq!(f.taker_side, side);
                let best = touch.expect("a fill needs a touched best");
                match side {
                    Side::Buy => {
                        assert!(f.price_tick >= best);
                        if let Some(l) = limit {
                            assert!(f.price_tick <= l);
                        }
                    }
                    Side::Sell => {
                        assert!(f.price_tick <= best);
                        if let Some(l) = limit {
                            assert!(f.price_tick >= l);
                        }
                    }
                }
                if let Some((price, maker)) = last {
                    match side {
                        Side::Buy => assert!(f.price_tick >= price, "buy sweep non-decreasing"),
                        Side::Sell => assert!(f.price_tick <= price, "sell sweep non-increasing"),
                    }
                    if f.price_tick == price {
                        assert!(f.maker_id > maker, "FIFO within a level");
                    }
                }
                last = Some((f.price_tick, f.maker_id));
            }
        }
    }
}

// --- Accounting -------------------------------------------------------------------------

fn nav(cash: f64, shares: &[f64], mids: &[f64]) -> f64 {
    cash + shares.iter().zip(mids).map(|(sh, m)| sh * m).sum::<f64>()
}

/// Binds `clear_bar` (`market.rs:1032-1081`). Per agent per bar, recomputed in the
/// kernel's own per-symbol order from the reported fills and the prior observation:
/// `cash_after == cash_before - sum(size * fill_price)` (exact: the same subtractions in
/// the same order), `shares_after == shares_before + size` (exact),
/// `navs[i] == cash_after + sum(shares_after * cleared_mids)` (exact, same fold), and
/// `rewards[i] == (nav_post - nav_prev) / nav_prev` with `nav_prev` marked at the prior
/// cleared mids (exact). Read through `observations[i].cash` and
/// `.portfolio[s].shares`; no kernel change.
#[test]
fn clear_bar_accounting_identities_hold_exactly() {
    let mut rng = SplitMix64::new(0x5EED_0008);
    let params = MarketParams::default();
    for _ in 0..CASES {
        let (seed, n_sym, n_days, n_agents) = shape(&mut rng);
        let data = Dataset::synthetic(n_sym, n_days, seed);
        let mut market = MarketClearing::from_dataset(&data, n_agents, CAPITAL);
        let mut prev_obs = market.initial_observations();
        let mut prev_mids = market.exo_mid_at_cursor();
        while !market.is_done() {
            let w = weights(&mut rng, n_agents, n_sym);
            let result = market.step(&w, &params);
            for (i, before) in prev_obs.iter().enumerate() {
                let after = &result.observations[i];
                let mut cash = before.cash;
                let mut shares: Vec<f64> = before.portfolio.iter().map(|p| p.shares).collect();
                let nav_prev = nav(cash, &shares, &prev_mids);
                for (s, fill) in result.fills[i].iter().enumerate() {
                    cash -= fill.size * fill.fill_price;
                    shares[s] += fill.size;
                    assert_eq!(
                        after.portfolio[s].shares, shares[s],
                        "shares agent {i} sym {s}"
                    );
                }
                assert_eq!(after.cash, cash, "cash agent {i}");
                let nav_post = nav(cash, &shares, &result.cleared_mids);
                assert_eq!(result.navs[i], nav_post, "nav agent {i}");
                let expected = if nav_prev.abs() > EPS {
                    (nav_post - nav_prev) / nav_prev
                } else {
                    0.0
                };
                assert_eq!(result.rewards[i], expected, "reward agent {i}");
            }
            prev_mids = result.cleared_mids.clone();
            prev_obs = result.observations;
        }
    }
}

/// Binds `clear_bar` (`market.rs:1032-1081`). Repeating the previous bar's weights sizes
/// every order to exactly zero, so cash and holdings are unchanged; and two agents with
/// exactly opposite weights on a fresh book net to zero flow, so the permanent-impact
/// multiplier stays at one and the next bar clears at the exogenous mid.
#[test]
fn clear_bar_zero_and_antisymmetric_orders_move_nothing() {
    let mut rng = SplitMix64::new(0x5EED_0009);
    let params = MarketParams::default();
    for _ in 0..CASES {
        let (seed, n_sym, n_days, _) = shape(&mut rng);
        let data = Dataset::synthetic(n_sym, n_days, seed);

        let n_agents = rng.range(1, 3);
        let mut market = MarketClearing::from_dataset(&data, n_agents, CAPITAL);
        let w = weights(&mut rng, n_agents, n_sym);
        let traded = market.step(&w, &params);
        let held = market.step(&w, &params);
        for i in 0..n_agents {
            assert_eq!(held.observations[i].cash, traded.observations[i].cash);
            for s in 0..n_sym {
                assert_eq!(held.fills[i][s].size, 0.0);
                assert_eq!(
                    held.observations[i].portfolio[s].shares,
                    traded.observations[i].portfolio[s].shares
                );
            }
        }

        let mut pair = MarketClearing::from_dataset(&data, 2, CAPITAL);
        let long: Vec<f64> = (0..n_sym).map(|_| rng.signed(0.5)).collect();
        let short: Vec<f64> = long.iter().map(|x| -x).collect();
        let result = pair.step(&[long, short], &params);
        assert_eq!(
            result.net_flow,
            vec![0.0; n_sym],
            "antisymmetric orders net to zero"
        );
        for s in 0..n_sym {
            assert_eq!(result.fills[0][s].size, -result.fills[1][s].size);
        }
        let exo = pair.exo_mid_at_cursor();
        let next = pair.step(&[vec![0.0; n_sym], vec![0.0; n_sym]], &params);
        assert_eq!(
            next.cleared_mids, exo,
            "zero net flow leaves no permanent impact"
        );
    }
}
