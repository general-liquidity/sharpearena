//! Throughput benchmark for the native engine: scenario generation, scalar `reset`/`step`
//! and batched `step_batch`. Reports steps per second and per-episode wall time so the
//! paper's compute-cost table is measured rather than estimated.
//!
//! Run with `cargo run --release -p sharpearena --example bench-steps`.

use std::time::Instant;

use sharpearena::{
    generate_scenario, Action, CostModel, Dataset, Decision, LaneConfig, MarketObservation, Order,
    ScenarioSpec, TradingEnv, VecTradingEnv, Window,
};

const N_SYMBOLS: usize = 4;
const N_DAYS: usize = 120;

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

fn bench_generation(episodes: u64) -> (f64, f64) {
    let spec = ScenarioSpec {
        n_symbols: N_SYMBOLS,
        n_days: N_DAYS,
        ..ScenarioSpec::default()
    };
    // Serialize to the canonical JSON bytes so this row measures the same work as the
    // Python and WebAssembly generation benchmarks.
    let start = Instant::now();
    let mut acc = 0.0f64;
    for seed in 0..episodes {
        let data = generate_scenario(&spec, seed);
        acc += serde_json::to_string(&data).unwrap().len() as f64;
    }
    let elapsed = start.elapsed().as_secs_f64();
    assert!(acc > 0.0);
    (episodes as f64 / elapsed, elapsed)
}

fn bench_scalar(episodes: u64) -> (f64, f64, u64) {
    let start = Instant::now();
    let mut steps = 0u64;
    for seed in 0..episodes {
        let mut env = TradingEnv::new(
            Dataset::synthetic(N_SYMBOLS, N_DAYS, seed),
            Window {
                start: 0,
                end: N_DAYS,
            },
            CostModel::default(),
            seed,
        );
        let mut obs = env.reset();
        loop {
            let res = env.step(decision_for(&obs, 0.25));
            steps += 1;
            if res.done {
                break;
            }
            obs = res.observation;
        }
    }
    let elapsed = start.elapsed().as_secs_f64();
    (steps as f64 / elapsed, elapsed, steps)
}

fn bench_batched(lanes: usize, bars: usize) -> f64 {
    let configs: Vec<LaneConfig> = (0..lanes as u64)
        .map(|s| LaneConfig::new(N_SYMBOLS, N_DAYS, s))
        .collect();
    let mut batch = VecTradingEnv::from_configs(&configs);
    let mut obs = batch.reset_batch();
    let start = Instant::now();
    let mut steps = 0u64;
    for _ in 0..bars {
        let decisions: Vec<Decision> = obs.iter().map(|o| decision_for(o, 0.25)).collect();
        let res = batch.step_batch(&decisions);
        steps += lanes as u64;
        obs = res.observations;
    }
    steps as f64 / start.elapsed().as_secs_f64()
}

fn main() {
    // Warm the allocator and the code paths before timing.
    let _ = bench_scalar(4);
    let _ = bench_generation(4);

    let episodes = 500u64;
    let (gen_per_s, gen_elapsed) = bench_generation(episodes);
    let (scalar_sps, scalar_elapsed, scalar_steps) = bench_scalar(episodes);
    let batched_sps = bench_batched(64, 1000);

    println!("panel: {N_SYMBOLS} symbols x {N_DAYS} days");
    println!("episodes: {episodes}");
    println!(
        "generation: {gen_per_s:.1} scenarios/s ({:.1} us/scenario, {gen_elapsed:.3} s total)",
        1e6 / gen_per_s
    );
    println!(
        "scalar step: {scalar_sps:.0} steps/s over {scalar_steps} steps ({scalar_elapsed:.3} s total)"
    );
    println!(
        "scalar episode wall time: {:.3} ms",
        1e3 * scalar_elapsed / episodes as f64
    );
    println!("batched step (64 lanes): {batched_sps:.0} steps/s");
}
