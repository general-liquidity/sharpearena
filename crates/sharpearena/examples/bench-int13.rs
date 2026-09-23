//! INT-13 native baseline harness: scalar and vector transition throughput, reset cost
//! and native checkpoint cost, measured over repeated timed repetitions so the report can
//! carry uncertainty instead of one peak steps-per-second figure.
//!
//! The workload comes from `docs/integration/int13/workload-manifest.json`, the same
//! manifest the Python boundary harness reads, so the two halves measure one workload.
//!
//! Run with:
//! `cargo run --release -p sharpearena --example bench-int13 -- --out results-native.json`
//!
//! This example measures; it does not change dynamics, scoring or goldens.

use std::time::Instant;

use sharpearena::{
    Action, CostModel, Dataset, Decision, LaneConfig, MarketObservation, Order, TradingEnv,
    VecTradingEnv, Window,
};

/// The subset of the shared manifest this harness needs. Deliberately a narrow mirror:
/// a missing field is a hard error, so the two harnesses cannot silently diverge from
/// the committed workload.
#[derive(serde::Deserialize)]
struct Manifest {
    manifest_id: String,
    panel: Panel,
    action_policy: ActionPolicy,
    scaling: Scaling,
    measurement: Measurement,
}

#[derive(serde::Deserialize)]
struct Panel {
    n_symbols: usize,
    n_days: usize,
}

#[derive(serde::Deserialize)]
struct ActionPolicy {
    target_weight: f64,
    confidence: f64,
}

#[derive(serde::Deserialize)]
struct Scaling {
    lane_counts: Vec<usize>,
    thread_counts: Vec<usize>,
    thread_scaling_lanes: usize,
}

#[derive(serde::Deserialize)]
struct Measurement {
    repetitions: usize,
}

/// Median, mean, sample standard deviation and a 95 percent normal-approximation
/// interval on the mean of `samples`. `n` travels with the summary so a reader can
/// re-derive the interval rather than trust it.
#[derive(serde::Serialize)]
struct Summary {
    n: usize,
    median: f64,
    mean: f64,
    sd: f64,
    rel_sd: f64,
    ci95_lo: f64,
    ci95_hi: f64,
    min: f64,
    max: f64,
}

fn summarize(mut samples: Vec<f64>) -> Summary {
    assert!(!samples.is_empty(), "no samples to summarize");
    samples.sort_by(|a, b| a.partial_cmp(b).expect("non-finite sample"));
    let n = samples.len();
    let median = if n % 2 == 1 {
        samples[n / 2]
    } else {
        0.5 * (samples[n / 2 - 1] + samples[n / 2])
    };
    let mean = samples.iter().sum::<f64>() / n as f64;
    let sd = if n > 1 {
        (samples.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt()
    } else {
        0.0
    };
    let se = if n > 1 { sd / (n as f64).sqrt() } else { 0.0 };
    Summary {
        n,
        median,
        mean,
        sd,
        rel_sd: if mean != 0.0 { sd / mean } else { 0.0 },
        ci95_lo: mean - 1.96 * se,
        ci95_hi: mean + 1.96 * se,
        min: samples[0],
        max: samples[n - 1],
    }
}

fn decision_for(obs: &MarketObservation, policy: &ActionPolicy) -> Decision {
    let orders = obs
        .symbols
        .iter()
        .map(|s| Order {
            symbol: s.symbol.clone(),
            action: if policy.target_weight > 0.0 {
                Action::Buy
            } else {
                Action::Hold
            },
            target_weight: policy.target_weight,
            confidence: Some(policy.confidence),
            rationale: String::new(),
        })
        .collect();
    Decision {
        orders,
        reasoning: String::new(),
        cost: None,
    }
}

fn build_scalar(m: &Manifest, seed: u64) -> TradingEnv {
    TradingEnv::new(
        Dataset::synthetic(m.panel.n_symbols, m.panel.n_days, seed),
        Window {
            start: 0,
            end: m.panel.n_days,
        },
        CostModel::default(),
        seed,
    )
}

/// One timed repetition of the scalar loop: run `episodes` full episodes end to end and
/// return steps per second. Construction is inside the timed region only for the
/// `construct` cell; here the env is rebuilt per episode because an episode ends at the
/// window edge, which is the honest scalar cost.
fn scalar_rep(m: &Manifest, episodes: u64, seed_base: u64) -> (f64, u64) {
    let start = Instant::now();
    let mut steps = 0u64;
    for k in 0..episodes {
        let mut env = build_scalar(m, seed_base + k);
        let mut obs = env.reset();
        loop {
            let res = env.step(decision_for(&obs, &m.action_policy));
            steps += 1;
            if res.done {
                break;
            }
            obs = res.observation;
        }
    }
    (
        steps as f64 / start.elapsed().as_secs_f64(),
        steps,
    )
}

/// Scalar stepping with construction hoisted out of the timed region: the env is built
/// and reset first, then `bars` steps are timed. This isolates transition cost from
/// scenario generation, which the paired `construct` cell reports separately.
fn scalar_steps_only_rep(m: &Manifest, bars: usize, seed: u64) -> f64 {
    let mut env = build_scalar(m, seed);
    let mut obs = env.reset();
    let start = Instant::now();
    let mut steps = 0u64;
    for _ in 0..bars {
        let res = env.step(decision_for(&obs, &m.action_policy));
        steps += 1;
        if res.done {
            obs = env.reset();
        } else {
            obs = res.observation;
        }
    }
    steps as f64 / start.elapsed().as_secs_f64()
}

fn lane_configs(m: &Manifest, lanes: usize) -> Vec<LaneConfig> {
    (0..lanes as u64)
        .map(|s| LaneConfig::new(m.panel.n_symbols, m.panel.n_days, s))
        .collect()
}

/// One timed repetition of the batched loop at `lanes` lanes for `bars` batched calls.
/// Construction and the first reset sit outside the timed region.
fn vector_rep(batch: &mut VecTradingEnv, m: &Manifest, bars: usize) -> f64 {
    let mut obs = batch.reset_batch();
    let lanes = batch.len();
    let start = Instant::now();
    let mut steps = 0u64;
    for _ in 0..bars {
        let decisions: Vec<Decision> = obs
            .iter()
            .map(|o| decision_for(o, &m.action_policy))
            .collect();
        let res = batch.step_batch(&decisions);
        steps += lanes as u64;
        obs = res.observations;
    }
    steps as f64 / start.elapsed().as_secs_f64()
}

#[derive(serde::Serialize)]
struct Cell {
    name: String,
    unit: String,
    lanes: Option<usize>,
    threads: Option<usize>,
    summary: Summary,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out_path = args
        .windows(2)
        .find(|w| w[0] == "--out")
        .map(|w| w[1].clone())
        .unwrap_or_else(|| "int13-native.json".to_string());
    let manifest_path = args
        .windows(2)
        .find(|w| w[0] == "--manifest")
        .map(|w| w[1].clone())
        .unwrap_or_else(|| "docs/integration/int13/workload-manifest.json".to_string());

    let text = std::fs::read_to_string(&manifest_path)
        .unwrap_or_else(|e| panic!("cannot read manifest {manifest_path}: {e}"));
    let m: Manifest = serde_json::from_str(&text).expect("manifest does not match this harness");
    let reps = m.measurement.repetitions;
    let warmup = 2usize;

    let mut cells: Vec<Cell> = Vec::new();

    // Scalar: full episodes including per-episode construction.
    let episodes = 200u64;
    for _ in 0..warmup {
        let _ = scalar_rep(&m, 20, 900_000);
    }
    let mut samples = Vec::with_capacity(reps);
    let mut steps_per_episode = 0u64;
    for r in 0..reps {
        let (sps, steps) = scalar_rep(&m, episodes, r as u64 * episodes);
        steps_per_episode = steps / episodes;
        samples.push(sps);
    }
    cells.push(Cell {
        name: "native_scalar_episode_loop".into(),
        unit: "steps/s".into(),
        lanes: Some(1),
        threads: Some(1),
        summary: summarize(samples),
    });

    // Scalar: transitions only, construction hoisted out.
    let bars = 20_000usize;
    for _ in 0..warmup {
        let _ = scalar_steps_only_rep(&m, 2_000, 900_001);
    }
    let samples: Vec<f64> = (0..reps)
        .map(|r| scalar_steps_only_rep(&m, bars, r as u64))
        .collect();
    cells.push(Cell {
        name: "native_scalar_transition_only".into(),
        unit: "steps/s".into(),
        lanes: Some(1),
        threads: Some(1),
        summary: summarize(samples),
    });

    // Construction plus first reset: the per-episode setup cost the scalar loop pays.
    for _ in 0..warmup {
        let mut e = build_scalar(&m, 900_002);
        let _ = e.reset();
    }
    let samples: Vec<f64> = (0..reps)
        .map(|r| {
            let iters = 200u64;
            let start = Instant::now();
            for k in 0..iters {
                let mut e = build_scalar(&m, r as u64 * iters + k);
                let _ = e.reset();
            }
            1e6 * start.elapsed().as_secs_f64() / iters as f64
        })
        .collect();
    cells.push(Cell {
        name: "native_construct_and_first_reset".into(),
        unit: "us/env".into(),
        lanes: Some(1),
        threads: Some(1),
        summary: summarize(samples),
    });

    // Reset of an already-constructed env: the recurring auto-reset cost.
    let samples: Vec<f64> = (0..reps)
        .map(|r| {
            let mut e = build_scalar(&m, r as u64);
            let _ = e.reset();
            let iters = 5_000u64;
            let start = Instant::now();
            for _ in 0..iters {
                let _ = e.reset();
            }
            1e6 * start.elapsed().as_secs_f64() / iters as f64
        })
        .collect();
    cells.push(Cell {
        name: "native_reset_existing_env".into(),
        unit: "us/reset".into(),
        lanes: Some(1),
        threads: Some(1),
        summary: summarize(samples),
    });

    // Native checkpoint: clone_state / restore_state at mid-episode depth.
    let samples: Vec<f64> = (0..reps)
        .map(|r| {
            let mut e = build_scalar(&m, r as u64);
            let mut obs = e.reset();
            for _ in 0..(m.panel.n_days / 2) {
                let res = e.step(decision_for(&obs, &m.action_policy));
                if res.done {
                    break;
                }
                obs = res.observation;
            }
            let iters = 2_000u64;
            let start = Instant::now();
            for _ in 0..iters {
                let s = e.clone_state();
                e.restore_state(s);
            }
            1e6 * start.elapsed().as_secs_f64() / iters as f64
        })
        .collect();
    cells.push(Cell {
        name: "native_clone_restore_state_roundtrip".into(),
        unit: "us/roundtrip".into(),
        lanes: Some(1),
        threads: Some(1),
        summary: summarize(samples),
    });

    // Vector scaling curve at the ambient rayon thread count.
    let ambient_threads = rayon::current_num_threads();
    for &lanes in &m.scaling.lane_counts {
        let bars = (200_000usize / lanes).max(50);
        let mut batch = VecTradingEnv::from_configs(&lane_configs(&m, lanes));
        for _ in 0..warmup {
            let _ = vector_rep(&mut batch, &m, bars.min(200));
        }
        let samples: Vec<f64> = (0..reps).map(|_| vector_rep(&mut batch, &m, bars)).collect();
        cells.push(Cell {
            name: "native_vector_step_batch".into(),
            unit: "steps/s".into(),
            lanes: Some(lanes),
            threads: Some(ambient_threads),
            summary: summarize(samples),
        });
    }

    // Thread scaling at a fixed lane count, using an explicit rayon pool so the cell
    // records the thread allocation it actually ran under.
    let lanes = m.scaling.thread_scaling_lanes;
    for &threads in &m.scaling.thread_counts {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("rayon pool");
        let bars = (200_000usize / lanes).max(50);
        let mut batch = VecTradingEnv::from_configs(&lane_configs(&m, lanes));
        pool.install(|| {
            for _ in 0..warmup {
                let _ = vector_rep(&mut batch, &m, bars.min(200));
            }
        });
        let samples: Vec<f64> = (0..reps)
            .map(|_| pool.install(|| vector_rep(&mut batch, &m, bars)))
            .collect();
        cells.push(Cell {
            name: "native_vector_thread_scaling".into(),
            unit: "steps/s".into(),
            lanes: Some(lanes),
            threads: Some(threads),
            summary: summarize(samples),
        });
    }

    let record = serde_json::json!({
        "harness": "bench-int13 (native)",
        "manifest_id": m.manifest_id,
        "steps_per_episode": steps_per_episode,
        "rustc": option_env!("RUSTC_VERSION").unwrap_or("see report"),
        "ambient_rayon_threads": ambient_threads,
        "profile": if cfg!(debug_assertions) { "debug" } else { "release" },
        "repetitions": reps,
        "warmup_repetitions": warmup,
        "cells": cells,
    });
    let text = serde_json::to_string_pretty(&record).expect("serialize results");
    std::fs::write(&out_path, &text).unwrap_or_else(|e| panic!("cannot write {out_path}: {e}"));
    println!("{text}");
}
