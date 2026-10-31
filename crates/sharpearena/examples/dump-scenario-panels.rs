//! Dump generated close panels as raw little-endian `f64` for offline measurement.
//!
//! Used by `studies/generator-calibration/`. Reads the shipped generator through its
//! public entry point, so what the study measures is what an agent receives.
//!
//! ```text
//! cargo run --release -p sharpearena --example dump-scenario-panels -- \
//!     <out-dir> <start-seed> <n-seeds> <n-symbols> <n-days> <mode>...
//! ```
//!
//! Each mode writes `<out-dir>/<mode>.f64`, shaped `(n_seeds, n_days, n_symbols)` in C
//! order with symbols in sorted order. `calm_candidate` is the name for
//! [`ScenarioSpec::calm_calibration_candidate`]; every other name is a serde
//! `DistributionMode` label with all opt-in knobs at their defaults.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use sharpearena::{generate_scenario, DistributionMode, ScenarioSpec};

fn spec_for(mode: &str, n_symbols: usize, n_days: usize) -> Result<ScenarioSpec, String> {
    let base = if mode == "calm_candidate" {
        ScenarioSpec::calm_calibration_candidate()
    } else {
        let parsed: DistributionMode = serde_json::from_value(serde_json::json!(mode))
            .map_err(|e| format!("unknown mode {mode}: {e}"))?;
        ScenarioSpec {
            distribution_mode: parsed,
            ..ScenarioSpec::default()
        }
    };
    Ok(ScenarioSpec {
        n_symbols,
        n_days,
        ..base
    })
}

fn parse<T: std::str::FromStr>(arg: Option<String>, name: &str) -> Result<T, String> {
    arg.ok_or_else(|| format!("missing {name}"))?
        .parse()
        .map_err(|_| format!("invalid {name}"))
}

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let out = PathBuf::from(args.next().ok_or("missing out-dir")?);
    let start: u64 = parse(args.next(), "start-seed")?;
    let n_seeds: u64 = parse(args.next(), "n-seeds")?;
    let n_symbols: usize = parse(args.next(), "n-symbols")?;
    let n_days: usize = parse(args.next(), "n-days")?;
    let modes: Vec<String> = args.collect();
    if modes.is_empty() {
        return Err("name at least one mode".into());
    }
    std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    for mode in modes {
        let spec = spec_for(&mode, n_symbols, n_days)?;
        let path = out.join(format!("{mode}.f64"));
        let file = File::create(&path).map_err(|e| e.to_string())?;
        let mut writer = BufWriter::new(file);
        for seed in start..start + n_seeds {
            let data = generate_scenario(&spec, seed);
            for t in 0..n_days {
                for series in data.closes.values() {
                    writer
                        .write_all(&series[t].to_le_bytes())
                        .map_err(|e| e.to_string())?;
                }
            }
        }
        writer.flush().map_err(|e| e.to_string())?;
    }
    Ok(())
}
