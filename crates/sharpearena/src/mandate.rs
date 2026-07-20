//! Per-scenario trading mandates — MiniGrid's `Mission` pattern for trading.
//!
//! In MiniGrid's *Fetch* task each episode ships a per-episode objective ("pick up the
//! red key") and the agent is graded on satisfying *that* objective. The trading analogue:
//! each scenario draws a [`Mandate`] (a sampled trading objective — a style constraint, an
//! optional drawdown cap, an optional benchmark) the episode is graded against.
//!
//! [`sample_mandate`] is **deterministic and leak-free**: it derives the whole mandate from
//! the scenario `seed` (known at `reset`), never from future bars, via the same SplitMix64
//! PRNG family `scenario_gen` uses — so it is byte-identical across Rust / WASM / Python.
//! [`mandate_breach`] is a pure penalty in `[0, 1]` (`0` = clean, `1` = fully breached) the
//! reward layer turns into `1 - breach` — bounded, hence GRPO-safe. Both use only
//! `mul`/`add`/`div`/`max`/`abs` (no `ln`/`exp`, which differ across libm implementations).

use serde::{Deserialize, Serialize};

/// The constraint families a scenario can draw. `Unconstrained` is the permissive control
/// (no structural breach); the others each carry a distinct structural rule. Serializes to
/// the wire labels the Python contract speaks (`long_only` / `market_neutral` / …).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MandateStyle {
    LongOnly,
    MarketNeutral,
    Momentum,
    Unconstrained,
    /// Pairs-convergence: hold offsetting weights on a cointegrated pair so the book is
    /// dollar-neutral (`w_x*beta + w_y ≈ 0`). Generalizes [`MandateStyle::MarketNeutral`]
    /// toward the pairs reader; the breach uses a beta-free dollar-neutrality proxy (see
    /// [`mandate_breach`]).
    PairsConvergence,
}

impl MandateStyle {
    /// All styles, in the canonical draw order (mirrors the Python `STYLES` tuple).
    pub const ALL: [MandateStyle; 5] = [
        MandateStyle::LongOnly,
        MandateStyle::MarketNeutral,
        MandateStyle::Momentum,
        MandateStyle::Unconstrained,
        MandateStyle::PairsConvergence,
    ];

    /// Whether the style needs shorting — excluded when the env disallows shorts, so a
    /// sampled mandate is never unsatisfiable-by-construction. Both market-neutral and
    /// pairs-convergence require an offsetting short leg.
    fn requires_short(self) -> bool {
        matches!(
            self,
            MandateStyle::MarketNeutral | MandateStyle::PairsConvergence
        )
    }
}

/// A per-scenario objective the episode is graded on satisfying.
///
/// `style` is the structural constraint; `max_drawdown` an optional realized-DD cap (a
/// fraction, e.g. `0.10` = 10%); `max_inventory` an optional per-bar gross-exposure cap on
/// `Σ|w_i|` (e.g. `1.0` = at most 100% gross, from the Avellaneda-Stoikov inventory penalty
/// — exceeding it draws a *squared* breach); `benchmark` an optional symbol to beat (carried
/// in the prompt text only — informational, not breach-scored); `text` the human-readable
/// rendering shown to the model. The plain-JSON form round-trips through a trace/replay;
/// `max_inventory` is `#[serde(default)]` so older traces (without the key) still parse.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Mandate {
    pub style: MandateStyle,
    #[serde(default)]
    pub max_drawdown: Option<f64>,
    #[serde(default)]
    pub max_inventory: Option<f64>,
    #[serde(default)]
    pub benchmark: Option<String>,
    #[serde(default)]
    pub text: String,
}

/// The realized-DD caps a mandate can draw (fractions).
const DRAWDOWN_CAPS: [f64; 4] = [0.05, 0.10, 0.15, 0.20];

/// The per-bar gross-exposure caps a mandate can draw (`Σ|w_i|` ceilings).
const INVENTORY_CAPS: [f64; 4] = [0.5, 1.0, 1.5, 2.0];

const EPS: f64 = 1e-9;

/// SplitMix64 — the same dependency-free PRNG family `scenario_gen` uses, so the mandate
/// draw stays cross-runtime deterministic (no transcendental calls).
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        SplitMix64(seed ^ 0x1234_5678_9ABC_DEF0)
    }

    /// Next draw in `[0, 1)`.
    fn next_unit(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Render the human-readable objective string. Mirrors the Python `_render_text`: a base
/// clause per style, then optional `keep max drawdown under N%` / `aim to beat <sym>`
/// clauses, joined by `; ` and terminated with `.`.
fn render_text(
    style: MandateStyle,
    max_drawdown: Option<f64>,
    max_inventory: Option<f64>,
    benchmark: Option<&str>,
) -> String {
    let base = match style {
        MandateStyle::LongOnly => "Long-only mandate: hold no short positions",
        MandateStyle::MarketNeutral => {
            "Market-neutral mandate: keep net exposure near zero (balance longs and shorts)"
        }
        MandateStyle::Momentum => "Momentum mandate: lean into recent winners, cut losers",
        MandateStyle::Unconstrained => "Unconstrained mandate: trade freely",
        MandateStyle::PairsConvergence => {
            "Pairs-convergence mandate: hold offsetting weights on the pair (stay dollar-neutral)"
        }
    };
    let mut clauses = vec![base.to_string()];
    if let Some(dd) = max_drawdown {
        // Python's `{:.0%}` — percent rounded to a whole number (5% / 10% / 15% / 20%).
        clauses.push(format!(
            "keep max drawdown under {}%",
            (dd * 100.0).round() as i64
        ));
    }
    if let Some(inv) = max_inventory {
        // Gross-exposure ceiling, rendered as a percent of capital (50% / 100% / …).
        clauses.push(format!(
            "keep gross exposure under {}%",
            (inv * 100.0).round() as i64
        ));
    }
    if let Some(b) = benchmark {
        clauses.push(format!("aim to beat {b}"));
    }
    clauses.join("; ") + "."
}

/// Deterministically draw a [`Mandate`] from a scenario `seed`.
///
/// Leak-free: the draw depends only on `seed` (and the static `n_symbols` / `allow_short`
/// env shape), so it is reproducible at `reset` and never peeks at future bars. When
/// `allow_short` is `false` the short-requiring styles are dropped so the mandate stays
/// satisfiable on a long-only market. The draw order matches the Python: style, then the
/// drawdown coin/value, then the benchmark coin/value.
pub fn sample_mandate(seed: u64, n_symbols: usize, allow_short: bool) -> Mandate {
    let mut rng = SplitMix64::new(seed);

    let styles: Vec<MandateStyle> = MandateStyle::ALL
        .iter()
        .copied()
        .filter(|s| allow_short || !s.requires_short())
        .collect();
    let style = styles[(rng.next_unit() * styles.len() as f64) as usize];

    let max_drawdown = if rng.next_unit() < 0.5 {
        Some(DRAWDOWN_CAPS[(rng.next_unit() * DRAWDOWN_CAPS.len() as f64) as usize])
    } else {
        None
    };

    let benchmark = if rng.next_unit() < 0.3 {
        let n = n_symbols.max(1);
        let idx = (rng.next_unit() * n as f64) as usize;
        Some(format!("SYM{idx:02}"))
    } else {
        None
    };

    // Drawn last so adding it leaves the style/drawdown/benchmark draws above byte-stable.
    let max_inventory = if rng.next_unit() < 0.4 {
        Some(INVENTORY_CAPS[(rng.next_unit() * INVENTORY_CAPS.len() as f64) as usize])
    } else {
        None
    };

    let text = render_text(style, max_drawdown, max_inventory, benchmark.as_deref());
    Mandate {
        style,
        max_drawdown,
        max_inventory,
        benchmark,
        text,
    }
}

/// Realized max drawdown of the per-bar return series, as a positive fraction. Pure
/// `mul`/`add`/`div` — no transcendentals, so byte-identical across runtimes.
fn max_drawdown(returns: &[f64]) -> f64 {
    let mut equity = 1.0_f64;
    let mut peak = 1.0_f64;
    let mut mdd = 0.0_f64;
    for &r in returns {
        equity *= 1.0 + r;
        if equity > peak {
            peak = equity;
        }
        if peak > EPS {
            let dd = (peak - equity) / peak;
            if dd > mdd {
                mdd = dd;
            }
        }
    }
    mdd
}

/// A bounded breach penalty in `[0, 1]` (0 = clean, 1 = fully breached).
///
/// Three independent breach sources, combined by worst-case (`max`) so a clean structure
/// with a blown drawdown — or a blown inventory cap — still scores the violation:
///
/// * **structural** — a short under `LongOnly` (fraction of bars holding a short); net
///   exposure away from zero under `MarketNeutral` *or* `PairsConvergence` (mean
///   `|net| / gross`). Read off the per-bar target-weight vectors. `Momentum` /
///   `Unconstrained` carry no structural rule.
/// * **inventory** — per-bar gross exposure `Σ|w_i|` over `max_inventory`, normalized by the
///   cap and *squared* (the Avellaneda-Stoikov inventory penalty), saturated at 1 per bar,
///   then meaned across bars.
/// * **drawdown** — realized max drawdown over the cap, normalized by the cap and saturated
///   at 1.
///
/// The `PairsConvergence` structural rule is a beta-free proxy: with no per-pair beta in the
/// mandate it rewards *dollar*-neutrality (`mean |Σ w_i| / Σ|w_i| → 0`), the same measure as
/// `MarketNeutral`. It scores the directional-exposure component of a true `w_x*beta + w_y`
/// hedge; the unit-beta case is exact, off-unit-beta is approximated.
///
/// `weights[i]` is the target-weight vector the rollout recorded on bar `i` (empty vectors
/// and an empty outer slice are safe — they contribute no structural/inventory breach). Pure
/// and allocation-light; safe on empty inputs (returns 0).
pub fn mandate_breach(m: &Mandate, returns: &[f64], weights: &[Vec<f64>]) -> f64 {
    let mut worst = 0.0_f64;

    if !weights.is_empty() {
        match m.style {
            MandateStyle::LongOnly => {
                let short_steps = weights
                    .iter()
                    .filter(|w| w.iter().cloned().fold(f64::INFINITY, f64::min) < -EPS)
                    .count();
                worst = worst.max(short_steps as f64 / weights.len() as f64);
            }
            MandateStyle::MarketNeutral | MandateStyle::PairsConvergence => {
                let mut sum = 0.0_f64;
                for w in weights {
                    let gross: f64 = w.iter().map(|x| x.abs()).sum();
                    let net: f64 = w.iter().sum::<f64>().abs();
                    sum += if gross > EPS { net / gross } else { 0.0 };
                }
                worst = worst.max((sum / weights.len() as f64).min(1.0));
            }
            MandateStyle::Momentum | MandateStyle::Unconstrained => {}
        }

        if let Some(cap) = m.max_inventory {
            let denom = cap.max(EPS);
            let mut sum = 0.0_f64;
            for w in weights {
                let gross: f64 = w.iter().map(|x| x.abs()).sum();
                if gross > cap {
                    let over = (gross - cap) / denom;
                    sum += (over * over).min(1.0);
                }
            }
            worst = worst.max((sum / weights.len() as f64).min(1.0));
        }
    }

    if let Some(cap) = m.max_drawdown {
        if !returns.is_empty() {
            let mdd = max_drawdown(returns);
            if mdd > cap {
                worst = worst.max(((mdd - cap) / cap.max(EPS)).min(1.0));
            }
        }
    }

    worst.min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_is_deterministic() {
        assert_eq!(sample_mandate(42, 4, true), sample_mandate(42, 4, true));
        assert_eq!(sample_mandate(7, 3, false), sample_mandate(7, 3, false));
    }

    #[test]
    fn style_varies_across_seeds() {
        let styles: std::collections::HashSet<MandateStyle> =
            (0..64).map(|s| sample_mandate(s, 4, true).style).collect();
        assert!(
            styles.len() > 1,
            "expected style variation across the seed space"
        );
    }

    #[test]
    fn no_short_never_yields_market_neutral() {
        for s in 0..64 {
            assert_ne!(
                sample_mandate(s, 4, false).style,
                MandateStyle::MarketNeutral
            );
        }
    }

    #[test]
    fn style_serializes_snake_case() {
        // The wire labels the Python contract (STYLES / validate_mandate) speaks.
        let labels: Vec<String> = MandateStyle::ALL
            .iter()
            .map(|s| serde_json::to_string(s).unwrap())
            .collect();
        assert_eq!(
            labels,
            vec![
                "\"long_only\"",
                "\"market_neutral\"",
                "\"momentum\"",
                "\"unconstrained\"",
                "\"pairs_convergence\""
            ]
        );
    }

    #[test]
    fn mandate_round_trips_through_json() {
        let m = sample_mandate(42, 4, true);
        let json = serde_json::to_string(&m).unwrap();
        let back: Mandate = serde_json::from_str(&json).unwrap();
        assert_eq!(m, back);
        // The plain-JSON form carries every key (null for absent options).
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        for key in [
            "style",
            "max_drawdown",
            "max_inventory",
            "benchmark",
            "text",
        ] {
            assert!(v.get(key).is_some(), "missing key {key}");
        }
    }

    #[test]
    fn old_json_without_new_fields_still_parses() {
        // A pre-extension trace (no max_inventory key) must still deserialize — the field
        // defaults to None.
        let m: Mandate = serde_json::from_str(
            r#"{"style":"long_only","max_drawdown":null,"benchmark":null,"text":""}"#,
        )
        .unwrap();
        assert_eq!(m.max_inventory, None);
        assert_eq!(m.style, MandateStyle::LongOnly);
    }

    #[test]
    fn breach_zero_on_clean_long_only() {
        let m = Mandate {
            style: MandateStyle::LongOnly,
            max_drawdown: None,
            max_inventory: None,
            benchmark: None,
            text: String::new(),
        };
        let weights = vec![vec![0.5, 0.3]; 4];
        assert_eq!(mandate_breach(&m, &[0.01, 0.0, 0.01, 0.0], &weights), 0.0);
    }

    #[test]
    fn breach_positive_on_short_under_long_only() {
        let m = Mandate {
            style: MandateStyle::LongOnly,
            max_drawdown: None,
            max_inventory: None,
            benchmark: None,
            text: String::new(),
        };
        let weights = vec![vec![-0.5, 0.2]; 4];
        assert!(mandate_breach(&m, &[0.01, 0.0], &weights) > 0.0);
    }

    #[test]
    fn breach_positive_on_drawdown_cap() {
        let m = Mandate {
            style: MandateStyle::LongOnly,
            max_drawdown: Some(0.10),
            max_inventory: None,
            benchmark: None,
            text: String::new(),
        };
        let clean = vec![vec![0.5, 0.3]; 2];
        // A -30% bar against a 10% cap penalizes even when long-only.
        assert!(mandate_breach(&m, &[-0.30, 0.0], &clean) > 0.0);
    }

    #[test]
    fn market_neutral_balanced_clean_one_sided_breaches() {
        let m = Mandate {
            style: MandateStyle::MarketNeutral,
            max_drawdown: None,
            max_inventory: None,
            benchmark: None,
            text: String::new(),
        };
        assert_eq!(mandate_breach(&m, &[], &[vec![0.5, -0.5]]), 0.0);
        assert!(mandate_breach(&m, &[], &[vec![0.5, 0.5]]) > 0.0);
    }

    #[test]
    fn breach_is_bounded() {
        let m = Mandate {
            style: MandateStyle::LongOnly,
            max_drawdown: Some(0.10),
            max_inventory: None,
            benchmark: None,
            text: String::new(),
        };
        let weights = vec![vec![-0.5, 0.2]; 2];
        let b = mandate_breach(&m, &[-0.9, -0.9], &weights);
        assert!((0.0..=1.0).contains(&b));
    }

    #[test]
    fn empty_inputs_are_clean() {
        let m = sample_mandate(1, 4, true);
        assert_eq!(mandate_breach(&m, &[], &[]), 0.0);
    }

    #[test]
    fn no_short_never_yields_pairs_convergence() {
        // Pairs-convergence needs an offsetting short leg, so a long-only market never draws it.
        for s in 0..64 {
            assert_ne!(
                sample_mandate(s, 4, false).style,
                MandateStyle::PairsConvergence
            );
        }
    }

    #[test]
    fn inventory_cap_clean_under_breaches_over_and_bounded() {
        let m = Mandate {
            style: MandateStyle::Unconstrained,
            max_drawdown: None,
            max_inventory: Some(1.0),
            benchmark: None,
            text: String::new(),
        };
        // Gross |0.5| + |0.3| = 0.8 ≤ 1.0 cap -> clean.
        assert_eq!(mandate_breach(&m, &[], &[vec![0.5, 0.3]]), 0.0);
        // Gross |0.9| + |0.8| = 1.7 > 1.0 cap -> squared breach, strictly positive.
        let b = mandate_breach(&m, &[], &[vec![0.9, 0.8]]);
        assert!(b > 0.0 && b <= 1.0);
        // A grossly over-leveraged book stays bounded at 1.
        let blown = mandate_breach(&m, &[], &[vec![5.0, 5.0], vec![5.0, 5.0], vec![5.0, 5.0]]);
        assert!((0.0..=1.0).contains(&blown));
        assert_eq!(blown, 1.0);
    }

    #[test]
    fn inventory_breach_is_squared() {
        // The penalty grows with the *square* of the normalized excess: a 2x-over book is
        // penalized ~4x a 1x-over book (until each saturates), unlike a linear term.
        let m = Mandate {
            style: MandateStyle::Unconstrained,
            max_drawdown: None,
            max_inventory: Some(1.0),
            benchmark: None,
            text: String::new(),
        };
        // excess 0.1 -> 0.01 ; excess 0.2 -> 0.04 ; ratio ~4x.
        let small = mandate_breach(&m, &[], &[vec![1.1]]);
        let big = mandate_breach(&m, &[], &[vec![1.2]]);
        assert!(small > 0.0);
        assert!(big > small * 3.0);
    }

    #[test]
    fn pairs_convergence_clean_when_dollar_neutral() {
        let m = Mandate {
            style: MandateStyle::PairsConvergence,
            max_drawdown: None,
            max_inventory: None,
            benchmark: None,
            text: String::new(),
        };
        // Offsetting legs (dollar-neutral) -> clean; a directional book -> breach.
        assert_eq!(mandate_breach(&m, &[], &[vec![0.5, -0.5]]), 0.0);
        assert!(mandate_breach(&m, &[], &[vec![0.5, 0.5]]) > 0.0);
    }
}
