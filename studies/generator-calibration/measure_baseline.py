r"""Measure the signal already in each generator tier, before any tier is designed.

    cargo run --release -p sharpearena --example dump-scenario-panels -- \
        <panels> 0 256 4 120 calm hard extreme cointegrated_pairs regime_shift calm_candidate
    cargo run --release -p sharpearena --example dump-scenario-panels -- <panels>/long 0 16 4 20000 calm
    python studies/generator-calibration/measure_baseline.py <panels>

The long calm paths separate the population value from the short-sample demeaning bias
of a 119-return estimator. Writes ``baseline.json`` beside this file. The 2026-09 audit
figures are not an input.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from stats import annualized_sharpe, lag1_autocorr, load_panels, measure, realism_counts, simple_returns

MODES = ("calm", "hard", "extreme", "cointegrated_pairs", "regime_shift", "calm_candidate")
SEEDS = {"start": 0, "count": 256}
N_DAYS, N_SYMBOLS = 120, 4
LONG_DAYS, LONG_SEEDS = 20000, 16


def long_calm(root: Path) -> dict:
    """Population check: pooled over 16 seeds of 20,000 bars, 4 symbols each."""
    panels = load_panels(root / "long" / "calm.f64", LONG_DAYS, N_SYMBOLS)
    if len(panels) != LONG_SEEDS:
        raise SystemExit(f"long calm: expected {LONG_SEEDS} panels, got {len(panels)}")
    lag1, sharpe = [], []
    for panel in panels:
        r = simple_returns(panel)
        lag1.append(lag1_autocorr(r))
        sharpe.append(annualized_sharpe(r))
    return {
        "seeds": LONG_SEEDS,
        "n_days": LONG_DAYS,
        "lag1_autocorr_mean": float(np.mean(lag1)),
        "buy_and_hold_sharpe_by_symbol": [float(x) for x in np.mean(sharpe, axis=0)],
    }


def main() -> None:
    root = Path(sys.argv[1])
    tiers = {}
    for mode in MODES:
        panels = load_panels(root / f"{mode}.f64", N_DAYS, N_SYMBOLS)
        if len(panels) != SEEDS["count"]:
            raise SystemExit(f"{mode}: expected {SEEDS['count']} panels, got {len(panels)}")
        tiers[mode] = {"stats": measure(panels), "realism_dev": realism_counts(panels)}
    out = {
        "seeds": SEEDS,
        "geometry": {"n_days": N_DAYS, "n_symbols": N_SYMBOLS, "n_returns": N_DAYS - 1},
        "interval": "t-based 95% across seeds, the seed being the replicate",
        "strategy_costs": "gross of all costs",
        "tiers": tiers,
        "long_calm": long_calm(root),
    }
    path = Path(__file__).with_name("baseline.json")
    path.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print("long calm", out["long_calm"])
    keys = (
        "lag1_autocorr",
        "mean_daily_return",
        "daily_vol",
        "buy_and_hold_sharpe_mean_symbol",
        "excess_kurtosis",
        "abs_return_autocorr",
        "vol_forecast_corr",
        "cross_symbol_corr",
        "sharpe_momentum_1bar",
        "sharpe_trend_20bar",
    )
    for mode, body in tiers.items():
        print(mode, body["realism_dev"])
        for key in keys:
            s = body["stats"][key]
            print(f"  {key:34s} {s['mean']:+.5f} [{s['ci95_low']:+.5f},{s['ci95_high']:+.5f}] sd {s['sd']:.5f}")
        print("  sym sharpe", [round(body["stats"][f"buy_and_hold_sharpe_sym{i:02d}"]["mean"], 3) for i in range(4)])


if __name__ == "__main__":
    main()
