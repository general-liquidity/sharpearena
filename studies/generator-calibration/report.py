r"""Grade ``martingale_v1`` on report seeds against the frozen criteria.

    cargo run --release -p sharpearena --example dump-scenario-panels -- <panels>/report 128 128 4 120 \
        calm hard extreme cointegrated_pairs regime_shift calm_candidate martingale_v1
    python studies/generator-calibration/report.py <panels>/report

Writes ``report.json`` beside this file. It reads the shipped tier, so it also confirms
that the port the fit used reproduces that tier bit for bit; if not, the fit is void.
The gate counts are a development measurement, not the frozen realism study.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from fit import PROTOCOL, draws_for, martingale_panels
from stats import load_panels, measure, realism, realism_counts

HERE = Path(__file__).resolve().parent
TIERS = ("calm", "hard", "extreme", "cointegrated_pairs", "regime_shift", "calm_candidate", "martingale_v1")
STRATEGIES = (
    "sharpe_buy_and_hold_equal_weight",
    "sharpe_momentum_1bar",
    "sharpe_reversal_1bar",
    "sharpe_trend_20bar",
)


def inside(stat: dict, margin: float) -> bool:
    return -margin <= stat["ci95_low"] and stat["ci95_high"] <= margin


def grade(stats: dict, calm: dict, gate: dict, n_returns: int, n_symbols: int) -> dict:
    c = PROTOCOL["criteria"]
    null = realism._abs_acf_iid_upper(n_returns, n_symbols, 10)
    return {
        "C1_lag1": inside(stats["lag1_autocorr"], c["C1_lag1_margin"]),
        "C2_mean_daily_return": inside(stats["mean_daily_return"], c["C2_mean_daily_return_margin"]),
        "C3_buy_and_hold": inside(stats["buy_and_hold_sharpe_mean_symbol"], c["C3_buy_and_hold_sharpe_margin"]),
        "C4_strategies": all(inside(stats[s], c["C4_strategy_sharpe_margin"]) for s in STRATEGIES),
        "C5_kurtosis": stats["excess_kurtosis"]["median"] > 0.0
        and stats["excess_kurtosis"]["median"] > calm["excess_kurtosis"]["median"],
        "C6_clustering": stats["abs_return_autocorr"]["ci95_low"] > null,
        "C7_gate_share": gate["passed"] / gate["panels"] >= c["C7_min_pass_share"],
        "abs_acf_iid_null": null,
    }


def port_matches(panels: np.ndarray) -> bool:
    fit = json.loads((HERE / "fit.json").read_text(encoding="utf-8"))["chosen"]
    seeds = PROTOCOL["seeds"]["report"]
    g = PROTOCOL["geometry"]
    draws = draws_for(range(seeds["start"], seeds["start"] + seeds["count"]), g["n_symbols"], g["n_days"])
    port = martingale_panels(draws, a=fit["a"], b=fit["b"], w=fit["w"], p=fit["p"], k=fit["k"], sbar=fit["sbar"])
    return bool(np.array_equal(port, panels))


def main() -> None:
    root = Path(sys.argv[1])
    g = PROTOCOL["geometry"]
    seeds = PROTOCOL["seeds"]["report"]
    tiers = {}
    loaded = {}
    for tier in TIERS:
        panels = load_panels(root / f"{tier}.f64", g["n_days"], g["n_symbols"])
        if len(panels) != seeds["count"]:
            raise SystemExit(f"{tier}: expected {seeds['count']} panels, got {len(panels)}")
        loaded[tier] = panels
        tiers[tier] = {"stats": measure(panels), "realism_dev": realism_counts(panels)}
    calm = tiers["calm"]["stats"]
    for tier, body in tiers.items():
        body["criteria"] = grade(body["stats"], calm, body["realism_dev"], g["n_returns"], g["n_symbols"])
    out = {
        "protocol_version": PROTOCOL["version"],
        "seeds": seeds,
        "port_reproduces_tier_exactly": port_matches(loaded["martingale_v1"]),
        "realism_counts_are": "a development measurement, not the frozen realism study",
        "tiers": tiers,
    }
    path = HERE / "report.json"
    path.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print("port exact", out["port_reproduces_tier_exactly"])
    show = (
        "lag1_autocorr",
        "mean_daily_return",
        "daily_vol",
        "buy_and_hold_sharpe_mean_symbol",
        "sharpe_buy_and_hold_equal_weight",
        "sharpe_momentum_1bar",
        "sharpe_reversal_1bar",
        "sharpe_trend_20bar",
        "excess_kurtosis",
        "abs_return_autocorr",
        "vol_forecast_corr",
        "cross_symbol_corr",
    )
    for tier, body in tiers.items():
        print(tier, body["realism_dev"])
        print("  ", {k: v for k, v in body["criteria"].items()})
        for key in show:
            s = body["stats"][key]
            print(f"  {key:34s} {s['mean']:+.5f} [{s['ci95_low']:+.5f},{s['ci95_high']:+.5f}] med {s['median']:+.4f}")


if __name__ == "__main__":
    main()
