"""Fit ``martingale_v1`` on fit seeds only, under the frozen protocol.

    python studies/generator-calibration/fit.py <sharpebench-data-dir>

Writes ``fit.json`` beside this file: the targets, every grid cell and the chosen
parameters. Nothing here reads a report seed. ``martingale_panels`` is the exact port of
the Rust recursion, which ``report.py`` checks bit for bit against the shipped tier.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from fractions import Fraction
from itertools import product
from pathlib import Path

import numpy as np

from stats import realism, sha256

HERE = Path(__file__).resolve().parent
PROTOCOL = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
MIX_XOR = 0x1234_5678_9ABC_DEF0
GAMMA = 0x9E37_79B9_7F4A_7C15
E_ABS_Z = Fraction(7, 15)


def uniforms(seed: int, n: int) -> np.ndarray:
    """The first ``n`` draws of ``SplitMix64::new(seed ^ salt)``, as the crate makes them."""
    salt = int(PROTOCOL["recursion"]["stream_salt"].replace("_", ""), 16)
    state = (seed ^ salt ^ MIX_XOR) & 0xFFFF_FFFF_FFFF_FFFF
    idx = np.arange(1, n + 1, dtype=np.uint64)
    z = np.uint64(state) + idx * np.uint64(GAMMA)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58_476D_1CE4_E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D0_49BB_1331_11EB)
    z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def draws_for(seeds: range, n_symbols: int, n_days: int) -> np.ndarray:
    """Uniforms shaped (seed, symbol, bar, draw). The count never depends on parameters."""
    burn = PROTOCOL["recursion"]["burn_in_bars"]
    per = PROTOCOL["recursion"]["draws_per_bar"]
    bars = burn + n_days - 1
    out = np.empty((len(seeds), n_symbols, bars, per))
    for i, seed in enumerate(seeds):
        out[i] = uniforms(seed, n_symbols * bars * per).reshape(n_symbols, bars, per)
    return out


def coefficients(g: float, persistence: float, p: float, k: float, sbar: float) -> dict[str, float]:
    m = float(E_ABS_Z) * ((1.0 - p) + p * k)
    a = g / m
    b = persistence - g
    w = sbar * (1.0 - a * m - b)
    return {"a": a, "b": b, "w": w, "m": m}


def martingale_panels(draws: np.ndarray, *, a, b, w, p, k, sbar) -> np.ndarray:
    """Close panels shaped (seed, T, N). Operation order matches the Rust recursion."""
    burn = PROTOCOL["recursion"]["burn_in_bars"]
    cap = PROTOCOL["recursion"]["cap"] * sbar
    n_seeds, n_symbols, bars, _ = draws.shape
    s = np.full((n_seeds, n_symbols), sbar)
    price = np.full((n_seeds, n_symbols), 100.0)
    closes = np.empty((n_seeds, bars - burn + 1, n_symbols))
    closes[:, 0, :] = price
    for bar in range(bars):
        u = draws[:, :, bar, :]
        z = u[..., 0] + u[..., 1] + u[..., 2] + u[..., 3] - 2.0
        z = np.where(u[..., 4] < p, k * z, z)
        r = s * z
        if bar >= burn:
            price = price * (1.0 + np.maximum(r, -0.95))
            closes[:, bar - burn + 1, :] = price
        s = np.minimum(w + a * np.maximum(r, -r) + b * s, cap)
    return closes


def series_facts(log_returns: np.ndarray) -> tuple[float, float, float]:
    """Per-series kurtosis, mean |r| ACF over lags 1..10, and daily volatility."""
    x = log_returns.reshape(-1, 1)
    return (
        realism._excess_kurtosis(x),
        realism._abs_return_autocorr(x, 10),
        float(x.std(ddof=1)),
    )


def read_series(path: Path) -> dict[str, list[float]]:
    series: dict[str, list[float]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            series[row["symbol"]].append(float(row["close"]))
    return dict(series)


def tape_stats(data: Path, names: list[str]) -> dict:
    """Medians over windows, with a dropped-window count. Holds no observation."""
    width = PROTOCOL["targets"]["window_closes"]
    facts = []
    dropped = 0
    files = {}
    for name in names:
        path = data / name
        digest = sha256(path)
        expected = (data / f"{name}.sha256").read_text(encoding="utf-8").split()[0]
        if digest != expected:
            raise SystemExit(f"{name}: sha256 {digest} does not match its sidecar")
        files[name] = digest
        for closes in read_series(path).values():
            for start in range(0, len(closes) - width + 1, width):
                window = np.array(closes[start : start + width])
                if not np.all(window > 0.0):
                    dropped += 1
                    continue
                facts.append(series_facts(np.diff(np.log(window))))
    arr = np.array(facts)
    return {
        "files": files,
        "windows": len(facts),
        "windows_dropped_nonpositive": dropped,
        "excess_kurtosis": float(np.median(arr[:, 0])),
        "abs_return_autocorr": float(np.median(arr[:, 1])),
        "daily_vol": float(np.median(arr[:, 2])),
    }


def panel_facts(closes: np.ndarray) -> dict[str, float]:
    log_r = np.log(closes[:, 1:, :] / closes[:, :-1, :])
    per = np.array(
        [series_facts(log_r[i, :, c]) for i in range(log_r.shape[0]) for c in range(log_r.shape[2])]
    )
    passed = [realism.certify_realism(panel, kind="return").passed for panel in log_r]
    return {
        "median_excess_kurtosis": float(np.median(per[:, 0])),
        "median_abs_return_autocorr": float(np.median(per[:, 1])),
        "median_daily_vol": float(np.median(per[:, 2])),
        "pass_share": float(np.mean(passed)),
    }


def round_sig(x: float, digits: int) -> float:
    return float(f"{x:.{digits - 1}e}")


def main() -> None:
    data = Path(sys.argv[1])
    target_cfg = PROTOCOL["targets"]
    target = tape_stats(data, target_cfg["fit_files"])
    comparison = {name: tape_stats(data, [name]) for name in target_cfg["comparison_files"]}
    fit = PROTOCOL["seeds"]["fit"]
    geometry = PROTOCOL["geometry"]
    draws = draws_for(
        range(fit["start"], fit["start"] + fit["count"]),
        geometry["n_symbols"],
        geometry["n_days"],
    )
    grid = PROTOCOL["grid"]
    cells = []
    for g, persistence, p, k in product(
        grid["alpha_share_g"], grid["persistence_P"], grid["tail_probability_p"], grid["tail_scale_k"]
    ):
        # Log returns make the facts weakly scale-dependent, so each cell is compared at
        # the volatility target. A unit-scale pass sets that scale; it fits no shape.
        unit = coefficients(g, persistence, p, k, 1e-3)
        trial = martingale_panels(draws, a=unit["a"], b=unit["b"], w=unit["w"], p=p, k=k, sbar=1e-3)
        trial_r = trial[:, 1:, :] / trial[:, :-1, :] - 1.0
        sbar = 1e-3 * target["daily_vol"] / float(np.median(trial_r.std(axis=1, ddof=1)))
        co = coefficients(g, persistence, p, k, sbar)
        closes = martingale_panels(draws, a=co["a"], b=co["b"], w=co["w"], p=p, k=k, sbar=sbar)
        facts = panel_facts(closes)
        j = (
            (facts["median_excess_kurtosis"] - target["excess_kurtosis"]) / target["excess_kurtosis"]
        ) ** 2 + (
            (facts["median_abs_return_autocorr"] - target["abs_return_autocorr"])
            / target["abs_return_autocorr"]
        ) ** 2
        cells.append(
            {"g": g, "P": persistence, "p": p, "k": k, "a": co["a"], "b": co["b"], "sbar": sbar, "J": j, **facts}
        )
    bar = PROTOCOL["selection"]["min_fit_pass_share"]
    eligible = [c for c in cells if c["pass_share"] >= bar]
    pool = eligible or cells
    best = min(pool, key=lambda c: (c["J"], c["a"], c["p"], c["k"]))
    sbar = round_sig(best["sbar"], PROTOCOL["selection"]["sbar_significant_figures"])
    co = coefficients(best["g"], best["P"], best["p"], best["k"], sbar)
    cap_scale = PROTOCOL["recursion"]["cap"] * sbar
    floor_margin = cap_scale * 2.0 * best["k"]
    if floor_margin >= 0.95:
        raise SystemExit(f"floor reachable: {floor_margin}")
    chosen = {
        "g": best["g"],
        "P": best["P"],
        "p": best["p"],
        "k": best["k"],
        "a": co["a"],
        "b": co["b"],
        "w": co["w"],
        "sbar": sbar,
        "cap_scale": cap_scale,
        "largest_possible_move": floor_margin,
    }
    closes = martingale_panels(draws, a=co["a"], b=co["b"], w=co["w"], p=best["p"], k=best["k"], sbar=sbar)
    fitted_vol = panel_facts(closes)["median_daily_vol"]
    if abs(fitted_vol / target["daily_vol"] - 1.0) > 0.02:
        raise SystemExit(f"volatility target missed: {fitted_vol} vs {target['daily_vol']}")
    out = {
        "protocol_version": PROTOCOL["version"],
        "target": target,
        "comparison_not_fit": comparison,
        "eligible_cells": len(eligible),
        "grid_cells": len(cells),
        "fit_pass_bar_met": bool(eligible),
        "chosen": chosen,
        "chosen_fit_seed_facts": panel_facts(closes),
        "note": "cells are compared at the volatility target; sbar is then rounded once",
        "cells": cells,
    }
    path = HERE / "fit.json"
    path.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print("target", target)
    print("comparison", json.dumps(comparison, indent=0))
    print("eligible", len(eligible), "of", len(cells))
    print("chosen", {k: repr(v) for k, v in chosen.items()})
    print("fit facts", out["chosen_fit_seed_facts"])
    for c in sorted(cells, key=lambda c: c["J"])[:8]:
        print({k: round(c[k], 4) for k in ("g", "P", "p", "k", "J", "pass_share", "median_excess_kurtosis", "median_abs_return_autocorr")})


if __name__ == "__main__":
    main()
