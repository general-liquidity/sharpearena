#!/usr/bin/env python3
"""Predictability probe: the in-band generator-inversion attack, measured.

The limitations section names a threat: scenarios are a pure function of a public
64-bit seed under a public deterministic transform, so future bars are in principle
a computable function of past bars, and the leak-free interface property says
nothing about how hard that computation is. This script converts the named threat
into a measurement with three adversaries evaluated on next-bar return prediction
from a causal prefix, across the three volatility tiers, 16 seeds each:

* baseline: the unconditional adversary, predicting the expanding prefix mean
  return per symbol (the martingale-with-drift null).
* honest: a statistical adversary with no knowledge of the generator, a ridge
  AR(p) regressor refit causally on the prefix only at every bar.
* oracle: the known-seed adversary, an upper bound on any inversion attack; with
  the seed in hand the generator replays the panel and every future bar is known
  exactly.
* seed-search: the strongest cheap in-band inversion we could implement, a brute
  scan over a bounded seed band matching the observed first-bar closes, then
  verifying the surviving candidate against a longer prefix. Full 2^64 inversion
  is infeasible by enumeration; this characterizes the gap between the bounded
  search and the oracle it collapses to whenever the band covers the true seed.

Metrics: directional accuracy and MSE against the unconditional baseline, plus
the trading value of each predictor as a frictionless sign-following long/short
policy scored through the public ``score_run`` kernel (deflated Sharpe). Writes
``paper/evidence/predictability.json`` and ``paper/figures/predictability.pdf``.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from sharpearena import TradingEnv, score_run
except ImportError:  # --figures-only reads the committed JSON and needs no bindings
    TradingEnv = score_run = None

PAPER = Path(__file__).resolve().parents[1]
EVIDENCE = PAPER / "evidence"
FIGURES = PAPER / "figures"

TIERS = ("calm", "hard", "extreme")
N_SYMBOLS = 4
N_DAYS = 120
N_SEEDS = 16
SEED_START = 50_000  # the published start_level of the deployed seed band
SEEDS = list(range(SEED_START, SEED_START + N_SEEDS))
WARMUP = 30  # causal prefix length before the first scored prediction
AR_ORDER = 5
RIDGE_LAMBDA = 1e-4
# The in-band adversary's plausible seed set: a 2^16-wide band around the
# published start_level. Specs publish (start_level, num_levels), so a bounded
# deployment band is public knowledge; an unbounded band is the full u64 space.
SEARCH_HALF_WIDTH = 1 << 15
MATCH_TOL = 1e-9
FLAT = json.dumps({"orders": []})


def extract_closes(seed: int, mode: str) -> np.ndarray:
    """Replay the public environment with flat decisions and collect the tape.

    This is exactly what an in-band adversary sees: past bars only, one at a
    time, through the observation interface. Returns (n_bars, n_symbols).
    """
    env = TradingEnv(
        n_symbols=N_SYMBOLS, n_days=N_DAYS, seed=seed, distribution_mode=mode
    )
    obs = json.loads(env.reset())
    rows = [[s["close_history"][-1] for s in obs["symbols"]]]
    done = False
    while not done:
        obs_json, _r, done, _info = env.step(FLAT)
        obs = json.loads(obs_json)
        rows.append([s["close_history"][-1] for s in obs["symbols"]])
    return np.asarray(rows, dtype=np.float64)


def first_closes(seed: int) -> np.ndarray:
    """First-bar closes for a candidate seed (tier-invariant: the volatility
    post-pass rewrites bars 1.. and never bar 0)."""
    env = TradingEnv(
        n_symbols=N_SYMBOLS, n_days=2, seed=seed, distribution_mode="calm"
    )
    obs = json.loads(env.reset())
    return np.asarray([s["close_history"][-1] for s in obs["symbols"]])


def ridge_ar_predictions(rets: np.ndarray) -> np.ndarray:
    """Causal ridge AR(p) next-bar predictions per symbol.

    rets: (T, S) simple returns. Returns preds (T, S) where preds[t] is the
    prediction of rets[t] made from rets[:t] only; NaN before WARMUP.
    """
    T, S = rets.shape
    preds = np.full((T, S), np.nan)
    p = AR_ORDER
    for s in range(S):
        x = rets[:, s]
        for t in range(WARMUP, T):
            hist = x[:t]
            n = len(hist) - p
            X = np.column_stack(
                [hist[p - 1 - k : p - 1 - k + n] for k in range(p)]
                + [np.ones(n)]
            )
            y = hist[p:]
            A = X.T @ X + RIDGE_LAMBDA * len(y) * np.eye(p + 1)
            w = np.linalg.solve(A, X.T @ y)
            preds[t, s] = np.concatenate([hist[t - p : t][::-1], [1.0]]) @ w
    return preds


def prefix_mean_predictions(rets: np.ndarray) -> np.ndarray:
    """The unconditional baseline: expanding causal mean return per symbol."""
    T, S = rets.shape
    preds = np.full((T, S), np.nan)
    csum = np.cumsum(rets, axis=0)
    for t in range(WARMUP, T):
        preds[t] = csum[t - 1] / t
    return preds


def evaluate(preds: np.ndarray, rets: np.ndarray) -> dict:
    """Directional accuracy, MSE, and the sign-following policy return series."""
    mask = ~np.isnan(preds[:, 0])
    p, a = preds[mask], rets[mask]
    acc = float(np.mean(np.sign(p) == np.sign(a)))
    mse = float(np.mean((p - a) ** 2))
    policy_rets = np.mean(np.where(p >= 0.0, 1.0, -1.0) * a, axis=1)
    return {"accuracy": acc, "mse": mse, "policy_returns": policy_rets}


def dsr(returns: np.ndarray) -> float:
    return float(json.loads(score_run(returns.tolist()))["deflated_sharpe"])


def seed_search(observed_first: np.ndarray, band: range) -> dict:
    """Scan the band for seeds whose first-bar closes match the observation."""
    t0 = time.time()
    matches = [
        int(c)
        for c in band
        if np.max(np.abs(first_closes(c) - observed_first)) < MATCH_TOL
    ]
    return {
        "matches": matches,
        "candidates_checked": len(band),
        "elapsed_s": time.time() - t0,
    }


def main() -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    # Seed-search inversion (tier-invariant first-bar fingerprint, run once per
    # true seed over the bounded band).
    band = range(SEED_START - SEARCH_HALF_WIDTH, SEED_START + SEARCH_HALF_WIDTH)
    search_results = []
    for seed in SEEDS:
        obs_first = first_closes(seed)
        res = seed_search(obs_first, band)
        res["true_seed"] = seed
        res["recovered"] = res["matches"] == [seed]
        search_results.append(res)
    recovered = sum(r["recovered"] for r in search_results)
    total_search_s = sum(r["elapsed_s"] for r in search_results)

    per_tier: dict[str, dict] = {}
    for tier in TIERS:
        rows = {
            name: {"accuracy": [], "mse": [], "dsr": []}
            for name in ("baseline", "honest", "oracle")
        }
        verified = 0
        for i, seed in enumerate(SEEDS):
            closes = extract_closes(seed, tier)
            rets = closes[1:] / closes[:-1] - 1.0  # rets[t] = bar t+1 return

            base = evaluate(prefix_mean_predictions(rets), rets)
            honest = evaluate(ridge_ar_predictions(rets), rets)
            # Oracle: the recovered (or known) seed replays the generator, so
            # the prediction IS the realized next bar. Verify the seed-search
            # winner reproduces the observed tier prefix before granting it.
            if search_results[i]["recovered"]:
                replay = extract_closes(search_results[i]["matches"][0], tier)
                if np.max(np.abs(replay[:WARMUP] - closes[:WARMUP])) < MATCH_TOL:
                    verified += 1
            oracle = evaluate(rets.copy(), rets)

            for name, ev in (("baseline", base), ("honest", honest), ("oracle", oracle)):
                rows[name]["accuracy"].append(ev["accuracy"])
                rows[name]["mse"].append(ev["mse"])
                rows[name]["dsr"].append(dsr(ev["policy_returns"]))

        per_tier[tier] = {
            "prefix_verified_recoveries": verified,
            "adversaries": {
                name: {
                    "accuracy_mean": float(np.mean(v["accuracy"])),
                    "accuracy_std": float(np.std(v["accuracy"])),
                    "mse_mean": float(np.mean(v["mse"])),
                    "dsr_mean": float(np.mean(v["dsr"])),
                    "dsr_std": float(np.std(v["dsr"])),
                    "per_seed_accuracy": v["accuracy"],
                    "per_seed_dsr": v["dsr"],
                }
                for name, v in rows.items()
            },
        }

    evidence = {
        "config": {
            "tiers": list(TIERS),
            "n_symbols": N_SYMBOLS,
            "n_days": N_DAYS,
            "seeds": SEEDS,
            "warmup_bars": WARMUP,
            "ar_order": AR_ORDER,
            "ridge_lambda": RIDGE_LAMBDA,
            "search_band_width": 2 * SEARCH_HALF_WIDTH,
            "match_tolerance": MATCH_TOL,
            "policy": "frictionless unit-gross long/short sign following, "
            "equal weight across symbols, scored via score_run deflated_sharpe",
        },
        "seed_search": {
            "recovered": recovered,
            "n_seeds": N_SEEDS,
            "band_width": 2 * SEARCH_HALF_WIDTH,
            "total_elapsed_s": total_search_s,
            "per_candidate_us": 1e6 * total_search_s / (N_SEEDS * len(band)),
            "collisions": [
                r for r in search_results if len(r["matches"]) != 1
            ],
            "full_u64_extrapolation_note": (
                "recovery cost scales linearly in band width; extrapolating the "
                "measured per-candidate cost to the full 2^64 space gives the "
                "brute-force wall-clock recorded here"
            ),
            "full_u64_extrapolated_years": float(
                (2**64) * (total_search_s / (N_SEEDS * len(band))) / (86400 * 365.25)
            ),
            "prefix_bars_needed": 1,
        },
        "tiers": per_tier,
    }
    out = EVIDENCE / "predictability.json"
    out.write_text(json.dumps(evidence, indent=2))
    print(f"wrote {out}")

    make_figure(per_tier)


def make_figure(per_tier: dict) -> None:
    # Figure: accuracy by tier (left), DSR by tier (right), three adversaries.
    # Two square-ish panels side by side, drawn at column width (5.5 in) so the
    # 9 to 10 pt fonts land at print size.
    FIGURES.mkdir(parents=True, exist_ok=True)
    names = ("baseline", "honest", "oracle")
    labels = ("baseline (prefix mean)", "honest (ridge AR)", "oracle (known seed)")
    colors = ("#9aa0a6", "#1a73e8", "#d93025")
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 3.3))
    x = np.arange(len(TIERS))
    w = 0.26
    for ax, metric, title in (
        (axes[0], "accuracy_mean", "Directional accuracy (next bar)"),
        (axes[1], "dsr_mean", "Deflated Sharpe (sign policy)"),
    ):
        err_key = "accuracy_std" if metric == "accuracy_mean" else "dsr_std"
        for k, (name, label, color) in enumerate(zip(names, labels, colors)):
            vals = [per_tier[t]["adversaries"][name][metric] for t in TIERS]
            errs = [per_tier[t]["adversaries"][name][err_key] for t in TIERS]
            ax.bar(x + (k - 1) * w, vals, w, yerr=errs, capsize=2,
                   label=label, color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(TIERS, fontsize=9)
        ax.tick_params(labelsize=9)
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].axhline(0.5, ls=":", c="k", lw=0.8)
    axes[0].set_ylim(0, 1.05)
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, fontsize=8.5, frameon=False, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(FIGURES / "predictability.pdf")
    print(f"wrote {FIGURES / 'predictability.pdf'}")


if __name__ == "__main__":
    if "--figures-only" in sys.argv:
        data = json.loads((EVIDENCE / "predictability.json").read_text())
        make_figure(data["tiers"])
    else:
        main()
