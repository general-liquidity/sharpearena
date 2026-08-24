#!/usr/bin/env python3
"""F1: the baseline leaderboard under the corrected kernel.

Rolls the reference-policy field over fixed seeds in each scenario tier via
``sharpearena.run_baselines`` (which scores pooled returns with the SharpeBench
``score_run`` kernel and attaches a seed-paired bootstrap CI per row), and writes
the rows, the rendered leaderboard tables, and a grouped-bar figure.

Deterministic: seeds, tiers and bootstrap parameters are fixed below.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sharpearena

PAPER = Path(__file__).resolve().parents[1]
EVIDENCE = PAPER / "evidence"
FIGURES = PAPER / "figures"

TIERS = ("calm", "hard", "extreme")
SEEDS = list(range(16))
N_SYMBOLS = 4
N_DAYS = 120
# Bootstrap parameters, stated explicitly on the committed command surface.
# These equal the library defaults in sharpearena.confidence
# (DEFAULT_N_BOOT, DEFAULT_RESAMPLE_SEED): percentile bootstrap, alpha 0.05.
N_BOOT = 2000
RESAMPLE_SEED = 0x5BA7_2026


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("sharpearena")
    except Exception:
        return getattr(sharpearena, "__version__", "unknown")


def main() -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    tiers: dict[str, dict] = {}
    for tier in TIERS:
        rows = sharpearena.run_baselines(
            n_symbols=N_SYMBOLS,
            n_days=N_DAYS,
            seeds=SEEDS,
            distribution_mode=tier,
            confidence=True,
            n_boot=N_BOOT,
            resample_seed=RESAMPLE_SEED,
        )
        tiers[tier] = {
            "rows": rows,
            "leaderboard_markdown": sharpearena.leaderboard_markdown(rows, show_ci=True),
        }

    out = {
        "finding": "F1",
        "package_version": _version(),
        "config": {
            "n_symbols": N_SYMBOLS,
            "n_days": N_DAYS,
            "seeds": SEEDS,
            "tiers": list(TIERS),
            "bootstrap": {
                "n_boot": N_BOOT,
                "resample_seed": RESAMPLE_SEED,
                "alpha": 0.05,
                "convention": "seed-paired percentile bootstrap on the deflated Sharpe",
            },
        },
        "tiers": tiers,
    }
    (EVIDENCE / "f1-baselines.json").write_text(json.dumps(out, indent=2))

    # Figure: deflated Sharpe per policy, grouped by tier, with bootstrap CI bars.
    policies = [r["policy"] for r in tiers[TIERS[0]]["rows"]]
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.8 / len(TIERS)
    for j, tier in enumerate(TIERS):
        rows = {r["policy"]: r for r in tiers[tier]["rows"]}
        xs = [i + (j - (len(TIERS) - 1) / 2) * width for i in range(len(policies))]
        ys = [rows[p]["deflated_sharpe"] for p in policies]
        ci = [rows[p].get("deflated_sharpe_ci") or {} for p in policies]
        lo = [y - c.get("lo", y) for y, c in zip(ys, ci)]
        hi = [c.get("hi", y) - y for y, c in zip(ys, ci)]
        ax.bar(xs, ys, width=width, yerr=[lo, hi], capsize=2, label=tier)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(policies)))
    ax.set_xticklabels(policies, rotation=30, ha="right")
    ax.set_ylabel("deflated Sharpe (score_run)")
    ax.legend(title="tier", frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "f1-baselines.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
