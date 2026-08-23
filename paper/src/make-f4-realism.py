#!/usr/bin/env python3
"""F4: stylized-facts realism certification of the scenario generator.

Rolls a flat (zero-weight) policy through seeded episodes per tier, collecting
the point-in-time ``closes`` vector each bar into a (T, n_symbols) price panel,
then grades each panel with ``sharpearena.certify_realism`` against the
directional Cont-stylized-facts bounds. Writes per-seed reports, per-tier
aggregates, and a grouped-bar figure of the mean fact values.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sharpearena import SharpeArenaEnv, certify_realism

PAPER = Path(__file__).resolve().parents[1]
EVIDENCE = PAPER / "evidence"
FIGURES = PAPER / "figures"

TIERS = ("calm", "hard", "extreme")
SEEDS = list(range(8))
N_SYMBOLS = 4
N_DAYS = 120
MAX_STEPS = 512


def _collect_panel(tier: str, seed: int) -> np.ndarray:
    env = SharpeArenaEnv(
        n_symbols=N_SYMBOLS, n_days=N_DAYS, seed=seed, distribution_mode=tier
    )
    obs, _ = env.reset(seed=seed)
    closes = [np.asarray(obs["closes"], dtype=np.float64).reshape(-1)]
    flat = np.zeros(N_SYMBOLS, dtype=np.float32)
    for _ in range(MAX_STEPS):
        obs, _reward, terminated, truncated, _info = env.step(flat)
        closes.append(np.asarray(obs["closes"], dtype=np.float64).reshape(-1))
        if terminated or truncated:
            break
    return np.stack(closes, axis=0)


def main() -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    tiers: dict[str, dict] = {}
    for tier in TIERS:
        per_seed = []
        for seed in SEEDS:
            panel = _collect_panel(tier, seed)
            report = certify_realism(panel, kind="price")
            per_seed.append(
                {
                    "seed": seed,
                    "n_bars": int(panel.shape[0]),
                    "facts": report.facts,
                    "checks": report.checks,
                    "passed": report.passed,
                }
            )
        fact_names = sorted(per_seed[0]["facts"])
        mean_facts = {
            f: float(np.nanmean([r["facts"][f] for r in per_seed])) for f in fact_names
        }
        tiers[tier] = {
            "per_seed": per_seed,
            "mean_facts": mean_facts,
            "pass_rate": float(np.mean([r["passed"] for r in per_seed])),
        }

    out = {
        "finding": "F4",
        "config": {
            "n_symbols": N_SYMBOLS,
            "n_days": N_DAYS,
            "seeds": SEEDS,
            "max_steps": MAX_STEPS,
            "tiers": list(TIERS),
        },
        "tiers": tiers,
    }
    (EVIDENCE / "f4-realism.json").write_text(json.dumps(out, indent=2))

    # Figure: mean fact value per tier, grouped by fact.
    fact_names = sorted(tiers[TIERS[0]]["mean_facts"])
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.8 / len(TIERS)
    for j, tier in enumerate(TIERS):
        xs = [i + (j - (len(TIERS) - 1) / 2) * width for i in range(len(fact_names))]
        ys = [tiers[tier]["mean_facts"][f] for f in fact_names]
        ax.bar(xs, ys, width=width, label=f"{tier} (pass {tiers[tier]['pass_rate']:.0%})")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(fact_names)))
    ax.set_xticklabels(fact_names, rotation=30, ha="right")
    ax.set_ylabel("mean stylized-fact value")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "f4-realism.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
