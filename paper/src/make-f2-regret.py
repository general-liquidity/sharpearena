#!/usr/bin/env python3
"""F2: regret against the closed-form Avellaneda-Stoikov optimum.

``sharpearena.mm_regret`` runs the candidate and ``analytically_optimal_policy``
on identical seeded episodes and reports the mean reward gap. Candidates: the
optimum itself (must be ~0 by construction) and fixed-spread quoters across a
half-spread grid. Writes the regret curve as JSON and a figure.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sharpearena import (
    MMParams,
    analytically_optimal_policy,
    fixed_spread_policy,
    mm_regret,
)

PAPER = Path(__file__).resolve().parents[1]
EVIDENCE = PAPER / "evidence"
FIGURES = PAPER / "figures"

N_EPISODES = 16
SEED_BASE = 0
HALF_SPREADS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0)


def main() -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    params = MMParams()
    optimal_regret = mm_regret(
        analytically_optimal_policy(params),
        params=params,
        n_episodes=N_EPISODES,
        seed_base=SEED_BASE,
    )
    fixed = {
        hs: mm_regret(
            fixed_spread_policy(hs),
            params=params,
            n_episodes=N_EPISODES,
            seed_base=SEED_BASE,
        )
        for hs in HALF_SPREADS
    }

    out = {
        "finding": "F2",
        "config": {
            "params": {
                "sigma": params.sigma,
                "gamma": params.gamma,
                "kappa": params.kappa,
                "arrival_rate": params.arrival_rate,
                "n_steps": params.n_steps,
                "dt": params.dt,
                "inventory_cap": params.inventory_cap,
                "phi": params.phi,
            },
            "n_episodes": N_EPISODES,
            "seed_base": SEED_BASE,
            "half_spreads": list(HALF_SPREADS),
        },
        "optimal_regret": optimal_regret,
        "fixed_spread_regret": {str(k): v for k, v in fixed.items()},
    }
    (EVIDENCE / "f2-regret.json").write_text(json.dumps(out, indent=2))

    fig, ax = plt.subplots(figsize=(6, 4))
    xs = list(HALF_SPREADS)
    ys = [fixed[h] for h in HALF_SPREADS]
    ax.plot(xs, ys, marker="o", label="fixed-spread quoter")
    ax.axhline(
        optimal_regret, linestyle="--", color="black", linewidth=0.8,
        label="A-S closed-form optimum",
    )
    ax.set_xscale("log")
    ax.set_xlabel("fixed half-spread (price units)")
    ax.set_ylabel("mean regret vs optimum")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "f2-regret.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
