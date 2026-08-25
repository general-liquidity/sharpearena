#!/usr/bin/env python3
"""F2: regret against the closed-form Avellaneda-Stoikov reference policy.

``sharpearena.mm_regret`` runs the candidate and ``closed_form_reference_policy``
on identical seeded episodes and reports the mean reward gap. Candidates: the
closed-form policy itself (must be ~0 by construction) and fixed-spread quoters
across a half-spread grid. The script additionally rolls both policies per
episode through the public ``MarketMakingEnv`` to serialize the per-episode
regret vectors and a t-based 95% CI per grid point (the aggregate API returns
only the mean). Writes the regret curve as JSON and a figure.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sharpearena import (
    MarketMakingEnv,
    MMParams,
    closed_form_reference_policy,
    fixed_spread_policy,
    mm_regret,
)

PAPER = Path(__file__).resolve().parents[1]
EVIDENCE = PAPER / "evidence"
FIGURES = PAPER / "figures"

N_EPISODES = 16
SEED_BASE = 0
HALF_SPREADS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
# Two-sided 97.5% Student-t critical value at df = N_EPISODES - 1 = 15.
T_CRIT_DF15 = 2.131


def _episode_reward(env: MarketMakingEnv, policy, seed: int) -> float:
    obs, _ = env.reset(seed=seed)
    total = 0.0
    while True:
        obs, reward, terminated, truncated, _ = env.step(policy(obs))
        total += reward
        if terminated or truncated:
            return total


def _per_episode_regret(policy, params: MMParams) -> list[float]:
    """Per-episode reward gap to the closed-form policy on shared seeds."""
    optimal = closed_form_reference_policy(params)
    env = MarketMakingEnv(params)
    out = []
    for i in range(N_EPISODES):
        seed = SEED_BASE + i
        out.append(
            _episode_reward(env, optimal, seed) - _episode_reward(env, policy, seed)
        )
    return out


def _stats(values: list[float]) -> dict:
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    std = math.sqrt(var)
    half = T_CRIT_DF15 * std / math.sqrt(n)
    return {"mean": mean, "std": std, "ci95_lo": mean - half, "ci95_hi": mean + half}


def main() -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    params = MMParams()
    optimal_regret = mm_regret(
        closed_form_reference_policy(params),
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

    per_episode = {
        str(hs): _per_episode_regret(fixed_spread_policy(hs), params)
        for hs in HALF_SPREADS
    }
    dispersion = {hs: _stats(per_episode[str(hs)]) for hs in HALF_SPREADS}
    for hs in HALF_SPREADS:
        assert abs(dispersion[hs]["mean"] - fixed[hs]) < 1e-9, hs

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
            "ci_convention": "t-based 95% over per-episode paired gaps, df=15",
        },
        "optimal_regret": optimal_regret,
        "fixed_spread_regret": {str(k): v for k, v in fixed.items()},
        "per_episode_regret": per_episode,
        "regret_dispersion": {str(k): v for k, v in dispersion.items()},
    }
    (EVIDENCE / "f2-regret.json").write_text(json.dumps(out, indent=2))

    fig, ax = plt.subplots(figsize=(6, 4))
    xs = list(HALF_SPREADS)
    ys = [fixed[h] for h in HALF_SPREADS]
    lo = [ys[i] - dispersion[h]["ci95_lo"] for i, h in enumerate(HALF_SPREADS)]
    hi = [dispersion[h]["ci95_hi"] - ys[i] for i, h in enumerate(HALF_SPREADS)]
    ax.errorbar(
        xs, ys, yerr=[lo, hi], marker="o", capsize=3, label="fixed-spread quoter"
    )
    ax.axhline(
        optimal_regret, linestyle="--", color="black", linewidth=0.8,
        label="A-S closed-form reference",
    )
    ax.set_xscale("log")
    ax.set_xlabel("fixed half-spread (price units)")
    ax.set_ylabel("mean regret vs closed-form reference")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "f2-regret.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
