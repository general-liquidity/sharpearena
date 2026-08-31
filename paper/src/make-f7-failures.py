#!/usr/bin/env python3
"""F7: failure-mode distributions over the baseline field.

Rolls every baseline policy (reference set plus behavioral counterparties)
over fixed seeds per tier, under a DrawdownStopper wrapper, with a per-scenario
sampled mandate. Each episode's return series and event stream (per-bar target
weights plus stop-out markers folded in) is classified by
``classify_episode_failure``; ``rollup_failure_modes`` tallies per tier and
overall. Writes JSON plus a grouped-bar figure of the mode counts per tier.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sharpearena import (
    DrawdownStopper,
    FailureMode,
    SharpeArenaEnv,
    check_env_effective_config,
    classify_episode_failure,
    merge_effective_configs,
    rollup_failure_modes,
    sample_mandate,
)
from sharpearena.baselines import BASELINE_POLICIES, BEHAVIORAL_POLICIES

PAPER = Path(__file__).resolve().parents[1]
EVIDENCE = PAPER / "evidence"
FIGURES = PAPER / "figures"

TIERS = ("calm", "hard", "extreme")
SEEDS = list(range(16))
N_SYMBOLS = 4
N_DAYS = 120
MAX_STEPS = 512
MAX_DRAWDOWN = 0.5

FIELD = list(BASELINE_POLICIES) + list(BEHAVIORAL_POLICIES)
_READBACK: dict[str, dict[int, dict]] = {tier: {} for tier in TIERS}


def _episode(tier: str, seed: int, policy) -> tuple[list[float], list[dict]]:
    base = SharpeArenaEnv(
        n_symbols=N_SYMBOLS, n_days=N_DAYS, seed=seed, distribution_mode=tier
    )
    if seed not in _READBACK[tier]:
        _READBACK[tier][seed] = check_env_effective_config(
            base,
            seed=seed,
            n_symbols=N_SYMBOLS,
            n_days=N_DAYS,
            distribution_mode=tier,
        )
    env = DrawdownStopper(base, max_drawdown=MAX_DRAWDOWN)
    obs, _ = env.reset(seed=seed)
    returns: list[float] = []
    events: list[dict] = []
    for _ in range(MAX_STEPS):
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        returns.append(float(reward))
        events.append(
            {
                "event": "target_weights",
                "weights": [float(x) for x in np.asarray(action).reshape(-1)],
            }
        )
        if info.get("stopped_out"):
            events.append({"event": "stopped_out"})
        if terminated or truncated:
            break
    return returns, events


def main() -> None:
    for readbacks in _READBACK.values():
        readbacks.clear()
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    episodes: list[dict] = []
    modes_by_tier: dict[str, list[FailureMode]] = {t: [] for t in TIERS}
    for tier in TIERS:
        for name, factory in FIELD:
            for seed in SEEDS:
                returns, events = _episode(tier, seed, factory())
                mandate = sample_mandate(seed, n_symbols=N_SYMBOLS)
                mode = classify_episode_failure(returns, events, mandate)
                modes_by_tier[tier].append(mode)
                episodes.append(
                    {
                        "tier": tier,
                        "policy": name,
                        "seed": seed,
                        "n_bars": len(returns),
                        "mandate_style": mandate.style,
                        "mode": mode.value,
                    }
                )

    rollups = {
        tier: rollup_failure_modes(modes_by_tier[tier]).__dict__
        for tier in TIERS
    }
    overall = rollup_failure_modes(
        [m for tier in TIERS for m in modes_by_tier[tier]]
    ).__dict__

    out = {
        "finding": "F7",
        "config": {
            "n_symbols": N_SYMBOLS,
            "n_days": N_DAYS,
            "seeds": SEEDS,
            "max_steps": MAX_STEPS,
            "max_drawdown": MAX_DRAWDOWN,
            "tiers": list(TIERS),
            "policies": [name for name, _ in FIELD],
        },
        "effective_config": {
            tier: merge_effective_configs(_READBACK[tier]) for tier in TIERS
        },
        "episodes": episodes,
        "rollup_by_tier": rollups,
        "rollup_overall": overall,
    }
    (EVIDENCE / "f7-failures.json").write_text(json.dumps(out, indent=2))

    # Figure: mode counts per tier.
    mode_names = [m.value for m in FailureMode]
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.8 / len(TIERS)
    for j, tier in enumerate(TIERS):
        counts = rollups[tier]["counts"]
        xs = [i + (j - (len(TIERS) - 1) / 2) * width for i in range(len(mode_names))]
        ax.bar(xs, [counts[m] for m in mode_names], width=width, label=tier)
    ax.set_xticks(range(len(mode_names)))
    ax.set_xticklabels(mode_names, rotation=30, ha="right")
    ax.set_ylabel("episodes")
    ax.legend(title="tier", frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "f7-failures.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
