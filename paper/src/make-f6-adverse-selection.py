#!/usr/bin/env python3
"""F6: adverse-selection markout decomposition.

``compare_informed_vs_uninformed`` runs paired episodes (identical flow, sided
on alpha vs a coin, price path held fixed) and reports mean maker markout per
filled unit per horizon; one seeded ``run_adverse_selection`` episode
additionally decomposes markout by maker. The aggregate API pools quantities
over episodes, so the script additionally reruns the public
``run_adverse_selection`` per episode and leg to serialize per-episode
markout-per-unit vectors and t-based 95% CIs on the per-episode paired gap
(the per-episode ratio is episode markout over episode filled quantity; the
API's pooled aggregate is the quantity-weighted counterpart). Writes JSON plus
a figure.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sharpearena import (
    AdverseSelectionParams,
    compare_informed_vs_uninformed,
    run_adverse_selection,
)

PAPER = Path(__file__).resolve().parents[1]
EVIDENCE = PAPER / "evidence"
FIGURES = PAPER / "figures"

N_EPISODES = 24
SEED_BASE = 0
DETAIL_SEED = 0


def main() -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    params = AdverseSelectionParams()
    comparison = compare_informed_vs_uninformed(
        params=params, n_episodes=N_EPISODES, seed_base=SEED_BASE
    )

    detail = run_adverse_selection(params=params, seed=DETAIL_SEED)
    per_maker = [asdict(m) for m in detail.makers]

    # Per-episode markout per unit, per leg, per horizon (t-based 95% CI, df = 23).
    horizons = list(params.markout_horizons)
    t_crit = 2.069

    def _episode_per_unit(informed: bool, seed: int) -> dict[int, float]:
        report = run_adverse_selection(
            params=replace(params, informed=informed), seed=seed
        )
        qty = sum(m.filled_qty for m in report.makers)
        return {
            h: sum(m.markout[h] for m in report.makers) / qty for h in horizons
        }

    per_episode = {"informed": [], "uninformed": [], "gap": []}
    for i in range(N_EPISODES):
        inf = _episode_per_unit(True, SEED_BASE + i)
        uni = _episode_per_unit(False, SEED_BASE + i)
        per_episode["informed"].append({str(h): inf[h] for h in horizons})
        per_episode["uninformed"].append({str(h): uni[h] for h in horizons})
        per_episode["gap"].append({str(h): uni[h] - inf[h] for h in horizons})

    def _stats(values: list[float]) -> dict:
        n = len(values)
        mean = sum(values) / n
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
        half = t_crit * std / math.sqrt(n)
        return {
            "mean": mean,
            "std": std,
            "ci95_lo": mean - half,
            "ci95_hi": mean + half,
            "excludes_zero": mean - half > 0.0 or mean + half < 0.0,
        }

    gap_stats = {
        str(h): _stats([row[str(h)] for row in per_episode["gap"]]) for h in horizons
    }

    out = {
        "finding": "F6",
        "config": {"n_episodes": N_EPISODES, "seed_base": SEED_BASE, "detail_seed": DETAIL_SEED},
        "comparison": {
            "n_episodes": comparison["n_episodes"],
            "horizons": list(comparison["horizons"]),
            "informed_markout_per_unit": {
                str(h): v for h, v in comparison["informed_markout_per_unit"].items()
            },
            "uninformed_markout_per_unit": {
                str(h): v for h, v in comparison["uninformed_markout_per_unit"].items()
            },
            "gap_per_unit": {str(h): v for h, v in comparison["gap_per_unit"].items()},
            "informed_is_worse": {
                str(h): v for h, v in comparison["informed_is_worse"].items()
            },
        },
        "detail_episode_makers": per_maker,
        "per_episode_markout_per_unit": per_episode,
        "gap_stats": gap_stats,
        "ci_convention": (
            "t-based 95% over per-episode paired gaps (episode markout over "
            "episode filled quantity), df=23"
        ),
    }
    (EVIDENCE / "f6-adverse-selection.json").write_text(json.dumps(out, indent=2, default=str))

    # Figure: markout per filled unit by horizon, informed vs uninformed legs.
    horizons = list(comparison["horizons"])
    informed = [comparison["informed_markout_per_unit"][h] for h in horizons]
    uninformed = [comparison["uninformed_markout_per_unit"][h] for h in horizons]
    fig, ax = plt.subplots(figsize=(6, 4))
    width = 0.35
    xs = range(len(horizons))
    ax.bar([x - width / 2 for x in xs], informed, width=width, label="informed flow")
    ax.bar([x + width / 2 for x in xs], uninformed, width=width, label="uninformed flow")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([str(h) for h in horizons])
    ax.set_xlabel("markout horizon (steps)")
    ax.set_ylabel("maker markout per filled unit")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "f6-markouts.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
