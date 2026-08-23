#!/usr/bin/env python3
"""F6: adverse-selection markout decomposition.

``compare_informed_vs_uninformed`` runs paired episodes (identical flow, sided
on alpha vs a coin, price path held fixed) and reports mean maker markout per
filled unit per horizon; one seeded ``run_adverse_selection`` episode
additionally decomposes markout by maker. Writes JSON plus a figure.
"""
from __future__ import annotations

import json
from dataclasses import asdict
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
