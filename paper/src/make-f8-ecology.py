#!/usr/bin/env python3
"""F8: ecology outcomes under shocks, replicated over seeds.

``run_ecology`` runs the replicator over the baseline species (behavioral
counterparties included) with shared-book payoffs from ``market_payoffs``, once
under a steady control schedule and once under ``regime_shocks`` rotating
calm/hard/extreme, with the ``mutating_innovator`` breeding variants. The run
is replicated over ``SEEDS`` replicator seeds per schedule; the committed JSON
carries the full seed-0 reports (the figures' source) plus a per-seed outcome
summary and the cross-seed winner distribution. Writes JSON plus a
stacked-share figure per seed-0 run.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sharpearena import (
    baseline_species,
    market_payoffs,
    mutating_innovator,
    population_table,
    regime_shocks,
    run_ecology,
    steady_shocks,
)

PAPER = Path(__file__).resolve().parents[1]
EVIDENCE = PAPER / "evidence"
FIGURES = PAPER / "figures"

GENERATIONS = 12
FIELD_SIZE = 8
SEEDS = list(range(8))  # seed 0 is the detailed (figure) run
N_SYMBOLS = 4
N_DAYS = 120
MAX_STEPS = 256
INNOVATE_EVERY = 4
SHOCK_PERIOD = 4


def _run(shocks, seed: int) -> dict:
    return run_ecology(
        baseline_species(include_behavioral=True),
        market_payoffs(
            n_symbols=N_SYMBOLS, n_days=N_DAYS, max_steps=MAX_STEPS
        ),
        generations=GENERATIONS,
        field_size=FIELD_SIZE,
        innovate_every=INNOVATE_EVERY,
        innovator=mutating_innovator(),
        shocks=shocks,
        seed=seed,
    )


def _root(name: str, species: list[dict]) -> str:
    """Follow a bred variant's parent chain back to its founding species."""
    parents = {s["name"]: s["parent"] for s in species}
    while parents.get(name):
        name = parents[name]
    return name


def _summarize(report: dict) -> dict:
    """One seed's outcome: the final dominant species and the outcome counts."""
    final = report["final_shares"]
    winner = max(final, key=final.get)
    counts: dict[str, int] = {}
    for entry in report["outcomes"].values():
        counts[entry["outcome"]] = counts.get(entry["outcome"], 0) + 1
    return {
        "winner": winner,
        "winner_root": _root(winner, report["species"]),
        "winner_final_share": final[winner],
        "outcome_counts": counts,
    }


def _plot(report: dict, title: str, path: Path) -> None:
    names = [s["name"] for s in report["species"]]
    shares = np.asarray(report["shares"], dtype=float)  # (G+1, n_species)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.stackplot(range(shares.shape[0]), shares.T, labels=names)
    ax.set_xlabel("generation")
    ax.set_ylabel("population share")
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=6, ncol=2, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    control_runs = {s: _run(steady_shocks(GENERATIONS), s) for s in SEEDS}
    shocked_runs = {
        s: _run(regime_shocks(GENERATIONS, period=SHOCK_PERIOD), s) for s in SEEDS
    }
    control = control_runs[SEEDS[0]]
    shocked = shocked_runs[SEEDS[0]]

    per_seed = {
        str(s): {
            "control": _summarize(control_runs[s]),
            "shocked": _summarize(shocked_runs[s]),
        }
        for s in SEEDS
    }
    control_winners: dict[str, int] = {}
    shocked_winners: dict[str, int] = {}
    replaced = 0
    for s in SEEDS:
        cw = per_seed[str(s)]["control"]["winner_root"]
        sw = per_seed[str(s)]["shocked"]["winner_root"]
        control_winners[cw] = control_winners.get(cw, 0) + 1
        shocked_winners[sw] = shocked_winners.get(sw, 0) + 1
        if sw != cw:
            replaced += 1

    out = {
        "finding": "F8",
        "config": {
            "generations": GENERATIONS,
            "field_size": FIELD_SIZE,
            "seeds": SEEDS,
            "detail_seed": SEEDS[0],
            "n_symbols": N_SYMBOLS,
            "n_days": N_DAYS,
            "max_steps": MAX_STEPS,
            "innovate_every": INNOVATE_EVERY,
            "shock_period": SHOCK_PERIOD,
        },
        "control": control,
        "shocked": shocked,
        "control_table": population_table(control),
        "shocked_table": population_table(shocked),
        "multi_seed": {
            "per_seed": per_seed,
            "control_winner_distribution": control_winners,
            "shocked_winner_distribution": shocked_winners,
            "winner_replaced_count": replaced,
            "n_seeds": len(SEEDS),
        },
    }
    (EVIDENCE / "f8-ecology.json").write_text(json.dumps(out, indent=2))

    _plot(control, "steady control", FIGURES / "f8-ecology-control.pdf")
    _plot(shocked, "regime shocks (calm/hard/extreme)", FIGURES / "f8-ecology-shocked.pdf")


if __name__ == "__main__":
    main()
