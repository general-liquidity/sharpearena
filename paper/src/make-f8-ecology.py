#!/usr/bin/env python3
"""F8: ecology outcomes under shocks.

``run_ecology`` runs the replicator over the baseline species (behavioral
counterparties included) with shared-book payoffs from ``market_payoffs``, once
under a steady control schedule and once under ``regime_shocks`` rotating
calm/hard/extreme, with the ``mutating_innovator`` breeding variants. Writes
both reports as JSON plus a stacked-share figure per run.
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
SEED = 0
N_SYMBOLS = 4
N_DAYS = 120
MAX_STEPS = 256
INNOVATE_EVERY = 4
SHOCK_PERIOD = 4


def _run(shocks) -> dict:
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
        seed=SEED,
    )


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

    control = _run(steady_shocks(GENERATIONS))
    shocked = _run(regime_shocks(GENERATIONS, period=SHOCK_PERIOD))

    out = {
        "finding": "F8",
        "config": {
            "generations": GENERATIONS,
            "field_size": FIELD_SIZE,
            "seed": SEED,
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
    }
    (EVIDENCE / "f8-ecology.json").write_text(json.dumps(out, indent=2))

    _plot(control, "steady control", FIGURES / "f8-ecology-control.pdf")
    _plot(shocked, "regime shocks (calm/hard/extreme)", FIGURES / "f8-ecology-shocked.pdf")


if __name__ == "__main__":
    main()
