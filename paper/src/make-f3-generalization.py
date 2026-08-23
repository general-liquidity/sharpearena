#!/usr/bin/env python3
"""F3: generalization gap and the cross-regime transfer matrix.

``generalization_gap`` scores the default reference policy on disjoint train and
held-out seed bands within each tier; ``cross_regime_transfer`` holds the seed
band fixed and varies the regime, producing the full tier-by-tier matrix (the
diagonal is 0 by construction). Writes JSON plus a transfer-matrix heatmap.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sharpearena import SharpeArenaEnv, cross_regime_transfer, generalization_gap

PAPER = Path(__file__).resolve().parents[1]
EVIDENCE = PAPER / "evidence"
FIGURES = PAPER / "figures"

TIERS = ("calm", "hard", "extreme")
N_SYMBOLS = 4
N_DAYS = 120
N_TRAIN = 16
N_TEST = 16
SEED_GAP = 10_000
TRANSFER_SEEDS = list(range(16))


def _make_env(seed: int, mode: str) -> SharpeArenaEnv:
    return SharpeArenaEnv(
        n_symbols=N_SYMBOLS, n_days=N_DAYS, seed=seed, distribution_mode=mode
    )


def main() -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    gaps = {
        tier: generalization_gap(
            lambda seed, m=tier: _make_env(seed, m),
            n_train=N_TRAIN,
            n_test=N_TEST,
            gap=SEED_GAP,
        )
        for tier in TIERS
    }

    matrix = {}
    for a in TIERS:
        for b in TIERS:
            matrix[f"{a}->{b}"] = cross_regime_transfer(
                _make_env, a, b, TRANSFER_SEEDS
            )

    out = {
        "finding": "F3",
        "config": {
            "n_symbols": N_SYMBOLS,
            "n_days": N_DAYS,
            "n_train": N_TRAIN,
            "n_test": N_TEST,
            "seed_gap": SEED_GAP,
            "transfer_seeds": TRANSFER_SEEDS,
            "tiers": list(TIERS),
        },
        "generalization_gap": gaps,
        "cross_regime_transfer": matrix,
    }
    (EVIDENCE / "f3-generalization.json").write_text(json.dumps(out, indent=2))

    # Figure: 3x3 heatmap of transfer_gap_deflated_sharpe.
    grid = [
        [matrix[f"{a}->{b}"]["transfer_gap_deflated_sharpe"] for b in TIERS]
        for a in TIERS
    ]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(grid, cmap="coolwarm")
    ax.set_xticks(range(len(TIERS)), labels=TIERS)
    ax.set_yticks(range(len(TIERS)), labels=TIERS)
    ax.set_xlabel("scored on (zero-shot)")
    ax.set_ylabel("selected on")
    for i in range(len(TIERS)):
        for j in range(len(TIERS)):
            ax.text(j, i, f"{grid[i][j]:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="transfer gap (deflated Sharpe)")
    fig.tight_layout()
    fig.savefig(FIGURES / "f3-transfer-matrix.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
