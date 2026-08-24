#!/usr/bin/env python3
"""F5: the manipulation payoff boundary and the size response.

``impact_boundary_sweep`` sweeps one impact axis (permanent impact, temporary
impact, follower gain) and reports where the pump-and-unwind round trip stops
paying against its zero-impact paired reference; ``size_response`` sweeps the
push weight and reports whether the payoff is bounded. The sweep APIs return
seed means only, so the script additionally collects per-seed impact PnL at
every grid point via the public ``run_manipulation_probe`` and serializes
per-seed vectors plus t-based 95% CIs (asserting the per-seed means reproduce
the API sweeps). Writes JSON plus two figures. These probes diagnose the
simulator's impact specification, not agents.
"""
from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sharpearena import (
    ManipulationParams,
    impact_boundary_sweep,
    run_manipulation_probe,
    size_response,
)

PAPER = Path(__file__).resolve().parents[1]
EVIDENCE = PAPER / "evidence"
FIGURES = PAPER / "figures"

SEEDS = tuple(range(8))
# Axis grids: the impact coefficients use the module defaults; follower_gain
# needs its own grid because its natural scale is tens, not tenths.
AXES: dict[str, tuple[float, ...] | None] = {
    "kyle_lambda": None,
    "eta": None,
    "follower_gain": (0.0, 5.0, 15.0, 30.0, 60.0, 120.0),
}


def main() -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    params = ManipulationParams()
    boundaries = {
        axis: impact_boundary_sweep(
            params=params, axis=axis, values=values, seeds=SEEDS
        ).to_dict()
        for axis, values in AXES.items()
    }
    size = size_response(params=params, seeds=SEEDS).to_dict()

    # Per-seed impact PnL at every grid point (t-based 95% CI, df = 7).
    t_crit = 2.365
    def _per_seed(p: ManipulationParams) -> list[float]:
        return [
            run_manipulation_probe(params=p, seed=s).impact_pnl for s in SEEDS
        ]

    def _stats(values: list[float]) -> dict:
        n = len(values)
        mean = sum(values) / n
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
        half = t_crit * std / math.sqrt(n)
        return {"mean": mean, "std": std, "ci95_lo": mean - half, "ci95_hi": mean + half}

    dispersion: dict[str, dict] = {}
    for axis, rep in boundaries.items():
        rows = [_per_seed(replace(params, **{axis: v})) for v in rep["values"]]
        for row, mean in zip(rows, rep["impact_pnl"]):
            assert abs(sum(row) / len(row) - mean) < 1e-12
        dispersion[axis] = {
            "values": rep["values"],
            "per_seed_impact_pnl": rows,
            "stats": [_stats(r) for r in rows],
        }
    size_rows = [
        _per_seed(replace(params, push_weight=w)) for w in size["push_weights"]
    ]
    for row, mean in zip(size_rows, size["impact_pnl"]):
        assert abs(sum(row) / len(row) - mean) < 1e-12
    dispersion["push_weight"] = {
        "values": size["push_weights"],
        "per_seed_impact_pnl": size_rows,
        "stats": [_stats(r) for r in size_rows],
    }

    out = {
        "finding": "F5",
        "config": {
            "seeds": list(SEEDS),
            "base_params": {
                "n_symbols": params.n_symbols,
                "n_days": params.n_days,
                "n_followers": params.n_followers,
                "kyle_lambda": params.kyle_lambda,
                "eta": params.eta,
                "push_weight": params.push_weight,
                "follower_gain": params.follower_gain,
            },
        },
        "boundaries": boundaries,
        "size_response": size,
        "dispersion": dispersion,
        "ci_convention": "t-based 95% over per-seed impact PnL, df=7",
    }
    (EVIDENCE / "f5-manipulation.json").write_text(json.dumps(out, indent=2))

    # Figure 1: impact P&L along each swept axis, boundary marked where found.
    fig, axes = plt.subplots(1, len(boundaries), figsize=(10, 3.2), sharey=True)
    for ax, (axis, rep) in zip(axes, boundaries.items()):
        ax.plot(rep["values"], rep["impact_pnl"], marker="o")
        ax.axhline(0.0, color="black", linewidth=0.8)
        if rep["boundary"] is not None:
            ax.axvline(rep["boundary"], linestyle="--", color="red", linewidth=0.8)
        ax.set_xlabel(axis)
    axes[0].set_ylabel("impact P&L")
    fig.tight_layout()
    fig.savefig(FIGURES / "f5-boundaries.pdf")
    plt.close(fig)

    # Figure 2: payoff vs push size, with the peak marked.
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(size["push_weights"], size["impact_pnl"], marker="o")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axvline(size["peak_push_weight"], linestyle="--", color="red", linewidth=0.8)
    verdict = "bounded" if size["bounded"] else "UNBOUNDED"
    ax.set_title(f"size response ({verdict})", fontsize=10)
    ax.set_xlabel("push weight")
    ax.set_ylabel("impact P&L")
    fig.tight_layout()
    fig.savefig(FIGURES / "f5-size-response.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
