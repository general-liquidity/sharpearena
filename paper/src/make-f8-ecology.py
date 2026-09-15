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
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from figure_style import OKABE_ITO, save_pdf

try:
    from sharpearena import (
        baseline_species,
        market_payoffs,
        mutating_innovator,
        population_table,
        regime_shocks,
        run_ecology,
        steady_shocks,
    )
except ImportError:  # --figures-only reads the committed JSON and needs no bindings
    baseline_species = market_payoffs = mutating_innovator = None
    population_table = regime_shocks = run_ecology = steady_shocks = None

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


# Band fills use the Okabe-Ito colours (black excluded, so hatching stays visible) and
# every band gets its own hatch, so bands stay separable in greyscale.
BAND_COLORS = OKABE_ITO[:-1]
BAND_HATCHES = ("///", "\\\\\\", "|||", "---", "++", "xx", "...", "oo", "**", "OO", "/.", "\\|")
LABEL_PT = 7.0


def _band_colors(report: dict, shares: np.ndarray) -> list[str]:
    """Founders cycle the palette by position, so a founder keeps its colour in both
    figures; a bred variant takes the first colour no band it ever touches already has."""
    colors: list[str] = []
    for i, species in enumerate(report["species"]):
        if species["parent"] is None:
            colors.append(BAND_COLORS[i % len(BAND_COLORS)])
            continue
        taken = set()
        for row in shares:
            stack = [j for j in range(len(row)) if row[j] > 0.0]
            if i in stack:
                k = stack.index(i)
                taken.update(colors[j] for j in stack[max(k - 1, 0):k + 2] if j < i)
        colors.append(next(c for c in BAND_COLORS if c not in taken))
    return colors


def _spread(targets: list[float], gap: float, top: float) -> list[float]:
    """Label heights in target order, at least ``gap`` apart and no higher than ``top``."""
    ys = list(targets)
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + gap)
    if ys and ys[-1] > top:
        ys[-1] = top
        for i in range(len(ys) - 2, -1, -1):
            ys[i] = min(ys[i], ys[i + 1] - gap)
    return ys


def _plot(report: dict, title: str, path: Path) -> None:
    # Every band is named on the plot, not in a legend: a band present at the first
    # generation is labelled in the left margin (outside the share tick labels), one
    # present at the last generation in the right margin (a band spanning the run gets
    # both), and a band born and extinct inside the run above the axes at its widest
    # generation. Leader lines join each label to the middle of its band. Axes geometry
    # is fixed in inches, so label spacing is exact.
    names = [s["name"] for s in report["species"]]
    shares = np.asarray(report["shares"], dtype=float)  # (G+1, n_species)
    last = shares.shape[0] - 1
    gens = np.arange(shares.shape[0])
    centres = np.cumsum(shares, axis=1) - shares / 2.0

    width, height = 5.5, 3.5
    left, right, bottom, top = 1.55, 1.15, 0.45, 0.6
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes(
        (left / width, bottom / height, 1 - (left + right) / width, 1 - (bottom + top) / height)
    )
    axis_width = width - left - right
    axis_height = height - bottom - top
    polys = ax.stackplot(
        gens,
        shares.T,
        colors=[to_rgba(c, 0.55) for c in _band_colors(report, shares)],
        edgecolor="black",
        linewidth=0.4,
    )
    for i, poly in enumerate(polys):
        poly.set_hatch(BAND_HATCHES[i % len(BAND_HATCHES)])

    def wrapped(name: str) -> str:
        return name.replace("~", "\n~")

    ax.set_xlim(0, last)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(labelsize=7)
    ax.set_xlabel("generation", fontsize=8)
    ax.set_title(title, fontsize=9, pad=20)

    line_gap = 1.25 * LABEL_PT / 72.0 / axis_height
    leader = {"arrowstyle": "-", "color": "0.3", "linewidth": 0.5, "shrinkA": 1, "shrinkB": 0}
    # Left labels end clear of the share tick labels; right labels start past a short lead.
    sides = (("left", 0, -0.36 / axis_width, "right"), ("right", last, 1 + 0.12 / axis_width, "left"))
    left_texts = []
    for side, gen, x_text, ha in sides:
        present = [i for i in range(len(names)) if shares[gen, i] > 0.0]
        labels = [wrapped(names[i]) if side == "right" else names[i] for i in present]
        lines = max((lab.count("\n") + 1 for lab in labels), default=1)
        ys = _spread([centres[gen, i] for i in present], lines * line_gap, 1.0)
        for i, label, y in zip(present, labels, ys):
            text = ax.annotate(
                label, xy=(gen, centres[gen, i]), xytext=(x_text, y),
                textcoords=ax.get_yaxis_transform(), ha=ha, va="center",
                fontsize=LABEL_PT, arrowprops=leader, annotation_clip=False,
            )
            if side == "left":
                left_texts.append(text)
    for i in range(len(names)):
        if shares[0, i] == 0.0 and shares[last, i] == 0.0:
            gen = int(np.argmax(shares[:, i]))
            ax.annotate(
                names[i], xy=(gen, centres[gen, i]), xytext=(gen, 1.04), ha="center",
                va="bottom", fontsize=LABEL_PT, arrowprops=leader, annotation_clip=False,
            )
    # The share axis label sits outside the widest left label.
    renderer = fig.canvas.get_renderer()
    x0 = min(t.get_window_extent(renderer).x0 for t in left_texts)
    x_axes = ax.transAxes.inverted().transform((x0, 0.0))[0]
    ax.set_ylabel("population share", fontsize=8)
    ax.yaxis.set_label_coords(x_axes - 0.06 / axis_width, 0.5)
    ax.yaxis.label.set_horizontalalignment("center")
    ax.yaxis.label.set_verticalalignment("bottom")
    save_pdf(fig, path)
    plt.close(fig)


def make_figures(out: dict) -> None:
    """Render both F8 figures from the evidence dict (the committed JSON shape)."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    _plot(out["control"], "steady control", FIGURES / "f8-ecology-control.pdf")
    _plot(out["shocked"], "regime shocks (calm/hard/extreme)", FIGURES / "f8-ecology-shocked.pdf")


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

    make_figures(out)


if __name__ == "__main__":
    if "--figures-only" in sys.argv:
        make_figures(json.loads((EVIDENCE / "f8-ecology.json").read_text()))
        print(f"wrote {FIGURES / 'f8-ecology-control.pdf'}, {FIGURES / 'f8-ecology-shocked.pdf'}")
    else:
        main()
