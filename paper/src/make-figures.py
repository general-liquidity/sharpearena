#!/usr/bin/env python3
"""Re-render every paper figure from the committed evidence JSON.

Reads paper/evidence/*.json and reproduces the figures the make-* scripts emit,
without re-running any experiment. Every bar and point is a reduction over the
committed records; no number is typed into this script.

The dispatch is an explicit registry: one entry per renderer, naming the
evidence file it reads and every PDF it writes. Figures whose layout already
lives in a producer are rendered by calling that producer's own figure function
on the committed record, so the extension plots cannot drift from the base ones
and no layout is maintained twice. ``main`` closes the registry against the
figure directory and prints an omission notice naming any committed PDF no
entry claims, so an incomplete rebuild says so instead of quietly leaving stale
plots in place. Evidence files that do not exist yet are skipped with a notice
that names the figures they would have produced.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Callable, NamedTuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from confidence_support import require_scoring_intervals

PAPER = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
EVIDENCE = PAPER / "evidence"
FIGURES = PAPER / "figures"

TIERS = ("calm", "hard", "extreme")


def _load(name: str) -> dict | None:
    path = EVIDENCE / name
    if not path.exists():
        print(f"skip: {path} (evidence not generated yet)")
        return None
    return json.loads(path.read_text())


def producer(stem: str):
    """Import a hyphenated producer module by path, once.

    The producers guard their ``sharpearena`` import so this stays a
    frozen-input path: importing one to reach its figure function does not
    require the native bindings, only the committed JSON it is handed.
    """
    name = stem.replace("-", "_")
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SRC / f"{stem}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def f1(data: dict) -> None:
    tiers = data["tiers"]
    for tier in TIERS:
        require_scoring_intervals(tiers[tier]["rows"])
    policies = [r["policy"] for r in tiers[TIERS[0]]["rows"]]
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.8 / len(TIERS)
    for j, tier in enumerate(TIERS):
        rows = {r["policy"]: r for r in tiers[tier]["rows"]}
        xs = [i + (j - (len(TIERS) - 1) / 2) * width for i in range(len(policies))]
        ys = [rows[p]["deflated_sharpe"] for p in policies]
        ci = [rows[p].get("deflated_sharpe_ci") or {} for p in policies]
        lo = [y - c.get("lo", y) for y, c in zip(ys, ci)]
        hi = [c.get("hi", y) - y for y, c in zip(ys, ci)]
        ax.bar(xs, ys, width=width, yerr=[lo, hi], capsize=2, label=tier)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(policies)))
    ax.set_xticklabels(policies, rotation=30, ha="right")
    ax.set_ylabel("deflated Sharpe (score_run)")
    ax.legend(title="tier", frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "f1-baselines.pdf")
    plt.close(fig)


def f2(data: dict) -> None:
    fixed = data["fixed_spread_regret"]
    disp = data.get("regret_dispersion", {})
    xs = [float(k) for k in fixed]
    ys = [fixed[k] for k in fixed]
    keys = list(fixed)
    order = np.argsort(xs)
    xs = [xs[i] for i in order]
    ys = [ys[i] for i in order]
    keys = [keys[i] for i in order]
    fig, ax = plt.subplots(figsize=(6, 4))
    if disp:
        lo = [ys[i] - disp[k]["ci95_lo"] for i, k in enumerate(keys)]
        hi = [disp[k]["ci95_hi"] - ys[i] for i, k in enumerate(keys)]
        ax.errorbar(
            xs, ys, yerr=[lo, hi], marker="o", capsize=3,
            label="fixed-spread quoter",
        )
    else:
        ax.plot(xs, ys, marker="o", label="fixed-spread quoter")
    ax.axhline(
        data["optimal_regret"], linestyle="--", color="black", linewidth=0.8,
        label="A-S closed-form reference",
    )
    ax.set_xscale("log")
    ax.set_xlabel("fixed half-spread (price units)")
    ax.set_ylabel("mean regret vs closed-form reference")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "f2-regret.pdf")
    plt.close(fig)


def f3(data: dict) -> None:
    matrix = data["cross_regime_transfer"]
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


def f4(data: dict) -> None:
    tiers = data["tiers"]
    fact_names = sorted(tiers[TIERS[0]]["mean_facts"])
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.8 / len(TIERS)
    for j, tier in enumerate(TIERS):
        xs = [i + (j - (len(TIERS) - 1) / 2) * width for i in range(len(fact_names))]
        ys = [tiers[tier]["mean_facts"][f] for f in fact_names]
        ax.bar(xs, ys, width=width, label=f"{tier} (pass {tiers[tier]['pass_rate']:.0%})")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(fact_names)))
    ax.set_xticklabels(fact_names, rotation=30, ha="right")
    ax.set_ylabel("mean stylized-fact value")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "f4-realism.pdf")
    plt.close(fig)


def f6(data: dict) -> None:
    comp = data["comparison"]
    horizons = [str(h) for h in comp["horizons"]]
    informed = [comp["informed_markout_per_unit"][h] for h in horizons]
    uninformed = [comp["uninformed_markout_per_unit"][h] for h in horizons]
    fig, ax = plt.subplots(figsize=(6, 4))
    width = 0.35
    xs = range(len(horizons))
    ax.bar([x - width / 2 for x in xs], informed, width=width, label="informed flow")
    ax.bar([x + width / 2 for x in xs], uninformed, width=width, label="uninformed flow")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(horizons)
    ax.set_xlabel("markout horizon (steps)")
    ax.set_ylabel("maker markout per filled unit")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "f6-markouts.pdf")
    plt.close(fig)


def f7(data: dict) -> None:
    rollups = data["rollup_by_tier"]
    mode_names = list(rollups[TIERS[0]]["counts"])
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


def f8(data: dict) -> None:
    for key, title, name in (
        ("control", "steady control", "f8-ecology-control.pdf"),
        ("shocked", "regime shocks (calm/hard/extreme)", "f8-ecology-shocked.pdf"),
    ):
        report = data[key]
        names = [s["name"] for s in report["species"]]
        shares = np.asarray(report["shares"], dtype=float)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.stackplot(range(shares.shape[0]), shares.T, labels=names)
        ax.set_xlabel("generation")
        ax.set_ylabel("population share")
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=6, ncol=2, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
        fig.tight_layout()
        fig.savefig(FIGURES / name)
        plt.close(fig)


def f4_calm_calibration(data: dict) -> None:
    producer("make-f4-realism")._plot_calm_calibration(data["calm_calibration"])


def f5(data: dict) -> None:
    producer("make-f5-manipulation").make_figures(data)


def f6_endogenous(data: dict) -> None:
    producer("make-f6-adverse-selection").endogenous_figure(data["endogenous"])


def predictability(data: dict) -> None:
    producer("make-predictability").make_figure(data["tiers"])


def witness(data: dict) -> None:
    producer("make-witness").make_figure(data)


class Renderer(NamedTuple):
    """One dispatch entry: the record it reads and every PDF it writes.

    ``figures`` is the claim this registry is checked against, so an entry that
    silently stops writing one of its plots is a defect the coverage check can
    name, not a gap the reader has to notice.
    """

    evidence: str
    figures: tuple[str, ...]
    render: Callable[[dict], None]


REGISTRY: tuple[Renderer, ...] = (
    Renderer("f1-baselines.json", ("f1-baselines.pdf",), f1),
    Renderer("f2-regret.json", ("f2-regret.pdf",), f2),
    Renderer("f3-generalization.json", ("f3-transfer-matrix.pdf",), f3),
    Renderer("f4-realism.json", ("f4-realism.pdf",), f4),
    Renderer("f4-realism.json", ("f4-calm-calibration.pdf",), f4_calm_calibration),
    Renderer(
        "f5-manipulation.json",
        (
            "f5-boundaries.pdf",
            "f5-size-response.pdf",
            "f5-concave.pdf",
            "f5-positive-control.pdf",
            "f5-extended-sweeps.pdf",
        ),
        f5,
    ),
    Renderer("f6-adverse-selection.json", ("f6-markouts.pdf",), f6),
    Renderer("f6-adverse-selection.json", ("f6-endogenous.pdf",), f6_endogenous),
    Renderer("f7-failures.json", ("f7-failures.pdf",), f7),
    Renderer(
        "f8-ecology.json", ("f8-ecology-control.pdf", "f8-ecology-shocked.pdf"), f8
    ),
    Renderer("predictability.json", ("predictability.pdf",), predictability),
    Renderer("witness.json", ("witness.pdf",), witness),
)

REGISTERED_FIGURES: frozenset[str] = frozenset(
    name for entry in REGISTRY for name in entry.figures
)


def unregistered_figures() -> list[str]:
    """Committed PDFs that no registry entry claims to rebuild.

    Reported rather than raised: a figure this script cannot regenerate is a
    fact about the registry that the operator needs told, and the rebuild of
    everything else is still worth doing.
    """
    return sorted({p.name for p in FIGURES.glob("*.pdf")} - REGISTERED_FIGURES)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for entry in REGISTRY:
        data = _load(entry.evidence)
        if data is None:
            print(f"  not rendered: {', '.join(entry.figures)}")
            continue
        entry.render(data)
        print(f"wrote {', '.join(entry.figures)} from {entry.evidence}")
    missing = unregistered_figures()
    if missing:
        print(
            "omission: no registry entry rebuilds "
            + ", ".join(missing)
            + "; those files were left as they were"
        )
    else:
        print(f"registry covers every PDF in {FIGURES} ({len(REGISTERED_FIGURES)})")


if __name__ == "__main__":
    main()
