#!/usr/bin/env python3
"""Re-render every paper figure from the committed evidence JSON.

Reads paper/evidence/f*.json and reproduces the figures the make-f* scripts
emit, without re-running any experiment. Every bar and point is a reduction
over the committed records; no number is typed into this script. Evidence
files that do not exist yet are skipped with a notice.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PAPER = Path(__file__).resolve().parents[1]
EVIDENCE = PAPER / "evidence"
FIGURES = PAPER / "figures"

TIERS = ("calm", "hard", "extreme")


def _load(name: str) -> dict | None:
    path = EVIDENCE / name
    if not path.exists():
        print(f"skip: {path} (evidence not generated yet)")
        return None
    return json.loads(path.read_text())


def f1() -> None:
    data = _load("f1-baselines.json")
    if data is None:
        return
    tiers = data["tiers"]
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


def f2() -> None:
    data = _load("f2-regret.json")
    if data is None:
        return
    fixed = data["fixed_spread_regret"]
    xs = [float(k) for k in fixed]
    ys = [fixed[k] for k in fixed]
    order = np.argsort(xs)
    xs = [xs[i] for i in order]
    ys = [ys[i] for i in order]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys, marker="o", label="fixed-spread quoter")
    ax.axhline(
        data["optimal_regret"], linestyle="--", color="black", linewidth=0.8,
        label="A-S closed-form optimum",
    )
    ax.set_xscale("log")
    ax.set_xlabel("fixed half-spread (price units)")
    ax.set_ylabel("mean regret vs optimum")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "f2-regret.pdf")
    plt.close(fig)


def f3() -> None:
    data = _load("f3-generalization.json")
    if data is None:
        return
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


def f4() -> None:
    data = _load("f4-realism.json")
    if data is None:
        return
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


def f5() -> None:
    data = _load("f5-manipulation.json")
    if data is None:
        return
    boundaries = data["boundaries"]
    fig, axes = plt.subplots(1, len(boundaries), figsize=(10, 3.2), sharey=True)
    for ax, (axis, rep) in zip(np.atleast_1d(axes), boundaries.items()):
        ax.plot(rep["values"], rep["impact_pnl"], marker="o")
        ax.axhline(0.0, color="black", linewidth=0.8)
        if rep["boundary"] is not None:
            ax.axvline(rep["boundary"], linestyle="--", color="red", linewidth=0.8)
        ax.set_xlabel(axis)
    np.atleast_1d(axes)[0].set_ylabel("impact P&L")
    fig.tight_layout()
    fig.savefig(FIGURES / "f5-boundaries.pdf")
    plt.close(fig)

    size = data["size_response"]
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


def f6() -> None:
    data = _load("f6-adverse-selection.json")
    if data is None:
        return
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


def f7() -> None:
    data = _load("f7-failures.json")
    if data is None:
        return
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


def f8() -> None:
    data = _load("f8-ecology.json")
    if data is None:
        return
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


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for fn in (f1, f2, f3, f4, f5, f6, f7, f8):
        fn()


if __name__ == "__main__":
    main()
