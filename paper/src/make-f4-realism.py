#!/usr/bin/env python3
"""F4: stylized-facts realism certification of the scenario generator.

Rolls a flat (zero-weight) policy through seeded episodes per tier, collecting
the point-in-time ``closes`` vector each bar into a (T, n_symbols) price panel,
then grades each panel with ``sharpearena.certify_realism`` against the
directional Cont-stylized-facts bounds. Writes per-seed reports, per-tier
aggregates, and a grouped-bar figure of the mean fact values.

Also certifies the same tiers with the generator's opt-in volatility-clustering
driver enabled (``vol_clustering = VOL_CLUSTERING``) under ``clustered_tiers``,
so the JSON carries the before/after evidence for the clustering fact. The
default (unclustered) tiers remain the canonical configuration.

``calm_calibration`` is the Calm-tier calibration sweep over the two existing
opt-in knob families (``vol_clustering`` and the ``jump_burst_*`` triple). Every
grid cell is certified on the diagnostic seeds and on a disjoint confirmation
band; the selection rule is fixed before the sweep runs (see ``CALM_RULE``) and
the chosen cell is then re-certified on two further disjoint bands that played
no part in selection. The sweep never touches the canonical Calm tape.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sharpearena import SharpeArenaEnv, certify_realism

PAPER = Path(__file__).resolve().parents[1]
EVIDENCE = PAPER / "evidence"
FIGURES = PAPER / "figures"

TIERS = ("calm", "hard", "extreme")
SEEDS = list(range(8))
N_SYMBOLS = 4
N_DAYS = 120
MAX_STEPS = 512
# Opt-in volatility-clustering strength for the clustered variants (matches the
# committed strength in tests/test_vol_clustering.py; leaderboard stays at 0.0).
VOL_CLUSTERING = 0.5

# --- Calm calibration sweep (existing knobs only) --------------------------------
# Grid over the two opt-in families. vol_clustering = 0.0 rows are the jump-only
# control; the (0.0, 0.0, 0.0, 0.0) cell is the canonical Calm tape itself.
CALM_VC_GRID = (0.0, 0.3, 0.5)
CALM_JUMP_PROB_GRID = (0.01, 0.02, 0.03, 0.05)
CALM_JUMP_PERSIST_GRID = (0.0, 0.5)
CALM_JUMP_SIZE_GRID = (0.005, 0.010, 0.015, 0.020)
# Seed bands. The diagnostic band is F4's; the others are disjoint from it and
# from each other. Confirmation participates in selection; final and wide do not.
CALM_CONFIRM_SEEDS = list(range(100, 108))
CALM_FINAL_SEEDS = list(range(1000, 1008))
CALM_WIDE_SEEDS = list(range(2000, 2032))
# Pre-declared selection rule (fixed before the sweep is read):
#   1. three-check conjunction passes on >= MIN_PASS of the diagnostic seeds,
#   2. mean realized-volatility ratio to default Calm <= VOL_RATIO_MAX,
#   3. conjunction passes on >= MIN_PASS of the confirmation seeds,
#   4. among survivors, smallest mean normalized tape perturbation
#      (RMS of the per-bar simple-return difference from the default Calm tape,
#      divided by the default tape's realized volatility).
CALM_RULE = {
    "min_pass_of_8": 7,
    "vol_ratio_max": 1.25,
    "objective": "min normalized_rms_perturbation",
}


def _collect_panel(
    tier: str,
    seed: int,
    vol_clustering: float = 0.0,
    jump_burst_probability: float = 0.0,
    jump_burst_persistence: float = 0.0,
    jump_burst_size: float = 0.0,
) -> np.ndarray:
    env = SharpeArenaEnv(
        n_symbols=N_SYMBOLS,
        n_days=N_DAYS,
        seed=seed,
        distribution_mode=tier,
        vol_clustering=vol_clustering,
        jump_burst_probability=jump_burst_probability,
        jump_burst_persistence=jump_burst_persistence,
        jump_burst_size=jump_burst_size,
    )
    obs, _ = env.reset(seed=seed)
    closes = [np.asarray(obs["closes"], dtype=np.float64).reshape(-1)]
    flat = np.zeros(N_SYMBOLS, dtype=np.float32)
    for _ in range(MAX_STEPS):
        obs, _reward, terminated, truncated, _info = env.step(flat)
        closes.append(np.asarray(obs["closes"], dtype=np.float64).reshape(-1))
        if terminated or truncated:
            break
    return np.stack(closes, axis=0)


def _simple_returns(panel: np.ndarray) -> np.ndarray:
    return panel[1:] / panel[:-1] - 1.0


def _realized_vol(panel: np.ndarray) -> float:
    """Mean per-symbol standard deviation of simple returns (the crate's proxy)."""
    return float(_simple_returns(panel).std(axis=0).mean())


def _certify_calm_cell(
    knobs: dict[str, float], seeds: list[int], baseline: dict[int, np.ndarray]
) -> dict:
    """Certify one Calm knob setting over ``seeds`` against the default panels."""
    per_seed = []
    for seed in seeds:
        panel = _collect_panel("calm", seed, **knobs)
        report = certify_realism(panel, kind="price")
        base = baseline[seed]
        base_vol = _realized_vol(base)
        diff = _simple_returns(panel) - _simple_returns(base)
        per_seed.append(
            {
                "seed": seed,
                "facts": report.facts,
                "checks": report.checks,
                "thresholds": {k: list(v) for k, v in report.thresholds.items()},
                "passed": report.passed,
                "vol_ratio": _realized_vol(panel) / base_vol,
                "rms_perturbation": float(np.sqrt(np.mean(diff**2)) / base_vol),
            }
        )
    fact_names = sorted(per_seed[0]["facts"])
    return {
        "per_seed": per_seed,
        "n_pass": int(sum(r["passed"] for r in per_seed)),
        "n_seeds": len(seeds),
        "check_pass_counts": {
            c: int(sum(r["checks"][c] for r in per_seed)) for c in per_seed[0]["checks"]
        },
        "mean_facts": {
            f: float(np.nanmean([r["facts"][f] for r in per_seed])) for f in fact_names
        },
        "mean_vol_ratio": float(np.mean([r["vol_ratio"] for r in per_seed])),
        "mean_rms_perturbation": float(
            np.mean([r["rms_perturbation"] for r in per_seed])
        ),
    }


def calm_calibration() -> dict:
    baseline = {
        seed: _collect_panel("calm", seed)
        for seed in SEEDS + CALM_CONFIRM_SEEDS + CALM_FINAL_SEEDS + CALM_WIDE_SEEDS
    }
    jump_cells = [(0.0, 0.0, 0.0)] + [
        (p, per, size)
        for p in CALM_JUMP_PROB_GRID
        for per in CALM_JUMP_PERSIST_GRID
        for size in CALM_JUMP_SIZE_GRID
    ]
    cells = []
    for vc in CALM_VC_GRID:
        for p, per, size in jump_cells:
            knobs = {
                "vol_clustering": vc,
                "jump_burst_probability": p,
                "jump_burst_persistence": per,
                "jump_burst_size": size,
            }
            diag = _certify_calm_cell(knobs, SEEDS, baseline)
            conf = _certify_calm_cell(knobs, CALM_CONFIRM_SEEDS, baseline)
            cells.append({"knobs": knobs, "diagnostic": diag, "confirmation": conf})

    min_pass = CALM_RULE["min_pass_of_8"]
    vol_max = CALM_RULE["vol_ratio_max"]
    for cell in cells:
        d, c = cell["diagnostic"], cell["confirmation"]
        cell["passes_diagnostic"] = d["n_pass"] >= min_pass
        cell["within_vol_bound"] = d["mean_vol_ratio"] <= vol_max
        cell["passes_confirmation"] = c["n_pass"] >= min_pass
        cell["qualifies"] = (
            cell["passes_diagnostic"]
            and cell["within_vol_bound"]
            and cell["passes_confirmation"]
        )
    qualifying = [c for c in cells if c["qualifies"]]
    chosen = (
        min(qualifying, key=lambda c: c["diagnostic"]["mean_rms_perturbation"])
        if qualifying
        else None
    )
    # The in-sample minimum (rule steps 1, 2 and 4 without step 3) is reported so
    # the reader can see whether selecting on the diagnostic seeds alone replicates.
    in_sample = [c for c in cells if c["passes_diagnostic"] and c["within_vol_bound"]]
    in_sample_min = (
        min(in_sample, key=lambda c: c["diagnostic"]["mean_rms_perturbation"])
        if in_sample
        else None
    )
    out = {
        "grid": {
            "vol_clustering": list(CALM_VC_GRID),
            "jump_burst_probability": list(CALM_JUMP_PROB_GRID),
            "jump_burst_persistence": list(CALM_JUMP_PERSIST_GRID),
            "jump_burst_size": list(CALM_JUMP_SIZE_GRID),
        },
        "seed_bands": {
            "diagnostic": SEEDS,
            "confirmation": CALM_CONFIRM_SEEDS,
            "final": CALM_FINAL_SEEDS,
            "wide": CALM_WIDE_SEEDS,
        },
        "rule": CALM_RULE,
        "cells": cells,
        "n_qualifying": len(qualifying),
        "in_sample_minimum": None,
        "chosen": None,
    }
    if in_sample_min is not None:
        out["in_sample_minimum"] = {
            "knobs": in_sample_min["knobs"],
            "confirmation_n_pass": in_sample_min["confirmation"]["n_pass"],
            "qualifies": in_sample_min["qualifies"],
        }
    if chosen is not None:
        knobs = chosen["knobs"]
        out["chosen"] = {
            "knobs": knobs,
            "diagnostic": chosen["diagnostic"],
            "confirmation": chosen["confirmation"],
            "final": _certify_calm_cell(knobs, CALM_FINAL_SEEDS, baseline),
            "wide": _certify_calm_cell(knobs, CALM_WIDE_SEEDS, baseline),
            "default_calm_wide": _certify_calm_cell(
                {
                    "vol_clustering": 0.0,
                    "jump_burst_probability": 0.0,
                    "jump_burst_persistence": 0.0,
                    "jump_burst_size": 0.0,
                },
                CALM_WIDE_SEEDS,
                baseline,
            ),
        }
    return out


def _plot_calm_calibration(calib: dict) -> None:
    cells = calib["cells"]
    chosen = calib["chosen"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9, 3.8))
    markers = {0.0: "o", 0.3: "s", 0.5: "^"}
    for vc in CALM_VC_GRID:
        sub = [c for c in cells if c["knobs"]["vol_clustering"] == vc]
        xs = [c["diagnostic"]["mean_rms_perturbation"] for c in sub]
        ys = [c["diagnostic"]["n_pass"] for c in sub]
        inb = [c["within_vol_bound"] for c in sub]
        ax.scatter(
            [x for x, b in zip(xs, inb) if b],
            [y for y, b in zip(ys, inb) if b],
            marker=markers[vc],
            label=f"vol_clustering={vc} (within vol bound)",
            alpha=0.85,
        )
        ax.scatter(
            [x for x, b in zip(xs, inb) if not b],
            [y for y, b in zip(ys, inb) if not b],
            marker=markers[vc],
            facecolors="none",
            edgecolors="gray",
            alpha=0.6,
        )
    ax.axhline(CALM_RULE["min_pass_of_8"] - 0.5, color="black", linewidth=0.8, ls="--")
    if chosen is not None:
        ax.scatter(
            [chosen["diagnostic"]["mean_rms_perturbation"]],
            [chosen["diagnostic"]["n_pass"]],
            marker="*",
            s=220,
            color="red",
            zorder=5,
            label="chosen preset",
        )
    ax.set_xlabel("normalized tape perturbation (RMS / default vol)")
    ax.set_ylabel("diagnostic seeds passing (of 8)")
    ax.set_yticks(range(0, 9))
    ax.legend(frameon=False, fontsize=7)

    if chosen is not None:
        bands = ("diagnostic", "confirmation", "final", "wide")
        gated = ("excess_kurtosis", "abs_return_autocorr", "aggregational_gaussianity")
        width = 0.8 / len(bands)
        for j, band in enumerate(bands):
            rep = chosen[band]
            xs = [i + (j - (len(bands) - 1) / 2) * width for i in range(len(gated))]
            ys = [rep["check_pass_counts"][g] / rep["n_seeds"] for g in gated]
            bx.bar(xs, ys, width=width, label=f"{band} ({rep['n_pass']}/{rep['n_seeds']})")
        bx.set_xticks(range(len(gated)))
        bx.set_xticklabels(gated, rotation=20, ha="right", fontsize=8)
        bx.set_ylim(0, 1.05)
        bx.set_ylabel("per-check pass fraction, chosen preset")
        bx.legend(frameon=False, fontsize=7, title="band (conjunction)")
    fig.tight_layout()
    fig.savefig(FIGURES / "f4-calm-calibration.pdf")
    plt.close(fig)


def main() -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    def certify_tiers(vol_clustering: float) -> dict[str, dict]:
        tiers: dict[str, dict] = {}
        for tier in TIERS:
            per_seed = []
            for seed in SEEDS:
                panel = _collect_panel(tier, seed, vol_clustering)
                report = certify_realism(panel, kind="price")
                per_seed.append(
                    {
                        "seed": seed,
                        "n_bars": int(panel.shape[0]),
                        "facts": report.facts,
                        "checks": report.checks,
                        "thresholds": {k: list(v) for k, v in report.thresholds.items()},
                        "passed": report.passed,
                    }
                )
            fact_names = sorted(per_seed[0]["facts"])
            mean_facts = {
                f: float(np.nanmean([r["facts"][f] for r in per_seed]))
                for f in fact_names
            }
            tiers[tier] = {
                "per_seed": per_seed,
                "mean_facts": mean_facts,
                "pass_rate": float(np.mean([r["passed"] for r in per_seed])),
            }
        return tiers

    tiers = certify_tiers(0.0)
    clustered_tiers = certify_tiers(VOL_CLUSTERING)
    calib = calm_calibration()

    out = {
        "finding": "F4",
        "config": {
            "n_symbols": N_SYMBOLS,
            "n_days": N_DAYS,
            "seeds": SEEDS,
            "max_steps": MAX_STEPS,
            "tiers": list(TIERS),
            "vol_clustering": VOL_CLUSTERING,
        },
        "tiers": tiers,
        "clustered_tiers": clustered_tiers,
        "calm_calibration": calib,
    }
    (EVIDENCE / "f4-realism.json").write_text(json.dumps(out, indent=2))
    _plot_calm_calibration(calib)

    # Figure: mean fact value per tier, grouped by fact.
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


if __name__ == "__main__":
    main()
