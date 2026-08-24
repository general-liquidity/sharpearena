#!/usr/bin/env python3
"""F5: the manipulation payoff boundary and the size response.

``impact_boundary_sweep`` sweeps one impact axis (permanent impact, temporary
impact, follower gain) and reports where the pump-and-unwind round trip stops
paying against its zero-impact paired reference; ``size_response`` sweeps the
push weight and reports whether the payoff is bounded. The sweep APIs return
seed means only, so the script additionally collects per-seed impact PnL at
every grid point via the public ``run_manipulation_probe`` and serializes
per-seed vectors plus t-based 95% CIs (asserting the per-seed means reproduce
the API sweeps). Writes JSON plus figures. These probes diagnose the
simulator's impact specification, not agents.

Concave arm (the falsifiability ablation): under linear permanent impact,
round-trip unprofitability is a theorem (Huberman-Stanzl 2004), so the linear
probe can only confirm theory. The ``concave`` key reruns the same sweeps at
``impact_exponent`` 0.5 and 0.7 (permanent impact concave in flow, the regime
in which theory predicts manipulation can pay), with the same per-seed CIs.

Positive control (the ``positive_control`` key): the symmetric schedule above never
searches asymmetric round trips. This block runs an exploratory schedule search over the
``AsymmetricSchedule`` family (up/down duration ratio and block-fraction size split)
at exponents 1.0, 0.7 and 0.5, on a pure-impact theory arm (no followers, no
temporary impact), a temporary-impact arm and the canonical follower arm, looking
for profitable sampled round trips with per-seed intervals. Because the best cell is
selected from 135 points, its report also carries a Bonferroni familywise 95% interval.
The linear and concave arms above
are untouched: their code paths and evidence numbers are byte-identical.
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
    AsymmetricSchedule,
    ManipulationParams,
    impact_boundary_sweep,
    run_asymmetric_probe,
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
# The concavity ablation: permanent-impact exponents below 1.0 (0.5 = square-root law).
CONCAVE_EXPONENTS = (0.5, 0.7)

# Positive control: the asymmetric-schedule search. Exponents include 1.0 so the linear
# theorem is tested on the same shapes. Leg lengths keep the symmetric trip's 10 bars of
# trading; (9, 1) is slow-accumulate / block-liquidate, the Gatheral (2010) shape, and
# (1, 9) is its mirror. Size splits are the block fraction at the turn; None = uniform.
POSITIVE_CONTROL_EXPONENTS = (1.0, 0.7, 0.5)
POSITIVE_CONTROL_LEGS = ((1, 9), (2, 8), (5, 5), (8, 2), (9, 1))
POSITIVE_CONTROL_SPLITS = (None, 0.5, 0.9)
# Arms: pure permanent impact (the theory case), temporary impact added, and the canonical
# follower ecology of the linear/concave arms.
POSITIVE_CONTROL_ARMS = {
    "theory": {"eta": 0.0, "follower_gain": 0.0},
    "temporary_impact": {"eta": 0.05, "follower_gain": 0.0},
    "canonical": {"eta": 0.05, "follower_gain": 30.0},
}
POSITIVE_CONTROL_FAMILY_SIZE = (
    len(POSITIVE_CONTROL_EXPONENTS)
    * len(POSITIVE_CONTROL_LEGS)
    * len(POSITIVE_CONTROL_SPLITS)
    * len(POSITIVE_CONTROL_ARMS)
)
# Student-t quantile t_{1 - .05/(2*135), 7}. Kept explicit so regenerating the
# evidence does not add SciPy as a runtime dependency of the paper pipeline.
BONFERRONI_T_CRIT_DF7 = 6.391202695754376


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

    def _familywise_stats(values: list[float]) -> dict:
        n = len(values)
        mean = sum(values) / n
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
        half = BONFERRONI_T_CRIT_DF7 * std / math.sqrt(n)
        return {
            "method": "Bonferroni two-sided familywise 95% interval",
            "family_size": POSITIVE_CONTROL_FAMILY_SIZE,
            "df": n - 1,
            "critical_value": BONFERRONI_T_CRIT_DF7,
            "ci95_lo": mean - half,
            "ci95_hi": mean + half,
        }

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

    # Concave arm: the same boundary/size sweeps, per-seed rows and CIs at each
    # impact_exponent < 1. The scientific question is whether pump-and-dump becomes
    # profitable anywhere once permanent impact is concave, as Huberman-Stanzl predict.
    concave: dict[str, dict] = {}
    for exponent in CONCAVE_EXPONENTS:
        cp = replace(params, impact_exponent=exponent)
        c_boundaries = {
            axis: impact_boundary_sweep(
                params=cp, axis=axis, values=values, seeds=SEEDS
            ).to_dict()
            for axis, values in AXES.items()
        }
        c_size = size_response(params=cp, seeds=SEEDS).to_dict()
        c_dispersion: dict[str, dict] = {}
        for axis, rep in c_boundaries.items():
            rows = [_per_seed(replace(cp, **{axis: v})) for v in rep["values"]]
            for row, mean in zip(rows, rep["impact_pnl"]):
                assert abs(sum(row) / len(row) - mean) < 1e-12
            c_dispersion[axis] = {
                "values": rep["values"],
                "per_seed_impact_pnl": rows,
                "stats": [_stats(r) for r in rows],
            }
        c_size_rows = [
            _per_seed(replace(cp, push_weight=w)) for w in c_size["push_weights"]
        ]
        for row, mean in zip(c_size_rows, c_size["impact_pnl"]):
            assert abs(sum(row) / len(row) - mean) < 1e-12
        c_dispersion["push_weight"] = {
            "values": c_size["push_weights"],
            "per_seed_impact_pnl": c_size_rows,
            "stats": [_stats(r) for r in c_size_rows],
        }
        base_row = _per_seed(cp)
        concave[str(exponent)] = {
            "impact_exponent": exponent,
            "base": {"per_seed_impact_pnl": base_row, "stats": _stats(base_row)},
            "boundaries": c_boundaries,
            "size_response": c_size,
            "dispersion": c_dispersion,
            "profitable_anywhere": any(
                any(rep["profitable"]) for rep in c_boundaries.values()
            )
            or any(v > 0.0 for v in c_size["impact_pnl"]),
        }

    # Positive control: the asymmetric schedule search. Additive to the arms above; the
    # symmetric probe is never re-run here and its numbers are not touched.
    positive_control: dict = {"arms": {}, "summary": {}}
    for arm_name, arm_overrides in POSITIVE_CONTROL_ARMS.items():
        arm_p = replace(params, **arm_overrides)
        by_exponent: dict[str, dict] = {}
        for exponent in POSITIVE_CONTROL_EXPONENTS:
            ep = replace(arm_p, impact_exponent=exponent)
            points = []
            for up, down in POSITIVE_CONTROL_LEGS:
                for split in POSITIVE_CONTROL_SPLITS:
                    sched = (
                        AsymmetricSchedule.uniform(up, down)
                        if split is None
                        else AsymmetricSchedule(up, down, split)
                    )
                    row = [
                        run_asymmetric_probe(params=ep, schedule=sched, seed=s).impact_pnl
                        for s in SEEDS
                    ]
                    st = _stats(row)
                    points.append(
                        {
                            "schedule": sched.to_dict(),
                            "size_split_label": "uniform" if split is None else split,
                            "per_seed_impact_pnl": row,
                            "stats": st,
                            "profitable_mean": st["mean"] > 0.0,
                            "profitable_ci": st["ci95_lo"] > 0.0,
                        }
                    )
            best = max(points, key=lambda pt: pt["stats"]["mean"])
            best_familywise = _familywise_stats(best["per_seed_impact_pnl"])
            by_exponent[str(exponent)] = {
                "impact_exponent": exponent,
                "points": points,
                "n_profitable_mean": sum(pt["profitable_mean"] for pt in points),
                "n_profitable_ci": sum(pt["profitable_ci"] for pt in points),
                "best": {
                    "schedule": best["schedule"],
                    "stats": best["stats"],
                    "familywise_inference": best_familywise,
                },
            }
        positive_control["arms"][arm_name] = {
            "params": {
                "eta": arm_p.eta,
                "follower_gain": arm_p.follower_gain,
                "kyle_lambda": arm_p.kyle_lambda,
                "push_weight": arm_p.push_weight,
            },
            "by_exponent": by_exponent,
        }
    positive_control["summary"] = {
        arm: {
            k: {
                "n_points": len(v["points"]),
                "n_profitable_mean": v["n_profitable_mean"],
                "n_profitable_ci": v["n_profitable_ci"],
                "best_mean": v["best"]["stats"]["mean"],
                "best_schedule": v["best"]["schedule"],
            }
            for k, v in a["by_exponent"].items()
        }
        for arm, a in positive_control["arms"].items()
    }
    positive_control["config"] = {
        "exponents": list(POSITIVE_CONTROL_EXPONENTS),
        "legs": [list(l) for l in POSITIVE_CONTROL_LEGS],
        "size_splits": ["uniform" if s is None else s for s in POSITIVE_CONTROL_SPLITS],
        "arms": POSITIVE_CONTROL_ARMS,
        "seeds": list(SEEDS),
        "family_size": POSITIVE_CONTROL_FAMILY_SIZE,
        "selection_inference": "Bonferroni two-sided familywise 95% interval for each selected best cell",
        "not_searched": [
            "push_weight (fixed at the canonical 0.8)",
            "flow scale / volume_scale (the sub-unit flow calibration is unchanged)",
            "kyle_lambda (fixed at 0.1)",
            "legs longer than 10 trading bars, hold longer than 1 bar",
            "short-side or overshooting round trips",
            "follower gains other than 0 and 30",
        ],
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
                "impact_exponent": params.impact_exponent,
            },
            "concave_exponents": list(CONCAVE_EXPONENTS),
        },
        "boundaries": boundaries,
        "size_response": size,
        "dispersion": dispersion,
        "concave": concave,
        "ci_convention": "t-based 95% over per-seed impact PnL, df=7",
        "positive_control": positive_control,
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

    # Figure 3: the concavity ablation. Impact P&L (with 95% CIs) along the permanent-
    # impact axis and the push-size axis, linear vs each concave exponent.
    fig, (ax_l, ax_s) = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)

    def _with_ci(ax, values, disp, label):
        means = [s["mean"] for s in disp["stats"]]
        lo = [s["ci95_lo"] for s in disp["stats"]]
        hi = [s["ci95_hi"] for s in disp["stats"]]
        ax.plot(values, means, marker="o", label=label)
        ax.fill_between(values, lo, hi, alpha=0.2)

    _with_ci(ax_l, dispersion["kyle_lambda"]["values"], dispersion["kyle_lambda"], "linear")
    _with_ci(ax_s, dispersion["push_weight"]["values"], dispersion["push_weight"], "linear")
    for key, arm in concave.items():
        _with_ci(
            ax_l,
            arm["dispersion"]["kyle_lambda"]["values"],
            arm["dispersion"]["kyle_lambda"],
            f"exponent {key}",
        )
        _with_ci(
            ax_s,
            arm["dispersion"]["push_weight"]["values"],
            arm["dispersion"]["push_weight"],
            f"exponent {key}",
        )
    for ax, xlabel in ((ax_l, "kyle_lambda"), (ax_s, "push weight")):
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel(xlabel)
    ax_l.set_ylabel("impact P&L")
    ax_l.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "f5-concave.pdf")
    plt.close(fig)

    # Figure 4: the positive control. One column per exponent, one row per arm; impact P&L
    # with 95% CIs against the up/down duration ratio, one line per size split.
    arms = list(positive_control["arms"].items())
    exps = [str(e) for e in POSITIVE_CONTROL_EXPONENTS]
    fig, grid_axes = plt.subplots(
        len(arms), len(exps), figsize=(3.2 * len(exps), 2.6 * len(arms)),
        sharex=True, sharey="row",
    )
    for i, (arm_name, arm) in enumerate(arms):
        for j, e in enumerate(exps):
            ax = grid_axes[i][j]
            pts = arm["by_exponent"][e]["points"]
            for split in POSITIVE_CONTROL_SPLITS:
                label = "uniform" if split is None else split
                sel = [pt for pt in pts if pt["size_split_label"] == label]
                xs = [pt["schedule"]["duration_ratio"] for pt in sel]
                ys = [pt["stats"]["mean"] for pt in sel]
                lo = [pt["stats"]["ci95_lo"] for pt in sel]
                hi = [pt["stats"]["ci95_hi"] for pt in sel]
                ax.plot(xs, ys, marker="o", markersize=3, label=f"split {label}")
                ax.fill_between(xs, lo, hi, alpha=0.15)
            ax.axhline(0.0, color="black", linewidth=0.8)
            ax.set_xscale("log")
            if i == 0:
                ax.set_title(f"exponent {e}", fontsize=10)
            if i == len(arms) - 1:
                ax.set_xlabel("up/down duration ratio")
            if j == 0:
                ax.set_ylabel(f"{arm_name}\nimpact P&L", fontsize=9)
    grid_axes[0][0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "f5-positive-control.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
