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
The single positive-control cell selected by that exploratory grid is then evaluated once
on a disjoint, fixed 32-seed confirmation band.  That confirmation is one prespecified
cell, not another search; its interval is therefore reported separately from the 135-cell
selection-adjusted result.
The linear and concave arms above
are untouched: their code paths and evidence numbers are byte-identical.

Extended sweeps (the ``extended_sweeps`` key): the positive control lists the axes it
did not search. This block sweeps each of them, one at a time, through the positive
control's best cell (theory arm, uniform 9:1, exponent 0.5) and through the same cell
under linear impact: push size, permanent-impact coefficient, leg length at fixed 9:1
ratio, an interleaved hold, the mirrored short-side round trip, and follower gain on the
canonical follower arm. Five points per axis, eight seeds per cell, 60 cells; every cell
stores a pointwise t interval and a Bonferroni familywise interval over all 60. The
anchor cell is asserted to reproduce the positive control's winner exactly, and every
earlier key is byte-identical.
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from sharpearena import (
        AsymmetricSchedule,
        ManipulationParams,
        impact_boundary_sweep,
        run_asymmetric_probe,
        run_manipulation_probe,
        size_response,
    )
except ImportError:  # --figures-only reads the committed JSON and needs no bindings
    AsymmetricSchedule = ManipulationParams = None
    impact_boundary_sweep = run_asymmetric_probe = None
    run_manipulation_probe = size_response = None

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

# Confirmation follows the exploratory grid on a fresh band.  The cell is deliberately
# spelled out rather than re-selected from the grid: it is one hypothesis on 32 new seeds.
# The band is disjoint from the 0..7 selection/evidence seeds and all F4 calibration bands.
POSITIVE_CONTROL_CONFIRM_SEEDS = tuple(range(30_000, 30_032))
CONFIRM_T_CRIT_DF31 = 2.0395134463964077

# Extended sweeps (the ``extended_sweeps`` key): the axes the positive control lists as
# unsearched, each swept one at a time through the best-known asymmetric cell (uniform
# 9:1 accumulate-then-block, theory arm) at the linear exponent and at 0.5. Five points
# per axis, the base cell included as an anchor so every axis is comparable to the
# positive control's winner. ``leg_length`` keeps the 9:1 ratio and scales total trading
# bars; ``hold_bars`` interleaves a flat-weight hold between the legs; ``short_side``
# mirrors the sign of the five uniform duration ratios the positive control searched;
# ``follower_gain`` runs on the canonical follower arm (temporary impact on, three
# momentum followers) rather than the theory arm, since that is where followers exist.
EXTENDED_EXPONENTS = (1.0, 0.5)
EXTENDED_BASE_SCHEDULE = (
    None if AsymmetricSchedule is None else AsymmetricSchedule.uniform(9, 1)
)
EXTENDED_AXES: dict[str, tuple] = {
    "push_weight": (0.2, 0.4, 0.6, 0.8, 1.0),
    "kyle_lambda": (0.025, 0.05, 0.1, 0.2, 0.4),
    "leg_length": ((9, 1), (18, 2), (27, 3), (36, 4), (45, 5)),
    "hold_bars": (0, 1, 3, 6, 12),
    "short_side": ((1, 9), (2, 8), (5, 5), (8, 2), (9, 1)),
    "follower_gain": (0.0, 15.0, 30.0, 60.0, 120.0),
}
EXTENDED_ARMS = {
    "push_weight": "theory",
    "kyle_lambda": "theory",
    "leg_length": "theory",
    "hold_bars": "theory",
    "short_side": "theory",
    "follower_gain": "canonical",
}
# The extension selects its anchor from the earlier 135-cell grid on the same eight seeds.
# Its inference must therefore include that earlier search as well as the 60 new cells.
# Every extension cell (including anchor repeats) counts toward this global family.
EXTENDED_FAMILY_SIZE = sum(len(v) for v in EXTENDED_AXES.values()) * len(EXTENDED_EXPONENTS)
GLOBAL_MANIPULATION_FAMILY_SIZE = POSITIVE_CONTROL_FAMILY_SIZE + EXTENDED_FAMILY_SIZE
# Student-t quantile t_{1 - .05/(2*195), 7}, computed once with SciPy and pinned so
# regenerating evidence does not add SciPy as a runtime requirement.
BONFERRONI_T_CRIT_DF7_GLOBAL = 6.7861752374916025
assert EXTENDED_FAMILY_SIZE == 60
assert GLOBAL_MANIPULATION_FAMILY_SIZE == 195


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

    def _stats(values: list[float], critical_value: float = t_crit) -> dict:
        n = len(values)
        mean = sum(values) / n
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
        half = critical_value * std / math.sqrt(n)
        return {"mean": mean, "std": std, "ci95_lo": mean - half, "ci95_hi": mean + half}

    def _familywise_stats(
        values: list[float],
        family_size: int = POSITIVE_CONTROL_FAMILY_SIZE,
        critical_value: float = BONFERRONI_T_CRIT_DF7,
    ) -> dict:
        n = len(values)
        mean = sum(values) / n
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
        half = critical_value * std / math.sqrt(n)
        return {
            "method": "Bonferroni two-sided familywise 95% interval",
            "family_size": family_size,
            "df": n - 1,
            "critical_value": critical_value,
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
    # A single fresh-band confirmation of the cell selected in the exploratory grid.
    # It is intentionally not maximized over any schedule, arm or exponent on these seeds.
    confirm_params = replace(params, eta=0.0, follower_gain=0.0, impact_exponent=0.5)
    confirm_schedule = AsymmetricSchedule.uniform(9, 1)
    confirm_row = [
        run_asymmetric_probe(params=confirm_params, schedule=confirm_schedule, seed=s).impact_pnl
        for s in POSITIVE_CONTROL_CONFIRM_SEEDS
    ]
    positive_control["confirmation"] = {
        "protocol": (
            "one fixed cell on a disjoint fresh seed band; no schedule/arm/exponent selection "
            "or multiplicity adjustment is applied to this one-hypothesis confirmation"
        ),
        "selection_relation": (
            "cell was selected by the 135-cell exploratory grid on seeds 0..7; these 32 seeds "
            "were not used by that grid or the extended sweeps"
        ),
        "cell": {
            "arm": "theory",
            "impact_exponent": 0.5,
            "schedule": confirm_schedule.to_dict(),
            "eta": confirm_params.eta,
            "follower_gain": confirm_params.follower_gain,
            "kyle_lambda": confirm_params.kyle_lambda,
            "push_weight": confirm_params.push_weight,
        },
        "seeds": list(POSITIVE_CONTROL_CONFIRM_SEEDS),
        "per_seed_impact_pnl": confirm_row,
        "stats": {
            **_stats(confirm_row, CONFIRM_T_CRIT_DF31),
            "method": "two-sided Student-t 95% interval, df=31, one fixed confirmation cell",
            "critical_value": CONFIRM_T_CRIT_DF31,
        },
    }

    # Extended sweeps: one axis at a time through the best-known asymmetric cell. Additive;
    # the positive control's numbers above are not touched, and the anchor cell (theory
    # arm, exponent 0.5, push 0.8, lambda 0.1, uniform 9:1, hold 1) is asserted to
    # reproduce the positive control's selected winner exactly.
    def _extended_cell(
        axis: str,
        value,
        exponent: float,
        cell_params: ManipulationParams,
        sched: AsymmetricSchedule,
        side: int,
    ) -> dict:
        row = [
            run_asymmetric_probe(params=cell_params, schedule=sched, seed=s, side=side).impact_pnl
            for s in SEEDS
        ]
        st = _stats(row)
        fw = _familywise_stats(
            row, GLOBAL_MANIPULATION_FAMILY_SIZE, BONFERRONI_T_CRIT_DF7_GLOBAL
        )
        return {
            "axis": axis,
            "value": list(value) if isinstance(value, tuple) else value,
            "impact_exponent": exponent,
            "schedule": sched.to_dict(),
            "side": side,
            "params": {
                "kyle_lambda": cell_params.kyle_lambda,
                "eta": cell_params.eta,
                "push_weight": cell_params.push_weight,
                "follower_gain": cell_params.follower_gain,
                "n_followers": cell_params.n_followers,
                "hold_bars": cell_params.hold_bars,
            },
            "per_seed_impact_pnl": row,
            "stats": st,
            "familywise_inference": fw,
            "profitable_mean": st["mean"] > 0.0,
            "profitable_ci": st["ci95_lo"] > 0.0,
            "profitable_familywise": fw["ci95_lo"] > 0.0,
            "unprofitable_familywise": fw["ci95_hi"] < 0.0,
        }

    def _paired_stats(a: list[float], b: list[float]) -> dict:
        # Descriptive pointwise paired-t interval on same-seed differences. It is retained
        # for schedule-shape diagnosis only, not used for any selected-cell inference.
        diffs = [x - y for x, y in zip(a, b)]
        st = _stats(diffs)
        return {
            "mean_diff": st["mean"],
            "ci95_lo": st["ci95_lo"],
            "ci95_hi": st["ci95_hi"],
            "method": "descriptive pointwise paired Student-t 95% interval; no multiplicity claim",
        }

    extended: dict = {"axes": {}, "summary": {}}
    pc_theory_uniform = {
        (e, pt["schedule"]["up_bars"], pt["schedule"]["down_bars"]): pt
        for e, blk in positive_control["arms"]["theory"]["by_exponent"].items()
        for pt in blk["points"]
        if pt["size_split_label"] == "uniform"
    }
    n_cells_run = 0
    for axis, values in EXTENDED_AXES.items():
        arm_p = replace(params, **POSITIVE_CONTROL_ARMS[EXTENDED_ARMS[axis]])
        by_exponent: dict[str, dict] = {}
        for exponent in EXTENDED_EXPONENTS:
            ep = replace(arm_p, impact_exponent=exponent)
            cells = []
            for value in values:
                sched, side, cp = EXTENDED_BASE_SCHEDULE, 1, ep
                if axis == "push_weight":
                    cp = replace(ep, push_weight=value)
                elif axis == "kyle_lambda":
                    cp = replace(ep, kyle_lambda=value)
                elif axis == "leg_length":
                    sched = AsymmetricSchedule.uniform(*value)
                elif axis == "hold_bars":
                    cp = replace(ep, hold_bars=value)
                elif axis == "short_side":
                    sched, side = AsymmetricSchedule.uniform(*value), -1
                elif axis == "follower_gain":
                    cp = replace(ep, follower_gain=value)
                cell = _extended_cell(axis, value, exponent, cp, sched, side)
                if axis == "short_side":
                    long_ref = pc_theory_uniform[(str(exponent), value[0], value[1])]
                    cell["long_side_reference"] = {
                        "per_seed_impact_pnl": long_ref["per_seed_impact_pnl"],
                        "stats": long_ref["stats"],
                    }
                    cell["short_minus_long"] = _paired_stats(
                        cell["per_seed_impact_pnl"], long_ref["per_seed_impact_pnl"]
                    )
                cells.append(cell)
                n_cells_run += 1
            best = max(cells, key=lambda c: c["stats"]["mean"])
            by_exponent[str(exponent)] = {
                "impact_exponent": exponent,
                "cells": cells,
                "n_cells": len(cells),
                "n_profitable_mean": sum(c["profitable_mean"] for c in cells),
                "n_profitable_ci": sum(c["profitable_ci"] for c in cells),
                "n_profitable_familywise": sum(c["profitable_familywise"] for c in cells),
                "n_unprofitable_familywise": sum(c["unprofitable_familywise"] for c in cells),
                "best": {
                    "value": best["value"],
                    "stats": best["stats"],
                    "familywise_inference": best["familywise_inference"],
                },
            }
        extended["axes"][axis] = {
            "arm": EXTENDED_ARMS[axis],
            "arm_params": POSITIVE_CONTROL_ARMS[EXTENDED_ARMS[axis]],
            "values": [list(v) if isinstance(v, tuple) else v for v in values],
            "by_exponent": by_exponent,
        }
    assert n_cells_run == EXTENDED_FAMILY_SIZE

    # Anchor cross-check: the extended base cell must be the positive control's winner.
    pc_best = positive_control["arms"]["theory"]["by_exponent"]["0.5"]["best"]
    assert pc_best["schedule"] == EXTENDED_BASE_SCHEDULE.to_dict()
    anchor = next(
        c
        for c in extended["axes"]["push_weight"]["by_exponent"]["0.5"]["cells"]
        if c["value"] == params.push_weight
    )
    assert anchor["stats"] == pc_best["stats"], "anchor cell drifted from positive control"

    extended["summary"] = {
        axis: {
            e: {
                "n_cells": blk["n_cells"],
                "n_profitable_mean": blk["n_profitable_mean"],
                "n_profitable_ci": blk["n_profitable_ci"],
                "n_profitable_familywise": blk["n_profitable_familywise"],
                "n_unprofitable_familywise": blk["n_unprofitable_familywise"],
                "best_value": blk["best"]["value"],
                "best_mean": blk["best"]["stats"]["mean"],
                "best_familywise_lo": blk["best"]["familywise_inference"]["ci95_lo"],
            }
            for e, blk in a["by_exponent"].items()
        }
        for axis, a in extended["axes"].items()
    }
    extended["config"] = {
        "exponents": list(EXTENDED_EXPONENTS),
        "base_schedule": EXTENDED_BASE_SCHEDULE.to_dict(),
        "axes": {k: [list(v) if isinstance(v, tuple) else v for v in vals] for k, vals in EXTENDED_AXES.items()},
        "arms": EXTENDED_ARMS,
        "seeds": list(SEEDS),
        "n_extended_cells": EXTENDED_FAMILY_SIZE,
        "family_size": GLOBAL_MANIPULATION_FAMILY_SIZE,
        "pointwise_critical_value": t_crit,
        "familywise_critical_value": BONFERRONI_T_CRIT_DF7_GLOBAL,
        "selection_inference": (
            "Bonferroni two-sided familywise 95% interval over the global 195-cell family "
            "(the 135-cell antecedent selection grid plus all 60 extended cells), stored for "
            "every extension cell; repeated anchors are counted conservatively"
        ),
        "not_searched": [
            "two-axis interactions (each axis is swept alone through the anchor cell)",
            "exponent 0.7",
            "block size splits other than uniform on the extended axes",
            "temporary impact on the push, lambda, leg, hold and short axes (theory arm, eta = 0)",
            "n_followers other than 3, volume_scale, other distribution modes",
            "overshooting round trips (net position never crosses zero mid-trip)",
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
        "extended_sweeps": extended,
    }
    (EVIDENCE / "f5-manipulation.json").write_text(json.dumps(out, indent=2))
    make_figures(out)


def make_figures(out: dict, wanted: list[str] | None = None) -> None:
    """Render the F5 figures from the evidence dict (the committed JSON shape).

    ``wanted`` restricts which PDFs are written; every figure is still laid out
    so the code path stays exercised, only the save is skipped.
    """
    FIGURES.mkdir(parents=True, exist_ok=True)
    boundaries = out["boundaries"]
    size = out["size_response"]
    dispersion = out["dispersion"]
    concave = out["concave"]
    positive_control = out["positive_control"]
    extended = out["extended_sweeps"]

    def _save(fig, name: str) -> None:
        if wanted is None or name in wanted:
            fig.savefig(FIGURES / name)
            print(f"wrote {FIGURES / name}")
        plt.close(fig)

    # Figure 1: impact P&L along each swept axis, boundary marked where found.
    # A 2+1 grid sized for its 0.55-linewidth subfigure slot (about 3.0 in),
    # so fonts land at 8 to 9 pt at print size.
    fig = plt.figure(figsize=(3.05, 3.5))
    gs = fig.add_gridspec(2, 4)
    slots = [gs[0, 0:2], gs[0, 2:4], gs[1, 1:3]]
    first_ax = None
    for slot, (axis, rep) in zip(slots, boundaries.items()):
        ax = fig.add_subplot(slot, sharey=first_ax)
        first_ax = first_ax or ax
        ax.plot(rep["values"], rep["impact_pnl"], marker="o", markersize=3,
                linewidth=1.2)
        ax.axhline(0.0, color="black", linewidth=0.8)
        if rep["boundary"] is not None:
            ax.axvline(rep["boundary"], linestyle="--", color="red", linewidth=0.8)
        ax.set_xlabel(axis, fontsize=9)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
        ax.yaxis.get_offset_text().set_fontsize(8)
        ax.tick_params(labelsize=8)
        if slot is slots[1]:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel("impact P&L", fontsize=9)
    fig.tight_layout()
    _save(fig, "f5-boundaries.pdf")

    # Figure 2: payoff vs push size, with the peak marked. Sized for its
    # 0.42-linewidth slot and matched in height to Figure 1.
    fig, ax = plt.subplots(figsize=(2.35, 3.5))
    ax.plot(size["push_weights"], size["impact_pnl"], marker="o", markersize=3,
            linewidth=1.2)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axvline(size["peak_push_weight"], linestyle="--", color="red", linewidth=0.8)
    verdict = "bounded" if size["bounded"] else "UNBOUNDED"
    ax.set_title(f"size response ({verdict})", fontsize=9)
    ax.set_xlabel("push weight", fontsize=9)
    ax.set_ylabel("impact P&L", fontsize=9)
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    ax.yaxis.get_offset_text().set_fontsize(8)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    _save(fig, "f5-size-response.pdf")

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
    _save(fig, "f5-concave.pdf")

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
    _save(fig, "f5-positive-control.pdf")

    # Figure 5: the extended sweeps. One panel per axis; impact P&L against the axis value,
    # one line per exponent, pointwise 95% shading and familywise 95% error bars. The
    # short-side panel also draws the positive control's long-side theory-arm cells dashed.
    def _x_of(axis: str, value) -> float:
        if axis == "leg_length":
            return float(value[0] + value[1])
        if axis == "short_side":
            return float(value[0]) / float(value[1])
        return float(value)

    xlabels = {
        "push_weight": "push weight",
        "kyle_lambda": "kyle_lambda (log)",
        "leg_length": "total trading bars (9:1 legs)",
        "hold_bars": "hold bars between legs",
        "short_side": "up/down duration ratio (log), short side",
        "follower_gain": "follower gain (canonical arm)",
    }
    fig, panels = plt.subplots(2, 3, figsize=(10.5, 6.0))
    for ax, (axis, blk) in zip(panels.flatten(), extended["axes"].items()):
        for e, per_e in blk["by_exponent"].items():
            cells = per_e["cells"]
            xs = [_x_of(axis, c["value"]) for c in cells]
            ys = [c["stats"]["mean"] for c in cells]
            lo = [c["stats"]["ci95_lo"] for c in cells]
            hi = [c["stats"]["ci95_hi"] for c in cells]
            fw_lo = [c["familywise_inference"]["ci95_lo"] for c in cells]
            fw_hi = [c["familywise_inference"]["ci95_hi"] for c in cells]
            line, = ax.plot(xs, ys, marker="o", markersize=3, label=f"exponent {e}")
            ax.fill_between(xs, lo, hi, alpha=0.15, color=line.get_color())
            ax.errorbar(
                xs, ys,
                yerr=[[y - l for y, l in zip(ys, fw_lo)], [h - y for y, h in zip(ys, fw_hi)]],
                fmt="none", ecolor=line.get_color(), elinewidth=0.6, capsize=2,
            )
            if axis == "short_side":
                ref = [c["long_side_reference"]["stats"]["mean"] for c in cells]
                ax.plot(
                    xs, ref, linestyle="--", marker="x", markersize=3,
                    color=line.get_color(), label=f"long side, exponent {e}",
                )
        ax.axhline(0.0, color="black", linewidth=0.8)
        if axis in ("kyle_lambda", "short_side"):
            ax.set_xscale("log")
        ax.set_xlabel(xlabels[axis], fontsize=9)
        ax.set_ylabel("impact P&L", fontsize=9)
        ax.tick_params(labelsize=8)
    panels[0][0].legend(fontsize=7, frameon=False)
    panels[1][1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    _save(fig, "f5-extended-sweeps.pdf")


if __name__ == "__main__":
    if "--figures-only" in sys.argv:
        names = [a for a in sys.argv[1:] if not a.startswith("--")]
        data = json.loads((EVIDENCE / "f5-manipulation.json").read_text())
        make_figures(data, wanted=names or None)
    else:
        main()
