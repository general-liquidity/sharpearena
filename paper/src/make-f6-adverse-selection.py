#!/usr/bin/env python3
"""F6: adverse-selection markout decomposition.

``compare_informed_vs_uninformed`` runs paired episodes (identical flow, sided
on alpha vs a coin, price path held fixed) and reports mean maker markout per
filled unit per horizon; one seeded ``run_adverse_selection`` episode
additionally decomposes markout by maker. The aggregate API pools quantities
over episodes, so the script additionally reruns the public
``run_adverse_selection`` per episode and leg to serialize per-episode
markout-per-unit vectors and t-based 95% CIs on the per-episode paired gap
(the per-episode ratio is episode markout over episode filled quantity; the
API's pooled aggregate is the quantity-weighted counterpart). Writes JSON plus
a figure.

The ``endogenous`` key adds the arm in which filled taker flow moves the mid
through the clearing engine's permanent-impact law (``EndogenousImpact``): the
same 24 paired episodes are rerun with the price path exogenous and endogenous,
per-episode vectors and t-based 95% CIs are serialized for maker markout levels,
the informed-uninformed gap and toxic-fill rates under both arms, plus a
``kyle_lambda`` sweep at the longest horizon. The pre-existing keys are produced
by the unchanged code above it and are byte-identical to the previous run; the
script asserts that the comparison's exogenous arm reproduces the committed
per-episode vectors exactly. Writes ``f6-endogenous.pdf``.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sharpearena import (
    AdverseSelectionParams,
    compare_informed_vs_uninformed,
    run_adverse_selection,
)
from sharpearena.adverse_selection import EndogenousImpact, compare_endogenous_arms

PAPER = Path(__file__).resolve().parents[1]
EVIDENCE = PAPER / "evidence"
FIGURES = PAPER / "figures"

N_EPISODES = 24
SEED_BASE = 0
DETAIL_SEED = 0

# Endogenous arm: the engine's default Kyle coefficient over one episode's realized taker
# volume (see EndogenousImpact), plus a sweep that brackets where the informed-flow
# markout level changes sign.
IMPACT = EndogenousImpact(kyle_lambda=0.1, volume_scale=240.0)
LAMBDA_SWEEP = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4)


def _t_crit(n: int) -> float:
    """Two-sided 97.5% Student t quantile at ``n - 1`` degrees of freedom."""
    try:
        from scipy.stats import t as student_t

        return float(student_t.ppf(0.975, n - 1))
    except Exception:  # noqa: BLE001 - scipy is optional for the paper scripts
        return {16: 2.131, 24: 2.069}[n]


def _stats_t(values: list[float]) -> dict:
    n = len(values)
    mean = sum(values) / n
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    half = _t_crit(n) * std / math.sqrt(n)
    lo, hi = mean - half, mean + half
    return {
        "mean": mean,
        "std": std,
        "ci95_lo": lo,
        "ci95_hi": hi,
        "excludes_zero": lo > 0.0 or hi < 0.0,
        "sign": "positive" if lo > 0.0 else ("negative" if hi < 0.0 else "indeterminate"),
        "per_episode": list(values),
    }


def endogenous_block(params: AdverseSelectionParams, committed: dict) -> dict:
    """Both arms of the paired control, per horizon, with CIs, plus the lambda sweep."""
    horizons = list(params.markout_horizons)
    result = compare_endogenous_arms(
        params=params, impact=IMPACT, n_episodes=N_EPISODES, seed_base=SEED_BASE
    )
    per = result["per_episode"]

    # Gate: the comparison's exogenous arm is the committed scenario, episode for episode.
    for leg in ("informed", "uninformed"):
        for h in horizons:
            got = per["exogenous"][leg]["markout_per_unit"][h]
            want = [row[str(h)] for row in committed[leg]]
            if got != want:
                raise SystemExit(f"exogenous arm drifted from committed vectors: {leg} h={h}")

    arms = {}
    for arm in ("exogenous", "endogenous"):
        arms[arm] = {
            leg: {
                key: {str(h): _stats_t(per[arm][leg][key][h]) for h in horizons}
                for key in ("markout_per_unit", "toxic_fill_rate")
            }
            for leg in ("informed", "uninformed")
        }
        arms[arm]["gap_per_unit"] = {
            str(h): _stats_t(per[arm]["gap_per_unit"][h]) for h in horizons
        }
        arms[arm]["informed_displacement"] = _stats_t(per[arm]["informed_displacement"])
        arms[arm]["uninformed_displacement"] = _stats_t(per[arm]["uninformed_displacement"])

    # Same episodes under both arms make these direct paired changes the relevant quantity
    # for endogenous-versus-exogenous comparison, rather than a visual comparison of two
    # separately estimated level intervals.
    paired_endogenous_minus_exogenous = {}
    for leg in ("informed", "uninformed"):
        paired_endogenous_minus_exogenous[leg] = {
            key: {
                str(h): _stats_t([
                    endo - exo
                    for endo, exo in zip(
                        per["endogenous"][leg][key][h], per["exogenous"][leg][key][h]
                    )
                ])
                for h in horizons
            }
            for key in ("markout_per_unit", "toxic_fill_rate")
        }
    paired_endogenous_minus_exogenous["gap_per_unit"] = {
        str(h): _stats_t([
            endo - exo
            for endo, exo in zip(
                per["endogenous"]["gap_per_unit"][h],
                per["exogenous"]["gap_per_unit"][h],
            )
        ])
        for h in horizons
    }

    h_max = max(horizons)
    sweep = []
    for lam in LAMBDA_SWEEP:
        imp = EndogenousImpact(kyle_lambda=lam, volume_scale=IMPACT.volume_scale)
        r = compare_endogenous_arms(
            params=params, impact=imp, n_episodes=N_EPISODES, seed_base=SEED_BASE
        )
        e = r["per_episode"]["endogenous"]
        sweep.append(
            {
                "kyle_lambda": lam,
                "impact_per_unit_at_s0": imp.impact_per_unit(params.mm.s0),
                "informed_markout_per_unit": _stats_t(e["informed"]["markout_per_unit"][h_max]),
                "uninformed_markout_per_unit": _stats_t(
                    e["uninformed"]["markout_per_unit"][h_max]
                ),
                "gap_per_unit": _stats_t(e["gap_per_unit"][h_max]),
                "informed_toxic_fill_rate": _stats_t(e["informed"]["toxic_fill_rate"][h_max]),
                "uninformed_toxic_fill_rate": _stats_t(
                    e["uninformed"]["toxic_fill_rate"][h_max]
                ),
                "informed_displacement": _stats_t(e["informed_displacement"]),
                "inference": (
                    "exploratory pointwise t-based 95% intervals across six lambda values; "
                    "no multiplicity-adjusted sweep claim"
                ),
            }
        )

    return {
        "design": (
            "Same 24 paired episodes (seeds 0-23), informed vs uninformed, run with the "
            "price path exogenous (the committed F6 scenario) and endogenous: the bar's "
            "filled taker flow Q_t updates the clearing engine's permanent-impact "
            "multiplier M_{t+1} = M_t (1 + lambda Q_t / V) and makers quote around "
            "mid_t = S_t M_t, S_t the efficient path shared by both legs and both arms. "
            "Temporary impact is the maker ladder itself. Per-episode values are episode "
            "markout over episode filled quantity (all makers); toxic-fill rate is the "
            "share of all maker fills with negative markout at the horizon."
        ),
        "config": {
            "n_episodes": N_EPISODES,
            "seed_base": SEED_BASE,
            "kyle_lambda": IMPACT.kyle_lambda,
            "volume_scale": IMPACT.volume_scale,
            "impact_per_unit_at_s0": IMPACT.impact_per_unit(params.mm.s0),
            "lambda_sweep": list(LAMBDA_SWEEP),
            "sweep_horizon": h_max,
        },
        "horizons": horizons,
        "arms": arms,
        "informed_is_worse": {
            arm: {str(h): v for h, v in d.items()}
            for arm, d in result["informed_is_worse"].items()
        },
        "makers_profit_against_informed_flow": {
            arm: {str(h): v for h, v in d.items()}
            for arm, d in result["makers_profit_against_informed_flow"].items()
        },
        "lambda_sweep": sweep,
        "paired_endogenous_minus_exogenous": paired_endogenous_minus_exogenous,
        "exogenous_arm_matches_committed_vectors": True,
        "ci_convention": (
            "t-based 95% over 24 per-episode values (df=23) for every cell; gap CIs over "
            "per-episode paired gaps. Lambda-sweep intervals are descriptive pointwise intervals."
        ),
    }


def endogenous_figure(block: dict) -> None:
    """Left: markout per filled unit by horizon, both legs, both arms (hatched is the
    endogenous path), with 95% CIs. Right: the lambda sweep at the longest horizon."""
    horizons = [str(h) for h in block["horizons"]]
    arms = block["arms"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 4))

    width = 0.2
    xs = list(range(len(horizons)))
    spec = [
        ("exogenous", "informed", "C0", None, "informed, exogenous path"),
        ("exogenous", "uninformed", "C1", None, "uninformed, exogenous path"),
        ("endogenous", "informed", "C0", "//", "informed, endogenous path"),
        ("endogenous", "uninformed", "C1", "//", "uninformed, endogenous path"),
    ]
    for k, (arm, leg, color, hatch, label) in enumerate(spec):
        cells = [arms[arm][leg]["markout_per_unit"][h] for h in horizons]
        means = [c["mean"] for c in cells]
        err = [
            [c["mean"] - c["ci95_lo"] for c in cells],
            [c["ci95_hi"] - c["mean"] for c in cells],
        ]
        ax.bar(
            [x + (k - 1.5) * width for x in xs],
            means,
            width=width * 0.9,
            color=color,
            alpha=0.55 if hatch else 0.9,
            hatch=hatch,
            edgecolor=color,
            yerr=err,
            capsize=2,
            error_kw={"linewidth": 0.8},
            label=label,
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(horizons)
    ax.set_xlabel("markout horizon (bars)")
    ax.set_ylabel("maker markout per filled unit (price units)")
    ax.set_title(f"lambda = {block['config']['kyle_lambda']}, V = {block['config']['volume_scale']:.0f}")
    ax.legend(frameon=False, fontsize=8)

    lams = [row["kyle_lambda"] for row in block["lambda_sweep"]]
    for key, color, label in (
        ("informed_markout_per_unit", "C0", "informed"),
        ("uninformed_markout_per_unit", "C1", "uninformed"),
    ):
        mean = [row[key]["mean"] for row in block["lambda_sweep"]]
        lo = [row[key]["ci95_lo"] for row in block["lambda_sweep"]]
        hi = [row[key]["ci95_hi"] for row in block["lambda_sweep"]]
        bx.plot(lams, mean, color=color, marker="o", markersize=4, linewidth=1.5, label=label)
        bx.fill_between(lams, lo, hi, color=color, alpha=0.2, linewidth=0)
    bx.axhline(0.0, color="black", linewidth=0.8)
    bx.axvline(
        block["config"]["kyle_lambda"], color="gray", linewidth=0.8, linestyle="--"
    )
    bx.set_xlabel("Kyle lambda (permanent impact per unit dimensionless flow)")
    bx.set_ylabel(f"markout per filled unit at h = {block['config']['sweep_horizon']}")
    bx.set_title("endogenous path, exploratory pointwise 95% CI bands")
    bx.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "f6-endogenous.pdf")
    plt.close(fig)


def main() -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    params = AdverseSelectionParams()
    comparison = compare_informed_vs_uninformed(
        params=params, n_episodes=N_EPISODES, seed_base=SEED_BASE
    )

    detail = run_adverse_selection(params=params, seed=DETAIL_SEED)
    per_maker = [asdict(m) for m in detail.makers]

    # Per-episode markout per unit, per leg, per horizon (t-based 95% CI, df = 23).
    horizons = list(params.markout_horizons)
    t_crit = 2.069

    def _episode_per_unit(informed: bool, seed: int) -> dict[int, float]:
        report = run_adverse_selection(
            params=replace(params, informed=informed), seed=seed
        )
        qty = sum(m.filled_qty for m in report.makers)
        return {
            h: sum(m.markout[h] for m in report.makers) / qty for h in horizons
        }

    per_episode = {"informed": [], "uninformed": [], "gap": []}
    for i in range(N_EPISODES):
        inf = _episode_per_unit(True, SEED_BASE + i)
        uni = _episode_per_unit(False, SEED_BASE + i)
        per_episode["informed"].append({str(h): inf[h] for h in horizons})
        per_episode["uninformed"].append({str(h): uni[h] for h in horizons})
        per_episode["gap"].append({str(h): uni[h] - inf[h] for h in horizons})

    def _stats(values: list[float]) -> dict:
        n = len(values)
        mean = sum(values) / n
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
        half = t_crit * std / math.sqrt(n)
        return {
            "mean": mean,
            "std": std,
            "ci95_lo": mean - half,
            "ci95_hi": mean + half,
            "excludes_zero": mean - half > 0.0 or mean + half < 0.0,
        }

    gap_stats = {
        str(h): _stats([row[str(h)] for row in per_episode["gap"]]) for h in horizons
    }

    out = {
        "finding": "F6",
        "config": {"n_episodes": N_EPISODES, "seed_base": SEED_BASE, "detail_seed": DETAIL_SEED},
        "comparison": {
            "n_episodes": comparison["n_episodes"],
            "horizons": list(comparison["horizons"]),
            "informed_markout_per_unit": {
                str(h): v for h, v in comparison["informed_markout_per_unit"].items()
            },
            "uninformed_markout_per_unit": {
                str(h): v for h, v in comparison["uninformed_markout_per_unit"].items()
            },
            "gap_per_unit": {str(h): v for h, v in comparison["gap_per_unit"].items()},
            "informed_is_worse": {
                str(h): v for h, v in comparison["informed_is_worse"].items()
            },
        },
        "detail_episode_makers": per_maker,
        "per_episode_markout_per_unit": per_episode,
        "gap_stats": gap_stats,
        "ci_convention": (
            "t-based 95% over per-episode paired gaps (episode markout over "
            "episode filled quantity), df=23"
        ),
    }
    out["endogenous"] = endogenous_block(params, per_episode)
    (EVIDENCE / "f6-adverse-selection.json").write_text(json.dumps(out, indent=2, default=str))
    endogenous_figure(out["endogenous"])

    # Figure: markout per filled unit by horizon, informed vs uninformed legs.
    horizons = list(comparison["horizons"])
    informed = [comparison["informed_markout_per_unit"][h] for h in horizons]
    uninformed = [comparison["uninformed_markout_per_unit"][h] for h in horizons]
    fig, ax = plt.subplots(figsize=(6, 4))
    width = 0.35
    xs = range(len(horizons))
    ax.bar([x - width / 2 for x in xs], informed, width=width, label="informed flow")
    ax.bar([x + width / 2 for x in xs], uninformed, width=width, label="uninformed flow")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([str(h) for h in horizons])
    ax.set_xlabel("markout horizon (steps)")
    ax.set_ylabel("maker markout per filled unit")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "f6-markouts.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
