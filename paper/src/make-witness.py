#!/usr/bin/env python3
"""Rank-eligibility existence witness: is the SharpeBench acceptance region non-empty here?

The baseline board (F1) reports that no reference policy is rank-eligible on any tier
and that even ``flat`` fails the per-run gate on all 16 seeds. The devil's advocate asked
the right question: is eligibility attainable at all on this task family, or is the
number an entrant has to beat unbeatable? This script answers by construction, the way
SharpeBench's own pass witness does: inject a KNOWN edge of controlled strength and find
the strength at which the gates first open.

THE ORACLE IS OUT-OF-BAND AND IS NOT AN ENTRANT. The generator is deterministic, so the
true next-bar return of every symbol is recoverable in-script by replaying the same seed
one step ahead with flat decisions (the tape is policy-invariant in the canonical
position-trading environment; asserted below). No entrant can do this through the
observation interface (see the predictability probe for what an honest adversary
recovers). The oracle exists only to place a signal of known strength ``s`` in the
policy's hands:

    signal_t = s * z_t + sqrt(1 - s^2) * eps_t,   z_t = standardized true next-bar return,
                                                  eps_t ~ N(0, 1), seeded,

so ``s`` is the correlation between the signal and the truth (``s = 0`` is a pure noise
trader, ``s = 1`` is omniscience). Two policy variants consume the signal. ``sign_follow``
trades sign(signal) at 1/n gross per symbol every bar, the momentum baseline's exposure
convention; it pays the round-trip cost of flipping every bar, which on Calm exceeds
even the oracle's edge. ``deadband_hold`` trades sign(signal) only when |signal| > 1
and otherwise holds its previous position, so it spends turnover only on strong
signals. Both are reported; the first is the canonical family, the second shows what a
turnover-aware entrant would face. Every strength is scored
EXACTLY as the baselines are: per-seed ``score_run`` for the per-run gate (pass^k), the
pooled series through ``score_run`` for the deflated Sharpe, the bootstrap p-value and
the kernel's own ``rank_eligible`` flag, and a seed-paired bootstrap CI on the pooled
DSR, with the F1 deflation footprint (``n_trials = len(BASELINE_POLICIES)``).

Eligibility is the paper's rule: the per-run gate passes on every seed (pass^k rate
exactly 1.00) AND the kernel's ``rank_eligible`` conjunction holds on the pooled run
(DSR >= 0.95, bootstrap p < 0.05, process and mandate clean). After the coarse sweep,
the boundary is refined by bisection to a resolution of 0.005 in ``s`` and the gate(s)
still failing on the ineligible side of the boundary are named: that is the binding gate.

Bands: the primary band is the canonical held-out band (16 seeds, 10,000-seed gap from
the train band, as in F3); the F1 table band (seeds 0-15) is rerun as a cross-check so
the reader can place the witness next to Table 1. Tiers: Calm, Hard, Extreme.

Noise replication (audit M-13). A threshold located under one ``eps`` path is a
functional of that draw, and the bisection bracket is numerical resolution, not a
statistical interval. So every (variant, band, tier) cell is rerun under
``N_NOISE_REPS`` independent noise paths: replicate 0 is the primary path serialized
under ``boundaries`` (byte-identical to the single-path run), replicates 1.. redraw
``eps`` (still common across strengths within a replicate) and repeat the coarse sweep,
the monotonicity check and the bisection. ``noise_replicates`` records every crossing
bracket and its midpoint, with mean, min, max over replicates, and names any replicate
on which eligibility is never attained.

Writes ``paper/evidence/witness.json`` and ``paper/figures/witness.pdf``.
"""
from __future__ import annotations

import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sharpearena import SharpeArenaEnv, score_run
from sharpearena.baselines import BASELINE_POLICIES
from sharpearena.confidence import deflated_sharpe_ci
from sharpearena.generalization import train_test_seeds

PAPER = Path(__file__).resolve().parents[1]
EVIDENCE = PAPER / "evidence"
FIGURES = PAPER / "figures"

TIERS = ("calm", "hard", "extreme")
N_SYMBOLS = 4
N_DAYS = 120
MAX_STEPS = 512
N_SEEDS = 16
SEED_GAP = 10_000
# The F1 deflation footprint: declared in-sample search breadth of the baseline board.
N_TRIALS = len(BASELINE_POLICIES)
N_BOOT = 2000
RESAMPLE_SEED = 0x5BA7_2026
# Coarse strength grid, then bisection to this resolution.
STRENGTHS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8, 1.0)
BISECT_RESOLUTION = 0.005
# Independent eps paths per (variant, band, tier). Replicate 0 is the primary path.
N_NOISE_REPS = 5
NOISE_REP_STRIDE = 0x2545_F491
# Kernel gate thresholds (sharpebench-core ScoreConfig defaults), restated for the report.
KERNEL_GATES = {
    "per_run_psr_bar": 0.90,
    "per_run_benchmark_annual_sharpe": 0.0,
    "dsr_bar": 0.95,
    "alpha": 0.05,
}

ORACLE_DISCLOSURE = (
    "OUT-OF-BAND ORACLE. The true next-bar return is obtained by replaying the "
    "deterministic generator one step ahead in-script. It is not available through the "
    "observation interface, it is not an entrant, it is not a baseline, and it is not "
    "scored on any leaderboard. It exists only to inject a known edge of controlled "
    "strength so the acceptance region's boundary can be located."
)

TRAIN_SEEDS, HELD_OUT_SEEDS = train_test_seeds(N_SEEDS, N_SEEDS, 0, SEED_GAP)
BANDS = {"held_out": HELD_OUT_SEEDS, "f1_table": TRAIN_SEEDS}
# Policy variants: (deadband on |signal|, hold previous position below the deadband).
VARIANTS = {"sign_follow": (0.0, False), "deadband_hold": (1.0, True)}


def _make_env(seed: int, tier: str) -> SharpeArenaEnv:
    return SharpeArenaEnv(
        n_symbols=N_SYMBOLS, n_days=N_DAYS, seed=seed, distribution_mode=tier
    )


def replay_tape(seed: int, tier: str) -> np.ndarray:
    """The out-of-band channel: the full close panel for ``seed`` under flat decisions.
    ``(n_bars, n_symbols)``; row ``t`` is the observation the policy sees at step ``t``."""
    env = _make_env(seed, tier)
    obs, _ = env.reset()
    rows = [np.asarray(obs["closes"], dtype=np.float64)]
    for _ in range(MAX_STEPS):
        obs, _r, term, trunc, _info = env.step(np.zeros(N_SYMBOLS, dtype=np.float32))
        rows.append(np.asarray(obs["closes"], dtype=np.float64))
        if term or trunc:
            break
    return np.vstack(rows)


def _noise_seed(seed: int, variant: str, noise_rep: int = 0) -> int:
    # Common random numbers across strengths: changing s mixes the same eps path
    # with the same truth path, rather than silently changing the experiment.
    # noise_rep = 0 reproduces the single-path run exactly.
    variant_offset = 0 if variant == "sign_follow" else 0x9E37_79B9
    return (int(seed) * 1_000_003 + variant_offset + noise_rep * NOISE_REP_STRIDE) % (2**32)


def rollout(
    seed: int, tier: str, strength: float, variant: str, noise_rep: int = 0
) -> dict:
    """One episode of the mixed-signal policy. Returns the per-bar reward series, the
    realized directional accuracy of the signal, and a tape-invariance flag."""
    deadband, hold = VARIANTS[variant]
    tape = replay_tape(seed, tier)
    rets = tape[1:] / tape[:-1] - 1.0  # rets[t] is what the action at step t earns
    sigma = rets.std(axis=0, ddof=1)
    sigma = np.where(sigma > 0.0, sigma, 1.0)
    z = rets / sigma
    rng = np.random.default_rng(_noise_seed(seed, variant, noise_rep))
    eps = rng.standard_normal(rets.shape)
    signal = strength * z + math.sqrt(1.0 - strength * strength) * eps

    env = _make_env(seed, tier)
    obs, _ = env.reset()
    rewards: list[float] = []
    invariant = True
    hits = 0
    t = 0
    prev = np.zeros(N_SYMBOLS)
    for _ in range(MAX_STEPS):
        if t < len(signal):
            strong = np.abs(signal[t]) > deadband
            fresh = np.where(signal[t] >= 0.0, 1.0, -1.0) / N_SYMBOLS
            weak = prev if hold else np.zeros(N_SYMBOLS)
            action = np.where(strong, fresh, weak)
            hits += int(np.sum(np.sign(signal[t]) == np.sign(rets[t])))
        else:
            action = np.zeros(N_SYMBOLS)
        prev = action
        obs, reward, term, trunc, _info = env.step(action.astype(np.float32))
        t += 1
        if t < len(tape):
            invariant &= bool(np.array_equal(np.asarray(obs["closes"]), tape[t]))
        rewards.append(float(reward))
        if term or trunc:
            break
    return {
        "returns": rewards,
        "accuracy": hits / float(signal.size),
        "tape_invariant": invariant,
    }


def score_strength(
    seeds: list[int], tier: str, strength: float, variant: str, noise_rep: int = 0
) -> dict:
    """Score one strength exactly like ``run_baselines``: per-seed gate, pooled kernel."""
    per_seed: list[list[float]] = []
    per_seed_pass: list[bool] = []
    per_seed_psr: list[float] = []
    accuracies: list[float] = []
    invariant = True
    for seed in seeds:
        ro = rollout(seed, tier, strength, variant, noise_rep)
        per_seed.append(ro["returns"])
        accuracies.append(ro["accuracy"])
        invariant &= ro["tape_invariant"]
        comp = json.loads(score_run(ro["returns"], N_TRIALS))
        per_seed_pass.append(bool(comp["passed_k"]))
        per_seed_psr.append(float(comp["psr"]))
    pooled = [r for series in per_seed for r in series]
    comp = json.loads(score_run(pooled, N_TRIALS))
    ci = deflated_sharpe_ci(
        per_seed, N_TRIALS, n_boot=N_BOOT, resample_seed=RESAMPLE_SEED
    )
    pass_rate = float(np.mean(per_seed_pass))
    gates = {
        "per_run_gate_all_seeds": pass_rate == 1.0,
        "dsr_bar": float(comp["deflated_sharpe"]) >= KERNEL_GATES["dsr_bar"],
        "bootstrap_alpha": float(comp["bootstrap_p"]) < KERNEL_GATES["alpha"],
        "process_ok": bool(comp["process_ok"]),
        "mandate_ok": bool(comp["mandate_ok"]),
        "kernel_rank_eligible": bool(comp["rank_eligible"]),
    }
    eligible = gates["per_run_gate_all_seeds"] and gates["kernel_rank_eligible"]
    return {
        "strength": strength,
        "eligible": eligible,
        "gates": gates,
        "failing_gates": [k for k, v in gates.items() if not v and k != "kernel_rank_eligible"],
        "pass_k_rate": pass_rate,
        "n_seeds_passing": int(sum(per_seed_pass)),
        "per_seed_passed_k": per_seed_pass,
        "per_seed_psr": per_seed_psr,
        "min_seed_psr": float(min(per_seed_psr)),
        "deflated_sharpe": float(comp["deflated_sharpe"]),
        "deflated_sharpe_ci": ci,
        "psr": float(comp["psr"]),
        "bootstrap_p": float(comp["bootstrap_p"]),
        "mean_return": float(np.mean(pooled)),
        "signal_accuracy": float(np.mean(accuracies)),
        "tape_invariant": invariant,
    }


def locate_boundary(
    seeds: list[int], tier: str, rows: list[dict], variant: str, noise_rep: int = 0
) -> dict:
    """Refine a threshold only when the observed coarse eligibility set is monotone."""
    eligible_idx = [i for i, r in enumerate(rows) if r["eligible"]]
    if not eligible_idx:
        return {
            "attained": False,
            "threshold_identified": False,
            "accepted_strengths": [],
            "note": "no strength in the grid is eligible",
        }
    non_monotone = any(
        rows[i]["eligible"] and not rows[i + 1]["eligible"] for i in range(len(rows) - 1)
    )
    accepted = [r["strength"] for r in rows if r["eligible"]]
    if non_monotone:
        return {
            "attained": True,
            "threshold_identified": False,
            "accepted_strengths": accepted,
            "grid_non_monotone": True,
            "note": "eligibility is non-monotone on the coarse grid; no threshold is reported",
        }
    first = eligible_idx[0]
    if first == 0:
        return {
            "attained": True,
            "threshold_identified": True,
            "lo": None,
            "hi": rows[0]["strength"],
            "accepted_strengths": accepted,
            "points": [],
        }
    lo, hi = rows[first - 1], rows[first]
    points: list[dict] = []
    while hi["strength"] - lo["strength"] > BISECT_RESOLUTION:
        mid = round(0.5 * (lo["strength"] + hi["strength"]), 5)
        row = score_strength(seeds, tier, mid, variant, noise_rep)
        points.append(row)
        if row["eligible"]:
            hi = row
        else:
            lo = row
    return {
        "attained": True,
        "threshold_identified": True,
        "lo": lo["strength"],
        "hi": hi["strength"],
        "binding_gates": lo["failing_gates"],
        "lo_row": {k: lo[k] for k in ("strength", "pass_k_rate", "n_seeds_passing",
                                      "min_seed_psr", "deflated_sharpe", "bootstrap_p",
                                      "signal_accuracy")},
        "hi_row": {k: hi[k] for k in ("strength", "pass_k_rate", "n_seeds_passing",
                                      "min_seed_psr", "deflated_sharpe", "bootstrap_p",
                                      "signal_accuracy")},
        "accepted_strengths": accepted,
        "grid_non_monotone": False,
        "points": points,
    }


SLIM_ROW_KEYS = ("strength", "eligible", "pass_k_rate", "n_seeds_passing", "min_seed_psr",
                 "deflated_sharpe", "bootstrap_p", "signal_accuracy")


def run_cell(task: tuple[str, str, str, int]) -> tuple[tuple[str, str, str, int], dict]:
    """Coarse sweep + boundary for one (variant, band, tier, noise replicate)."""
    variant, band_name, tier, noise_rep = task
    seeds = BANDS[band_name]
    rows = [score_strength(seeds, tier, s, variant, noise_rep) for s in STRENGTHS]
    assert all(r["tape_invariant"] for r in rows), "tape is not policy-invariant"
    boundary = locate_boundary(seeds, tier, rows, variant, noise_rep)
    return task, {"sweep": rows, "boundary": boundary}


def _crossing(boundary: dict) -> float | None:
    """Point estimate of the crossing: the midpoint of the bisection bracket. Its
    half-width is BISECT_RESOLUTION/2 (numerical), separate from the noise range."""
    if not boundary.get("threshold_identified"):
        return None
    if boundary.get("lo") is None:
        return float(boundary["hi"])
    return round(0.5 * (boundary["lo"] + boundary["hi"]), 6)


def summarize_replicates(cells: list[dict]) -> dict:
    """Per-(variant, band, tier) summary over noise replicates."""
    reps = []
    for k, cell in enumerate(cells):
        b = cell["boundary"]
        reps.append({
            "noise_rep": k,
            "attained": bool(b.get("attained")),
            "threshold_identified": bool(b.get("threshold_identified")),
            "grid_non_monotone": bool(b.get("grid_non_monotone", False)),
            "lo": b.get("lo"),
            "hi": b.get("hi"),
            "crossing": _crossing(b),
            "binding_gates": b.get("binding_gates"),
            "accepted_strengths": b.get("accepted_strengths", []),
            "hi_row": b.get("hi_row"),
            "lo_row": b.get("lo_row"),
            "sweep": [{k2: r[k2] for k2 in SLIM_ROW_KEYS} for r in cell["sweep"]],
            "bisection_points": [
                {k2: r[k2] for k2 in SLIM_ROW_KEYS} for r in b.get("points", [])
            ],
        })
    crossings = [r["crossing"] for r in reps if r["crossing"] is not None]
    binding = [tuple(r["binding_gates"]) for r in reps if r["binding_gates"] is not None]
    summary = {
        "n_reps": len(reps),
        "n_attained": sum(r["attained"] for r in reps),
        "n_threshold_identified": len(crossings),
        "unattained_reps": [r["noise_rep"] for r in reps if not r["attained"]],
        "non_monotone_reps": [r["noise_rep"] for r in reps if r["grid_non_monotone"]],
        "crossings": crossings,
        "brackets": [[r["lo"], r["hi"]] for r in reps if r["crossing"] is not None],
        "mean": float(np.mean(crossings)) if crossings else None,
        "min": float(min(crossings)) if crossings else None,
        "max": float(max(crossings)) if crossings else None,
        "range": float(max(crossings) - min(crossings)) if crossings else None,
        "sd_ddof1": float(np.std(crossings, ddof=1)) if len(crossings) > 1 else None,
        "bracket_half_width": BISECT_RESOLUTION / 2.0,
        "binding_gates_consistent": len(set(binding)) <= 1,
        "binding_gates": sorted({g for bg in binding for g in bg}),
    }
    return {"replicates": reps, "summary": summary}


def main() -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    all_noise_seeds = [
        _noise_seed(seed, variant, rep)
        for variant in VARIANTS for seed in {s for b in BANDS.values() for s in b}
        for rep in range(N_NOISE_REPS)
    ]
    assert len(set(all_noise_seeds)) == len(all_noise_seeds), "noise seeds collide"

    tasks = [
        (variant, band_name, tier, rep)
        for variant in VARIANTS
        for band_name in BANDS
        for tier in TIERS
        for rep in range(N_NOISE_REPS)
    ]
    cells: dict[tuple[str, str, str, int], dict] = {}
    workers = max(1, min(len(tasks), os.cpu_count() or 1))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for task, cell in pool.map(run_cell, tasks):
            cells[task] = cell
            b = cell["boundary"]
            print(
                f"{task[0]:14s} {task[1]:9s} {task[2]:8s} rep={task[3]} boundary s in "
                f"[{b.get('lo')}, {b.get('hi')}] binding={b.get('binding_gates')}"
            )

    results: dict[str, dict] = {
        variant: {
            band_name: {tier: cells[(variant, band_name, tier, 0)] for tier in TIERS}
            for band_name in BANDS
        }
        for variant in VARIANTS
    }
    noise_replicates = {
        "config": {
            "n_noise_reps": N_NOISE_REPS,
            "noise_rep_stride": NOISE_REP_STRIDE,
            "noise_seed_rule": (
                "(seed * 1_000_003 + variant_offset + noise_rep * noise_rep_stride) mod 2^32; "
                "noise_rep = 0 is the primary path serialized under 'boundaries'"
            ),
            "crossing_point": (
                "midpoint of the bisection bracket [lo, hi]; half-width = "
                "bisect_resolution / 2 is numerical resolution, separate from the "
                "min..max range over noise replicates"
            ),
            "summary_statistics": "mean, min, max, range, sd (ddof=1) over identified crossings",
        },
        "by_cell": {
            variant: {
                band_name: {
                    tier: summarize_replicates(
                        [cells[(variant, band_name, tier, k)] for k in range(N_NOISE_REPS)]
                    )
                    for tier in TIERS
                }
                for band_name in BANDS
            }
            for variant in VARIANTS
        },
    }
    for variant in VARIANTS:
        for band_name in BANDS:
            for tier in TIERS:
                s = noise_replicates["by_cell"][variant][band_name][tier]["summary"]
                print(
                    f"{variant:14s} {band_name:9s} {tier:8s} crossings={s['crossings']} "
                    f"mean={s['mean']} min={s['min']} max={s['max']} "
                    f"attained={s['n_attained']}/{s['n_reps']} "
                    f"non_monotone={s['non_monotone_reps']}"
                )

    out = {
        "finding": "witness",
        "oracle_disclosure": ORACLE_DISCLOSURE,
        "config": {
            "tiers": list(TIERS),
            "n_symbols": N_SYMBOLS,
            "n_days": N_DAYS,
            "bands": {k: list(v) for k, v in BANDS.items()},
            "seed_gap": SEED_GAP,
            "n_trials": N_TRIALS,
            "n_trials_convention": "F1 footprint: len(BASELINE_POLICIES)",
            "strengths": list(STRENGTHS),
            "bisect_resolution": BISECT_RESOLUTION,
            "kernel_gates": KERNEL_GATES,
            "eligibility_rule": (
                "per-run gate passes on every seed (pass^k rate exactly 1.00) and the "
                "kernel's rank_eligible conjunction holds on the pooled run"
            ),
    "policy": (
                "signal = s*z + sqrt(1-s^2)*eps; z = standardized true next-bar return "
                "(oracle), eps ~ N(0,1) seeded by (seed, variant) and reused at every s; "
                "action = sign(signal)/n "
                "where |signal| > deadband, else flat (sign_follow) or the previous "
                "position (deadband_hold)"
            ),
            "variants": {k: {"deadband": v[0], "hold": v[1]} for k, v in VARIANTS.items()},
            "bootstrap": {
                "n_boot": N_BOOT,
                "resample_seed": RESAMPLE_SEED,
                "alpha": 0.05,
                "convention": "seed-paired percentile bootstrap on the deflated Sharpe",
            },
        },
        "results": results,
        "boundaries": {
            variant: {
                band: {
                    tier: {
                        k: v
                        for k, v in results[variant][band][tier]["boundary"].items()
                        if k != "points"
                    }
                    for tier in TIERS
                }
                for band in BANDS
            }
            for variant in VARIANTS
        },
        "noise_replicates": noise_replicates,
    }
    (EVIDENCE / "witness.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {EVIDENCE / 'witness.json'}")

    # Figure: held-out band, one row per policy variant. Left, pooled DSR with CI against
    # s (filled markers where eligible, boundary marked). Right, pass^k rate against s.
    # The shaded band and the horizontal error bar are the min..max crossing range over
    # the noise replicates; the dashed line is the primary-path bracket.
    colors = {"calm": "#1a73e8", "hard": "#d93025", "extreme": "#5f6368"}
    fig, axes = plt.subplots(len(VARIANTS), 2, figsize=(9.0, 3.1 * len(VARIANTS)))
    for row_i, variant in enumerate(VARIANTS):
        ax_d, ax_p = axes[row_i]
        for tier_i, tier in enumerate(TIERS):
            res = results[variant]["held_out"][tier]
            summ = noise_replicates["by_cell"][variant]["held_out"][tier]["summary"]
            if summ["n_threshold_identified"] > 0:
                for ax in (ax_d, ax_p):
                    ax.axvspan(summ["min"], summ["max"], color=colors[tier], alpha=0.15,
                               linewidth=0)
                ax_p.errorbar(
                    [summ["mean"]], [0.15 + 0.12 * tier_i],
                    xerr=[[summ["mean"] - summ["min"]], [summ["max"] - summ["mean"]]],
                    fmt="|", color=colors[tier], capsize=3, linewidth=1.0, zorder=4,
                )
            rows = sorted(
                res["sweep"] + res["boundary"].get("points", []),
                key=lambda r: r["strength"],
            )
            xs = [r["strength"] for r in rows]
            ys = [r["deflated_sharpe"] for r in rows]
            lo = [r["deflated_sharpe_ci"]["lo"] for r in rows]
            hi = [r["deflated_sharpe_ci"]["hi"] for r in rows]
            ax_d.plot(xs, ys, color=colors[tier], linewidth=1.0, label=tier)
            ax_d.fill_between(xs, lo, hi, color=colors[tier], alpha=0.12)
            el = [r for r in rows if r["eligible"]]
            ne = [r for r in rows if not r["eligible"]]
            ax_d.scatter([r["strength"] for r in ne], [r["deflated_sharpe"] for r in ne],
                         facecolors="white", edgecolors=colors[tier], s=18, zorder=3)
            ax_d.scatter([r["strength"] for r in el], [r["deflated_sharpe"] for r in el],
                         color=colors[tier], s=18, zorder=3)
            b = res["boundary"]
            if b.get("attained") and b.get("hi") is not None:
                ax_d.axvline(b["hi"], color=colors[tier], linestyle="--", linewidth=0.8)
                ax_p.axvline(b["hi"], color=colors[tier], linestyle="--", linewidth=0.8)
            ax_p.plot(xs, [r["pass_k_rate"] for r in rows], color=colors[tier],
                      marker="o", markersize=3, linewidth=1.0, label=tier)
        ax_d.axhline(KERNEL_GATES["dsr_bar"], color="black", linestyle=":", linewidth=0.8)
        ax_d.set_ylabel(f"{variant}\npooled deflated Sharpe")
        ax_d.set_title("filled = rank-eligible; dotted = DSR bar; shaded = noise range",
                       fontsize=9)
        ax_p.axhline(1.0, color="black", linestyle=":", linewidth=0.8)
        ax_p.set_ylabel("pass^k rate (16 held-out seeds)")
        ax_p.set_title(f"bars = crossing mean, min..max over {N_NOISE_REPS} noise paths",
                       fontsize=9)
        ax_p.set_ylim(-0.02, 1.05)
        if row_i == 0:
            ax_d.legend(fontsize=8, frameon=False)
        if row_i == len(VARIANTS) - 1:
            ax_d.set_xlabel("signal strength s (corr. with true next-bar return)")
            ax_p.set_xlabel("signal strength s")
        for ax in (ax_d, ax_p):
            ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "witness.pdf")
    print(f"wrote {FIGURES / 'witness.pdf'}")


if __name__ == "__main__":
    main()
