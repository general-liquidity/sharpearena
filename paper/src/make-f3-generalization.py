#!/usr/bin/env python3
"""F3: generalization gap and the cross-regime transfer matrix, with seed bootstrap.

``generalization_gap`` scores the default reference policy on disjoint train and
held-out seed bands within each tier; ``cross_regime_transfer`` holds the seed
band fixed and varies the regime, producing the full tier-by-tier matrix (the
diagonal is 0 by construction). Both public APIs report pooled aggregates only,
so the script additionally rolls the same equal-weight reference per seed
through the public ``SharpeArenaEnv`` and ``score_run`` to serialize per-seed
return series and seed-resampled bootstrap 95% CIs: per-band pooled DSR CIs for
the within-tier gaps, and seed-paired resampling for each transfer-matrix cell
(the same resampled seed indices feed both regimes, cancelling shared path
luck). Writes JSON plus a transfer-matrix heatmap.
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sharpearena import (
    SharpeArenaEnv,
    check_env_effective_config,
    cross_regime_transfer,
    generalization_gap,
    merge_effective_configs,
    score_run,
)

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
MAX_STEPS = 512
N_BOOT = 2000
BOOT_SEED = 0
ALPHA = 0.05


# Every environment in this experiment is built here, so this is where the
# configuration each arm actually ran is read back out of it and checked against what
# was asked for. A mismatch raises and the arm never reaches the evidence file: a
# silently inverted tier survives fixed seeds, goldens and provenance digests intact,
# so nothing else in the pipeline would notice.
_READBACK: dict[str, dict[int, dict]] = {tier: {} for tier in TIERS}


def _make_env(seed: int, mode: str) -> SharpeArenaEnv:
    env = SharpeArenaEnv(
        n_symbols=N_SYMBOLS, n_days=N_DAYS, seed=seed, distribution_mode=mode
    )
    seen = _READBACK.setdefault(mode, {})
    if seed not in seen:
        seen[seed] = check_env_effective_config(
            env,
            seed=seed,
            n_symbols=N_SYMBOLS,
            n_days=N_DAYS,
            distribution_mode=mode,
        )
    return env


def _equal_weight(obs: dict) -> np.ndarray:
    n = int(np.asarray(obs["closes"]).reshape(-1).shape[0])
    return np.full((n,), 1.0 / n, dtype=np.float32)


def _rollout(seed: int, mode: str) -> list[float]:
    env = _make_env(seed, mode)
    obs, _ = env.reset()
    out: list[float] = []
    for _ in range(MAX_STEPS):
        obs, reward, terminated, truncated, _info = env.step(_equal_weight(obs))
        out.append(float(reward))
        if bool(terminated) or bool(truncated):
            break
    return out


def _pooled_dsr(series: list[list[float]]) -> float:
    pooled = [r for s in series for r in s]
    return float(json.loads(score_run(pooled, 0)).get("deflated_sharpe", 0.0))


def _band_ci(args: tuple[int, list[list[float]]]) -> dict:
    """Percentile bootstrap CI on the pooled DSR, resampling seeds.

    Each CI job owns an independent RNG stream seeded (BOOT_SEED, job id), so
    the set of intervals is deterministic and jobs can run in parallel.
    """
    job_id, series = args
    rng = np.random.default_rng((BOOT_SEED, job_id))
    n = len(series)
    draws = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        draws.append(_pooled_dsr([series[i] for i in idx]))
    lo, hi = np.quantile(draws, [ALPHA / 2, 1 - ALPHA / 2])
    return {"point": _pooled_dsr(series), "lo": float(lo), "hi": float(hi)}


def _paired_gap_ci(args: tuple[int, list[list[float]], list[list[float]]]) -> dict:
    """Seed-paired bootstrap CI on DSR(a) - DSR(b) over a shared seed band."""
    job_id, a, b = args
    rng = np.random.default_rng((BOOT_SEED, job_id))
    n = len(a)
    draws = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        draws.append(
            _pooled_dsr([a[i] for i in idx]) - _pooled_dsr([b[i] for i in idx])
        )
    lo, hi = np.quantile(draws, [ALPHA / 2, 1 - ALPHA / 2])
    return {"point": _pooled_dsr(a) - _pooled_dsr(b), "lo": float(lo), "hi": float(hi)}


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

    # Per-seed return series for the dispersion layer (same policy, same envs).
    train_seeds = list(range(N_TRAIN))
    test_seeds = list(range(N_TRAIN + SEED_GAP, N_TRAIN + SEED_GAP + N_TEST))
    series = {
        tier: {
            "train": [_rollout(s, tier) for s in train_seeds],
            "test": [_rollout(s, tier) for s in test_seeds],
        }
        for tier in TIERS
    }
    # The transfer band is the train band, per TRANSFER_SEEDS above.
    transfer_series = {tier: series[tier]["train"] for tier in TIERS}

    band_jobs = [
        (tier, band) for tier in TIERS for band in ("train", "test")
    ]
    pair_jobs = [(a, b) for a in TIERS for b in TIERS]
    with ProcessPoolExecutor(max_workers=len(band_jobs) + len(pair_jobs)) as pool:
        band_results = list(
            pool.map(
                _band_ci,
                [
                    (i, series[tier][band])
                    for i, (tier, band) in enumerate(band_jobs)
                ],
            )
        )
        pair_results = list(
            pool.map(
                _paired_gap_ci,
                [
                    (100 + i, transfer_series[a], transfer_series[b])
                    for i, (a, b) in enumerate(pair_jobs)
                ],
            )
        )
    band_cis: dict[str, dict] = {tier: {} for tier in TIERS}
    for (tier, band), res in zip(band_jobs, band_results):
        band_cis[tier][band] = res
    transfer_cis = {
        f"{a}->{b}": res for (a, b), res in zip(pair_jobs, pair_results)
    }

    # The bootstrap layer must reproduce the API's pooled points.
    for tier in TIERS:
        assert abs(band_cis[tier]["train"]["point"] - gaps[tier]["train"]["deflated_sharpe"]) < 1e-9
        assert abs(band_cis[tier]["test"]["point"] - gaps[tier]["test"]["deflated_sharpe"]) < 1e-9
    for key, cell in matrix.items():
        assert abs(transfer_cis[key]["point"] - cell["transfer_gap_deflated_sharpe"]) < 1e-9

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
            "bootstrap": {
                "n_boot": N_BOOT,
                "resample_seed": BOOT_SEED,
                "alpha": ALPHA,
                "convention": (
                    "percentile bootstrap over resampled seeds; transfer cells "
                    "are seed-paired (shared indices across regimes); one RNG "
                    "stream per CI job, seeded (resample_seed, job id)"
                ),
            },
        },
        "effective_config": {
            tier: merge_effective_configs(_READBACK[tier]) for tier in TIERS
        },
        "generalization_gap": gaps,
        "cross_regime_transfer": matrix,
        "per_seed_returns": series,
        "band_dsr_ci": band_cis,
        "transfer_gap_ci": transfer_cis,
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
