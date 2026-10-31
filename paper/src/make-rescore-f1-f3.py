#!/usr/bin/env python3
"""Rescore the frozen F1 and F3 return series under the current kernel pin.

``f1-baselines.json`` and ``f3-generalization.json`` predate the corrected
standardized moments and the constant-track refusal. Both serialize the return
series behind every scored cell, so a current-method result is a recomputation:
nothing here steps an environment, and the only inputs are the two committed
files, which this script reads and never writes.

Each output is a new file whose name carries the pinned SharpeBench version, so a
later pin produces a later file beside this one. Three checks stop the run rather
than publish a number:

* a frozen input whose digest is not the one recorded here;
* a return series that does not reproduce the mean return the frozen cell records,
  which is the sufficiency check: the series scored here are the series scored then;
* the manuscript's worked instance, which rescores the Hard train band at six trials
  and must read 0.1114 to the precision the manuscript prints.

A row the kernel will not score reports the kernel's typed reason in the place of
the number. The flat policy's constant track is that case: its deflated Sharpe,
pass^k rate and interval are unavailable, never the frozen 0.0007 and never zero.

Deterministic: no clock, no ambient randomness, sorted keys, one RNG stream per
interval seeded as the frozen producers seeded it.

    python paper/src/make-rescore-f1-f3.py          # write, or confirm the bytes
    python paper/src/make-rescore-f1-f3.py --check  # recompute, write nothing
"""
from __future__ import annotations

import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from sharpearena import EXPECTED_SPEC_HASH, score_run
from sharpearena.baselines import BASELINE_POLICIES, leaderboard_markdown
from sharpearena.confidence import deflated_sharpe_ci
from sharpearena.kernel_score import (
    KernelScoreUnavailable,
    is_kernel_score_unavailable,
    kernel_deflated_sharpe,
    kernel_score_difference,
    kernel_score_or_unavailable,
    seed_pass_tally,
)

from provenance_common import sha256

PAPER = Path(__file__).resolve().parents[1]
REPO = PAPER.parent
EVIDENCE = PAPER / "evidence"

RESCORE_VERSION = 1
TIERS = ("calm", "hard", "extreme")

# The bytes this rescore is a statement about. A changed input is a different
# experiment, so it stops the run instead of quietly moving every number below.
FROZEN_F1 = "f1-baselines.json"
FROZEN_F1_SHA256 = "ec61a9a1f4340d58d497545ce401c159bd1e61201cb7c001773dd66b9e381f67"
FROZEN_F3 = "f3-generalization.json"
FROZEN_F3_SHA256 = "3bff445e13dd4d094a45ebe20e713ec0bc63b06e017c30a399d3aa41a8cf50e1"

# paper/sections/08-findings.tex: "rescoring the committed F3 return series at six
# trials reproduces 0.1114 exactly".
WORKED_INSTANCE_TRIALS = 6
WORKED_INSTANCE_PRINTED = "0.1114"
F3_DECLARED_TRIALS = 0


def _reject_constant(token: str) -> float:
    raise ValueError(f"non-finite token {token} in frozen evidence")


def load_frozen(name: str, expected_sha256: str, root: Path = EVIDENCE) -> dict:
    path = root / name
    actual = sha256(path)
    if actual != expected_sha256:
        raise SystemExit(f"{name} hashes to {actual}, this rescore is bound to {expected_sha256}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)


def kernel_identity(repo: Path = REPO) -> dict:
    """The scorer that produced these numbers, read from what the build locks."""
    manifest = (repo / "crates" / "sharpearena" / "Cargo.toml").read_text(encoding="utf-8")
    declared = re.search(r'^sharpebench-core = "=([^"]+)"', manifest, re.MULTILINE)
    lock = (repo / "Cargo.lock").read_text(encoding="utf-8")
    locked = re.search(r'name = "sharpebench-core"\nversion = "([^"]+)"', lock)
    if declared is None or locked is None or declared.group(1) != locked.group(1):
        raise SystemExit("cannot read one sharpebench-core version from Cargo.toml and Cargo.lock")
    package = (repo / "crates" / "sharpearena-py" / "pyproject.toml").read_text(encoding="utf-8")
    version = re.search(r'^version = "([^"]+)"', package, re.MULTILINE)
    return {
        "sharpebench_pin": declared.group(1),
        "sharpearena_version": version.group(1),
        "spec_hash": EXPECTED_SPEC_HASH,
    }


def _composite(returns: list[float], trials: int) -> dict:
    return json.loads(score_run(returns, trials))


def rescore_policy(
    policy: str,
    per_seed: list[list[float]],
    trials: int,
    *,
    n_boot: int,
    resample_seed: int,
    alpha: float,
) -> dict:
    """One baseline row from recorded series, by the rules of ``run_baselines``.

    Only the rollout is absent. A test holds this function to the row
    ``run_baselines`` writes for the same series, so the two cannot part silently.
    """
    pooled = [value for series in per_seed for value in series]
    # Every seed is scored, as run_baselines does, so an unscorable seed stays in the
    # pass-rate denominator as not passed.
    tally = seed_pass_tally([_composite(series, trials) for series in per_seed])
    composite = _composite(pooled, trials) if len(pooled) >= 2 else {}
    row: dict = {
        "policy": policy,
        "deflated_sharpe": kernel_score_or_unavailable(composite),
        **tally,
        "mean_return": float(np.mean(pooled)) if pooled else 0.0,
    }
    try:
        interval = deflated_sharpe_ci(
            per_seed, trials, n_boot=n_boot, resample_seed=resample_seed, alpha=alpha
        )
        interval_error = None
    except ValueError as error:
        interval = None
        interval_error = f"unavailable_seed_bootstrap: {error}"
    row["arena_deflated_sharpe_ci"] = interval
    if is_kernel_score_unavailable(row["deflated_sharpe"]):
        row["deflated_sharpe_ci"] = None
        row["confidence_status"] = row["deflated_sharpe"]
    elif interval is None:
        row["deflated_sharpe_ci"] = None
        row["confidence_status"] = interval_error
    elif composite and interval["point"] == row["deflated_sharpe"]:
        row["deflated_sharpe_ci"] = interval
        row["confidence_status"] = "scoring_kernel_reproduced"
    else:
        row["deflated_sharpe_ci"] = None
        row["confidence_status"] = "unavailable_scoring_kernel_mismatch"
    return row


def require_series_match(where: str, per_seed: list[list[float]], recorded_mean: float) -> None:
    """Sufficiency: the recorded series must be the series the frozen cell scored."""
    pooled = [value for series in per_seed for value in series]
    mean = float(np.mean(pooled)) if pooled else 0.0
    if mean != recorded_mean:
        raise SystemExit(f"{where}: series mean {mean!r} is not the recorded {recorded_mean!r}")


def rescore_f1(frozen: dict) -> dict:
    bootstrap = frozen["config"]["bootstrap"]
    trials = len(BASELINE_POLICIES)
    tiers: dict[str, dict] = {}
    for tier in TIERS:
        frozen_rows = frozen["tiers"][tier]["rows"]
        if [row["policy"] for row in frozen_rows] != [name for name, _ in BASELINE_POLICIES]:
            raise SystemExit(f"F1 {tier}: the frozen rows are not the declared policy field")
        rows = []
        for old in frozen_rows:
            require_series_match(
                f"F1 {tier} {old['policy']}", old["per_seed_returns"], old["mean_return"]
            )
            new = rescore_policy(
                old["policy"],
                old["per_seed_returns"],
                trials,
                n_boot=bootstrap["n_boot"],
                resample_seed=bootstrap["resample_seed"],
                alpha=bootstrap["alpha"],
            )
            new["frozen"] = {
                "deflated_sharpe": old["deflated_sharpe"],
                "passed_k_rate": old["passed_k_rate"],
                "deflated_sharpe_ci": old.get("deflated_sharpe_ci"),
            }
            new["deflated_sharpe_delta"] = kernel_score_difference(
                new["deflated_sharpe"], old["deflated_sharpe"]
            )
            new["passed_k_rate_delta"] = kernel_score_difference(new["passed_k_rate"], old["passed_k_rate"])
            rows.append(new)
        rates = [row["passed_k_rate"] for row in rows if not isinstance(row["passed_k_rate"], str)]
        tiers[tier] = {
            "rows": rows,
            "leaderboard_markdown": leaderboard_markdown(rows, show_ci=True),
            "best_passed_k_rate": max(rates) if rates else None,
            "n_policies_all_seeds_pass": sum(rate == 1.0 for rate in rates),
            "unavailable_policies": [
                row["policy"]
                for row in rows
                if is_kernel_score_unavailable(row["deflated_sharpe"])
            ],
        }
    return {
        "finding": "F1",
        "declared_trials": trials,
        "bootstrap": bootstrap,
        "tiers": tiers,
    }


def _pooled(series: list[list[float]]) -> list[float]:
    return [value for run in series for value in run]


def _pooled_dsr(series: list[list[float]]) -> float:
    return kernel_deflated_sharpe(_composite(_pooled(series), F3_DECLARED_TRIALS))


def evaluate_band(per_seed: list[list[float]], trials: int) -> dict:
    """``evaluate_seeds`` over recorded series. A test holds the two together."""
    pooled = _pooled(per_seed)
    return {
        "n_seeds": len(per_seed),
        "deflated_sharpe": kernel_score_or_unavailable(_composite(pooled, trials)),
        **seed_pass_tally([_composite(series, trials) for series in per_seed]),
        "mean_return": float(np.mean(pooled)) if pooled else 0.0,
    }


def _withheld(error: KernelScoreUnavailable) -> dict:
    return {"point": error.reason, "lo": error.reason, "hi": error.reason}


def band_interval(args: tuple[int, int, int, float, list[list[float]]]) -> dict:
    """Percentile bootstrap on the pooled DSR, resampling seeds, as the producer did."""
    boot_seed, job_id, n_boot, alpha, series = args
    rng = np.random.default_rng((boot_seed, job_id))
    n = len(series)
    try:
        draws = []
        for _ in range(n_boot):
            indices = rng.integers(0, n, size=n)
            draws.append(_pooled_dsr([series[i] for i in indices]))
        lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
        return {"point": _pooled_dsr(series), "lo": float(lo), "hi": float(hi)}
    except KernelScoreUnavailable as error:
        return _withheld(error)


def paired_gap_interval(args: tuple[int, int, int, float, list, list]) -> dict:
    """Seed-paired bootstrap on DSR(a) - DSR(b), as the producer did."""
    boot_seed, job_id, n_boot, alpha, a, b = args
    rng = np.random.default_rng((boot_seed, job_id))
    n = len(a)
    try:
        draws = []
        for _ in range(n_boot):
            indices = rng.integers(0, n, size=n)
            draws.append(
                _pooled_dsr([a[i] for i in indices]) - _pooled_dsr([b[i] for i in indices])
            )
        lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
        return {"point": _pooled_dsr(a) - _pooled_dsr(b), "lo": float(lo), "hi": float(hi)}
    except KernelScoreUnavailable as error:
        return _withheld(error)


def rescore_f3(frozen: dict) -> dict:
    bootstrap = frozen["config"]["bootstrap"]
    boot_seed, n_boot, alpha = bootstrap["resample_seed"], bootstrap["n_boot"], bootstrap["alpha"]
    series = frozen["per_seed_returns"]
    gaps: dict[str, dict] = {}
    for tier in TIERS:
        old = frozen["generalization_gap"][tier]
        for band in ("train", "test"):
            require_series_match(f"F3 {tier} {band}", series[tier][band], old[band]["mean_return"])
        train = evaluate_band(series[tier]["train"], F3_DECLARED_TRIALS)
        test = evaluate_band(series[tier]["test"], F3_DECLARED_TRIALS)
        gap = kernel_score_difference(train["deflated_sharpe"], test["deflated_sharpe"])
        gaps[tier] = {
            "train": train,
            "test": test,
            "gap_deflated_sharpe": gap,
            "gap_mean_return": train["mean_return"] - test["mean_return"],
            "frozen_gap_deflated_sharpe": old["gap_deflated_sharpe"],
            "gap_deflated_sharpe_delta": kernel_score_difference(gap, old["gap_deflated_sharpe"]),
        }

    # The transfer band is the train band: the frozen producer scored both from the
    # same seeds, and the mean-return check below confirms it cell by cell.
    matrix: dict[str, dict] = {}
    for a in TIERS:
        for b in TIERS:
            key = f"{a}->{b}"
            old = frozen["cross_regime_transfer"][key]
            require_series_match(
                f"F3 {key} in", series[a]["train"], old["in_distribution"]["mean_return"]
            )
            require_series_match(
                f"F3 {key} out", series[b]["train"], old["out_of_distribution"]["mean_return"]
            )
            inside = evaluate_band(series[a]["train"], F3_DECLARED_TRIALS)
            outside = evaluate_band(series[b]["train"], F3_DECLARED_TRIALS)
            gap = kernel_score_difference(inside["deflated_sharpe"], outside["deflated_sharpe"])
            matrix[key] = {
                "train_mode": a,
                "test_mode": b,
                "in_distribution": inside,
                "out_of_distribution": outside,
                "transfer_gap_deflated_sharpe": gap,
                "frozen_transfer_gap_deflated_sharpe": old["transfer_gap_deflated_sharpe"],
                "transfer_gap_deflated_sharpe_delta": kernel_score_difference(
                    gap, old["transfer_gap_deflated_sharpe"]
                ),
            }

    band_jobs = [(tier, band) for tier in TIERS for band in ("train", "test")]
    pair_jobs = [(a, b) for a in TIERS for b in TIERS]
    with ProcessPoolExecutor(max_workers=len(band_jobs) + len(pair_jobs)) as pool:
        band_results = list(
            pool.map(
                band_interval,
                [
                    (boot_seed, i, n_boot, alpha, series[tier][band])
                    for i, (tier, band) in enumerate(band_jobs)
                ],
            )
        )
        pair_results = list(
            pool.map(
                paired_gap_interval,
                [
                    (boot_seed, 100 + i, n_boot, alpha, series[a]["train"], series[b]["train"])
                    for i, (a, b) in enumerate(pair_jobs)
                ],
            )
        )
    band_cis: dict[str, dict] = {tier: {} for tier in TIERS}
    for (tier, band), result in zip(band_jobs, band_results):
        band_cis[tier][band] = {**result, "frozen": frozen["band_dsr_ci"][tier][band]}
    transfer_cis = {}
    for (a, b), result in zip(pair_jobs, pair_results):
        key = f"{a}->{b}"
        transfer_cis[key] = {**result, "frozen": frozen["transfer_gap_ci"][key]}
    return {
        "finding": "F3",
        "declared_trials": F3_DECLARED_TRIALS,
        "bootstrap": bootstrap,
        "generalization_gap": gaps,
        "cross_regime_transfer": matrix,
        "band_dsr_ci": band_cis,
        "transfer_gap_ci": transfer_cis,
    }


def worked_instance(f1: dict, f3: dict) -> dict:
    """The manuscript's instance, run rather than remembered."""
    series = f3["per_seed_returns"]["hard"]["train"]
    long_row = next(
        row for row in f1["tiers"]["hard"]["rows"] if row["policy"] == "equal_weight_long"
    )
    if long_row["per_seed_returns"] != series:
        raise SystemExit("the F1 and F3 Hard series are not the same recorded returns")
    value = kernel_deflated_sharpe(
        _composite(_pooled(series), WORKED_INSTANCE_TRIALS)
    )
    printed = f"{value:.4f}"
    if printed != WORKED_INSTANCE_PRINTED:
        raise SystemExit(f"worked instance reads {printed}, the manuscript prints 0.1114")
    return {
        "statement": (
            "rescoring the committed F3 Hard train series at six trials reproduces the "
            "printed 0.1114"
        ),
        "series": "f3-generalization.json per_seed_returns.hard.train",
        "series_equals_f1_hard_equal_weight_long": True,
        "trials": WORKED_INSTANCE_TRIALS,
        "deflated_sharpe": value,
        "printed": printed,
        "frozen_f1_value": long_row["deflated_sharpe"],
        "delta_from_frozen_f1_value": value - long_row["deflated_sharpe"],
        "agreement": "four displayed decimals; the bits differ because the moments changed",
    }


def envelope(body: dict, source: str, source_sha256: str, kernel: dict, worked: dict) -> dict:
    return {
        "artifact_kind": "rescore",
        "rescore_version": RESCORE_VERSION,
        "kernel": kernel,
        "frozen_input": {"path": f"paper/evidence/{source}", "sha256": source_sha256},
        "method": (
            "recomputation from the frozen file's recorded return series; no environment "
            "was stepped and the frozen file is unchanged"
        ),
        "worked_instance": worked,
        **body,
    }


def render(document: dict) -> str:
    # allow_nan=False: a non-finite value is a bug here, not a number to publish.
    return json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"


def write_versioned(path: Path, text: str) -> str:
    """Create ``path``, or confirm it already holds these bytes. Never replace it."""
    data = text.encode("utf-8")
    if path.exists():
        if path.read_bytes() == data:
            return "unchanged"
        raise SystemExit(
            f"{path.name} exists with other bytes; a rescore is never overwritten. "
            "Bump RESCORE_VERSION to publish a new one beside it."
        )
    path.write_bytes(data)
    return "wrote"


def output_path(name: str, kernel: dict, root: Path = EVIDENCE) -> Path:
    return root / f"{name}.rescore-v{RESCORE_VERSION}-sb{kernel['sharpebench_pin']}.json"


def main(argv: list[str]) -> int:
    check_only = "--check" in argv
    kernel = kernel_identity()
    f1 = load_frozen(FROZEN_F1, FROZEN_F1_SHA256)
    f3 = load_frozen(FROZEN_F3, FROZEN_F3_SHA256)
    worked = worked_instance(f1, f3)
    documents = {
        "f1-baselines": envelope(rescore_f1(f1), FROZEN_F1, FROZEN_F1_SHA256, kernel, worked),
        "f3-generalization": envelope(rescore_f3(f3), FROZEN_F3, FROZEN_F3_SHA256, kernel, worked),
    }
    for name, document in documents.items():
        text = render(document)
        path = output_path(name, kernel)
        if check_only:
            if not path.exists() or path.read_bytes() != text.encode("utf-8"):
                print(f"{path.name} does not hold what the frozen inputs recompute to")
                return 1
            print(f"{path.name}: reproduced")
        else:
            print(f"{write_versioned(path, text)} {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
