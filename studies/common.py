"""Shared plumbing for the study producers: resumable jobs, manifests and interval maths.

Everything that decides a result is deterministic. Wall time is kept out of the job files
so two runs of one commit can be compared byte for byte; it goes to ``timing/``.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from scipy import stats

REPO = Path(__file__).resolve().parents[1]


class StudyRefused(RuntimeError):
    """A precondition of the protocol does not hold, so nothing was run."""


def load_protocol(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + "\n"


def write_json(path: Path, value: Any) -> None:
    """Write through a sibling temp file, so a killed job never leaves a partial result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = canonical(value)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def band(start: int, count: int) -> list[int]:
    if count < 0:
        raise StudyRefused(f"negative band size {count}")
    return list(range(start, start + count))


def require_disjoint(bands: dict[str, Iterable[int]]) -> None:
    """Refuse overlapping seed bands, naming the pair, whatever the caller meant."""
    items = [(name, set(seeds)) for name, seeds in bands.items()]
    for i, (a, sa) in enumerate(items):
        for b, sb in items[i + 1 :]:
            shared = sa & sb
            if shared:
                raise StudyRefused(f"bands {a} and {b} share seeds, first {min(shared)}")


def git_head() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
    )
    return out.stdout.strip()


def tree_dirty() -> bool:
    """True when a tracked file differs from HEAD. Run outputs are untracked and ignored."""
    out = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    return bool(out.stdout.strip())


def require_clean_tree(stage: str) -> None:
    if tree_dirty():
        raise StudyRefused(f"stage {stage} needs a clean tree: commit first")


def package_versions(names: Sequence[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    return versions


def manifest(protocol_id: str, version: int, sha: str, stage: str, extra: dict) -> dict:
    from sharpearena import engine_spec_hash
    import sharpearena

    spec = engine_spec_hash(sharpearena._engine_module)
    return {
        "protocol_id": protocol_id,
        "protocol_version": version,
        "protocol_sha256": sha,
        "stage": stage,
        "git_head": git_head(),
        "spec_hash": spec,
        "packages": package_versions(
            ["sharpearena", "numpy", "scipy", "stable-baselines3", "torch", "gymnasium", "pettingzoo"]
        ),
        **extra,
    }


# -- jobs ---------------------------------------------------------------------


@dataclass(frozen=True)
class Job:
    """One resumable unit. ``key`` is its relative output path without a suffix."""

    key: str
    payload: dict


def job_path(out: Path, key: str) -> Path:
    return out / "jobs" / f"{key}.json"


def _run_one(func: Callable[[dict], dict], job: Job, out: Path) -> tuple[str, str, float]:
    started = time.perf_counter()
    try:
        result = {"status": "ok", "key": job.key, "result": func(job.payload)}
    except Exception as exc:  # noqa: BLE001 - a failed job stays in the denominator
        result = {
            "status": "failed",
            "key": job.key,
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
            "trace_tail": traceback.format_exc()[-800:],
        }
    write_json(job_path(out, job.key), result)
    elapsed = time.perf_counter() - started
    write_json(out / "timing" / f"{job.key}.json", {"seconds": elapsed, "pid": os.getpid()})
    return job.key, result["status"], elapsed


def run_jobs(
    jobs: Sequence[Job],
    func: Callable[[dict], dict],
    out: Path,
    workers: int,
    *,
    retry_failed: bool = False,
) -> dict[str, int]:
    """Run every job without a finished file. An ok file is never recomputed.

    A failed file stays unless ``retry_failed`` is set, because a failure is a result of
    the run until someone records that it was infrastructure.
    """
    keys = [job.key for job in jobs]
    if len(set(keys)) != len(keys):
        raise StudyRefused("duplicate job keys")
    pending = []
    skipped = 0
    for job in jobs:
        path = job_path(out, job.key)
        if path.exists():
            status = read_json(path).get("status")
            if status == "ok" or (status == "failed" and not retry_failed):
                skipped += 1
                continue
        pending.append(job)
    counts = {"ok": 0, "failed": 0, "skipped": skipped}
    if not pending:
        return counts
    if workers <= 1:
        for job in pending:
            _, status, _ = _run_one(func, job, out)
            counts[status] += 1
        return counts
    with ProcessPoolExecutor(max_workers=min(workers, len(pending))) as pool:
        futures = [pool.submit(_run_one, func, job, out) for job in pending]
        for future in as_completed(futures):
            _, status, _ = future.result()
            counts[status] += 1
    return counts


def collect(out: Path, keys: Sequence[str]) -> tuple[dict[str, dict], list[dict]]:
    """Finished results by key, and every job that is failed or missing, typed."""
    results: dict[str, dict] = {}
    absent: list[dict] = []
    for key in keys:
        path = job_path(out, key)
        if not path.exists():
            absent.append({"key": key, "status": "missing"})
            continue
        record = read_json(path)
        if record["status"] == "ok":
            results[key] = record["result"]
        else:
            absent.append(
                {"key": key, "status": "failed", "error_type": record.get("error_type")}
            )
    return results, absent


def core_seconds(out: Path) -> float:
    return sum(read_json(p)["seconds"] for p in (out / "timing").rglob("*.json"))


# -- intervals ----------------------------------------------------------------


def cp_lower(successes: int, trials: int, confidence: float) -> float:
    """One-sided Clopper-Pearson lower bound."""
    if trials <= 0:
        raise StudyRefused("a rate with no trials has no bound")
    if successes == 0:
        return 0.0
    return float(stats.beta.ppf(1.0 - confidence, successes, trials - successes + 1))


def cp_upper(successes: int, trials: int, confidence: float) -> float:
    """One-sided Clopper-Pearson upper bound."""
    if trials <= 0:
        raise StudyRefused("a rate with no trials has no bound")
    if successes == trials:
        return 1.0
    return float(stats.beta.ppf(confidence, successes + 1, trials - successes))


def t_interval(values: Sequence[float], confidence: float) -> dict:
    """Two-sided t interval, or a typed unavailable below two finite values."""
    clean = [float(v) for v in values]
    if len(clean) < 2 or not all(math.isfinite(v) for v in clean):
        return {"unavailable": "fewer_than_two_finite_values", "n": len(clean)}
    n = len(clean)
    mean = math.fsum(clean) / n
    sd = math.sqrt(math.fsum((v - mean) ** 2 for v in clean) / (n - 1))
    half = float(stats.t.ppf(0.5 + confidence / 2.0, n - 1)) * sd / math.sqrt(n)
    return {"mean": mean, "lo": mean - half, "hi": mean + half, "half": half, "sd": sd, "n": n}


def holm(pvalues: dict[str, float], alpha: float) -> dict[str, bool]:
    ordered = sorted(pvalues, key=lambda k: (pvalues[k], k))
    rejected: dict[str, bool] = {}
    still = True
    for i, key in enumerate(ordered):
        still = still and pvalues[key] <= alpha / (len(ordered) - i)
        rejected[key] = still
    return rejected
