"""Shared plumbing for the study producers: resumable jobs, manifests and interval maths.

Everything that decides a result is deterministic. Wall time is kept out of the job files
so two runs of one commit can be compared byte for byte; it goes to ``timing/``.

Every attempt at a job is kept. ``attempts/<key>/NNN.json`` is written once and never
replaced, and its wall time goes to ``timing/<key>/NNN.json``. ``jobs/<key>.json`` holds
the latest attempt's result for collection. A failed key is retried only with a recorded
infrastructure reason, which is stored in the new attempt's record.
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
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

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


def attempt_dir(out: Path, key: str) -> Path:
    return out / "attempts" / key


def attempt_records(out: Path, key: str) -> list[Path]:
    """The key's attempt records in attempt order (``001.json``, ``002.json``, ...)."""
    folder = attempt_dir(out, key)
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.glob("*.json") if p.stem.isdigit())


def write_once(path: Path, value: Any) -> None:
    """Write ``path`` only if it does not exist. An attempt record is never replaced.

    The text goes to a temp file first and is then hard-linked into place, which fails
    rather than overwrites when the name is taken, on POSIX and on NTFS alike.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(canonical(value), encoding="utf-8", newline="\n")
    try:
        os.link(tmp, path)
    except FileExistsError as exc:
        raise StudyRefused(f"attempt record {path} already exists; attempts are append-only") from exc
    finally:
        tmp.unlink()


def _run_one(
    func: Callable[[dict], dict], job: Job, out: Path, attempt: int, retry_reason: Optional[str]
) -> tuple[str, str, float]:
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
    elapsed = time.perf_counter() - started
    record = {**result, "attempt": attempt, "retry_reason": retry_reason}
    write_once(attempt_dir(out, job.key) / f"{attempt:03d}.json", record)
    write_once(
        out / "timing" / job.key / f"{attempt:03d}.json", {"seconds": elapsed, "pid": os.getpid()}
    )
    write_json(job_path(out, job.key), result)
    return job.key, result["status"], elapsed


def load_retry_reasons(path: Optional[Path]) -> dict[str, str]:
    """Read a ``{key: infrastructure reason}`` JSON file, refusing an empty reason."""
    if path is None:
        return {}
    reasons = read_json(path)
    if not isinstance(reasons, dict) or not all(
        isinstance(k, str) and isinstance(v, str) and v.strip() for k, v in reasons.items()
    ):
        raise StudyRefused(f"{path} must map each job key to a non-empty infrastructure reason")
    return reasons


def require_failed_keys(out: Path, reasons: Mapping[str, str]) -> None:
    """Refuse a retry reason for any key that is not a failed job in ``out``.

    A producer that splits one stage into several job lists (the learner's controls,
    training, search and design lists) hands each ``run_jobs`` call only its own keys,
    so a reason naming a key outside every list would never reach a check. Calling
    this once per stage, before any job runs, refuses it instead of ignoring it.
    """
    bad = []
    for key in sorted(reasons):
        path = job_path(out, key)
        if not path.exists() or read_json(path).get("status") != "failed":
            bad.append(key)
    if bad:
        raise StudyRefused(f"retry reasons given for keys that are not failed jobs in {out}: {bad}")


def run_jobs(
    jobs: Sequence[Job],
    func: Callable[[dict], dict],
    out: Path,
    workers: int,
    *,
    retry_reasons: Optional[Mapping[str, str]] = None,
) -> dict[str, int]:
    """Run every job without a finished file. An ok file is never recomputed.

    A failed job stays failed, because a failure is a result of the run until someone
    records that it was infrastructure. ``retry_reasons`` maps a failed key to that
    recorded reason; only such keys are retried, and the reason is stored in the new
    attempt's record beside the earlier attempts, which are kept. A reason for a key
    that is not failed in ``out`` is refused, so a reason cannot be attached to a
    result it does not describe.
    """
    keys = [job.key for job in jobs]
    if len(set(keys)) != len(keys):
        raise StudyRefused("duplicate job keys")
    reasons = dict(retry_reasons or {})
    for key, reason in reasons.items():
        if not isinstance(reason, str) or not reason.strip():
            raise StudyRefused(f"retry of {key} needs a non-empty infrastructure reason")
    pending: list[tuple[Job, int, Optional[str]]] = []
    skipped = 0
    retried: set[str] = set()
    # inv: every job before this one is either skipped or in `pending` with the next
    # unused attempt number, and `retried` holds the keys among them retried by reason.
    for job in jobs:
        path = job_path(out, job.key)
        previous = attempt_records(out, job.key)
        if path.exists():
            if not previous:
                raise StudyRefused(
                    f"{path} has no attempt ledger; it predates append-only attempts"
                )
            status = read_json(path).get("status")
            if status == "ok" or (status == "failed" and job.key not in reasons):
                skipped += 1
                continue
            retried.add(job.key)
            pending.append((job, len(previous) + 1, reasons[job.key]))
        else:
            pending.append((job, len(previous) + 1, None))
    stray = set(reasons) - retried
    if stray:
        raise StudyRefused(f"retry reasons given for keys that are not failed: {sorted(stray)}")
    counts = {"ok": 0, "failed": 0, "skipped": skipped}
    if not pending:
        return counts
    if workers <= 1:
        for job, attempt, reason in pending:
            _, status, _ = _run_one(func, job, out, attempt, reason)
            counts[status] += 1
        return counts
    with ProcessPoolExecutor(max_workers=min(workers, len(pending))) as pool:
        futures = [
            pool.submit(_run_one, func, job, out, attempt, reason)
            for job, attempt, reason in pending
        ]
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
    """Wall seconds summed over every attempt of every job in ``out``, failed ones too."""
    return sum(read_json(p)["seconds"] for p in (out / "timing").rglob("*.json"))


def attempt_ledger(out: Path) -> dict:
    """Attempt counts and core-seconds over every attempt in ``out``.

    ``attempts`` counts every attempt record, so a retried key contributes each of its
    tries. ``core_seconds`` is wall time and differs between runs of one commit, which is
    why it lives here and in the manifest and never in a job file.
    """
    by_status: dict[str, int] = {}
    keys = 0
    retried = 0
    attempts = 0
    root = out / "attempts"
    folders = sorted({p.parent for p in root.rglob("*.json")}) if root.is_dir() else []
    # inv: the counts cover exactly the folders already read.
    for folder in folders:
        records = sorted(p for p in folder.glob("*.json") if p.stem.isdigit())
        keys += 1
        attempts += len(records)
        retried += len(records) > 1
        for record in records:
            status = read_json(record)["status"]
            by_status[status] = by_status.get(status, 0) + 1
    return {
        "jobs_attempted": keys,
        "attempts": attempts,
        "jobs_retried": retried,
        "attempts_by_status": dict(sorted(by_status.items())),
        "core_seconds_all_attempts": core_seconds(out),
    }


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
