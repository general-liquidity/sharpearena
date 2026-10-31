"""Separate-verifier grader for the INT-09 Harbor feasibility task.

Runs inside the verifier container only. Its inputs are the single declared
agent artifact (/logs/artifacts/actions.jsonl) and the private evaluator file
shipped with this tests/ directory, which the agent container never receives.

Every refusal writes a distinct code to /logs/verifier/report.json so a failing
fixture names the reason it failed instead of collapsing into "reward 0". Each
refusal is also labelled as an agent fault or an evaluator fault, so a job metric
can score the first as 0 and drop the second from its denominator.

Agent-supplied files are read through one descriptor, opened without following a
link and without blocking on a FIFO. Every check runs on that descriptor and the
bytes are read from it once into memory, so swapping the path after a check
cannot change what is graded.
"""

from __future__ import annotations

import errno
import json
import os
from pathlib import Path
import shutil
import stat

ARTIFACTS_DIR = Path("/logs/artifacts")
ARTIFACT = ARTIFACTS_DIR / "actions.jsonl"
# Written by the [[verifier.collect]] hook in task.toml. The hook runs inside the
# agent container, so this file reports what the hook did; it proves nothing about
# a tree the hook never saw, which is why _require_safe_tree() walks the upload too.
TRANSFER_CHECK = ARTIFACTS_DIR / ".sa-transfer-check.json"
TRANSFER_CHECK_VERSION = 2
PRIVATE = Path("/tests/private/future_prices.json")
VERIFIER_DIR = Path("/logs/verifier")
REWARD_PATH = VERIFIER_DIR / "reward.txt"
# Harbor prefers reward.json and hands only that dict to a job metric, never
# report.json, so "accepted" and the fault labels must travel here for metric.py
# to separate a refusal from a scored zero. Harbor requires every value numeric.
REWARD_JSON_PATH = VERIFIER_DIR / "reward.json"
REPORT_PATH = VERIFIER_DIR / "report.json"

MAX_ARTIFACT_BYTES = 64 * 1024
MAX_TRANSFER_CHECK_BYTES = 1024
EXPECTED_STEPS = 8
LEGAL_ACTIONS = {"hold", "buy", "sell"}
TERMINAL_ACTION = "close"

# Refusals the agent cannot cause: the verifier image owns /tests. Everything else
# is an agent fault, including codes an evaluator defect could also produce, so an
# agent can never move a trial out of the denominator by tripping an ambiguous code.
EVALUATOR_FAULT_CODES = frozenset({"private_input_missing", "private_input_unreadable"})
# Resource exhaustion the agent may have caused (Harbor's Docker provider bounds
# neither processes nor disk). Not scored and not dropped: metric.py makes the
# headline unavailable, so exhausting a resource can neither earn nor dodge a 0.
INFRASTRUCTURE_FAULT_CODES = frozenset({"verifier_resources_exhausted"})
# Room the verifier needs to write its three outputs, with a wide margin.
MIN_VERIFIER_FREE_BYTES = 1024 * 1024

# O_NOFOLLOW refuses a link as the final path component (ELOOP), O_NONBLOCK keeps a
# FIFO with no writer from blocking the open. Both are Linux flags; the getattr
# defaults only let this module import on a host that lacks them.
_SAFE_OPEN_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_NONBLOCK", 0)
    | getattr(os, "O_NOCTTY", 0)
    | getattr(os, "O_BINARY", 0)
)


class Refusal(Exception):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def read_regular_bounded(path: Path, limit: int, *, missing: str, not_regular: str, unreadable: str) -> bytes:
    """Return at most ``limit + 1`` bytes of ``path`` if it is a regular file.

    Refuses with ``not_regular`` for a symlink (dangling or not), a FIFO, a device or
    a directory, with ``missing`` when nothing is there, and with ``unreadable`` for
    any other OS error. Reading ``limit + 1`` lets the caller tell "exactly at the
    bound" from "over it" without reading an unbounded file.
    """
    try:
        fd = os.open(path, _SAFE_OPEN_FLAGS)
    except FileNotFoundError as exc:
        raise Refusal(missing, f"{path} was not collected") from exc
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise Refusal(not_regular, f"{path} is a symlink") from exc
        raise Refusal(unreadable, str(exc)) from exc
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise Refusal(not_regular, f"{path} is not a regular file")
        chunks: list[bytes] = []
        remaining = limit + 1
        # inv: sum(len(c) for c in chunks) == limit + 1 - remaining.
        # term: each pass either ends at EOF or lowers remaining by at least one byte.
        while remaining > 0:
            chunk = os.read(fd, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)
    except OSError as exc:
        raise Refusal(unreadable, str(exc)) from exc
    finally:
        os.close(fd)


def _require_verifier_headroom() -> None:
    """Refuse to grade when the verifier could not reliably write its verdict."""
    try:
        free = shutil.disk_usage(VERIFIER_DIR).free
    except OSError as exc:
        raise Refusal("verifier_resources_exhausted", f"disk usage unreadable: {exc}") from exc
    if free < MIN_VERIFIER_FREE_BYTES:
        raise Refusal(
            "verifier_resources_exhausted",
            f"{free} bytes free under {VERIFIER_DIR}, below {MIN_VERIFIER_FREE_BYTES}",
        )


def _require_clean_verifier_dir() -> None:
    """A reward file present before grading means a stale or injected output."""
    for name in ("reward.txt", "reward.json", "rewards.json"):
        if os.path.lexists(VERIFIER_DIR / name):
            raise Refusal(
                "pre_existing_reward_file",
                f"{name} existed in the verifier directory before grading",
            )


def _require_transfer_check() -> None:
    """Refuse when the pre-transfer hook removed entries or left no result."""
    code = "transfer_check_unavailable"
    payload = read_regular_bounded(
        TRANSFER_CHECK, MAX_TRANSFER_CHECK_BYTES, missing=code, not_regular=code, unreadable=code
    )
    if len(payload) > MAX_TRANSFER_CHECK_BYTES:
        raise Refusal(code, f"{TRANSFER_CHECK} exceeds {MAX_TRANSFER_CHECK_BYTES} bytes")
    try:
        check = json.loads(payload)
    except ValueError as exc:
        raise Refusal(code, f"unparseable: {exc}") from exc
    counts = {}
    for key in ("removed", "oversized"):
        value = check.get(key) if isinstance(check, dict) else None
        if type(value) is not int or value < 0:
            raise Refusal(code, f"malformed result {check!r}")
        counts[key] = value
    if check.get("version") != TRANSFER_CHECK_VERSION:
        raise Refusal(code, f"malformed result {check!r}")
    if counts["oversized"]:
        raise Refusal(
            "artifact_transfer_oversized",
            f"{counts['oversized']} files over the transfer bound were removed before transfer",
        )
    if counts["removed"]:
        raise Refusal(
            "artifact_tree_unsafe",
            f"{counts['removed']} non-regular entries were removed before artifact transfer",
        )


def _require_safe_tree() -> None:
    """Refuse a link or special file anywhere in the uploaded artifact tree."""
    if ARTIFACTS_DIR.is_symlink():
        raise Refusal("artifact_tree_unsafe", f"{ARTIFACTS_DIR} is a symlink")

    def walk_failed(exc: OSError) -> None:
        raise Refusal("artifact_unreadable", str(exc)) from exc

    for directory, subdirs, files in os.walk(ARTIFACTS_DIR, onerror=walk_failed):
        for name in [*subdirs, *files]:
            path = Path(directory) / name
            if path == ARTIFACT:
                continue  # _read_artifact() owns the graded path's own reason codes.
            mode = os.lstat(path).st_mode
            if not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
                raise Refusal("artifact_tree_unsafe", f"{path} is not a regular entry")


def _read_artifact() -> list[dict[str, object]]:
    payload = read_regular_bounded(
        ARTIFACT,
        MAX_ARTIFACT_BYTES,
        missing="artifact_missing",
        not_regular="artifact_not_regular_file",
        unreadable="artifact_unreadable",
    )
    size = len(payload)
    if size == 0:
        raise Refusal("artifact_empty", f"{ARTIFACT} is empty")
    if size > MAX_ARTIFACT_BYTES:
        raise Refusal("artifact_oversized", f"more than {MAX_ARTIFACT_BYTES} bytes")

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise Refusal("artifact_unparseable", f"invalid UTF-8: {exc}") from exc
    records: list[dict[str, object]] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError as exc:
            raise Refusal("artifact_unparseable", f"line {lineno}: {exc}") from exc
        if not isinstance(record, dict):
            raise Refusal("artifact_unparseable", f"line {lineno} is not an object")
        records.append(record)
    return records


def _validate(records: list[dict[str, object]]) -> list[dict[str, object]]:
    if len(records) != EXPECTED_STEPS:
        raise Refusal(
            "step_count_mismatch",
            f"expected {EXPECTED_STEPS} decisions, got {len(records)}",
        )
    for index, record in enumerate(records):
        if type(record.get("step")) is not int or record.get("step") != index:
            raise Refusal(
                "step_index_mismatch",
                f"record {index} carries step {record.get('step')!r}",
            )
        action = record.get("action")
        size = record.get("size")
        if not isinstance(size, (int, float)) or isinstance(size, bool):
            raise Refusal("size_not_numeric", f"step {index} size {size!r}")
        # Compare before conversion so an arbitrarily large JSON integer cannot
        # escape the refusal contract through float overflow.
        if not 0.0 <= size <= 1.0:
            raise Refusal("size_out_of_range", f"step {index} size {size!r}")
        if index == len(records) - 1:
            if action != TERMINAL_ACTION:
                raise Refusal(
                    "missing_final_action",
                    f"last record action {action!r}, expected {TERMINAL_ACTION!r}",
                )
            # instruction.md: the final line must be {"step": <n>, "action": "close", "size": 0}.
            if size != 0:
                raise Refusal("final_close_size_nonzero", f"close at step {index} has size {size!r}")
        elif not isinstance(action, str) or action not in LEGAL_ACTIONS:
            raise Refusal("illegal_action", f"step {index} action {action!r}")
    return records


def _score(records: list[dict[str, object]]) -> float:
    if not PRIVATE.exists():
        raise Refusal("private_input_missing", f"{PRIVATE} is not available")
    try:
        returns = json.loads(PRIVATE.read_text(encoding="utf-8"))["returns"]
        values = [float(returns[index]) for index in range(len(records))]
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        raise Refusal("private_input_unreadable", f"{type(exc).__name__}: {exc}") from exc
    signs = {"buy": 1.0, "sell": -1.0, "hold": 0.0, TERMINAL_ACTION: 0.0}
    pnl = sum(
        signs[str(record["action"])] * float(record["size"]) * values[index]
        for index, record in enumerate(records)
    )
    # Bounded, monotone in pnl, so a reward is comparable across fixtures.
    return max(0.0, min(1.0, 0.5 + 50.0 * pnl))


def main() -> int:
    VERIFIER_DIR.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {"grader": "sa-replay-feasibility", "version": 3}
    try:
        _require_verifier_headroom()
        _require_clean_verifier_dir()
        _require_transfer_check()
        _require_safe_tree()
        records = _validate(_read_artifact())
        reward = _score(records)
        report.update({"accepted": True, "refusal": None, "fault": None, "reward": reward})
    except Refusal as refusal:
        reward = 0.0
        if refusal.code in EVALUATOR_FAULT_CODES:
            fault = "evaluator"
        elif refusal.code in INFRASTRUCTURE_FAULT_CODES:
            fault = "infrastructure"
        else:
            fault = "agent"
        report.update(
            {
                "accepted": False,
                "refusal": refusal.code,
                "fault": fault,
                "detail": refusal.detail,
                "reward": 0.0,
            }
        )
    report["private_input_visible"] = PRIVATE.exists()
    report["tests_dir_visible"] = os.path.isdir("/tests")
    REPORT_PATH.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    REWARD_JSON_PATH.write_text(
        json.dumps(
            {
                "accepted": int(report["accepted"]),
                "agent_fault": int(report["fault"] == "agent"),
                "evaluator_fault": int(report["fault"] == "evaluator"),
                "infrastructure_fault": int(report["fault"] == "infrastructure"),
                "reward": reward,
            }
        ),
        encoding="utf-8",
    )
    REWARD_PATH.write_text(f"{reward}", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
