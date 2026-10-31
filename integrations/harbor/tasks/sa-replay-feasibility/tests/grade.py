"""Separate-verifier grader for the INT-09 Harbor feasibility task.

Runs inside the verifier container only. Its inputs are the single declared
agent artifact (/logs/artifacts/actions.jsonl) and the private evaluator file
shipped with this tests/ directory, which the agent container never receives.

Every refusal writes a distinct code to /logs/verifier/report.json so a failing
fixture names the reason it failed instead of collapsing into "reward 0".
"""

from __future__ import annotations

import json
import os
from pathlib import Path

ARTIFACTS_DIR = Path("/logs/artifacts")
ARTIFACT = ARTIFACTS_DIR / "actions.jsonl"
# Written by the [[verifier.collect]] hook in task.toml. The hook runs inside the
# agent container, so this file reports what the hook did; it proves nothing about
# a tree the hook never saw, which is why _require_safe_tree() walks the upload too.
TRANSFER_CHECK = ARTIFACTS_DIR / ".sa-transfer-check.json"
PRIVATE = Path("/tests/private/future_prices.json")
VERIFIER_DIR = Path("/logs/verifier")
REWARD_PATH = VERIFIER_DIR / "reward.txt"
# Harbor prefers reward.json and hands only that dict to a job metric, never
# report.json, so "accepted" must travel here for metric.py to separate a refusal
# from a scored zero.
REWARD_JSON_PATH = VERIFIER_DIR / "reward.json"
REPORT_PATH = VERIFIER_DIR / "report.json"

MAX_ARTIFACT_BYTES = 64 * 1024
MAX_TRANSFER_CHECK_BYTES = 1024
EXPECTED_STEPS = 8
LEGAL_ACTIONS = {"hold", "buy", "sell"}
TERMINAL_ACTION = "close"


class Refusal(Exception):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _require_clean_verifier_dir() -> None:
    """A reward file present before grading means a stale or injected output."""
    for name in ("reward.txt", "reward.json", "rewards.json"):
        if (VERIFIER_DIR / name).exists():
            raise Refusal(
                "pre_existing_reward_file",
                f"{name} existed in the verifier directory before grading",
            )


def _require_transfer_check() -> None:
    """Refuse when the pre-transfer hook found unsafe entries or left no result."""
    try:
        if TRANSFER_CHECK.is_symlink() or not TRANSFER_CHECK.is_file():
            raise Refusal("transfer_check_unavailable", f"{TRANSFER_CHECK} is not a file")
        with TRANSFER_CHECK.open("rb") as stream:
            payload = stream.read(MAX_TRANSFER_CHECK_BYTES + 1)
    except OSError as exc:
        raise Refusal("transfer_check_unavailable", str(exc)) from exc
    try:
        check = json.loads(payload)
    except ValueError as exc:
        raise Refusal("transfer_check_unavailable", f"unparseable: {exc}") from exc
    removed = check.get("removed") if isinstance(check, dict) else None
    if type(removed) is not int or removed < 0 or check.get("version") != 1:
        raise Refusal("transfer_check_unavailable", f"malformed result {check!r}")
    if removed:
        raise Refusal(
            "artifact_tree_unsafe",
            f"{removed} non-regular entries were removed before artifact transfer",
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
            if path.is_symlink() or not (path.is_dir() or path.is_file()):
                raise Refusal("artifact_tree_unsafe", f"{path} is not a regular entry")


def _read_artifact() -> list[dict[str, object]]:
    try:
        # Check symlinks before existence: dangling links are non-regular too.
        if ARTIFACT.is_symlink():
            raise Refusal("artifact_not_regular_file", f"{ARTIFACT} is a symlink")
        if not ARTIFACT.exists():
            raise Refusal("artifact_missing", f"{ARTIFACT} was not collected")
        if not ARTIFACT.is_file():
            raise Refusal("artifact_not_regular_file", f"{ARTIFACT} is not a regular file")
        # Bound the actual read, not just a prior stat of a mutable file.
        with ARTIFACT.open("rb") as stream:
            payload = stream.read(MAX_ARTIFACT_BYTES + 1)
    except OSError as exc:
        raise Refusal("artifact_unreadable", str(exc)) from exc
    size = len(payload)
    if size == 0:
        raise Refusal("artifact_empty", f"{ARTIFACT} is empty")
    if size > MAX_ARTIFACT_BYTES:
        raise Refusal("artifact_oversized", f"{size} bytes exceeds {MAX_ARTIFACT_BYTES}")

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
        elif not isinstance(action, str) or action not in LEGAL_ACTIONS:
            raise Refusal("illegal_action", f"step {index} action {action!r}")
    return records


def _score(records: list[dict[str, object]]) -> float:
    if not PRIVATE.exists():
        raise Refusal("private_input_missing", f"{PRIVATE} is not available")
    returns = json.loads(PRIVATE.read_text(encoding="utf-8"))["returns"]
    signs = {"buy": 1.0, "sell": -1.0, "hold": 0.0, TERMINAL_ACTION: 0.0}
    pnl = sum(
        signs[str(record["action"])] * float(record["size"]) * float(returns[index])
        for index, record in enumerate(records)
    )
    # Bounded, monotone in pnl, so a reward is comparable across fixtures.
    return max(0.0, min(1.0, 0.5 + 50.0 * pnl))


def main() -> int:
    VERIFIER_DIR.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {"grader": "sa-replay-feasibility", "version": 2}
    try:
        _require_clean_verifier_dir()
        _require_transfer_check()
        _require_safe_tree()
        records = _validate(_read_artifact())
        reward = _score(records)
        report.update({"accepted": True, "refusal": None, "reward": reward})
    except Refusal as refusal:
        reward = 0.0
        report.update(
            {"accepted": False, "refusal": refusal.code, "detail": refusal.detail, "reward": 0.0}
        )
    report["private_input_visible"] = PRIVATE.exists()
    report["tests_dir_visible"] = os.path.isdir("/tests")
    REPORT_PATH.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    REWARD_JSON_PATH.write_text(
        json.dumps({"accepted": int(report["accepted"]), "reward": reward}), encoding="utf-8"
    )
    REWARD_PATH.write_text(f"{reward}", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
