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

ARTIFACT = Path("/logs/artifacts/actions.jsonl")
PRIVATE = Path("/tests/private/future_prices.json")
VERIFIER_DIR = Path("/logs/verifier")
REWARD_PATH = VERIFIER_DIR / "reward.txt"
REPORT_PATH = VERIFIER_DIR / "report.json"

MAX_ARTIFACT_BYTES = 64 * 1024
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
    for name in ("reward.txt", "rewards.json"):
        if (VERIFIER_DIR / name).exists():
            raise Refusal(
                "pre_existing_reward_file",
                f"{name} existed in the verifier directory before grading",
            )


def _read_artifact() -> list[dict[str, object]]:
    if not ARTIFACT.exists():
        raise Refusal("artifact_missing", f"{ARTIFACT} was not collected")
    if ARTIFACT.is_symlink() or not ARTIFACT.is_file():
        raise Refusal("artifact_not_regular_file", f"{ARTIFACT} is not a regular file")
    size = ARTIFACT.stat().st_size
    if size == 0:
        raise Refusal("artifact_empty", f"{ARTIFACT} is empty")
    if size > MAX_ARTIFACT_BYTES:
        raise Refusal("artifact_oversized", f"{size} bytes exceeds {MAX_ARTIFACT_BYTES}")

    records: list[dict[str, object]] = []
    for lineno, line in enumerate(ARTIFACT.read_text(encoding="utf-8").splitlines(), 1):
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
        if record.get("step") != index:
            raise Refusal(
                "step_index_mismatch",
                f"record {index} carries step {record.get('step')!r}",
            )
        action = record.get("action")
        size = record.get("size")
        if not isinstance(size, (int, float)) or isinstance(size, bool):
            raise Refusal("size_not_numeric", f"step {index} size {size!r}")
        if not 0.0 <= float(size) <= 1.0:
            raise Refusal("size_out_of_range", f"step {index} size {size!r}")
        if index == len(records) - 1:
            if action != TERMINAL_ACTION:
                raise Refusal(
                    "missing_final_action",
                    f"last record action {action!r}, expected {TERMINAL_ACTION!r}",
                )
        elif action not in LEGAL_ACTIONS:
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
    report: dict[str, object] = {"grader": "sa-replay-feasibility", "version": 1}
    try:
        _require_clean_verifier_dir()
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
    REWARD_PATH.write_text(f"{reward}", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
