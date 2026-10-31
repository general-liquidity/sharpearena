# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Job aggregation for the INT-09 task that keeps crashes apart from scored zeros.

Harbor's default mean counts a trial with no reward as 0, so a crash and an honest
0.0 look the same. Here every trial is one of three outcomes, and the mean covers
accepted trials only:

- accepted: the grader scored the submission. A reward of 0.0 stays in the mean.
- refused: the grader rejected the submission with a reason code.
- crashed: no trustworthy grader result exists. Anything unclassifiable lands here.

Two entry points, because Harbor gives a job metric the reward dicts and nothing else:

    python metric.py --job-dir jobs/<job-name>   # reads each trial's report.json
    uv run metric.py -i rewards.jsonl -o out.json  # Harbor's uv-script contract

Only the first can name refusal codes. The second relies on the ``accepted`` field
grade.py writes to reward.json. ``mean_accepted_reward`` is null when nothing was
accepted: unavailable is not zero.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

GRADER = "sa-replay-feasibility"

ACCEPTED = "accepted"
REFUSED = "refused"
CRASHED = "crashed"

Outcome = tuple[str, float | None, str | None]
CRASH: Outcome = (CRASHED, None, None)


def _finite(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def classify_reward(reward: object) -> Outcome:
    """Classify one line of Harbor's rewards.jsonl."""
    if not isinstance(reward, dict) or not _finite(reward.get("reward")):
        return CRASH
    if reward.get("accepted") == 1:
        return (ACCEPTED, float(reward["reward"]), None)
    if reward.get("accepted") == 0 and reward["reward"] == 0:
        return (REFUSED, None, None)
    return CRASH


def classify_report(report: object) -> Outcome:
    """Classify one trial's verifier/report.json."""
    if not isinstance(report, dict) or report.get("grader") != GRADER:
        return CRASH
    accepted, reward, refusal = (report.get(key) for key in ("accepted", "reward", "refusal"))
    if not _finite(reward):
        return CRASH
    if accepted is True and refusal is None:
        return (ACCEPTED, float(reward), None)
    if accepted is False and isinstance(refusal, str) and refusal and reward == 0:
        return (REFUSED, None, refusal)
    return CRASH


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_bytes())
    except (OSError, ValueError):
        return None


def classify_trial(trial_dir: Path) -> Outcome:
    result = _load_json(trial_dir / "result.json")
    if not isinstance(result, dict) or result.get("exception_info") is not None:
        return CRASH
    return classify_report(_load_json(trial_dir / "verifier" / "report.json"))


def summarize(outcomes: list[Outcome]) -> dict[str, object]:
    rewards = [reward for kind, reward, _ in outcomes if kind == ACCEPTED]
    summary: dict[str, object] = {
        "trials": len(outcomes),
        ACCEPTED: len(rewards),
        REFUSED: sum(kind == REFUSED for kind, _, _ in outcomes),
        CRASHED: sum(kind == CRASHED for kind, _, _ in outcomes),
        "mean_accepted_reward": sum(rewards) / len(rewards) if rewards else None,
    }
    codes = sorted(code for kind, _, code in outcomes if kind == REFUSED and code)
    if codes:
        summary["refusal_codes"] = {code: codes.count(code) for code in dict.fromkeys(codes)}
    return summary


def job_outcomes(job_dir: Path) -> list[Outcome]:
    trials = sorted(path for path in job_dir.iterdir() if path.is_dir())
    if not trials:
        raise SystemExit(f"{job_dir} holds no trial directories")
    return [classify_trial(trial) for trial in trials]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("-i", "--input", type=Path, help="Harbor rewards.jsonl")
    source.add_argument("--job-dir", type=Path, help="a Harbor job directory")
    parser.add_argument("-o", "--output", type=Path, help="write here instead of stdout")
    args = parser.parse_args(argv)

    if args.input is not None:
        lines = args.input.read_text(encoding="utf-8").splitlines()
        outcomes = [classify_reward(json.loads(line)) for line in lines if line.strip()]
    else:
        outcomes = job_outcomes(args.job_dir)

    payload = json.dumps(summarize(outcomes), allow_nan=False, sort_keys=True)
    if args.output is None:
        print(payload)
    else:
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
