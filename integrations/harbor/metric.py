# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Job aggregation for the INT-09 task that keeps crashes apart from scored zeros.

Every trial is one of four outcomes:

- accepted: the grader scored the submission. A reward of 0.0 stays a 0.0.
- agent_fault: the grader refused the submission for something the agent did.
- evaluator_fault: the grader refused for something only the evaluator could
  cause (its own private input). The agent was never graded.
- infrastructure: the verifier ran out of a resource (disk, processes, time) that
  the agent may have exhausted, and said so. Not scored and not dropped.
- crashed: no trustworthy grader result exists. Anything unclassifiable lands here.

The headline, ``mean_reward_all_trials``, averages every accepted and agent-fault
trial with each agent fault scored 0. Evaluator faults leave the denominator and
are counted. ``mean_reward_accepted_only`` is a secondary figure that drops agent
faults too, so it is never the headline. A crash or an infrastructure verdict cannot
be attributed to the agent or the evaluator, so either makes the headline
unavailable; ``mean_reward_crashes_as_zero`` then gives the figure with every crash
and infrastructure trial scored 0.

Trials from different agents or job configurations are never pooled. ``--job-dir``
groups trials by the agent identity and resolved trial configuration in each
``result.json`` and reports one summary per group; with more than one group there
is no job-level mean. Harbor's uv-script contract hands this script one group per
agent name, model and dataset, and nothing that names the agent version or the job
configuration, so that entry point cannot check them and says so.

    python metric.py --job-dir jobs/<job-name>   # reads each trial's result.json and report.json
    uv run metric.py -i rewards.jsonl -o out.json  # Harbor's uv-script contract
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

GRADER = "sa-replay-feasibility"

ACCEPTED = "accepted"
AGENT_FAULT = "agent_fault"
EVALUATOR_FAULT = "evaluator_fault"
INFRASTRUCTURE = "infrastructure"
CRASHED = "crashed"

Outcome = tuple[str, float | None, str | None]
CRASH: Outcome = (CRASHED, None, None)

# Fields of Harbor 0.23.0's TrialConfig that differ between trials of one job
# configuration. Everything else in the resolved config identifies the group.
PER_TRIAL_CONFIG_FIELDS = ("trial_name", "trials_dir", "job_id")

UV_SCRIPT_GROUPING = (
    "one Harbor evals key (agent name, model, dataset); agent version and job "
    "configuration are not visible to this entry point, use --job-dir to check them"
)


def _finite(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _flag(value: object) -> int | None:
    return value if type(value) is int and value in (0, 1) else None


def classify_reward(reward: object) -> Outcome:
    """Classify one line of Harbor's rewards.jsonl (the grader's reward.json)."""
    if not isinstance(reward, dict) or not _finite(reward.get("reward")):
        return CRASH
    keys = ("accepted", "agent_fault", "evaluator_fault", "infrastructure_fault")
    flags = tuple(_flag(reward.get(key)) for key in keys)
    if None in flags or sum(flags) != 1:
        return CRASH
    accepted, agent_fault, evaluator_fault, _ = flags
    if accepted:
        return (ACCEPTED, float(reward["reward"]), None)
    if reward["reward"] != 0:
        return CRASH
    if agent_fault:
        return (AGENT_FAULT, None, None)
    return (EVALUATOR_FAULT, None, None) if evaluator_fault else (INFRASTRUCTURE, None, None)


def classify_report(report: object) -> Outcome:
    """Classify one trial's verifier/report.json."""
    if not isinstance(report, dict) or report.get("grader") != GRADER:
        return CRASH
    accepted, reward, refusal, fault = (
        report.get(key) for key in ("accepted", "reward", "refusal", "fault")
    )
    if not _finite(reward):
        return CRASH
    if accepted is True and refusal is None and fault is None:
        return (ACCEPTED, float(reward), None)
    if accepted is False and isinstance(refusal, str) and refusal and reward == 0:
        if fault == "agent":
            return (AGENT_FAULT, None, refusal)
        if fault == "evaluator":
            return (EVALUATOR_FAULT, None, refusal)
        if fault == "infrastructure":
            return (INFRASTRUCTURE, None, refusal)
    return CRASH


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_bytes())
    except (OSError, ValueError):
        return None


def trial_group(result: object) -> str | None:
    """A digest of the agent identity and resolved trial configuration, or None."""
    if not isinstance(result, dict):
        return None
    config, agent = result.get("config"), result.get("agent_info")
    if not isinstance(config, dict) or not isinstance(agent, dict):
        return None
    shared = {k: v for k, v in config.items() if k not in PER_TRIAL_CONFIG_FIELDS}
    identity = {
        "agent_info": agent,
        "config": shared,
        "task_checksum": result.get("task_checksum"),
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _group_label(result: dict) -> dict[str, object]:
    agent = result.get("agent_info") or {}
    config_agent = (result.get("config") or {}).get("agent") or {}
    model = agent.get("model_info") if isinstance(agent, dict) else None
    return {
        "agent_name": agent.get("name") if isinstance(agent, dict) else None,
        "agent_version": agent.get("version") if isinstance(agent, dict) else None,
        "agent_import_path": config_agent.get("import_path") if isinstance(config_agent, dict) else None,
        "model": model.get("name") if isinstance(model, dict) else None,
    }


def classify_trial(trial_dir: Path) -> Outcome:
    result = _load_json(trial_dir / "result.json")
    if not isinstance(result, dict) or result.get("exception_info") is not None:
        return CRASH
    return classify_report(_load_json(trial_dir / "verifier" / "report.json"))


def _mean(values: list[float], count: int) -> float | None:
    return math.fsum(values) / count if count else None


def summarize(outcomes: list[Outcome]) -> dict[str, object]:
    """Summarize one group of trials that share an agent and job configuration."""
    rewards = [reward for kind, reward, _ in outcomes if kind == ACCEPTED]
    counts = {
        kind: sum(k == kind for k, _, _ in outcomes)
        for kind in (ACCEPTED, AGENT_FAULT, EVALUATOR_FAULT, INFRASTRUCTURE, CRASHED)
    }
    scored = counts[ACCEPTED] + counts[AGENT_FAULT]
    unattributable = counts[CRASHED] + counts[INFRASTRUCTURE]
    summary: dict[str, object] = {"trials": len(outcomes), **counts, "scored_trials": scored}
    if unattributable:
        summary["mean_reward_all_trials"] = None
        summary["mean_reward_all_trials_unavailable"] = (
            f"{counts[CRASHED]} crashed and {counts[INFRASTRUCTURE]} infrastructure-unavailable "
            "trials have no attributable fault"
        )
        summary["mean_reward_crashes_as_zero"] = _mean(rewards, scored + unattributable)
    elif scored:
        summary["mean_reward_all_trials"] = _mean(rewards, scored)
    else:
        summary["mean_reward_all_trials"] = None
        summary["mean_reward_all_trials_unavailable"] = "no scored trials"
    summary["mean_reward_accepted_only"] = _mean(rewards, len(rewards))
    summary["mean_reward_accepted_only_note"] = (
        "secondary: excludes agent faults, so it is not the headline"
    )
    for kind in (AGENT_FAULT, EVALUATOR_FAULT, INFRASTRUCTURE):
        codes = sorted(code for k, _, code in outcomes if k == kind and code)
        if codes:
            summary[f"{kind}_codes"] = {code: codes.count(code) for code in dict.fromkeys(codes)}
    return summary


def summarize_job(job_dir: Path) -> dict[str, object]:
    """One summary per agent and configuration group; no job-level mean across groups."""
    trials = sorted(path for path in job_dir.iterdir() if path.is_dir())
    if not trials:
        raise SystemExit(f"{job_dir} holds no trial directories")
    groups: dict[str, list[Outcome]] = {}
    labels: dict[str, dict[str, object]] = {}
    unattributed: list[str] = []
    for trial in trials:
        result = _load_json(trial / "result.json")
        key = trial_group(result)
        if key is None:
            unattributed.append(trial.name)
            continue
        labels.setdefault(key, _group_label(result))
        groups.setdefault(key, []).append(classify_trial(trial))
    out: dict[str, object] = {
        "groups": [
            {"group": key, **labels[key], **summarize(groups[key])} for key in sorted(groups)
        ],
        "unattributed_trials": unattributed,
    }
    if len(groups) > 1:
        out["pooled"] = None
        out["pooled_refused"] = (
            f"{len(groups)} agent or job-configuration groups; their trials are not pooled"
        )
    elif unattributed:
        out["pooled"] = None
        out["pooled_refused"] = (
            f"{len(unattributed)} trials have no readable result.json and cannot be "
            "assigned to a group"
        )
    elif groups:
        out["pooled"] = out["groups"][0]
    else:
        out["pooled"] = None
        out["pooled_refused"] = "no trial has a readable result.json"
    return out


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
        result: dict[str, object] = {**summarize(outcomes), "grouping": UV_SCRIPT_GROUPING}
    else:
        result = summarize_job(args.job_dir)

    payload = json.dumps(result, allow_nan=False, sort_keys=True)
    if args.output is None:
        print(payload)
    else:
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
