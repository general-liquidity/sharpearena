"""The HUD local feasibility fixture, run through the pinned SDK.

Skipped when `hud` is not installed; the rest of the package works without it.
The fixture lives in `examples/hud/` because INT-08 is a feasibility check, not a
shipped adapter: the adapter waits on the INT-01 contracts.

Every run here uses a deterministic agent double and never calls a model. The
telemetry exporter is disabled per process, so no span leaves the machine.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("hud")

# Disabling the exporter before `hud.settings` is first read is what keeps the
# fixture offline: with telemetry on and an API key in the environment, rollouts
# would report to the platform.
os.environ["HUD_TELEMETRY_ENABLED"] = "false"
os.environ["HUD_CLI_ANALYTICS_ENABLED"] = "false"
os.environ["HUD_FILE_TRACKING_ENABLED"] = "false"
os.environ.pop("HUD_API_KEY", None)

FIXTURE = Path(__file__).resolve().parents[3] / "examples" / "hud"
if str(FIXTURE) not in sys.path:
    sys.path.insert(0, str(FIXTURE))

from hud.eval import LocalRuntime, SubprocessRuntime, Task, Taskset  # noqa: E402

from agents import (  # noqa: E402
    CompletingAgent,
    DuplicateSubmissionAgent,
    EarlyExitAgent,
    MalformedThenValidAgent,
    StallingAgent,
)
from episode import SCENARIO_SEEDS, TASK_IDENTITY, open_episode  # noqa: E402
from grading import INCOMPLETE_FLOOR, export_evidence, grade_episode  # noqa: E402

HORIZON = 5


def _task(scenario="calm-a", horizon=HORIZON):
    return Task(
        env="sharpearena-feasibility",
        id="bounded_episode",
        args={"scenario": scenario, "horizon": horizon},
        # Required: HUD keys `Job.results` off `slug`. Two unslugged Task rows in one
        # Taskset deadlock the run instead of erroring (see the INT-08 report).
        slug=scenario,
    )


def _run(agent, *, scenario="calm-a", horizon=HORIZON, rollout_timeout=None, runtime=None):
    """One rollout through the SDK, returning the single `Run`."""
    job = asyncio.run(
        _task(scenario, horizon).run(
            agent,
            runtime=runtime or LocalRuntime(FIXTURE / "env.py"),
            rollout_timeout=rollout_timeout,
        )
    )
    assert len(job.runs) == 1
    return job, job.runs[0]


def _info(run):
    return run.evaluation["info"]


def test_completed_episode_is_graded_from_engine_evidence():
    job, run = _run(CompletingAgent())
    info = _info(run)

    assert info["status"] == "completed"
    assert info["bars_advanced"] == HORIZON
    assert len(info["returns"]) == HORIZON
    assert info["counts"] == {
        "expected_bars": HORIZON,
        "attempted": HORIZON,
        "accepted": HORIZON,
        "refused": 0,
        "invalid_decision": 0,
        "duplicate_submission": 0,
        "out_of_order": 0,
        "episode_closed": 0,
    }
    assert info["identity"]["task_identity"] == TASK_IDENTITY
    # The identity binds the scenario without publishing the generator seed.
    assert str(SCENARIO_SEEDS["calm-a"]["seed"]) not in json.dumps(info["identity"])
    assert run.reward == pytest.approx(job.reward)


def test_grade_ignores_the_agents_self_report():
    """A double that claims a different outcome does not move the score."""

    class LyingAgent(CompletingAgent):
        async def drive(self, run, client):
            await super().drive(run, client)
            return {"double": "lying", "claimed_reward": 1.0, "bars_submitted": 999}

    _, honest = _run(CompletingAgent())
    _, lying = _run(LyingAgent())

    assert json.loads(lying.trace.content)["claimed_reward"] == 1.0
    assert lying.reward == pytest.approx(honest.reward)
    assert _info(lying)["agent_report"] != _info(honest)["agent_report"]


def test_malformed_decision_is_refused_without_advancing_the_engine():
    _, run = _run(MalformedThenValidAgent())
    info = _info(run)

    assert info["counts"]["invalid_decision"] == 1
    # The refused attempt cost an attempt, not a bar.
    assert info["counts"]["attempted"] == HORIZON + 1
    assert info["bars_advanced"] == HORIZON
    assert info["status"] == "completed"


def test_duplicate_submission_does_not_double_advance_the_engine():
    _, run = _run(DuplicateSubmissionAgent())
    info = _info(run)

    assert info["counts"]["duplicate_submission"] == 1
    assert info["bars_advanced"] == HORIZON
    assert len(info["returns"]) == HORIZON


def test_early_exit_takes_the_floor_rather_than_a_favourable_prefix():
    _, run = _run(EarlyExitAgent(stop_after=2))
    info = _info(run)

    assert info["status"] == "incomplete"
    assert info["bars_advanced"] == 2
    assert run.reward == INCOMPLETE_FLOOR
    # The observed prefix stays in the evidence; only the score is withheld.
    assert len(info["returns"]) == 2


def test_rollout_timeout_is_an_error_not_a_zero_score():
    job, run = _run(StallingAgent(stall_seconds=30.0), rollout_timeout=6.0)

    assert run.trace.stop_reason == "timeout"
    assert run.trace.is_error
    # A timed-out rollout is excluded from the batch mean rather than averaged
    # in as a zero, which is the C04 requirement.
    assert job.errors == [run]


def test_subprocess_runtime_serves_the_same_env_out_of_process():
    _, run = _run(CompletingAgent(), runtime=SubprocessRuntime(FIXTURE / "env.py"))

    assert _info(run)["status"] == "completed"
    assert run.runtime.startswith("tcp://127.0.0.1:")


def test_two_scenarios_run_as_one_taskset_without_crossing_state():
    tasks = [_task("calm-a"), _task("fat-a")]
    job = asyncio.run(
        Taskset("int08-feasibility", tasks).run(
            CompletingAgent(), runtime=LocalRuntime(FIXTURE / "env.py")
        )
    )

    by_slug = job.results
    assert len(by_slug) == 2
    returns = [tuple(_info(runs[0])["returns"]) for runs in by_slug.values()]
    assert all(len(series) == HORIZON for series in returns)
    assert returns[0] != returns[1]


def test_repeated_rollouts_of_one_scenario_are_identical():
    """The engine is deterministic, so two rollouts of one row must agree."""
    job = asyncio.run(
        _task().run(CompletingAgent(), runtime=LocalRuntime(FIXTURE / "env.py"), group=2)
    )

    first, second = (_info(run)["returns"] for run in job.runs)
    assert first == second


# ─── interrupted export ──────────────────────────────────────────────────────


def _completed_grade(bars=3):
    from agents import equal_weight_decision

    episode = open_episode("calm-a", bars)
    decision = equal_weight_decision(list(episode.env.symbols))
    for bar in range(bars):
        episode.submit(decision, bar)
    return grade_episode(episode, None)


def test_interrupted_export_leaves_the_previous_run_intact(tmp_path, monkeypatch):
    """A write that dies mid-serialization must not replace a good export."""
    target = tmp_path / "evidence.json"
    export_evidence(_completed_grade(3), target)
    good = target.read_text(encoding="utf-8")

    def die(*args, **kwargs):
        raise KeyboardInterrupt("export interrupted")

    monkeypatch.setattr(json, "dumps", die)
    with pytest.raises(KeyboardInterrupt):
        export_evidence(_completed_grade(4), target)

    assert target.read_text(encoding="utf-8") == good
    assert json.loads(good)["info"]["bars_advanced"] == 3


def test_interrupted_export_never_publishes_a_truncated_file(tmp_path, monkeypatch):
    """A crash after the temporary write keeps the partial bytes off `destination`."""
    target = tmp_path / "evidence.json"
    real_replace = Path.replace

    def die(self, other):
        raise KeyboardInterrupt("killed before rename")

    monkeypatch.setattr(Path, "replace", die)
    with pytest.raises(KeyboardInterrupt):
        export_evidence(_completed_grade(3), target)

    monkeypatch.setattr(Path, "replace", real_replace)
    assert not target.exists()
    assert (tmp_path / "evidence.json.partial").exists()
