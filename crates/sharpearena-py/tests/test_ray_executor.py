"""The Ray executor (INT-06): guarded import, engine parity, and cross-worker-count
determinism.

Env-behaviour tests skip when ``ray`` is not installed; the rest of the package works
without it. The optional-import test runs *only* when ``ray`` is absent and asserts the
module still imports cleanly, with ``num_workers=0`` (the local route) usable with no
Ray at all, and any positive worker count raising the named :class:`RayUnavailable`.
"""

from __future__ import annotations

import importlib.util

import pytest

import sharpearena  # noqa: F401  (ensures the package imports without ray)
import sharpearena.ray_executor as rx
from sharpearena.integrations.parity import CORE_FIXTURES, deterministic_actions, engine_rollout
from sharpearena.ray_executor import (
    EpisodeSpec,
    ExecutorViolation,
    RayUnavailable,
    execute_episode,
    reduce_results,
    report_digest,
    run_episodes,
)

_HAS_RAY = importlib.util.find_spec("ray") is not None
needs_ray = pytest.mark.skipif(not _HAS_RAY, reason="ray not installed")


@pytest.fixture
def ray_cluster():
    """A local Ray instance, torn down after the test.

    Module-local rather than a `conftest.py` fixture: nothing else in this suite
    needs Ray, and a fixture that starts a cluster belongs next to the one file that
    uses it.
    """
    import ray

    ray.init(num_cpus=4, include_dashboard=False, log_to_driver=False)
    try:
        yield
    finally:
        ray.shutdown()


# ---------------------------------------------------------------------------
# Optional-import contract (runs when ray is NOT installed)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(_HAS_RAY, reason="ray is installed")
def test_import_works_and_distributed_route_raises_without_ray():
    """`import sharpearena.ray_executor` succeeds without ray, the local route
    (`num_workers=0`) still runs, and asking for workers raises a clear error."""
    assert rx._HAS_RAY is False
    spec = EpisodeSpec(episode_id="e0", n_symbols=2, n_days=10, actions=((0.0, 0.0),))
    report = run_episodes([spec], num_workers=0)
    assert report.results[0].status == "completed"
    with pytest.raises(RayUnavailable, match="ray is not installed"):
        run_episodes([spec], num_workers=2)


# ---------------------------------------------------------------------------
# reduce_results: the double-counting guard, exercised with no ray required
# ---------------------------------------------------------------------------


def _spec(episode_id: str = "e0") -> EpisodeSpec:
    return EpisodeSpec(episode_id=episode_id, n_symbols=2, n_days=10, actions=((0.0, 0.0),))


def _result(episode_id: str, total_reward: float = 0.0) -> rx.EpisodeResult:
    return rx.EpisodeResult(
        episode_id=episode_id,
        status="completed",
        steps=1,
        total_reward=total_reward,
        final_nav=1.0,
        terminated=False,
        truncated=False,
        spec_hash="deadbeef",
    )


def test_an_identical_retry_result_is_dropped_not_double_counted():
    """A Ray retry that reproduces the same result for an episode_id already placed is
    the harmless case `reduce_results` documents: dropped, and recorded as a dropped
    duplicate rather than silently doubling the run's evidence."""
    spec = _spec()
    result = _result("e0")
    report = reduce_results([spec], [result, result])
    assert report.duplicates_dropped == 1
    assert report.results == (result,)
    assert report.counts.attempted == 1


def test_a_conflicting_retry_result_is_refused_not_averaged():
    """Two different results for one episode_id falsify the purity a retry depends on;
    the run must be refused, not quietly reduced to one of them or an average."""
    spec = _spec()
    report = [_result("e0", total_reward=1.0), _result("e0", total_reward=2.0)]
    with pytest.raises(ExecutorViolation, match="two different results"):
        reduce_results([spec], report)


def test_a_result_for_an_unscheduled_episode_id_is_refused():
    with pytest.raises(ExecutorViolation, match="never scheduled"):
        reduce_results([_spec("e0")], [_result("e1")])


def test_duplicate_episode_ids_in_the_spec_list_are_refused():
    with pytest.raises(ExecutorViolation, match="must be unique"):
        reduce_results([_spec("e0"), _spec("e0")], [])


def test_an_episode_id_missing_from_the_specs_is_refused_at_construction():
    with pytest.raises(ExecutorViolation, match="non-empty string"):
        EpisodeSpec(episode_id="", actions=((0.0,),))


def test_actions_and_policy_are_mutually_exclusive_and_required():
    with pytest.raises(ExecutorViolation, match="exactly one"):
        EpisodeSpec(episode_id="e0")
    with pytest.raises(ExecutorViolation, match="exactly one"):
        EpisodeSpec(episode_id="e0", actions=((0.0,),), policy="m:f")


# ---------------------------------------------------------------------------
# Parity: the local route reproduces the engine, step for step, on CORE_FIXTURES
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture", CORE_FIXTURES, ids=lambda f: f.name)
def test_local_route_reproduces_the_engine_on_core_fixtures(fixture):
    """`run_episodes_local` (num_workers=0) is a pure function of `execute_episode`,
    which is meant to reproduce `integrations.parity.engine_rollout` bar for bar. This
    checks that claim directly, on the same fixtures every other adapter is held to,
    rather than trusting the module docstring's description of it."""
    reference = engine_rollout(fixture)
    spec = EpisodeSpec(
        episode_id=fixture.name,
        seed=fixture.seed,
        n_symbols=fixture.n_symbols,
        n_days=fixture.n_days,
        distribution_mode=fixture.distribution_mode,
        max_weight=fixture.max_weight,
        allow_short=fixture.allow_short,
        mode=fixture.mode,
        max_steps=len(fixture.actions),
        actions=fixture.actions,
    )
    result = execute_episode(spec)

    assert len(result.evidence) == len(reference.steps[: len(result.evidence)])
    for expected, actual in zip(reference.steps, result.evidence):
        assert actual.reward == expected.reward
        assert actual.nav == expected.nav
        assert actual.terminated == expected.terminated
        assert actual.truncated == expected.truncated
        assert actual.observation == expected.observation

    if fixture.expect == "refused":
        assert result.status == "refused"
    else:
        assert result.status == "completed"


# ---------------------------------------------------------------------------
# Determinism across worker counts: same seed, same reduced report, regardless of how
# many Ray workers ran it or in what order their results completed.
# ---------------------------------------------------------------------------


@needs_ray
def test_report_digest_is_identical_across_worker_counts(ray_cluster):
    """Ray gives no ordering guarantee over completing tasks, and retries at least
    once. `reduce_results` places by `episode_id` rather than completion order, so the
    reduced, canonically-digested report must be byte-identical whether it ran locally,
    on 2 workers, or on 3 -- this is the concrete claim the module docstring makes about
    determinism not coming from Ray, checked rather than asserted in prose."""
    specs = [
        EpisodeSpec(
            episode_id=f"ep-{i}",
            seed=i,
            n_symbols=3,
            n_days=20,
            max_steps=15,
            actions=deterministic_actions(n_symbols=3, n_steps=15, seed=i),
        )
        for i in range(6)
    ]

    local = run_episodes(specs, num_workers=0)
    two_workers = run_episodes(specs, num_workers=2)
    three_workers = run_episodes(specs, num_workers=3)

    digests = {report_digest(r) for r in (local, two_workers, three_workers)}
    assert len(digests) == 1, "report_digest must not depend on num_workers"
    assert local.counts.is_complete
    assert two_workers.num_workers == 2
    assert three_workers.num_workers == 3


@needs_ray
def test_a_deadline_reports_unscheduled_rather_than_dropping_them(ray_cluster):
    """A run that hits `deadline_s` before every spec was even scheduled must say so in
    `unscheduled`, not report a shorter run as if it had been asked for."""
    specs = [
        EpisodeSpec(
            episode_id=f"ep-{i}",
            seed=i,
            n_symbols=2,
            n_days=10,
            max_steps=5,
            actions=deterministic_actions(n_symbols=2, n_steps=5, seed=i),
        )
        for i in range(50)
    ]
    report = run_episodes(specs, num_workers=1, deadline_s=0.0)
    assert report.counts.expected == 50
    assert report.counts.attempted + len(report.unscheduled) == 50
