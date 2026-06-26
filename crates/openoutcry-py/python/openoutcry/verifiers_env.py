"""PrimeIntellect ``verifiers`` environment for OpenOutcry.

The rubric is scored by the **real SharpeBench kernel** — :func:`deflated_sharpe_reward`
and :func:`pass_k_reward` call :func:`openoutcry.score_run`, the exact Rust deflated
Sharpe / pass^k the benchmark computes, not a Python reimplementation. Verified against
``verifiers`` 0.1.14 (`vf.Rubric(funcs=, weights=)`).

Usage::

    import verifiers as vf
    from openoutcry.verifiers_env import build_rubric, load_environment
    rubric = build_rubric()                 # the SharpeBench-calibrated reward bundle
    # env = load_environment(dataset=...)   # a SingleTurnEnv backed by the rubric

A rollout drives :class:`~openoutcry.gym.OpenOutcryEnv` and records the per-step reward
(portfolio return) into ``state['returns']`` (and the env's process events into
``state['events']``); the rubric then scores the run with the real kernel.
"""

from __future__ import annotations

import json
from typing import Any, Optional, Sequence

from .openoutcry_py import score_run  # the real SharpeBench scorer (pyo3)

try:  # pragma: no cover - exercised only when verifiers is installed
    import verifiers as vf

    _HAS_VERIFIERS = True
except Exception:  # noqa: BLE001 - any import failure means "not available"
    vf = None  # type: ignore[assignment]
    _HAS_VERIFIERS = False


# ---------------------------------------------------------------------------
# SharpeBench-calibrated reward functions (the Rust kernel, not approximations)
# ---------------------------------------------------------------------------

def _returns_from_state(state: Optional[dict]) -> list[float]:
    """Per-step portfolio returns recorded by the rollout, if any."""
    return [float(r) for r in (state or {}).get("returns", []) or []]


def _composite(returns: list[float], n_trials: int) -> dict:
    """The real SharpeBench ``CompositeScore`` for a return series (Rust kernel)."""
    if len(returns) < 2:
        return {}
    return json.loads(score_run(returns, n_trials))


def deflated_sharpe_reward(
    completion: Any = None,
    state: Optional[dict] = None,
    *,
    n_trials: int = 0,
    **kwargs: Any,
) -> float:
    """The **real** deflated Sharpe (SharpeBench kernel), deflated for ``n_trials``
    of declared in-sample search — the metric the benchmark ranks on."""
    return float(_composite(_returns_from_state(state), n_trials).get("deflated_sharpe", 0.0))


def pass_k_reward(
    completion: Any = None,
    state: Optional[dict] = None,
    *,
    n_trials: int = 0,
    **kwargs: Any,
) -> float:
    """1.0 iff the run clears the per-run PSR bar (the kernel's ``passed_k`` gate)."""
    return 1.0 if _composite(_returns_from_state(state), n_trials).get("passed_k", False) else 0.0


def process_check_reward(
    completion: Any = None,
    state: Optional[dict] = None,
    **kwargs: Any,
) -> float:
    """Penalize block-severity events surfaced in the env's per-step ``info`` (the
    sim-exploitation guard, e.g. a manipulative order). 1.0 = clean."""
    events: Sequence[dict] = (state or {}).get("events", []) if state else []
    bad = sum(1 for e in events if "manipulative" in str(e.get("event", "")).lower())
    return 1.0 if bad == 0 else max(0.0, 1.0 - 0.25 * bad)


def build_rubric():
    """The SharpeBench-calibrated reward bundle: deflated Sharpe (rank) + pass^k +
    process discipline, weighted. Raises if ``verifiers`` is unavailable."""
    if not _HAS_VERIFIERS:
        raise RuntimeError("verifiers is not installed; cannot build a Rubric")
    return vf.Rubric(
        funcs=[deflated_sharpe_reward, pass_k_reward, process_check_reward],
        weights=[1.0, 0.5, 0.5],
    )


def load_environment(dataset: Any = None, **kwargs: Any):
    """``verifiers`` entry point: a single-turn environment backed by the
    SharpeBench-calibrated rubric.

    The rubric is the verified, directly-reusable piece. How a rollout maps a model
    completion to an OpenOutcry trajectory — a single full-strategy turn, or a
    multi-turn per-bar loop driving :class:`~openoutcry.gym.OpenOutcryEnv` — is the
    integration point for your training setup.
    """
    if not _HAS_VERIFIERS:
        raise RuntimeError(
            "verifiers is not installed. Install PrimeIntellect 'verifiers' to load "
            "this environment; the rest of the openoutcry package works without it."
        )
    if dataset is None:
        # A minimal one-row default so the env constructs out of the box; supply a
        # real dataset of market scenarios for training.
        from datasets import Dataset

        dataset = Dataset.from_dict(
            {
                "question": [
                    "Trade the OpenOutcry market scenario to maximize the deflated Sharpe."
                ],
                "answer": [""],
            }
        )
    return vf.SingleTurnEnv(dataset=dataset, rubric=build_rubric(), **kwargs)


__all__ = [
    "deflated_sharpe_reward",
    "pass_k_reward",
    "process_check_reward",
    "build_rubric",
    "load_environment",
]
