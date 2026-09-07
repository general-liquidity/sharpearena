"""Trusted, offline Gym replay inputs for producer regressions.

These inputs include seeds and possibly a complete CSV. Keep them operator-only,
like private checkpoints. This is a fixed local adapter, not a plugin loader or
an isolation boundary. It replays supplied actions, never an agent or a model.
"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from typing import Any, Mapping, Sequence

import numpy as np

from .checkpoint import _build_env, _extract_params
from .gym import SharpeArenaEnv
from .trace import _jsonify

GYM_REPLAY_PRODUCER = "sharpearena.gym-actions/1"
GYM_PARAM_FIELDS = frozenset(
    {
        "n_symbols",
        "n_days",
        "seed",
        "window_start",
        "window_end",
        "csv_text",
        "max_weight",
        "allow_short",
        "distribution_mode",
        "vol_clustering",
        "jump_burst_probability",
        "jump_burst_persistence",
        "jump_burst_size",
        "mode",
        "env_kwargs",
    }
)


def gym_replay_inputs(env: SharpeArenaEnv, actions: Sequence[Any]) -> dict[str, Any]:
    """Capture construction parameters and a complete action prefix from reset.

    The caller supplies the prefix, not just the actions in a minimized excerpt.
    Promotion additionally reruns this recipe and compares its complete output to
    the source trace before an operator can approve the resulting candidate.
    """
    from ._action_validation import validated_action

    inputs = {
        "producer": GYM_REPLAY_PRODUCER,
        "params": _extract_params(env),
        "actions": [
            validated_action(action, env.action_space).tolist() for action in actions
        ],
    }
    validate_replay_inputs(inputs)
    return inputs


def validate_replay_inputs(inputs: Mapping[str, Any]) -> None:
    from .trace_promotion import PromotionError

    if not isinstance(inputs, dict) or set(inputs) != {"producer", "params", "actions"}:
        raise PromotionError(
            "replay inputs require exactly producer, params and actions"
        )
    if inputs["producer"] != GYM_REPLAY_PRODUCER:
        raise PromotionError(
            "unsupported replay producer; no code is loaded from case data"
        )
    if (
        not isinstance(inputs["params"], dict)
        or set(inputs["params"]) != GYM_PARAM_FIELDS
    ):
        raise PromotionError(
            "replay inputs require every captured Gym construction parameter"
        )
    actions = inputs["actions"]
    if not isinstance(actions, list) or len(actions) < 2:
        raise PromotionError(
            "replay inputs require at least two ordered actions from reset"
        )
    for action in actions:
        try:
            valid = (
                isinstance(action, list)
                and bool(action)
                and all(
                    type(value) in (int, float) and math.isfinite(value)
                    for value in action
                )
            )
        except OverflowError:
            valid = False
        if not valid:
            raise PromotionError(
                "replay actions must be nonempty finite numeric vectors"
            )
    try:
        json.dumps(inputs, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as error:
        raise PromotionError("replay inputs must be finite JSON data") from error


def replay_gym_inputs(inputs: Mapping[str, Any], meta: Mapping[str, Any]):
    """Rerun the installed Gym/native producer and return its new trace.

    Observations are the pre-action values the policy would see. Metadata other
    than the effective construction config and step count is supplied lineage,
    not an independently authenticated claim about the original run.
    """
    from .trace_promotion import PromotionError, StrictTrace, _digest

    validate_replay_inputs(inputs)
    params = deepcopy(inputs["params"])
    env = _build_env(params)
    try:
        effective = _extract_params(env)
        if _digest(effective) != _digest(params):
            raise PromotionError(
                "replay construction readback differs from frozen inputs"
            )
        obs, _ = env.reset()
        steps = []
        for ordinal, action in enumerate(inputs["actions"]):
            # Capture before step: an environment may mutate its observation buffers.
            observed = deepcopy(_jsonify(obs))
            decision = json.loads(env._action_to_decision_json(np.asarray(action)))
            obs, reward, terminated, truncated, info = env.step(np.asarray(action))
            steps.append(
                {
                    "kind": "step",
                    "step": ordinal,
                    "observation": observed,
                    "decision": decision,
                    "reward": reward,
                    "info": {
                        **_jsonify(info),
                        "terminated": terminated,
                        "truncated": truncated,
                    },
                }
            )
            if (terminated or truncated) and ordinal + 1 != len(inputs["actions"]):
                raise PromotionError(
                    "replay producer ended before the frozen action prefix"
                )
        actual_meta = deepcopy(dict(meta))
        actual_meta["config"] = effective
        actual_meta["n_steps"] = len(steps)
        # Outputs may contain a nonfinite reward: the named check must see it,
        # rather than a serialization exception silently substituting zero.
        return StrictTrace(tuple(steps), actual_meta, _digest(inputs))
    finally:
        env.close()


__all__ = ["GYM_REPLAY_PRODUCER", "gym_replay_inputs", "replay_gym_inputs"]
