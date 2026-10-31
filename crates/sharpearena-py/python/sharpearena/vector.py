"""A gymnasium-vector wrapper around the native batched SharpeArena binding.

:class:`SharpeArenaVectorEnv` adapts the JSON-at-the-boundary native ``VecTradingEnv``
(``B`` independent leak-free lanes) to the :class:`gymnasium.vector.VectorEnv` API:
``reset() -> (obs_batch, infos)`` and ``step(actions) -> (obs_batch, rewards,
terminated, truncated, infos)``, with numpy arrays of leading dim ``B``.

**Autoreset mode — selectable.** ``autoreset_mode`` chooses how a finished lane
(``terminated`` on a blow-up, ``truncated`` on a horizon or episode cap) is recycled:

- ``"next_step"`` (default, Gymnasium 1.x): the terminal step is returned verbatim and
  the reset surfaces on the *following* step (reward 0, flags ``False``, ``first`` ``True``).
- ``"same_step"`` (gym3): the lane resets *in place*, so ``obs_batch[i]`` is already the
  new episode's t0 and ``infos["first"][i]`` is ``True``; the terminal obs/info ride in
  ``infos["final_obs"][i]`` / ``infos["final_info"][i]``.
- ``"disabled"``: the lane never auto-resets and stays at its terminal bar.

Either way the batch never stalls; ``rewards``/``infos`` describe the step that executed.

**Scenario on recycle — selectable.** ``autoreset_scenarios`` chooses what a recycled lane
runs next. ``"replay"`` (default) restarts the lane's scenario, so a run sees only as many
scenarios as lanes. ``"fresh_v1"`` draws a new scenario per recycled episode from a frozen
per-lane seed stream confined to the training band ``[0, EVAL_SEED_BASE)``. It is
training-only: evaluation mode and lanes outside the band are refused. Drawn seeds appear
in ``infos`` and in the native ``effective_configs`` readback, so a run replays.

The action is a per-lane **target-weight vector** over the environment's symbols (shape
``(B, n_symbols)``), converted into the wire-format ``Decision`` JSON the binding expects.
"""

from __future__ import annotations

import json
import operator
from typing import Any, Optional, Sequence

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space

try:  # gymnasium >= 1.0 exposes the AutoresetMode enum; fall back to the string label.
    from gymnasium.vector import AutoresetMode

    _AUTORESET_ENUM = {
        "next_step": AutoresetMode.NEXT_STEP,
        "same_step": AutoresetMode.SAME_STEP,
        "disabled": AutoresetMode.DISABLED,
    }
except Exception:  # noqa: BLE001
    _AUTORESET_ENUM = {}

from .sharpearena_py import VecTradingEnv
from ._action_validation import validate_weight_limit, validated_action
from ._seed_bands import EVAL_SEED_BASE as _EVAL_SEED_BASE

_BUY, _SELL, _HOLD = "buy", "sell", "hold"


def _action_label(weight: float) -> str:
    if weight > 0.0:
        return _BUY
    if weight < 0.0:
        return _SELL
    return _HOLD


class SharpeArenaVectorEnv(VectorEnv):
    """Vectorized gymnasium env over ``B`` leak-free, point-in-time market lanes.

    Pass either an explicit list of ``seeds`` (one synthetic scenario per seed) or
    ``num_envs`` (seeds become ``range(num_envs)``). All lanes share the panel shape,
    window and cost overrides. ``max_weight`` bounds the per-symbol target weight the
    action space allows (set ``allow_short=False`` to clip the lower bound to 0).
    """

    metadata = {"render_modes": [], "autoreset_mode": "next_step"}

    def __init__(
        self,
        num_envs: Optional[int] = None,
        *,
        seeds: Optional[Sequence[int]] = None,
        n_symbols: int = 4,
        n_days: int = 120,
        window_start: Optional[int] = None,
        window_end: Optional[int] = None,
        max_weight: float = 1.0,
        allow_short: bool = True,
        distribution_mode: str = "calm",
        vol_clustering: float = 0.0,
        jump_burst_probability: float = 0.0,
        jump_burst_persistence: float = 0.0,
        jump_burst_size: float = 0.0,
        autoreset_mode: str = "next_step",
        mode: str = "train",
        max_episode_steps: Optional[int] = None,
        env_kwargs: Optional[dict] = None,
        autoreset_scenarios: str = "replay",
    ) -> None:
        if max_episode_steps is not None:
            if isinstance(max_episode_steps, (bool, np.bool_)):
                raise ValueError("max_episode_steps must be a positive integer")
            try:
                max_episode_steps = operator.index(max_episode_steps)
            except TypeError as exc:
                raise ValueError("max_episode_steps must be a positive integer") from exc
            if max_episode_steps <= 0:
                raise ValueError("max_episode_steps must be a positive integer")
        validate_weight_limit(max_weight, allow_short)
        if mode not in ("train", "eval"):
            raise ValueError("mode must be 'train' or 'eval'")
        if mode == "eval" and autoreset_scenarios != "replay":
            raise ValueError(
                "autoreset_scenarios draws from the training band; eval mode must replay"
            )
        if seeds is None:
            if num_envs is None:
                raise ValueError("pass either num_envs or seeds")
            base = _EVAL_SEED_BASE if mode == "eval" else 0
            seeds = list(range(base, base + int(num_envs)))
        else:
            seeds = [int(s) for s in seeds]
            if num_envs is not None and num_envs != len(seeds):
                raise ValueError("num_envs must match len(seeds) when both are given")
        if not seeds:
            raise ValueError("seeds must be non-empty")

        self._seeds = list(seeds)
        self.num_envs = len(self._seeds)
        self._autoreset_mode = autoreset_mode
        self._fresh_scenarios = autoreset_scenarios != "replay"
        self._pending_actions: Optional[np.ndarray] = None
        self.metadata = {
            **self.metadata,
            "autoreset_mode": _AUTORESET_ENUM.get(autoreset_mode, autoreset_mode),
        }

        kwargs: dict[str, Any] = dict(env_kwargs or {})
        self._native_kwargs = dict(
            n_symbols=n_symbols,
            n_days=n_days,
            window_start=window_start,
            window_end=window_end,
            distribution_mode=distribution_mode,
            vol_clustering=float(vol_clustering),
            jump_burst_probability=float(jump_burst_probability),
            jump_burst_persistence=float(jump_burst_persistence),
            jump_burst_size=float(jump_burst_size),
            autoreset_mode=autoreset_mode,
            max_episode_steps=max_episode_steps,
            autoreset_scenarios=autoreset_scenarios,
            **kwargs,
        )
        self._env = VecTradingEnv(seeds=self._seeds, **self._native_kwargs)

        # Discover the symbol axis from the first lane's first observation (every lane
        # shares the same panel shape), so the spaces match the dataset exactly.
        first = json.loads(self._env.reset_batch())
        self._symbols = [s["symbol"] for s in first["observations"][0]["symbols"]]
        n = len(self._symbols)

        low = -max_weight if allow_short else 0.0
        self.single_action_space = spaces.Box(
            low=low, high=max_weight, shape=(n,), dtype=np.float32
        )
        self.single_observation_space = spaces.Dict(
            {
                "closes": spaces.Box(low=0.0, high=np.inf, shape=(n,), dtype=np.float64),
                "positions": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(n,), dtype=np.float64
                ),
                "cash": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64),
            }
        )
        self.action_space = batch_space(self.single_action_space, self.num_envs)
        self.observation_space = batch_space(
            self.single_observation_space, self.num_envs
        )

    # -- internal helpers --------------------------------------------------

    @property
    def symbols(self) -> list[str]:
        return list(self._symbols)

    @property
    def scenario_seeds(self) -> list[int]:
        return list(self._seeds)

    @property
    def autoreset_scenarios(self) -> str:
        return self._env.autoreset_scenarios

    @property
    def effective_configs(self) -> list[dict]:
        """Per-lane readback of what the native batch actually built and, under
        ``fresh_v1``, every seed it has drawn since this batch was constructed."""
        return [json.loads(config) for config in self._env.effective_configs]

    def _decode_obs(self, obs: dict) -> dict[str, np.ndarray]:
        by_symbol = {s["symbol"]: s for s in obs["symbols"]}
        closes = np.array(
            [by_symbol[sym]["close_history"][-1] for sym in self._symbols],
            dtype=np.float64,
        )
        pos = {p["symbol"]: p["shares"] for p in obs.get("portfolio", [])}
        positions = np.array(
            [pos.get(sym, 0.0) for sym in self._symbols], dtype=np.float64
        )
        return {
            "closes": closes,
            "positions": positions,
            "cash": np.array([obs["cash"]], dtype=np.float64),
        }

    def _stack_obs(self, observations: list[dict]) -> dict[str, np.ndarray]:
        decoded = [self._decode_obs(o) for o in observations]
        return {
            "closes": np.stack([d["closes"] for d in decoded]),
            "positions": np.stack([d["positions"] for d in decoded]),
            "cash": np.stack([d["cash"] for d in decoded]),
        }

    def _actions_to_decisions_json(self, actions: np.ndarray) -> str:
        actions = validated_action(actions, self.action_space)
        decisions = [
            {
                "orders": [
                    {
                        "symbol": sym,
                        "action": _action_label(float(w)),
                        "target_weight": float(w),
                        "confidence": 0.5,
                    }
                    for sym, w in zip(self._symbols, lane)
                ],
                "reasoning": "SharpeArenaVectorEnv.step",
            }
            for lane in actions
        ]
        return json.dumps(decisions)

    def _record_draws(self, infos: dict, previous: list[int], recycled, seeds: list[int]) -> None:
        """Fold one step's ``fresh_v1`` draws into ``infos``. ``scenario_seed`` is the
        scenario each returned observation belongs to, so a recycled lane's finished
        episode is named by ``previous`` and carried in ``final_info`` when present."""
        self._seeds = list(seeds)
        infos["scenario_seed"] = np.array(self._seeds, dtype=np.int64)
        drawn = np.empty(self.num_envs, dtype=object)
        drawn[recycled] = np.array(self._seeds, dtype=object)[recycled]
        infos["drawn_scenario_seed"] = drawn
        final_info = infos.get("final_info")
        if final_info is not None:
            for lane in np.flatnonzero(recycled):
                final_info[lane]["scenario_seed"] = previous[lane]

    # -- gymnasium vector API ----------------------------------------------

    def reset(
        self, *, seed: Optional[int | Sequence[int]] = None, options: Optional[dict] = None
    ) -> tuple[dict[str, np.ndarray], dict]:
        """Rebuild native lanes for an integer base seed or an exact per-lane list.

        Seeds are nonnegative signed-64-bit integers, used literally even in eval
        mode. None keeps current scenarios and, under ``fresh_v1``, the stream position.
        Automatic resets draw new seeds only under ``fresh_v1``. An explicit seed reopens
        that stream from the new seeds. A failed seed validation leaves the active batch
        and queued action intact.
        """
        if seed is not None:
            def checked(value):
                if isinstance(value, (bool, np.bool_)):
                    raise ValueError("reset seeds must be nonnegative integers")
                try:
                    value = operator.index(value)
                except TypeError as exc:
                    raise ValueError("reset seeds must be nonnegative integers") from exc
                if not 0 <= value <= np.iinfo(np.int64).max:
                    raise ValueError("reset seeds must fit nonnegative int64")
                return value

            if isinstance(seed, (int, np.integer)) and not isinstance(seed, (bool, np.bool_)):
                base = checked(seed)
                seeds = [checked(base + i) for i in range(self.num_envs)]
                rng_seed = base
            else:
                try:
                    seeds = [checked(value) for value in seed]
                except TypeError as exc:
                    raise ValueError("seed must be an integer or one integer per lane") from exc
                if len(seeds) != self.num_envs:
                    raise ValueError("seed list must match num_envs")
                rng_seed = seeds[0]
            replacement = VecTradingEnv(seeds=seeds, **self._native_kwargs)
            super().reset(seed=rng_seed, options=options)
            self._env = replacement
            self._seeds = seeds
        self._pending_actions = None
        out = json.loads(self._env.reset_batch())
        obs = self._stack_obs(out["observations"])
        infos = {
            "scenario_seed": np.array(self._seeds, dtype=np.int64),
            "first": np.ones(self.num_envs, dtype=bool),
        }
        return obs, infos

    def step_async(self, actions: np.ndarray) -> None:
        """Stash ``actions`` for the next ``step_wait`` without advancing the env, so a
        caller can overlap policy/LLM inference with env stepping. Raises if a previous
        ``step_async`` is still pending (call ``step_wait`` first)."""
        if self._pending_actions is not None:
            raise RuntimeError("step_async called twice without an intervening step_wait")
        self._pending_actions = validated_action(actions, self.action_space)

    def step_wait(
        self,
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, dict]:
        """Run the batched step queued by ``step_async`` and return the 5-tuple. Raises
        if there is no pending ``step_async``."""
        if self._pending_actions is None:
            raise RuntimeError("step_wait called without a pending step_async")
        actions = self._pending_actions
        self._pending_actions = None
        out = json.loads(self._env.step_batch(self._actions_to_decisions_json(actions)))
        obs = self._stack_obs(out["observations"])
        rewards = np.asarray(out["rewards"], dtype=np.float64)
        terminated = np.asarray(out["terminated"], dtype=bool)
        truncated = np.asarray(out["truncated"], dtype=bool)
        infos = {
            "scenario_seed": np.array(self._seeds, dtype=np.int64),
            "first": np.asarray(out["first"], dtype=bool),
            "nav": np.array([i["nav"] for i in out["infos"]], dtype=np.float64),
        }
        if self._autoreset_mode == "same_step":
            # SAME_STEP surfaces each finished lane's terminal obs/info alongside the
            # already-reset batch (None for lanes that did not finish this step).
            infos["final_obs"] = np.array(
                [self._decode_obs(o) if o is not None else None for o in out["final_obs"]],
                dtype=object,
            )
            infos["final_info"] = np.array(out["final_info"], dtype=object)
        if self._fresh_scenarios:
            self._seeds = out["scenario_seeds"]
            infos["scenario_seed"] = np.array(self._seeds, dtype=np.int64)
            infos["drawn_scenario_seed"] = np.array(out["drawn_seeds"], dtype=object)
        return obs, rewards, terminated, truncated, infos

    def step_array(
        self, actions: np.ndarray
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, dict]:
        """Run one batched step through the binary path, with no JSON at the boundary.

        Takes the same actions and returns the same batch as :meth:`step`, byte for byte,
        including recycling and ``fresh_v1`` draws, and leaves the engine in the same
        state. ``final_info`` entries carry ``nav`` but no ``events``: process events stay
        in the engine's trace, so use :meth:`step` to read them. :meth:`step` stays the
        canonical wire format. Raises if a ``step_async`` is pending.
        """
        if self._pending_actions is not None:
            raise RuntimeError("step_array called with a step_async pending")
        weights = validated_action(actions, self.action_space)
        b, n = self.num_envs, len(self._symbols)
        closes = np.empty((b, n), dtype=np.float64)
        positions = np.empty((b, n), dtype=np.float64)
        cash = np.empty((b, 1), dtype=np.float64)
        rewards = np.empty(b, dtype=np.float64)
        navs = np.empty(b, dtype=np.float64)
        terminated = np.empty(b, dtype=np.uint8)
        truncated = np.empty(b, dtype=np.uint8)
        first = np.empty(b, dtype=np.uint8)
        same_step = self._autoreset_mode == "same_step"
        finals = (
            {
                "final_closes": np.empty((b, n), dtype=np.float64),
                "final_positions": np.empty((b, n), dtype=np.float64),
                "final_cash": np.empty((b, 1), dtype=np.float64),
                "final_navs": np.empty(b, dtype=np.float64),
            }
            if same_step
            else {}
        )
        self._env.step_batch_array(
            weights,
            closes,
            positions,
            cash,
            rewards,
            terminated,
            truncated,
            first,
            navs,
            reasoning="SharpeArenaVectorEnv.step",
            **finals,
        )
        recycled = first.view(bool)
        infos = {
            "scenario_seed": np.array(self._seeds, dtype=np.int64),
            "first": recycled,
            "nav": navs,
        }
        if same_step:
            final_obs = np.empty(b, dtype=object)
            final_info = np.empty(b, dtype=object)
            for lane in np.flatnonzero(recycled):
                final_obs[lane] = {
                    "closes": finals["final_closes"][lane],
                    "positions": finals["final_positions"][lane],
                    "cash": finals["final_cash"][lane],
                }
                final_info[lane] = {"nav": float(finals["final_navs"][lane])}
            infos["final_obs"] = final_obs
            infos["final_info"] = final_info
        if self._fresh_scenarios:
            self._record_draws(infos, self._seeds, recycled, self._env.scenario_seeds)
        obs = {"closes": closes, "positions": positions, "cash": cash}
        return obs, rewards, terminated.view(bool), truncated.view(bool), infos

    def step(
        self, actions: np.ndarray
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, dict]:
        self.step_async(actions)
        return self.step_wait()

    def render(self):  # pragma: no cover - no visual rendering
        return None

    def close(self, **kwargs):  # pragma: no cover
        return None
