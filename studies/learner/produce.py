#!/usr/bin/env python3
"""Producer for the learner generalization study. See PROTOCOL.md and protocol.json.

    python studies/learner/produce.py --stage smoke --workers 4
    python studies/learner/produce.py --stage pilot --workers 32
    python studies/learner/produce.py --stage pilot-decide
    python studies/learner/produce.py --stage main --workers 32

A job is one training run, or the closed-form controls of one seed band. Finished jobs are
skipped; an interrupted training run restarts from its seed and reproduces the same bytes.

MERGE DEPENDENCY: ``pilot`` and ``main`` need ``state_dict``, ``load_state_dict`` and
``training`` on ``CausalNormalizeObservation`` from branch ``feat/normalizer-freeze``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import common  # noqa: E402
from common import Job, StudyRefused  # noqa: E402

PROTOCOL_PATH = HERE / "protocol.json"
EPS = 1e-8


# -- the normaliser boundary --------------------------------------------------


class SmokeFreezeShim:
    """Smoke-only stand-in with the ``feat/normalizer-freeze`` interface.

    Exists so the pipeline can be exercised before that branch merges. ``resolve_normalizer``
    refuses it outside ``smoke``.
    """

    def __new__(cls, env: Any, *, training: bool = True):
        import gymnasium as gym
        from sharpearena.wrappers import CausalNormalizeObservation

        class Shim(CausalNormalizeObservation):
            def __init__(self, inner: gym.Env, training: bool) -> None:
                super().__init__(inner)
                self.training = training

            def _normalize(self, obs: dict) -> dict:
                if self.training:
                    return super()._normalize(obs)
                out = {}
                for key, value in obs.items():
                    stat = self._stats[key]
                    out[key] = (np.asarray(value, np.float64) - stat.mean) / (stat.std() + EPS)
                return out

            def state_dict(self) -> dict:
                return {
                    k: {"count": s.count, "mean": s.mean.tolist(), "m2": s.m2.tolist()}
                    for k, s in self._stats.items()
                }

            def load_state_dict(self, state: dict) -> None:
                for key, row in state.items():
                    stat = self._stats[key]
                    stat.count = int(row["count"])
                    stat.mean = np.asarray(row["mean"], np.float64)
                    stat.m2 = np.asarray(row["m2"], np.float64)

        return Shim(env, training)


def resolve_normalizer(stage: str) -> tuple[Callable[..., Any], str]:
    from sharpearena.wrappers import CausalNormalizeObservation

    missing = [
        name for name in ("state_dict", "load_state_dict") if not hasattr(CausalNormalizeObservation, name)
    ]
    if not missing:
        return CausalNormalizeObservation, "sharpearena"
    if stage == "smoke":
        return SmokeFreezeShim, "smoke_shim"
    raise StudyRefused(
        f"stage {stage} needs feat/normalizer-freeze merged: CausalNormalizeObservation lacks {missing}"
    )


# -- reference policies -------------------------------------------------------


class BuyAndHold:
    """Equal weight at t0, then the weights the closes drift to. Never rebalances."""

    name = "buy_and_hold"

    def __init__(self) -> None:
        self._first: np.ndarray | None = None

    def __call__(self, obs: dict) -> np.ndarray:
        closes = np.asarray(obs["closes"], dtype=np.float64).reshape(-1)
        if self._first is None:
            self._first = closes.copy()
        held = closes / self._first
        return (held / held.sum()).astype(np.float32)


class RandomAllocator:
    """A fully invested long-only draw each bar. One member of the zero-skill floor."""

    name = "random_allocator"

    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng

    def __call__(self, obs: dict) -> np.ndarray:
        n = int(np.asarray(obs["closes"]).reshape(-1).shape[0])
        return self._rng.dirichlet(np.ones(n)).astype(np.float32)


def tape_closes(seed: int, env_spec: dict, tier: str) -> np.ndarray:
    from sharpearena import SharpeArenaEnv

    env = SharpeArenaEnv(
        n_symbols=env_spec["n_symbols"], n_days=env_spec["n_days"], seed=seed, distribution_mode=tier
    )
    obs, _ = env.reset()
    rows = [np.asarray(obs["closes"], np.float64).reshape(-1)]
    flat = np.zeros(env_spec["n_symbols"], dtype=np.float32)
    while True:
        obs, _, terminated, truncated, _ = env.step(flat)
        rows.append(np.asarray(obs["closes"], np.float64).reshape(-1))
        if terminated or truncated:
            return np.stack(rows)


def memoriser_actions(closes: np.ndarray, cost: float = 0.0) -> np.ndarray:
    """Foresight on one known tape: per symbol, the best path over short, flat and long.

    Each symbol holds ``-1/n``, ``0`` or ``1/n`` and earns its next-bar return, less ``cost``
    per unit of weight traded. With ``cost = 0`` this is the next-bar sign. With the
    environment's cost it stops trading moves too small to pay for, which the plain sign
    rule cannot: on Calm that rule earns 31.5 bp gross and loses 4.1 bp net. An instrument
    check only, never a policy.
    """
    returns = np.diff(closes, axis=0) / closes[:-1]
    bars, n = returns.shape
    levels = np.array([-1.0, 0.0, 1.0]) / n
    out = np.zeros((bars, n))
    switch = cost * np.abs(levels[:, None] - levels[None, :])
    for c in range(n):
        value = -cost * np.abs(levels) + levels * returns[0, c]
        back = np.zeros((bars, 3), dtype=np.int64)
        for t in range(1, bars):
            total = value[None, :] - switch
            back[t] = np.argmax(total, axis=1)
            value = total[np.arange(3), back[t]] + levels * returns[t, c]
        state = int(np.argmax(value))
        for t in range(bars - 1, -1, -1):
            out[t, c] = levels[state]
            state = int(back[t, state])
    return out.astype(np.float32)


def cost_per_turnover(seed: int, env_spec: dict, tier: str) -> float:
    """The environment's cost per unit of weight traded, read from one probe episode.

    Alternate a full long and a full short book every bar, then divide what the ledger lost
    against the frictionless return by the weight traded. A measurement of the engine, run
    on a reserved seed, never on a seed any arm is scored on.
    """
    closes = tape_closes(seed, env_spec, tier)
    bars, n = closes.shape[0] - 1, closes.shape[1]
    sign = np.where(np.arange(bars) % 2 == 0, 1.0, -1.0)[:, None]
    actions = np.repeat(sign, n, axis=1) / n
    returns = np.diff(closes, axis=0) / closes[:-1]
    gross = float((actions * returns).sum(axis=1).mean())
    net = float(np.mean(rollout_fixed(seed, env_spec, tier, actions.astype(np.float32))))
    traded = np.abs(np.diff(np.vstack([np.zeros(n), actions]), axis=0)).sum(axis=1).mean()
    return max(0.0, (gross - net) / float(traded))


# -- rollouts -----------------------------------------------------------------


def behaviour(rewards: list[float], actions: list[np.ndarray]) -> dict:
    """The per-episode line every evaluation keeps, including the inactivity measures."""
    returns = np.asarray(rewards, dtype=np.float64)
    weights = np.asarray(actions, dtype=np.float64)
    gross = np.abs(weights).sum(axis=1)
    turnover = (
        0.5 * np.abs(np.diff(weights, axis=0)).sum(axis=1) if len(weights) > 1 else np.zeros(1)
    )
    return {
        "bars": int(returns.size),
        "mean_bp": float(returns.mean() * 1e4),
        "gross": float(gross.mean()),
        "turnover": float(turnover.mean()),
        "action_std": float(weights.std(axis=0).mean()),
    }


def rollout_fixed(seed: int, env_spec: dict, tier: str, actions: np.ndarray) -> list[float]:
    """Replay a recorded weight sequence on the raw environment."""
    from sharpearena import SharpeArenaEnv

    env = SharpeArenaEnv(
        n_symbols=env_spec["n_symbols"], n_days=env_spec["n_days"], seed=seed, distribution_mode=tier
    )
    env.reset()
    rewards = []
    for weights in actions:
        _, reward, terminated, truncated, _ = env.step(np.asarray(weights, dtype=np.float32))
        rewards.append(float(reward))
        if terminated or truncated:
            break
    return rewards


def rollout_policy(seed: int, env_spec: dict, tier: str, policy: Callable[[dict], np.ndarray]) -> tuple[list[float], list[np.ndarray]]:
    from sharpearena import SharpeArenaEnv

    env = SharpeArenaEnv(
        n_symbols=env_spec["n_symbols"], n_days=env_spec["n_days"], seed=seed, distribution_mode=tier
    )
    obs, _ = env.reset()
    rewards, actions = [], []
    while True:
        action = np.asarray(policy(obs), dtype=np.float32)
        obs, reward, terminated, truncated, _ = env.step(action)
        rewards.append(float(reward))
        actions.append(action)
        if terminated or truncated:
            return rewards, actions


def fit_normalizer(make: Callable[..., Any], seeds: Sequence[int], env_spec: dict, tier: str, seed: int) -> dict:
    """Fit once on the seen band under a seeded random allocator, then hand back the state."""
    from sharpearena import SharpeArenaEnv

    rng = np.random.default_rng((0x4E4F, seed))
    inner = SharpeArenaEnv(
        n_symbols=env_spec["n_symbols"], n_days=env_spec["n_days"], seed=seeds[0], distribution_mode=tier
    )
    env = make(inner, training=True)
    policy = RandomAllocator(rng)
    for scenario in seeds:
        obs, _ = env.reset(seed=int(scenario))
        while True:
            obs, _, terminated, truncated, _ = env.step(policy({"closes": np.zeros(env_spec["n_symbols"])}))
            if terminated or truncated:
                break
    return env.state_dict()


def frozen_env(make: Callable[..., Any], state: dict, env_spec: dict, tier: str, seed: int):
    from sharpearena import SharpeArenaEnv

    env = make(
        SharpeArenaEnv(
            n_symbols=env_spec["n_symbols"], n_days=env_spec["n_days"], seed=seed, distribution_mode=tier
        ),
        training=False,
    )
    env.load_state_dict(state)
    return env


def make_training_env(make, state: dict, seeds: Sequence[int], env_spec: dict, tier: str, lanes: int, scale: float):
    import gymnasium as gym
    from stable_baselines3.common.vec_env import DummyVecEnv

    class Lane(gym.Wrapper):
        """Deals this lane's seen seeds in turn. Scales reward for training alone."""

        def __init__(self, env: gym.Env, assigned: list[int]) -> None:
            super().__init__(env)
            self._assigned = assigned
            self._next = 0

        def reset(self, *, seed=None, options=None):
            scenario = self._assigned[self._next % len(self._assigned)]
            self._next += 1
            return self.env.reset(seed=scenario)

        def step(self, action):
            obs, reward, terminated, truncated, info = self.env.step(action)
            return obs, float(reward) * scale, terminated, truncated, info

    count = min(lanes, len(seeds))
    dealt = [[int(s) for s in seeds[i::count]] for i in range(count)]

    def builder(assigned: list[int]):
        return lambda: Lane(frozen_env(make, state, env_spec, tier, assigned[0]), assigned)

    return DummyVecEnv([builder(a) for a in dealt])


def evaluate_model(model, make, state: dict, seeds: Sequence[int], env_spec: dict, tier: str, keep: int = 0) -> dict:
    """Deterministic policy on each seed, frozen normaliser, unscaled ledger returns."""
    rows, pooled, recorded = [], [], []
    for index, seed in enumerate(seeds):
        env = frozen_env(make, state, env_spec, tier, int(seed))
        obs, _ = env.reset()
        rewards, actions = [], []
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            rewards.append(float(reward))
            actions.append(np.asarray(action, dtype=np.float32))
            if terminated or truncated:
                break
        rows.append({"seed": int(seed), **behaviour(rewards, actions)})
        pooled.extend(rewards)
        if index < keep:
            recorded.append(np.stack(actions))
    return {"rows": rows, "pooled": pooled, "actions": recorded}


def deflated(pooled: list[float], n_trials: int) -> dict:
    """The kernel's deflated Sharpe, or the kernel's reason it has none. Never a zero."""
    from sharpearena import kernel_deflated_sharpe, score_run

    try:
        return {"dsr": kernel_deflated_sharpe(json.loads(score_run(pooled, n_trials)))}
    except Exception as exc:  # noqa: BLE001 - typed below, never coerced
        return {"unavailable": type(exc).__name__, "detail": str(exc)[:200]}


# -- timing null --------------------------------------------------------------


def holding_periods(gross: Sequence[float], floor: float) -> list[tuple[int, int]]:
    """Maximal invested runs as ``(first bar, length)``, as Bench's ``holding_periods``."""
    periods, start = [], None
    for bar, level in enumerate(gross):
        if level > floor and start is None:
            start = bar
        elif level <= floor and start is not None:
            periods.append((start, bar - start))
            start = None
    if start is not None:
        periods.append((start, len(gross) - start))
    return periods


def random_layout(lengths: Sequence[int], bars: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    """Mirror of Bench's ``random_layout``: a random order, with flat gaps as a uniform weak
    composition and at least one flat bar between periods. Same law, not the same stream."""
    count = len(lengths)
    order = rng.permutation(count)
    spare = bars - int(sum(lengths)) + 1 - count
    separators = sorted(int(x) for x in rng.choice(spare + count, size=count, replace=False))
    layout, cursor, previous = [], 0, None
    for position, (period, separator) in enumerate(zip(order, separators)):
        before = separator if previous is None else separator - previous - 1
        cursor += before + (1 if position > 0 else 0)
        layout.append((int(period), cursor))
        cursor += lengths[period]
        previous = separator
    return layout


def moved_actions(actions: np.ndarray, floor: float, rng: np.random.Generator) -> tuple[np.ndarray, str]:
    """One null placement of one episode, and which construction produced it."""
    gross = np.abs(actions).sum(axis=1)
    periods = holding_periods(gross, floor)
    bars = len(actions)
    if not periods:
        return actions.copy(), "never_invested"
    if len(periods) == 1 and periods[0][1] == bars:
        return np.roll(actions, int(rng.integers(1, bars)), axis=0), "always_invested:circular_shift"
    placed = np.zeros_like(actions)
    lengths = [p[1] for p in periods]
    for period, first in random_layout(lengths, bars, rng):
        source, length = periods[period]
        placed[first : first + length] = actions[source : source + length]
    return placed, "moved_holding_periods"


def timing_null(recorded: list[np.ndarray], seeds: Sequence[int], env_spec: dict, tier: str, draws: int, floor: float, seed: int) -> dict:
    """Place the realised held-out mean among mean returns at random timing, same exposure."""
    rng = np.random.default_rng((0x7154, seed))
    observed = float(
        np.mean([np.mean(rollout_fixed(s, env_spec, tier, a)) for s, a in zip(seeds, recorded)]) * 1e4
    )
    values, kinds = [], set()
    for _ in range(draws):
        means = []
        for scenario, actions in zip(seeds, recorded):
            placed, kind = moved_actions(actions, floor, rng)
            kinds.add(kind)
            means.append(np.mean(rollout_fixed(scenario, env_spec, tier, placed)))
        values.append(float(np.mean(means) * 1e4))
    if kinds == {"never_invested"}:
        return {"unavailable": "never_invested", "observed_bp": observed}
    below = sum(1 for v in values if v < observed)
    ties = sum(1 for v in values if v == observed)
    return {
        "observed_bp": observed,
        "percentile": (below + 0.5 * ties) / draws,
        "null_mean_bp": float(np.mean(values)),
        "draws": draws,
        "constructions": sorted(kinds),
    }


def closes_are_exogenous(seed: int, env_spec: dict, tier: str) -> bool:
    """The null's validity condition: trading must not move the closes it is marked at."""
    from sharpearena import SharpeArenaEnv

    def path(weight: float) -> list[bytes]:
        env = SharpeArenaEnv(
            n_symbols=env_spec["n_symbols"], n_days=env_spec["n_days"], seed=seed, distribution_mode=tier
        )
        obs, _ = env.reset()
        out = [np.asarray(obs["closes"]).tobytes()]
        action = np.full(env_spec["n_symbols"], weight, dtype=np.float32)
        while True:
            obs, _, terminated, truncated, _ = env.step(action)
            out.append(np.asarray(obs["closes"]).tobytes())
            if terminated or truncated:
                return out

    return path(0.0) == path(0.25)


# -- jobs ---------------------------------------------------------------------


def policy_digest(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.policy.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def run_controls(p: dict) -> dict:
    """Closed-form policies and the random floor on one band. No training involved."""
    from sharpearena.baselines import EqualWeightLongPolicy, FlatPolicy, KellyVolTargetPolicy

    env_spec, tier, seeds = p["env"], p["tier"], p["seeds"]
    makers = {
        "equal_weight": EqualWeightLongPolicy,
        "flat": FlatPolicy,
        "buy_and_hold": BuyAndHold,
        "kelly_vol_target": KellyVolTargetPolicy,
    }
    rows: dict[str, list[dict]] = {}
    for name, maker in makers.items():
        rows[name] = []
        for seed in seeds:
            rewards, actions = rollout_policy(seed, env_spec, tier, maker())
            rows[name].append({"seed": seed, **behaviour(rewards, actions)})
    rows["memoriser_if_seen"] = []
    cost = cost_per_turnover(p["cost_probe_seed"], env_spec, tier)
    for seed in seeds:
        actions = memoriser_actions(tape_closes(seed, env_spec, tier), cost)
        rows["memoriser_if_seen"].append({"seed": seed, **behaviour(rollout_fixed(seed, env_spec, tier, actions), list(actions))})
    floor = []
    for draw in range(p["floor_draws"]):
        rng = np.random.default_rng((0xF100, p["band_tag"], draw))
        values = [rollout_policy(s, env_spec, tier, RandomAllocator(rng))[0] for s in seeds]
        floor.append(float(np.mean([np.mean(v) for v in values]) * 1e4))
    return {"band": p["band"], "tier": tier, "rows": rows, "floor_draws_bp": floor, "cost_per_turnover": cost}


def run_training(p: dict) -> dict:
    """One independent training run: fit, train, checkpoint on dev, evaluate final and best."""
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.utils import set_random_seed

    torch.set_num_threads(p["algorithm"]["torch_threads"])
    make, backend = resolve_normalizer(p["stage"])
    env_spec, tier = p["env"], p["tier"]
    ppo, config = p["ppo"], p["config"]
    seed = p["train_seed"]
    set_random_seed(seed)
    state = fit_normalizer(make, p["seen"], env_spec, tier, seed)
    train_env = make_training_env(make, state, p["seen"], env_spec, tier, p["lanes"], ppo["reward_scale"])
    arch = list(p["net_arch"])
    model = PPO(
        p["algorithm"]["policy"],
        train_env,
        learning_rate=config["learning_rate"],
        ent_coef=config["ent_coef"],
        n_steps=ppo["n_steps"],
        batch_size=ppo["batch_size"],
        n_epochs=ppo["n_epochs"],
        gamma=ppo["gamma"],
        gae_lambda=ppo["gae_lambda"],
        clip_range=ppo["clip_range"],
        policy_kwargs={"net_arch": {"pi": arch, "vf": arch}, "log_std_init": ppo["log_std_init"]},
        seed=seed,
        device=p["algorithm"]["device"],
        verbose=0,
    )
    per_chunk = p["env_steps"] // p["checkpoints"]
    curve, best_value, best_state, best_index = [], -math.inf, None, -1
    for index in range(p["checkpoints"]):
        model.learn(total_timesteps=per_chunk, reset_num_timesteps=index == 0)
        dev = evaluate_model(model, make, state, p["dev"], env_spec, tier)
        dev_mean = float(np.mean([r["mean_bp"] for r in dev["rows"]]))
        seen = evaluate_model(model, make, state, p["seen"][:16], env_spec, tier)
        curve.append(
            {
                "checkpoint": index,
                "env_steps": int(model.num_timesteps),
                "dev_mean_bp": dev_mean,
                "seen_mean_bp": float(np.mean([r["mean_bp"] for r in seen["rows"]])),
            }
        )
        if dev_mean > best_value:
            best_value, best_index = dev_mean, index
            best_state = {k: v.clone() for k, v in model.policy.state_dict().items()}
    final_digest = policy_digest(model)
    keep = p["timing_episodes"]
    out: dict[str, Any] = {
        "cell": p["cell"], "band": p["band_index"], "run": p["run_index"], "config": p["config_name"],
        "train_seed": seed, "normalizer_backend": backend, "env_steps": int(model.num_timesteps),
        "curve": curve, "best_checkpoint": best_index,
    }

    def read(tag: str) -> dict:
        seen = evaluate_model(model, make, state, p["seen"], env_spec, tier)
        unseen = evaluate_model(model, make, state, p["unseen"], env_spec, tier, keep=keep)
        return {
            "policy_sha256": policy_digest(model),
            "seen": seen["rows"],
            "unseen": unseen["rows"],
            "seen_dsr": deflated(seen["pooled"], p["n_trials"][tag]),
            "unseen_dsr": deflated(unseen["pooled"], p["n_trials"][tag]),
        }, unseen

    final, unseen_final = read("final")
    out["final"] = final
    if not closes_are_exogenous(p["unseen"][0], env_spec, tier):
        raise StudyRefused("own orders moved the closes, so the timing null is invalid")
    out["timing_null"] = timing_null(
        unseen_final["actions"], p["unseen"][:keep], env_spec, tier, p["timing_draws"], p["invested_floor"], seed
    )
    out["transfer"] = {}
    for other in p["transfer_tiers"]:
        rows = evaluate_model(model, make, state, p["unseen"], env_spec, other)
        out["transfer"][other] = rows["rows"]
    if best_index == p["checkpoints"] - 1:
        out["best"] = {"same_as_final": True, "policy_sha256": final_digest}
    else:
        model.policy.load_state_dict(best_state)
        out["best"], _ = read("best")
    return out


def run_job(p: dict) -> dict:
    return run_controls(p) if p["kind"] == "controls" else run_training(p)


# -- design -------------------------------------------------------------------


def stage_design(protocol: dict, stage: str, decision: dict | None, smoke_steps: int) -> dict:
    bands = protocol["bands"]
    controls = protocol["controls"]
    if stage == "smoke":
        return {
            "seen_start": bands["smoke_seen"]["start"], "seen_width": bands["smoke_seen"]["width_per_band"],
            "unseen": common.band(bands["smoke_unseen"]["start"], bands["smoke_unseen"]["count"]),
            "dev": common.band(bands["smoke_unseen"]["start"] + 100, 4),
            "n_bands": 2, "first_band": 0, "runs_per_band": 2, "env_steps": smoke_steps, "checkpoints": 2,
            "floor_draws": 4, "timing_draws": 4, "timing_episodes": 2, "configs": ["c0"], "cells": ["A", "D"],
        }
    dev = common.band(bands["dev"]["start"], bands["dev"]["count"])
    common_part = {
        "floor_draws": controls["random_floor_draws"], "timing_draws": controls["timing_null_draws"],
        "timing_episodes": controls["timing_null_episodes"], "checkpoints": protocol["budget"]["checkpoints"], "dev": dev,
    }
    if stage == "main":
        if decision is None:
            raise StudyRefused("main needs a committed pilot_decision.json")
        return {
            **common_part, "seen_start": bands["main_seen_16"]["start"], "seen_width": bands["main_seen_16"]["width_per_band"],
            "unseen": common.band(bands["heldout"]["start"], bands["heldout"]["count"]),
            "n_bands": decision["bands"], "first_band": 0, "runs_per_band": protocol["design"]["runs_per_band"],
            "env_steps": decision["env_steps"], "configs": [decision["config"]], "cells": decision["cells"],
        }
    raise StudyRefused(f"no design for {stage}")


def pilot_design(protocol: dict, part: str, config: str | None, env_steps: int) -> dict:
    bands, controls, pilot = protocol["bands"], protocol["controls"], protocol["pilot"]
    search = part == "search"
    return {
        "seen_start": bands["pilot_seen"]["start"], "seen_width": bands["pilot_seen"]["width_per_band"],
        "unseen": common.band(bands["pilot_unseen"]["start"], bands["pilot_unseen"]["count"]),
        "dev": common.band(bands["dev"]["start"], bands["dev"]["count"]),
        "n_bands": pilot["config_search_bands"] if search else pilot["bands"],
        "first_band": 0 if search else pilot["config_search_bands"],
        "runs_per_band": pilot["config_search_runs_per_band"] if search else pilot["runs_per_band"],
        "env_steps": env_steps, "checkpoints": protocol["budget"]["checkpoints"],
        "floor_draws": controls["random_floor_draws"], "timing_draws": controls["timing_null_draws"],
        "timing_episodes": controls["timing_null_episodes"],
        "configs": list(protocol["configs"]) if search else [config], "cells": ["A"],
    }


def seen_seeds(design: dict, cell: dict, band_index: int, protocol: dict, stage: str) -> list[int]:
    width = cell["n_seen"]
    if stage == "smoke":
        width = design["seen_width"]
    elif width != design["seen_width"]:
        block = protocol["bands"]["main_seen_256"]
        return common.band(block["start"] + band_index * block["width_per_band"], width)
    return common.band(design["seen_start"] + band_index * design["seen_width"], width)


def build_jobs(protocol: dict, stage: str, design: dict, n_trials_runs: int, prefix: str) -> tuple[list[Job], list[Job]]:
    env_spec = protocol["env"]
    control_bands: dict[str, tuple[str, list[int]]] = {}
    training: list[Job] = []
    run_tag = 0
    for cell_name in design["cells"]:
        cell = protocol["cells"][cell_name]
        tier = cell["tier"]
        control_bands[f"{tier}/unseen"] = (tier, design["unseen"])
        control_bands[f"{tier}/dev"] = (tier, design["dev"])
        for config_name in design["configs"]:
            for band_index in range(design["first_band"], design["first_band"] + design["n_bands"]):
                seen = seen_seeds(design, cell, band_index, protocol, stage)
                control_bands[f"{tier}/seen_{cell_name}_{band_index}"] = (tier, seen)
                common.require_disjoint({"seen": seen, "dev": design["dev"], "unseen": design["unseen"]})
                for run_index in range(design["runs_per_band"]):
                    train_seed = int.from_bytes(
                        hashlib.sha256(
                            f"{protocol['protocol_id']}/{stage}/{prefix}/{cell_name}/{config_name}/{band_index}/{run_index}".encode()
                        ).digest()[:4],
                        "big",
                    ) >> 1
                    run_tag += 1
                    training.append(
                        Job(
                            f"{prefix}runs/{cell_name}_{config_name}_b{band_index}_r{run_index}",
                            {
                                "kind": "training", "stage": stage, "cell": cell_name, "band_index": band_index, "run_index": run_index,
                                "config_name": config_name, "config": protocol["configs"][config_name], "net_arch": cell["net_arch"],
                                "tier": tier, "transfer_tiers": list(cell["transfer_tiers"]),
                                "env": env_spec, "algorithm": protocol["algorithm"], "ppo": protocol["ppo"], "lanes": protocol["lanes"],
                                "seen": seen, "dev": design["dev"], "unseen": design["unseen"], "train_seed": train_seed,
                                "env_steps": design["env_steps"], "checkpoints": design["checkpoints"],
                                "timing_draws": design["timing_draws"], "timing_episodes": design["timing_episodes"],
                                "invested_floor": protocol["controls"]["invested_gross_floor"],
                                "n_trials": {"final": n_trials_runs, "best": n_trials_runs * design["checkpoints"]},
                            },
                        )
                    )
    controls = [
        Job(
            f"{prefix}controls/{name}",
            {"kind": "controls", "band": name, "seeds": seeds, "env": env_spec, "tier": tier,
             "cost_probe_seed": protocol["bands"]["cost_probe"]["start"],
             "floor_draws": design["floor_draws"], "band_tag": int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)},
        )
        for name, (tier, seeds) in control_bands.items()
    ]
    return controls, training


# -- analysis -----------------------------------------------------------------


def excess_by_seed(rows: list[dict], reference: list[dict]) -> list[float]:
    base = {r["seed"]: r["mean_bp"] for r in reference}
    return [r["mean_bp"] - base[r["seed"]] for r in rows]


def nested_summary(values: dict[tuple[int, int], float], confidence: float) -> dict:
    """Interval over band means, with band and run variance beside it."""
    bands: dict[int, list[float]] = {}
    for (band_index, _), value in sorted(values.items()):
        bands.setdefault(band_index, []).append(value)
    means = [float(np.mean(v)) for v in bands.values()]
    interval = common.t_interval(means, confidence)
    within = [float(np.var(v, ddof=1)) for v in bands.values() if len(v) > 1]
    return {
        "interval": interval,
        "n_bands": len(means),
        "n_runs": len(values),
        "band_means": means,
        "within_band_run_variance": float(np.mean(within)) if within else None,
        "between_band_variance": float(np.var(means, ddof=1)) if len(means) > 1 else None,
    }


def analyze_cell(results: dict, controls: dict, prefix: str, cell: str, config: str, protocol: dict) -> dict:
    confidence = 1.0 - protocol["design"]["alpha"]
    inactive = protocol["inactivity"]
    tier = protocol["cells"][cell]["tier"]
    needed = [f"{prefix}controls/{tier}/unseen"] + [
        f"{prefix}controls/{tier}/seen_{cell}_{run['band']}"
        for run in results.values()
        if run["cell"] == cell and run["config"] == config
    ]
    lost = sorted({key for key in needed if key not in controls})
    if lost:
        return {"cell": cell, "config": config, "unavailable": "controls_missing", "controls_missing": lost}
    reference_unseen = controls[f"{prefix}controls/{tier}/unseen"]["rows"]["equal_weight"]
    gaps: dict[str, dict] = {"final": {}, "best": {}}
    seen_excess, unseen_excess, flags, memoriser = {}, {}, [], {}
    floors = []
    for key, run in sorted(results.items()):
        if run["cell"] != cell or run["config"] != config:
            continue
        where = (run["band"], run["run"])
        seen_key = f"{prefix}controls/{tier}/seen_{cell}_{run['band']}"
        reference_seen = controls[seen_key]["rows"]["equal_weight"]
        final = run["final"]
        s = float(np.mean(excess_by_seed(final["seen"], reference_seen)))
        u = float(np.mean(excess_by_seed(final["unseen"], reference_unseen)))
        gaps["final"][where] = s - u
        seen_excess[where], unseen_excess[where] = s, u
        if "seen" in run["best"]:
            sb = float(np.mean(excess_by_seed(run["best"]["seen"], reference_seen)))
            ub = float(np.mean(excess_by_seed(run["best"]["unseen"], reference_unseen)))
            gaps["best"][where] = sb - ub
        else:
            gaps["best"][where] = s - u
        lines = final["unseen"]
        if (
            np.mean([r["gross"] for r in lines]) < inactive["min_mean_gross"]
            or np.mean([r["turnover"] for r in lines]) < inactive["min_mean_turnover"]
            or np.mean([r["action_std"] for r in lines]) < inactive["min_action_std"]
        ):
            flags.append(key)
        band_controls = controls[seen_key]
        memoriser[run["band"]] = float(
            np.mean(excess_by_seed(band_controls["rows"]["memoriser_if_seen"], reference_seen))
        )
        quantile = protocol["controls"]["floor_quantile"]
        floors.append(float(np.quantile(band_controls["floor_draws_bp"], quantile)) - float(np.mean([r["mean_bp"] for r in reference_seen])))
    memoriser_unseen_excess = -float(np.mean([r["mean_bp"] for r in reference_unseen]))
    memoriser_gaps = [v - memoriser_unseen_excess for v in memoriser.values()]
    return {
        "cell": cell,
        "tier": tier,
        "config": config,
        "gap_final": nested_summary(gaps["final"], confidence),
        "gap_best": nested_summary(gaps["best"], confidence),
        "seen_excess": nested_summary(seen_excess, confidence),
        "unseen_excess": nested_summary(unseen_excess, confidence),
        "memoriser_gap": common.t_interval(memoriser_gaps, confidence),
        "floor_excess_bp": float(np.mean(floors)) if floors else None,
        "inactive_runs": flags,
    }


def verdict(row: dict, protocol: dict) -> str:
    """The protocol's decision rule. Each earlier failure makes the gap uninformative."""
    if "unavailable" in row:
        return f"unavailable:{row['unavailable']}"
    sesoi = protocol["design"]["sesoi_bp"]
    memo = row["memoriser_gap"]
    if "lo" not in memo or memo["lo"] <= 0.0:
        return "unavailable:instrument"
    seen = row["seen_excess"]["interval"]
    if "lo" not in seen or seen["lo"] <= sesoi:
        return "unavailable:no_capacity"
    floor = row["floor_excess_bp"]
    if floor is None or seen["mean"] <= floor:
        return "unavailable:no_capacity"
    gap = row["gap_final"]["interval"]
    if "lo" not in gap:
        return "unavailable:gap"
    if gap["lo"] > 0.0:
        return "overfit_detected"
    if gap["lo"] >= -sesoi and gap["hi"] <= sesoi:
        return "within_margin"
    return "inconclusive"


def signal_verdict(row: dict, protocol: dict) -> str:
    """The declared-signal read: does the held-out excess show the tier's known edge?

    Reported beside the gap verdict, never instead of it. On a tier built with no edge the
    expected answer is ``no_heldout_edge``; on Calm it is ``heldout_edge``.
    """
    if "unavailable" in row:
        return f"unavailable:{row['unavailable']}"
    interval = row["unseen_excess"]["interval"]
    if "lo" not in interval:
        return "unavailable:unseen_excess"
    sesoi = protocol["design"]["sesoi_bp"]
    if interval["lo"] > 0.0:
        return "heldout_edge"
    if interval["lo"] >= -sesoi and interval["hi"] <= sesoi:
        return "no_heldout_edge"
    return "inconclusive"


def required_bands(sd: float, n_pilot: int, protocol: dict) -> dict:
    """The pilot rule: least K reaching the half-width at the inflated SD, up to the cap."""
    from scipy import stats

    design = protocol["design"]
    upper = sd * math.sqrt((n_pilot - 1) / stats.chi2.ppf(1.0 - design["pilot_sd_upper_confidence"], n_pilot - 1))
    half = design["target_half_width_bp"]
    for k in range(2, design["max_bands"] + 1):
        reached = stats.t.ppf(1.0 - design["alpha"] / 2.0, k - 1) * upper / math.sqrt(k)
        if reached <= half:
            return {"bands": k, "sd_upper": upper, "half_width": float(reached), "cap_binds": False}
    k = design["max_bands"]
    reached = stats.t.ppf(1.0 - design["alpha"] / 2.0, k - 1) * upper / math.sqrt(k)
    return {"bands": k, "sd_upper": upper, "half_width": float(reached), "cap_binds": True}


def run_stage(protocol: dict, sha: str, stage: str, out: Path, design: dict, workers: int, retry: dict[str, str], n_trial_runs: int, prefix: str = "", *, consumed: set[str] | None = None) -> tuple[dict, dict, list[Job], list[dict], dict]:
    controls, training = build_jobs(protocol, stage, design, n_trial_runs, prefix)
    keys_c = {j.key for j in controls}
    keys_t = {j.key for j in training}
    retry_c = {k: v for k, v in retry.items() if k in keys_c}
    retry_t = {k: v for k, v in retry.items() if k in keys_t}
    if consumed is not None:
        consumed.update(retry_c, retry_t)
    counts_c = common.run_jobs(controls, run_job, out, workers, retry_reasons=retry_c)
    counts_t = common.run_jobs(training, run_job, out, workers, retry_reasons=retry_t)
    control_results, absent_c = common.collect(out, [j.key for j in controls])
    results, absent_t = common.collect(out, [j.key for j in training])
    return control_results, results, training, absent_c + absent_t, {"controls": counts_c, "training": counts_t}


def require_reasons_used(reasons: dict[str, str], used: set[str]) -> None:
    """Refuse a retry reason that no run_jobs call of this stage received."""
    unused = sorted(set(reasons) - used)
    if unused:
        raise StudyRefused(f"retry reasons named failed jobs outside this stage's design: {unused}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("smoke", "pilot", "pilot-decide", "main"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--retry-failed",
        type=Path,
        default=None,
        metavar="REASONS.json",
        help="retry the failed keys named in this {key: infrastructure reason} file",
    )
    parser.add_argument("--smoke-steps", type=int, default=2048)
    parser.add_argument("--escalate", action="store_true", help="use the protocol's one budget escalation")
    args = parser.parse_args(argv)
    retry_reasons = common.load_retry_reasons(args.retry_failed)
    # Every reason must name a failed job of this stage and reach a run_jobs call.
    used: set[str] = set()

    protocol, sha = common.load_protocol(PROTOCOL_PATH)
    root = args.out if args.out is not None else HERE / "runs"
    decision_path = HERE / "pilot_decision.json"
    if args.stage != "smoke":
        common.require_clean_tree(args.stage)
        resolve_normalizer(args.stage)
    extra: dict[str, Any] = {}

    if args.stage == "smoke":
        out = root / "smoke"
        common.require_failed_keys(out, retry_reasons)
        design = stage_design(protocol, "smoke", None, args.smoke_steps)
        controls, results, training, absent, counts = run_stage(
            protocol, sha, "smoke", out, design, args.workers, retry_reasons,
            len(design["configs"]) * design["n_bands"] * design["runs_per_band"],
            consumed=used,
        )
        require_reasons_used(retry_reasons, used)
        summary = {"cells": {}}
        for cell in design["cells"]:
            summary["cells"][cell] = analyze_cell(results, controls, "", cell, "c0", protocol)
            summary["cells"][cell]["verdict"] = "unavailable:not_final_stage"
    elif args.stage == "pilot":
        out = root / "pilot"
        common.require_failed_keys(out, retry_reasons)
        steps = protocol["budget"]["escalated_env_steps" if args.escalate else "env_steps"]
        search = pilot_design(protocol, "search", None, steps)
        search_trials = len(protocol["configs"]) * search["n_bands"] * search["runs_per_band"]
        controls, results, _, absent, counts = run_stage(
            protocol, sha, "pilot", out, search, args.workers, retry_reasons, search_trials, "search_",
            consumed=used,
        )
        ranked = []
        for name in protocol["configs"]:
            row = analyze_cell(results, controls, "search_", "A", name, protocol)
            means = row.get("seen_excess", {}).get("band_means", [])
            mean = float(np.mean(means)) if means else None
            ranked.append({"config": name, "seen_excess_mean": mean, "inactive": len(row.get("inactive_runs", []))})
        eligible = [r for r in ranked if r["seen_excess_mean"] is not None and r["inactive"] == 0]
        if not eligible:
            raise StudyRefused("no eligible configuration: every one failed or was inactive")
        chosen = max(eligible, key=lambda r: (r["seen_excess_mean"], r["config"]))["config"]
        design = pilot_design(protocol, "design", chosen, steps)
        trials = search_trials + design["n_bands"] * design["runs_per_band"]
        controls2, results2, training, absent2, counts2 = run_stage(
            protocol, sha, "pilot", out, design, args.workers, retry_reasons, trials, "design_",
            consumed=used,
        )
        require_reasons_used(retry_reasons, used)
        row = analyze_cell(results2, controls2, "design_", "A", chosen, protocol)
        row["verdict"] = verdict(row, protocol)
        summary = {"search": ranked, "chosen": chosen, "env_steps": steps, "escalated": args.escalate, "trial_runs": trials, "cells": {"A": row}}
        absent = absent + absent2
        counts = {"search": counts, "design": counts2}
        extra["pilot_is_never_evidence"] = True
    elif args.stage == "pilot-decide":
        if retry_reasons:
            raise StudyRefused("pilot-decide runs no jobs, so it takes no retry reasons")
        pilot = common.read_json(root / "pilot" / "summary.json")
        row = pilot["cells"]["A"]["gap_final"]
        interval = row["interval"]
        if "sd" not in interval:
            raise StudyRefused("the pilot has no gap dispersion to size from")
        choice = required_bands(interval["sd"], interval["n"], protocol)
        capacity = pilot["cells"]["A"]["verdict"] != "unavailable:no_capacity"
        common.write_json(
            decision_path,
            {
                **choice,
                "config": pilot["chosen"],
                "env_steps": pilot["env_steps"],
                "pilot_trial_runs": pilot["trial_runs"],
                "cells": ["A", "D", "B", "C"] if capacity else ["A", "D"],
                "protocol_sha256": sha,
            },
        )
        print(f"decision written to {decision_path}; commit it before main")
        return 0
    else:
        out = root / "main"
        common.require_failed_keys(out, retry_reasons)
        decision = common.read_json(decision_path) if decision_path.exists() else None
        if decision is None:
            raise StudyRefused("main needs a committed pilot_decision.json")
        if decision["protocol_sha256"] != sha:
            raise StudyRefused("pilot_decision.json is for another protocol version")
        design = stage_design(protocol, "main", decision, 0)
        per_cell = design["n_bands"] * design["runs_per_band"]
        trials = decision["pilot_trial_runs"] + per_cell * len(design["cells"])
        controls, results, training, absent, counts = run_stage(
            protocol, sha, "main", out, design, args.workers, retry_reasons, trials,
            consumed=used,
        )
        require_reasons_used(retry_reasons, used)
        summary = {"cells": {}, "trial_runs": trials}
        for cell in design["cells"]:
            row = analyze_cell(results, controls, "", cell, design["configs"][0], protocol)
            row["verdict"] = verdict(row, protocol)
            row["signal_verdict"] = signal_verdict(row, protocol)
            row["expected_signal"] = protocol["cells"][cell]["expected_signal"]
            summary["cells"][cell] = row

    summary["absent_jobs"] = absent
    summary["jobs_expected"] = len(training)
    ledger = common.attempt_ledger(out)
    summary["attempt_ledger"] = ledger
    extra["attempt_ledger"] = ledger
    common.write_json(out / "summary.json", summary)
    common.write_json(
        out / "manifest.json",
        common.manifest(protocol["protocol_id"], protocol["version"], sha, args.stage, extra),
    )
    print(f"{args.stage}: {counts}; core-seconds {common.core_seconds(out):.1f}")
    return 1 if absent else 0


if __name__ == "__main__":
    sys.exit(main())
