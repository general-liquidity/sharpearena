#!/usr/bin/env python
"""INT-13 Python-boundary harness: the cost of reaching the native engine from Python.

Reads the same workload manifest as the native harness
(``docs/integration/int13/workload-manifest.json``) so the two halves measure one
workload, and decomposes the boundary into three separately reported parts:

1. the engine transition itself (measured natively by ``bench-int13``),
2. the binding round trip: pyo3 call plus the Rust-side ``serde_json`` parse of the
   decision and serialize of the observation,
3. the Python-side ``json.dumps`` / ``json.loads`` and numpy decode.

Each cell is repeated; the report carries median, mean, sample standard deviation and a
95 percent normal-approximation interval, not a single peak steps-per-second number.

Usage::

    python scripts/bench/int13_python_boundary.py --out int13-python.json

Requires the built wheel installed (``maturin build --release`` then ``pip install``);
a debug build would measure the wrong thing.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

import sharpearena
from sharpearena import SharpeArenaEnv, TradingEnv, VecTradingEnv

MANIFEST_DEFAULT = Path("docs/integration/int13/workload-manifest.json")


def summarize(samples: list[float]) -> dict[str, float | int]:
    """Median, mean, sd and a 95 percent normal-approximation interval on the mean.

    ``n`` travels with the summary so a reader can re-derive the interval instead of
    trusting it.
    """
    if not samples:
        raise ValueError("no samples to summarize")
    n = len(samples)
    mean = statistics.fmean(samples)
    sd = statistics.stdev(samples) if n > 1 else 0.0
    se = sd / (n**0.5) if n > 1 else 0.0
    return {
        "n": n,
        "median": statistics.median(samples),
        "mean": mean,
        "sd": sd,
        "rel_sd": (sd / mean) if mean else 0.0,
        "ci95_lo": mean - 1.96 * se,
        "ci95_hi": mean + 1.96 * se,
        "min": min(samples),
        "max": max(samples),
    }


def repeat(fn: Callable[[], float], reps: int, warmup: int) -> list[float]:
    """Run ``fn`` ``warmup`` times discarding the results, then ``reps`` times keeping them."""
    for _ in range(warmup):
        fn()
    return [fn() for _ in range(reps)]


def decision_dict(symbols: list[str], policy: dict[str, Any]) -> dict[str, Any]:
    return {
        "orders": [
            {
                "symbol": s,
                "action": policy["action"],
                "target_weight": policy["target_weight"],
                "confidence": policy["confidence"],
                "rationale": policy["rationale"],
            }
            for s in symbols
        ],
        "reasoning": "",
    }


def build_env(panel: dict[str, Any], seed: int) -> TradingEnv:
    return TradingEnv(
        n_symbols=panel["n_symbols"],
        n_days=panel["n_days"],
        seed=seed,
        distribution_mode=panel["distribution_mode"],
        exec_seed=seed,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=str(MANIFEST_DEFAULT))
    ap.add_argument("--out", default="int13-python.json")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    panel = manifest["panel"]
    policy = manifest["action_policy"]
    reps = manifest["measurement"]["repetitions"]
    warmup = 2
    lane_counts = manifest["scaling"]["lane_counts"]

    cells: list[dict[str, Any]] = []
    sizes: dict[str, Any] = {}

    def add(name: str, unit: str, samples: list[float], **extra: Any) -> None:
        cells.append(
            {"name": name, "unit": unit, "summary": summarize(samples), **extra}
        )

    # --- payload sizes: what the JSON boundary actually moves per step --------------
    probe = build_env(panel, 0)
    obs_json = probe.reset()
    obs = json.loads(obs_json)
    symbols = [s["symbol"] for s in obs["symbols"]]
    dec = decision_dict(symbols, policy)
    dec_json = json.dumps(dec)
    step_obs_json, _reward, _done, step_info_json = probe.step(dec_json)
    sizes["decision_json_bytes"] = len(dec_json.encode())
    sizes["observation_json_bytes"] = len(step_obs_json.encode())
    sizes["info_json_bytes"] = len(step_info_json.encode())
    # The RL-facing numeric payload the Gymnasium space actually exposes: closes(n),
    # positions(n) and cash(1) as float64.
    sizes["gym_observation_float64_values"] = 2 * len(symbols) + 1
    sizes["gym_observation_float64_bytes"] = 8 * (2 * len(symbols) + 1)
    sizes["json_to_numeric_ratio"] = (
        sizes["observation_json_bytes"] / sizes["gym_observation_float64_bytes"]
    )
    state_json = probe.clone_state()
    sizes["clone_state_json_bytes_at_bar_1"] = len(state_json.encode())

    # --- cell 1: raw binding round trip, no Python-side JSON ------------------------
    def raw_binding() -> float:
        env = build_env(panel, 11)
        env.reset()
        bars = 20_000
        t0 = time.perf_counter()
        steps = 0
        for _ in range(bars):
            _o, _r, done, _i = env.step(dec_json)
            steps += 1
            if done:
                env.reset()
        return steps / (time.perf_counter() - t0)

    add(
        "py_scalar_binding_no_python_json",
        "steps/s",
        repeat(raw_binding, reps, warmup),
        lanes=1,
        note="pyo3 call plus Rust-side serde parse/serialize; decision string prebuilt, returned strings discarded",
    )

    # --- cell 2: binding round trip plus Python-side JSON ---------------------------
    def binding_with_pyjson() -> float:
        env = build_env(panel, 11)
        env.reset()
        bars = 20_000
        t0 = time.perf_counter()
        steps = 0
        for _ in range(bars):
            payload = json.dumps(dec)
            o, _r, done, i = env.step(payload)
            _obs = json.loads(o)
            _info = json.loads(i)
            steps += 1
            if done:
                env.reset()
        return steps / (time.perf_counter() - t0)

    add(
        "py_scalar_binding_with_python_json",
        "steps/s",
        repeat(binding_with_pyjson, reps, warmup),
        lanes=1,
        note="adds json.dumps of the decision and json.loads of the observation and info",
    )

    # --- cell 3: Python-side JSON alone, no engine ----------------------------------
    def pyjson_only() -> float:
        iters = 20_000
        t0 = time.perf_counter()
        for _ in range(iters):
            payload = json.dumps(dec)
            _obs = json.loads(step_obs_json)
            _info = json.loads(step_info_json)
            if not payload:
                raise AssertionError("empty payload")
        return iters / (time.perf_counter() - t0)

    add(
        "py_json_encode_decode_only",
        "ops/s",
        repeat(pyjson_only, reps, warmup),
        lanes=1,
        note="no engine call; same payloads as cell 2",
    )

    # --- cell 4: the Gymnasium path users actually train against --------------------
    def gym_step() -> float:
        env = SharpeArenaEnv(
            n_symbols=panel["n_symbols"], n_days=panel["n_days"], seed=7
        )
        env.reset(seed=7)
        action = np.full(
            (panel["n_symbols"],), policy["target_weight"], dtype=np.float32
        )
        bars = 20_000
        t0 = time.perf_counter()
        steps = 0
        for _ in range(bars):
            _o, _r, term, trunc, _i = env.step(action)
            steps += 1
            if term or trunc:
                env.reset()
        return steps / (time.perf_counter() - t0)

    add(
        "py_gymnasium_env_step",
        "steps/s",
        repeat(gym_step, reps, warmup),
        lanes=1,
        note="SharpeArenaEnv.step: action encode, JSON boundary, observation decode into numpy",
    )

    # --- cell 5: reset and checkpoint overhead at the boundary ----------------------
    def py_reset() -> float:
        env = build_env(panel, 13)
        env.reset()
        iters = 3_000
        t0 = time.perf_counter()
        for _ in range(iters):
            env.reset()
        return 1e6 * (time.perf_counter() - t0) / iters

    add("py_reset_existing_env", "us/reset", repeat(py_reset, reps, warmup), lanes=1)

    def py_construct() -> float:
        iters = 200
        t0 = time.perf_counter()
        for k in range(iters):
            env = build_env(panel, 5_000 + k)
            env.reset()
        return 1e6 * (time.perf_counter() - t0) / iters

    add(
        "py_construct_and_first_reset",
        "us/env",
        repeat(py_construct, reps, warmup),
        lanes=1,
    )

    def py_checkpoint() -> float:
        env = build_env(panel, 17)
        env.reset()
        for _ in range(panel["n_days"] // 2):
            _o, _r, done, _i = env.step(dec_json)
            if done:
                break
        iters = 1_000
        t0 = time.perf_counter()
        for _ in range(iters):
            env.restore_state(env.clone_state())
        return 1e6 * (time.perf_counter() - t0) / iters

    add(
        "py_clone_restore_state_roundtrip",
        "us/roundtrip",
        repeat(py_checkpoint, reps, warmup),
        lanes=1,
        note="JSON snapshot out and back in across the boundary at mid-episode depth",
    )

    mid_env = build_env(panel, 17)
    mid_env.reset()
    for _ in range(panel["n_days"] // 2):
        _o, _r, done, _i = mid_env.step(dec_json)
        if done:
            break
    sizes["clone_state_json_bytes_mid_episode"] = len(mid_env.clone_state().encode())

    # --- cell 6: vector boundary scaling curve --------------------------------------
    for lanes in lane_counts:
        seeds = list(range(lanes))
        decisions_json = json.dumps([dec] * lanes)
        batch_probe = VecTradingEnv(seeds=seeds, **_panel_kwargs(panel))
        batch_probe.reset_batch()
        result_json = batch_probe.step_batch(decisions_json)
        sizes.setdefault("batch_result_json_bytes", {})[str(lanes)] = len(
            result_json.encode()
        )

        def vec_no_pyjson(lanes: int = lanes, seeds: list[int] = seeds) -> float:
            batch = VecTradingEnv(seeds=seeds, **_panel_kwargs(panel))
            batch.reset_batch()
            calls = max(50, 200_000 // lanes)
            payload = json.dumps([dec] * lanes)
            t0 = time.perf_counter()
            for _ in range(calls):
                batch.step_batch(payload)
            return (calls * lanes) / (time.perf_counter() - t0)

        def vec_with_pyjson(lanes: int = lanes, seeds: list[int] = seeds) -> float:
            batch = VecTradingEnv(seeds=seeds, **_panel_kwargs(panel))
            batch.reset_batch()
            calls = max(50, 200_000 // lanes)
            t0 = time.perf_counter()
            for _ in range(calls):
                payload = json.dumps([dec] * lanes)
                out = batch.step_batch(payload)
                _decoded = json.loads(out)
            return (calls * lanes) / (time.perf_counter() - t0)

        add(
            "py_vector_step_batch_no_python_json",
            "steps/s",
            repeat(vec_no_pyjson, reps, warmup),
            lanes=lanes,
        )
        add(
            "py_vector_step_batch_with_python_json",
            "steps/s",
            repeat(vec_with_pyjson, reps, warmup),
            lanes=lanes,
        )

    record = {
        "harness": "int13_python_boundary",
        "manifest_id": manifest["manifest_id"],
        "repetitions": reps,
        "warmup_repetitions": warmup,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "numpy": np.__version__,
            "sharpearena": getattr(sharpearena, "__version__", "unknown"),
        },
        "payload_sizes": sizes,
        "cells": cells,
    }
    Path(args.out).write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))
    return 0


def _panel_kwargs(panel: dict[str, Any]) -> dict[str, Any]:
    return {
        "n_symbols": panel["n_symbols"],
        "n_days": panel["n_days"],
        "distribution_mode": panel["distribution_mode"],
        "autoreset_mode": "next_step",
    }


if __name__ == "__main__":
    raise SystemExit(main())
