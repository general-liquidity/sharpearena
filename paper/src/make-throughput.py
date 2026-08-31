"""Measure environment throughput on all three surfaces and write the evidence JSON.

Times the Python surface directly (scenario generation, Gymnasium-adapter stepping, and
one full protocol-conformant evaluation over the canonical held-out band), then shells
out to the native and WebAssembly benchmarks over the same 4-symbol, 120-day panel so
the three rows are produced by one command. Writes ``paper/evidence/throughput.json``.

Native:  cargo run --release -p sharpearena --example bench-steps
WASM:    node npm/sharpearena/bench/throughput.js
"""

from __future__ import annotations

import json
import platform
import re
import subprocess
import time
from pathlib import Path

import numpy as np

import sharpearena
from sharpearena.effective_config import (
    check_env_effective_config,
    merge_effective_configs,
)
from sharpearena.gym import SharpeArenaEnv
from sharpearena.sharpearena_py import generate_scenario_json

N_SYMBOLS = 4
N_DAYS = 120
EPISODES = 500
REPEATS = 3
# The canonical held-out namespace of the evaluation protocol.
EVAL_BAND = range(10_256, 10_512)
EVAL_TIERS = ("calm", "hard", "extreme")

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
EVIDENCE = HERE.parent / "evidence" / "throughput.json"


def effective_config_preflight() -> dict:
    """Verify the environment-backed timed arms without charging checks to them.

    The step microbenchmark uses seeds ``0..EPISODES`` at the default Calm tier;
    the canonical evaluation uses the held-out band on all three tiers. Boundary
    seeds from each arm are read back separately so the artifact does not imply
    that the held-out preflight also verified the microbenchmark's constructor.
    The raw generation benchmark calls ``generate_scenario_json`` directly, which
    is itself the independent regeneration path used by the check.
    """
    micro_readback: dict[int, dict] = {}
    for seed in (0, EPISODES - 1):
        env = SharpeArenaEnv(n_symbols=N_SYMBOLS, n_days=N_DAYS, seed=seed)
        micro_readback[seed] = check_env_effective_config(
            env,
            seed=seed,
            n_symbols=N_SYMBOLS,
            n_days=N_DAYS,
            distribution_mode="calm",
        )

    canonical_readback: dict[str, dict[int, dict]] = {
        tier: {} for tier in EVAL_TIERS
    }
    for tier in EVAL_TIERS:
        for seed in (EVAL_BAND.start, EVAL_BAND.stop - 1):
            env = SharpeArenaEnv(
                n_symbols=N_SYMBOLS,
                n_days=N_DAYS,
                seed=seed,
                distribution_mode=tier,
            )
            canonical_readback[tier][seed] = check_env_effective_config(
                env,
                seed=seed,
                n_symbols=N_SYMBOLS,
                n_days=N_DAYS,
                distribution_mode=tier,
            )
    return {
        "step_microbenchmark": merge_effective_configs(micro_readback),
        "canonical_evaluation": {
            tier: merge_effective_configs(canonical_readback[tier])
            for tier in EVAL_TIERS
        },
    }


def bench_generation(episodes: int) -> tuple[float, float]:
    start = time.perf_counter()
    total = 0
    for seed in range(episodes):
        total += len(generate_scenario_json(seed, n_symbols=N_SYMBOLS, n_days=N_DAYS))
    elapsed = time.perf_counter() - start
    assert total > 0
    return episodes / elapsed, elapsed


def bench_steps(episodes: int) -> tuple[float, float, int]:
    action = np.full(N_SYMBOLS, 0.25, dtype=np.float32)
    steps = 0
    start = time.perf_counter()
    for seed in range(episodes):
        env = SharpeArenaEnv(n_symbols=N_SYMBOLS, n_days=N_DAYS, seed=seed)
        env.reset(seed=seed)
        while True:
            _, _, terminated, truncated, _ = env.step(action)
            steps += 1
            if terminated or truncated:
                break
    elapsed = time.perf_counter() - start
    return steps / elapsed, elapsed, steps


def bench_canonical_evaluation() -> tuple[float, int]:
    """Time the protocol's recommended evaluation: the full reference field over the
    canonical held-out band on all three tiers, scored by the SharpeBench kernel."""
    seeds = list(EVAL_BAND)
    start = time.perf_counter()
    n_rows = 0
    for tier in EVAL_TIERS:
        rows = sharpearena.run_baselines(
            n_symbols=N_SYMBOLS,
            n_days=N_DAYS,
            seeds=seeds,
            distribution_mode=tier,
            confidence=False,
        )
        n_rows += len(rows)
    return time.perf_counter() - start, n_rows


def run_native() -> dict:
    """Run the native benchmark and parse its printed rates, one repeat per invocation."""
    gen, steps, ep_ms = [], [], []
    for i in range(REPEATS + 1):
        proc = subprocess.run(
            [
                "cargo",
                "run",
                "--release",
                "-q",
                "-p",
                "sharpearena",
                "--example",
                "bench-steps",
            ],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        )
        if i == 0:
            continue  # the first invocation also pays the build check
        out = proc.stdout
        gen.append(float(re.search(r"generation: ([\d.]+) scenarios/s", out).group(1)))
        steps.append(float(re.search(r"scalar step: ([\d.]+) steps/s", out).group(1)))
        ep_ms.append(float(re.search(r"episode wall time: ([\d.]+) ms", out).group(1)))
    return {
        "generation_scenarios_per_s": {"median": float(np.median(gen)), "runs": gen},
        "steps_per_s": {"median": float(np.median(steps)), "runs": steps},
        "episode_wall_ms": {"median": float(np.median(ep_ms)), "runs": ep_ms},
    }


def run_wasm() -> dict:
    proc = subprocess.run(
        ["node", "bench/throughput.js"],
        cwd=REPO / "npm" / "sharpearena",
        capture_output=True,
        text=True,
        check=True,
        shell=True,
    )
    return json.loads(proc.stdout)


def main() -> None:
    effective_config = effective_config_preflight()
    # Warm the binding and the allocator before timing.
    bench_steps(4)
    bench_generation(4)

    gen_rates, step_rates, episode_ms = [], [], []
    steps_total = 0
    for _ in range(REPEATS):
        gen_rate, _ = bench_generation(EPISODES)
        step_rate, step_elapsed, steps_total = bench_steps(EPISODES)
        gen_rates.append(gen_rate)
        step_rates.append(step_rate)
        episode_ms.append(1e3 * step_elapsed / EPISODES)

    eval_runs = [bench_canonical_evaluation() for _ in range(REPEATS)]
    eval_seconds = [s for s, _ in eval_runs]
    eval_rows = eval_runs[0][1]

    payload = {
        "panel": {"n_symbols": N_SYMBOLS, "n_days": N_DAYS},
        "effective_config": effective_config,
        "episodes_per_repeat": EPISODES,
        "repeats": REPEATS,
        "steps_per_repeat": steps_total,
        "python_generation_scenarios_per_s": {
            "median": float(np.median(gen_rates)),
            "runs": [float(x) for x in gen_rates],
        },
        "python_steps_per_s": {
            "median": float(np.median(step_rates)),
            "runs": [float(x) for x in step_rates],
        },
        "python_episode_wall_ms": {
            "median": float(np.median(episode_ms)),
            "runs": [float(x) for x in episode_ms],
        },
        "canonical_evaluation": {
            "band": [min(EVAL_BAND), max(EVAL_BAND) + 1],
            "seeds": len(EVAL_BAND),
            "tiers": list(EVAL_TIERS),
            "policy_rows": eval_rows,
            "episodes": len(EVAL_BAND) * len(EVAL_TIERS) * (eval_rows // len(EVAL_TIERS)),
            "seconds": {
                "median": float(np.median(eval_seconds)),
                "runs": [float(x) for x in eval_seconds],
            },
        },
        "native": run_native(),
        "wasm": run_wasm(),
        "host": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
    }
    EVIDENCE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
