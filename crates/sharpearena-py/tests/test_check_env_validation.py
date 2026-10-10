"""Conformance checks remain effective under ordinary and optimized Python."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize("fault", ["reward", "observation", "info", "reset", "flags"])
def test_checker_rejects_invalid_environment_with_or_without_optimization(
    optimized, fault
):
    program = r"""
import sys
import numpy as np
from gymnasium import spaces
from sharpearena.check_env import check_env, check_determinism_across_constructors

fault = sys.argv[1]
class Env:
    observation_space = spaces.Dict({"x": spaces.Box(0, 1, shape=(1,), dtype=np.float64)})
    action_space = spaces.Box(0, 1, shape=(1,), dtype=np.float32)
    def __init__(self, flag=False):
        self.flag = flag
    def reset(self):
        if fault == "reset":
            return ({"x": np.array([0.5])}, {}, "extra")
        return {"x": np.array([0.5])}, (None if fault == "info" else {})
    def step(self, action):
        return ({"x": np.array([2.0 if fault == "observation" else 0.5])},
                float("nan") if fault == "reward" else 0.0,
                self.flag, not self.flag, {})

try:
    if fault == "flags":
        made = []
        def factory():
            made.append(1)
            return Env(flag=len(made) == 2)
        check_determinism_across_constructors(factory)
    else:
        check_env(Env())
except AssertionError:
    print("VALIDATION_REJECTED")
else:
    raise RuntimeError("checker reported success for an invalid environment")
"""
    command = [sys.executable]
    if optimized:
        command.append("-O")
    result = subprocess.run(
        command + ["-c", program, fault], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "VALIDATION_REJECTED" in result.stdout


def test_checker_accepts_an_installed_native_environment():
    from sharpearena.check_env import check_env
    from sharpearena.gym import SharpeArenaEnv

    check_env(SharpeArenaEnv(n_symbols=2, n_days=12))
