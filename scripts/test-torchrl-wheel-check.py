"""Check installed TorchRL parity and reject corrupted rewards and end flags."""

from unittest.mock import patch

from optional_extras import _exercise_torchrl
from sharpearena.torchrl_env import SharpeArenaTorchRLEnv


original = SharpeArenaTorchRLEnv._step


def corrupted_reward(self, inputs):
    result = original(self, inputs)
    result.set("reward", result["reward"] + 0.5)
    return result


def corrupted_flags(self, inputs):
    result = original(self, inputs)
    result.set("truncated", ~result["truncated"])
    return result


print(_exercise_torchrl())
for mutation in (corrupted_reward, corrupted_flags):
    with patch.object(SharpeArenaTorchRLEnv, "_step", mutation):
        try:
            _exercise_torchrl()
        except AssertionError:
            print(f"expected failure: {mutation.__name__} rejected")
        else:
            raise SystemExit(
                f"installed TorchRL mutation survived: {mutation.__name__}"
            )
