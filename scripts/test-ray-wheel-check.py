"""Exercise installed Ray/RLlib execution without building or training a model."""

from optional_extras import _exercise_ray
from unittest.mock import patch

import sharpearena.ray_executor as executor
from sharpearena.spaces import FlattenObservation


print(_exercise_ray())

original_run = executor.run_episodes
original_step = FlattenObservation.step


def local_only(specs, **kwargs):
    return original_run(specs, num_workers=0)


def corrupted_reward(self, action):
    observation, reward, terminated, truncated, info = original_step(self, action)
    return observation, reward + 0.5, terminated, truncated, info


for target, name, mutation in (
    (executor, "run_episodes", local_only),
    (FlattenObservation, "step", corrupted_reward),
):
    with patch.object(target, name, mutation):
        try:
            _exercise_ray()
        except AssertionError:
            print(f"expected failure: {mutation.__name__} rejected")
        else:
            raise SystemExit(f"installed Ray mutation survived: {mutation.__name__}")
