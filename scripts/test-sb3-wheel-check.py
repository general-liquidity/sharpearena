"""Run the installed SB3 check and prove it rejects corrupted adapter rewards."""

from unittest.mock import patch

from optional_extras import _exercise_sb3
from sharpearena.sb3_env import SharpeArenaSB3VecEnv


original = SharpeArenaSB3VecEnv.step


def corrupted_step(self, actions):
    observations, rewards, done, infos = original(self, actions)
    return observations, rewards + 0.5, done, infos


print(_exercise_sb3())
with patch.object(SharpeArenaSB3VecEnv, "step", corrupted_step):
    try:
        _exercise_sb3()
    except AssertionError:
        print("expected failure: altered installed adapter reward rejected")
    else:
        raise SystemExit("the installed adapter reward mutation survived")
