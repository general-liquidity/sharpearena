"""Run the CleanRL-style PPO example against the installed wheel at a tiny CPU budget.

The Python job has no torch, so its pytest cases skip there. This runs in the SB3
wheel cell, which installs CPU torch, and exits nonzero if the example breaks.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import math
from pathlib import Path
import sys

import torch

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/cleanrl-style/ppo_vector.py"

spec = importlib.util.spec_from_file_location("ppo_vector", EXAMPLE)
ppo = importlib.util.module_from_spec(spec)
sys.modules["ppo_vector"] = ppo
spec.loader.exec_module(ppo)

torch.set_num_threads(1)
args = dataclasses.replace(
    ppo.Args(), n_days=40, total_timesteps=256, num_steps=32, update_epochs=1
)
first, episodes = ppo.train(args)
second, _ = ppo.train(args)
held_out = ppo.evaluate(first, args)

assert episodes, "the tiny budget must finish at least one training episode"
assert ppo.parameter_digest(first) == ppo.parameter_digest(second), "training is not seeded"
assert len(held_out) == args.num_envs and all(map(math.isfinite, held_out))
print(f"cleanrl-style example ran: {len(episodes)} episodes, held-out {held_out}")
