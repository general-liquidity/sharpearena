"""Train PPO on SharpeArena through the Stable-Baselines3 VecEnv route, then evaluate.

Runnable end to end on CPU in under a minute at the default budget, with no account,
no download and no model call. The budget is deliberately tiny: this script exists to
demonstrate that the route works, not to produce a result. Nothing it prints is a
benchmark number, and the reported reward at this many timesteps carries no information
about whether a policy learned anything.

    pip install "sharpearena[sb3]"
    python examples/sb3/train_ppo.py

The observation is a single-level ``Dict``, so the policy is ``MultiInputPolicy``. The
training lanes and the evaluation lanes are drawn from different seed bands: the eval
band starts at ``sharpearena.eval_seeds.EVAL_SEED_BASE``, which is what keeps the
evaluation tapes held out from the ones the agent trained on.

See ``crates/sharpearena-py/python/sharpearena/sb3_env.py`` for the autoreset and
terminal-observation mapping this route rests on, and ``docs/integrations/INT-11-stable-baselines3.md``
for how far it has been checked.
"""

from __future__ import annotations

import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from sharpearena._seed_bands import EVAL_SEED_BASE
from sharpearena.sb3_env import SharpeArenaSB3VecEnv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=2048)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--n-symbols", type=int, default=4)
    parser.add_argument("--n-days", type=int, default=120)
    parser.add_argument("--eval-episodes", type=int, default=4)
    args = parser.parse_args()

    train = SharpeArenaSB3VecEnv(
        seeds=list(range(args.num_envs)),
        n_symbols=args.n_symbols,
        n_days=args.n_days,
    )
    evaluation = SharpeArenaSB3VecEnv(
        seeds=[EVAL_SEED_BASE + i for i in range(args.num_envs)],
        n_symbols=args.n_symbols,
        n_days=args.n_days,
    )

    model = PPO("MultiInputPolicy", train, n_steps=64, device="cpu", verbose=0)
    model.learn(total_timesteps=args.timesteps)

    mean_reward, std_reward = evaluate_policy(
        model, evaluation, n_eval_episodes=args.eval_episodes, warn=False
    )
    print(f"train lanes:      {train.scenario_seeds}")
    print(f"evaluation lanes: {evaluation.scenario_seeds}")
    print(f"timesteps:        {args.timesteps}")
    print(f"episode reward:   {mean_reward:.6f} +/- {std_reward:.6f}")
    print("Not a benchmark result: the budget is a demonstration, not an experiment.")


if __name__ == "__main__":
    main()
