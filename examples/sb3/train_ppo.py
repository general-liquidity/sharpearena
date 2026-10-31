"""Train PPO on SharpeArena through Stable-Baselines3, checkpoint, resume, then evaluate.

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

A checkpoint here is two files, because ``VecNormalize`` statistics are not part of the
model archive: without them a restored policy reads differently scaled observations.
Evaluation restores those statistics with ``training=False``, so held-out tapes cannot
move them, and ``norm_reward=False``, so the printed reward is the environment's own.

See ``crates/sharpearena-py/python/sharpearena/sb3_env.py`` for the autoreset and
terminal-observation mapping this route rests on, and
``docs/integrations/support-status.md`` for how far it has been checked.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize

from sharpearena._seed_bands import EVAL_SEED_BASE
from sharpearena.sb3_env import SharpeArenaSB3VecEnv


def run(args: argparse.Namespace, directory: Path) -> None:
    def lanes(first_seed: int) -> SharpeArenaSB3VecEnv:
        return SharpeArenaSB3VecEnv(
            seeds=[first_seed + i for i in range(args.num_envs)],
            n_symbols=args.n_symbols,
            n_days=args.n_days,
        )

    model_path = directory / "ppo.zip"
    stats_path = directory / "vecnormalize.pkl"

    train = VecNormalize(lanes(0))
    model = PPO(
        "MultiInputPolicy", train, n_steps=64, seed=args.seed, device="cpu", verbose=0
    )
    model.learn(total_timesteps=args.timesteps)
    model.save(model_path)
    train.save(stats_path)

    # Resume from the two files alone, as a separate process would.
    resumed_train = VecNormalize.load(str(stats_path), lanes(0))
    resumed = PPO.load(model_path, env=resumed_train, device="cpu")
    resumed.learn(total_timesteps=args.resume_timesteps, reset_num_timesteps=False)
    resumed.save(model_path)
    resumed_train.save(stats_path)

    evaluation = VecNormalize.load(str(stats_path), lanes(EVAL_SEED_BASE))
    evaluation.training = False
    evaluation.norm_reward = False
    final = PPO.load(model_path, device="cpu")
    mean_reward, std_reward = evaluate_policy(
        final, evaluation, n_eval_episodes=args.eval_episodes, warn=False
    )
    print(f"train lanes:      {resumed_train.venv.scenario_seeds}")
    print(f"evaluation lanes: {evaluation.venv.scenario_seeds}")
    print(f"timesteps:        {final.num_timesteps}")
    print(f"episode reward:   {mean_reward:.6f} +/- {std_reward:.6f}")
    print("Not a benchmark result: the budget is a demonstration, not an experiment.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=2048)
    parser.add_argument("--resume-timesteps", type=int, default=256)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--n-symbols", type=int, default=4)
    parser.add_argument("--n-days", type=int, default=120)
    parser.add_argument("--eval-episodes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out-dir", type=Path, help="keep the checkpoint here; default is a temporary dir"
    )
    args = parser.parse_args()

    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        run(args, args.out_dir)
        return
    with tempfile.TemporaryDirectory() as directory:
        run(args, Path(directory))


if __name__ == "__main__":
    main()
