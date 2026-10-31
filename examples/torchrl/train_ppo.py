"""Train PPO on SharpeArena through TorchRL, checkpoint, resume, then evaluate.

Runnable end to end on CPU in under a minute at the default budget, with no account,
no download and no model call. The budget is deliberately tiny: this script exists to
demonstrate that the route works, not to produce a result. Nothing it prints is a
benchmark number.

    pip install "sharpearena[torchrl]"
    python examples/torchrl/train_ppo.py

The environment crosses at ``float64``, the engine's width, so narrowing to the network's
``float32`` is a stated transform here rather than something a pipeline did quietly, and
the three observation keys are concatenated into one vector for the MLPs. Advantages come
from ``GAE``, which reads ``terminated`` and ``truncated`` separately: running out of bars
bootstraps the value and bankruptcy does not. Training and evaluation use different seed
bands, and evaluation runs with no exploration and no gradient.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import torch
from tensordict.nn import NormalParamExtractor, TensorDictModule
from torch import nn
from torchrl.collectors import Collector
from torchrl.envs import CatTensors, Compose, DoubleToFloat, ExplorationType, TransformedEnv
from torchrl.envs.utils import set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from sharpearena._seed_bands import EVAL_SEED_BASE
from sharpearena.torchrl_env import OBSERVATION_KEYS, SharpeArenaTorchRLEnv


def make_env(seed: int, n_symbols: int, n_days: int) -> TransformedEnv:
    return TransformedEnv(
        SharpeArenaTorchRLEnv(seed=seed, n_symbols=n_symbols, n_days=n_days),
        Compose(
            DoubleToFloat(
                in_keys=list(OBSERVATION_KEYS) + ["reward"],
            ),
            CatTensors(in_keys=list(OBSERVATION_KEYS), out_key="observation"),
        ),
    )


def make_modules(env: TransformedEnv, hidden: int = 32):
    width = env.observation_spec["observation"].shape[-1]
    action_spec = env.action_spec
    actions = action_spec.shape[-1]

    def mlp(output: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(width, hidden), nn.Tanh(), nn.Linear(hidden, output)
        )

    actor = ProbabilisticActor(
        TensorDictModule(
            nn.Sequential(mlp(2 * actions), NormalParamExtractor()),
            in_keys=["observation"],
            out_keys=["loc", "scale"],
        ),
        in_keys=["loc", "scale"],
        spec=action_spec,
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": action_spec.space.low,
            "high": action_spec.space.high,
        },
        return_log_prob=True,
    )
    critic = ValueOperator(mlp(1), in_keys=["observation"])
    return actor, critic


def make_estimator(critic: ValueOperator) -> GAE:
    return GAE(gamma=0.99, lmbda=0.95, value_network=critic)


def train(
    actor,
    critic,
    loss: ClipPPOLoss,
    optimizer: torch.optim.Optimizer,
    env: TransformedEnv,
    frames: int,
    frames_per_batch: int,
    epochs: int = 2,
) -> int:
    """Collect and update until ``frames`` are spent; returns the frames consumed."""
    estimator = make_estimator(critic)
    collector = Collector(
        env,
        actor,
        frames_per_batch=frames_per_batch,
        total_frames=frames,
        device="cpu",
    )
    consumed = 0
    try:
        for batch in collector:
            consumed += batch.numel()
            with torch.no_grad():
                estimator(batch)
            for _ in range(epochs):
                losses = loss(batch)
                total = (
                    losses["loss_objective"]
                    + losses["loss_critic"]
                    + losses["loss_entropy"]
                )
                optimizer.zero_grad()
                total.backward()
                optimizer.step()
    finally:
        collector.shutdown()
    return consumed


def evaluate(actor, env: TransformedEnv, max_steps: int) -> float:
    """One held-out episode's summed reward, with the distribution's deterministic action."""
    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        rollout = env.rollout(max_steps, actor)
    return float(rollout["next", "reward"].sum())


def run(args: argparse.Namespace, directory: Path) -> dict:
    torch.manual_seed(args.seed)
    env = make_env(0, args.n_symbols, args.n_days)
    env.set_seed(0)
    actor, critic = make_modules(env)
    loss = ClipPPOLoss(actor, critic, entropy_coeff=0.01, normalize_advantage=False)
    optimizer = torch.optim.Adam(loss.parameters(), lr=3e-4)
    consumed = train(
        actor, critic, loss, optimizer, env, args.frames, args.frames_per_batch
    )

    path = directory / "ppo.pt"
    torch.save(
        {
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "optimizer": optimizer.state_dict(),
            "frames": consumed,
        },
        path,
    )

    # Resume from the file alone, as a separate process would.
    resumed_env = make_env(0, args.n_symbols, args.n_days)
    resumed_actor, resumed_critic = make_modules(resumed_env)
    resumed_loss = ClipPPOLoss(
        resumed_actor, resumed_critic, entropy_coeff=0.01, normalize_advantage=False
    )
    resumed_optimizer = torch.optim.Adam(resumed_loss.parameters(), lr=3e-4)
    checkpoint = torch.load(path, weights_only=True)
    resumed_actor.load_state_dict(checkpoint["actor"])
    resumed_critic.load_state_dict(checkpoint["critic"])
    resumed_optimizer.load_state_dict(checkpoint["optimizer"])
    consumed = checkpoint["frames"] + train(
        resumed_actor,
        resumed_critic,
        resumed_loss,
        resumed_optimizer,
        resumed_env,
        args.frames_per_batch,
        args.frames_per_batch,
    )

    held_out = make_env(EVAL_SEED_BASE, args.n_symbols, args.n_days)
    reward = evaluate(resumed_actor, held_out, args.n_days)
    print(f"train seed:       0, evaluation seed: {EVAL_SEED_BASE}")
    print(f"frames:           {consumed}")
    print(f"episode reward:   {reward:.6f}")
    print("Not a benchmark result: the budget is a demonstration, not an experiment.")
    return {"frames": consumed, "reward": reward}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=2048)
    parser.add_argument("--frames-per-batch", type=int, default=256)
    parser.add_argument("--n-symbols", type=int, default=4)
    parser.add_argument("--n-days", type=int, default=120)
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
