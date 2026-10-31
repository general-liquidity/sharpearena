"""Single-file PPO on SharpeArena's Gymnasium 1.x vector env, in the CleanRL style.

This is not CleanRL and contains no CleanRL code. It borrows one idea, the
everything-in-one-readable-file layout of CleanRL (Huang et al., JMLR 2022,
https://github.com/vwxyzjn/cleanrl, MIT licensed), and was written fresh for this
repository. CleanRL itself pins ``gymnasium==0.29.1``, which cannot share an
environment with this package's ``gymnasium>=1.0``; see
``docs/integrations/support-status.md``.

    pip install sharpearena torch
    python examples/cleanrl-style/ppo_vector.py

The default budget exists to show the loop runs on a CPU. Nothing it prints is a
benchmark number, and the reported return says nothing about whether a policy
learned anything.

The vector env runs in ``same_step`` autoreset mode so every stored transition is a
real one. The ``next_step`` default would need a mask for the zero-reward reset step
it inserts after each episode. The two ways an episode ends are kept apart: a
termination is a real end with no future value, while a truncation is a horizon limit,
so its target bootstraps from the value of the terminal observation in ``final_obs``.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from sharpearena._seed_bands import EVAL_SEED_BASE
from sharpearena.vector import SharpeArenaVectorEnv


@dataclasses.dataclass
class Args:
    seed: int = 1
    num_envs: int = 4
    n_symbols: int = 4
    n_days: int = 120
    total_timesteps: int = 2048
    num_steps: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    update_epochs: int = 4
    num_minibatches: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    # Per-symbol weight cap, so a sample at the action bound stays a modest portfolio.
    max_weight: float = 0.25


def make_env(args: Args, base_seed: int) -> SharpeArenaVectorEnv:
    return SharpeArenaVectorEnv(
        seeds=[base_seed + lane for lane in range(args.num_envs)],
        n_symbols=args.n_symbols,
        n_days=args.n_days,
        max_weight=args.max_weight,
        autoreset_mode="same_step",
    )


def features(obs: dict[str, np.ndarray]) -> torch.Tensor:
    """Turn prices and holdings into scale-free inputs: relative log price and weights."""
    closes, positions, cash = obs["closes"], obs["positions"], obs["cash"]
    nav = cash + (closes * positions).sum(axis=-1, keepdims=True)
    log_closes = np.log(closes)
    relative = log_closes - log_closes.mean(axis=-1, keepdims=True)
    weights = closes * positions / nav
    flat = np.concatenate([relative, weights, cash / nav], axis=-1)
    return torch.as_tensor(flat, dtype=torch.float32)


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> torch.Tensor:
    """Generalized advantage estimation over a ``(steps, lanes)`` rollout.

    ``next_values[t]`` is the value of the observation that truly followed step ``t``,
    so on a truncated step it is the value of the terminal observation, not of the
    reset observation. Termination zeroes the bootstrap. Either end stops the
    advantage from leaking across the episode boundary.
    """
    not_terminated = 1.0 - terminated
    continues = not_terminated * (1.0 - truncated)
    advantages = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[0])
    for t in reversed(range(rewards.shape[0])):
        delta = rewards[t] + gamma * not_terminated[t] * next_values[t] - values[t]
        running = delta + gamma * gae_lambda * continues[t] * running
        advantages[t] = running
    return advantages


class Agent(nn.Module):
    def __init__(self, input_size: int, action_size: int) -> None:
        super().__init__()

        def network(output_size: int, gain: float) -> nn.Sequential:
            layers = [nn.Linear(input_size, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh()]
            layers.append(nn.Linear(64, output_size))
            for layer in layers:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    nn.init.zeros_(layer.bias)
            nn.init.orthogonal_(layers[-1].weight, gain=gain)
            return nn.Sequential(*layers)

        self.critic = network(1, gain=1.0)
        self.actor_mean = network(action_size, gain=0.01)
        self.actor_logstd = nn.Parameter(torch.full((action_size,), -1.0))

    def value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x).squeeze(-1)

    def policy(self, x: torch.Tensor) -> Normal:
        mean = self.actor_mean(x)
        return Normal(mean, self.actor_logstd.exp().expand_as(mean))


def clipped_to(action: torch.Tensor, env: SharpeArenaVectorEnv) -> np.ndarray:
    space = env.single_action_space
    return np.clip(action.numpy(), space.low, space.high, dtype=space.dtype)


def true_next_values(
    agent: Agent, next_obs: dict[str, np.ndarray], infos: dict, ended: np.ndarray
) -> torch.Tensor:
    """Value of what followed each lane: its terminal observation if the lane just ended."""
    bootstrap_obs = {key: batch.copy() for key, batch in next_obs.items()}
    for lane in np.flatnonzero(ended):
        for key, value in infos["final_obs"][lane].items():
            bootstrap_obs[key][lane] = value
    return agent.value(features(bootstrap_obs))


def train(args: Args) -> tuple[Agent, list[float]]:
    """Run PPO and return the agent with the finished-episode training returns."""
    torch.manual_seed(args.seed)
    env = make_env(args, base_seed=args.seed)
    obs, _ = env.reset()
    input_size = features(obs).shape[-1]
    agent = Agent(input_size, env.single_action_space.shape[0])
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    shape = (args.num_steps, args.num_envs)
    rollout_inputs = torch.zeros((*shape, input_size))
    rollout_actions = torch.zeros((*shape, env.single_action_space.shape[0]))
    log_probs, rewards = torch.zeros(shape), torch.zeros(shape)
    terminated, truncated = torch.zeros(shape), torch.zeros(shape)
    values, next_values = torch.zeros(shape), torch.zeros(shape)

    episode_returns: list[float] = []
    running_return = np.zeros(args.num_envs)
    batch_size = args.num_steps * args.num_envs
    minibatch_size = batch_size // args.num_minibatches

    for _ in range(args.total_timesteps // batch_size):
        for t in range(args.num_steps):
            inputs = features(obs)
            with torch.no_grad():
                distribution = agent.policy(inputs)
                action = distribution.sample()
                log_probs[t] = distribution.log_prob(action).sum(-1)
                values[t] = agent.value(inputs)
            rollout_inputs[t], rollout_actions[t] = inputs, action

            obs, reward, term, trunc, infos = env.step(clipped_to(action, env))
            ended = term | trunc
            rewards[t] = torch.as_tensor(reward, dtype=torch.float32)
            terminated[t] = torch.as_tensor(term, dtype=torch.float32)
            truncated[t] = torch.as_tensor(trunc, dtype=torch.float32)
            with torch.no_grad():
                next_values[t] = true_next_values(agent, obs, infos, ended)

            running_return += reward
            episode_returns.extend(running_return[ended].tolist())
            running_return[ended] = 0.0

        advantages = compute_gae(
            rewards, values, next_values, terminated, truncated, args.gamma, args.gae_lambda
        )
        returns = advantages + values

        b_inputs = rollout_inputs.flatten(0, 1)
        b_actions = rollout_actions.flatten(0, 1)
        b_log_probs, b_advantages = log_probs.flatten(), advantages.flatten()
        b_returns = returns.flatten()
        for _epoch in range(args.update_epochs):
            for index in torch.randperm(batch_size).split(minibatch_size):
                distribution = agent.policy(b_inputs[index])
                new_log_prob = distribution.log_prob(b_actions[index]).sum(-1)
                ratio = (new_log_prob - b_log_probs[index]).exp()
                advantage = b_advantages[index]
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
                clipped = ratio.clamp(1.0 - args.clip_coef, 1.0 + args.clip_coef)
                policy_loss = -torch.min(advantage * ratio, advantage * clipped).mean()
                value_loss = (agent.value(b_inputs[index]) - b_returns[index]).pow(2).mean()
                entropy = distribution.entropy().sum(-1).mean()
                loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

    return agent, episode_returns


def evaluate(agent: Agent, args: Args) -> list[float]:
    """Play one greedy episode per held-out lane; these seeds are never trained on."""
    env = make_env(args, base_seed=EVAL_SEED_BASE)
    obs, _ = env.reset()
    totals = np.zeros(args.num_envs)
    finished = np.zeros(args.num_envs, dtype=bool)
    while not finished.all():
        with torch.no_grad():
            mean = agent.policy(features(obs)).mean
        obs, reward, term, trunc, _ = env.step(clipped_to(mean, env))
        totals += np.where(finished, 0.0, reward)
        finished |= term | trunc
    return totals.tolist()


def parameter_digest(agent: Agent) -> str:
    digest = hashlib.sha256()
    for tensor in agent.state_dict().values():
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for field in dataclasses.fields(Args):
        flag = "--" + field.name.replace("_", "-")
        parser.add_argument(flag, type=type(field.default), default=field.default)
    args = Args(**vars(parser.parse_args()))

    agent, episode_returns = train(args)
    held_out = evaluate(agent, args)
    print(f"training episodes finished:     {len(episode_returns)}")
    print(f"held-out mean episode return:   {np.mean(held_out):.6f}")
    print(f"parameter digest:               {parameter_digest(agent)}")
    print("Not a benchmark result: the budget is a demonstration, not an experiment.")


if __name__ == "__main__":
    main()
