# Gymnasium guide

Use the Python package when you need a conventional Gymnasium lifecycle over the
deterministic SharpeArena engine. The package supports direct construction, registered
difficulty IDs, vector rollout, held-out seed bands, and optional adapters for
PettingZoo, `verifiers`, and Minari.

## Install

```bash
pip install sharpearena
```

The base wheel includes NumPy, Gymnasium, the pure-Python wrappers, and the compiled
Rust extension. Integrations are optional:

```bash
pip install "sharpearena[pettingzoo]"
pip install "sharpearena[verifiers]"
pip install "sharpearena[minari]"
pip install "sharpearena[mcp]"
```

## Direct construction

```python
from sharpearena import SharpeArenaEnv

env = SharpeArenaEnv(
    n_symbols=4,
    n_days=120,
    seed=7,
    distribution_mode="hard",
    allow_short=True,
    max_weight=1.0,
)

observation, info = env.reset(seed=7)
terminated = truncated = False

while not (terminated or truncated):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
```

The action is one target weight per symbol in `env.symbols`. Positive values are long,
negative values are short, and zero is flat. `allow_short=False` clips the lower action
bound to zero. The observation contains current closes, positions, and cash; the native
wire observation remains available through the low-level binding.

Calling `reset(seed=k)` rebuilds a deterministic scenario and derives independent
scenario and execution-noise seeds from `k`. Calling `reset()` without a seed retains
the current scenario.

## Registered environments

Importing `sharpearena` registers six versioned IDs:

| Training band | Held-out evaluation band |
|---|---|
| `SharpeArena/Calm-v1` | `SharpeArena/Calm-Eval-v1` |
| `SharpeArena/Hard-v1` | `SharpeArena/Hard-Eval-v1` |
| `SharpeArena/Extreme-v1` | `SharpeArena/Extreme-Eval-v1` |

```python
import gymnasium
import sharpearena

env = gymnasium.make("SharpeArena/Hard-v1")
eval_env = gymnasium.make("SharpeArena/Hard-Eval-v1")
```

The evaluation IDs offset scenarios into the disjoint band beginning at
`EVAL_SEED_BASE = 1_000_000`. The suffix `-v1` freezes the scenario, fill, cost, and
point-in-time semantics. A change to those rules requires a new environment ID.

## Vector rollout

```python
import gymnasium
import sharpearena

envs = gymnasium.make_vec("SharpeArena/Hard-v1", num_envs=8)
observations, info = envs.reset(seed=7)
observations, rewards, terminations, truncations, info = envs.step(
    envs.action_space.sample()
)
```

The registered vector entry point is `SharpeArenaVectorEnv`. It drives the native
batched environment rather than a separate Python market implementation.

## Reproducible evaluation

Keep scenario and evaluation identity with the results:

- environment ID or complete constructor arguments;
- user seed plus resolved scenario and execution seeds from `info`;
- window and data identity;
- cost, mandate, wrapper, and reward configuration;
- package version, `CONTRACT_VERSION`, and `SPEC_HASH`;
- whether the run used the train band, public eval band, or a sealed seed set.

Evidence producers in this repository also read effective configuration back from the
constructed environment and refuse a requested/effective mismatch. Echoing the
driver's request is not a readback.

## Optional integrations

| Workflow | Entry point |
|---|---|
| PettingZoo shared market | `MultiAgentSharpeArenaEnv`, `make_aec_env` |
| RLVR / Prime RL | `load_environment()` |
| Minari dataset export | `to_minari`, `to_minari_train_test` |
| Checkpoint and branching | `CheckpointableEnv` |
| Lookahead checks | `LookaheadGuard` and causal wrappers |
| Functional transforms | `SharpeArenaFuncEnv` |

For reward design, disjointness conventions, sealed evaluation, and Prime RL, continue
to the [training guide](training.md). For external model processes, read the
[agent contract](agent-contract.md).

## Trust boundary

Gymnasium controls what market data reaches the policy through the environment API. It
does not prevent Python policy code from reading host files, environment variables, or
the network. Use only trusted local policy code here. Use SharpeBench's digest-pinned
container runner when process containment is required.
