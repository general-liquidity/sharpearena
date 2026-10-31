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

`reset(seed=k)` rebuilds the native batch on consecutive scenario seeds
`k, k+1, ...`. `reset(seed=[a, b, ...])` accepts exactly one integer per lane.
Reset seeds must be nonnegative int64 values; invalid inputs leave the batch
and any queued action unchanged. The window, scenario knobs, costs and an
explicit execution seed are preserved; without an execution override, each
lane's execution seed follows its scenario seed. `scenario_seeds`, native
dataset readbacks and `info["scenario_seed"]` describe the rebuilt lanes.
`reset()` keeps the current scenarios and clears pending actions and counters.
By default autoreset also replays those scenarios, not fresh independent draws;
see [Fresh scenarios on recycle](#fresh-scenarios-on-recycle) for the opt-in.

Without constructor seeds, training lanes begin at zero and evaluation lanes
begin at `EVAL_SEED_BASE`. Constructor and explicit reset seeds are literal,
including in evaluation mode: choose the evaluation band yourself when passing
them. For example, use `reset(seed=EVAL_SEED_BASE + 7)` for evaluation lanes.

`max_episode_steps` is now enforced as a positive integer cap on executed
steps per lane, without shortening the data window or changing rewards before
the cap. The earlier native horizon still wins. A cap sets `truncated`, not
`terminated` (bankruptcy can independently set that flag). `next_step` resets
on the following call with zero reward, `same_step` preserves the terminal
observation/info in `final_obs`/`final_info`, and `disabled` freezes a capped
lane with zero subsequent reward until explicit reset. Reset calls do not
consume the cap. `None` preserves the native horizon. Record the cap alongside
the environment ID and window; a custom cap changes the task length, not the
frozen scenario/fill rules behind `-v1`.

## Fresh scenarios on recycle

With the default `autoreset_scenarios="replay"`, a recycled lane restarts its own
scenario, so a run sees only as many scenarios as lanes. Training over more scenarios
than lanes needs the opt-in:

```python
from sharpearena import SharpeArenaVectorEnv

envs = SharpeArenaVectorEnv(num_envs=8, autoreset_scenarios="fresh_v1")
```

Under `fresh_v1` each recycled episode draws a new scenario seed. The seed is a pure
function of the lane's construction seed, its lane index and its episode count, so two
runs with the same arguments draw the same seeds. An explicit execution seed in
`env_kwargs` is kept; otherwise the lane's execution seed follows the drawn seed, as it
does at construction.

- Every draw lands in the training band `[0, EVAL_SEED_BASE)`. Evaluation mode and any
  lane or `reset(seed=...)` outside that band are refused, so a run cannot mix held-out
  scenarios into training.
- `info["scenario_seed"]` names the scenario each returned observation belongs to, and
  `info["drawn_scenario_seed"]` holds the seed a lane drew on that step, or `None`. Under
  `same_step`, `final_info[i]["scenario_seed"]` names the episode that just finished.
- `effective_configs` adds the stream's root seed, lane and band, every drawn seed in
  order, the live scenario seed and a fingerprint of the dataset the lane holds now. The
  base fields keep describing the lane as constructed.
- `reset()` keeps the current scenarios and the stream position. `reset(seed=...)`
  reopens the stream from the new seeds. `disabled` lanes never recycle, so they never
  draw. CSV lanes share one frozen path and have no fresh mode.
- Draws are with replacement, so a seed can repeat within or across lanes. The label is
  versioned: `fresh_v1` always yields the same stream, and a different derivation would
  ship under a new label.

## Binary step path

`step` crosses the native boundary as JSON, which stays the canonical wire format.
`step_array` is a second method on `SharpeArenaEnv` and `SharpeArenaVectorEnv` for
trainers that only need numbers. It moves float64 arrays through the buffer protocol
with no JSON per step:

```python
observation, reward, terminated, truncated, info = env.step_array(action)
```

It accepts the same actions as `step` and returns the same observations, rewards and
flags byte for byte, with the same recycling and `fresh_v1` draws. It builds the same
decision the adapter would send as JSON, including action labels and confidence, and the
engine's state snapshot matches too. The one difference is what comes back:
`info` and `final_info` carry `nav` but no `events`. The events stay in the engine's
trace, and `step` is the way to read them per step. Each call returns newly allocated
arrays, so a stored observation is never overwritten. The two paths can be mixed within
one episode.

The native `TradingEnv.step_array` and `VecTradingEnv.step_batch_array` write into
caller-owned arrays, which lets a tight loop reuse its buffers. They refuse a wrong dtype,
a wrong shape, a read-only output or a non-finite weight before the engine advances.

## Reproducible evaluation

Keep scenario and evaluation identity with the results:

- environment ID or complete constructor arguments;
- user seed plus resolved scenario and execution seeds from `info`;
- the `autoreset_scenarios` label and, under `fresh_v1`, the drawn seeds;
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

### Checkpoint exports

`CheckpointableEnv.clone_state()` returns private restoration state, including the
scenario seed and, for a CSV-backed environment, the entire input CSV. Keep the object,
its pickle, and any private export away from evaluated agents.

```python
snapshot = checkpointed_env.clone_state()
public_record = snapshot.to_dict()  # step and past actions only; not restorable
private_record = snapshot.to_dict(include_private=True)  # operator-only
checkpointed_env.restore_state(CheckpointState.from_dict(private_record))
```

This is an export-policy distinction, not process isolation. The public record does
not sanitize information deliberately encoded in actions. `from_dict` rejects public
records and unknown schemas; it continues to read legacy private records containing
`params`. Do not unpickle untrusted input. Native snapshots avoid action replay, but
copying, serialization and restoration scale with state size rather than taking
constant time.

## Trust boundary

Gymnasium controls what market data reaches the policy through the environment API. It
does not prevent Python policy code from reading host files, environment variables, or
the network. Use only trusted local policy code here. Use SharpeBench's digest-pinned
container runner when process containment is required.
