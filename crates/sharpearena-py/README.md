# SharpeArena for Python

The Python distribution combines the native SharpeArena engine with a
Gymnasium-compatible environment and optional adapters for multi-agent, RLVR,
offline-RL, MCP, and local-model workflows.

> Point-in-time observations prevent future-bar access through the environment API.
> They do not isolate Python code from the host. Run trusted local code here, or use
> SharpeBench's digest-pinned container path for an untrusted entrant.

## Install

```bash
pip install sharpearena
```

The base install includes NumPy, Gymnasium, the Python package, and the compiled pyo3
extension. Optional integrations are explicit:

```bash
pip install "sharpearena[pettingzoo]"
pip install "sharpearena[verifiers]"
pip install "sharpearena[minari]"
pip install "sharpearena[mcp]"
```

## Step a Gymnasium environment

```python
from sharpearena import SharpeArenaEnv

env = SharpeArenaEnv(n_symbols=4, n_days=120, seed=7)
observation, info = env.reset(seed=7)

terminated = truncated = False
while not (terminated or truncated):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
```

Actions are signed target-weight vectors in the environment's symbol order. The
default range is `[-1, 1]`; pass `allow_short=False` to make it `[0, 1]`, or set
`max_weight` to change the per-symbol bound.

Importing `sharpearena` registers the following Gymnasium IDs:

```text
SharpeArena/Calm-v1          SharpeArena/Calm-Eval-v1
SharpeArena/Hard-v1          SharpeArena/Hard-Eval-v1
SharpeArena/Extreme-v1       SharpeArena/Extreme-Eval-v1
```

The `-Eval-v1` environments draw from the disjoint evaluation seed band. The `-v1`
suffix freezes the environment semantics; a rule change requires a new versioned ID.

## Native boundary

`sharpearena.sharpearena_py.TradingEnv` is the low-level binding. Its boundary is
JSON: `reset()` returns an observation string, and `step(decision_json)` returns
`(observation_json, reward, done, info_json)`. The Python Gymnasium wrapper converts
between that wire format and NumPy spaces.

The wheel ships `py.typed` and a stub for the compiled extension. Typed boundary
exceptions distinguish invalid input, invalid JSON, invalid salt, unavailable data,
and engine failures.

## Other surfaces

| Task | Surface |
|---|---|
| Batched Gymnasium rollout | `SharpeArenaVectorEnv` or `gymnasium.make_vec(...)` |
| Multi-agent environment | `MultiAgentSharpeArenaEnv` with the `pettingzoo` extra |
| RLVR / Prime RL | `load_environment()` with the `verifiers` extra |
| Offline RL export | `to_minari`, `to_minari_train_test` with the `minari` extra |
| External tool server | MCP server with the `mcp` extra |
| Evidence field | `sharpearena-local-field` and the model transport shims |
| Benchmark compilation | `sharpearena-compile-bench` |
| Strategy search | `sharpearena-strategy-search` |

See the repository's [Gymnasium guide](../../docs/gymnasium.md),
[training guide](../../docs/training.md), and
[agent contract](../../docs/agent-contract.md) for the full workflows.

## Build from source

```bash
python -m maturin develop --manifest-path crates/sharpearena-py/Cargo.toml
```

## License

MIT OR Apache-2.0
