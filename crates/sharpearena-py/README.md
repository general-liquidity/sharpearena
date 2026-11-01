# SharpeArena for Python

The Python distribution combines SharpeArena's deterministic point-in-time
evaluation sandbox with a Gymnasium-compatible environment and optional adapters
for multi-agent, RLVR, offline-RL, MCP, and local-model workflows.

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

## Scoring and typed unavailability

`score_run(returns, n_trials, periods_per_year)` returns the pinned SharpeBench
`CompositeScore` as JSON. When the kernel cannot score a series (a non-finite
observation, fewer than two observations) the composite carries a typed error
beside a no-skill floor rather than an estimate: `deflation_error` (the deflated
Sharpe, its bar and its interval are the floor), `bootstrap_error` (the bootstrap
p-value is the conservative `1.0` sentinel) and `selection_error` (the selection
diagnostic is absent). Reading `deflated_sharpe` past such a key publishes the
floor as a score.

A constant track (every return equal, such as the `flat` baseline's all-zero
series) has a sample variance of zero and so no Sharpe ratio. The pinned kernel
would still score it, so `score_run` withholds it: `deflation_error` reads
`returns must not be constant: a constant series has no Sharpe ratio` (or
`Sharpe ratio is not finite` for a track whose computed standard deviation
underflows to zero), `deflated_sharpe` and `psr` read the `0.0` floor, the
interval and rolling Sharpe fields are absent or null, and `passed_k` and
`rank_eligible` are false. The seed bootstrap (`deflated_sharpe_ci`,
`paired_dsr_diff`) raises `ValueError` with the same reason. The predicate is
value equality, not a volatility tolerance, so a dispersed track keeps its score.

`sharpearena.kernel_score` is the one place a consumer reads the ranked numbers:

- `kernel_deflated_sharpe(composite)` and `kernel_psr(composite)` return the
  finite value or raise. An unavailability key (`deflation_error`,
  `bootstrap_error`, `selection_error`) raises `KernelScoreUnavailable`; a gate
  refusal key (`standard_error_refusal`, `exposure_path_error`) raises
  `KernelScoreRefused`; both derive from `KernelScoreWithheld` and name every key
  present. Any other `*_error` or `*_refusal` key raises
  `KernelFieldUnrecognized` and is never recorded. Selection, ranking inputs and
  the paper producers use these and stop. The
  [kernel field convention](../../docs/kernel-field-convention.md) lists the
  allowlists.
- `kernel_score_or_unavailable(composite, key="deflated_sharpe")` returns the
  float or the reason string `unavailable_scoring_kernel_error: <key>: <message>`
  (or `refused_scoring_kernel_gate: <key>: <message>` for a refusal) for rows that must be recorded (eval-seed snapshots, generalization and regime
  rows, the PettingZoo leaderboard, trace `meta`, the local-field journal).
  `kernel_score_difference` makes a gap unavailable when either side is;
  `is_kernel_score_unavailable` tests a recorded cell and
  `is_kernel_score_refused` picks out a recorded refusal.

Every `score_run` consumer in this package goes through these helpers; none
substitutes a default. The multi-agent ranking places an unscorable agent after
every scored one with the reason in its cell.

For the private-copy PettingZoo tournament, fewer than two observed returns are
also explicitly unavailable, not a numeric zero. `env.final_ranking` exposes a
detached whole-roster result after all agents finish, including agents removed
earlier; it is `None` before completion and after reset. Step dictionaries retain
the Parallel live-agent contract. Row `rank` is a display position, not a
SharpeBench eligibility certificate or a full benchmark-field score.

## Other surfaces

| Task | Surface |
|---|---|
| Batched Gymnasium rollout | `SharpeArenaVectorEnv` or `gymnasium.make_vec(...)` |
| Multi-agent environment | `MultiAgentSharpeArenaEnv` with the `pettingzoo` extra |
| RLVR / Prime RL | `load_environment()` with the `verifiers` extra |
| Offline RL export | `to_minari`, `to_minari_train_test` with the `minari` extra |
| External tool server | MCP server with the `mcp` extra |
| Evidence field | `sharpearena-local-field` and the model transport shims; raw schema 2 records source-labelled per-request duration and token/retry accounting |
| Benchmark compilation | `sharpearena-compile-bench`; bridge schema 2 adds a validated, rank-neutral p50/p95 operational profile beside score submissions |
| Strategy search | `sharpearena-strategy-search` |

Vector `reset(seed=k)` rebuilds native lanes on `k, k+1, ...`; a list supplies
exactly one nonnegative int64 seed per lane. Unseeded reset and autoreset replay
the current scenarios. Explicit seeds are literal, not shifted into the eval
band. `max_episode_steps` enforces a positive executed-step cap per lane, setting
truncation while preserving the selected autoreset mode; `None` keeps the native
horizon. Disabled capped lanes freeze with zero subsequent reward. Log custom
caps with the window and seeds when comparing runs.

See the repository's [Gymnasium guide](../../docs/gymnasium.md),
[training guide](../../docs/training.md), and
[agent contract](../../docs/agent-contract.md) for the full workflows.

## Build from source

```bash
python -m maturin develop --manifest-path crates/sharpearena-py/Cargo.toml
```

## License

MIT OR Apache-2.0
