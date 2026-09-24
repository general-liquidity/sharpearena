# Interface examples

The README's [Choose an interface](../README.md#choose-an-interface) table names
every route into the engine. Gymnasium has a worked example in the root
[README](../README.md#quick-start) already. This page gives a short, runnable
example for each of the others, so a reader can start on the route they care about
without reading the module source first.

Every example on this page was run against this tree and this page states which
package versions it ran against. Install the extra it names before you run it:
`pip install "sharpearena[pettingzoo,verifiers,minari,mcp]"` installs those four
at once; `sb3` is left out of that combined install because it pulls in torch,
the largest dependency any extra here declares, and `ray` is left out because it
is bounded separately (see below).

## PettingZoo

`MultiAgentSharpeArenaEnv` (`crates/sharpearena-py/python/sharpearena/pettingzoo_env.py`)
runs a batched tournament: each agent trades its own copy of the same frozen
scenario, so every agent sees an identical price path, and agents are ranked by
realized Sharpe at episode end. It does not model shared market impact between
agents; see the module docstring for that distinction.

```python
from sharpearena.pettingzoo_env import MultiAgentSharpeArenaEnv

env = MultiAgentSharpeArenaEnv(n_agents=2, n_symbols=4, n_days=30, seed=1)
observations, infos = env.reset(seed=1)
for agent in env.agents:
    env.action_space(agent).seed(0)

for _ in range(5):
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)

print(sorted(rewards.items()))
```

Run against `pettingzoo` 1.26.1, installed with `pip install "sharpearena[pettingzoo]"`.
Output:

```
[('agent_0', -0.027227086041561654), ('agent_1', -0.005926892224436098)]
```

`lob_env.py` and `market_env.py` cover the limit-order-book and shared-impact tasks
under the same package; start from this one and follow the imports if you need
those.

## `verifiers` and Prime-RL

`sharpearena.verifiers_env` is a multi-turn `verifiers` environment: one bar per
turn, a parsed `<action>` decision, and a reward built from the real SharpeBench
kernel. The module docstring states it was verified against `verifiers` 0.3.1,
which is also the CI pin (INT-10, `docs/integrations/inventory.md`, is closed).
0.3.1 keeps the same v0/`MultiTurnEnv` API this module targets as
`verifiers.legacy`, aliased transparently onto the top-level `verifiers.*` names, so
no source change was needed; a structural capability probe
(`sharpearena.verifiers_env.UnsupportedVerifiersAPIError`) now guards that surface so
a future release that drops or reshapes it fails loudly instead of silently.

```python
from sharpearena.verifiers_env import load_environment

env = load_environment(n_windows=2, n_symbols=4, n_days=30, max_episode_bars=10)
print(type(env).__name__)
print(env.dataset)
```

Run against `verifiers` 0.3.1, installed with `pip install "sharpearena[verifiers]"`.
Output:

```
SharpeArenaVerifiersEnv
Dataset({
    features: ['question', 'answer', 'info', 'example_id', 'prompt'],
    num_rows: 2
})
```

This builds the dataset and the rollout environment only. Driving a rollout needs a
model client (Prime-RL or any `verifiers`-compatible trainer); see
`examples/prime-rl/README.md` for that wiring.

## Minari export

`sharpearena.minari_export.to_minari_train_test` converts two recorded rollout
traces into a train and a test `minari.MinariDataset`, split over disjoint seed
bands rather than a random episode shuffle, so a train scenario can never land in
the test set. Record a trace with `sharpearena.trace.RolloutTraceWriter`, then
export it:

```python
import tempfile, os
from sharpearena import SharpeArenaEnv
from sharpearena.trace import RolloutTraceWriter
from sharpearena.minari_export import to_minari_train_test
from sharpearena.dataset import EVAL_SEED_BASE

def record(seed, path):
    env = SharpeArenaEnv(n_symbols=4, n_days=30, seed=seed)
    obs, info = env.reset(seed=seed)
    writer = RolloutTraceWriter(path, config={"n_symbols": 4, "n_days": 30})
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        writer.record_step(step=step, observation=obs, decision=action,
                            reward=reward, info=info)
        if terminated or truncated:
            break
    writer.finalize()
    writer.close()
    return env

tmp = tempfile.mkdtemp()
train_env = record(1, os.path.join(tmp, "train.jsonl"))
test_env = record(EVAL_SEED_BASE, os.path.join(tmp, "test.jsonl"))

train_ds, test_ds = to_minari_train_test(
    os.path.join(tmp, "train.jsonl"), os.path.join(tmp, "test.jsonl"),
    "sharpearena/quickstart-v0",
    observation_space=train_env.observation_space,
    action_space=train_env.action_space,
)
print(train_ds.id, train_ds.total_episodes)
print(test_ds.id, test_ds.total_episodes)
```

Run against `minari` 0.5.3, installed with `pip install "sharpearena[minari]"`.
Output (Minari's own dataset-metadata warnings, which fire on every export that
skips optional fields like `author`, are omitted below):

```
sharpearena/quickstart-train-v0 1
sharpearena/quickstart-test-v0 1
```

Minari stores an exported dataset under a local dataset-id path and refuses to
recreate an id that already exists on disk; pick a new `dataset_id` or delete the
old one before re-running this example a second time.

## Stable-Baselines3

`sharpearena.sb3_env.SharpeArenaSB3VecEnv` is a Stable-Baselines3 `VecEnv` over the
same native batched engine `SharpeArenaVectorEnv` wraps for Gymnasium, fixed at
`autoreset_mode="same_step"` because that is the only mode whose autoreset and
terminal-observation semantics match SB3's own contract; see the module docstring
in `crates/sharpearena-py/python/sharpearena/sb3_env.py` for the mapping and why
the other two modes are refused. The observation is a single-level `Dict`, so the
policy is `MultiInputPolicy`.

```python
from sharpearena.sb3_env import SharpeArenaSB3VecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

train = SharpeArenaSB3VecEnv(seeds=[0, 1], n_symbols=3, n_days=24)
model = PPO(
    "MultiInputPolicy", train, n_steps=32, batch_size=16, n_epochs=1,
    policy_kwargs={"net_arch": [16]}, device="cpu", verbose=0, seed=0,
)
model.learn(total_timesteps=64)
mean_reward, std_reward = evaluate_policy(model, train, n_eval_episodes=2, warn=False)
print(f"{mean_reward:.6f} +/- {std_reward:.6f}")
```

Run against `stable-baselines3` 2.9.0, installed with
`pip install "sharpearena[sb3]"`, and CPU-only `torch`. Output (mean episode
reward at this tiny a budget is not a benchmark result; see
`examples/sb3/train_ppo.py` for the runnable recipe with training and evaluation
lanes drawn from disjoint seed bands):

```
-0.000219 +/- 0.001918
```

## TorchRL

`SharpeArenaTorchRLEnv` (`crates/sharpearena-py/python/sharpearena/torchrl_env.py`)
is a `torchrl.envs.EnvBase` subclass driving the same engine as
`SharpeArenaEnv` underneath. TorchRL environments read and write `TensorDict`
instances rather than the Gymnasium 5-tuple, and the step output lives under a
`"next"` entry; a policy is any callable that sets an `"action"` key on the
tensordict it is handed. `equal_weight_policy` below is not a learner, only a
deterministic policy that makes the rollout reproducible for this example.

```python
from sharpearena.torchrl_env import SharpeArenaTorchRLEnv, equal_weight_policy

env = SharpeArenaTorchRLEnv(n_symbols=4, n_days=30, seed=1)
policy = equal_weight_policy(env)
rollout = env.rollout(5, policy=policy)
print(rollout.get(("next", "reward")).squeeze(-1).tolist())
```

Run against `torchrl` 0.14.0 (`tensordict` 0.14.2, `torch` 2.14.0+cpu), installed
with `pip install "sharpearena[torchrl]"`. Output:

```
[-0.0031250825670572357, 0.0016629164935610952, 0.001159297585380914, 0.001013520026747372, 0.0003075342205489662]
```

Observations and the reward cross the TensorDict boundary as `torch.float64`,
the engine's own width, unless `obs_dtype=torch.float32` is named explicitly.
Actions cross as `torch.float32`, matching the `Box` `SharpeArenaEnv.action_space`
advertises; `_action_validation.validated_action` widens them back to `float64`
before they reach the engine. `docs/rl-contract-coverage.md` states the native
dtype-parity standard this adapter matches, and `tests/test_torchrl.py` pins
both directions of the boundary plus a `torchrl.collectors.Collector` round trip
through serialization and replay.

## MCP

`sharpearena.mcp_server.build_server` returns a `FastMCP` server exposing `reset`,
`step`, and `spec` tools over an episode env. `main()` (`python -m
sharpearena.mcp_server`) runs it over streamable HTTP for a real client. This lists
the tools without starting a transport or calling a model:

```python
import asyncio
from sharpearena.mcp_server import build_server

server = build_server(env_kwargs={"n_symbols": 4, "n_days": 30, "seed": 1})

async def main():
    tools = await server.list_tools()
    for tool in tools:
        print(tool.name)

asyncio.run(main())
```

Run against `mcp` 1.28.1, installed with `pip install "sharpearena[mcp]"`. Output:

```
reset
step
spec
```

## The JSON contract (stdin/stdout or `POST /decide`)

The [agent contract guide](agent-contract.md) is the authoritative spec for the
wire shapes; this is the shortest way to see them move. The reference stdio agent
(`crates/sharpearena/examples/reference-agent.py`) reads one `MarketObservation`
JSON line and writes one `Decision` JSON line, with no dependency beyond the
standard library, so it works from any language that can speak newline-delimited
JSON on stdio or send the same body to `POST /decide`:

```bash
echo '{"date":"2025-01-02","cash":1.0,"symbols":[{"symbol":"AAPL","close_history":[187.2,188.0,190.4]}],"portfolio":[]}' \
  | python crates/sharpearena/examples/reference-agent.py
```

Output:

```
{"orders": [{"symbol": "AAPL", "action": "buy", "target_weight": 1.0, "confidence": 0.5, "rationale": "equal-weight hold"}], "reasoning": "equal-weight buy-and-hold"}
```

The schemas behind these shapes are
[`observation.schema.json`](../crates/sharpearena/contract/observation.schema.json)
and
[`decision.schema.json`](../crates/sharpearena/contract/decision.schema.json).
`target_weight` is signed for shorts and must lie in `[-1, 1]`; `confidence` is
optional and lies in `[0, 1]`; an empty `orders` array is a deliberate hold, not an
error.

## The SharpeBench bridge

`sharpearena.bench_bridge.compile_benchmark_evidence` turns a completed local-field
journal (one JSONL record per model call, written by
`sharpearena.local_agents.LocalFieldRunner`) into the submissions file and manifest
SharpeBench scores. It validates the field is complete before it will produce
anything: an incomplete grid, a failed cell, a coordinate collision, or an invalid
return hash all refuse rather than emit a partial result.

This example uses a fixed, deterministic stand-in model (the same fixture shape
the test suite uses) instead of a real local model, so it runs with no model
weights and no network access:

```python
import tempfile
from pathlib import Path
from sharpearena.local_agents import (
    DatasetSpec, EvidenceJournal, FieldPlan, InferenceOutcome, InferenceResult,
    LocalFieldRunner, ModelIdentity, ModelRunConfig, SamplingConfig,
)
from sharpearena.bench_bridge import compile_benchmark_evidence

class FixedModel:
    def identity(self, model):
        return ModelIdentity(model=model.model, digest="sha256:fixed",
                              parameter_size="test", quantization="none",
                              family="fixture", server="fixture", server_version="1")

    def decide_many(self, observations, model, renderer, *, max_workers, sampling_seeds=None):
        return [
            InferenceOutcome(result=InferenceResult(
                decision={
                    "orders": [{"symbol": s["symbol"], "action": "buy",
                                "target_weight": 0.1, "confidence": 0.5,
                                "rationale": "fixture"} for s in obs["symbols"]],
                    "reasoning": "fixture",
                    "cost": {"cost_usd": 0.0, "tokens_in": 10, "tokens_out": 5,
                             "reasoning_tokens": 0},
                },
                raw_response_sha256=f"response-{seed}", prompt_tokens=10, output_tokens=5,
                reasoning_tokens=None, total_duration_ns=100,
                duration_source="host-monotonic-request", raw_response=f"raw-{seed}",
            ))
            for obs, seed in zip(observations, sampling_seeds)
        ]

plan = FieldPlan(
    models=(ModelRunConfig("test-fixture:synthetic", SamplingConfig(seed=40),
                            decision_cadence=2, entry_class="field",
                            source_url="https://example.test/models/test-fixture",
                            source_revision="0123456789abcdef", license_id="MIT"),),
    datasets=(DatasetSpec("synthetic-calm", tier="calm", n_symbols=2, n_days=12),),
    seeds=(1, 2), repetitions=2, max_steps=5,
)

with tempfile.TemporaryDirectory() as tmp:
    journal = Path(tmp) / "field.jsonl"
    counts = LocalFieldRunner(FixedModel()).run(plan, EvidenceJournal(journal))
    print("run counts:", counts)
    manifest = compile_benchmark_evidence([journal], Path(tmp) / "compiled")
    print("outputs:", [o["submissions_path"] for o in manifest["outputs"]])
```

Run against this tree, `sharpearena` 0.31.0. Output (the temp path will differ on
your machine):

```
run counts: {'completed': 4, 'failed': 0, 'skipped': 0}
outputs: ['/tmp/.../compiled/00-synthetic-calm.submissions.json']
```

A real field run replaces `FixedModel` with `OllamaClient` or
`OpenAICompatibleClient` from the same module; see the [local-agent
architecture](LOCAL_AGENT_ARCHITECTURE.md) guide for that path.

## Ray and RLlib

`sharpearena.ray_executor` (INT-06) runs episodes across Ray tasks with a
deterministic reduction: `run_episodes` places results by `episode_id` rather than
completion order, so the reduced, canonically-digested report is identical whether
it ran locally (`num_workers=0`, no Ray import at all) or on any number of Ray
workers. Ray retries a task at least once, not exactly once; the guard against
double-counting a retry is in `reduce_results`, not a comment: a duplicate result
for an `episode_id` already placed is dropped when identical and raises
`ExecutorViolation` when it differs, because two different answers under one
identity means the episode was not the pure function of its spec the retry safety
depends on.

```python
import ray
from sharpearena.ray_executor import EpisodeSpec, run_episodes, report_digest
from sharpearena.integrations.parity import deterministic_actions

specs = [
    EpisodeSpec(
        episode_id=f"ep-{i}", seed=i, n_symbols=3, n_days=20, max_steps=15,
        actions=deterministic_actions(n_symbols=3, n_steps=15, seed=i),
    )
    for i in range(4)
]
ray.init(num_cpus=2, include_dashboard=False, log_to_driver=False)
report = run_episodes(specs, num_workers=2)
ray.shutdown()
print(report_digest(report))
```

Run against `ray[rllib]` 2.58.0. Output:

```
59cd261135f95e3b4d359b0ec0b9bac335edb0bbb392219072a7fb70de35fac1
```

`run_episodes(specs, num_workers=0)` on the same `specs` produces the identical
digest; `tests/test_ray_executor.py::test_report_digest_is_identical_across_worker_counts`
checks this at 0, 2 and 3 workers rather than asserting it once by hand.

`sharpearena.rllib_env` (INT-07) registers single-agent and multi-agent PPO routes
through `ray.tune.registry.register_env`, not Gymnasium's own registry, because
`SingleAgentEnvRunner.make_env` resolves environments from the Tune registry and
Gymnasium's registry does not reach a remote `EnvRunner` actor. Both routes wrap
the environment's `Dict` observation in `FlattenObservation`
(`sharpearena.spaces`) before RLlib sees it, because the default RLModules have no
encoder for a `Dict` observation space; `tests/test_rllib.py` proves the
flatten/unflatten round trip reproduces the native engine's observation bit for
bit, through the same `integrations.parity` checker every other adapter is held
to, rather than only checking the tensor shape.

```python
from sharpearena.rllib_env import single_agent_config

cfg = single_agent_config(env_config={"n_symbols": 3, "n_days": 20}, num_env_runners=0)
print(cfg.env, cfg.num_env_runners)
```

Run against `ray[rllib]` 2.58.0. Output:

```
sharpearena-single-agent 0
```

Both `single_agent_config` and `multi_agent_config` (the latter with every seat on
one shared policy, since the seats are symmetric by construction) were built and
trained for one PPO iteration by hand against this Ray build with
`torch==2.14.0+cpu` installed, including the multi-agent route with
`num_envs_per_env_runner=2`: both upstream `rllib-env` and `multi-agent-envs`
pages still state multi-agent setups are not vectorizable, and that restriction is
stale at 2.58.0 (`MultiAgentEnvRunner.make_env` calls `gymnasium.make_vec` and
asserts a `VectorMultiAgentEnv` result). `torch` is not part of this package's
`ray` extra or its CI dependency set, so that training step is local evidence, not
CI-verified; `docs/integrations/support-status.md` records the distinction. The
config-construction and flatten-parity tests above run with `ray[rllib]` alone.

`pip install "sharpearena[ray]"` installs `ray[rllib]>=2.58.0,<3`, bounded rather
than open like the four extras above: the registry route, the Dict-observation
RLModule failure, and the multi-agent vectorisation behaviour were each checked
against 2.58.0 specifically, not assumed from an older Ray release. `ray[rllib]`
2.58.0 also pins `gymnasium==1.2.2` exactly, which conflicts with this repository's
own CI pin of `gymnasium==1.3.0`
(`crates/sharpearena-py/ci-requirements.txt`); the base package's
`gymnasium>=1.0` floor admits 1.2.2 and the full suite passes under it (verified
here), but that conflict is why `ray`/`rllib` is not installed in CI today and
these two test files run locally rather than as part of the pinned CI matrix.

## HUD and Harbor

These are not supported routes; treat them as local feasibility fixtures, not as
something to build on yet. No new example is added here.

- HUD: a task and grader exercised locally with a deterministic agent double,
  covered in `examples/hud/` and `crates/sharpearena-py/tests/test_hud_local.py`.
  Read [`INT-08`](integrations/INT-08-hud-local-feasibility.md) for what that run
  did and did not establish.
- Harbor: a Harbor task package with a separate-container verifier and tampering
  fixtures, in `integrations/harbor/`. Read
  [`INT-09`](integrations/INT-09-harbor-local-feasibility.md) for what that run did
  and did not establish.

## WASM and TypeScript

`crates/sharpearena-wasm/` compiles the same Rust engine to WebAssembly with
`wasm-bindgen`, and the [npm package](../npm/sharpearena/) is the supported
JavaScript and TypeScript surface built on it; its `runBaseline` example is in the
root [README](../README.md#javascript--typescript). No example calling the raw
`sharpearena-wasm` bindings directly exists in this tree yet: there is no
JavaScript or TypeScript file anywhere under `crates/sharpearena-wasm/` or `npm/`
that does so outside the npm package's own generated bindings and tests. If you
need the raw crate rather than the npm wrapper, start from
`crates/sharpearena-wasm/src/lib.rs`.
