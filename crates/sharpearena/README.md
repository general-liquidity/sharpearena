# SharpeArena for Rust

The deterministic, point-in-time engine and governed agent contract behind
SharpeArena.

SharpeArena owns the market cursor, exposes history only through each
`MarketObservation`, validates each `Decision`, and advances one simulation step.
It also provides procedural scenarios, vector environments, execution noise,
market clearing, a limit-order book, capture and replay, and checked external-agent
execution.

> SharpeArena is an evaluation environment, not a process sandbox. Rust code linked
> into the evaluator has the evaluator's filesystem, network, and process access. Use
> SharpeBench's digest-pinned container path for an untrusted entrant.

## Install

```bash
cargo add sharpearena
```

## Step an environment

```rust
use sharpearena::{Agent, BuyAndHold, CostModel, Dataset, TradingEnv, Window};

let data = Dataset::synthetic(4, 120, 1);
let mut env = TradingEnv::new(
    data,
    Window { start: 20, end: 120 },
    CostModel::default(),
    7,
);
let mut agent = BuyAndHold;
let mut observation = env.reset();

loop {
    let step = env.step(agent.decide(&observation));
    observation = step.observation;
    if step.done {
        break;
    }
}
```

`TradingEnv` and `run_backtest` share one step implementation. The
`env_step_matches_run_backtest` test pins equivalent native runs to the same output.

## Connect an external agent

The wire contract is one newline-delimited JSON `MarketObservation` in and one
`Decision` out. Rust exposes stdio and HTTP transports through `ExternalAgent` and
`HttpAgent`.

For any wire transport, use `run_backtest_checked`. A transport or protocol fault is
returned as `CellOutcome::Failed`, so it cannot be scored as an empty-order hold. The
lower-level `run_backtest` API remains available for trusted in-process policies, but
it does not provide that failure guarantee.

The wire evolves additively under `CONTRACT_VERSION`. The authoritative schemas and
fixtures live in [`contract/`](contract/); compatibility rules live in
[`GOVERNANCE.md`](GOVERNANCE.md).

## Capture and replay

`run_backtest_capture` records decisions with the window and seed coordinates.
`replay_run` and `replay_submission` recompute returns from those decisions, the
provided dataset, and the provided cost model.

Replay does not treat `DecisionStep.step` or `observation_id` as engine inputs. Those
fields remain evidence metadata. Bind the trajectory, dataset, costs, engine identity,
and expected geometry in a higher-level evidence contract when tamper detection is
required. The end-to-end example shows the recompute-and-score path:

```bash
cargo run -p sharpearena --example score-a-trajectory
```

## Public areas

| Area | Main types and functions |
|---|---|
| Environment | `TradingEnv`, `VecTradingEnv`, `Dataset`, `Window`, `CostModel` |
| Scenarios | `ScenarioSpec`, `generate_scenario`, train/eval splits, `SealedSalt` |
| Agent wire | `MarketObservation`, `Decision`, `Order`, `CONTRACT_VERSION` |
| Checked transport | `run_backtest_checked`, `CellOutcome`, `TransportFault` |
| Replay | `run_backtest_capture`, `replay_run`, `replay_submission` |
| Markets | `OrderBook`, `clear_bar`, `MarketParams`, `Mandate`, `ExecNoise` |
| Compatibility | `SPEC_HASH`, canonical golden fixtures, closed input schemas |

Python/Gymnasium and npm/WASM are separate distributions over this engine. Start at
the [repository README](../../README.md) for the product map.

## License

MIT OR Apache-2.0
