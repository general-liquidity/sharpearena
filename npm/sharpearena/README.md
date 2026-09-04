# @general-liquidity/sharpearena

A typed Node and Bun API for SharpeArena's deterministic point-in-time evaluation
sandbox, including baselines, replay, synthetic data, stress scenarios,
walk-forward windows, and regime tags. The package loads the same Rust engine
compiled to WebAssembly and refuses a wrapper/engine `SPEC_HASH` mismatch at startup.

This package is built as CommonJS for Node 18 or newer. Browser execution is not a
supported distribution target.

> The API provides a point-in-time market model, not process containment. JavaScript
> called by the host has the host's permissions.

## Install

```bash
npm install @general-liquidity/sharpearena
```

## Run a baseline

```ts
import { runBaseline } from "@general-liquidity/sharpearena";

const run = runBaseline({
  agent: "momentum",
  dataset: { synthetic: { n_symbols: 4, n_days: 120, seed: 1 } },
  seed: 7,
});

console.log(run.returns.length, run.cost);
```

Named agents are `buy_and_hold`, `hold`, `momentum`, and `random`.

## Replay decisions

```ts
import { replayRun } from "@general-liquidity/sharpearena";

const replayed = replayRun(dataset, trajectory, costs);
```

`replayRun` recomputes returns from the trajectory's decisions, window, and seed using
the supplied dataset and cost model. It does not use `DecisionStep.step` or
`observation_id` as engine inputs. Bind those metadata fields and the expected run
geometry separately when the trajectory is an evidence artifact.

## API

| Export | Purpose |
|---|---|
| `runBaseline(config)` | Run one named in-process baseline. |
| `replayRun(dataset, trajectory, costs?)` | Recompute a captured decision trajectory. |
| `datasetSynthetic(params?)` | Build a deterministic synthetic panel. |
| `stressSuite(seed?)` | Return the named adversarial stress scenarios. |
| `walkForward(params)` | Generate disjoint out-of-sample windows. |
| `tagRegime(dataset, window)` | Classify a window as bull, bear, or chop. |
| `SPEC_HASH`, `checkSpecHash(...)` | Inspect or verify wrapper/engine compatibility. |

The package also exports the TypeScript wire and engine types, including
`MarketObservation`, `Decision`, `Run`, `RunTrajectory`, `Dataset`, `Window`, and
`CostModel`.

## Links

- [SharpeArena repository](https://github.com/general-liquidity/sharpearena)
- [Agent contract](https://github.com/general-liquidity/sharpearena/blob/main/docs/agent-contract.md)
- [SharpeBench scoring package](https://www.npmjs.com/package/@general-liquidity/sharpebench)

## License

MIT OR Apache-2.0
